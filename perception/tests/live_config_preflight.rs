use std::sync::atomic::Ordering;
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use billiards_perception::calibration::{TableCalibrationV1, ValidatedTableCalibrationV1};
use billiards_perception::capture::{
    CaptureAggregateFrameLimit, CaptureAppliedCameraControls, CaptureAppliedFrameGeometry,
    CaptureCapability, CaptureControlReadback, CaptureExposureMode, CaptureFrameBudget,
    CaptureFrameLimit, CaptureFrameRate, CaptureInterrupt, CaptureOpenError,
    CaptureOutputFrameGeometry, CaptureOwnerStatus, CapturePreflightStatus, CapturePreflightWait,
    CaptureReadOutcome, CaptureReceiveOutcome, CaptureRuntimeCoordinator, CaptureRuntimeWait,
    CaptureSource, CaptureSourceCapabilities, CaptureSourceFactory, CaptureSourceMetadata,
    CaptureTimestampSemantics, CaptureVideoProfile,
};
use billiards_perception::capture_settings_wire::{
    parse_capture_settings_json, CaptureSettingsSnapshotV3,
};
use billiards_perception::detection_batch::{ClockKind, TimeUnit, TimestampStage};
use billiards_perception::live_config::{
    CaptureControlRequirements, CapturePreflightError, CaptureSessionCompatibility,
    ExpectedCaptureSettings, RequestedCaptureControlValues,
};
use billiards_perception::ImageSpace;

const EXPECTED_FRAME_LIMIT_BYTES: usize = 2_000_000;
const EXPECTED_AGGREGATE_LIMIT_BYTES: usize = 4_000_000;

// CaptureRuntimeCoordinator deliberately gates the one source for the whole
// process. Keep these fixture sessions sequential, as real camera ownership
// must be.
static RUNTIME_TEST_LOCK: Mutex<()> = Mutex::new(());

#[derive(Debug)]
struct InterruptState {
    interrupted: Mutex<bool>,
    changed: Condvar,
}

#[derive(Debug)]
struct TestInterrupt(Arc<InterruptState>);

impl CaptureInterrupt for TestInterrupt {
    fn interrupt(&self) {
        *self
            .0
            .interrupted
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = true;
        self.0.changed.notify_all();
    }
}

#[derive(Debug)]
struct TestFactory {
    metadata: CaptureSourceMetadata,
    interrupt: Arc<InterruptState>,
    metadata_ready: Option<std::sync::mpsc::Sender<()>>,
}

impl CaptureSourceFactory for TestFactory {
    fn interrupt_handle(&self) -> Arc<dyn CaptureInterrupt> {
        Arc::new(TestInterrupt(Arc::clone(&self.interrupt)))
    }

    fn open(self: Box<Self>) -> Result<Box<dyn CaptureSource>, CaptureOpenError> {
        Ok(Box::new(TestSource {
            metadata: self.metadata,
            interrupt: self.interrupt,
            metadata_ready: self.metadata_ready,
        }))
    }
}

#[derive(Debug)]
struct TestSource {
    metadata: CaptureSourceMetadata,
    interrupt: Arc<InterruptState>,
    metadata_ready: Option<std::sync::mpsc::Sender<()>>,
}

#[derive(Debug, Default)]
struct ImmediateCounts {
    opens: std::sync::atomic::AtomicUsize,
    reads: std::sync::atomic::AtomicUsize,
    releases: std::sync::atomic::AtomicUsize,
}

#[derive(Debug)]
struct ImmediateFactory {
    metadata: CaptureSourceMetadata,
    counts: Arc<ImmediateCounts>,
    open_error: Option<String>,
}

impl CaptureSourceFactory for ImmediateFactory {
    fn interrupt_handle(&self) -> Arc<dyn CaptureInterrupt> {
        Arc::new(TestInterrupt(Arc::new(InterruptState {
            interrupted: Mutex::new(false),
            changed: Condvar::new(),
        })))
    }

    fn open(self: Box<Self>) -> Result<Box<dyn CaptureSource>, CaptureOpenError> {
        self.counts.opens.fetch_add(1, Ordering::SeqCst);
        if let Some(message) = self.open_error {
            return Err(CaptureOpenError::Failed(message));
        }
        Ok(Box::new(ImmediateSource {
            metadata: self.metadata,
            counts: self.counts,
        }))
    }
}

#[derive(Debug)]
struct ImmediateSource {
    metadata: CaptureSourceMetadata,
    counts: Arc<ImmediateCounts>,
}

#[derive(Debug)]
struct OneFrameFactory {
    metadata: CaptureSourceMetadata,
    counts: Arc<ImmediateCounts>,
    frame_width_px: u32,
    frame_height_px: u32,
}

impl CaptureSourceFactory for OneFrameFactory {
    fn interrupt_handle(&self) -> Arc<dyn CaptureInterrupt> {
        Arc::new(TestInterrupt(Arc::new(InterruptState {
            interrupted: Mutex::new(false),
            changed: Condvar::new(),
        })))
    }

    fn open(self: Box<Self>) -> Result<Box<dyn CaptureSource>, CaptureOpenError> {
        self.counts.opens.fetch_add(1, Ordering::SeqCst);
        Ok(Box::new(OneFrameSource {
            metadata: self.metadata,
            counts: self.counts,
            frame_width_px: self.frame_width_px,
            frame_height_px: self.frame_height_px,
            emitted: false,
        }))
    }
}

#[derive(Debug)]
struct OneFrameSource {
    metadata: CaptureSourceMetadata,
    counts: Arc<ImmediateCounts>,
    frame_width_px: u32,
    frame_height_px: u32,
    emitted: bool,
}

impl CaptureSource for OneFrameSource {
    fn metadata(&self) -> CaptureSourceMetadata {
        self.metadata.clone()
    }

    fn read(
        &mut self,
        session_id: &str,
        frame_limit: CaptureFrameLimit,
        frame_budget: &CaptureFrameBudget,
    ) -> Result<CaptureReadOutcome, String> {
        self.counts.reads.fetch_add(1, Ordering::SeqCst);
        if std::mem::replace(&mut self.emitted, true) {
            return Ok(CaptureReadOutcome::End);
        }
        let rgb8 = frame_budget
            .allocate_rgb8_buffer(self.frame_width_px, self.frame_height_px, frame_limit)
            .map_err(|error| error.to_string())?;
        let frame = billiards_perception::capture::RgbFrameEnvelope::new(
            billiards_perception::detection_batch::SourceFrame {
                session_id: session_id.to_owned(),
                frame_id: "wrong-size-frame".to_owned(),
                timestamp: None,
                width_px: self.frame_width_px,
                height_px: self.frame_height_px,
            },
            rgb8,
            Instant::now(),
            frame_limit,
        )
        .map_err(|error| error.to_string())?;
        Ok(CaptureReadOutcome::Frame(frame))
    }

    fn release(&mut self) -> Result<(), String> {
        self.counts.releases.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}

#[derive(Debug)]
struct BlockingReleaseFactory {
    metadata: CaptureSourceMetadata,
    release_started: std::sync::mpsc::Sender<()>,
    allow_release: std::sync::mpsc::Receiver<()>,
}

impl CaptureSourceFactory for BlockingReleaseFactory {
    fn interrupt_handle(&self) -> Arc<dyn CaptureInterrupt> {
        Arc::new(TestInterrupt(Arc::new(InterruptState {
            interrupted: Mutex::new(false),
            changed: Condvar::new(),
        })))
    }

    fn open(self: Box<Self>) -> Result<Box<dyn CaptureSource>, CaptureOpenError> {
        Ok(Box::new(BlockingReleaseSource {
            metadata: self.metadata,
            release_started: self.release_started,
            allow_release: self.allow_release,
        }))
    }
}

#[derive(Debug)]
struct BlockingReleaseSource {
    metadata: CaptureSourceMetadata,
    release_started: std::sync::mpsc::Sender<()>,
    allow_release: std::sync::mpsc::Receiver<()>,
}

impl CaptureSource for BlockingReleaseSource {
    fn metadata(&self) -> CaptureSourceMetadata {
        self.metadata.clone()
    }

    fn read(
        &mut self,
        _session_id: &str,
        _frame_limit: CaptureFrameLimit,
        _frame_budget: &CaptureFrameBudget,
    ) -> Result<CaptureReadOutcome, String> {
        Ok(CaptureReadOutcome::End)
    }

    fn release(&mut self) -> Result<(), String> {
        let _ = self.release_started.send(());
        self.allow_release
            .recv()
            .map_err(|error| format!("release gate disconnected: {error}"))
    }
}

impl CaptureSource for ImmediateSource {
    fn metadata(&self) -> CaptureSourceMetadata {
        self.metadata.clone()
    }

    fn read(
        &mut self,
        _session_id: &str,
        _frame_limit: CaptureFrameLimit,
        _frame_budget: &CaptureFrameBudget,
    ) -> Result<CaptureReadOutcome, String> {
        self.counts.reads.fetch_add(1, Ordering::SeqCst);
        Ok(CaptureReadOutcome::End)
    }

    fn release(&mut self) -> Result<(), String> {
        self.counts.releases.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}

#[derive(Debug)]
struct BlockingOpenFactory {
    interrupt: Arc<InterruptState>,
    open_started: std::sync::mpsc::Sender<()>,
}

#[derive(Debug)]
struct SuccessfulOpenAfterInterruptFactory {
    metadata: CaptureSourceMetadata,
    counts: Arc<ImmediateCounts>,
    interrupt: Arc<InterruptState>,
    open_started: std::sync::mpsc::Sender<()>,
}

impl CaptureSourceFactory for SuccessfulOpenAfterInterruptFactory {
    fn interrupt_handle(&self) -> Arc<dyn CaptureInterrupt> {
        Arc::new(TestInterrupt(Arc::clone(&self.interrupt)))
    }

    fn open(self: Box<Self>) -> Result<Box<dyn CaptureSource>, CaptureOpenError> {
        self.counts.opens.fetch_add(1, Ordering::SeqCst);
        let _ = self.open_started.send(());
        let mut interrupted = self
            .interrupt
            .interrupted
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        while !*interrupted {
            interrupted = self
                .interrupt
                .changed
                .wait(interrupted)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
        }
        drop(interrupted);
        Ok(Box::new(ImmediateSource {
            metadata: self.metadata,
            counts: self.counts,
        }))
    }
}

impl CaptureSourceFactory for BlockingOpenFactory {
    fn interrupt_handle(&self) -> Arc<dyn CaptureInterrupt> {
        Arc::new(TestInterrupt(Arc::clone(&self.interrupt)))
    }

    fn open(self: Box<Self>) -> Result<Box<dyn CaptureSource>, CaptureOpenError> {
        let _ = self.open_started.send(());
        let mut interrupted = self
            .interrupt
            .interrupted
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        while !*interrupted {
            interrupted = self
                .interrupt
                .changed
                .wait(interrupted)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
        }
        Err(CaptureOpenError::Interrupted)
    }
}

impl CaptureSource for TestSource {
    fn metadata(&self) -> CaptureSourceMetadata {
        if let Some(sender) = &self.metadata_ready {
            let _ = sender.send(());
        }
        self.metadata.clone()
    }

    fn read(
        &mut self,
        _session_id: &str,
        _frame_limit: CaptureFrameLimit,
        _frame_budget: &CaptureFrameBudget,
    ) -> Result<CaptureReadOutcome, String> {
        let mut interrupted = self
            .interrupt
            .interrupted
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        while !*interrupted {
            interrupted = self
                .interrupt
                .changed
                .wait(interrupted)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
        }
        Ok(CaptureReadOutcome::End)
    }

    fn release(&mut self) -> Result<(), String> {
        Ok(())
    }
}

fn profile(width_px: u32, height_px: u32, pixel_format: &str) -> CaptureVideoProfile {
    CaptureVideoProfile::new(
        width_px,
        height_px,
        CaptureFrameRate::new(30, 1).unwrap(),
        billiards_perception::capture::CapturePixelFormat::new(pixel_format).unwrap(),
    )
    .unwrap()
}

fn base_profile() -> CaptureVideoProfile {
    profile(1000, 500, "RGB24")
}

fn source_timestamp_semantics() -> CaptureTimestampSemantics {
    CaptureTimestampSemantics::SourceTimestamp {
        unit: TimeUnit::Nanoseconds,
        clock: ClockKind::Monotonic,
        domain_id: "camera-clock".to_owned(),
        stage: TimestampStage::Exposure,
    }
}

fn read_completion_semantics() -> CaptureTimestampSemantics {
    CaptureTimestampSemantics::ReadCompletionOnly
}

fn source_metadata(
    device_id: &str,
    requested_profile: CaptureVideoProfile,
    negotiated_profile: CaptureVideoProfile,
    manual_exposure: CaptureCapability,
    timestamp_semantics: CaptureTimestampSemantics,
    source_timestamps: CaptureCapability,
) -> CaptureSourceMetadata {
    source_metadata_with_backend(
        "test-adapter",
        device_id,
        requested_profile,
        negotiated_profile,
        manual_exposure,
        timestamp_semantics,
        source_timestamps,
    )
}

fn source_metadata_with_backend(
    backend_id: &str,
    device_id: &str,
    requested_profile: CaptureVideoProfile,
    negotiated_profile: CaptureVideoProfile,
    manual_exposure: CaptureCapability,
    timestamp_semantics: CaptureTimestampSemantics,
    source_timestamps: CaptureCapability,
) -> CaptureSourceMetadata {
    let pixel_format = billiards_perception::capture::CapturePixelFormat::new("RGB24").unwrap();
    let capabilities = CaptureSourceCapabilities::new(
        Some(vec![pixel_format]),
        manual_exposure,
        CaptureCapability::Unknown,
        CaptureCapability::Unsupported,
        source_timestamps,
    )
    .unwrap();
    source_metadata_with_reported_controls(
        backend_id,
        device_id,
        requested_profile,
        negotiated_profile,
        capabilities,
        CaptureAppliedCameraControls::unknown(),
        timestamp_semantics,
    )
}

fn source_metadata_with_reported_controls(
    backend_id: &str,
    device_id: &str,
    requested_profile: CaptureVideoProfile,
    negotiated_profile: CaptureVideoProfile,
    capabilities: CaptureSourceCapabilities,
    applied_controls: CaptureAppliedCameraControls,
    timestamp_semantics: CaptureTimestampSemantics,
) -> CaptureSourceMetadata {
    source_metadata_with_geometry_and_controls(
        backend_id,
        device_id,
        requested_profile,
        negotiated_profile,
        capabilities,
        (
            CaptureAppliedFrameGeometry::new(12, "test-camera-profile", ImageSpace::Rectified, 2)
                .unwrap(),
            applied_controls,
        ),
        timestamp_semantics,
    )
}

fn source_metadata_with_geometry_and_controls(
    backend_id: &str,
    device_id: &str,
    requested_profile: CaptureVideoProfile,
    negotiated_profile: CaptureVideoProfile,
    capabilities: CaptureSourceCapabilities,
    applied_readbacks: (CaptureAppliedFrameGeometry, CaptureAppliedCameraControls),
    timestamp_semantics: CaptureTimestampSemantics,
) -> CaptureSourceMetadata {
    let (applied_frame_geometry, applied_controls) = applied_readbacks;
    CaptureSourceMetadata::new(
        device_id,
        Some("test device".to_owned()),
        backend_id,
        requested_profile,
        negotiated_profile,
        capabilities,
        applied_frame_geometry,
        applied_controls,
        timestamp_semantics,
    )
    .unwrap()
}

fn metadata_with_geometry(geometry: CaptureAppliedFrameGeometry) -> CaptureSourceMetadata {
    let pixel_format = billiards_perception::capture::CapturePixelFormat::new("RGB24").unwrap();
    let capabilities = CaptureSourceCapabilities::new(
        Some(vec![pixel_format]),
        CaptureCapability::Supported,
        CaptureCapability::Supported,
        CaptureCapability::Supported,
        CaptureCapability::Unknown,
    )
    .unwrap();
    source_metadata_with_geometry_and_controls(
        "test-adapter",
        "camera-0",
        base_profile(),
        base_profile(),
        capabilities,
        (geometry, CaptureAppliedCameraControls::unknown()),
        read_completion_semantics(),
    )
}

fn opened_report(
    metadata: CaptureSourceMetadata,
) -> billiards_perception::capture::CaptureSessionReport {
    let _runtime_guard = RUNTIME_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let (metadata_ready_sender, metadata_ready_receiver) = std::sync::mpsc::channel();
    let interrupt = Arc::new(InterruptState {
        interrupted: Mutex::new(false),
        changed: Condvar::new(),
    });
    let frame_limit = CaptureFrameLimit::new(EXPECTED_FRAME_LIMIT_BYTES).unwrap();
    let aggregate_limit = CaptureAggregateFrameLimit::new(EXPECTED_AGGREGATE_LIMIT_BYTES).unwrap();
    let mut runtime = CaptureRuntimeCoordinator::new(frame_limit, aggregate_limit).unwrap();
    runtime
        .start_unvalidated(TestFactory {
            metadata,
            interrupt,
            metadata_ready: Some(metadata_ready_sender),
        })
        .unwrap();
    metadata_ready_receiver
        .recv_timeout(Duration::from_secs(2))
        .expect("capture worker reaches its opened metadata state");
    let deadline = Instant::now() + Duration::from_secs(2);
    let report = loop {
        if let Some(report) = runtime.active_source_report() {
            break report;
        }
        assert!(
            Instant::now() < deadline,
            "opened source report is available while the source remains active"
        );
        std::thread::yield_now();
    };
    assert!(matches!(
        runtime.stop(Duration::from_secs(2)),
        CaptureRuntimeWait::Completed(_)
    ));
    report
}

fn calibration() -> ValidatedTableCalibrationV1 {
    serde_json::from_str::<TableCalibrationV1>(include_str!(
        "../contracts/calibration/v1/table-calibration-center-origin.json"
    ))
    .unwrap()
    .into_validated()
    .unwrap()
}

fn expected_settings(
    device_id: &str,
    requested_profile: CaptureVideoProfile,
    expected_negotiated_profile: CaptureVideoProfile,
    control_requirements: CaptureControlRequirements,
    timestamp_semantics: CaptureTimestampSemantics,
    frame_limit_bytes: usize,
    aggregate_limit_bytes: usize,
) -> Result<ExpectedCaptureSettings, CapturePreflightError> {
    ExpectedCaptureSettings::new(
        12,
        device_id,
        requested_profile,
        expected_negotiated_profile,
        "test-camera-profile",
        ImageSpace::Rectified,
        2,
        3,
        Some(calibration()),
        timestamp_semantics,
        control_requirements,
        Duration::from_millis(250),
        CaptureFrameLimit::new(frame_limit_bytes).unwrap(),
        CaptureAggregateFrameLimit::new(aggregate_limit_bytes).unwrap(),
    )
}

fn report_for_expected_settings(
    expected: &ExpectedCaptureSettings,
    applied_controls: CaptureAppliedCameraControls,
) -> billiards_perception::capture::CaptureSessionReport {
    let pixel_format = expected
        .expected_negotiated_profile()
        .pixel_format()
        .clone();
    let source_timestamps = match expected.timestamp_semantics() {
        CaptureTimestampSemantics::ReadCompletionOnly => CaptureCapability::Unsupported,
        CaptureTimestampSemantics::SourceTimestamp { .. } => CaptureCapability::Supported,
    };
    let capabilities = CaptureSourceCapabilities::new(
        Some(vec![pixel_format]),
        CaptureCapability::Supported,
        CaptureCapability::Supported,
        CaptureCapability::Supported,
        source_timestamps,
    )
    .unwrap();
    let applied_geometry = CaptureAppliedFrameGeometry::new(
        expected.configuration_revision(),
        expected.camera_profile_id(),
        expected.image_space(),
        expected.rectification_revision(),
    )
    .unwrap();
    let metadata = CaptureSourceMetadata::new(
        expected.device_id(),
        Some("test-camera".to_owned()),
        "test-adapter",
        expected.requested_profile().clone(),
        expected.expected_negotiated_profile().clone(),
        capabilities,
        applied_geometry,
        applied_controls,
        expected.timestamp_semantics().clone(),
    )
    .unwrap();
    let metadata = match expected.output_frame_geometry() {
        Some(geometry) => metadata.with_output_frame_geometry(geometry),
        None => metadata,
    };
    opened_report(metadata)
}

fn default_metadata() -> CaptureSourceMetadata {
    source_metadata(
        "camera-0",
        base_profile(),
        base_profile(),
        CaptureCapability::Supported,
        read_completion_semantics(),
        CaptureCapability::Unknown,
    )
}

fn new_runtime() -> CaptureRuntimeCoordinator {
    CaptureRuntimeCoordinator::new(
        CaptureFrameLimit::new(EXPECTED_FRAME_LIMIT_BYTES).unwrap(),
        CaptureAggregateFrameLimit::new(EXPECTED_AGGREGATE_LIMIT_BYTES).unwrap(),
    )
    .unwrap()
}

fn immediate_factory(metadata: CaptureSourceMetadata) -> (ImmediateFactory, Arc<ImmediateCounts>) {
    let counts = Arc::new(ImmediateCounts::default());
    (
        ImmediateFactory {
            metadata,
            counts: Arc::clone(&counts),
            open_error: None,
        },
        counts,
    )
}

#[test]
fn valid_expected_profile_binds_revision_to_opened_session() {
    let report = opened_report(default_metadata());
    let expected = expected_settings(
        "camera-0",
        base_profile(),
        base_profile(),
        CaptureControlRequirements::default(),
        read_completion_semantics(),
        EXPECTED_FRAME_LIMIT_BYTES,
        EXPECTED_AGGREGATE_LIMIT_BYTES,
    )
    .unwrap();

    let compatible = expected.validate_session(&report).unwrap();
    assert_eq!(compatible.configuration_revision(), 12);
    assert_eq!(
        compatible.expected_settings().camera_profile_id(),
        "test-camera-profile"
    );
    assert_eq!(compatible.expected_settings().calibration_revision(), 3);
    assert_eq!(
        compatible.expected_settings().image_space(),
        ImageSpace::Rectified
    );
    assert_eq!(compatible.session_id(), report.session_id());
    assert_eq!(compatible.device_id(), "camera-0");
    assert_eq!(compatible.negotiated_profile(), report.negotiated_profile());
}

#[test]
fn v2_preflight_requires_the_requested_emitted_rgb8_geometry() {
    let input_profile = base_profile();
    let output_geometry = CaptureOutputFrameGeometry::new(16, 16).unwrap();
    let expected = ExpectedCaptureSettings::new_with_output_frame_geometry(
        12,
        "camera-0",
        input_profile.clone(),
        input_profile,
        output_geometry,
        "test-camera-profile",
        ImageSpace::Rectified,
        2,
        3,
        None,
        read_completion_semantics(),
        CaptureControlRequirements::default(),
        Duration::from_millis(250),
        CaptureFrameLimit::new(EXPECTED_FRAME_LIMIT_BYTES).unwrap(),
        CaptureAggregateFrameLimit::new(EXPECTED_AGGREGATE_LIMIT_BYTES).unwrap(),
    )
    .unwrap();

    let omitted = opened_report(default_metadata());
    assert_eq!(
        expected.validate_session(&omitted).unwrap_err(),
        CapturePreflightError::OutputFrameGeometryMissing
    );

    let matching_metadata = default_metadata()
        .with_output_frame_geometry(CaptureOutputFrameGeometry::new(16, 16).unwrap());
    let matching = opened_report(matching_metadata);
    let compatibility = expected.validate_session(&matching).unwrap();
    assert_eq!(compatibility.output_frame_geometry(), output_geometry);

    let mismatched_metadata = default_metadata()
        .with_output_frame_geometry(CaptureOutputFrameGeometry::new(15, 16).unwrap());
    let mismatched = opened_report(mismatched_metadata);
    assert_eq!(
        expected.validate_session(&mismatched).unwrap_err(),
        CapturePreflightError::OutputFrameGeometryMismatch
    );

    // Even a dimension-preserving source must negotiate v2 before adding the
    // new report field; otherwise a v1 acknowledgement would change shape.
    let legacy_expected = expected_settings(
        "camera-0",
        base_profile(),
        base_profile(),
        CaptureControlRequirements::default(),
        read_completion_semantics(),
        EXPECTED_FRAME_LIMIT_BYTES,
        EXPECTED_AGGREGATE_LIMIT_BYTES,
    )
    .unwrap();
    let profile = base_profile();
    let v2_report = opened_report(default_metadata().with_output_frame_geometry(
        CaptureOutputFrameGeometry::new(profile.width_px(), profile.height_px()).unwrap(),
    ));
    assert_eq!(
        legacy_expected.validate_session(&v2_report).unwrap_err(),
        CapturePreflightError::OutputFrameGeometryMismatch
    );
}

#[test]
fn applied_source_geometry_must_match_revision_profile_and_image_space() {
    let expected = expected_settings(
        "camera-0",
        base_profile(),
        base_profile(),
        CaptureControlRequirements::default(),
        read_completion_semantics(),
        EXPECTED_FRAME_LIMIT_BYTES,
        EXPECTED_AGGREGATE_LIMIT_BYTES,
    )
    .unwrap();
    let cases = [
        (
            CaptureAppliedFrameGeometry::new(11, "test-camera-profile", ImageSpace::Rectified, 2)
                .unwrap(),
            CapturePreflightError::ConfigurationRevisionMismatch,
        ),
        (
            CaptureAppliedFrameGeometry::new(
                12,
                "another-camera-profile",
                ImageSpace::Rectified,
                2,
            )
            .unwrap(),
            CapturePreflightError::CameraProfileMismatch,
        ),
        (
            CaptureAppliedFrameGeometry::new(12, "test-camera-profile", ImageSpace::Raw, 2)
                .unwrap(),
            CapturePreflightError::ImageSpaceMismatch,
        ),
        (
            CaptureAppliedFrameGeometry::new(12, "test-camera-profile", ImageSpace::Rectified, 3)
                .unwrap(),
            CapturePreflightError::RectificationRevisionMismatch,
        ),
    ];
    for (geometry, expected_error) in cases {
        let report = opened_report(metadata_with_geometry(geometry));
        assert_eq!(
            expected.validate_session(&report).unwrap_err(),
            expected_error
        );
    }
}

#[test]
fn source_control_readbacks_are_distinct_and_required_values_must_match() {
    let pixel_format = billiards_perception::capture::CapturePixelFormat::new("RGB24").unwrap();
    let capabilities = CaptureSourceCapabilities::new(
        Some(vec![pixel_format]),
        CaptureCapability::Supported,
        CaptureCapability::Supported,
        CaptureCapability::Supported,
        CaptureCapability::Unknown,
    )
    .unwrap();
    let applied_controls = CaptureAppliedCameraControls::new(
        CaptureControlReadback::Applied(CaptureExposureMode::Automatic),
        CaptureControlReadback::Unknown,
        CaptureControlReadback::Applied(1.0),
        CaptureControlReadback::Unknown,
        CaptureControlReadback::Unsupported,
        CaptureControlReadback::Unknown,
        CaptureControlReadback::Applied(true),
        CaptureControlReadback::Applied(2),
    )
    .unwrap();
    let report = opened_report(source_metadata_with_reported_controls(
        "test-adapter",
        "camera-0",
        base_profile(),
        base_profile(),
        capabilities,
        applied_controls,
        read_completion_semantics(),
    ));
    let expected = expected_settings(
        "camera-0",
        base_profile(),
        base_profile(),
        CaptureControlRequirements {
            automatic_exposure: true,
            gain_control: true,
            ..CaptureControlRequirements::default()
        },
        read_completion_semantics(),
        EXPECTED_FRAME_LIMIT_BYTES,
        EXPECTED_AGGREGATE_LIMIT_BYTES,
    )
    .unwrap()
    .with_requested_control_values(
        RequestedCaptureControlValues::new(CaptureExposureMode::Automatic, None, 1.0).unwrap(),
    )
    .unwrap();

    let compatible = expected.validate_session(&report).unwrap();
    assert_eq!(
        compatible.applied_controls().gain(),
        &CaptureControlReadback::Applied(1.0)
    );
    assert_eq!(
        compatible.applied_controls().contrast(),
        &CaptureControlReadback::Unsupported
    );
    assert_eq!(
        compatible.applied_controls().brightness(),
        &CaptureControlReadback::Unknown
    );

    let mismatch_controls = CaptureAppliedCameraControls::new(
        CaptureControlReadback::Applied(CaptureExposureMode::Manual),
        CaptureControlReadback::Unknown,
        CaptureControlReadback::Applied(1.0),
        CaptureControlReadback::Unknown,
        CaptureControlReadback::Unknown,
        CaptureControlReadback::Unknown,
        CaptureControlReadback::Unknown,
        CaptureControlReadback::Unknown,
    )
    .unwrap();
    let pixel_format = billiards_perception::capture::CapturePixelFormat::new("RGB24").unwrap();
    let capabilities = CaptureSourceCapabilities::new(
        Some(vec![pixel_format]),
        CaptureCapability::Supported,
        CaptureCapability::Supported,
        CaptureCapability::Supported,
        CaptureCapability::Unknown,
    )
    .unwrap();
    let mismatch_report = opened_report(source_metadata_with_reported_controls(
        "test-adapter",
        "camera-0",
        base_profile(),
        base_profile(),
        capabilities,
        mismatch_controls,
        read_completion_semantics(),
    ));
    assert_eq!(
        expected.validate_session(&mismatch_report).unwrap_err(),
        CapturePreflightError::RequiredControlReadbackMismatch("exposure_mode")
    );
}

#[test]
fn required_manual_exposure_and_gain_need_available_exact_readbacks() {
    let requirements = CaptureControlRequirements {
        manual_exposure: true,
        gain_control: true,
        ..CaptureControlRequirements::default()
    };
    let expected = expected_settings(
        "camera-0",
        base_profile(),
        base_profile(),
        requirements,
        read_completion_semantics(),
        EXPECTED_FRAME_LIMIT_BYTES,
        EXPECTED_AGGREGATE_LIMIT_BYTES,
    )
    .unwrap()
    .with_requested_control_values(
        RequestedCaptureControlValues::new(CaptureExposureMode::Manual, Some(-4.0), 1.0).unwrap(),
    )
    .unwrap();

    let pixel_format = billiards_perception::capture::CapturePixelFormat::new("RGB24").unwrap();
    let capabilities = CaptureSourceCapabilities::new(
        Some(vec![pixel_format]),
        CaptureCapability::Supported,
        CaptureCapability::Unknown,
        CaptureCapability::Supported,
        CaptureCapability::Unknown,
    )
    .unwrap();
    let matching_report = opened_report(source_metadata_with_reported_controls(
        "test-adapter",
        "camera-0",
        base_profile(),
        base_profile(),
        capabilities.clone(),
        CaptureAppliedCameraControls::new(
            CaptureControlReadback::Applied(CaptureExposureMode::Manual),
            CaptureControlReadback::Applied(-4.0),
            CaptureControlReadback::Applied(1.0),
            CaptureControlReadback::Unknown,
            CaptureControlReadback::Unknown,
            CaptureControlReadback::Unknown,
            CaptureControlReadback::Unknown,
            CaptureControlReadback::Unknown,
        )
        .unwrap(),
        read_completion_semantics(),
    ));
    expected.validate_session(&matching_report).unwrap();

    let missing_request = expected_settings(
        "camera-0",
        base_profile(),
        base_profile(),
        requirements,
        read_completion_semantics(),
        EXPECTED_FRAME_LIMIT_BYTES,
        EXPECTED_AGGREGATE_LIMIT_BYTES,
    )
    .unwrap();
    assert_eq!(
        missing_request
            .validate_session(&matching_report)
            .unwrap_err(),
        CapturePreflightError::RequestedControlValuesMissing
    );

    for (applied_controls, expected_error) in [
        (
            CaptureAppliedCameraControls::new(
                CaptureControlReadback::Unknown,
                CaptureControlReadback::Unknown,
                CaptureControlReadback::Applied(1.0),
                CaptureControlReadback::Unknown,
                CaptureControlReadback::Unknown,
                CaptureControlReadback::Unknown,
                CaptureControlReadback::Unknown,
                CaptureControlReadback::Unknown,
            )
            .unwrap(),
            CapturePreflightError::RequiredControlReadbackUnavailable("exposure_mode"),
        ),
        (
            CaptureAppliedCameraControls::new(
                CaptureControlReadback::Applied(CaptureExposureMode::Manual),
                CaptureControlReadback::Unknown,
                CaptureControlReadback::Applied(1.0),
                CaptureControlReadback::Unknown,
                CaptureControlReadback::Unknown,
                CaptureControlReadback::Unknown,
                CaptureControlReadback::Unknown,
                CaptureControlReadback::Unknown,
            )
            .unwrap(),
            CapturePreflightError::RequiredControlReadbackUnavailable("exposure_value"),
        ),
        (
            CaptureAppliedCameraControls::new(
                CaptureControlReadback::Applied(CaptureExposureMode::Manual),
                CaptureControlReadback::Applied(-3.5),
                CaptureControlReadback::Applied(1.0),
                CaptureControlReadback::Unknown,
                CaptureControlReadback::Unknown,
                CaptureControlReadback::Unknown,
                CaptureControlReadback::Unknown,
                CaptureControlReadback::Unknown,
            )
            .unwrap(),
            CapturePreflightError::RequiredControlReadbackMismatch("exposure_value"),
        ),
        (
            CaptureAppliedCameraControls::new(
                CaptureControlReadback::Applied(CaptureExposureMode::Manual),
                CaptureControlReadback::Applied(-4.0),
                CaptureControlReadback::Unknown,
                CaptureControlReadback::Unknown,
                CaptureControlReadback::Unknown,
                CaptureControlReadback::Unknown,
                CaptureControlReadback::Unknown,
                CaptureControlReadback::Unknown,
            )
            .unwrap(),
            CapturePreflightError::RequiredControlReadbackUnavailable("gain"),
        ),
    ] {
        let report = opened_report(source_metadata_with_reported_controls(
            "test-adapter",
            "camera-0",
            base_profile(),
            base_profile(),
            capabilities.clone(),
            applied_controls,
            read_completion_semantics(),
        ));
        assert_eq!(
            expected.validate_session(&report).unwrap_err(),
            expected_error
        );
    }
}

#[test]
fn requested_backend_is_verified_against_the_opened_source_report() {
    let report = opened_report(source_metadata_with_backend(
        "v4l2",
        "camera-0",
        base_profile(),
        base_profile(),
        CaptureCapability::Supported,
        read_completion_semantics(),
        CaptureCapability::Unsupported,
    ));
    let expected = expected_settings(
        "camera-0",
        base_profile(),
        base_profile(),
        CaptureControlRequirements::default(),
        read_completion_semantics(),
        EXPECTED_FRAME_LIMIT_BYTES,
        EXPECTED_AGGREGATE_LIMIT_BYTES,
    )
    .unwrap()
    .with_requested_backend("v4l2")
    .unwrap();
    let compatible = expected.validate_session(&report).unwrap();
    assert_eq!(compatible.backend_id(), "v4l2");

    let mismatch = expected_settings(
        "camera-0",
        base_profile(),
        base_profile(),
        CaptureControlRequirements::default(),
        read_completion_semantics(),
        EXPECTED_FRAME_LIMIT_BYTES,
        EXPECTED_AGGREGATE_LIMIT_BYTES,
    )
    .unwrap()
    .with_requested_backend("dshow")
    .unwrap();
    assert!(matches!(
        mismatch.validate_session(&report),
        Err(CapturePreflightError::RequestedBackendMismatch)
    ));

    let automatic = expected_settings(
        "camera-0",
        base_profile(),
        base_profile(),
        CaptureControlRequirements::default(),
        read_completion_semantics(),
        EXPECTED_FRAME_LIMIT_BYTES,
        EXPECTED_AGGREGATE_LIMIT_BYTES,
    )
    .unwrap()
    .with_requested_backend("auto")
    .unwrap();
    assert_eq!(
        automatic.validate_session(&report).unwrap().backend_id(),
        "v4l2"
    );
    assert!(matches!(
        automatic.clone().with_requested_backend("invented"),
        Err(CapturePreflightError::UnsupportedCaptureBackend)
    ));
}

#[test]
fn wrong_device_requested_profile_and_negotiated_mode_are_rejected() {
    let report = opened_report(default_metadata());
    let wrong_device = expected_settings(
        "other-camera",
        base_profile(),
        base_profile(),
        CaptureControlRequirements::default(),
        read_completion_semantics(),
        EXPECTED_FRAME_LIMIT_BYTES,
        EXPECTED_AGGREGATE_LIMIT_BYTES,
    )
    .unwrap();
    assert_eq!(
        wrong_device.validate_session(&report).unwrap_err(),
        CapturePreflightError::DeviceMismatch
    );

    let wrong_requested_profile = expected_settings(
        "camera-0",
        profile(800, 500, "RGB24"),
        base_profile(),
        CaptureControlRequirements::default(),
        read_completion_semantics(),
        EXPECTED_FRAME_LIMIT_BYTES,
        EXPECTED_AGGREGATE_LIMIT_BYTES,
    )
    .unwrap();
    assert_eq!(
        wrong_requested_profile
            .validate_session(&report)
            .unwrap_err(),
        CapturePreflightError::RequestedProfileMismatch
    );

    let wrong_negotiated_mode = expected_settings(
        "camera-0",
        base_profile(),
        profile(800, 500, "RGB24"),
        CaptureControlRequirements::default(),
        read_completion_semantics(),
        EXPECTED_FRAME_LIMIT_BYTES,
        EXPECTED_AGGREGATE_LIMIT_BYTES,
    )
    .unwrap();
    assert_eq!(
        wrong_negotiated_mode.validate_session(&report).unwrap_err(),
        CapturePreflightError::NegotiatedProfileMismatch
    );
}

#[test]
fn unsupported_and_unknown_requested_controls_are_rejected() {
    for reported in [CaptureCapability::Unsupported, CaptureCapability::Unknown] {
        let metadata = source_metadata(
            "camera-0",
            base_profile(),
            base_profile(),
            reported,
            read_completion_semantics(),
            CaptureCapability::Unknown,
        );
        let report = opened_report(metadata);
        let expected = expected_settings(
            "camera-0",
            base_profile(),
            base_profile(),
            CaptureControlRequirements {
                manual_exposure: true,
                ..CaptureControlRequirements::default()
            },
            read_completion_semantics(),
            EXPECTED_FRAME_LIMIT_BYTES,
            EXPECTED_AGGREGATE_LIMIT_BYTES,
        )
        .unwrap();
        assert_eq!(
            expected.validate_session(&report).unwrap_err(),
            CapturePreflightError::RequiredCapabilityUnavailable {
                capability: "manual_exposure",
                reported,
            }
        );
    }
}

#[test]
fn v3_source_defaults_accepts_and_preserves_unknown_or_unsupported_control_reports() {
    let resolved = CaptureSettingsSnapshotV3::parse_json(include_str!(
        "../contracts/capture-settings/v3/capture-settings-default.json"
    ))
    .unwrap();
    let expected = &resolved.expected;
    assert_eq!(expected.requested_control_values(), None);

    for (capability, unknown_readbacks) in [
        (CaptureCapability::Unsupported, false),
        (CaptureCapability::Unknown, true),
        (CaptureCapability::Unsupported, true),
        (CaptureCapability::Unknown, false),
    ] {
        let applied_controls = if unknown_readbacks {
            CaptureAppliedCameraControls::unknown()
        } else {
            CaptureAppliedCameraControls::new(
                CaptureControlReadback::Unsupported,
                CaptureControlReadback::Unsupported,
                CaptureControlReadback::Unsupported,
                CaptureControlReadback::Unsupported,
                CaptureControlReadback::Unsupported,
                CaptureControlReadback::Unsupported,
                CaptureControlReadback::Unsupported,
                CaptureControlReadback::Unsupported,
            )
            .unwrap()
        };
        let pixel_format = expected
            .expected_negotiated_profile()
            .pixel_format()
            .clone();
        let capabilities = CaptureSourceCapabilities::new(
            Some(vec![pixel_format]),
            capability,
            capability,
            capability,
            CaptureCapability::Unsupported,
        )
        .unwrap();
        let applied_geometry = CaptureAppliedFrameGeometry::new(
            expected.configuration_revision(),
            expected.camera_profile_id(),
            expected.image_space(),
            expected.rectification_revision(),
        )
        .unwrap();
        let metadata = CaptureSourceMetadata::new(
            expected.device_id(),
            Some("test-camera".to_owned()),
            "test-adapter",
            expected.requested_profile().clone(),
            expected.expected_negotiated_profile().clone(),
            capabilities,
            applied_geometry,
            applied_controls,
            expected.timestamp_semantics().clone(),
        )
        .unwrap()
        .with_output_frame_geometry(expected.output_frame_geometry().unwrap());
        let report = opened_report(metadata);
        let compatible = expected
            .validate_session(&report)
            .expect("source-default policy must not require unavailable controls");
        assert_eq!(
            matches!(
                compatible.applied_controls().exposure_mode(),
                CaptureControlReadback::Unknown
            ),
            unknown_readbacks
        );
        assert_eq!(
            matches!(
                compatible.applied_controls().gain(),
                CaptureControlReadback::Unknown
            ),
            unknown_readbacks
        );
        assert_eq!(
            compatible.source_report().capabilities().manual_exposure(),
            capability
        );
        assert_eq!(
            compatible
                .source_report()
                .capabilities()
                .automatic_exposure(),
            capability
        );
        assert_eq!(
            compatible.source_report().capabilities().gain_control(),
            capability
        );
        assert_eq!(
            compatible.expected_settings().requested_control_values(),
            None
        );
    }
}

#[test]
fn strict_v1_v2_and_v3_require_reject_missing_and_mismatched_control_readbacks() {
    let settings = [
        include_str!("../contracts/capture-settings/v1/capture-settings-default.json"),
        include_str!("../contracts/capture-settings/v2/capture-settings-default.json"),
        include_str!("../contracts/capture-settings/v3/capture-settings-require-controls.json"),
    ];
    for json in settings {
        let resolved = parse_capture_settings_json(json).unwrap();
        let expected = &resolved.expected;
        assert!(expected.requested_control_values().is_some());

        let missing_readbacks =
            report_for_expected_settings(expected, CaptureAppliedCameraControls::unknown());
        assert_eq!(
            expected.validate_session(&missing_readbacks).unwrap_err(),
            CapturePreflightError::RequiredControlReadbackUnavailable("exposure_mode"),
            "schema v{} must reject unknown required readbacks",
            resolved.schema_version
        );

        let mismatched_readbacks = CaptureAppliedCameraControls::new(
            CaptureControlReadback::Applied(CaptureExposureMode::Automatic),
            CaptureControlReadback::Unknown,
            CaptureControlReadback::Applied(0.5),
            CaptureControlReadback::Unknown,
            CaptureControlReadback::Unknown,
            CaptureControlReadback::Unknown,
            CaptureControlReadback::Unknown,
            CaptureControlReadback::Unknown,
        )
        .unwrap();
        let mismatch_report = report_for_expected_settings(expected, mismatched_readbacks);
        assert_eq!(
            expected.validate_session(&mismatch_report).unwrap_err(),
            CapturePreflightError::RequiredControlReadbackMismatch("gain"),
            "schema v{} must reject mismatched required readbacks",
            resolved.schema_version
        );
    }
}

#[test]
fn source_timestamp_request_rejects_unknown_or_unsupported_host_only_sources() {
    for reported in [CaptureCapability::Unsupported, CaptureCapability::Unknown] {
        let metadata = source_metadata(
            "camera-0",
            base_profile(),
            base_profile(),
            CaptureCapability::Supported,
            read_completion_semantics(),
            reported,
        );
        let report = opened_report(metadata);
        let expected = expected_settings(
            "camera-0",
            base_profile(),
            base_profile(),
            CaptureControlRequirements::default(),
            source_timestamp_semantics(),
            EXPECTED_FRAME_LIMIT_BYTES,
            EXPECTED_AGGREGATE_LIMIT_BYTES,
        )
        .unwrap();
        assert_eq!(
            expected.validate_session(&report).unwrap_err(),
            CapturePreflightError::TimestampSemanticsMismatch
        );
    }
}

#[test]
fn calibration_revisions_and_image_geometry_must_match_expected_settings() {
    let mismatched_profile = ExpectedCaptureSettings::new(
        12,
        "camera-0",
        base_profile(),
        base_profile(),
        "different-camera-profile",
        ImageSpace::Rectified,
        2,
        3,
        Some(calibration()),
        read_completion_semantics(),
        CaptureControlRequirements::default(),
        Duration::from_millis(250),
        CaptureFrameLimit::new(EXPECTED_FRAME_LIMIT_BYTES).unwrap(),
        CaptureAggregateFrameLimit::new(EXPECTED_AGGREGATE_LIMIT_BYTES).unwrap(),
    );
    assert!(matches!(
        mismatched_profile,
        Err(CapturePreflightError::CalibrationMismatch(_))
    ));

    let mismatched_revision = ExpectedCaptureSettings::new(
        12,
        "camera-0",
        base_profile(),
        base_profile(),
        "test-camera-profile",
        ImageSpace::Rectified,
        2,
        4,
        Some(calibration()),
        read_completion_semantics(),
        CaptureControlRequirements::default(),
        Duration::from_millis(250),
        CaptureFrameLimit::new(EXPECTED_FRAME_LIMIT_BYTES).unwrap(),
        CaptureAggregateFrameLimit::new(EXPECTED_AGGREGATE_LIMIT_BYTES).unwrap(),
    );
    assert!(matches!(
        mismatched_revision,
        Err(CapturePreflightError::CalibrationMismatch(_))
    ));
}

#[test]
fn zero_configuration_revision_is_rejected_before_expected_settings_exist() {
    let result = ExpectedCaptureSettings::new(
        0,
        "camera-0",
        base_profile(),
        base_profile(),
        "test-camera-profile",
        ImageSpace::Rectified,
        2,
        3,
        Some(calibration()),
        read_completion_semantics(),
        CaptureControlRequirements::default(),
        Duration::from_millis(250),
        CaptureFrameLimit::new(EXPECTED_FRAME_LIMIT_BYTES).unwrap(),
        CaptureAggregateFrameLimit::new(EXPECTED_AGGREGATE_LIMIT_BYTES).unwrap(),
    );

    assert_eq!(
        result.unwrap_err(),
        CapturePreflightError::InvalidRevision("configuration_revision")
    );
}

#[test]
fn invalid_memory_limits_and_zero_frame_age_are_rejected() {
    assert!(matches!(
        expected_settings(
            "camera-0",
            base_profile(),
            base_profile(),
            CaptureControlRequirements::default(),
            read_completion_semantics(),
            1_000_000,
            EXPECTED_AGGREGATE_LIMIT_BYTES,
        ),
        Err(CapturePreflightError::FrameExceedsConfiguredLimit { .. })
    ));

    assert!(matches!(
        expected_settings(
            "camera-0",
            base_profile(),
            base_profile(),
            CaptureControlRequirements::default(),
            read_completion_semantics(),
            EXPECTED_FRAME_LIMIT_BYTES,
            1_400_000,
        ),
        Err(CapturePreflightError::FrameExceedsAggregateLimit { .. })
    ));

    let zero_age = ExpectedCaptureSettings::new(
        12,
        "camera-0",
        base_profile(),
        base_profile(),
        "test-camera-profile",
        ImageSpace::Rectified,
        2,
        3,
        Some(calibration()),
        read_completion_semantics(),
        CaptureControlRequirements::default(),
        Duration::ZERO,
        CaptureFrameLimit::new(EXPECTED_FRAME_LIMIT_BYTES).unwrap(),
        CaptureAggregateFrameLimit::new(EXPECTED_AGGREGATE_LIMIT_BYTES).unwrap(),
    );
    assert_eq!(
        zero_age.err(),
        Some(CapturePreflightError::ZeroMaximumFrameAge)
    );
}

#[test]
fn aggregate_limit_can_be_below_per_frame_ceiling_when_mode_fits_both() {
    let aggregate_limit_bytes = 1_600_000;
    let expected = expected_settings(
        "camera-0",
        base_profile(),
        base_profile(),
        CaptureControlRequirements::default(),
        read_completion_semantics(),
        EXPECTED_FRAME_LIMIT_BYTES,
        aggregate_limit_bytes,
    )
    .expect("profile frame size fits both independently configured limits");

    assert!(
        expected.aggregate_frame_limit().maximum_rgb8_bytes()
            < expected.frame_limit().maximum_rgb8_bytes()
    );
    let report = opened_report(default_metadata());
    expected.validate_session(&report).unwrap();
}

#[test]
fn preflighted_start_validates_before_read_and_binds_compatible_status_to_session() {
    let _runtime_guard = RUNTIME_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let mut runtime = new_runtime();
    let (factory, counts) = immediate_factory(default_metadata());
    let expected = expected_settings(
        "camera-0",
        base_profile(),
        base_profile(),
        CaptureControlRequirements::default(),
        read_completion_semantics(),
        EXPECTED_FRAME_LIMIT_BYTES,
        EXPECTED_AGGREGATE_LIMIT_BYTES,
    )
    .unwrap();

    let session_id = runtime.start_preflighted(expected, factory).unwrap();
    let status = match runtime
        .wait_for_preflight(Duration::from_secs(2))
        .expect("preflight status exists")
    {
        CapturePreflightWait::Ready(status) => status,
        CapturePreflightWait::TimedOut(status) => panic!("preflight timed out: {status:?}"),
    };
    let CapturePreflightStatus::Compatible(compatibility) = status else {
        panic!("expected compatible source metadata, got {status:?}");
    };
    assert_eq!(compatibility.session_id(), session_id);
    assert_eq!(runtime.preflight_status().unwrap().session_id(), session_id);

    let CaptureRuntimeWait::Completed(report) = runtime.wait_timeout(Duration::from_secs(2)) else {
        panic!("immediate source did not finish");
    };
    assert_eq!(
        report.end,
        billiards_perception::capture::CaptureWorkerEnd::NaturalEnd
    );
    assert_eq!(counts.opens.load(Ordering::SeqCst), 1);
    assert_eq!(counts.reads.load(Ordering::SeqCst), 1);
    assert_eq!(counts.releases.load(Ordering::SeqCst), 1);
    assert_eq!(runtime.owner_status(), CaptureOwnerStatus::Available);
}

#[test]
fn required_control_readback_mismatch_rejects_source_before_first_read() {
    let _runtime_guard = RUNTIME_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let pixel_format = billiards_perception::capture::CapturePixelFormat::new("RGB24").unwrap();
    let capabilities = CaptureSourceCapabilities::new(
        Some(vec![pixel_format]),
        CaptureCapability::Unknown,
        CaptureCapability::Supported,
        CaptureCapability::Supported,
        CaptureCapability::Unknown,
    )
    .unwrap();
    let applied_controls = CaptureAppliedCameraControls::new(
        CaptureControlReadback::Applied(CaptureExposureMode::Automatic),
        CaptureControlReadback::Unknown,
        CaptureControlReadback::Applied(2.0),
        CaptureControlReadback::Unknown,
        CaptureControlReadback::Unknown,
        CaptureControlReadback::Unknown,
        CaptureControlReadback::Unknown,
        CaptureControlReadback::Unknown,
    )
    .unwrap();
    let metadata = source_metadata_with_reported_controls(
        "test-adapter",
        "camera-0",
        base_profile(),
        base_profile(),
        capabilities,
        applied_controls,
        read_completion_semantics(),
    );
    let (factory, counts) = immediate_factory(metadata);
    let expected = expected_settings(
        "camera-0",
        base_profile(),
        base_profile(),
        CaptureControlRequirements {
            automatic_exposure: true,
            gain_control: true,
            ..CaptureControlRequirements::default()
        },
        read_completion_semantics(),
        EXPECTED_FRAME_LIMIT_BYTES,
        EXPECTED_AGGREGATE_LIMIT_BYTES,
    )
    .unwrap()
    .with_requested_control_values(
        RequestedCaptureControlValues::new(CaptureExposureMode::Automatic, None, 1.0).unwrap(),
    )
    .unwrap();
    let mut runtime = new_runtime();
    runtime.start_preflighted(expected, factory).unwrap();
    let mut receiver = runtime.claim_receiver().unwrap();
    match runtime.wait_for_preflight(Duration::from_secs(2)).unwrap() {
        CapturePreflightWait::Ready(CapturePreflightStatus::Rejected { error, .. }) => {
            assert_eq!(
                error,
                CapturePreflightError::RequiredControlReadbackMismatch("gain")
            );
        }
        other => panic!("expected applied-control rejection, got {other:?}"),
    }
    let CaptureRuntimeWait::Completed(report) = runtime.wait_timeout(Duration::from_secs(2)) else {
        panic!("source did not finish after preflight rejection");
    };
    assert_eq!(report.frames_published, 0);
    assert_eq!(counts.reads.load(Ordering::SeqCst), 0);
    assert_eq!(counts.releases.load(Ordering::SeqCst), 1);
    assert!(matches!(
        receiver.receive_with_timeout(Duration::from_millis(1), |_| ()),
        CaptureReceiveOutcome::ProducerFailed(_)
    ));
}

#[test]
fn every_opened_report_mismatch_fails_before_reading_or_publishing() {
    let _runtime_guard = RUNTIME_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let mut runtime = new_runtime();
    let required_manual = CaptureControlRequirements {
        manual_exposure: true,
        ..CaptureControlRequirements::default()
    };
    let unsupported_manual_metadata = source_metadata(
        "camera-0",
        base_profile(),
        base_profile(),
        CaptureCapability::Unsupported,
        read_completion_semantics(),
        CaptureCapability::Unknown,
    );
    let cases = [
        (
            metadata_with_geometry(
                CaptureAppliedFrameGeometry::new(
                    11,
                    "test-camera-profile",
                    ImageSpace::Rectified,
                    2,
                )
                .unwrap(),
            ),
            expected_settings(
                "camera-0",
                base_profile(),
                base_profile(),
                CaptureControlRequirements::default(),
                read_completion_semantics(),
                EXPECTED_FRAME_LIMIT_BYTES,
                EXPECTED_AGGREGATE_LIMIT_BYTES,
            )
            .unwrap(),
            CapturePreflightError::ConfigurationRevisionMismatch,
        ),
        (
            default_metadata(),
            expected_settings(
                "other-camera",
                base_profile(),
                base_profile(),
                CaptureControlRequirements::default(),
                read_completion_semantics(),
                EXPECTED_FRAME_LIMIT_BYTES,
                EXPECTED_AGGREGATE_LIMIT_BYTES,
            )
            .unwrap(),
            CapturePreflightError::DeviceMismatch,
        ),
        (
            default_metadata(),
            expected_settings(
                "camera-0",
                profile(800, 500, "RGB24"),
                base_profile(),
                CaptureControlRequirements::default(),
                read_completion_semantics(),
                EXPECTED_FRAME_LIMIT_BYTES,
                EXPECTED_AGGREGATE_LIMIT_BYTES,
            )
            .unwrap(),
            CapturePreflightError::RequestedProfileMismatch,
        ),
        (
            default_metadata(),
            expected_settings(
                "camera-0",
                base_profile(),
                profile(800, 500, "RGB24"),
                CaptureControlRequirements::default(),
                read_completion_semantics(),
                EXPECTED_FRAME_LIMIT_BYTES,
                EXPECTED_AGGREGATE_LIMIT_BYTES,
            )
            .unwrap(),
            CapturePreflightError::NegotiatedProfileMismatch,
        ),
        (
            default_metadata(),
            expected_settings(
                "camera-0",
                base_profile(),
                base_profile(),
                CaptureControlRequirements::default(),
                source_timestamp_semantics(),
                EXPECTED_FRAME_LIMIT_BYTES,
                EXPECTED_AGGREGATE_LIMIT_BYTES,
            )
            .unwrap(),
            CapturePreflightError::TimestampSemanticsMismatch,
        ),
        (
            unsupported_manual_metadata,
            expected_settings(
                "camera-0",
                base_profile(),
                base_profile(),
                required_manual,
                read_completion_semantics(),
                EXPECTED_FRAME_LIMIT_BYTES,
                EXPECTED_AGGREGATE_LIMIT_BYTES,
            )
            .unwrap(),
            CapturePreflightError::RequiredCapabilityUnavailable {
                capability: "manual_exposure",
                reported: CaptureCapability::Unsupported,
            },
        ),
    ];

    for (metadata, expected, expected_error) in cases {
        let (factory, counts) = immediate_factory(metadata);
        runtime.start_preflighted(expected, factory).unwrap();
        let mut receiver = runtime.claim_receiver().unwrap();
        match runtime.wait_for_preflight(Duration::from_secs(2)).unwrap() {
            CapturePreflightWait::Ready(CapturePreflightStatus::Rejected { error, .. })
                if error == expected_error => {}
            other => panic!("expected rejection {expected_error:?}, got {other:?}"),
        }
        let CaptureRuntimeWait::Completed(report) = runtime.wait_timeout(Duration::from_secs(2))
        else {
            panic!("rejected source did not release and finish");
        };
        assert_eq!(
            report.end,
            billiards_perception::capture::CaptureWorkerEnd::Failed
        );
        assert_eq!(report.frames_published, 0);
        assert_eq!(counts.opens.load(Ordering::SeqCst), 1);
        assert_eq!(counts.reads.load(Ordering::SeqCst), 0);
        assert_eq!(counts.releases.load(Ordering::SeqCst), 1);
        assert_eq!(runtime.owner_status(), CaptureOwnerStatus::Available);
        assert!(matches!(
            receiver.receive_with_timeout(Duration::from_millis(1), |_| ()),
            CaptureReceiveOutcome::ProducerFailed(_)
        ));
    }
}

#[test]
fn exact_coordinator_memory_limits_are_checked_before_source_start() {
    let _runtime_guard = RUNTIME_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let mut runtime = new_runtime();
    for (frame_limit_bytes, aggregate_limit_bytes, frame_limit_mismatch) in [
        (
            EXPECTED_FRAME_LIMIT_BYTES + 1,
            EXPECTED_AGGREGATE_LIMIT_BYTES,
            true,
        ),
        (
            EXPECTED_FRAME_LIMIT_BYTES,
            EXPECTED_AGGREGATE_LIMIT_BYTES + 1,
            false,
        ),
    ] {
        let expected = expected_settings(
            "camera-0",
            base_profile(),
            base_profile(),
            CaptureControlRequirements::default(),
            read_completion_semantics(),
            frame_limit_bytes,
            aggregate_limit_bytes,
        )
        .unwrap();
        let (factory, counts) = immediate_factory(default_metadata());
        let result = runtime.start_preflighted(expected, factory);
        match result {
            Err(
                billiards_perception::capture::CaptureStartError::PreflightFrameLimitMismatch {
                    expected_bytes,
                    coordinator_bytes,
                },
            ) if frame_limit_mismatch => {
                assert_eq!(expected_bytes, EXPECTED_FRAME_LIMIT_BYTES + 1);
                assert_eq!(coordinator_bytes, EXPECTED_FRAME_LIMIT_BYTES);
            }
            Err(
                billiards_perception::capture::CaptureStartError::PreflightAggregateLimitMismatch {
                    expected_bytes,
                    coordinator_bytes,
                },
            ) if !frame_limit_mismatch => {
                assert_eq!(expected_bytes, EXPECTED_AGGREGATE_LIMIT_BYTES + 1);
                assert_eq!(coordinator_bytes, EXPECTED_AGGREGATE_LIMIT_BYTES);
            }
            other => panic!("expected exact runtime memory-limit rejection, got {other:?}"),
        }
        assert_eq!(counts.opens.load(Ordering::SeqCst), 0);
        assert_eq!(counts.reads.load(Ordering::SeqCst), 0);
        assert_eq!(counts.releases.load(Ordering::SeqCst), 0);
        assert_eq!(runtime.owner_status(), CaptureOwnerStatus::Available);
    }
}

#[test]
fn readiness_timeout_keeps_worker_and_owner_gate_until_cancelled_open_exits() {
    let _runtime_guard = RUNTIME_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let mut runtime = new_runtime();
    let expected = expected_settings(
        "camera-0",
        base_profile(),
        base_profile(),
        CaptureControlRequirements::default(),
        read_completion_semantics(),
        EXPECTED_FRAME_LIMIT_BYTES,
        EXPECTED_AGGREGATE_LIMIT_BYTES,
    )
    .unwrap();
    let interrupt = Arc::new(InterruptState {
        interrupted: Mutex::new(false),
        changed: Condvar::new(),
    });
    let (open_started_sender, open_started_receiver) = std::sync::mpsc::channel();
    let session_id = runtime
        .start_preflighted(
            expected,
            BlockingOpenFactory {
                interrupt,
                open_started: open_started_sender,
            },
        )
        .unwrap();
    open_started_receiver
        .recv_timeout(Duration::from_secs(2))
        .expect("worker is blocked inside source open");

    match runtime
        .wait_for_preflight(Duration::from_millis(20))
        .unwrap()
    {
        CapturePreflightWait::TimedOut(CapturePreflightStatus::Pending {
            session_id: pending_id,
        }) => assert_eq!(pending_id, session_id),
        other => panic!("expected a bounded pending result, got {other:?}"),
    }
    assert_eq!(
        runtime.owner_status(),
        CaptureOwnerStatus::Owned {
            session_id: session_id.clone()
        }
    );

    let mut replacement = new_runtime();
    let (replacement_factory, _) = immediate_factory(default_metadata());
    assert!(matches!(
        replacement.start_unvalidated(replacement_factory),
        Err(billiards_perception::capture::CaptureStartError::OwnerGate(
            _
        ))
    ));

    let CaptureRuntimeWait::Completed(report) = runtime.stop(Duration::from_secs(2)) else {
        panic!("interruptible open did not exit after cancellation");
    };
    assert_eq!(
        report.end,
        billiards_perception::capture::CaptureWorkerEnd::Stopped
    );
    assert_eq!(
        runtime.preflight_status().unwrap().session_id(),
        session_id.as_str()
    );
    assert!(matches!(
        runtime.preflight_status().unwrap(),
        CapturePreflightStatus::Stopped { .. }
    ));
    assert_eq!(runtime.owner_status(), CaptureOwnerStatus::Available);
}

#[test]
fn cancellation_racing_a_successful_open_sets_stopped_preflight_without_metadata_or_read() {
    let _runtime_guard = RUNTIME_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let mut runtime = new_runtime();
    let expected = expected_settings(
        "camera-0",
        base_profile(),
        base_profile(),
        CaptureControlRequirements::default(),
        read_completion_semantics(),
        EXPECTED_FRAME_LIMIT_BYTES,
        EXPECTED_AGGREGATE_LIMIT_BYTES,
    )
    .unwrap();
    let interrupt = Arc::new(InterruptState {
        interrupted: Mutex::new(false),
        changed: Condvar::new(),
    });
    let counts = Arc::new(ImmediateCounts::default());
    let (open_started_sender, open_started_receiver) = std::sync::mpsc::channel();
    runtime
        .start_preflighted(
            expected,
            SuccessfulOpenAfterInterruptFactory {
                metadata: default_metadata(),
                counts: Arc::clone(&counts),
                interrupt,
                open_started: open_started_sender,
            },
        )
        .unwrap();
    open_started_receiver
        .recv_timeout(Duration::from_secs(2))
        .expect("factory is blocked in open before cancellation");

    let CaptureRuntimeWait::Completed(report) = runtime.stop(Duration::from_secs(2)) else {
        panic!("successful-open cancellation race did not finish");
    };
    assert_eq!(
        report.end,
        billiards_perception::capture::CaptureWorkerEnd::Stopped
    );
    assert!(matches!(
        runtime.preflight_status().unwrap(),
        CapturePreflightStatus::Stopped { .. }
    ));
    assert_eq!(counts.opens.load(Ordering::SeqCst), 1);
    assert_eq!(counts.reads.load(Ordering::SeqCst), 0);
    assert_eq!(counts.releases.load(Ordering::SeqCst), 1);
    assert_eq!(runtime.owner_status(), CaptureOwnerStatus::Available);
}

#[test]
fn open_failure_is_reported_as_preflight_failure_and_wakes_receiver() {
    let _runtime_guard = RUNTIME_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let mut runtime = new_runtime();
    let expected = expected_settings(
        "camera-0",
        base_profile(),
        base_profile(),
        CaptureControlRequirements::default(),
        read_completion_semantics(),
        EXPECTED_FRAME_LIMIT_BYTES,
        EXPECTED_AGGREGATE_LIMIT_BYTES,
    )
    .unwrap();
    let (mut factory, counts) = immediate_factory(default_metadata());
    factory.open_error = Some("device unavailable".to_owned());
    runtime.start_preflighted(expected, factory).unwrap();
    let mut receiver = runtime.claim_receiver().unwrap();
    assert!(matches!(
        runtime.wait_for_preflight(Duration::from_secs(2)).unwrap(),
        CapturePreflightWait::Ready(CapturePreflightStatus::Failed { .. })
    ));
    let CaptureRuntimeWait::Completed(report) = runtime.wait_timeout(Duration::from_secs(2)) else {
        panic!("open failure did not finish");
    };
    assert_eq!(
        report.end,
        billiards_perception::capture::CaptureWorkerEnd::Failed
    );
    assert_eq!(counts.opens.load(Ordering::SeqCst), 1);
    assert_eq!(counts.reads.load(Ordering::SeqCst), 0);
    assert_eq!(counts.releases.load(Ordering::SeqCst), 0);
    assert!(matches!(
        receiver.receive_with_timeout(Duration::from_millis(1), |_| ()),
        CaptureReceiveOutcome::ProducerFailed(_)
    ));
    assert_eq!(runtime.owner_status(), CaptureOwnerStatus::Available);
}

#[test]
fn preflight_rejection_wakes_receiver_before_slow_source_release_finishes() {
    let _runtime_guard = RUNTIME_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let mut runtime = new_runtime();
    let expected = expected_settings(
        "other-camera",
        base_profile(),
        base_profile(),
        CaptureControlRequirements::default(),
        read_completion_semantics(),
        EXPECTED_FRAME_LIMIT_BYTES,
        EXPECTED_AGGREGATE_LIMIT_BYTES,
    )
    .unwrap();
    let (release_started_sender, release_started_receiver) = std::sync::mpsc::channel();
    let (allow_release_sender, allow_release_receiver) = std::sync::mpsc::channel();
    runtime
        .start_preflighted(
            expected,
            BlockingReleaseFactory {
                metadata: default_metadata(),
                release_started: release_started_sender,
                allow_release: allow_release_receiver,
            },
        )
        .unwrap();
    let mut receiver = runtime.claim_receiver().unwrap();
    assert!(matches!(
        runtime.wait_for_preflight(Duration::from_secs(2)).unwrap(),
        CapturePreflightWait::Ready(CapturePreflightStatus::Rejected {
            error: CapturePreflightError::DeviceMismatch,
            ..
        })
    ));
    release_started_receiver
        .recv_timeout(Duration::from_secs(2))
        .expect("worker has entered source release and remains owned");

    assert!(matches!(
        receiver.receive_with_timeout(Duration::from_millis(50), |_| ()),
        CaptureReceiveOutcome::ProducerFailed(_)
    ));
    assert!(matches!(
        runtime.owner_status(),
        CaptureOwnerStatus::Owned { .. }
    ));
    allow_release_sender.send(()).unwrap();
    assert!(matches!(
        runtime.wait_timeout(Duration::from_secs(2)),
        CaptureRuntimeWait::Completed(_)
    ));
    assert_eq!(runtime.owner_status(), CaptureOwnerStatus::Available);
}

#[test]
fn preflighted_session_rejects_frame_dimensions_that_disagree_with_negotiated_profile() {
    let _runtime_guard = RUNTIME_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let mut runtime = new_runtime();
    let counts = Arc::new(ImmediateCounts::default());
    let expected = expected_settings(
        "camera-0",
        base_profile(),
        base_profile(),
        CaptureControlRequirements::default(),
        read_completion_semantics(),
        EXPECTED_FRAME_LIMIT_BYTES,
        EXPECTED_AGGREGATE_LIMIT_BYTES,
    )
    .unwrap();
    runtime
        .start_preflighted(
            expected,
            OneFrameFactory {
                metadata: default_metadata(),
                counts: Arc::clone(&counts),
                frame_width_px: 16,
                frame_height_px: 16,
            },
        )
        .unwrap();
    let mut receiver = runtime.claim_receiver().unwrap();
    assert!(matches!(
        runtime.wait_for_preflight(Duration::from_secs(2)).unwrap(),
        CapturePreflightWait::Ready(CapturePreflightStatus::Compatible(_))
    ));
    let CaptureRuntimeWait::Completed(report) = runtime.wait_timeout(Duration::from_secs(2)) else {
        panic!("wrong-size frame did not fail and release");
    };
    assert_eq!(
        report.end,
        billiards_perception::capture::CaptureWorkerEnd::Failed
    );
    assert_eq!(report.frames_published, 0);
    assert!(report.failures.iter().any(|failure| matches!(
        failure,
        billiards_perception::capture::CaptureWorkerFailure::FrameProfileMismatch {
            expected_width_px: 1000,
            expected_height_px: 500,
            actual_width_px: 16,
            actual_height_px: 16,
        }
    )));
    assert_eq!(counts.reads.load(Ordering::SeqCst), 1);
    assert_eq!(counts.releases.load(Ordering::SeqCst), 1);
    assert!(matches!(
        receiver.receive_with_timeout(Duration::from_millis(1), |_| ()),
        CaptureReceiveOutcome::ProducerFailed(_)
    ));
    assert_eq!(runtime.owner_status(), CaptureOwnerStatus::Available);
}

#[test]
fn v2_capture_worker_rejects_frames_outside_acknowledged_output_geometry() {
    let _runtime_guard = RUNTIME_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let mut runtime = new_runtime();
    let counts = Arc::new(ImmediateCounts::default());
    let input_profile = base_profile();
    let expected = ExpectedCaptureSettings::new_with_output_frame_geometry(
        12,
        "camera-0",
        input_profile.clone(),
        input_profile,
        CaptureOutputFrameGeometry::new(16, 16).unwrap(),
        "test-camera-profile",
        ImageSpace::Rectified,
        2,
        3,
        None,
        read_completion_semantics(),
        CaptureControlRequirements::default(),
        Duration::from_millis(250),
        CaptureFrameLimit::new(EXPECTED_FRAME_LIMIT_BYTES).unwrap(),
        CaptureAggregateFrameLimit::new(EXPECTED_AGGREGATE_LIMIT_BYTES).unwrap(),
    )
    .unwrap();
    let metadata = default_metadata()
        .with_output_frame_geometry(CaptureOutputFrameGeometry::new(16, 16).unwrap());
    runtime
        .start_preflighted(
            expected,
            OneFrameFactory {
                metadata,
                counts: Arc::clone(&counts),
                frame_width_px: 15,
                frame_height_px: 16,
            },
        )
        .unwrap();
    let mut receiver = runtime.claim_receiver().unwrap();
    assert!(matches!(
        runtime.wait_for_preflight(Duration::from_secs(2)).unwrap(),
        CapturePreflightWait::Ready(CapturePreflightStatus::Compatible(_))
    ));
    let CaptureRuntimeWait::Completed(report) = runtime.wait_timeout(Duration::from_secs(2)) else {
        panic!("v2 wrong-geometry frame did not fail and release");
    };
    assert!(report.failures.iter().any(|failure| matches!(
        failure,
        billiards_perception::capture::CaptureWorkerFailure::FrameProfileMismatch {
            expected_width_px: 16,
            expected_height_px: 16,
            actual_width_px: 15,
            actual_height_px: 16,
        }
    )));
    assert_eq!(report.frames_published, 0);
    assert_eq!(counts.releases.load(Ordering::SeqCst), 1);
    assert!(matches!(
        receiver.receive_with_timeout(Duration::from_millis(1), |_| ()),
        CaptureReceiveOutcome::ProducerFailed(_)
    ));
}

#[test]
fn report_and_compatibility_data_have_no_authentication_fields() {
    let report = opened_report(default_metadata());
    let json = serde_json::to_value(&report).unwrap();
    assert_eq!(json["applied_frame_geometry"]["configuration_revision"], 12);
    assert_eq!(
        json["applied_frame_geometry"]["camera_profile_id"],
        "test-camera-profile"
    );
    assert_eq!(json["applied_frame_geometry"]["image_space"], "rectified");
    assert_eq!(json["applied_frame_geometry"]["rectification_revision"], 2);
    let keys = json
        .as_object()
        .unwrap()
        .keys()
        .map(String::as_str)
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(
        keys,
        [
            "applied_controls",
            "applied_frame_geometry",
            "backend_id",
            "capabilities",
            "device_id",
            "device_name",
            "negotiated_profile",
            "requested_profile",
            "session_id",
            "timestamp_semantics",
        ]
        .into_iter()
        .collect()
    );

    let expected = expected_settings(
        "camera-0",
        base_profile(),
        base_profile(),
        CaptureControlRequirements::default(),
        read_completion_semantics(),
        EXPECTED_FRAME_LIMIT_BYTES,
        EXPECTED_AGGREGATE_LIMIT_BYTES,
    )
    .unwrap();
    let compatibility: CaptureSessionCompatibility = expected.validate_session(&report).unwrap();
    assert_eq!(compatibility.session_id(), report.session_id());
    assert_eq!(
        compatibility
            .applied_frame_geometry()
            .configuration_revision(),
        12
    );
    assert_eq!(
        compatibility.applied_frame_geometry().camera_profile_id(),
        "test-camera-profile"
    );
    assert_eq!(
        compatibility.applied_frame_geometry().image_space(),
        ImageSpace::Rectified
    );
}
