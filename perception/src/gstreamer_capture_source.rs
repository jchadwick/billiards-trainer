//! Feature-gated adapter for supervised GStreamer capture helpers.
//!
//! This source exercises the capture worker, metadata handshake, bounded frame
//! allocation, and subprocess lifecycle. The default constructor starts only a
//! synthetic source. A separately gated constructor can select one physical
//! camera by a stable discovery selector and use driver defaults without
//! writing controls. The physical constructor is available to the live
//! executable only with `camera-gstreamer-source` and an explicit source flag;
//! the executable's normal source remains unavailable.

use std::error::Error;
use std::fmt::{Display, Formatter};
use std::io::Write;
use std::path::PathBuf;
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex, MutexGuard};
use std::thread;
use std::time::{Duration, Instant};

use crate::capture::{
    CaptureAppliedCameraControls, CaptureAppliedFrameGeometry, CaptureCapability,
    CaptureControlReadback, CaptureFrameBudget, CaptureFrameLimit, CaptureFrameRate,
    CaptureFrameTiming, CaptureOpenError, CaptureOutputFrameGeometry, CapturePixelFormat,
    CaptureReadOutcome, CaptureSource, CaptureSourceCapabilities, CaptureSourceFactory,
    CaptureSourceMetadata, CaptureTimestampSemantics, CaptureVideoProfile, RgbFrameEnvelope,
};
use crate::capture_helper_protocol::{
    read_ready_record, CaptureHelperReady, FrameBinding, FrameBindingValidator, FrameRecordReader,
};
use crate::capture_settings_wire::{parse_capture_settings_json, ResolvedCaptureSettings};
use crate::detection_batch::SourceFrame;
use crate::ImageSpace;

/// Request device identifier reserved for the synthetic GStreamer source.
pub const SYNTHETIC_GSTREAMER_DEVICE_ID: &str = "synthetic-gstreamer-videotestsrc";
/// Stable provenance label for the explicit remote stream source. The URL is
/// launch configuration and is never treated as a physical camera selector.
pub const HTTP_MJPEG_STREAM_DEVICE_ID: &str = "remote-http-mjpeg-stream";

/// Explicit launch settings for the opt-in synthetic helper.
///
/// Frame and memory limits are parsed from the exact capture-settings bytes;
/// the protocol header and readiness limits, subprocess timings, and run
/// duration are supplied by the caller. This type deliberately has no
/// production defaults.
#[derive(Debug, Clone)]
pub struct GstreamerCaptureSourceConfig {
    /// Executable path for `billiards-perception-gstreamer-probe`.
    pub helper_executable_path: PathBuf,
    /// Remote HTTP(S) multipart MJPEG URL for the explicit stream source.
    pub http_mjpeg_url: Option<String>,
    /// Exact serialized bytes produced by the Rust configuration authority.
    pub exact_capture_settings_json: Vec<u8>,
    /// Maximum number of request bytes the helper accepts.
    pub capture_settings_max_bytes: usize,
    /// Maximum JSON bytes in the helper-ready preface.
    pub ready_record_max_bytes: usize,
    /// Maximum JSON bytes in each frame header.
    pub frame_header_max_bytes: usize,
    /// Finite helper frame count. `None` runs a physical camera until its
    /// supervising capture worker interrupts the helper; synthetic sources
    /// must always use a finite count.
    pub frame_count: Option<u64>,
    /// Optional delay before each pull after frame zero, in milliseconds.
    pub slow_reader_delay_ms: u64,
    /// Maximum GStreamer sample-pull wait, in milliseconds.
    pub pull_timeout_ms: u64,
    /// Collect helper and parent read durations only when the live audit is
    /// active. Disabled sources do not create additional timing spans.
    pub capture_audit_timing: bool,
    /// Parent-side deadline for receiving and validating helper readiness.
    pub readiness_timeout: Duration,
    /// Maximum wait for natural helper exit after stdout reaches clean EOF.
    pub child_exit_timeout: Duration,
    /// Poll cadence used while confirming child exit.
    pub child_exit_poll_interval: Duration,
}

/// A validated, unopened synthetic or feature-gated physical helper source.
pub struct GstreamerCaptureSourceFactory {
    config: GstreamerCaptureSourceConfig,
    resolved_settings: ResolvedCaptureSettings,
    helper_mode: HelperMode,
    control: Arc<ChildControl>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HelperMode {
    Synthetic,
    #[cfg(feature = "camera-gstreamer-source")]
    Physical,
    #[cfg(feature = "camera-gstreamer-source")]
    HttpMjpeg,
}

impl HelperMode {
    fn is_physical(self) -> bool {
        match self {
            Self::Synthetic => false,
            #[cfg(feature = "camera-gstreamer-source")]
            Self::Physical => true,
            #[cfg(feature = "camera-gstreamer-source")]
            Self::HttpMjpeg => false,
        }
    }

    fn permits_unbounded_run(self) -> bool {
        match self {
            Self::Synthetic => false,
            #[cfg(feature = "camera-gstreamer-source")]
            Self::Physical | Self::HttpMjpeg => true,
        }
    }
}

impl std::fmt::Debug for GstreamerCaptureSourceFactory {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("GstreamerCaptureSourceFactory")
            .field(
                "helper_executable_path",
                &self.config.helper_executable_path,
            )
            .field("frame_count", &self.config.frame_count)
            .field(
                "child_started",
                &self.control.child_started.load(Ordering::Acquire),
            )
            .finish_non_exhaustive()
    }
}

impl GstreamerCaptureSourceFactory {
    /// Validate all explicit launch settings and parse, but do not rewrite, the
    /// exact request bytes that will be sent to the helper.
    pub fn new(
        config: GstreamerCaptureSourceConfig,
    ) -> Result<Self, GstreamerCaptureSourceConfigError> {
        Self::new_for_mode(config, HelperMode::Synthetic)
    }

    /// Construct a physical-camera helper factory. This remains opt-in and
    /// accepts only v3 source-default settings and a stable ID discovered for
    /// the current platform. Camera controls are never written by this path.
    #[cfg(feature = "camera-gstreamer-source")]
    pub fn new_physical(
        config: GstreamerCaptureSourceConfig,
    ) -> Result<Self, GstreamerCaptureSourceConfigError> {
        let factory = Self::new_for_mode(config, HelperMode::Physical)?;
        if factory.resolved_settings.schema_version != 3 {
            return Err(invalid_config(
                "physical GStreamer source requires capture-settings v3",
            ));
        }
        if !matches!(
            factory.resolved_settings.camera_controls,
            crate::capture_settings_wire::ResolvedCameraControlsPolicy::SourceDefaults
        ) {
            return Err(invalid_config(
                "physical GStreamer source supports source_defaults controls only; it never writes camera controls",
            ));
        }
        if !crate::camera_discovery::is_capture_selector(
            factory.resolved_settings.expected.device_id(),
        ) {
            return Err(invalid_config(
                "physical GStreamer source requires a gstreamer-device-v1 selector from camera discovery",
            ));
        }
        Ok(factory)
    }

    /// Construct an explicit remote HTTP-MJPEG source for development
    /// integration tests. It has a distinct capture identity and uses
    /// host-side read-completion time, never a claimed camera exposure time.
    #[cfg(feature = "camera-gstreamer-source")]
    pub fn new_http_mjpeg(
        mut config: GstreamerCaptureSourceConfig,
        url: String,
    ) -> Result<Self, GstreamerCaptureSourceConfigError> {
        if !crate::configuration_authority::valid_http_uri(&url) {
            return Err(invalid_config(
                "HTTP-MJPEG URL must be a valid credential-free HTTP(S) URI",
            ));
        }
        config.http_mjpeg_url = Some(url);
        let factory = Self::new_for_mode(config, HelperMode::HttpMjpeg)?;
        if factory.resolved_settings.schema_version != 3 {
            return Err(invalid_config(
                "HTTP-MJPEG source requires capture-settings v3",
            ));
        }
        if factory.resolved_settings.expected.device_id() != HTTP_MJPEG_STREAM_DEVICE_ID {
            return Err(invalid_config(format!(
                "HTTP-MJPEG capture settings must use source id {HTTP_MJPEG_STREAM_DEVICE_ID:?}"
            )));
        }
        if !matches!(
            factory.resolved_settings.camera_controls,
            crate::capture_settings_wire::ResolvedCameraControlsPolicy::SourceDefaults
        ) {
            return Err(invalid_config(
                "HTTP-MJPEG source supports source_defaults controls only",
            ));
        }
        Ok(factory)
    }

    fn new_for_mode(
        config: GstreamerCaptureSourceConfig,
        helper_mode: HelperMode,
    ) -> Result<Self, GstreamerCaptureSourceConfigError> {
        validate_config(&config, helper_mode)?;
        let request = std::str::from_utf8(&config.exact_capture_settings_json).map_err(|_| {
            GstreamerCaptureSourceConfigError::Invalid(
                "capture-settings JSON must be valid UTF-8".to_owned(),
            )
        })?;
        let resolved_settings = parse_capture_settings_json(request)
            .map_err(|error| GstreamerCaptureSourceConfigError::Invalid(error.to_string()))?;
        if !matches!(
            resolved_settings.requested_backend.as_str(),
            "auto" | "gstreamer"
        ) {
            return Err(GstreamerCaptureSourceConfigError::Invalid(format!(
                "GStreamer helper cannot satisfy requested backend {:?}",
                resolved_settings.requested_backend
            )));
        }
        if helper_mode == HelperMode::Synthetic
            && resolved_settings.expected.device_id() != SYNTHETIC_GSTREAMER_DEVICE_ID
        {
            return Err(invalid_config(format!(
                "synthetic source requests must use device id {SYNTHETIC_GSTREAMER_DEVICE_ID:?}"
            )));
        }
        #[cfg(feature = "camera-gstreamer-source")]
        if helper_mode == HelperMode::HttpMjpeg {
            if config.http_mjpeg_url.is_none() {
                return Err(invalid_config("HTTP-MJPEG source requires a remote URL"));
            }
        } else if config.http_mjpeg_url.is_some() {
            return Err(invalid_config(
                "HTTP-MJPEG URL is valid only for the HTTP-MJPEG source",
            ));
        }
        if resolved_settings.expected.image_space() != ImageSpace::Raw {
            return Err(GstreamerCaptureSourceConfigError::Invalid(
                "GStreamer capture helpers emit raw, unrectified frames only".to_owned(),
            ));
        }
        if helper_mode == HelperMode::Synthetic
            && resolved_settings.expected.rectification_revision() != 0
        {
            return Err(GstreamerCaptureSourceConfigError::Invalid(
                "synthetic raw frames cannot claim a non-zero rectification revision".to_owned(),
            ));
        }

        Ok(Self {
            config,
            resolved_settings,
            helper_mode,
            control: Arc::new(ChildControl::default()),
        })
    }

    /// Return a read-only lifecycle handle useful for observing whether the
    /// supervised helper has been reaped after natural completion or stop.
    pub fn monitor(&self) -> GstreamerCaptureSourceMonitor {
        GstreamerCaptureSourceMonitor(Arc::clone(&self.control))
    }
}

/// Read-only process-lifecycle status for a helper factory.
#[derive(Debug, Clone)]
pub struct GstreamerCaptureSourceMonitor(Arc<ChildControl>);

impl GstreamerCaptureSourceMonitor {
    /// Whether a child was launched and its process handle was reaped.
    pub fn child_is_reaped(&self) -> bool {
        self.0.child_started.load(Ordering::Acquire) && self.0.child_reaped.load(Ordering::Acquire)
    }
}

impl CaptureSourceFactory for GstreamerCaptureSourceFactory {
    fn interrupt_handle(&self) -> Arc<dyn crate::capture::CaptureInterrupt> {
        Arc::new(ChildInterrupt(Arc::clone(&self.control)))
    }

    fn open(self: Box<Self>) -> Result<Box<dyn CaptureSource>, CaptureOpenError> {
        Err(CaptureOpenError::Failed(
            "GStreamer helper source requires the capture worker session identity".to_owned(),
        ))
    }

    fn open_with_session_id(
        self: Box<Self>,
        session_id: &str,
    ) -> Result<Box<dyn CaptureSource>, CaptureOpenError> {
        open_for_session(*self, session_id)
    }
}

#[derive(Debug, Default)]
struct ChildControl {
    state: Mutex<ChildState>,
    interrupted: AtomicBool,
    child_started: AtomicBool,
    child_reaped: AtomicBool,
}

#[derive(Debug, Default)]
struct ChildState {
    child: Option<Child>,
}

impl ChildControl {
    fn lock_state(&self) -> MutexGuard<'_, ChildState> {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn interrupt(&self) {
        self.interrupted.store(true, Ordering::Release);
        let mut state = self.lock_state();
        if let Some(child) = state.child.as_mut() {
            // `kill` is non-blocking; the owner thread reaps the process from
            // its normal release path after the pipe read has been interrupted.
            let _ = child.kill();
        }
    }

    fn is_interrupted(&self) -> bool {
        self.interrupted.load(Ordering::Acquire)
    }

    fn kill_and_reap(&self) -> Result<(), String> {
        let mut state = self.lock_state();
        let Some(mut child) = state.child.take() else {
            return Ok(());
        };

        match child.try_wait() {
            Ok(Some(_)) => {}
            Ok(None) => {
                // A failed kill usually means the child exited in the small
                // interval after try_wait; wait still obtains its final status.
                let _ = child.kill();
            }
            Err(error) => {
                let _ = child.kill();
                let _ = child.wait();
                self.child_reaped.store(true, Ordering::Release);
                return Err(format!(
                    "could not inspect helper process before reaping: {error}"
                ));
            }
        }
        child
            .wait()
            .map_err(|error| format!("could not reap helper process: {error}"))?;
        self.child_reaped.store(true, Ordering::Release);
        Ok(())
    }

    fn wait_for_exit(
        &self,
        timeout: Duration,
        poll_interval: Duration,
    ) -> Result<ExitStatus, String> {
        let started_at = Instant::now();
        loop {
            if self.is_interrupted() {
                return Err("helper exit wait was interrupted".to_owned());
            }
            {
                let mut state = self.lock_state();
                let Some(child) = state.child.as_mut() else {
                    return Err("helper process handle was already consumed".to_owned());
                };
                match child.try_wait() {
                    Ok(Some(status)) => {
                        state.child.take();
                        self.child_reaped.store(true, Ordering::Release);
                        return Ok(status);
                    }
                    Ok(None) => {}
                    Err(error) => {
                        drop(state);
                        let _ = self.kill_and_reap();
                        return Err(format!("could not inspect helper exit status: {error}"));
                    }
                }
            }

            let remaining = timeout.saturating_sub(started_at.elapsed());
            if remaining.is_zero() {
                let _ = self.kill_and_reap();
                return Err("helper did not exit before the configured deadline".to_owned());
            }
            thread::sleep(poll_interval.min(remaining));
        }
    }
}

#[derive(Debug)]
struct ChildInterrupt(Arc<ChildControl>);

impl crate::capture::CaptureInterrupt for ChildInterrupt {
    fn interrupt(&self) {
        self.0.interrupt();
    }
}

struct GstreamerCaptureSource {
    reader: Option<FrameRecordReader<std::process::ChildStdout>>,
    validator: FrameBindingValidator,
    control: Arc<ChildControl>,
    metadata: CaptureSourceMetadata,
    capture_audit_timing: bool,
    child_exit_timeout: Duration,
    child_exit_poll_interval: Duration,
    ended: bool,
}

impl CaptureSource for GstreamerCaptureSource {
    fn metadata(&self) -> CaptureSourceMetadata {
        self.metadata.clone()
    }

    fn read(
        &mut self,
        session_id: &str,
        frame_limit: CaptureFrameLimit,
        frame_budget: &CaptureFrameBudget,
    ) -> Result<CaptureReadOutcome, String> {
        if self.ended {
            return Ok(CaptureReadOutcome::End);
        }
        let Some(reader) = self.reader.as_mut() else {
            self.ended = true;
            return Ok(CaptureReadOutcome::End);
        };

        let parent_read_started = self.capture_audit_timing.then(Instant::now);
        let record = reader.read_next_with_buffer(|header, payload_len| {
            // Validate all untrusted identity and geometry before reserving
            // application-owned image memory. The allocation closure runs only
            // after the framed length has also passed the configured bound.
            self.validator
                .validate(header, payload_len)
                .map_err(|error| error.to_string())?;
            if header.source_timestamp_ticks.is_some() {
                return Err("synthetic helper claimed an unverified source timestamp".to_owned());
            }
            if let Some(helper_elapsed) = header.host_read_completed_elapsed_ns.as_deref() {
                helper_elapsed
                    .parse::<u128>()
                    .map_err(|_| "legacy helper elapsed value is not a decimal u128".to_owned())?;
            }
            if header
                .unverified_pipeline_pts_ticks
                .as_deref()
                .is_some_and(|ticks| ticks.parse::<u64>().is_err())
            {
                return Err("helper pipeline PTS is not a decimal u64".to_owned());
            }
            let schema_matches_audit_mode = if self.capture_audit_timing {
                header.record_schema_version == 3
                    && header.capture_timings_ns.is_some()
                    && header.host_read_completed_elapsed_ns.is_none()
            } else {
                header.record_schema_version == 2
                    && header.capture_timings_ns.is_none()
                    && header.host_read_completed_elapsed_ns.is_some()
            };
            if !schema_matches_audit_mode {
                return Err(
                    "helper capture timing does not match the configured audit mode".to_owned(),
                );
            }
            frame_budget
                .allocate_rgb8_buffer(header.width_px, header.height_px, frame_limit)
                .map_err(|error| error.to_string())
        });
        let parent_framed_record_read_ns = elapsed_nanos(parent_read_started);

        match record {
            Ok(Some(record)) => {
                let host_read_completed_at = Instant::now();
                if record.header.session_id != session_id {
                    return Err(
                        "capture worker session identity changed during helper read".to_owned()
                    );
                }
                let frame = RgbFrameEnvelope::new_with_capture_timing(
                    SourceFrame {
                        session_id: record.header.session_id,
                        frame_id: record.header.frame_id,
                        timestamp: None,
                        width_px: record.header.width_px,
                        height_px: record.header.height_px,
                    },
                    record.payload,
                    host_read_completed_at,
                    CaptureFrameTiming {
                        helper: record.header.capture_timings_ns,
                        parent_framed_record_read_ns,
                        parent_receiver_queue_wait_ns: None,
                    },
                    frame_limit,
                )
                .map_err(|error| error.to_string())?;
                Ok(CaptureReadOutcome::Frame(frame))
            }
            Ok(None) => {
                self.reader.take();
                if self.control.is_interrupted() {
                    self.ended = true;
                    return Ok(CaptureReadOutcome::End);
                }
                let status = match self
                    .control
                    .wait_for_exit(self.child_exit_timeout, self.child_exit_poll_interval)
                {
                    Ok(status) => status,
                    Err(_) if self.control.is_interrupted() => {
                        self.ended = true;
                        return Ok(CaptureReadOutcome::End);
                    }
                    Err(error) => return Err(error),
                };
                if !status.success() {
                    return Err(format!(
                        "synthetic helper exited unsuccessfully with status {status}"
                    ));
                }
                self.ended = true;
                Ok(CaptureReadOutcome::End)
            }
            Err(_error) if self.control.is_interrupted() => {
                self.reader.take();
                self.ended = true;
                Ok(CaptureReadOutcome::End)
            }
            Err(error) => Err(error.to_string()),
        }
    }

    fn release(&mut self) -> Result<(), String> {
        self.reader.take();
        self.ended = true;
        self.control.kill_and_reap()
    }
}

fn validate_config(
    config: &GstreamerCaptureSourceConfig,
    helper_mode: HelperMode,
) -> Result<(), GstreamerCaptureSourceConfigError> {
    let positive_limits = [
        (config.capture_settings_max_bytes, "capture-settings byte"),
        (config.ready_record_max_bytes, "ready-record byte"),
        (config.frame_header_max_bytes, "frame-header byte"),
    ];
    if config.helper_executable_path.as_os_str().is_empty() {
        return Err(invalid_config("helper executable path must not be empty"));
    }
    for (limit, label) in positive_limits {
        if limit == 0 {
            return Err(invalid_config(format!("{label} limit must be positive")));
        }
    }
    if config.exact_capture_settings_json.is_empty() {
        return Err(invalid_config("capture-settings request must not be empty"));
    }
    if config.exact_capture_settings_json.len() > config.capture_settings_max_bytes {
        return Err(invalid_config(format!(
            "capture-settings request has {} bytes; configured maximum is {}",
            config.exact_capture_settings_json.len(),
            config.capture_settings_max_bytes
        )));
    }
    match config.frame_count {
        Some(0) => return Err(invalid_config("helper frame count must be positive")),
        None if !helper_mode.permits_unbounded_run() => {
            return Err(invalid_config(
                "synthetic GStreamer helpers require a finite frame count",
            ));
        }
        Some(_) | None => {}
    }
    if config.pull_timeout_ms == 0 {
        return Err(invalid_config("helper pull timeout must be positive"));
    }
    if config.slow_reader_delay_ms >= config.pull_timeout_ms {
        return Err(invalid_config(
            "slow-reader delay must be shorter than the helper pull timeout",
        ));
    }
    if config.readiness_timeout.is_zero() {
        return Err(invalid_config("readiness timeout must be positive"));
    }
    if config.child_exit_timeout.is_zero() {
        return Err(invalid_config("child exit timeout must be positive"));
    }
    if config.child_exit_poll_interval.is_zero() {
        return Err(invalid_config("child exit poll interval must be positive"));
    }
    Ok(())
}

fn invalid_config(message: impl Into<String>) -> GstreamerCaptureSourceConfigError {
    GstreamerCaptureSourceConfigError::Invalid(message.into())
}

/// Invalid synthetic-helper configuration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GstreamerCaptureSourceConfigError {
    Invalid(String),
}

impl Display for GstreamerCaptureSourceConfigError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Invalid(message) => formatter.write_str(message),
        }
    }
}

impl Error for GstreamerCaptureSourceConfigError {}

fn open_for_session(
    factory: GstreamerCaptureSourceFactory,
    session_id: &str,
) -> Result<Box<dyn CaptureSource>, CaptureOpenError> {
    let GstreamerCaptureSourceFactory {
        config,
        resolved_settings,
        helper_mode,
        control,
    } = factory;
    if control.is_interrupted() {
        return Err(CaptureOpenError::Interrupted);
    }

    let open_result = open_child_source(
        &config,
        &resolved_settings,
        helper_mode,
        &control,
        session_id,
    );
    match open_result {
        Ok(source) => Ok(Box::new(source)),
        Err(error) => {
            let cleanup_error = control.kill_and_reap().err();
            if control.is_interrupted() {
                return Err(CaptureOpenError::Interrupted);
            }
            let message = match cleanup_error {
                Some(cleanup_error) => format!("{error}; cleanup also failed: {cleanup_error}"),
                None => error,
            };
            Err(CaptureOpenError::Failed(message))
        }
    }
}

fn open_child_source(
    config: &GstreamerCaptureSourceConfig,
    resolved: &ResolvedCaptureSettings,
    helper_mode: HelperMode,
    control: &Arc<ChildControl>,
    session_id: &str,
) -> Result<GstreamerCaptureSource, String> {
    let exact_settings = std::str::from_utf8(&config.exact_capture_settings_json)
        .map_err(|error| format!("capture-settings request is not UTF-8: {error}"))?;
    // Parse at factory construction and again at the process boundary from
    // the same bytes, so accidental post-construction mutation cannot bypass
    // request validation.
    let current_resolved = parse_capture_settings_json(exact_settings)
        .map_err(|error| format!("invalid capture-settings request: {error}"))?;
    if current_resolved.request_sha256 != resolved.request_sha256 {
        return Err("capture-settings request changed after factory validation".to_owned());
    }

    let mut command = Command::new(&config.helper_executable_path);
    command
        .arg("--session-id")
        .arg(session_id)
        .arg("--capture-settings-max-bytes")
        .arg(config.capture_settings_max_bytes.to_string())
        .arg("--ready-record-max-bytes")
        .arg(config.ready_record_max_bytes.to_string());
    if config.capture_audit_timing {
        command.arg("--capture-audit-timing");
    }
    match config.frame_count {
        Some(frame_count) => {
            command.arg("--frames").arg(frame_count.to_string());
        }
        None => {
            command.arg("--until-stopped");
        }
    }
    command
        .arg("--slow-reader-ms")
        .arg(config.slow_reader_delay_ms.to_string())
        .arg("--pull-timeout-ms")
        .arg(config.pull_timeout_ms.to_string())
        .arg("--source-mode")
        .arg(match helper_mode {
            HelperMode::Synthetic => "synthetic",
            #[cfg(feature = "camera-gstreamer-source")]
            HelperMode::Physical => "physical",
            #[cfg(feature = "camera-gstreamer-source")]
            HelperMode::HttpMjpeg => "http-mjpeg",
        })
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        // Inherit stderr so diagnostics remain visible without a pipe that
        // could fill and stall frame delivery.
        .stderr(Stdio::inherit());
    #[cfg(feature = "camera-gstreamer-source")]
    if let Some(url) = config.http_mjpeg_url.as_deref() {
        command.arg("--http-mjpeg-url").arg(url);
    }

    let (mut child_stdin, child_stdout) = {
        let mut state = control
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if control.is_interrupted() {
            return Err("helper start was interrupted".to_owned());
        }
        let mut child = command
            .spawn()
            .map_err(|error| format!("could not start GStreamer helper: {error}"))?;
        control.child_started.store(true, Ordering::Release);
        control.child_reaped.store(false, Ordering::Release);
        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| "synthetic helper stdin pipe was unavailable".to_owned());
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| "synthetic helper stdout pipe was unavailable".to_owned());
        state.child = Some(child);
        (stdin?, stdout?)
    };

    if control.is_interrupted() {
        return Err("helper start was interrupted".to_owned());
    }
    child_stdin
        .write_all(&config.exact_capture_settings_json)
        .map_err(|error| format!("could not send exact capture-settings request: {error}"))?;
    drop(child_stdin);

    let (ready, child_stdout) = read_ready_with_deadline(
        child_stdout,
        config.ready_record_max_bytes,
        config.readiness_timeout,
        control,
    )?;
    validate_ready_binding(&ready, resolved, session_id, helper_mode)?;

    let output_geometry = resolved
        .expected
        .output_frame_geometry()
        .unwrap_or_else(|| {
            let profile = resolved.expected.expected_negotiated_profile();
            CaptureOutputFrameGeometry::new(profile.width_px(), profile.height_px())
                .expect("validated capture profile has non-zero geometry")
        });
    let frame_limit = resolved.expected.frame_limit();
    let binding = FrameBinding {
        session_id: session_id.to_owned(),
        configuration_revision: resolved.configuration_revision,
        camera_profile_id: resolved.expected.camera_profile_id().to_owned(),
        capture_settings_sha256: resolved.request_sha256.clone(),
        width_px: ready.width_px,
        height_px: ready.height_px,
        max_payload_bytes: frame_limit.maximum_rgb8_bytes(),
    };
    let validator = FrameBindingValidator::new(binding)
        .map_err(|error| format!("invalid helper frame binding: {error}"))?;
    let frame_reader = FrameRecordReader::new(
        child_stdout,
        config.frame_header_max_bytes,
        frame_limit.maximum_rgb8_bytes(),
    )
    .map_err(|error| format!("invalid helper frame-reader configuration: {error}"))?;

    let actual_profile = match helper_mode {
        HelperMode::Synthetic => {
            let actual_rate = CaptureFrameRate::new(
                ready.frame_rate_numerator,
                ready.frame_rate_seconds_denominator,
            )
            .map_err(|error| format!("invalid negotiated helper frame rate: {error}"))?;
            CaptureVideoProfile::new(
                ready.width_px,
                ready.height_px,
                actual_rate,
                CapturePixelFormat::new(ready.pixel_format.clone())
                    .map_err(|error| format!("invalid negotiated helper pixel format: {error}"))?,
            )
            .map_err(|error| format!("invalid negotiated helper profile: {error}"))?
        }
        #[cfg(feature = "camera-gstreamer-source")]
        HelperMode::Physical => {
            let input = ready
                .input_profile
                .as_ref()
                .ok_or_else(|| "physical helper omitted its input profile".to_owned())?;
            let actual_rate = CaptureFrameRate::new(
                input.frame_rate_numerator,
                input.frame_rate_seconds_denominator,
            )
            .map_err(|error| format!("invalid physical input frame rate: {error}"))?;
            CaptureVideoProfile::new(
                input.width_px,
                input.height_px,
                actual_rate,
                CapturePixelFormat::new(input.pixel_format.clone())
                    .map_err(|error| format!("invalid physical input pixel format: {error}"))?,
            )
            .map_err(|error| format!("invalid physical input profile: {error}"))?
        }
        #[cfg(feature = "camera-gstreamer-source")]
        HelperMode::HttpMjpeg => {
            let output_rate = resolved.expected.expected_negotiated_profile().frame_rate();
            CaptureVideoProfile::new(
                ready.width_px,
                ready.height_px,
                output_rate,
                CapturePixelFormat::new(ready.pixel_format.clone())
                    .map_err(|error| format!("invalid HTTP-MJPEG output format: {error}"))?,
            )
            .map_err(|error| format!("invalid HTTP-MJPEG output profile: {error}"))?
        }
    };
    let is_physical = helper_mode.is_physical();
    let control_capability = if is_physical {
        CaptureCapability::Unknown
    } else {
        CaptureCapability::Unsupported
    };
    let capabilities = CaptureSourceCapabilities::new(
        Some(vec![actual_profile.pixel_format().clone()]),
        control_capability,
        control_capability,
        control_capability,
        control_capability,
    )
    .map_err(|error| format!("invalid GStreamer source capabilities: {error}"))?;
    let controls = CaptureAppliedCameraControls::new(
        if is_physical {
            CaptureControlReadback::Unknown
        } else {
            CaptureControlReadback::Unsupported
        },
        if is_physical {
            CaptureControlReadback::Unknown
        } else {
            CaptureControlReadback::Unsupported
        },
        if is_physical {
            CaptureControlReadback::Unknown
        } else {
            CaptureControlReadback::Unsupported
        },
        CaptureControlReadback::Unknown,
        CaptureControlReadback::Unknown,
        CaptureControlReadback::Unknown,
        if is_physical {
            CaptureControlReadback::Unknown
        } else {
            CaptureControlReadback::Unsupported
        },
        if is_physical {
            CaptureControlReadback::Unknown
        } else {
            CaptureControlReadback::Unsupported
        },
    )
    .map_err(|error| format!("invalid GStreamer control readbacks: {error}"))?;
    let applied_geometry = CaptureAppliedFrameGeometry::new(
        resolved.configuration_revision,
        resolved.expected.camera_profile_id().to_owned(),
        resolved.expected.image_space(),
        resolved.expected.rectification_revision(),
    )
    .map_err(|error| format!("invalid synthetic frame-geometry identity: {error}"))?;
    let metadata = CaptureSourceMetadata::new(
        resolved.expected.device_id().to_owned(),
        Some(match helper_mode {
            HelperMode::Synthetic => "GStreamer synthetic videotestsrc".to_owned(),
            #[cfg(feature = "camera-gstreamer-source")]
            HelperMode::Physical => "GStreamer selected physical camera".to_owned(),
            #[cfg(feature = "camera-gstreamer-source")]
            HelperMode::HttpMjpeg => "Remote HTTP multipart MJPEG stream".to_owned(),
        }),
        match helper_mode {
            HelperMode::Synthetic => "gstreamer",
            #[cfg(feature = "camera-gstreamer-source")]
            HelperMode::Physical => "gstreamer",
            #[cfg(feature = "camera-gstreamer-source")]
            HelperMode::HttpMjpeg => "gstreamer",
        },
        resolved.expected.requested_profile().clone(),
        actual_profile,
        capabilities,
        applied_geometry,
        controls,
        CaptureTimestampSemantics::ReadCompletionOnly,
    )
    .map_err(|error| format!("invalid synthetic source metadata: {error}"))?;
    let metadata = if resolved.expected.output_frame_geometry().is_some() {
        metadata.with_output_frame_geometry(output_geometry)
    } else {
        metadata
    };

    Ok(GstreamerCaptureSource {
        reader: Some(frame_reader),
        validator,
        control: Arc::clone(control),
        metadata,
        capture_audit_timing: config.capture_audit_timing,
        child_exit_timeout: config.child_exit_timeout,
        child_exit_poll_interval: config.child_exit_poll_interval,
        ended: false,
    })
}

fn elapsed_nanos(started: Option<Instant>) -> Option<u64> {
    started.map(|started| u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX))
}

fn read_ready_with_deadline(
    mut stdout: std::process::ChildStdout,
    maximum_json_bytes: usize,
    timeout: Duration,
    control: &Arc<ChildControl>,
) -> Result<(CaptureHelperReady, std::process::ChildStdout), String> {
    let (sender, receiver) = mpsc::sync_channel(1);
    let reader_thread = thread::Builder::new()
        .name("capture-helper-ready".to_owned())
        .spawn(move || {
            let result = read_ready_record(&mut stdout, maximum_json_bytes)
                .map(|ready| (ready, stdout))
                .map_err(|error| error.to_string());
            let _ = sender.send(result);
        })
        .map_err(|error| format!("could not start helper-ready reader: {error}"))?;

    match receiver.recv_timeout(timeout) {
        Ok(Ok(ready_and_stdout)) => {
            reader_thread
                .join()
                .map_err(|_| "helper-ready reader thread panicked".to_owned())?;
            Ok(ready_and_stdout)
        }
        Ok(Err(error)) => {
            let _ = reader_thread.join();
            Err(format!("helper did not send a valid ready record: {error}"))
        }
        Err(mpsc::RecvTimeoutError::Timeout) => {
            let _ = control.kill_and_reap();
            let _ = reader_thread.join();
            Err("helper-ready record exceeded the configured deadline".to_owned())
        }
        Err(mpsc::RecvTimeoutError::Disconnected) => {
            let _ = control.kill_and_reap();
            let _ = reader_thread.join();
            Err("helper-ready reader exited without a result".to_owned())
        }
    }
}

fn validate_ready_binding(
    ready: &CaptureHelperReady,
    resolved: &ResolvedCaptureSettings,
    session_id: &str,
    helper_mode: HelperMode,
) -> Result<(), String> {
    let expected_geometry = resolved
        .expected
        .output_frame_geometry()
        .unwrap_or_else(|| {
            let profile = resolved.expected.expected_negotiated_profile();
            CaptureOutputFrameGeometry::new(profile.width_px(), profile.height_px())
                .expect("validated capture profile has non-zero geometry")
        });
    let expected_profile = resolved.expected.expected_negotiated_profile();
    let expected_rate = expected_profile.frame_rate();
    if !helper_mode.is_physical() {
        let comparisons = [
            (
                ready.record_schema_version == 1,
                "synthetic helper-ready schema version",
            ),
            (
                ready.width_px == expected_geometry.width_px()
                    && ready.height_px == expected_geometry.height_px(),
                "synthetic output frame geometry",
            ),
            (
                ready.frame_rate_numerator == expected_rate.frames_numerator()
                    && ready.frame_rate_seconds_denominator == expected_rate.seconds_denominator(),
                "synthetic rational frame rate",
            ),
            (
                ready.pixel_format == expected_profile.pixel_format().identifier()
                    && ready.pixel_format == "RGB8",
                "synthetic RGB8 output format",
            ),
        ];
        if let Some((_, field)) = comparisons.into_iter().find(|(matches, _)| !matches) {
            return Err(format!(
                "helper-ready {field} does not match the resolved request"
            ));
        }
    } else {
        #[cfg(feature = "camera-gstreamer-source")]
        {
            let Some(input) = ready.input_profile.as_ref() else {
                return Err("physical helper-ready record omitted input profile".to_owned());
            };
            let Some(provider_identifier) = ready.source_identifier.as_deref() else {
                return Err("physical helper-ready record omitted provider identity".to_owned());
            };
            let Some(provider_key) = ready.source_identifier_source.as_deref() else {
                return Err("physical helper-ready record omitted provider identity key".to_owned());
            };
            if !crate::camera_discovery::is_persistent_provider_identity(
                std::env::consts::OS,
                provider_key,
                provider_identifier,
            ) || crate::camera_discovery::capture_selector(
                std::env::consts::OS,
                provider_key,
                provider_identifier,
            ) != resolved.expected.device_id()
            {
                return Err(
                    "physical helper-ready provider identity does not resolve to the requested selector"
                        .to_owned(),
                );
            }
            let comparisons = [
                (
                    ready.record_schema_version == 2,
                    "physical helper-ready schema version",
                ),
                (
                    ready.source_device_id.as_deref() == Some(resolved.expected.device_id()),
                    "selected stable camera identity",
                ),
                (
                    (input.width_px, input.height_px)
                        == (expected_profile.width_px(), expected_profile.height_px()),
                    "negotiated camera input geometry",
                ),
                (
                    input.frame_rate_numerator == expected_rate.frames_numerator()
                        && input.frame_rate_seconds_denominator
                            == expected_rate.seconds_denominator(),
                    "negotiated camera input rate",
                ),
                (
                    input.pixel_format == expected_profile.pixel_format().identifier(),
                    "negotiated camera input pixel format",
                ),
                (
                    (ready.width_px, ready.height_px)
                        == (expected_geometry.width_px(), expected_geometry.height_px()),
                    "emitted RGB8 output geometry",
                ),
                (
                    ready.frame_rate_numerator == expected_rate.frames_numerator()
                        && ready.frame_rate_seconds_denominator
                            == expected_rate.seconds_denominator(),
                    "emitted RGB8 frame rate",
                ),
                (ready.pixel_format == "RGB8", "emitted RGB8 pixel format"),
            ];
            if let Some((_, field)) = comparisons.into_iter().find(|(matches, _)| !matches) {
                return Err(format!(
                    "helper-ready {field} does not match the resolved request"
                ));
            }
        }
        #[cfg(not(feature = "camera-gstreamer-source"))]
        return Err("physical GStreamer helper feature is disabled".to_owned());
    }
    let comparisons = [
        (ready.session_id == session_id, "session identity"),
        (
            ready.configuration_revision == resolved.configuration_revision,
            "configuration revision",
        ),
        (
            ready.camera_profile_id == resolved.expected.camera_profile_id(),
            "camera profile identity",
        ),
        (
            ready.capture_settings_sha256 == resolved.request_sha256,
            "capture-settings digest",
        ),
    ];
    if let Some((_, field)) = comparisons.into_iter().find(|(matches, _)| !matches) {
        return Err(format!(
            "helper-ready {field} does not match the resolved request"
        ));
    }
    Ok(())
}

#[cfg(all(test, feature = "camera-gstreamer-source"))]
mod physical_ready_tests {
    use super::{validate_ready_binding, HelperMode};
    use crate::capture_helper_protocol::{CaptureHelperInputProfile, CaptureHelperReady};
    use crate::capture_settings_wire::parse_capture_settings_json;
    use serde_json::Value;

    fn resolved_v3() -> crate::capture_settings_wire::ResolvedCaptureSettings {
        let (provider_key, provider_id) = match std::env::consts::OS {
            "macos" => ("avf.unique_id", "fixture-camera-42"),
            "linux" => ("device.path", "/dev/v4l/by-id/fixture-camera-42"),
            other => panic!("unsupported test platform {other}"),
        };
        let selector = crate::camera_discovery::capture_selector(
            std::env::consts::OS,
            provider_key,
            provider_id,
        );
        let mut request: Value = serde_json::from_str(include_str!(
            "../contracts/capture-settings/v3/capture-settings-default.json"
        ))
        .unwrap();
        request["requested"]["device_id"] = Value::String(selector);
        let bytes = serde_json::to_vec(&request).unwrap();
        parse_capture_settings_json(std::str::from_utf8(&bytes).unwrap()).unwrap()
    }

    fn ready(
        resolved: &crate::capture_settings_wire::ResolvedCaptureSettings,
    ) -> CaptureHelperReady {
        let profile = resolved.expected.expected_negotiated_profile();
        let geometry = resolved.expected.output_frame_geometry().unwrap();
        let (provider_key, provider_id) = match std::env::consts::OS {
            "macos" => ("avf.unique_id", "fixture-camera-42"),
            "linux" => ("device.path", "/dev/v4l/by-id/fixture-camera-42"),
            _ => unreachable!(),
        };
        CaptureHelperReady {
            record_schema_version: 2,
            session_id: "physical-session".to_owned(),
            configuration_revision: resolved.configuration_revision,
            camera_profile_id: resolved.expected.camera_profile_id().to_owned(),
            capture_settings_sha256: resolved.request_sha256.clone(),
            width_px: geometry.width_px(),
            height_px: geometry.height_px(),
            frame_rate_numerator: profile.frame_rate().frames_numerator(),
            frame_rate_seconds_denominator: profile.frame_rate().seconds_denominator(),
            pixel_format: "RGB8".to_owned(),
            source_device_id: Some(resolved.expected.device_id().to_owned()),
            source_identifier: Some(provider_id.to_owned()),
            source_identifier_source: Some(provider_key.to_owned()),
            input_profile: Some(CaptureHelperInputProfile {
                width_px: profile.width_px(),
                height_px: profile.height_px(),
                frame_rate_numerator: profile.frame_rate().frames_numerator(),
                frame_rate_seconds_denominator: profile.frame_rate().seconds_denominator(),
                pixel_format: profile.pixel_format().identifier().to_owned(),
            }),
        }
    }

    #[test]
    fn physical_ready_requires_exact_selector_and_camera_input_profile() {
        let resolved = resolved_v3();
        let ready = ready(&resolved);
        validate_ready_binding(&ready, &resolved, "physical-session", HelperMode::Physical)
            .unwrap();

        let mut mismatched_identity = ready.clone();
        mismatched_identity.source_device_id = Some("gstreamer-device-v1:other".to_owned());
        assert!(validate_ready_binding(
            &mismatched_identity,
            &resolved,
            "physical-session",
            HelperMode::Physical
        )
        .is_err());

        let mut mismatched_provider = ready.clone();
        mismatched_provider.source_identifier = Some("another-camera".to_owned());
        assert!(validate_ready_binding(
            &mismatched_provider,
            &resolved,
            "physical-session",
            HelperMode::Physical
        )
        .is_err());

        let mut mismatched_input = ready;
        mismatched_input.input_profile.as_mut().unwrap().width_px += 1;
        assert!(validate_ready_binding(
            &mismatched_input,
            &resolved,
            "physical-session",
            HelperMode::Physical
        )
        .is_err());
    }
}
