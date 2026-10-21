//! Process-level composition for the Rust-owned live perception path.
//!
//! The library accepts a source factory so deterministic integration tests
//! exercise the same model, tracker, readiness handshake, and transport used
//! by a platform camera adapter. Adapter selection stays outside this module;
//! the standalone binary fails clearly while no qualified adapter is built.

use std::io::{ErrorKind, Read};
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use crate::capture::{CaptureReceiveOutcome, CaptureRuntimeWait, CaptureSourceFactory};
use crate::capture_pipeline::{
    AppliedCaptureCalibration, CapturePipelineConfig, CapturePipelineOutcome,
};
use crate::capture_runtime::{LiveCaptureLifecycle, LiveCaptureRuntime};
use crate::capture_settings_wire::parse_capture_settings_json;
use crate::inference::{FrameInferenceEngine, FrameInferenceOptions, ProviderSelection};
use crate::live_audit::{LiveAuditConfig, LiveAuditExporter, LiveAuditShutdownSummary};
use crate::tracking::TrackingConfigV1;
use crate::{ImageSpace, SceneTransportConfig};
use anyhow::{anyhow, bail, ensure, Context, Result};

/// All runtime-affecting live settings. The Python supervisor resolves these
/// values before process creation; this type deliberately supplies no camera,
/// detector, transport, or shutdown defaults.
#[derive(Debug, Clone)]
pub struct LiveApplicationSettings {
    pub onnx_path: PathBuf,
    pub source_checkpoint_path: PathBuf,
    pub provider: ProviderSelection,
    pub image_size: u32,
    pub confidence: f32,
    pub iou: f32,
    pub max_detections: usize,
    pub intra_op_threads: Option<usize>,
    pub warmup_runs: usize,
    pub source_image_space: ImageSpace,
    pub engine_instance_id: String,
    pub source_session_id: String,
    pub tracking_ball_class_id: u32,
    pub tracking_cue_class_id: Option<u32>,
    pub tracking_camera_profile_id: String,
    pub tracking_association_gate_px: f64,
    pub tracking_association_prediction_horizon: Duration,
    pub tracking_lost_after: Duration,
    pub tracking_maximum_track_age: Duration,
    pub tracking_maximum_active_tracks: usize,
    pub tracking_recent_observation_capacity: usize,
    pub tracking_configuration_revision: u64,
    pub tracking_rectification_revision: u64,
    pub scene_transport: SceneTransportConfig,
    pub startup_timeout: Duration,
    pub poll_interval: Duration,
    pub graceful_stop_timeout: Duration,
    pub capture_settings_maximum_bytes: usize,
}

/// Read a one-shot capture request without allocating beyond its configured
/// byte limit. The exact UTF-8 bytes are returned unchanged because their
/// digest is part of the request-bound protocol-v3 acknowledgement.
pub fn read_capture_settings<R: Read>(reader: &mut R, maximum_bytes: usize) -> Result<String> {
    ensure!(
        maximum_bytes > 0,
        "capture settings byte limit must be positive"
    );

    let mut bytes = Vec::new();
    let mut chunk = [0_u8; 8 * 1024];
    loop {
        let remaining = maximum_bytes.saturating_sub(bytes.len());
        if remaining == 0 {
            let mut overflow_probe = [0_u8; 1];
            match read_retry_interrupted(reader, &mut overflow_probe)
                .context("failed while reading the one-shot capture settings request")?
            {
                0 => break,
                _ => bail!(
                    "capture settings request exceeds configured {}-byte limit",
                    maximum_bytes
                ),
            }
        }

        let chunk_length = remaining.min(chunk.len());
        let count = read_retry_interrupted(reader, &mut chunk[..chunk_length])
            .context("failed while reading the one-shot capture settings request")?;
        if count == 0 {
            break;
        }
        bytes.extend_from_slice(&chunk[..count]);
    }

    String::from_utf8(bytes).context("capture settings request is not valid UTF-8")
}

fn read_retry_interrupted<R: Read>(reader: &mut R, buffer: &mut [u8]) -> std::io::Result<usize> {
    loop {
        match reader.read(buffer) {
            Err(error) if error.kind() == ErrorKind::Interrupted => continue,
            result => return result,
        }
    }
}

/// Run one live application session using the supplied Rust source adapter.
///
/// The runner requests source release on every path after capture starts and
/// waits up to the configured deadline. If a native open/read operation does
/// not return by then, the runtime handle is intentionally retained so its
/// destructor cannot block past the Python supervisor's process-reap deadline.
/// The caller must then let this dedicated process exit; it must not continue
/// using this library process or attempt to start another camera owner.
pub fn run_live_application<F: CaptureSourceFactory>(
    exact_capture_settings_json: &str,
    settings: LiveApplicationSettings,
    source_factory: F,
    shutdown_requested: &AtomicBool,
) -> Result<()> {
    run_live_application_with_audit(
        exact_capture_settings_json,
        settings,
        source_factory,
        shutdown_requested,
        None,
    )
}

/// Run the existing live application with optional, separately bounded timing
/// export. Failure to create or use the audit exporter is reported on stderr
/// and does not fail or delay the scene-production path.
pub fn run_live_application_with_audit<F: CaptureSourceFactory>(
    exact_capture_settings_json: &str,
    settings: LiveApplicationSettings,
    source_factory: F,
    shutdown_requested: &AtomicBool,
    audit_config: Option<LiveAuditConfig>,
) -> Result<()> {
    ensure!(
        exact_capture_settings_json.len() <= settings.capture_settings_maximum_bytes,
        "capture settings request exceeds configured byte limit"
    );
    ensure!(
        settings.capture_settings_maximum_bytes > 0,
        "capture settings byte limit must be positive"
    );
    ensure!(
        !settings.startup_timeout.is_zero(),
        "live startup timeout must be positive"
    );
    ensure!(
        !settings.poll_interval.is_zero(),
        "live process poll interval must be positive"
    );
    ensure!(
        !settings.graceful_stop_timeout.is_zero(),
        "live graceful-stop timeout must be positive"
    );

    let mut audit_exporter = match audit_config {
        Some(config) => {
            config.validate()?;
            match LiveAuditExporter::start(config) {
                Ok(exporter) => Some(exporter),
                Err(error) => {
                    eprintln!("live audit exporter could not start: {error:#}");
                    None
                }
            }
        }
        None => None,
    };

    let resolved = parse_capture_settings_json(exact_capture_settings_json)
        .context("invalid capture settings request")?;
    let expected = &resolved.expected;
    ensure!(
        expected.image_space() == settings.source_image_space,
        "source image space differs from the exact capture request"
    );
    ensure!(
        expected.camera_profile_id() == settings.tracking_camera_profile_id,
        "tracking camera profile differs from the exact capture request"
    );
    ensure!(
        expected.configuration_revision() == settings.tracking_configuration_revision,
        "tracking configuration revision differs from the exact capture request"
    );
    ensure!(
        expected.rectification_revision() == settings.tracking_rectification_revision,
        "tracking rectification revision differs from the exact capture request"
    );

    let environment = ort::init()
        .with_name("billiards-perception-live")
        .with_telemetry(false)
        .build()
        .context("failed to initialize ONNX Runtime")?;
    let mut detector = FrameInferenceEngine::new(
        environment,
        FrameInferenceOptions {
            onnx_path: settings.onnx_path.clone(),
            source_checkpoint_path: settings.source_checkpoint_path.clone(),
            provider: settings.provider,
            image_size: settings.image_size,
            confidence: settings.confidence,
            iou: settings.iou,
            max_detections: settings.max_detections,
            intra_op_threads: settings.intra_op_threads,
        },
    )
    .context("failed to load the configured live detector")?;

    warm_up_detector(&mut detector, expected, settings.warmup_runs)
        .context("live detector warm-up failed")?;

    let calibration = expected
        .calibration()
        .cloned()
        .map(|bundle| {
            AppliedCaptureCalibration::new(
                bundle,
                expected.calibration_revision(),
                expected.camera_profile_id(),
                expected.rectification_revision(),
            )
        })
        .transpose()
        .context("capture request calibration could not be applied")?;

    let tracking = TrackingConfigV1::new(
        settings.tracking_ball_class_id,
        settings.tracking_cue_class_id,
        settings.tracking_camera_profile_id,
        settings.tracking_association_gate_px,
        settings.tracking_association_prediction_horizon,
        settings.tracking_lost_after,
        settings.tracking_maximum_track_age,
        settings.tracking_maximum_active_tracks,
        settings.tracking_recent_observation_capacity,
        settings.tracking_configuration_revision,
        settings.tracking_rectification_revision,
    )
    .context("invalid live tracking configuration")?;

    let pipeline = CapturePipelineConfig {
        engine_instance_id: settings.engine_instance_id,
        capture_configuration_revision: resolved.configuration_revision,
        image_space: batch_image_space(expected.image_space()),
        maximum_frame_age: expected.maximum_frame_age(),
        calibration,
        scene_transport: settings.scene_transport,
    };

    let audit_recorder = audit_exporter.as_ref().map(LiveAuditExporter::recorder);
    let mut runtime = LiveCaptureRuntime::start_with_session_id_and_audit(
        exact_capture_settings_json,
        settings.source_session_id,
        detector,
        tracking,
        pipeline,
        audit_recorder,
        source_factory,
    )
    .context("failed to start Rust live capture runtime")?;

    let session_result = run_ready_session(
        &mut runtime,
        settings.startup_timeout,
        settings.poll_interval,
        shutdown_requested,
    );
    let pipeline_counters = runtime.pipeline_stats();
    let stop_result = stop_runtime(runtime, settings.graceful_stop_timeout);
    let shutdown_summary = LiveAuditShutdownSummary {
        // A released source is not a clean stream if processing failed before
        // shutdown. Both the session and its runtime teardown must complete.
        clean_shutdown: session_result.is_ok() && stop_result.is_ok(),
        pipeline_counters: Some(pipeline_counters),
        scene_transport_counters: stop_result.as_ref().ok().cloned(),
    };

    if let Some(exporter) = audit_exporter.take() {
        let summary = exporter.finish_with_shutdown_summary(shutdown_summary);
        if summary.failure.is_some() || summary.dropped_records > 0 {
            eprintln!(
                "live_audit_export_summary={}",
                serde_json::to_string(&summary).unwrap_or_else(|_| "{}".to_owned())
            );
        }
    }

    match (session_result, stop_result) {
        (Ok(()), Ok(_stats)) => Ok(()),
        (Err(session_error), Ok(_stats)) => Err(session_error),
        (Ok(()), Err(stop_error)) => Err(stop_error),
        (Err(session_error), Err(stop_error)) => Err(anyhow!(
            "{session_error:#}; additionally, live source shutdown failed: {stop_error:#}"
        )),
    }
}

fn batch_image_space(image_space: ImageSpace) -> crate::detection_batch::ImageSpace {
    match image_space {
        ImageSpace::Raw => crate::detection_batch::ImageSpace::Raw,
        ImageSpace::Rectified => crate::detection_batch::ImageSpace::Rectified,
    }
}

fn warm_up_detector(
    detector: &mut FrameInferenceEngine,
    expected: &crate::live_config::ExpectedCaptureSettings,
    warmup_runs: usize,
) -> Result<()> {
    if warmup_runs == 0 {
        return Ok(());
    }
    let (width_px, height_px) = warmup_rgb8_dimensions(expected);
    let pixel_count = u64::from(width_px)
        .checked_mul(u64::from(height_px))
        .and_then(|pixels| pixels.checked_mul(3))
        .context("warm-up RGB frame dimensions overflow")?;
    let byte_count = usize::try_from(pixel_count)
        .context("warm-up RGB frame cannot fit in this process address space")?;
    ensure!(
        byte_count <= expected.frame_limit().maximum_rgb8_bytes(),
        "warm-up RGB frame exceeds the request's per-frame memory limit"
    );
    let pixels = vec![0_u8; byte_count];
    for _ in 0..warmup_runs {
        detector
            .infer_rgb8(width_px, height_px, &pixels)
            .context("live detector failed on a warm-up frame")?;
    }
    Ok(())
}

fn warmup_rgb8_dimensions(expected: &crate::live_config::ExpectedCaptureSettings) -> (u32, u32) {
    expected
        .output_frame_geometry()
        .map(|geometry| (geometry.width_px(), geometry.height_px()))
        .unwrap_or_else(|| {
            let profile = expected.expected_negotiated_profile();
            (profile.width_px(), profile.height_px())
        })
}

fn run_ready_session<D: crate::capture_pipeline::Rgb8FrameDetector>(
    runtime: &mut LiveCaptureRuntime<D>,
    startup_timeout: Duration,
    poll_interval: Duration,
    shutdown_requested: &AtomicBool,
) -> Result<()> {
    let startup_deadline = Instant::now()
        .checked_add(startup_timeout)
        .context("live startup deadline is outside the monotonic clock range")?;

    loop {
        if shutdown_requested.load(Ordering::SeqCst) {
            bail!("shutdown requested before live capture became ready");
        }
        let remaining = startup_deadline.saturating_duration_since(Instant::now());
        ensure!(!remaining.is_zero(), "live capture readiness timed out");
        let lifecycle = runtime
            .wait_until_ready(remaining.min(poll_interval))
            .clone();
        match lifecycle {
            LiveCaptureLifecycle::Pending { .. } => continue,
            LiveCaptureLifecycle::Ready { .. } => break,
            LiveCaptureLifecycle::Failed { reason, .. } => {
                bail!("live capture failed before readiness: {reason}");
            }
            LiveCaptureLifecycle::Stopping { .. } | LiveCaptureLifecycle::Stopped { .. } => {
                bail!("live capture stopped before readiness");
            }
        }
    }

    loop {
        if shutdown_requested.load(Ordering::SeqCst) {
            return Ok(());
        }
        match runtime
            .receive_one(poll_interval)
            .context("live capture frame processing failed")?
        {
            CaptureReceiveOutcome::Frame(Ok(CapturePipelineOutcome::Published(_)))
            | CaptureReceiveOutcome::Frame(Ok(CapturePipelineOutcome::Skipped(_))) => {}
            CaptureReceiveOutcome::Frame(Ok(CapturePipelineOutcome::Rejected(rejection))) => {
                bail!("live capture frame was rejected: {rejection:?}");
            }
            CaptureReceiveOutcome::Frame(Err(error)) => {
                return Err(error).context("live capture frame processing failed");
            }
            CaptureReceiveOutcome::TimedOut => {}
            CaptureReceiveOutcome::Closed => return Ok(()),
            CaptureReceiveOutcome::Cancelled if shutdown_requested.load(Ordering::SeqCst) => {
                return Ok(());
            }
            CaptureReceiveOutcome::Cancelled => bail!("live capture was cancelled"),
            CaptureReceiveOutcome::ProducerFailed(reason) => {
                bail!("live capture source failed: {reason}");
            }
        }
    }
}

fn stop_runtime<D: crate::capture_pipeline::Rgb8FrameDetector>(
    mut runtime: LiveCaptureRuntime<D>,
    timeout: Duration,
) -> Result<crate::SceneTransportStats> {
    let wait = runtime.stop(timeout);
    if wait == CaptureRuntimeWait::TimedOut {
        let owner_status = runtime.owner_status();
        // CaptureRuntimeCoordinator joins its owner thread when dropped. Once
        // the configured shutdown budget expires, preserve the handle until
        // this process exits so that a late destructor cannot exceed the
        // Python supervisor's independent process-reap deadline.
        std::mem::forget(runtime);
        bail!("source release exceeded its configured deadline ({owner_status:?})");
    }

    match runtime.lifecycle() {
        LiveCaptureLifecycle::Stopped { .. } => Ok(runtime.transport_stats()),
        LiveCaptureLifecycle::Failed { reason, .. } => {
            bail!("live capture source release was not confirmed: {reason}");
        }
        lifecycle => bail!("live capture did not stop cleanly: {lifecycle:?}"),
    }
}

/// Build the transport configuration from values parsed by the live CLI.
/// Kept public so another platform-specific binary can reuse this contract.
pub fn scene_transport_config(
    address: SocketAddr,
    maximum_message_bytes: usize,
    connect_timeout: Duration,
    write_timeout: Duration,
    reconnect_initial_delay: Duration,
    reconnect_maximum_delay: Duration,
) -> SceneTransportConfig {
    SceneTransportConfig {
        address,
        maximum_message_bytes,
        connect_timeout,
        write_timeout,
        reconnect_initial_delay,
        reconnect_maximum_delay,
    }
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::{read_capture_settings, warmup_rgb8_dimensions};
    use crate::capture_settings_wire::parse_capture_settings_json;

    #[test]
    fn capture_settings_reader_preserves_exact_bytes_and_enforces_the_configured_bound() {
        let original = b"{ \"schema_version\": 1 }\n";
        assert_eq!(
            read_capture_settings(&mut Cursor::new(original), original.len()).unwrap(),
            std::str::from_utf8(original).unwrap()
        );
        assert!(
            read_capture_settings(&mut Cursor::new(original), original.len() - 1)
                .unwrap_err()
                .to_string()
                .contains("exceeds configured")
        );
        assert!(read_capture_settings(&mut Cursor::new([0xff]), 1)
            .unwrap_err()
            .to_string()
            .contains("valid UTF-8"));
        assert!(read_capture_settings(&mut Cursor::new(b"{}"), 0).is_err());
    }

    #[test]
    fn detector_warmup_uses_emitted_geometry_for_v2_and_negotiated_geometry_for_v1() {
        let v1 = parse_capture_settings_json(include_str!(
            "../contracts/capture-settings/v1/capture-settings-default.json"
        ))
        .unwrap();
        assert_eq!(warmup_rgb8_dimensions(&v1.expected), (1920, 1080));

        let v2 = parse_capture_settings_json(include_str!(
            "../contracts/capture-settings/v2/capture-settings-default.json"
        ))
        .unwrap();
        assert_eq!(
            v2.expected
                .output_frame_geometry()
                .map(|geometry| (geometry.width_px(), geometry.height_px())),
            Some((1920, 1080))
        );
        assert_eq!(warmup_rgb8_dimensions(&v2.expected), (1920, 1080));

        let transformed_v2 = parse_capture_settings_json(include_str!(
            "../contracts/capture-settings/v2/capture-settings-transformed-output.json"
        ))
        .unwrap();
        assert_eq!(
            (
                transformed_v2.expected.requested_profile().width_px(),
                transformed_v2.expected.requested_profile().height_px(),
            ),
            (1920, 1080)
        );
        assert_eq!(
            warmup_rgb8_dimensions(&transformed_v2.expected),
            (1280, 720)
        );
    }
}
