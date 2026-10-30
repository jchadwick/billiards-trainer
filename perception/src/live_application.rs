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
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::calibration_store::{CalibrationCaptureContextV1, CalibrationStore};
use crate::capture::{CaptureReceiveOutcome, CaptureRuntimeWait, CaptureSourceFactory};
use crate::capture_pipeline::{
    AppliedCaptureCalibration, CapturePipelineConfig, CapturePipelineOutcome,
};
use crate::capture_runtime::{LiveCaptureLifecycle, LiveCaptureRuntime};
use crate::capture_settings_wire::parse_capture_settings_json;
use crate::frame_preview::{FramePreviewConfig, FramePreviewServer, LatestCapturedFrameSlot};
use crate::inference::{FrameInferenceEngine, FrameInferenceOptions, ProviderSelection};
use crate::live_audit::{LiveAuditConfig, LiveAuditExporter, LiveAuditShutdownSummary};
use crate::phase_scheduler::{PhaseCadence, SchedulerConfig};
use crate::table_visibility::TableVisibilityConfig;
use crate::tracking::TrackingConfigV1;
use crate::{ImageSpace, SceneTransportConfig};
use anyhow::{anyhow, bail, ensure, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Upper bound for the compact schedule passed as one child-process argument.
/// The fixed schema is far smaller than this limit; the bound protects process
/// creation if the wire contract grows unexpectedly.
const MAX_PHASE_SCHEDULE_JSON_BYTES: usize = 4 * 1024;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct PhaseCadenceWire {
    motion_hz: f64,
    ball_hz: f64,
    cue_hz: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct PhaseScheduleWire {
    recovering: PhaseCadenceWire,
    pre_shot: PhaseCadenceWire,
    aiming: PhaseCadenceWire,
    shot: PhaseCadenceWire,
    quiet_dwell_ms: u64,
    layout_consensus_count: usize,
    layout_position_tolerance_px: f64,
    motion_evidence_expiry_ms: u64,
    cue_evidence_expiry_ms: u64,
    layout_evidence_expiry_ms: u64,
}

impl PhaseScheduleWire {
    fn validate(&self) -> Result<()> {
        for (phase_name, cadence) in [
            ("recovering", &self.recovering),
            ("pre_shot", &self.pre_shot),
            ("aiming", &self.aiming),
            ("shot", &self.shot),
        ] {
            for (rate_name, rate) in [
                ("motion_hz", cadence.motion_hz),
                ("ball_hz", cadence.ball_hz),
                ("cue_hz", cadence.cue_hz),
            ] {
                ensure!(
                    rate.is_finite() && rate >= 0.0,
                    "phase_schedule.{phase_name}.{rate_name} must be finite and non-negative"
                );
            }
        }
        ensure!(
            self.quiet_dwell_ms > 0,
            "phase_schedule.quiet_dwell_ms must be positive"
        );
        ensure!(
            self.layout_consensus_count > 0,
            "phase_schedule.layout_consensus_count must be positive"
        );
        ensure!(
            self.layout_position_tolerance_px.is_finite()
                && self.layout_position_tolerance_px > 0.0,
            "phase_schedule.layout_position_tolerance_px must be finite and positive"
        );
        ensure!(
            self.motion_evidence_expiry_ms > 0,
            "phase_schedule.motion_evidence_expiry_ms must be positive"
        );
        ensure!(
            self.cue_evidence_expiry_ms > 0,
            "phase_schedule.cue_evidence_expiry_ms must be positive"
        );
        ensure!(
            self.layout_evidence_expiry_ms > 0,
            "phase_schedule.layout_evidence_expiry_ms must be positive"
        );
        Ok(())
    }

    fn into_scheduler_config(self) -> Result<SchedulerConfig> {
        self.validate()?;
        Ok(SchedulerConfig {
            recovering: self.recovering.into(),
            pre_shot: self.pre_shot.into(),
            aiming: self.aiming.into(),
            shot: self.shot.into(),
            quiet_dwell: Duration::from_millis(self.quiet_dwell_ms),
            layout_consensus_count: self.layout_consensus_count,
            layout_position_tolerance_px: self.layout_position_tolerance_px,
            motion_evidence_expiry: Duration::from_millis(self.motion_evidence_expiry_ms),
            cue_evidence_expiry: Duration::from_millis(self.cue_evidence_expiry_ms),
            layout_evidence_expiry: Duration::from_millis(self.layout_evidence_expiry_ms),
        })
    }
}

impl From<PhaseCadenceWire> for PhaseCadence {
    fn from(value: PhaseCadenceWire) -> Self {
        Self {
            motion_hz: value.motion_hz,
            ball_hz: value.ball_hz,
            cue_hz: value.cue_hz,
        }
    }
}

/// Parse the supervisor's bounded JSON phase schedule into the coordinator's
/// typed scheduler configuration.
pub fn parse_phase_schedule_json(json: &str) -> Result<SchedulerConfig> {
    ensure!(
        json.len() <= MAX_PHASE_SCHEDULE_JSON_BYTES,
        "phase schedule JSON exceeds the {MAX_PHASE_SCHEDULE_JSON_BYTES}-byte limit"
    );
    let wire: PhaseScheduleWire = serde_json::from_str(json)
        .context("phase schedule must match the Rust live phase-schedule contract")?;
    wire.into_scheduler_config()
}

/// Validate and compact the effective configuration schedule for the single
/// child-process argument. The configuration authority remains the source of
/// defaults; missing or malformed fields are rejected here rather than guessed.
pub(crate) fn serialize_phase_schedule_config(value: &Value) -> Result<String, String> {
    let wire: PhaseScheduleWire = serde_json::from_value(value.clone())
        .map_err(|error| format!("invalid perception.rust_live.phase_schedule: {error}"))?;
    wire.validate()
        .map_err(|error| format!("invalid perception.rust_live.phase_schedule: {error:#}"))?;
    let json = serde_json::to_string(&wire)
        .map_err(|error| format!("could not encode live phase schedule: {error}"))?;
    if json.len() > MAX_PHASE_SCHEDULE_JSON_BYTES {
        return Err(format!(
            "perception.rust_live.phase_schedule exceeds the {MAX_PHASE_SCHEDULE_JSON_BYTES}-byte child argument limit"
        ));
    }
    Ok(json)
}

/// All runtime-affecting live settings. The Rust gateway resolves these
/// values before process creation; this type deliberately supplies no camera,
/// detector, transport, or shutdown defaults.
#[derive(Debug, Clone)]
pub struct LiveApplicationSettings {
    pub onnx_path: Option<PathBuf>,
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
    /// Validated per-phase motion, full-layout, and cue processing cadence.
    pub phase_schedule: SchedulerConfig,
    /// Validated appearance policy bound to the operator-confirmed calibration
    /// reference. A missing reference always keeps guidance disabled.
    pub table_visibility: TableVisibilityConfig,
    pub scene_transport: SceneTransportConfig,
    pub startup_timeout: Duration,
    pub poll_interval: Duration,
    pub graceful_stop_timeout: Duration,
    pub capture_settings_maximum_bytes: usize,
    /// Optional loopback-only current-frame preview. `None` creates no slot,
    /// listener, or preview workers.
    pub frame_preview: Option<FramePreviewConfig>,
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
/// destructor cannot block past the gateway supervisor's process-reap deadline.
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
        settings.provider == ProviderSelection::Openvino,
        "live perception requires the OpenVINO GPU provider"
    );
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

    let mut resolved = parse_capture_settings_json(exact_capture_settings_json)
        .context("invalid capture settings request")?;
    let visibility_reference = if let Some(store) = settings
        .frame_preview
        .as_ref()
        .and_then(|config| config.calibration_control.as_ref())
        .map(|config| &config.store)
    {
        install_calibration_store_state(&mut resolved, store)
    } else {
        None
    };
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

    let inference_options = FrameInferenceOptions {
        onnx_path: settings.onnx_path.clone(),
        source_checkpoint_path: settings.source_checkpoint_path.clone(),
        provider: settings.provider,
        image_size: settings.image_size,
        confidence: settings.confidence,
        iou: settings.iou,
        max_detections: settings.max_detections,
        intra_op_threads: settings.intra_op_threads,
    };
    let mut detector = FrameInferenceEngine::new_without_onnx_runtime(inference_options)
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

    // Preview is opt-in. Its slot and server do not participate in capture
    // readiness, and no client request is needed for camera startup. The slot
    // is bound to the accepted source report by the capture-owner thread.
    let engine_instance_id = settings.engine_instance_id.clone();
    let capture_session_id = settings.source_session_id.clone();
    let calibration_store_enabled = settings
        .frame_preview
        .as_ref()
        .is_some_and(|config| config.calibration_control.is_some());
    let (frame_preview_slot, _frame_preview_server) = match settings.frame_preview {
        Some(config) => {
            let slot = Arc::new(
                LatestCapturedFrameSlot::new(engine_instance_id.clone())
                    .context("invalid frame-preview engine identity")?,
            );
            let server = FramePreviewServer::bind(config, Arc::clone(&slot))
                .context("failed to bind the loopback frame-preview listener")?;
            let readiness =
                server.readiness_record(engine_instance_id.clone(), capture_session_id.clone());
            eprintln!(
                "frame_preview_ready={}",
                serde_json::to_string(&readiness)
                    .context("failed to serialize frame-preview readiness")?
            );
            (Some(slot), Some(server))
        }
        None => (None, None),
    };

    let pipeline = CapturePipelineConfig {
        engine_instance_id: settings.engine_instance_id,
        capture_configuration_revision: resolved.configuration_revision,
        image_space: batch_image_space(expected.image_space()),
        maximum_frame_age: expected.maximum_frame_age(),
        phase_schedule: Some(settings.phase_schedule.clone()),
        calibration,
        scene_transport: settings.scene_transport,
    };

    let audit_recorder = audit_exporter.as_ref().map(LiveAuditExporter::recorder);
    let mut runtime = match (frame_preview_slot, calibration_store_enabled) {
        (Some(preview_slot), true) => {
            LiveCaptureRuntime::start_with_session_id_audit_frame_preview_and_calibration(
                exact_capture_settings_json,
                settings.source_session_id,
                detector,
                tracking,
                pipeline,
                audit_recorder,
                preview_slot,
                expected.calibration_revision(),
                expected.calibration().cloned(),
                settings.table_visibility.clone(),
                visibility_reference,
                source_factory,
            )
        }
        (Some(preview_slot), false) => {
            LiveCaptureRuntime::start_with_session_id_audit_frame_preview_and_visibility(
                exact_capture_settings_json,
                settings.source_session_id,
                detector,
                tracking,
                pipeline,
                audit_recorder,
                preview_slot,
                settings.table_visibility.clone(),
                visibility_reference,
                source_factory,
            )
        }
        (None, _) => LiveCaptureRuntime::start_with_session_id_audit_and_visibility(
            exact_capture_settings_json,
            settings.source_session_id,
            detector,
            tracking,
            pipeline,
            audit_recorder,
            settings.table_visibility.clone(),
            visibility_reference,
            source_factory,
        ),
    }
    .context("failed to start Rust live capture runtime")?;

    let session_result = run_ready_session(
        &mut runtime,
        settings.startup_timeout,
        settings.poll_interval,
        shutdown_requested,
    );
    if let Some(diagnostics) = runtime.phase_diagnostics() {
        eprintln!(
            "phase_runtime_diagnostics={}",
            serde_json::to_string(&diagnostics).unwrap_or_else(|_| "{}".to_owned())
        );
    }
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

/// Load the Rust-owned store before model/capture startup. An unavailable,
/// malformed, or incompatible calibration disables physical coordinates but
/// leaves image inference and ball detection available. Capture request bytes
/// and their digest remain unchanged for the source readiness acknowledgement.
fn install_calibration_store_state(
    resolved: &mut crate::capture_settings_wire::ResolvedCaptureSettings,
    store: &CalibrationStore,
) -> Option<crate::table_visibility::TableVisibilityReference> {
    let loaded = match store.load() {
        Ok(snapshot) => snapshot,
        Err(error) => {
            eprintln!("calibration_store_load_failed={error}");
            let _ = resolved.expected.apply_calibration_store_state(0, None);
            return None;
        }
    };
    let store_revision = loaded.revision.unwrap_or(0);
    let Some(calibration) = loaded.calibration else {
        // Keep a clear tombstone's revision visible, even though the active
        // pipeline has no calibration bundle.
        let _ = resolved
            .expected
            .apply_calibration_store_state(store_revision, None);
        return None;
    };
    let Some(binding) = loaded.binding_context.as_ref() else {
        eprintln!("calibration_store_bundle_inactive=missing_capture_binding");
        let _ = resolved
            .expected
            .apply_calibration_store_state(store_revision, None);
        return None;
    };
    if !binding_matches_request(binding, &resolved.expected) {
        eprintln!("calibration_store_bundle_inactive=capture_context_mismatch");
        let _ = resolved
            .expected
            .apply_calibration_store_state(store_revision, None);
        return None;
    }
    let validated = match calibration.into_validated() {
        Ok(calibration) => calibration,
        Err(error) => {
            eprintln!("calibration_store_bundle_inactive=invalid_bundle:{error}");
            let _ = resolved
                .expected
                .apply_calibration_store_state(store_revision, None);
            return None;
        }
    };
    if let Err(error) = resolved
        .expected
        .apply_calibration_store_state(store_revision, Some(validated))
    {
        eprintln!("calibration_store_bundle_inactive={error}");
        let _ = resolved
            .expected
            .apply_calibration_store_state(store_revision, None);
        return None;
    }
    loaded.visibility_reference
}

fn binding_matches_request(
    binding: &CalibrationCaptureContextV1,
    expected: &crate::live_config::ExpectedCaptureSettings,
) -> bool {
    let (width_px, height_px) = expected
        .output_frame_geometry()
        .map(|geometry| (geometry.width_px(), geometry.height_px()))
        .unwrap_or_else(|| {
            (
                expected.expected_negotiated_profile().width_px(),
                expected.expected_negotiated_profile().height_px(),
            )
        });
    binding.configuration_revision == expected.configuration_revision()
        && binding.camera_profile_id == expected.camera_profile_id()
        && binding.image_space == expected.image_space()
        && binding.source_width_px == width_px
        && binding.source_height_px == height_px
        && binding.rectification_revision == expected.rectification_revision()
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

#[cfg(test)]
mod calibration_store_tests {
    use std::time::Duration;

    use crate::calibration::{ImagePointPx, TableCalibrationSolverInput};
    use crate::calibration_store::{CalibrationCaptureContextV1, CalibrationStore};
    use crate::capture_settings_wire::parse_capture_settings_json;
    use crate::ImageSpace;

    use super::{binding_matches_request, install_calibration_store_state};

    const CAPTURE_SETTINGS: &str =
        include_str!("../contracts/capture-settings/v3/capture-settings-default.json");

    #[test]
    fn saved_binding_activates_on_a_matching_later_process_and_tombstone_stays_visible() {
        let directory = tempfile::tempdir().unwrap();
        let store = CalibrationStore::new(
            directory.path().join("table-calibration.json"),
            Duration::from_millis(100),
            64 * 1024,
        )
        .unwrap();
        let mut resolved = parse_capture_settings_json(CAPTURE_SETTINGS).unwrap();
        let request_digest = resolved.request_sha256.clone();
        let expected = &resolved.expected;
        let geometry = expected.output_frame_geometry().unwrap();
        let context = CalibrationCaptureContextV1 {
            engine_instance_id: "previous-engine".to_owned(),
            capture_session_id: "previous-session".to_owned(),
            source_frame_id: "previous-frame".to_owned(),
            configuration_revision: expected.configuration_revision(),
            camera_profile_id: expected.camera_profile_id().to_owned(),
            image_space: expected.image_space(),
            source_width_px: geometry.width_px(),
            source_height_px: geometry.height_px(),
            calibration_revision: expected.calibration_revision(),
            rectification_revision: expected.rectification_revision(),
        };
        let solver_input = TableCalibrationSolverInput {
            calibration_revision: 0,
            camera_profile_id: context.camera_profile_id.clone(),
            rectification_revision: context.rectification_revision,
            image_space: context.image_space,
            source_width_px: context.source_width_px,
            source_height_px: context.source_height_px,
            table_width_mm: 2000.0,
            table_height_mm: 1000.0,
            corners_px: [
                ImagePointPx {
                    x_px: 100.0,
                    y_px: 100.0,
                },
                ImagePointPx {
                    x_px: f64::from(context.source_width_px - 100),
                    y_px: 100.0,
                },
                ImagePointPx {
                    x_px: f64::from(context.source_width_px - 100),
                    y_px: f64::from(context.source_height_px - 100),
                },
                ImagePointPx {
                    x_px: 100.0,
                    y_px: f64::from(context.source_height_px - 100),
                },
            ],
        };
        let saved = store
            .save_with_context(None, solver_input, Some(context.clone()))
            .unwrap();
        // Process/session/frame IDs describe the source image; the next
        // process reuses calibration only when stable capture geometry agrees.
        assert!(binding_matches_request(&context, &resolved.expected));
        install_calibration_store_state(&mut resolved, &store);
        assert_eq!(resolved.request_sha256, request_digest);
        assert_eq!(resolved.expected.calibration_revision(), 1);
        assert_eq!(
            resolved
                .expected
                .calibration()
                .unwrap()
                .calibration_revision(),
            1
        );

        let mut incompatible = context.clone();
        incompatible.configuration_revision += 1;
        assert!(!binding_matches_request(&incompatible, &resolved.expected));

        store.clear(saved.revision).unwrap();
        install_calibration_store_state(&mut resolved, &store);
        assert_eq!(resolved.expected.calibration_revision(), 2);
        assert!(resolved.expected.calibration().is_none());
        assert!(matches!(resolved.expected.image_space(), ImageSpace::Raw));
    }
}

fn run_ready_session<D: crate::capture_pipeline::Rgb8FrameDetector + Send + 'static>(
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

fn stop_runtime<D: crate::capture_pipeline::Rgb8FrameDetector + Send + 'static>(
    mut runtime: LiveCaptureRuntime<D>,
    timeout: Duration,
) -> Result<crate::SceneTransportStats> {
    let wait = runtime.stop(timeout);
    if wait == CaptureRuntimeWait::TimedOut {
        let owner_status = runtime.owner_status();
        // CaptureRuntimeCoordinator joins its owner thread when dropped. Once
        // the configured shutdown budget expires, preserve the handle until
        // this process exits so that a late destructor cannot exceed the
        // Gateway supervisor's independent process-reap deadline.
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
        public_state_delivery_enabled: true,
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
