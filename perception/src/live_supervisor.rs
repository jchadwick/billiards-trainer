//! Rust-owned lifecycle for the live detector child.
//!
//! The gateway starts the detector only when explicitly configured for live
//! operation. The child receives one exact, authority-generated capture
//! request, publishes through the gateway's loopback ingress, and is restarted
//! with bounded backoff after an unexpected exit. This keeps camera/model
//! failures visible as not-ready state while the service remains a single
//! Rust-owned runtime.

use std::net::{IpAddr, SocketAddr};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;

use serde_json::Value;
use tokio::io::AsyncWriteExt;
use tokio::process::{Child, Command};
use tokio::sync::watch;
use tokio::time::{sleep, timeout};
use uuid::Uuid;

use crate::ball_localization_profile::BallLocalizationSettings;
use crate::capture_settings_wire::build_capture_settings_v3;
use crate::live_application::serialize_phase_schedule_config;
use crate::live_audit::LiveAuditConfig;
use crate::producer_diagnostics::{DiagnosticsRegistry, ReporterLaunch, REPORTER_ENV};
use crate::producer_diagnostics_contract::{DiagnosticsConfig, ProducerExitReason};

const PHYSICAL_SELECTOR_PREFIX: &str = "gstreamer-device-v1:";

/// Executable paths selected by the same rules used by the live supervisor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct LiveExecutablePaths {
    pub detector: PathBuf,
    pub capture_helper: PathBuf,
}

/// Resolve the live executables without launching them.
///
/// `sibling_directory` is supplied by managed audit tools that need to inspect
/// the gateway binary's colocated executables before the gateway is launched.
/// The live service passes `None`, which selects the directory of its own
/// executable. Explicit command-line overrides and configured paths retain the
/// exact path rules used by the live process supervisor.
pub(crate) fn resolve_live_executables(
    configuration: &Value,
    project_root: &Path,
    live_executable_override: Option<&Path>,
    capture_helper_override: Option<&Path>,
    sibling_directory: Option<&Path>,
) -> Result<LiveExecutablePaths, String> {
    if !project_root.is_absolute() {
        return Err("live process project root must be absolute".to_owned());
    }
    let project_root = project_root
        .canonicalize()
        .map_err(|error| format!("could not resolve project root: {error}"))?;
    let live = object_at(configuration, "perception.rust_live")?;
    let sibling_directory = match sibling_directory {
        Some(directory) => directory.to_path_buf(),
        None => std::env::current_exe()
            .map_err(|error| format!("could not locate the Rust gateway executable: {error}"))?
            .parent()
            .map(Path::to_path_buf)
            .ok_or_else(|| "Rust gateway executable has no parent directory".to_owned())?,
    };

    let detector = match live_executable_override {
        Some(path) => resolve_configured_path(path, &project_root)?,
        None => configured_or_sibling(
            live,
            "executable_path",
            &project_root,
            &sibling_directory,
            "billiards-perception-live",
        )?,
    };
    let capture_helper = match capture_helper_override {
        Some(path) => resolve_configured_path(path, &project_root)?,
        None => configured_or_sibling(
            live,
            "capture_helper_executable_path",
            &project_root,
            &sibling_directory,
            "billiards-perception-gstreamer-probe",
        )?,
    };
    Ok(LiveExecutablePaths {
        detector,
        capture_helper,
    })
}

/// Validated process arguments derived from Rust's effective configuration.
#[derive(Debug, Clone)]
pub struct LiveProcessConfig {
    executable: PathBuf,
    project_root: PathBuf,
    capture_request: Vec<u8>,
    arguments: Vec<String>,
    graceful_stop_timeout: Duration,
    restart_initial_delay: Duration,
    restart_maximum_delay: Duration,
    diagnostics: Option<(ReporterLaunch, DiagnosticsRegistry)>,
}

impl LiveProcessConfig {
    /// Resolve model, capture, and process settings from one effective config.
    ///
    /// Executable overrides are intended for deployment packaging and tests.
    /// When omitted, sibling Rust binaries are used so a colocated server
    /// distribution does not depend on a separate launcher or PATH lookup.
    pub fn from_effective_configuration(
        configuration: &Value,
        project_root: &Path,
        scene_address: SocketAddr,
        live_executable_override: Option<&Path>,
        capture_helper_override: Option<&Path>,
        audit_config: Option<LiveAuditConfig>,
    ) -> Result<Self, String> {
        if !project_root.is_absolute() {
            return Err("live process project root must be absolute".to_owned());
        }
        let project_root = project_root
            .canonicalize()
            .map_err(|error| format!("could not resolve project root: {error}"))?;
        if let Some(config) = audit_config {
            config
                .validate()
                .map_err(|error| format!("invalid live audit configuration: {error}"))?;
        }
        let live = object_at(configuration, "perception.rust_live")?;
        let capture_settings = build_capture_settings_v3(configuration)
            .map_err(|error| format!("could not resolve live capture settings: {error}"))?;
        let capture_request = serde_json::to_vec(&capture_settings)
            .map_err(|error| format!("could not serialize live capture request: {error}"))?;
        let request_text = std::str::from_utf8(&capture_request)
            .map_err(|error| format!("capture request is not UTF-8: {error}"))?;
        let resolved = crate::capture_settings_wire::parse_capture_settings_json(request_text)
            .map_err(|error| format!("capture request failed Rust validation: {error}"))?;
        if resolved.schema_version != 3 {
            return Err("live detector requires capture settings v3".to_owned());
        }

        let executable_paths = resolve_live_executables(
            configuration,
            &project_root,
            live_executable_override,
            capture_helper_override,
            None,
        )?;
        let executable = executable_paths.detector;
        let helper = executable_paths.capture_helper;
        let phase_schedule = effective_phase_schedule(live)?;
        let phase_schedule_json = serialize_phase_schedule_config(&phase_schedule)?;
        let table_visibility = live
            .get("table_visibility")
            .ok_or_else(|| "perception.rust_live.table_visibility is missing".to_owned())?;
        let table_visibility_config: crate::table_visibility::TableVisibilityConfig =
            serde_json::from_value(table_visibility.clone()).map_err(|error| {
                format!("invalid perception.rust_live.table_visibility: {error}")
            })?;
        table_visibility_config
            .validate()
            .map_err(|error| format!("invalid perception.rust_live.table_visibility: {error}"))?;
        let table_visibility_json =
            serde_json::to_string(&table_visibility_config).map_err(|error| {
                format!("could not serialize table visibility configuration: {error}")
            })?;
        let mut ball_localization = match live.get("ball_localization") {
            Some(value) => serde_json::from_value::<BallLocalizationSettings>(value.clone())
                .map_err(|error| {
                    format!("invalid perception.rust_live.ball_localization: {error}")
                })?,
            None => BallLocalizationSettings::default(),
        };
        ball_localization.validate().map_err(|error| {
            format!("invalid perception.rust_live.ball_localization: {error:#}")
        })?;
        let requested = capture_settings
            .get("requested")
            .and_then(Value::as_object)
            .ok_or_else(|| "capture request has no requested settings".to_owned())?;
        let image_space = required_string(requested, "image_space")?;
        let camera_profile = required_string(requested, "camera_profile_id")?;
        crate::browser_precision_calibration::select_assets(
            &mut ball_localization,
            &project_root,
            &camera_profile,
        );
        let ball_localization_json = serde_json::to_string(&ball_localization)
            .map_err(|error| format!("could not serialize ball localization settings: {error}"))?;
        let configuration_revision = capture_settings
            .get("configuration_revision")
            .and_then(Value::as_u64)
            .ok_or_else(|| "capture request has no configuration revision".to_owned())?;
        let rectification_revision = requested
            .get("rectification_revision")
            .and_then(Value::as_u64)
            .ok_or_else(|| "capture request has no rectification revision".to_owned())?;
        let capture_source = capture_source(configuration)?;
        let mut arguments = Vec::new();
        let provider = required_string(live, "provider")?;
        if provider != "openvino" {
            return Err(format!(
                "production live inference requires the OpenVINO GPU provider; unsupported perception.rust_live.provider: {provider}"
            ));
        }
        let canonical_onnx = "models/billiards-yolov8n.onnx";
        let source_export = "models/exports/billiards-yolov8n.onnx";
        let default_onnx = if project_root.join(canonical_onnx).is_file()
            || !project_root.join(source_export).is_file()
        {
            canonical_onnx
        } else {
            source_export
        };
        for (key, argument, fallback) in [
            ("onnx_path", "onnx", default_onnx),
            (
                "source_checkpoint_path",
                "source-checkpoint",
                "models/billiards-yolov8n.pt",
            ),
        ] {
            let raw = live
                .get(key)
                .and_then(Value::as_str)
                .filter(|value| !value.trim().is_empty())
                .unwrap_or(fallback);
            let path = resolve_configured_path(Path::new(raw), &project_root)?;
            if raw == fallback && !path.is_file() {
                return Err(format!("Model asset is missing: {}. Install the verified model bundle or configure a model path under Advanced.", path.display()));
            }
            push_argument(&mut arguments, argument, path.display().to_string());
        }
        for (config_key, argument) in [
            ("provider", "provider"),
            ("image_size", "image-size"),
            ("confidence", "confidence"),
            ("max_detections", "max-detections"),
            ("warmup_runs", "warmup-runs"),
            ("ball_class_id", "tracking-ball-class-id"),
            ("association_gate_px", "tracking-association-gate-px"),
            (
                "association_prediction_horizon_ms",
                "tracking-association-prediction-horizon-ms",
            ),
            ("lost_after_ms", "tracking-lost-after-ms"),
            ("maximum_track_age_ms", "tracking-maximum-track-age-ms"),
            ("maximum_active_tracks", "tracking-maximum-active-tracks"),
            (
                "recent_observation_capacity",
                "tracking-recent-observation-capacity",
            ),
            ("maximum_message_bytes", "scene-snapshots-max-message-bytes"),
            ("connect_timeout_ms", "scene-snapshots-connect-timeout-ms"),
            ("write_timeout_ms", "scene-snapshots-write-timeout-ms"),
            (
                "reconnect_initial_delay_ms",
                "scene-snapshots-reconnect-initial-delay-ms",
            ),
            (
                "reconnect_maximum_delay_ms",
                "scene-snapshots-reconnect-maximum-delay-ms",
            ),
            ("poll_interval_ms", "poll-interval-ms"),
        ] {
            push_value_argument(&mut arguments, live, config_key, argument)?;
        }
        push_value_argument(&mut arguments, live, "iou", "iou")?;
        if let Some(threads) = live
            .get("intra_op_threads")
            .filter(|value| !value.is_null())
        {
            push_value_argument_named(&mut arguments, "intra-op-threads", threads)?;
        }
        if let Some(cue_class) = live.get("cue_class_id").filter(|value| !value.is_null()) {
            push_value_argument_named(&mut arguments, "tracking-cue-class-id", cue_class)?;
        }
        push_argument(&mut arguments, "source-image-space", image_space);
        push_argument(&mut arguments, "phase-schedule-json", phase_schedule_json);
        push_argument(
            &mut arguments,
            "table-visibility-config-json",
            table_visibility_json,
        );
        push_argument(
            &mut arguments,
            "ball-localization-json",
            ball_localization_json,
        );
        push_argument(&mut arguments, "tracking-camera-profile-id", camera_profile);
        push_argument(
            &mut arguments,
            "tracking-configuration-revision",
            configuration_revision.to_string(),
        );
        push_argument(
            &mut arguments,
            "tracking-rectification-revision",
            rectification_revision.to_string(),
        );
        push_argument(&mut arguments, "capture-settings-stdin", "");
        push_argument(
            &mut arguments,
            "capture-settings-max-bytes",
            required_u64(live, "capture_settings_maximum_bytes")?.to_string(),
        );
        push_argument(
            &mut arguments,
            "scene-snapshots-tcp",
            scene_address.to_string(),
        );
        append_live_audit_arguments(&mut arguments, audit_config);
        push_argument(
            &mut arguments,
            "startup-timeout-ms",
            seconds_as_milliseconds(required_f64(live, "startup_timeout_seconds")?)?.to_string(),
        );
        push_argument(
            &mut arguments,
            "graceful-stop-timeout-ms",
            seconds_as_milliseconds(required_f64(live, "graceful_stop_timeout_seconds")?)?
                .to_string(),
        );

        if let Some(url) = capture_source.http_mjpeg_url.as_deref() {
            push_argument(&mut arguments, "capture-source", "gstreamer-http-mjpeg");
            push_argument(&mut arguments, "capture-http-mjpeg-url", url);
        } else {
            push_argument(&mut arguments, "capture-source", "gstreamer-physical");
        }
        push_argument(
            &mut arguments,
            "capture-helper-executable",
            helper.display().to_string(),
        );
        for (key, argument) in [
            (
                "capture_helper_ready_record_max_bytes",
                "capture-helper-ready-record-max-bytes",
            ),
            (
                "capture_helper_frame_header_max_bytes",
                "capture-helper-frame-header-max-bytes",
            ),
            (
                "capture_helper_slow_reader_delay_ms",
                "capture-helper-slow-reader-delay-ms",
            ),
            (
                "capture_helper_pull_timeout_ms",
                "capture-helper-pull-timeout-ms",
            ),
            (
                "capture_helper_readiness_timeout_ms",
                "capture-helper-readiness-timeout-ms",
            ),
            (
                "capture_helper_child_exit_timeout_ms",
                "capture-helper-child-exit-timeout-ms",
            ),
            (
                "capture_helper_child_exit_poll_interval_ms",
                "capture-helper-child-exit-poll-interval-ms",
            ),
        ] {
            push_value_argument(&mut arguments, live, key, argument)?;
        }

        append_frame_preview_arguments(&mut arguments, live, &project_root)?;

        let graceful_stop_timeout = Duration::from_millis(seconds_as_milliseconds(required_f64(
            live,
            "graceful_stop_timeout_seconds",
        )?)?);
        // The existing capture recovery policy supplies a conservative restart
        // interval and a bounded cap, avoiding a tight model reload loop when a
        // camera is disconnected. The detector itself retains its independent
        // transport reconnect policy.
        let restart_initial_delay = Duration::from_millis(seconds_as_milliseconds(required_f64(
            live,
            "source_silence_timeout_seconds",
        )?)?);
        let restart_maximum_delay = Duration::from_millis(seconds_as_milliseconds(required_f64(
            live,
            "startup_timeout_seconds",
        )?)?);
        if restart_initial_delay.is_zero() || restart_maximum_delay < restart_initial_delay {
            return Err("live restart backoff configuration is invalid".to_owned());
        }

        Ok(Self {
            executable,
            project_root,
            capture_request,
            arguments,
            graceful_stop_timeout,
            restart_initial_delay,
            restart_maximum_delay,
            diagnostics: None,
        })
    }

    /// Bind metadata reporting to the already allocated private listener. Reuse
    /// the scene writer's established timeout/backoff configuration.
    pub fn with_diagnostics(
        mut self,
        address: SocketAddr,
        policy: DiagnosticsConfig,
        registry: DiagnosticsRegistry,
        configuration: &Value,
    ) -> Result<Self, String> {
        let live = object_at(configuration, "perception.rust_live")?;
        let launch = ReporterLaunch {
            address,
            generation_id: Uuid::new_v4().to_string(),
            policy,
            connect_timeout_ms: required_u64(live, "connect_timeout_ms")?,
            write_timeout_ms: required_u64(live, "write_timeout_ms")?,
            reconnect_initial_delay_ms: required_u64(live, "reconnect_initial_delay_ms")?,
            reconnect_maximum_delay_ms: required_u64(live, "reconnect_maximum_delay_ms")?,
        };
        launch.validate()?;
        self.diagnostics = Some((launch, registry));
        Ok(self)
    }

    /// Keep the child alive, restarting unexpected exits with bounded backoff.
    pub async fn supervise(self, shutdown: watch::Receiver<bool>) {
        self.supervise_with_failure(shutdown, |_| {}).await;
    }

    pub(crate) async fn supervise_with_failure(
        self,
        mut shutdown: watch::Receiver<bool>,
        report_failure: impl Fn(String),
    ) {
        let mut restart_delay = self.restart_initial_delay;
        loop {
            if *shutdown.borrow() {
                return;
            }
            match self.spawn_child().await {
                Ok(mut child) => {
                    // Scope the active model to this exact supervised process lifetime.
                    let mut generation = self.diagnostics.as_ref().and_then(|(_, registry)| {
                        registry.snapshot().generation_id.map(|generation| {
                            ActiveDiagnosticsGeneration {
                                registry: registry.clone(),
                                generation,
                                exit_reason: None,
                            }
                        })
                    });
                    eprintln!("live_detector_started=true");
                    tokio::select! {
                        result = child.wait() => {
                            match result {
                                Ok(status) => {
                                    eprintln!("live_detector_exited={status}");
                                    report_failure(format!("Live producer exited ({status}). Check camera access, model assets, and the OpenVINO GPU runtime."));
                                    if let Some(generation) = generation.as_mut() {
                                        generation.set_exit_reason(if status.success() {
                                            ProducerExitReason::ProcessExitedSuccessfully
                                        } else {
                                            ProducerExitReason::ProcessExitedNonzero
                                        });
                                    }
                                }
                                Err(error) => {
                                    eprintln!("live_detector_wait_failed={error}");
                                    report_failure(format!("Could not wait for the live producer: {error}"));
                                    if let Some(generation) = generation.as_mut() {
                                        generation.set_exit_reason(ProducerExitReason::ProcessWaitFailed);
                                    }
                                }
                            }
                        }
                        changed = shutdown.changed() => {
                            if changed.is_err() || *shutdown.borrow() {
                                let outcome = stop_child(&mut child, self.graceful_stop_timeout).await;
                                if let Some(generation) = generation.as_mut() {
                                    generation.set_exit_reason(outcome);
                                }
                                return;
                            }
                        }
                    }
                }
                Err(error) => {
                    eprintln!("live_detector_start_failed={error}");
                    report_failure(error);
                }
            }
            tokio::select! {
                _ = sleep(restart_delay) => {},
                changed = shutdown.changed() => {
                    if changed.is_err() || *shutdown.borrow() {
                        return;
                    }
                }
            }
            restart_delay = restart_delay
                .saturating_mul(2)
                .min(self.restart_maximum_delay);
        }
    }

    async fn spawn_child(&self) -> Result<Child, String> {
        // These IDs scope transport watermarks and cached state to one producer
        // process and capture session. Generate them at the spawn boundary so
        // every supervised restart creates a new epoch, while reconnects made
        // by this child continue to use the same pair.
        let mut arguments = self.arguments.clone();
        push_argument(
            &mut arguments,
            "engine-instance-id",
            Uuid::new_v4().to_string(),
        );
        push_argument(
            &mut arguments,
            "source-session-id",
            Uuid::new_v4().to_string(),
        );
        let mut command = Command::new(&self.executable);
        let generation = if let Some((template, registry)) = &self.diagnostics {
            let mut launch = template.clone();
            launch.generation_id = Uuid::new_v4().to_string();
            let settings = serde_json::to_string(&launch).map_err(|error| error.to_string())?;
            registry.begin_generation(launch.generation_id.clone());
            command.env(REPORTER_ENV, settings);
            Some(launch.generation_id)
        } else {
            // Inherited launch routing must never register an unsupervised child.
            command.env_remove(REPORTER_ENV);
            None
        };
        let result = command
            .args(&arguments)
            .current_dir(&self.project_root)
            .stdin(Stdio::piped())
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit())
            .kill_on_drop(true)
            .spawn();
        let mut child = result.map_err(|error| {
            if let (Some(generation), Some((_, registry))) = (&generation, &self.diagnostics) {
                registry.retire_generation(generation, ProducerExitReason::SpawnFailed);
            }
            format!("could not spawn {}: {error}", self.executable.display())
        })?;
        let Some(mut stdin) = child.stdin.take() else {
            let _ = child.kill().await;
            if let (Some(generation), Some((_, registry))) = (&generation, &self.diagnostics) {
                registry.retire_generation(generation, ProducerExitReason::LaunchRequestFailed);
            }
            return Err("live detector stdin was not piped".to_owned());
        };
        if let Err(error) = stdin.write_all(&self.capture_request).await {
            let _ = child.kill().await;
            if let (Some(generation), Some((_, registry))) = (&generation, &self.diagnostics) {
                registry.retire_generation(generation, ProducerExitReason::LaunchRequestFailed);
            }
            return Err(format!("could not send exact capture request: {error}"));
        }
        drop(stdin);
        Ok(child)
    }
}

struct ActiveDiagnosticsGeneration {
    registry: DiagnosticsRegistry,
    generation: String,
    exit_reason: Option<ProducerExitReason>,
}
impl ActiveDiagnosticsGeneration {
    fn set_exit_reason(&mut self, exit_reason: ProducerExitReason) {
        self.exit_reason = Some(exit_reason);
    }
}
impl Drop for ActiveDiagnosticsGeneration {
    fn drop(&mut self) {
        self.registry.retire_generation(
            &self.generation,
            self.exit_reason.unwrap_or(ProducerExitReason::Unknown),
        );
    }
}

/// Forward the complete audit configuration as detector arguments. With no
/// configuration, the child receives no audit flags and creates no recorder.
fn append_live_audit_arguments(arguments: &mut Vec<String>, config: Option<LiveAuditConfig>) {
    let Some(config) = config else {
        return;
    };
    push_argument(arguments, "live-audit-tcp", config.address.to_string());
    push_argument(
        arguments,
        "live-audit-queue-capacity",
        config.queue_capacity.to_string(),
    );
    push_argument(
        arguments,
        "live-audit-max-record-bytes",
        config.maximum_record_bytes.to_string(),
    );
    push_argument(
        arguments,
        "live-audit-connect-timeout-ms",
        config.connect_timeout.as_millis().to_string(),
    );
    push_argument(
        arguments,
        "live-audit-write-timeout-ms",
        config.write_timeout.as_millis().to_string(),
    );
}

/// Translate gateway frame-preview settings into the detector's actual CLI
/// contract. Configuration stores the listener address and port separately,
/// while the detector accepts one SocketAddr in `--frame-preview-tcp`.
fn append_frame_preview_arguments(
    arguments: &mut Vec<String>,
    live: &serde_json::Map<String, Value>,
    project_root: &Path,
) -> Result<(), String> {
    if !required_bool(live, "frame_preview_enabled")? {
        return Ok(());
    }

    let bind_address = required_string(live, "frame_preview_bind_address")?
        .parse::<IpAddr>()
        .map_err(|error| format!("frame-preview bind address is invalid: {error}"))?;
    if !bind_address.is_loopback() {
        return Err("frame-preview bind address must be loopback".to_owned());
    }
    let port = u16::try_from(required_u64(live, "frame_preview_port")?)
        .map_err(|error| format!("frame-preview port is invalid: {error}"))?;
    push_argument(
        arguments,
        "frame-preview-tcp",
        SocketAddr::new(bind_address, port).to_string(),
    );

    for (key, argument) in [
        (
            "frame_preview_max_header_bytes",
            "frame-preview-max-header-bytes",
        ),
        (
            "frame_preview_max_jpeg_bytes",
            "frame-preview-max-jpeg-bytes",
        ),
        (
            "frame_preview_max_concurrent_encodes",
            "frame-preview-max-concurrent-encodes",
        ),
        ("frame_preview_jpeg_quality", "frame-preview-jpeg-quality"),
        (
            "frame_preview_request_timeout_ms",
            "frame-preview-request-timeout-ms",
        ),
        (
            "frame_preview_write_timeout_ms",
            "frame-preview-write-timeout-ms",
        ),
    ] {
        push_value_argument(arguments, live, key, argument)?;
    }

    if required_bool(live, "frame_preview_calibration_control_enabled")? {
        push_config_argument(
            arguments,
            live,
            "frame_preview_calibration_store_path",
            "frame-preview-calibration-store-path",
            project_root,
            true,
        )?;
        for (key, argument) in [
            (
                "frame_preview_calibration_store_lock_timeout_ms",
                "frame-preview-calibration-store-lock-timeout-ms",
            ),
            (
                "frame_preview_calibration_store_max_file_bytes",
                "frame-preview-calibration-store-max-file-bytes",
            ),
            (
                "frame_preview_calibration_max_request_bytes",
                "frame-preview-calibration-max-request-bytes",
            ),
        ] {
            push_value_argument(arguments, live, key, argument)?;
        }
    }
    Ok(())
}

struct CaptureSource {
    http_mjpeg_url: Option<String>,
}

fn capture_source(configuration: &Value) -> Result<CaptureSource, String> {
    let selector = value_at(configuration, "perception.capture.device_selector")?;
    if selector
        .as_str()
        .is_some_and(|value| value.starts_with(PHYSICAL_SELECTOR_PREFIX))
    {
        return Ok(CaptureSource {
            http_mjpeg_url: None,
        });
    }
    let device = value_at(configuration, "vision.camera.device_id")?;
    let Some(url) = device.as_str() else {
        return Err("live capture requires a stable camera selector or HTTP-MJPEG URL".into());
    };
    if crate::configuration_authority::valid_http_uri(url) {
        return Ok(CaptureSource {
            http_mjpeg_url: Some(url.to_owned()),
        });
    }
    Err("live physical camera capture requires a stable GStreamer selector".into())
}

fn object_at<'a>(
    root: &'a Value,
    path: &str,
) -> Result<&'a serde_json::Map<String, Value>, String> {
    value_at(root, path)?
        .as_object()
        .ok_or_else(|| format!("configuration field {path} must be an object"))
}

fn value_at<'a>(root: &'a Value, path: &str) -> Result<&'a Value, String> {
    path.split('.').try_fold(root, |value, segment| {
        value
            .get(segment)
            .ok_or_else(|| format!("configuration field {path} is missing"))
    })
}

fn required_string(values: &serde_json::Map<String, Value>, key: &str) -> Result<String, String> {
    values
        .get(key)
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
        .ok_or_else(|| format!("perception.rust_live.{key} must be a non-empty string"))
}

/// Validate and return the single shared phase schedule used by production.
fn effective_phase_schedule(live: &serde_json::Map<String, Value>) -> Result<Value, String> {
    live.get("phase_schedule")
        .cloned()
        .ok_or_else(|| "perception.rust_live.phase_schedule is missing".to_owned())
}

fn required_u64(values: &serde_json::Map<String, Value>, key: &str) -> Result<u64, String> {
    values
        .get(key)
        .and_then(Value::as_u64)
        .ok_or_else(|| format!("perception.rust_live.{key} must be an unsigned integer"))
}

fn required_f64(values: &serde_json::Map<String, Value>, key: &str) -> Result<f64, String> {
    values
        .get(key)
        .and_then(Value::as_f64)
        .filter(|value| value.is_finite())
        .ok_or_else(|| format!("perception.rust_live.{key} must be a finite number"))
}

fn required_bool(values: &serde_json::Map<String, Value>, key: &str) -> Result<bool, String> {
    values
        .get(key)
        .and_then(Value::as_bool)
        .ok_or_else(|| format!("perception.rust_live.{key} must be a boolean"))
}

fn configured_or_sibling(
    values: &serde_json::Map<String, Value>,
    key: &str,
    project_root: &Path,
    sibling_directory: &Path,
    sibling_name: &str,
) -> Result<PathBuf, String> {
    let configured = values.get(key).and_then(Value::as_str).unwrap_or_default();
    if configured.trim().is_empty() {
        return resolve_configured_path(&sibling_directory.join(sibling_name), project_root);
    }
    resolve_configured_path(Path::new(configured), project_root)
}

fn resolve_configured_path(path: &Path, project_root: &Path) -> Result<PathBuf, String> {
    let candidate = if path.is_absolute() {
        path.to_path_buf()
    } else {
        project_root.join(path)
    };
    Ok(candidate)
}

fn push_config_argument(
    arguments: &mut Vec<String>,
    live: &serde_json::Map<String, Value>,
    key: &str,
    argument: &str,
    project_root: &Path,
    required: bool,
) -> Result<(), String> {
    let raw = live.get(key).and_then(Value::as_str).unwrap_or_default();
    if raw.trim().is_empty() {
        if required {
            return Err(format!("perception.rust_live.{key} must be configured"));
        }
        return Ok(());
    }
    let path = resolve_configured_path(Path::new(raw), project_root)?;
    push_argument(arguments, argument, path.display().to_string());
    Ok(())
}

fn push_value_argument(
    arguments: &mut Vec<String>,
    live: &serde_json::Map<String, Value>,
    key: &str,
    argument: &str,
) -> Result<(), String> {
    let value = live
        .get(key)
        .ok_or_else(|| format!("perception.rust_live.{key} is missing"))?;
    push_value_argument_named(arguments, argument, value)
}

fn push_value_argument_named(
    arguments: &mut Vec<String>,
    argument: &str,
    value: &Value,
) -> Result<(), String> {
    let encoded = match value {
        Value::String(value) => value.clone(),
        Value::Number(value) => value.to_string(),
        _ => return Err(format!("live argument {argument} has an invalid value")),
    };
    push_argument(arguments, argument, encoded);
    Ok(())
}

fn push_argument(
    arguments: &mut Vec<String>,
    argument: impl Into<String>,
    value: impl Into<String>,
) {
    let value = value.into();
    let argument = argument.into();
    arguments.push(format!("--{argument}"));
    if !value.is_empty() {
        arguments.push(value);
    }
}

fn seconds_as_milliseconds(seconds: f64) -> Result<u64, String> {
    if seconds <= 0.0 || seconds > (u64::MAX as f64 / 1_000.0) {
        return Err("live timeout seconds are outside the supported range".to_owned());
    }
    Ok((seconds * 1_000.0).round() as u64)
}

async fn stop_child(child: &mut Child, graceful_timeout: Duration) -> ProducerExitReason {
    if let Ok(Some(status)) = child.try_wait() {
        return if status.success() {
            ProducerExitReason::ProcessExitedSuccessfully
        } else {
            ProducerExitReason::ProcessExitedNonzero
        };
    }
    #[cfg(unix)]
    if let Some(pid) = child.id() {
        // SAFETY: pid is the live child ID returned by the OS; SIGTERM is sent
        // to that process only. The child handles it and joins its capture task.
        let _ = unsafe { libc::kill(pid as i32, libc::SIGTERM) };
    }
    #[cfg(not(unix))]
    let _ = child.start_kill();

    match timeout(graceful_timeout, child.wait()).await {
        Ok(Ok(status)) if status.success() => ProducerExitReason::GracefulStop,
        Ok(Ok(_)) => ProducerExitReason::ProcessExitedNonzero,
        Ok(Err(_)) => ProducerExitReason::ProcessWaitFailed,
        Err(_) => {
            let _ = child.kill().await;
            ProducerExitReason::ForcedTermination
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{resolve_live_executables, stop_child, LiveProcessConfig};
    use crate::live_audit::LiveAuditConfig;
    use std::net::{Ipv4Addr, SocketAddr};
    use std::path::Path;
    use std::time::Duration;

    use crate::producer_diagnostics::{DiagnosticsRegistry, ReporterLaunch};
    use crate::producer_diagnostics_contract::{DiagnosticsConfig, ProducerExitReason};
    use serde_json::Value;
    use uuid::Uuid;

    fn remote_stream_config() -> Value {
        let mut configuration: Value =
            serde_json::from_str(include_str!("../contracts/configuration/v1/defaults.json"))
                .unwrap();
        configuration["vision"]["camera"]["device_id"] =
            Value::String("http://127.0.0.1:8001/stream.mjpg".to_owned());
        configuration["vision"]["camera"]["backend"] = Value::String("gstreamer".to_owned());
        configuration["vision"]["camera"]["video_file_path"] = Value::Null;
        configuration["perception"]["rust_live"]["onnx_path"] =
            Value::String("models/balls.onnx".to_owned());
        configuration["perception"]["rust_live"]["source_checkpoint_path"] =
            Value::String("models/balls.pt".to_owned());
        configuration["perception"]["rust_live"]["provider"] = Value::String("openvino".to_owned());
        configuration["perception"]["rust_live"]["capture_helper_executable_path"] = Value::String(
            "perception/target/release/billiards-perception-gstreamer-probe".to_owned(),
        );
        configuration["perception"]["rust_live"]["executable_path"] =
            Value::String("perception/target/release/billiards-perception-live".to_owned());
        configuration["perception"]["rust_live"]["frame_preview_enabled"] = Value::Bool(false);
        configuration
    }

    #[tokio::test]
    async fn supervisor_spawn_failure_is_recorded_and_clears_active_report() {
        let directory = tempfile::tempdir().unwrap();
        let registry = DiagnosticsRegistry::default();
        let launch = ReporterLaunch {
            address: SocketAddr::new(Ipv4Addr::LOCALHOST.into(), 45_678),
            generation_id: "template-generation".into(),
            policy: DiagnosticsConfig::default(),
            connect_timeout_ms: 1_000,
            write_timeout_ms: 1_000,
            reconnect_initial_delay_ms: 10,
            reconnect_maximum_delay_ms: 100,
        };
        let config = LiveProcessConfig {
            executable: directory.path().join("missing-detector"),
            project_root: directory.path().to_path_buf(),
            capture_request: br#"{"schema_version":3}"#.to_vec(),
            arguments: Vec::new(),
            graceful_stop_timeout: Duration::from_millis(100),
            restart_initial_delay: Duration::from_millis(10),
            restart_maximum_delay: Duration::from_millis(100),
            diagnostics: Some((launch, registry.clone())),
        };

        assert!(config.spawn_child().await.is_err());
        let snapshot = registry.snapshot();
        assert!(snapshot.exited);
        assert!(snapshot.report.is_none());
        assert_eq!(snapshot.exit_reason, Some(ProducerExitReason::SpawnFailed));
        assert_ne!(
            snapshot.generation_id.as_deref(),
            Some("template-generation")
        );
    }

    #[test]
    fn live_executable_resolution_shares_configured_override_and_sibling_rules() {
        let project = tempfile::tempdir().unwrap();
        let sibling_directory = project.path().join("distribution/bin");
        std::fs::create_dir_all(&sibling_directory).unwrap();
        let mut configuration = remote_stream_config();
        configuration["perception"]["rust_live"]["executable_path"] = Value::String(String::new());
        configuration["perception"]["rust_live"]["capture_helper_executable_path"] =
            Value::String(String::new());

        let siblings = resolve_live_executables(
            &configuration,
            project.path(),
            None,
            None,
            Some(&sibling_directory),
        )
        .unwrap();
        assert_eq!(
            siblings.detector,
            sibling_directory.join("billiards-perception-live")
        );
        assert_eq!(
            siblings.capture_helper,
            sibling_directory.join("billiards-perception-gstreamer-probe")
        );

        let overrides = resolve_live_executables(
            &configuration,
            project.path(),
            Some(Path::new("custom/live-detector")),
            Some(Path::new("custom/capture-helper")),
            Some(&sibling_directory),
        )
        .unwrap();
        let canonical_project = project.path().canonicalize().unwrap();
        assert_eq!(
            overrides.detector,
            canonical_project.join("custom/live-detector")
        );
        assert_eq!(
            overrides.capture_helper,
            canonical_project.join("custom/capture-helper")
        );
    }

    #[test]
    fn builds_live_arguments_from_authority_settings_and_remote_mjpeg_url() {
        let project = tempfile::tempdir().unwrap();
        let configuration = remote_stream_config();
        assert_eq!(configuration["vision"]["camera"]["fps"], 30);
        let config = LiveProcessConfig::from_effective_configuration(
            &configuration,
            project.path(),
            "127.0.0.1:49321".parse::<SocketAddr>().unwrap(),
            None,
            None,
            None,
        )
        .unwrap();

        assert_eq!(
            config.executable,
            project
                .path()
                .canonicalize()
                .unwrap()
                .join("perception/target/release/billiards-perception-live")
        );
        assert!(config
            .arguments
            .windows(2)
            .any(|pair| pair == ["--capture-source", "gstreamer-http-mjpeg"]));
        assert!(config.arguments.windows(2).any(|pair| {
            pair == [
                "--capture-http-mjpeg-url",
                "http://127.0.0.1:8001/stream.mjpg",
            ]
        }));
        assert!(config
            .arguments
            .windows(2)
            .any(|pair| { pair == ["--scene-snapshots-tcp", "127.0.0.1:49321"] }));
        assert!(config
            .arguments
            .contains(&"--capture-settings-stdin".to_owned()));
        assert!(config.arguments.contains(&"--onnx".to_owned()));
        assert!(config.arguments.contains(&"--source-checkpoint".to_owned()));
        let phase_schedule_arguments = config
            .arguments
            .iter()
            .enumerate()
            .filter(|(_, argument)| argument.as_str() == "--phase-schedule-json")
            .collect::<Vec<_>>();
        assert_eq!(phase_schedule_arguments.len(), 1);
        let schedule_json = &config.arguments[phase_schedule_arguments[0].0 + 1];
        assert!(schedule_json.len() <= 4 * 1024);
        let schedule: Value = serde_json::from_str(schedule_json).unwrap();
        assert_eq!(schedule["recovering"]["motion_hz"].as_f64(), Some(15.0));
        assert_eq!(schedule["recovering"]["ball_hz"].as_f64(), Some(5.0));
        assert_eq!(schedule["pre_shot"]["ball_hz"].as_f64(), Some(1.0));
        assert_eq!(schedule["shot"]["ball_hz"].as_f64(), Some(0.0));
        assert_eq!(schedule["motion_evidence_expiry_ms"], 200);
        assert_eq!(schedule["layout_evidence_expiry_ms"], 1500);
        assert_eq!(schedule["cue_evidence_expiry_ms"], 250);
        assert!(!config
            .arguments
            .iter()
            .any(|argument| argument.starts_with("--live-audit-")));
    }

    #[test]
    fn rejects_invalid_or_missing_phase_schedule_before_spawn() {
        let project = tempfile::tempdir().unwrap();
        for (path, invalid_value, expected_error) in [
            (
                "recovering.ball_hz",
                serde_json::json!(-1.0),
                "must be finite and non-negative",
            ),
            ("quiet_dwell_ms", serde_json::json!(0), "must be positive"),
            (
                "layout_position_tolerance_px",
                serde_json::json!(0.0),
                "must be finite and positive",
            ),
        ] {
            let mut configuration = remote_stream_config();
            let mut target = &mut configuration["perception"]["rust_live"]["phase_schedule"];
            let path_parts = path.split('.').collect::<Vec<_>>();
            for part in &path_parts[..path_parts.len() - 1] {
                target = &mut target[*part];
            }
            target[path_parts[path_parts.len() - 1]] = invalid_value;
            let error = LiveProcessConfig::from_effective_configuration(
                &configuration,
                project.path(),
                "127.0.0.1:49321".parse().unwrap(),
                None,
                None,
                None,
            )
            .unwrap_err();
            assert!(error.contains("invalid perception.rust_live.phase_schedule"));
            assert!(error.contains(expected_error));
        }

        let mut configuration = remote_stream_config();
        configuration["perception"]["rust_live"]
            .as_object_mut()
            .unwrap()
            .remove("phase_schedule");
        let error = LiveProcessConfig::from_effective_configuration(
            &configuration,
            project.path(),
            "127.0.0.1:49321".parse().unwrap(),
            None,
            None,
            None,
        )
        .unwrap_err();
        assert!(error.contains("perception.rust_live.phase_schedule is missing"));
    }

    #[test]
    fn rejects_non_openvino_production_provider_before_spawn() {
        let project = tempfile::tempdir().unwrap();
        for provider in ["cpu", "auto", "coral"] {
            let mut configuration = remote_stream_config();
            configuration["perception"]["rust_live"]["provider"] =
                Value::String(provider.to_owned());
            let error = LiveProcessConfig::from_effective_configuration(
                &configuration,
                project.path(),
                "127.0.0.1:49321".parse().unwrap(),
                None,
                None,
                None,
            )
            .unwrap_err();
            assert!(error.contains("requires the OpenVINO GPU provider"));
        }
    }

    #[test]
    fn forwards_exact_live_audit_configuration_to_detector() {
        let project = tempfile::tempdir().unwrap();
        let configuration = remote_stream_config();
        let audit = LiveAuditConfig {
            address: "127.0.0.1:49322".parse().unwrap(),
            queue_capacity: 37,
            maximum_record_bytes: 8_192,
            connect_timeout: Duration::from_millis(1_250),
            write_timeout: Duration::from_millis(875),
        };
        let config = LiveProcessConfig::from_effective_configuration(
            &configuration,
            project.path(),
            "127.0.0.1:49321".parse().unwrap(),
            None,
            None,
            Some(audit),
        )
        .unwrap();
        for (flag, expected) in [
            ("--live-audit-tcp", "127.0.0.1:49322"),
            ("--live-audit-queue-capacity", "37"),
            ("--live-audit-max-record-bytes", "8192"),
            ("--live-audit-connect-timeout-ms", "1250"),
            ("--live-audit-write-timeout-ms", "875"),
        ] {
            assert!(config
                .arguments
                .windows(2)
                .any(|pair| pair == [flag, expected]));
        }
    }

    #[test]
    fn rejects_invalid_live_audit_configuration_before_spawn() {
        let project = tempfile::tempdir().unwrap();
        let configuration = remote_stream_config();
        let error = LiveProcessConfig::from_effective_configuration(
            &configuration,
            project.path(),
            "127.0.0.1:49321".parse().unwrap(),
            None,
            None,
            Some(LiveAuditConfig {
                address: "0.0.0.0:49322".parse().unwrap(),
                queue_capacity: 37,
                maximum_record_bytes: 8_192,
                connect_timeout: Duration::from_millis(1_250),
                write_timeout: Duration::from_millis(875),
            }),
        )
        .unwrap_err();
        assert!(error.contains("invalid live audit configuration"));
        assert!(error.contains("must be loopback"));
    }

    #[test]
    fn frame_preview_arguments_match_live_detector_cli_contract() {
        let project = tempfile::tempdir().unwrap();
        let mut configuration = remote_stream_config();
        configuration["perception"]["rust_live"]["frame_preview_enabled"] = Value::Bool(true);
        let config = LiveProcessConfig::from_effective_configuration(
            &configuration,
            project.path(),
            "127.0.0.1:49321".parse().unwrap(),
            None,
            None,
            None,
        )
        .unwrap();
        let arguments = config.arguments;

        assert!(arguments
            .windows(2)
            .any(|pair| pair == ["--frame-preview-tcp", "127.0.0.1:49322"]));
        assert!(arguments
            .windows(2)
            .any(|pair| { pair == ["--frame-preview-max-header-bytes", "8192"] }));
        assert!(arguments
            .windows(2)
            .any(|pair| { pair == ["--frame-preview-max-jpeg-bytes", "4194304"] }));
        assert!(arguments
            .windows(2)
            .any(|pair| { pair == ["--frame-preview-max-concurrent-encodes", "1"] }));
        assert!(arguments
            .windows(2)
            .any(|pair| { pair == ["--frame-preview-calibration-max-request-bytes", "65536"] }));
        assert!(!arguments.iter().any(|argument| {
            argument == "--frame-preview-bind-address" || argument == "--frame-preview-port"
        }));
    }

    #[test]
    fn disabled_frame_preview_adds_no_preview_arguments() {
        let project = tempfile::tempdir().unwrap();
        let mut configuration = remote_stream_config();
        configuration["perception"]["rust_live"]["frame_preview_enabled"] = Value::Bool(false);
        let config = LiveProcessConfig::from_effective_configuration(
            &configuration,
            project.path(),
            "127.0.0.1:49321".parse().unwrap(),
            None,
            None,
            None,
        )
        .unwrap();

        assert!(!config
            .arguments
            .iter()
            .any(|argument| argument.starts_with("--frame-preview-")));
    }

    #[test]
    fn frame_preview_rejects_non_loopback_listener_address() {
        let project = tempfile::tempdir().unwrap();
        let mut configuration = remote_stream_config();
        configuration["perception"]["rust_live"]["frame_preview_enabled"] = Value::Bool(true);
        configuration["perception"]["rust_live"]["frame_preview_bind_address"] =
            Value::String("0.0.0.0".to_owned());
        let error = LiveProcessConfig::from_effective_configuration(
            &configuration,
            project.path(),
            "127.0.0.1:49321".parse().unwrap(),
            None,
            None,
            None,
        )
        .unwrap_err();

        assert!(error.contains("frame-preview bind address must be loopback"));
    }

    #[test]
    fn rejects_live_launch_when_capture_is_neither_stable_camera_nor_http_stream() {
        let project = tempfile::tempdir().unwrap();
        let mut configuration = remote_stream_config();
        configuration["vision"]["camera"]["device_id"] = serde_json::json!(0);
        let error = LiveProcessConfig::from_effective_configuration(
            &configuration,
            project.path(),
            "127.0.0.1:49321".parse().unwrap(),
            None,
            None,
            None,
        )
        .unwrap_err();
        assert!(
            error.contains("stable camera selector"),
            "unexpected configuration error: {error}"
        );
    }

    #[tokio::test]
    async fn each_child_restart_receives_a_new_transport_epoch() {
        let directory = tempfile::tempdir().unwrap();
        let executable = directory.path().join("fake-live");
        let arguments_log = directory.path().join("arguments.log");
        let arguments_log = arguments_log.display().to_string().replace('\'', "'\\''");
        std::fs::write(
            &executable,
            format!(
                "#!/bin/sh\nprintf '%s\\n' \"$@\" >> '{arguments_log}'\nprintf '\\n' >> '{arguments_log}'\ncat >/dev/null\n"
            ),
        )
        .unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&executable, std::fs::Permissions::from_mode(0o755)).unwrap();
        }

        let config = LiveProcessConfig {
            executable,
            project_root: directory.path().to_path_buf(),
            capture_request: br#"{"schema_version":3}"#.to_vec(),
            arguments: Vec::new(),
            graceful_stop_timeout: Duration::from_millis(100),
            restart_initial_delay: Duration::from_millis(10),
            restart_maximum_delay: Duration::from_millis(100),
            diagnostics: None,
        };
        for _ in 0..2 {
            let mut child = config.spawn_child().await.unwrap();
            let status = child.wait().await.unwrap();
            assert!(status.success());

            // The shutdown path is a no-op for an already reaped child and
            // remains safe if the service exits between signal and wait.
            stop_child(&mut child, Duration::from_millis(10)).await;
        }

        let logged_arguments =
            std::fs::read_to_string(directory.path().join("arguments.log")).unwrap();
        let launches: Vec<Vec<&str>> = logged_arguments
            .split("\n\n")
            .filter(|launch| !launch.trim().is_empty())
            .map(|launch| launch.lines().collect())
            .collect();
        assert_eq!(launches.len(), 2);
        fn value_after<'a>(arguments: &[&'a str], name: &str) -> &'a str {
            let index = arguments
                .iter()
                .position(|argument| *argument == name)
                .unwrap_or_else(|| panic!("missing {name} in {arguments:?}"));
            arguments[index + 1]
        }
        let engine_ids = launches
            .iter()
            .map(|arguments| value_after(arguments, "--engine-instance-id"))
            .collect::<Vec<_>>();
        let session_ids = launches
            .iter()
            .map(|arguments| value_after(arguments, "--source-session-id"))
            .collect::<Vec<_>>();
        assert_ne!(engine_ids[0], engine_ids[1]);
        assert_ne!(session_ids[0], session_ids[1]);
        assert!(engine_ids
            .iter()
            .all(|value| Uuid::parse_str(value).is_ok()));
        assert!(session_ids
            .iter()
            .all(|value| Uuid::parse_str(value).is_ok()));
    }
}
