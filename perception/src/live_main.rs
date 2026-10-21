use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{ensure, Context, Result};
use billiards_perception::capture::{
    CaptureInterrupt, CaptureOpenError, CaptureSource, CaptureSourceFactory,
};
#[cfg(feature = "camera-gstreamer-source")]
use billiards_perception::gstreamer_capture_source::{
    GstreamerCaptureSourceConfig, GstreamerCaptureSourceFactory,
};
use billiards_perception::inference::ProviderSelection;
use billiards_perception::live_application::{
    read_capture_settings, run_live_application_with_audit, scene_transport_config,
    LiveApplicationSettings,
};
use billiards_perception::live_audit::LiveAuditConfig;
use billiards_perception::ImageSpace;
use clap::{ArgAction, Parser, ValueEnum};

#[derive(Debug, Clone, Copy, ValueEnum)]
enum ProviderArgument {
    Cpu,
    Auto,
    Openvino,
}

impl From<ProviderArgument> for ProviderSelection {
    fn from(value: ProviderArgument) -> Self {
        match value {
            ProviderArgument::Cpu => Self::Cpu,
            ProviderArgument::Auto => Self::Auto,
            ProviderArgument::Openvino => Self::Openvino,
        }
    }
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum ImageSpaceArgument {
    Raw,
    Rectified,
}

#[derive(Debug, Clone, Copy, ValueEnum, PartialEq, Eq)]
enum CaptureSourceArgument {
    /// Keep camera capture unavailable unless the explicit opt-in is selected.
    Unavailable,
    /// Use the opt-in GStreamer helper with the exact v3 discovery selector.
    GstreamerPhysical,
}

impl From<ImageSpaceArgument> for ImageSpace {
    fn from(value: ImageSpaceArgument) -> Self {
        match value {
            ImageSpaceArgument::Raw => Self::Raw,
            ImageSpaceArgument::Rectified => Self::Rectified,
        }
    }
}

#[derive(Debug, Parser)]
#[command(
    name = "billiards-perception-live",
    about = "Run the Rust-owned live perception process",
    long_about = "Consumes one exact capture-settings JSON request from stdin and runs the live detector, tracker and protocol-v3 scene transport. A platform capture adapter must be compiled into this executable before it can open a camera."
)]
struct Arguments {
    /// Explicit ONNX export. Its sibling manifest and source checkpoint must match.
    #[arg(long)]
    onnx: PathBuf,

    /// Explicit source .pt checkpoint whose digest is recorded in the export manifest.
    #[arg(long)]
    source_checkpoint: PathBuf,

    /// Select ONNX Runtime CPU, native OpenVINO CPU, or ONNX Runtime's max-efficiency policy.
    #[arg(long, value_enum)]
    provider: ProviderArgument,

    /// Fixed square model input size; must match the export manifest.
    #[arg(long)]
    image_size: u32,

    /// Minimum class score.
    #[arg(long)]
    confidence: f32,

    /// Class-aware NMS IoU threshold.
    #[arg(long)]
    iou: f32,

    /// Maximum detections retained after NMS.
    #[arg(long)]
    max_detections: usize,

    /// Optional explicit inference thread count; applied to the selected backend when supported.
    #[arg(long)]
    intra_op_threads: Option<usize>,

    /// Number of synthetic zero-RGB inputs used to warm the real model before source open.
    #[arg(long)]
    warmup_runs: usize,

    /// The supervisor's one-shot exact request is supplied on stdin.
    #[arg(long, action = ArgAction::SetTrue, required = true)]
    capture_settings_stdin: bool,

    /// Maximum bytes accepted from the one-shot capture-settings stdin request.
    #[arg(long)]
    capture_settings_max_bytes: usize,

    /// Physical capture remains unavailable unless explicitly selected.
    #[arg(long, value_enum, default_value_t = CaptureSourceArgument::Unavailable)]
    capture_source: CaptureSourceArgument,

    /// Absolute path to the opt-in GStreamer capture helper executable.
    #[arg(long)]
    capture_helper_executable: Option<PathBuf>,

    /// Maximum bytes in the helper-ready record.
    #[arg(long)]
    capture_helper_ready_record_max_bytes: Option<usize>,

    /// Maximum bytes in each helper frame header.
    #[arg(long)]
    capture_helper_frame_header_max_bytes: Option<usize>,

    /// Artificial delay before helper pulls; production callers should pass zero.
    #[arg(long)]
    capture_helper_slow_reader_delay_ms: Option<u64>,

    /// Maximum time the helper waits for one camera sample.
    #[arg(long)]
    capture_helper_pull_timeout_ms: Option<u64>,

    /// Parent deadline for helper startup and the first valid camera frame.
    #[arg(long)]
    capture_helper_readiness_timeout_ms: Option<u64>,

    /// Maximum wait for clean helper exit after stdout reaches EOF.
    #[arg(long)]
    capture_helper_child_exit_timeout_ms: Option<u64>,

    /// Poll interval while confirming helper exit.
    #[arg(long)]
    capture_helper_child_exit_poll_interval_ms: Option<u64>,

    /// Source image space claimed by the capture request.
    #[arg(long, value_enum)]
    source_image_space: ImageSpaceArgument,

    /// Unique identity for this inference engine process.
    #[arg(long)]
    engine_instance_id: String,

    /// Source-session identity allocated by the Python gateway.
    #[arg(long)]
    source_session_id: String,

    /// Model class selected as pool balls.
    #[arg(long)]
    tracking_ball_class_id: u32,

    /// Optional model class selected as the cue ball.
    #[arg(long)]
    tracking_cue_class_id: Option<u32>,

    /// Camera profile associated with tracked coordinates.
    #[arg(long)]
    tracking_camera_profile_id: String,

    /// Maximum center displacement used for association, in source pixels.
    #[arg(long)]
    tracking_association_gate_px: f64,

    /// Maximum forward prediction horizon used only for association, in milliseconds.
    #[arg(long)]
    tracking_association_prediction_horizon_ms: u64,

    /// Age after which an unmatched ball is marked lost, in milliseconds.
    #[arg(long)]
    tracking_lost_after_ms: u64,

    /// Maximum observation age before a track is removed, in milliseconds.
    #[arg(long)]
    tracking_maximum_track_age_ms: u64,

    /// Bound on simultaneously retained ball tracks.
    #[arg(long)]
    tracking_maximum_active_tracks: usize,

    /// Number of recent source frame IDs retained for duplicate detection.
    #[arg(long)]
    tracking_recent_observation_capacity: usize,

    /// Effective tracking configuration revision, copied from the request.
    #[arg(long)]
    tracking_configuration_revision: u64,

    /// Effective source image rectification revision, copied from the request.
    #[arg(long)]
    tracking_rectification_revision: u64,

    /// Local protocol-v3 scene receiver.
    #[arg(long)]
    scene_snapshots_tcp: SocketAddr,

    /// Maximum bytes per outbound protocol-v3 scene frame.
    #[arg(long)]
    scene_snapshots_max_message_bytes: usize,

    /// TCP connect and handshake deadline, in milliseconds.
    #[arg(long)]
    scene_snapshots_connect_timeout_ms: u64,

    /// TCP write deadline, in milliseconds.
    #[arg(long)]
    scene_snapshots_write_timeout_ms: u64,

    /// Initial asynchronous scene reconnect delay, in milliseconds.
    #[arg(long)]
    scene_snapshots_reconnect_initial_delay_ms: u64,

    /// Maximum asynchronous scene reconnect delay, in milliseconds.
    #[arg(long)]
    scene_snapshots_reconnect_maximum_delay_ms: u64,

    /// Maximum time for request-bound source and receiver readiness, in milliseconds.
    #[arg(long)]
    startup_timeout_ms: u64,

    /// Maximum time to wait per readiness/frame iteration and signal check, in milliseconds.
    #[arg(long)]
    poll_interval_ms: u64,

    /// Child-side source-release deadline; the parent process-reap deadline must be longer.
    #[arg(long)]
    graceful_stop_timeout_ms: u64,

    /// Optional unauthenticated loopback endpoint for versioned timing records.
    /// All --live-audit-* settings must be supplied together.
    #[arg(long)]
    live_audit_tcp: Option<SocketAddr>,

    /// Maximum number of pending scalar audit records; accepted range is 1..=4096.
    #[arg(long)]
    live_audit_queue_capacity: Option<usize>,

    /// Maximum framed JSON audit record bytes; accepted range is 1024..=1048576.
    #[arg(long)]
    live_audit_max_record_bytes: Option<usize>,

    /// Audit receiver connection deadline, in milliseconds.
    #[arg(long)]
    live_audit_connect_timeout_ms: Option<u64>,

    /// Audit record socket-write deadline, in milliseconds.
    #[arg(long)]
    live_audit_write_timeout_ms: Option<u64>,
}

fn main() -> Result<()> {
    let arguments = Arguments::parse();
    ensure!(
        arguments.capture_settings_stdin,
        "--capture-settings-stdin is required"
    );

    let termination_requested = Arc::new(AtomicBool::new(false));
    signal_hook::flag::register(
        signal_hook::consts::signal::SIGTERM,
        Arc::clone(&termination_requested),
    )
    .context("failed to register SIGTERM shutdown handler")?;
    signal_hook::flag::register(
        signal_hook::consts::signal::SIGINT,
        Arc::clone(&termination_requested),
    )
    .context("failed to register SIGINT shutdown handler")?;

    let mut stdin = std::io::stdin().lock();
    let exact_capture_settings =
        read_capture_settings(&mut stdin, arguments.capture_settings_max_bytes)?;
    let live_audit_config = build_live_audit_config(&arguments)?;
    let capture_audit_timing = live_audit_config.is_some();
    let source_factory = build_capture_source_factory(
        &arguments,
        exact_capture_settings.as_bytes(),
        capture_audit_timing,
    )?;

    let settings = LiveApplicationSettings {
        onnx_path: arguments.onnx,
        source_checkpoint_path: arguments.source_checkpoint,
        provider: arguments.provider.into(),
        image_size: arguments.image_size,
        confidence: arguments.confidence,
        iou: arguments.iou,
        max_detections: arguments.max_detections,
        intra_op_threads: arguments.intra_op_threads,
        warmup_runs: arguments.warmup_runs,
        source_image_space: arguments.source_image_space.into(),
        engine_instance_id: arguments.engine_instance_id,
        source_session_id: arguments.source_session_id,
        tracking_ball_class_id: arguments.tracking_ball_class_id,
        tracking_cue_class_id: arguments.tracking_cue_class_id,
        tracking_camera_profile_id: arguments.tracking_camera_profile_id,
        tracking_association_gate_px: arguments.tracking_association_gate_px,
        tracking_association_prediction_horizon: Duration::from_millis(
            arguments.tracking_association_prediction_horizon_ms,
        ),
        tracking_lost_after: Duration::from_millis(arguments.tracking_lost_after_ms),
        tracking_maximum_track_age: Duration::from_millis(arguments.tracking_maximum_track_age_ms),
        tracking_maximum_active_tracks: arguments.tracking_maximum_active_tracks,
        tracking_recent_observation_capacity: arguments.tracking_recent_observation_capacity,
        tracking_configuration_revision: arguments.tracking_configuration_revision,
        tracking_rectification_revision: arguments.tracking_rectification_revision,
        scene_transport: scene_transport_config(
            arguments.scene_snapshots_tcp,
            arguments.scene_snapshots_max_message_bytes,
            Duration::from_millis(arguments.scene_snapshots_connect_timeout_ms),
            Duration::from_millis(arguments.scene_snapshots_write_timeout_ms),
            Duration::from_millis(arguments.scene_snapshots_reconnect_initial_delay_ms),
            Duration::from_millis(arguments.scene_snapshots_reconnect_maximum_delay_ms),
        ),
        startup_timeout: Duration::from_millis(arguments.startup_timeout_ms),
        poll_interval: Duration::from_millis(arguments.poll_interval_ms),
        graceful_stop_timeout: Duration::from_millis(arguments.graceful_stop_timeout_ms),
        capture_settings_maximum_bytes: arguments.capture_settings_max_bytes,
    };

    match source_factory {
        LiveCaptureSourceFactory::Unavailable(factory) => run_live_application_with_audit(
            &exact_capture_settings,
            settings,
            factory,
            &termination_requested,
            live_audit_config,
        ),
        #[cfg(feature = "camera-gstreamer-source")]
        LiveCaptureSourceFactory::GstreamerPhysical(factory) => run_live_application_with_audit(
            &exact_capture_settings,
            settings,
            *factory,
            &termination_requested,
            live_audit_config,
        ),
    }
}

fn build_live_audit_config(arguments: &Arguments) -> Result<Option<LiveAuditConfig>> {
    let supplied = [
        arguments.live_audit_tcp.is_some(),
        arguments.live_audit_queue_capacity.is_some(),
        arguments.live_audit_max_record_bytes.is_some(),
        arguments.live_audit_connect_timeout_ms.is_some(),
        arguments.live_audit_write_timeout_ms.is_some(),
    ];
    if supplied.iter().all(|value| !value) {
        return Ok(None);
    }
    ensure!(
        supplied.iter().all(|value| *value),
        "all --live-audit-* options must be supplied together"
    );

    let config = LiveAuditConfig {
        address: arguments.live_audit_tcp.expect("validated above"),
        queue_capacity: arguments
            .live_audit_queue_capacity
            .expect("validated above"),
        maximum_record_bytes: arguments
            .live_audit_max_record_bytes
            .expect("validated above"),
        connect_timeout: Duration::from_millis(
            arguments
                .live_audit_connect_timeout_ms
                .expect("validated above"),
        ),
        write_timeout: Duration::from_millis(
            arguments
                .live_audit_write_timeout_ms
                .expect("validated above"),
        ),
    };
    config.validate()?;
    Ok(Some(config))
}

enum LiveCaptureSourceFactory {
    Unavailable(UnavailableCaptureSourceFactory),
    #[cfg(feature = "camera-gstreamer-source")]
    GstreamerPhysical(Box<GstreamerCaptureSourceFactory>),
}

fn build_capture_source_factory(
    arguments: &Arguments,
    exact_capture_settings: &[u8],
    capture_audit_timing: bool,
) -> Result<LiveCaptureSourceFactory> {
    #[cfg(not(feature = "camera-gstreamer-source"))]
    let _ = capture_audit_timing;

    match arguments.capture_source {
        CaptureSourceArgument::Unavailable => {
            ensure!(
                !has_capture_helper_options(arguments),
                "capture-helper options require --capture-source gstreamer-physical"
            );
            Ok(LiveCaptureSourceFactory::Unavailable(
                UnavailableCaptureSourceFactory,
            ))
        }
        CaptureSourceArgument::GstreamerPhysical => {
            #[cfg(feature = "camera-gstreamer-source")]
            {
                let config = GstreamerCaptureSourceConfig {
                    helper_executable_path: required_capture_option(
                        arguments.capture_helper_executable.clone(),
                        "--capture-helper-executable",
                    )?,
                    exact_capture_settings_json: exact_capture_settings.to_vec(),
                    capture_settings_max_bytes: arguments.capture_settings_max_bytes,
                    ready_record_max_bytes: required_capture_option(
                        arguments.capture_helper_ready_record_max_bytes,
                        "--capture-helper-ready-record-max-bytes",
                    )?,
                    frame_header_max_bytes: required_capture_option(
                        arguments.capture_helper_frame_header_max_bytes,
                        "--capture-helper-frame-header-max-bytes",
                    )?,
                    frame_count: None,
                    slow_reader_delay_ms: required_capture_option(
                        arguments.capture_helper_slow_reader_delay_ms,
                        "--capture-helper-slow-reader-delay-ms",
                    )?,
                    pull_timeout_ms: required_capture_option(
                        arguments.capture_helper_pull_timeout_ms,
                        "--capture-helper-pull-timeout-ms",
                    )?,
                    capture_audit_timing,
                    readiness_timeout: Duration::from_millis(required_capture_option(
                        arguments.capture_helper_readiness_timeout_ms,
                        "--capture-helper-readiness-timeout-ms",
                    )?),
                    child_exit_timeout: Duration::from_millis(required_capture_option(
                        arguments.capture_helper_child_exit_timeout_ms,
                        "--capture-helper-child-exit-timeout-ms",
                    )?),
                    child_exit_poll_interval: Duration::from_millis(required_capture_option(
                        arguments.capture_helper_child_exit_poll_interval_ms,
                        "--capture-helper-child-exit-poll-interval-ms",
                    )?),
                };
                let factory = GstreamerCaptureSourceFactory::new_physical(config)
                    .context("invalid opt-in physical GStreamer source configuration")?;
                Ok(LiveCaptureSourceFactory::GstreamerPhysical(Box::new(
                    factory,
                )))
            }
            #[cfg(not(feature = "camera-gstreamer-source"))]
            {
                let _ = (arguments, exact_capture_settings);
                anyhow::bail!(
                    "physical GStreamer capture was not compiled; build with camera-gstreamer-source explicitly"
                )
            }
        }
    }
}

fn has_capture_helper_options(arguments: &Arguments) -> bool {
    arguments.capture_helper_executable.is_some()
        || arguments.capture_helper_ready_record_max_bytes.is_some()
        || arguments.capture_helper_frame_header_max_bytes.is_some()
        || arguments.capture_helper_slow_reader_delay_ms.is_some()
        || arguments.capture_helper_pull_timeout_ms.is_some()
        || arguments.capture_helper_readiness_timeout_ms.is_some()
        || arguments.capture_helper_child_exit_timeout_ms.is_some()
        || arguments
            .capture_helper_child_exit_poll_interval_ms
            .is_some()
}

#[cfg(feature = "camera-gstreamer-source")]
fn required_capture_option<T>(value: Option<T>, argument_name: &str) -> Result<T> {
    value.ok_or_else(|| anyhow::anyhow!("{argument_name} is required for physical capture"))
}

/// Default for builds and sessions without the explicit physical-source
/// selection. The feature-gated helper remains unqualified for production.
struct UnavailableCaptureSourceFactory;

struct NoopCaptureInterrupt;

impl CaptureInterrupt for NoopCaptureInterrupt {
    fn interrupt(&self) {}
}

impl CaptureSourceFactory for UnavailableCaptureSourceFactory {
    fn interrupt_handle(&self) -> Arc<dyn CaptureInterrupt> {
        Arc::new(NoopCaptureInterrupt)
    }

    fn open(self: Box<Self>) -> std::result::Result<Box<dyn CaptureSource>, CaptureOpenError> {
        Err(CaptureOpenError::Failed(
            "no qualified Rust camera adapter is compiled into this build".to_owned(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use clap::Parser;

    use super::{
        build_capture_source_factory, build_live_audit_config, Arguments, CaptureSourceArgument,
        LiveCaptureSourceFactory, ProviderArgument,
    };

    #[test]
    fn live_cli_accepts_the_explicit_supervisor_process_contract() {
        let arguments = Arguments::try_parse_from(supervisor_arguments()).unwrap();
        assert!(arguments.capture_settings_stdin);
        assert_eq!(arguments.source_session_id, "session-test");
        assert_eq!(arguments.engine_instance_id, "engine-test");
        assert_eq!(arguments.capture_settings_max_bytes, 32_768);
        assert_eq!(arguments.max_detections, 300);
        assert_eq!(arguments.intra_op_threads, Some(2));
        assert_eq!(arguments.startup_timeout_ms, 10_000);
        assert_eq!(arguments.poll_interval_ms, 25);
        assert_eq!(arguments.graceful_stop_timeout_ms, 2_000);
        assert_eq!(arguments.capture_source, CaptureSourceArgument::Unavailable);
        assert!(matches!(
            build_capture_source_factory(&arguments, &[], false).unwrap(),
            LiveCaptureSourceFactory::Unavailable(_)
        ));
        assert!(build_live_audit_config(&arguments).unwrap().is_none());
    }

    #[test]
    fn live_cli_accepts_native_openvino_as_an_explicit_runtime_choice() {
        let mut arguments = supervisor_arguments();
        let provider_position = arguments
            .iter()
            .position(|argument| *argument == "--provider")
            .unwrap();
        arguments[provider_position + 1] = "openvino";
        let parsed = Arguments::try_parse_from(arguments).unwrap();
        assert!(matches!(parsed.provider, ProviderArgument::Openvino));
        assert_eq!(
            billiards_perception::inference::ProviderSelection::from(parsed.provider),
            billiards_perception::inference::ProviderSelection::Openvino
        );
    }

    #[test]
    fn live_cli_requires_a_complete_valid_loopback_audit_configuration() {
        let mut arguments = supervisor_arguments();
        arguments.extend(["--live-audit-tcp", "127.0.0.1:9001"]);
        let parsed = Arguments::try_parse_from(arguments.clone()).unwrap();
        assert!(build_live_audit_config(&parsed).is_err());

        arguments.extend([
            "--live-audit-queue-capacity",
            "128",
            "--live-audit-max-record-bytes",
            "16384",
            "--live-audit-connect-timeout-ms",
            "500",
            "--live-audit-write-timeout-ms",
            "500",
        ]);
        let parsed = Arguments::try_parse_from(arguments.clone()).unwrap();
        let audit = build_live_audit_config(&parsed).unwrap().unwrap();
        assert_eq!(audit.queue_capacity, 128);
        assert_eq!(audit.maximum_record_bytes, 16_384);

        let remote_audit_option = arguments
            .iter()
            .position(|argument| *argument == "127.0.0.1:9001")
            .unwrap();
        arguments[remote_audit_option] = "192.0.2.1:9001";
        let parsed = Arguments::try_parse_from(arguments).unwrap();
        assert!(build_live_audit_config(&parsed).is_err());
    }

    #[test]
    fn live_cli_requires_one_shot_capture_settings_stdin() {
        let mut arguments = supervisor_arguments();
        let flag_position = arguments
            .iter()
            .position(|argument| *argument == "--capture-settings-stdin")
            .unwrap();
        arguments.remove(flag_position);
        assert!(Arguments::try_parse_from(arguments).is_err());
    }

    #[test]
    fn unavailable_source_rejects_ignored_helper_settings() {
        let mut arguments = supervisor_arguments();
        arguments.extend([
            "--capture-helper-executable",
            "/opt/billiards/billiards-perception-gstreamer-probe",
        ]);
        let parsed = Arguments::try_parse_from(arguments).unwrap();
        let error = build_capture_source_factory(&parsed, &[], false)
            .err()
            .unwrap();
        assert!(error.to_string().contains("require --capture-source"));
    }

    #[cfg(feature = "camera-gstreamer-source")]
    #[test]
    fn physical_source_factory_uses_the_exact_v3_selector_and_continuous_mode() {
        let mut arguments = supervisor_arguments();
        arguments.extend(physical_helper_arguments());
        let parsed = Arguments::try_parse_from(arguments).unwrap();
        let request = physical_capture_settings();
        assert!(matches!(
            build_capture_source_factory(&parsed, request.as_bytes(), false).unwrap(),
            LiveCaptureSourceFactory::GstreamerPhysical(_)
        ));
    }

    #[cfg(feature = "camera-gstreamer-source")]
    #[test]
    fn physical_source_factory_rejects_missing_helper_launch_settings() {
        let mut arguments = supervisor_arguments();
        arguments.extend(["--capture-source", "gstreamer-physical"]);
        let parsed = Arguments::try_parse_from(arguments).unwrap();
        let error =
            build_capture_source_factory(&parsed, physical_capture_settings().as_bytes(), false)
                .err()
                .unwrap();
        assert!(error
            .to_string()
            .contains("--capture-helper-executable is required"));
    }

    #[cfg(not(feature = "camera-gstreamer-source"))]
    #[test]
    fn physical_source_selection_fails_closed_without_opt_in_feature() {
        let mut arguments = supervisor_arguments();
        arguments.extend(physical_helper_arguments());
        let parsed = Arguments::try_parse_from(arguments).unwrap();
        let error =
            build_capture_source_factory(&parsed, physical_capture_settings().as_bytes(), false)
                .err()
                .unwrap();
        assert!(error.to_string().contains("was not compiled"));
    }

    fn physical_helper_arguments() -> Vec<&'static str> {
        vec![
            "--capture-source",
            "gstreamer-physical",
            "--capture-helper-executable",
            "/opt/billiards/billiards-perception-gstreamer-probe",
            "--capture-helper-ready-record-max-bytes",
            "4096",
            "--capture-helper-frame-header-max-bytes",
            "4096",
            "--capture-helper-slow-reader-delay-ms",
            "0",
            "--capture-helper-pull-timeout-ms",
            "2000",
            "--capture-helper-readiness-timeout-ms",
            "5000",
            "--capture-helper-child-exit-timeout-ms",
            "3000",
            "--capture-helper-child-exit-poll-interval-ms",
            "5",
        ]
    }

    fn physical_capture_settings() -> String {
        let mut request: serde_json::Value = serde_json::from_str(include_str!(
            "../contracts/capture-settings/v3/capture-settings-default.json"
        ))
        .unwrap();
        request["requested"]["device_id"] =
            serde_json::Value::String(format!("gstreamer-device-v1:{}", "a".repeat(64)));
        serde_json::to_string(&request).unwrap()
    }

    fn supervisor_arguments() -> Vec<&'static str> {
        vec![
            "billiards-perception-live",
            "--onnx",
            "/models/detector.onnx",
            "--source-checkpoint",
            "/models/detector.pt",
            "--provider",
            "cpu",
            "--image-size",
            "640",
            "--confidence",
            "0.4",
            "--iou",
            "0.45",
            "--max-detections",
            "300",
            "--intra-op-threads",
            "2",
            "--warmup-runs",
            "1",
            "--capture-settings-stdin",
            "--capture-settings-max-bytes",
            "32768",
            "--source-image-space",
            "raw",
            "--engine-instance-id",
            "engine-test",
            "--source-session-id",
            "session-test",
            "--tracking-ball-class-id",
            "1",
            "--tracking-cue-class-id",
            "0",
            "--tracking-camera-profile-id",
            "camera-profile-test",
            "--tracking-association-gate-px",
            "40.0",
            "--tracking-association-prediction-horizon-ms",
            "100",
            "--tracking-lost-after-ms",
            "1000",
            "--tracking-maximum-track-age-ms",
            "3000",
            "--tracking-maximum-active-tracks",
            "32",
            "--tracking-recent-observation-capacity",
            "8",
            "--tracking-configuration-revision",
            "1",
            "--tracking-rectification-revision",
            "0",
            "--scene-snapshots-tcp",
            "127.0.0.1:9000",
            "--scene-snapshots-max-message-bytes",
            "65536",
            "--scene-snapshots-connect-timeout-ms",
            "1000",
            "--scene-snapshots-write-timeout-ms",
            "1000",
            "--scene-snapshots-reconnect-initial-delay-ms",
            "100",
            "--scene-snapshots-reconnect-maximum-delay-ms",
            "2000",
            "--startup-timeout-ms",
            "10000",
            "--poll-interval-ms",
            "25",
            "--graceful-stop-timeout-ms",
            "2000",
        ]
    }
}
