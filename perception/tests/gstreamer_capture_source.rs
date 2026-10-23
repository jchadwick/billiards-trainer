#![cfg(feature = "camera-gstreamer-probe")]

use std::fs;
use std::path::PathBuf;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use billiards_perception::capture::{
    CaptureAggregateFrameLimit, CaptureCapability, CaptureControlReadback, CaptureFrameLimit,
    CapturePreflightStatus, CaptureReceiveOutcome, CaptureRuntimeCoordinator, CaptureRuntimeWait,
    CaptureTimestampSemantics, CaptureWorkerEnd,
};
use billiards_perception::capture_settings_wire::parse_capture_settings_json;
use billiards_perception::gstreamer_capture_source::{
    GstreamerCaptureSourceConfig, GstreamerCaptureSourceFactory,
};
use serde_json::Value;

const MAX_SETTINGS_BYTES: usize = 16_384;
const READY_RECORD_MAX_BYTES: usize = 4_096;
const FRAME_HEADER_MAX_BYTES: usize = 4_096;
const FRAME_COUNT: u64 = 6;
const SLOW_READER_DELAY_MS: u64 = 0;
const PULL_TIMEOUT_MS: u64 = 2_000;
const READINESS_TIMEOUT: Duration = Duration::from_secs(5);
const CHILD_EXIT_TIMEOUT: Duration = Duration::from_secs(3);
const CHILD_EXIT_POLL_INTERVAL: Duration = Duration::from_millis(5);
const EXPECTED_WIDTH: u32 = 64;
const EXPECTED_HEIGHT: u32 = 48;
const FRAME_LIMIT_BYTES: usize = 1_000_000;
const AGGREGATE_LIMIT_BYTES: usize = 2_000_000;

// The coordinator's runtime owner gate and aggregate budget are process-wide.
// These integration cases intentionally run one real source session at a time.
static RUNTIME_TEST_LOCK: Mutex<()> = Mutex::new(());

#[test]
fn synthetic_helper_runs_through_capture_worker_with_verified_metadata_and_frames() {
    let _runtime = RUNTIME_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let (factory, monitor, resolved) = helper_factory(FRAME_COUNT);
    let mut coordinator = runtime();
    let session_id = coordinator.start_unvalidated(factory).unwrap();
    let mut receiver = coordinator.claim_receiver().unwrap();
    let source_report = wait_for_active_report(&mut coordinator);

    assert_eq!(source_report.session_id(), session_id);
    assert_eq!(
        source_report.device_id(),
        "synthetic-gstreamer-videotestsrc"
    );
    assert_eq!(source_report.backend_id(), "gstreamer");
    assert_eq!(
        (
            source_report.negotiated_profile().width_px(),
            source_report.negotiated_profile().height_px()
        ),
        (EXPECTED_WIDTH, EXPECTED_HEIGHT)
    );
    assert_eq!(
        source_report
            .negotiated_profile()
            .pixel_format()
            .identifier(),
        "RGB8"
    );
    assert_eq!(
        source_report.timestamp_semantics(),
        &CaptureTimestampSemantics::ReadCompletionOnly
    );
    assert_eq!(
        source_report.capabilities().manual_exposure(),
        CaptureCapability::Unsupported
    );
    assert_eq!(
        source_report.capabilities().automatic_exposure(),
        CaptureCapability::Unsupported
    );
    assert_eq!(
        source_report.capabilities().gain_control(),
        CaptureCapability::Unsupported
    );
    assert_eq!(
        source_report.capabilities().source_timestamps(),
        CaptureCapability::Unsupported
    );
    assert!(matches!(
        source_report.applied_controls().exposure_mode(),
        CaptureControlReadback::Unsupported
    ));
    assert!(matches!(
        source_report.applied_controls().gain(),
        CaptureControlReadback::Unsupported
    ));
    assert_eq!(
        source_report.output_frame_geometry().unwrap().width_px(),
        EXPECTED_WIDTH
    );
    assert_eq!(
        source_report.output_frame_geometry().unwrap().height_px(),
        EXPECTED_HEIGHT
    );
    assert_eq!(
        resolved.expected.frame_limit().maximum_rgb8_bytes(),
        FRAME_LIMIT_BYTES
    );

    let mut observed = Vec::new();
    loop {
        match receiver.receive_with_timeout(Duration::from_secs(3), |frame| {
            (
                frame.source().session_id.clone(),
                frame.source().frame_id.clone(),
                frame.source().timestamp.clone(),
                (frame.source().width_px, frame.source().height_px),
                frame.rgb8().len(),
                frame.host_read_completed_at(),
                frame.capture_timing(),
            )
        }) {
            CaptureReceiveOutcome::Frame(frame) => observed.push(frame),
            CaptureReceiveOutcome::Closed => break,
            CaptureReceiveOutcome::TimedOut => panic!("capture helper did not finish in time"),
            CaptureReceiveOutcome::Cancelled => panic!("capture helper was unexpectedly cancelled"),
            CaptureReceiveOutcome::ProducerFailed(reason) => {
                panic!("capture helper failed while streaming: {reason}")
            }
        }
    }

    assert!(
        observed.len() >= 2,
        "received too few source frames: {observed:?}"
    );
    assert_eq!(observed.first().unwrap().1, "0");
    for frame in &observed {
        assert_eq!(frame.0, session_id);
        assert!(
            frame.2.is_none(),
            "synthetic PTS must not become source time"
        );
        assert_eq!(frame.3, (EXPECTED_WIDTH, EXPECTED_HEIGHT));
        assert_eq!(
            frame.4,
            EXPECTED_WIDTH as usize * EXPECTED_HEIGHT as usize * 3
        );
        assert_eq!(frame.6, Default::default());
        assert!(frame.5 <= Instant::now());
    }
    let frame_ids = observed
        .iter()
        .map(|frame| frame.1.parse::<u64>().unwrap())
        .collect::<Vec<_>>();
    assert!(frame_ids.windows(2).all(|pair| pair[0] < pair[1]));

    // The worker is allowed the configured source-readiness deadline and
    // child-exit deadline to report and reap this intentionally malformed
    // helper. Keep the test bound derived from those settings so slower CI
    // process startup does not turn the protocol assertion into a race.
    let test_deadline = READINESS_TIMEOUT + CHILD_EXIT_TIMEOUT + Duration::from_secs(2);
    let completed = coordinator.wait_timeout(test_deadline);
    let CaptureRuntimeWait::Completed(report) = completed else {
        panic!("capture worker did not complete naturally: {completed:?}");
    };
    assert_eq!(report.end, CaptureWorkerEnd::NaturalEnd);
    assert!(report.frames_published >= 2);
    let budget = coordinator.frame_budget_stats();
    assert_eq!(budget.configured_limit_bytes, AGGREGATE_LIMIT_BYTES);
    assert_eq!(budget.current_bytes, 0);
    assert!(budget.high_water_mark_bytes <= AGGREGATE_LIMIT_BYTES);
    assert!(
        monitor.child_is_reaped(),
        "natural EOF must reap the helper"
    );
}

#[test]
fn opt_in_helper_timings_arrive_with_the_parent_framed_read_duration() {
    let _runtime = RUNTIME_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let (mut config, _) = helper_config(FRAME_COUNT);
    config.capture_audit_timing = true;
    let factory = GstreamerCaptureSourceFactory::new(config).unwrap();
    let mut coordinator = runtime();
    coordinator.start_unvalidated(factory).unwrap();
    let mut receiver = coordinator.claim_receiver().unwrap();

    let timing = match receiver
        .receive_with_timeout(Duration::from_secs(3), |frame| frame.capture_timing())
    {
        CaptureReceiveOutcome::Frame(timing) => timing,
        outcome => panic!("timed helper did not publish a frame: {outcome:?}"),
    };
    let helper = timing
        .helper
        .expect("opt-in helper emits its process-local durations");
    assert!(helper.appsink_pull_wait_ns > 0);
    assert!(helper.rgb_buffer_allocation_ns > 0);
    assert!(helper.rgb_mapping_copy_ns > 0);
    assert!(timing.parent_framed_record_read_ns.is_some());

    let stopped = coordinator.stop(Duration::from_secs(2));
    assert!(matches!(stopped, CaptureRuntimeWait::Completed(_)));
}

#[test]
fn synthetic_helper_preflight_rejects_unsupported_camera_controls_before_reading() {
    let _runtime = RUNTIME_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let (factory, monitor, resolved) = helper_factory(FRAME_COUNT);
    let mut coordinator = runtime();
    let session_id = coordinator
        .start_preflighted(resolved.expected.clone(), factory)
        .unwrap();
    let mut receiver = coordinator.claim_receiver().unwrap();

    let preflight = coordinator
        .wait_for_preflight(Duration::from_secs(5))
        .expect("preflight state is available");
    let preflight_status = match preflight {
        billiards_perception::capture::CapturePreflightWait::Ready(status) => status,
        timeout => panic!("capture preflight did not settle: {timeout:?}"),
    };
    let CapturePreflightStatus::Rejected {
        session_id: rejected_session,
        error,
    } = preflight_status
    else {
        panic!("synthetic camera controls must fail preflight: {preflight_status:?}");
    };
    assert_eq!(rejected_session, session_id);
    assert!(matches!(
        error,
        billiards_perception::live_config::CapturePreflightError::RequiredCapabilityUnavailable {
            capability: "automatic_exposure",
            reported: CaptureCapability::Unsupported,
        }
    ));
    assert!(matches!(
        receiver.receive_with_timeout(Duration::from_secs(2), |_| ()),
        CaptureReceiveOutcome::ProducerFailed(_)
    ));

    let completed = coordinator.wait_timeout(Duration::from_secs(3));
    let CaptureRuntimeWait::Completed(report) = completed else {
        panic!("preflight-rejected helper did not complete release: {completed:?}");
    };
    assert_eq!(report.end, CaptureWorkerEnd::Failed);
    assert_eq!(report.frames_published, 0);
    assert!(monitor.child_is_reaped());
}

#[test]
fn synthetic_helper_v3_source_defaults_pass_preflight_and_preserve_unsupported_controls() {
    let _runtime = RUNTIME_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let (factory, monitor, resolved) = helper_factory_for_contract(
        FRAME_COUNT,
        include_str!("../contracts/capture-settings/v3/capture-settings-default.json"),
    );
    assert!(resolved.expected.requested_control_values().is_none());
    let mut coordinator = runtime();
    let session_id = coordinator
        .start_preflighted(resolved.expected.clone(), factory)
        .unwrap();
    let mut receiver = coordinator.claim_receiver().unwrap();
    let source_report = wait_for_active_report(&mut coordinator);
    assert_eq!(source_report.session_id(), session_id);
    for capability in [
        source_report.capabilities().manual_exposure(),
        source_report.capabilities().automatic_exposure(),
        source_report.capabilities().gain_control(),
    ] {
        assert_eq!(capability, CaptureCapability::Unsupported);
    }
    assert!(matches!(
        source_report.applied_controls().exposure_mode(),
        CaptureControlReadback::Unsupported
    ));
    assert!(matches!(
        source_report.applied_controls().gain(),
        CaptureControlReadback::Unsupported
    ));

    let preflight = coordinator
        .wait_for_preflight(Duration::from_secs(5))
        .expect("preflight state is available");
    assert!(matches!(
        preflight,
        billiards_perception::capture::CapturePreflightWait::Ready(
            CapturePreflightStatus::Compatible(_)
        )
    ));
    assert!(matches!(
        receiver.receive_with_timeout(Duration::from_secs(3), |frame| frame.rgb8().len()),
        CaptureReceiveOutcome::Frame(frame_bytes)
            if frame_bytes == EXPECTED_WIDTH as usize * EXPECTED_HEIGHT as usize * 3
    ));

    let completed = coordinator.wait_timeout(Duration::from_secs(5));
    let CaptureRuntimeWait::Completed(report) = completed else {
        panic!("source-default helper did not complete: {completed:?}");
    };
    assert_eq!(report.end, CaptureWorkerEnd::NaturalEnd);
    assert!(monitor.child_is_reaped());
}

#[test]
fn synthetic_helper_v3_require_policy_still_rejects_unsupported_controls_before_reading() {
    let _runtime = RUNTIME_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let (factory, monitor, resolved) = helper_factory_for_contract(
        FRAME_COUNT,
        include_str!("../contracts/capture-settings/v3/capture-settings-require-controls.json"),
    );
    let mut coordinator = runtime();
    let session_id = coordinator
        .start_preflighted(resolved.expected.clone(), factory)
        .unwrap();
    let mut receiver = coordinator.claim_receiver().unwrap();

    let preflight = coordinator
        .wait_for_preflight(Duration::from_secs(5))
        .expect("preflight state is available");
    let preflight_status = match preflight {
        billiards_perception::capture::CapturePreflightWait::Ready(status) => status,
        timeout => panic!("capture preflight did not settle: {timeout:?}"),
    };
    let CapturePreflightStatus::Rejected {
        session_id: rejected_session,
        error,
    } = preflight_status
    else {
        panic!("v3 require policy must fail on unsupported controls: {preflight_status:?}");
    };
    assert_eq!(rejected_session, session_id);
    assert!(matches!(
        error,
        billiards_perception::live_config::CapturePreflightError::RequiredCapabilityUnavailable {
            capability: "automatic_exposure",
            reported: CaptureCapability::Unsupported,
        }
    ));
    assert!(matches!(
        receiver.receive_with_timeout(Duration::from_secs(2), |_| ()),
        CaptureReceiveOutcome::ProducerFailed(_)
    ));

    let completed = coordinator.wait_timeout(Duration::from_secs(3));
    let CaptureRuntimeWait::Completed(report) = completed else {
        panic!("preflight-rejected helper did not complete: {completed:?}");
    };
    assert_eq!(report.end, CaptureWorkerEnd::Failed);
    assert_eq!(report.frames_published, 0);
    assert!(monitor.child_is_reaped());
}

#[test]
fn stopping_a_blocked_synthetic_read_kills_and_reaps_the_helper_promptly() {
    let _runtime = RUNTIME_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let (factory, monitor, _) = helper_factory(1_000_000);
    let mut coordinator = runtime();
    coordinator.start_unvalidated(factory).unwrap();
    let mut receiver = coordinator.claim_receiver().unwrap();
    assert!(matches!(
        receiver.receive_with_timeout(Duration::from_secs(3), |_| ()),
        CaptureReceiveOutcome::Frame(())
    ));

    let started_at = Instant::now();
    let stopped = coordinator.stop(Duration::from_secs(2));
    let CaptureRuntimeWait::Completed(report) = stopped else {
        panic!("helper stop was not prompt: {stopped:?}");
    };
    assert_eq!(report.end, CaptureWorkerEnd::Stopped);
    assert!(started_at.elapsed() < Duration::from_secs(2));
    assert!(monitor.child_is_reaped());
}

#[test]
fn malformed_truncated_helper_ready_record_fails_open_and_reaps_the_child() {
    let _runtime = RUNTIME_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let fixture_dir = tempfile::tempdir().unwrap();
    let helper_path = fixture_dir.path().join("truncated-helper.sh");
    fs::write(
        &helper_path,
        "#!/bin/sh\ncat >/dev/null\nprintf '\\000\\000'\n",
    )
    .unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut permissions = fs::metadata(&helper_path).unwrap().permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(&helper_path, permissions).unwrap();
    }

    let (mut config, _) = helper_config(FRAME_COUNT);
    config.helper_executable_path = helper_path;
    let factory = GstreamerCaptureSourceFactory::new(config).unwrap();
    let monitor = factory.monitor();
    let mut coordinator = runtime();
    coordinator.start_unvalidated(factory).unwrap();

    let completed = coordinator.wait_timeout(Duration::from_secs(3));
    let CaptureRuntimeWait::Completed(report) = completed else {
        panic!("truncated helper failed to open in time: {completed:?}");
    };
    assert_eq!(report.end, CaptureWorkerEnd::Failed);
    assert_eq!(report.frames_published, 0);
    assert!(monitor.child_is_reaped());
}

fn helper_factory(
    frame_count: u64,
) -> (
    GstreamerCaptureSourceFactory,
    billiards_perception::gstreamer_capture_source::GstreamerCaptureSourceMonitor,
    billiards_perception::capture_settings_wire::ResolvedCaptureSettings,
) {
    let (config, resolved) = helper_config(frame_count);
    let factory = GstreamerCaptureSourceFactory::new(config).unwrap();
    let monitor = factory.monitor();
    (factory, monitor, resolved)
}

fn helper_config(
    frame_count: u64,
) -> (
    GstreamerCaptureSourceConfig,
    billiards_perception::capture_settings_wire::ResolvedCaptureSettings,
) {
    helper_config_for_contract(
        frame_count,
        include_str!("../contracts/capture-settings/v2/capture-settings-default.json"),
    )
}

fn helper_config_for_contract(
    frame_count: u64,
    contract_json: &str,
) -> (
    GstreamerCaptureSourceConfig,
    billiards_perception::capture_settings_wire::ResolvedCaptureSettings,
) {
    let mut request: Value = serde_json::from_str(contract_json).unwrap();
    request["requested"]["device_id"] = Value::String(
        billiards_perception::gstreamer_capture_source::SYNTHETIC_GSTREAMER_DEVICE_ID.to_owned(),
    );
    request["requested"]["requested_profile"]["width_px"] = Value::from(EXPECTED_WIDTH);
    request["requested"]["requested_profile"]["height_px"] = Value::from(EXPECTED_HEIGHT);
    request["requested"]["expected_negotiated_profile"]["width_px"] = Value::from(EXPECTED_WIDTH);
    request["requested"]["expected_negotiated_profile"]["height_px"] = Value::from(EXPECTED_HEIGHT);
    request["requested"]["output_frame_geometry"]["width_px"] = Value::from(EXPECTED_WIDTH);
    request["requested"]["output_frame_geometry"]["height_px"] = Value::from(EXPECTED_HEIGHT);
    request["requested"]["maximum_rgb_frame_bytes"] = Value::from(FRAME_LIMIT_BYTES);
    request["requested"]["maximum_aggregate_rgb_frame_bytes"] = Value::from(AGGREGATE_LIMIT_BYTES);
    let exact_capture_settings_json = serde_json::to_vec(&request).unwrap();
    assert!(exact_capture_settings_json.len() <= MAX_SETTINGS_BYTES);
    let resolved =
        parse_capture_settings_json(std::str::from_utf8(&exact_capture_settings_json).unwrap())
            .unwrap();

    let config = GstreamerCaptureSourceConfig {
        helper_executable_path: PathBuf::from(env!(
            "CARGO_BIN_EXE_billiards-perception-gstreamer-probe"
        )),
        exact_capture_settings_json,
        capture_settings_max_bytes: MAX_SETTINGS_BYTES,
        http_mjpeg_url: None,
        ready_record_max_bytes: READY_RECORD_MAX_BYTES,
        frame_header_max_bytes: FRAME_HEADER_MAX_BYTES,
        frame_count: Some(frame_count),
        slow_reader_delay_ms: SLOW_READER_DELAY_MS,
        pull_timeout_ms: PULL_TIMEOUT_MS,
        capture_audit_timing: false,
        readiness_timeout: READINESS_TIMEOUT,
        child_exit_timeout: CHILD_EXIT_TIMEOUT,
        child_exit_poll_interval: CHILD_EXIT_POLL_INTERVAL,
    };
    (config, resolved)
}

#[cfg(feature = "camera-gstreamer-source")]
#[test]
fn physical_factory_requires_v3_source_defaults_and_versioned_capture_selector() {
    let valid_selector = format!("gstreamer-device-v1:{}", "a".repeat(64));
    let (mut config, _) = helper_config_for_contract(
        FRAME_COUNT,
        include_str!("../contracts/capture-settings/v3/capture-settings-default.json"),
    );
    replace_request_device_id(&mut config, &valid_selector);
    assert!(GstreamerCaptureSourceFactory::new_physical(config.clone()).is_ok());

    replace_request_device_id(&mut config, "/dev/v4l/by-id/usb-camera");
    assert!(GstreamerCaptureSourceFactory::new_physical(config).is_err());

    let (mut config, _) = helper_config_for_contract(
        FRAME_COUNT,
        include_str!("../contracts/capture-settings/v3/capture-settings-require-controls.json"),
    );
    replace_request_device_id(&mut config, &valid_selector);
    let error = GstreamerCaptureSourceFactory::new_physical(config).unwrap_err();
    assert!(error.to_string().contains("source_defaults"));
}

#[cfg(feature = "camera-gstreamer-source")]
#[test]
fn continuous_helper_configuration_is_physical_only_and_requires_v3_selector() {
    let selector = format!("gstreamer-device-v1:{}", "b".repeat(64));
    let (mut physical_config, _) = helper_config_for_contract(
        FRAME_COUNT,
        include_str!("../contracts/capture-settings/v3/capture-settings-default.json"),
    );
    replace_request_device_id(&mut physical_config, &selector);
    physical_config.frame_count = None;
    assert!(GstreamerCaptureSourceFactory::new_physical(physical_config.clone()).is_ok());

    let (mut synthetic_config, _) = helper_config(FRAME_COUNT);
    synthetic_config.frame_count = None;
    let error = GstreamerCaptureSourceFactory::new(synthetic_config).unwrap_err();
    assert!(error.to_string().contains("finite frame count"));
}

#[cfg(feature = "camera-gstreamer-source")]
fn replace_request_device_id(config: &mut GstreamerCaptureSourceConfig, device_id: &str) {
    let mut request: Value = serde_json::from_slice(&config.exact_capture_settings_json).unwrap();
    request["requested"]["device_id"] = Value::String(device_id.to_owned());
    config.exact_capture_settings_json = serde_json::to_vec(&request).unwrap();
}

fn helper_factory_for_contract(
    frame_count: u64,
    contract_json: &str,
) -> (
    GstreamerCaptureSourceFactory,
    billiards_perception::gstreamer_capture_source::GstreamerCaptureSourceMonitor,
    billiards_perception::capture_settings_wire::ResolvedCaptureSettings,
) {
    let (config, resolved) = helper_config_for_contract(frame_count, contract_json);
    let factory = GstreamerCaptureSourceFactory::new(config).unwrap();
    let monitor = factory.monitor();
    (factory, monitor, resolved)
}

fn runtime() -> CaptureRuntimeCoordinator {
    CaptureRuntimeCoordinator::new(
        CaptureFrameLimit::new(FRAME_LIMIT_BYTES).unwrap(),
        CaptureAggregateFrameLimit::new(AGGREGATE_LIMIT_BYTES).unwrap(),
    )
    .unwrap()
}

fn wait_for_active_report(
    coordinator: &mut CaptureRuntimeCoordinator,
) -> billiards_perception::capture::CaptureSessionReport {
    // Opening the helper includes process scheduling and initializing
    // GStreamer before it can publish verified source metadata. Reuse the
    // helper's configured readiness deadline, with a small observation margin,
    // rather than imposing a shorter integration-test deadline.
    let deadline = Instant::now() + READINESS_TIMEOUT + Duration::from_secs(2);
    loop {
        if let Some(report) = coordinator.active_source_report() {
            return report;
        }
        assert!(
            Instant::now() < deadline,
            "helper metadata was not published"
        );
        std::thread::yield_now();
    }
}
