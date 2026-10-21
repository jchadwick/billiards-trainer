#![cfg(feature = "camera-gstreamer-probe")]

use std::io::{self, Read, Write};
use std::process::{Child, Command, Output, Stdio};
use std::sync::mpsc::{self, Receiver};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use serde_json::Value;
use sha2::{Digest, Sha256};

use billiards_perception::capture_helper_protocol::{
    read_ready_record, FrameBinding, FrameBindingValidator, FrameHeader, FrameReaderError,
    FrameRecordReader,
};

const SESSION_ID: &str = "synthetic-gstreamer-session";
const MAX_READY_RECORD_BYTES: usize = 4_096;
const CAPTURE_SETTINGS_V2: &str =
    include_str!("../contracts/capture-settings/v2/capture-settings-default.json");
const CAPTURE_SETTINGS_V1: &str =
    include_str!("../contracts/capture-settings/v1/capture-settings-default.json");

#[test]
fn synthetic_helper_emits_framed_owned_rgb_with_no_claimed_source_timestamp() {
    let settings = ProbeSettings::new(321, 17, 3, 0);
    let expected_request_digest = request_sha256(&settings.capture_settings_json);
    let helper = HelperProcess::spawn(settings);
    let mut records = Vec::new();
    for _ in 0..3 {
        records.push(helper.read_record().unwrap());
    }
    assert!(helper.read_optional_record().unwrap().is_none());
    let result = helper.finish();
    assert!(
        result.status.success(),
        "probe exit failed: {}",
        String::from_utf8_lossy(&result.stderr)
    );

    for (sequence, (header, pixels)) in records.into_iter().enumerate() {
        assert_eq!(header.record_schema_version, 2);
        assert_eq!(header.capture_timings_ns, None);
        assert_eq!(header.session_id, SESSION_ID);
        assert_eq!(header.frame_id, sequence.to_string());
        assert_eq!(header.configuration_revision, 1);
        assert!(!header.camera_profile_id.is_empty());
        assert_eq!(header.capture_settings_sha256, expected_request_digest);
        assert_eq!((header.width_px, header.height_px), (321, 17));
        assert_eq!(header.pixel_format, "rgb8");
        assert!(header.row_stride_bytes >= 321 * 3);
        assert!(header.unverified_pipeline_pts_ticks.is_some());
        assert_eq!(header.source_timestamp_ticks, None);
        assert!(!pixels.is_empty());
        assert_eq!(pixels.len(), 321 * 17 * 3);
        assert!(header
            .host_read_completed_elapsed_ns
            .as_deref()
            .is_some_and(|value| value.parse::<u128>().is_ok()));
    }
}

#[test]
fn helper_emits_scalar_capture_durations_only_when_requested() {
    let helper =
        HelperProcess::spawn_with_request(64, 48, 2, 0, request_with_output_geometry(64, 48), true);
    let first = helper.read_record().unwrap();
    let second = helper.read_record().unwrap();
    assert!(helper.read_optional_record().unwrap().is_none());
    let result = helper.finish();
    assert!(result.status.success());

    for (header, pixels) in [first, second] {
        assert_eq!(header.record_schema_version, 3);
        assert_eq!(header.host_read_completed_elapsed_ns, None);
        let timing = header
            .capture_timings_ns
            .expect("requested audit timing is present");
        assert!(timing.appsink_pull_wait_ns > 0);
        assert!(timing.rgb_buffer_allocation_ns > 0);
        assert!(timing.rgb_mapping_copy_ns > 0);
        assert_eq!(pixels.len(), 64 * 48 * 3);
    }
}

#[test]
fn helper_accepts_v1_settings_and_uses_the_negotiated_geometry() {
    let request = request_v1_with_geometry(32, 24);
    let expected_request_digest = request_sha256(&request);
    let helper = HelperProcess::spawn_with_request(32, 24, 1, 0, request, false);
    let (header, pixels) = helper.read_record().unwrap();
    let result = helper.finish();

    assert!(result.status.success());
    assert_eq!(header.width_px, 32);
    assert_eq!(header.height_px, 24);
    assert_eq!(header.configuration_revision, 1);
    assert_eq!(header.capture_settings_sha256, expected_request_digest);
    assert_eq!(pixels.len(), 32 * 24 * 3);
}

#[test]
fn slow_helper_reader_observes_fresh_samples_instead_of_a_growing_backlog() {
    let helper = HelperProcess::spawn(ProbeSettings::new(32, 24, 4, 80));
    let mut pts = Vec::new();
    for _ in 0..4 {
        let (header, pixels) = helper.read_record().unwrap();
        assert_eq!(pixels.len(), 32 * 24 * 3);
        pts.push(
            header
                .unverified_pipeline_pts_ticks
                .unwrap()
                .parse::<u64>()
                .unwrap(),
        );
    }
    let result = helper.finish();
    assert!(
        result.status.success(),
        "probe exit failed: {}",
        String::from_utf8_lossy(&result.stderr)
    );

    let expected_frame_period_ns = 1_000_000_000_u64 / 30;
    assert!(
        pts.windows(2)
            .any(|pair| pair[1] - pair[0] > expected_frame_period_ns * 2),
        "slow-reader run should skip stale queued samples; observed PTS values: {pts:?}"
    );
}

#[test]
fn supervising_process_can_force_terminate_and_reap_the_helper() {
    let helper = HelperProcess::spawn(ProbeSettings::new(32, 24, 1_000_000, 0));
    let first = match helper.read_record() {
        Ok(frame) => frame,
        Err(error) => {
            let output = helper.finish();
            panic!(
                "failed to read the first helper frame: {error}; status {}; stderr: {}",
                output.status,
                String::from_utf8_lossy(&output.stderr)
            );
        }
    };
    assert_eq!(first.0.frame_id, "0");
    let result = helper.terminate();
    assert!(!result.status.success());
}

#[test]
fn helper_rejects_invalid_and_oversized_requests_before_emitting_frames() {
    for (capture_settings, maximum_bytes) in [("{}", 2), (CAPTURE_SETTINGS_V2, 8)] {
        let mut child = probe_command(maximum_bytes, 1, 0)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        child
            .stdin
            .take()
            .unwrap()
            .write_all(capture_settings.as_bytes())
            .unwrap();
        let output = child.wait_with_output().unwrap();
        assert!(!output.status.success());
        assert!(output.stdout.is_empty());
    }

    let mut unsupported: Value = serde_json::from_str(CAPTURE_SETTINGS_V2).unwrap();
    unsupported["requested"]["backend"] = serde_json::json!("v4l2");
    let unsupported = serde_json::to_string(&unsupported).unwrap();
    let output = run_probe_without_frames(&unsupported, unsupported.len());
    assert!(!output.status.success());
    assert!(output.stdout.is_empty());
}

struct ProbeSettings {
    width: u32,
    height: u32,
    frames: u64,
    slow_reader_ms: u64,
    capture_settings_json: String,
}

impl ProbeSettings {
    fn new(width: u32, height: u32, frames: u64, slow_reader_ms: u64) -> Self {
        Self {
            width,
            height,
            frames,
            slow_reader_ms,
            capture_settings_json: request_with_output_geometry(width, height),
        }
    }
}

struct HelperProcess {
    child: Option<Child>,
    reader: Option<FrameReader>,
}

impl HelperProcess {
    fn spawn(settings: ProbeSettings) -> Self {
        Self::spawn_with_request(
            settings.width,
            settings.height,
            settings.frames,
            settings.slow_reader_ms,
            settings.capture_settings_json,
            false,
        )
    }

    fn spawn_with_request(
        width: u32,
        height: u32,
        frames: u64,
        slow_reader_ms: u64,
        capture_settings_json: String,
        capture_audit_timing: bool,
    ) -> Self {
        let max_frame_bytes = width as usize * height as usize * 3;
        let settings_bytes = capture_settings_json.as_bytes();
        let resolved = billiards_perception::capture_settings_wire::parse_capture_settings_json(
            &capture_settings_json,
        )
        .unwrap();
        let geometry = resolved
            .expected
            .output_frame_geometry()
            .unwrap_or_else(|| {
                let profile = resolved.expected.expected_negotiated_profile();
                billiards_perception::capture::CaptureOutputFrameGeometry::new(
                    profile.width_px(),
                    profile.height_px(),
                )
                .unwrap()
            });
        assert_eq!((geometry.width_px(), geometry.height_px()), (width, height));
        let binding = FrameBindingOwned {
            session_id: SESSION_ID.to_owned(),
            configuration_revision: resolved.configuration_revision,
            camera_profile_id: resolved.expected.camera_profile_id().to_owned(),
            capture_settings_sha256: resolved.request_sha256.clone(),
            width_px: width,
            height_px: height,
            max_payload_bytes: resolved.expected.frame_limit().maximum_rgb8_bytes(),
        };
        let mut child = probe_command_with_timing(
            settings_bytes.len(),
            frames,
            slow_reader_ms,
            capture_audit_timing,
        )
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
        child
            .stdin
            .take()
            .unwrap()
            .write_all(settings_bytes)
            .unwrap();
        let mut output = child.stdout.take().unwrap();
        let ready = match read_ready_record(&mut output, MAX_READY_RECORD_BYTES) {
            Ok(ready) => ready,
            Err(error) => {
                drop(output);
                let failed = child.wait_with_output().unwrap();
                panic!(
                    "helper did not send valid readiness metadata: {error}; status {}; stderr: {}",
                    failed.status,
                    String::from_utf8_lossy(&failed.stderr)
                );
            }
        };
        let expected_rate = resolved.expected.expected_negotiated_profile().frame_rate();
        assert_eq!(ready.record_schema_version, 1);
        assert_eq!(ready.session_id, SESSION_ID);
        assert_eq!(
            ready.configuration_revision,
            resolved.configuration_revision
        );
        assert_eq!(
            ready.camera_profile_id,
            resolved.expected.camera_profile_id()
        );
        assert_eq!(ready.capture_settings_sha256, resolved.request_sha256);
        assert_eq!((ready.width_px, ready.height_px), (width, height));
        assert_eq!(ready.frame_rate_numerator, expected_rate.frames_numerator());
        assert_eq!(
            ready.frame_rate_seconds_denominator,
            expected_rate.seconds_denominator()
        );
        assert_eq!(
            ready.pixel_format,
            resolved
                .expected
                .expected_negotiated_profile()
                .pixel_format()
                .identifier()
        );
        let reader = FrameReader::new(output, 4_096, max_frame_bytes, binding);
        Self {
            child: Some(child),
            reader: Some(reader),
        }
    }

    fn read_record(&self) -> io::Result<FrameRecord> {
        self.read_optional_record()?
            .ok_or_else(|| io::Error::new(io::ErrorKind::UnexpectedEof, "expected frame record"))
    }

    fn read_optional_record(&self) -> io::Result<Option<FrameRecord>> {
        self.reader
            .as_ref()
            .expect("helper frame reader is active")
            .receive()
    }

    fn finish(mut self) -> Output {
        if let Some(reader) = self.reader.take() {
            reader.stop();
        }
        self.child
            .take()
            .expect("helper child is active")
            .wait_with_output()
            .unwrap()
    }

    fn terminate(mut self) -> Output {
        let mut child = self.child.take().expect("helper child is active");
        child.kill().unwrap();
        let output = child.wait_with_output().unwrap();
        if let Some(reader) = self.reader.take() {
            reader.stop();
        }
        output
    }
}

fn probe_command(capture_settings_max_bytes: usize, frames: u64, slow_reader_ms: u64) -> Command {
    probe_command_with_timing(capture_settings_max_bytes, frames, slow_reader_ms, false)
}

fn probe_command_with_timing(
    capture_settings_max_bytes: usize,
    frames: u64,
    slow_reader_ms: u64,
    capture_audit_timing: bool,
) -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_billiards-perception-gstreamer-probe"));
    command.args([
        "--session-id",
        SESSION_ID,
        "--capture-settings-max-bytes",
        &capture_settings_max_bytes.to_string(),
        "--ready-record-max-bytes",
        &MAX_READY_RECORD_BYTES.to_string(),
        "--frames",
        &frames.to_string(),
        "--slow-reader-ms",
        &slow_reader_ms.to_string(),
        "--pull-timeout-ms",
        "5000",
    ]);
    if capture_audit_timing {
        command.arg("--capture-audit-timing");
    }
    command
}

fn request_with_output_geometry(width: u32, height: u32) -> String {
    let mut request: Value = serde_json::from_str(CAPTURE_SETTINGS_V2).unwrap();
    request["requested"]["output_frame_geometry"]["width_px"] = serde_json::json!(width);
    request["requested"]["output_frame_geometry"]["height_px"] = serde_json::json!(height);
    serde_json::to_string(&request).unwrap()
}

fn request_v1_with_geometry(width: u32, height: u32) -> String {
    let mut request: Value = serde_json::from_str(CAPTURE_SETTINGS_V1).unwrap();
    for profile in ["requested_profile", "expected_negotiated_profile"] {
        request["requested"][profile]["width_px"] = serde_json::json!(width);
        request["requested"][profile]["height_px"] = serde_json::json!(height);
    }
    serde_json::to_string(&request).unwrap()
}

fn run_probe_without_frames(capture_settings: &str, maximum_bytes: usize) -> Output {
    let mut child = probe_command(maximum_bytes, 1, 0)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(capture_settings.as_bytes())
        .unwrap();
    child.wait_with_output().unwrap()
}

fn request_sha256(request: &str) -> String {
    format!("{:x}", Sha256::digest(request.as_bytes()))
}

impl Drop for HelperProcess {
    fn drop(&mut self) {
        if let Some(child) = self.child.as_mut() {
            let _ = child.kill();
            let _ = child.wait();
        }
        if let Some(reader) = self.reader.take() {
            reader.stop();
        }
    }
}

type FrameRecord = (FrameHeader, Vec<u8>);

#[derive(Clone)]
struct FrameBindingOwned {
    session_id: String,
    configuration_revision: u64,
    camera_profile_id: String,
    capture_settings_sha256: String,
    width_px: u32,
    height_px: u32,
    max_payload_bytes: usize,
}

impl FrameBindingOwned {
    fn validator(&self) -> FrameBindingValidator {
        FrameBindingValidator::new(FrameBinding {
            session_id: self.session_id.clone(),
            configuration_revision: self.configuration_revision,
            camera_profile_id: self.camera_profile_id.clone(),
            capture_settings_sha256: self.capture_settings_sha256.clone(),
            width_px: self.width_px,
            height_px: self.height_px,
            max_payload_bytes: self.max_payload_bytes,
        })
        .unwrap()
    }
}

struct FrameReader {
    receiver: Receiver<io::Result<Option<FrameRecord>>>,
    thread: JoinHandle<()>,
}

impl FrameReader {
    fn new(
        mut output: impl Read + Send + 'static,
        max_header_bytes: usize,
        max_pixel_bytes: usize,
        binding: FrameBindingOwned,
    ) -> Self {
        let (sender, receiver) = mpsc::sync_channel(1);
        let thread = thread::spawn(move || {
            let mut validator = binding.validator();
            loop {
                let result = read_optional_record(&mut output, max_header_bytes, max_pixel_bytes)
                    .and_then(|record| {
                        if let Some((header, pixels)) = &record {
                            validator.validate(header, pixels.len()).map_err(|error| {
                                io::Error::new(io::ErrorKind::InvalidData, error)
                            })?;
                        }
                        Ok(record)
                    });
                let stop = !matches!(result, Ok(Some(_)));
                if sender.send(result).is_err() || stop {
                    break;
                }
            }
        });
        Self { receiver, thread }
    }

    fn receive(&self) -> io::Result<Option<FrameRecord>> {
        match self.receiver.recv_timeout(Duration::from_secs(5)) {
            Ok(result) => result,
            Err(mpsc::RecvTimeoutError::Timeout) => Err(io::Error::new(
                io::ErrorKind::TimedOut,
                "timed out waiting for helper frame",
            )),
            Err(mpsc::RecvTimeoutError::Disconnected) => Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "helper frame reader stopped unexpectedly",
            )),
        }
    }

    fn stop(self) {
        drop(self.receiver);
        self.thread
            .join()
            .expect("helper frame reader thread panicked");
    }
}

fn read_optional_record(
    reader: &mut impl Read,
    max_header_bytes: usize,
    max_pixel_bytes: usize,
) -> io::Result<Option<(FrameHeader, Vec<u8>)>> {
    let mut decoder = FrameRecordReader::new(reader, max_header_bytes, max_pixel_bytes)
        .map_err(protocol_error)?;
    decoder
        .read_next()
        .map(|record| record.map(|record| (record.header, record.payload)))
        .map_err(protocol_error)
}

fn protocol_error(error: FrameReaderError) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, error)
}
