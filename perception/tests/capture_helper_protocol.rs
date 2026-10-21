use std::io::Cursor;

use billiards_perception::capture::CaptureHelperTimingNs;
use billiards_perception::capture_helper_protocol::{
    FrameBinding, FrameBindingError, FrameBindingValidator, FrameHeader, FrameReaderError,
    FrameRecordReader,
};

const MAX_HEADER_BYTES: usize = 4096;
const MAX_PAYLOAD_BYTES: usize = 1024;

fn header() -> FrameHeader {
    FrameHeader {
        record_schema_version: 2,
        session_id: "session-123".to_owned(),
        frame_id: "0".to_owned(),
        configuration_revision: 8,
        camera_profile_id: "camera-profile-a".to_owned(),
        capture_settings_sha256: "a".repeat(64),
        width_px: 2,
        height_px: 2,
        pixel_format: "rgb8".to_owned(),
        row_stride_bytes: 6,
        unverified_pipeline_pts_ticks: Some("789".to_owned()),
        source_timestamp_ticks: None,
        host_read_completed_elapsed_ns: Some("123456".to_owned()),
        capture_timings_ns: None,
    }
}

fn record_bytes(header_json: &[u8], payload: &[u8]) -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&(header_json.len() as u32).to_be_bytes());
    bytes.extend_from_slice(header_json);
    bytes.extend_from_slice(&(payload.len() as u32).to_be_bytes());
    bytes.extend_from_slice(payload);
    bytes
}

fn encode_record(frame_header: &FrameHeader, payload: &[u8]) -> Vec<u8> {
    record_bytes(&serde_json::to_vec(frame_header).unwrap(), payload)
}

fn reader(bytes: Vec<u8>) -> FrameRecordReader<Cursor<Vec<u8>>> {
    FrameRecordReader::new(Cursor::new(bytes), MAX_HEADER_BYTES, MAX_PAYLOAD_BYTES).unwrap()
}

fn binding() -> FrameBinding {
    FrameBinding {
        session_id: "session-123".to_owned(),
        configuration_revision: 8,
        camera_profile_id: "camera-profile-a".to_owned(),
        capture_settings_sha256: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
            .to_owned(),
        width_px: 2,
        height_px: 2,
        max_payload_bytes: MAX_PAYLOAD_BYTES,
    }
}

fn validate_one(frame_header: &FrameHeader, payload_len: usize) -> Result<(), FrameBindingError> {
    FrameBindingValidator::new(binding())?.validate(frame_header, payload_len)
}

#[test]
fn reads_a_valid_record_and_then_reports_clean_eof() {
    let payload = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
    let mut input = reader(encode_record(&header(), &payload));

    let record = input.read_next().unwrap().unwrap();
    assert_eq!(record.header, header());
    assert_eq!(record.payload, payload);
    assert!(input.read_next().unwrap().is_none());
}

#[test]
fn constructor_requires_positive_limits() {
    assert!(matches!(
        FrameRecordReader::new(Cursor::new([]), 0, MAX_PAYLOAD_BYTES),
        Err(FrameReaderError::InvalidLimit("header byte"))
    ));
    assert!(matches!(
        FrameRecordReader::new(Cursor::new([]), MAX_HEADER_BYTES, 0),
        Err(FrameReaderError::InvalidLimit("payload byte"))
    ));
}

#[test]
fn clean_eof_is_distinct_from_every_truncated_prefix_length() {
    assert!(reader(Vec::new()).read_next().unwrap().is_none());

    for count in 1..4 {
        let error = reader(vec![0; count]).read_next().unwrap_err();
        assert!(error.to_string().contains("truncated header length prefix"));
    }
}

#[test]
fn rejects_zero_and_oversized_header_lengths_before_payload_allocation() {
    let mut zero = Vec::from(0_u32.to_be_bytes());
    zero.extend_from_slice(b"{}");
    let error = reader(zero).read_next().unwrap_err();
    assert!(error.to_string().contains("header length must be positive"));

    let mut oversized = Vec::from(20_u32.to_be_bytes());
    oversized.extend_from_slice(&[0; 20]);
    let error = FrameRecordReader::new(Cursor::new(oversized), 19, MAX_PAYLOAD_BYTES)
        .unwrap()
        .read_next()
        .unwrap_err();
    assert!(error
        .to_string()
        .contains("header length exceeds configured limit"));
}

#[test]
fn rejects_truncated_header_and_malformed_or_unknown_json() {
    let error = reader(record_bytes(b"{not-json", &[1]))
        .read_next()
        .unwrap_err();
    assert!(error.to_string().contains("invalid frame JSON header"));

    let error = reader(record_bytes(b"{}", &[])).read_next().unwrap_err();
    assert!(error.to_string().contains("invalid frame JSON header"));

    let json = serde_json::to_value(header()).unwrap();
    let mut missing_required_null = json.clone();
    missing_required_null
        .as_object_mut()
        .unwrap()
        .remove("source_timestamp_ticks");
    let error = reader(record_bytes(
        &serde_json::to_vec(&missing_required_null).unwrap(),
        &[1],
    ))
    .read_next()
    .unwrap_err();
    assert!(error.to_string().contains("invalid frame JSON header"));

    let mut with_unknown = json;
    with_unknown
        .as_object_mut()
        .unwrap()
        .insert("unexpected".to_owned(), serde_json::json!(true));
    let error = reader(record_bytes(
        &serde_json::to_vec(&with_unknown).unwrap(),
        &[1],
    ))
    .read_next()
    .unwrap_err();
    assert!(error.to_string().contains("invalid frame JSON header"));

    let full_header = serde_json::to_vec(&header()).unwrap();
    let mut truncated = Vec::new();
    truncated.extend_from_slice(&(full_header.len() as u32).to_be_bytes());
    truncated.extend_from_slice(&full_header[..full_header.len() - 1]);
    let error = reader(truncated).read_next().unwrap_err();
    assert!(error.to_string().contains("truncated JSON header"));
}

#[test]
fn rejects_negative_helper_timing_values() {
    let mut timed = header();
    timed.record_schema_version = 3;
    timed.host_read_completed_elapsed_ns = None;
    timed.capture_timings_ns = Some(CaptureHelperTimingNs {
        appsink_pull_wait_ns: 1,
        rgb_buffer_allocation_ns: 2,
        rgb_mapping_copy_ns: 3,
    });
    let mut json = serde_json::to_value(timed).unwrap();
    json["capture_timings_ns"]["appsink_pull_wait_ns"] = serde_json::json!(-1);
    let error = reader(record_bytes(&serde_json::to_vec(&json).unwrap(), &[1; 12]))
        .read_next()
        .unwrap_err();
    assert!(error.to_string().contains("invalid frame JSON header"));
}

#[test]
fn rejects_truncated_zero_and_oversized_payload_lengths() {
    let json = serde_json::to_vec(&header()).unwrap();

    for prefix_bytes in 1..4 {
        let mut truncated_prefix = Vec::new();
        truncated_prefix.extend_from_slice(&(json.len() as u32).to_be_bytes());
        truncated_prefix.extend_from_slice(&json);
        truncated_prefix.extend_from_slice(&[0; 3][..prefix_bytes]);
        let error = reader(truncated_prefix).read_next().unwrap_err();
        assert!(error
            .to_string()
            .contains("truncated payload length prefix"));
    }

    let error = reader(record_bytes(&json, &[])).read_next().unwrap_err();
    assert!(error
        .to_string()
        .contains("payload length must be positive"));

    let mut oversized = Vec::new();
    oversized.extend_from_slice(&(json.len() as u32).to_be_bytes());
    oversized.extend_from_slice(&json);
    oversized.extend_from_slice(&((MAX_PAYLOAD_BYTES as u32) + 1).to_be_bytes());
    let error = reader(oversized).read_next().unwrap_err();
    assert!(error
        .to_string()
        .contains("payload length exceeds configured limit"));

    let mut truncated_payload = Vec::new();
    truncated_payload.extend_from_slice(&(json.len() as u32).to_be_bytes());
    truncated_payload.extend_from_slice(&json);
    truncated_payload.extend_from_slice(&4_u32.to_be_bytes());
    truncated_payload.extend_from_slice(&[1, 2, 3]);
    let error = reader(truncated_payload).read_next().unwrap_err();
    assert!(error.to_string().contains("truncated RGB payload"));
}

#[test]
fn payload_allocator_is_called_after_checks_and_must_return_exact_size() {
    let bytes = encode_record(&header(), &[1; 12]);
    let mut input = reader(bytes.clone());
    let mut allocated_for = None;
    let record = input
        .read_next_with_buffer(|seen_header, payload_len| {
            assert_eq!(seen_header.session_id, "session-123");
            allocated_for = Some(payload_len);
            Ok::<_, &'static str>(Box::<[u8]>::from(vec![0; payload_len]))
        })
        .unwrap()
        .unwrap();
    assert_eq!(allocated_for, Some(12));
    assert_eq!(record.payload.as_ref(), &[1; 12]);

    let mut oversized =
        FrameRecordReader::new(Cursor::new(bytes.clone()), MAX_HEADER_BYTES, 11).unwrap();
    let mut allocator_called = false;
    let error = oversized
        .read_next_with_buffer(|_, _| {
            allocator_called = true;
            Ok::<_, &'static str>(Box::<[u8]>::from(vec![0; 12]))
        })
        .unwrap_err();
    assert!(!allocator_called);
    assert!(error
        .to_string()
        .contains("payload length exceeds configured limit"));

    let mut wrong_size = reader(bytes);
    let error = wrong_size
        .read_next_with_buffer(|_, _| Ok::<_, &'static str>(Box::<[u8]>::from(vec![0; 11])))
        .unwrap_err();
    assert!(matches!(
        error,
        FrameReaderError::BufferSizeMismatch {
            expected: 12,
            actual: 11
        }
    ));

    let mut allocation_fails = reader(encode_record(&header(), &[1; 12]));
    let error = allocation_fails
        .read_next_with_buffer(|_, _| Err::<Box<[u8]>, _>("injected allocator failure"))
        .unwrap_err();
    assert!(error.to_string().contains("injected allocator failure"));
}

#[test]
fn binding_accepts_expected_identity_and_advances_contiguous_sequence() {
    let mut validator = FrameBindingValidator::new(binding()).unwrap();
    let first = header();
    validator.validate(&first, 12).unwrap();
    let mut second = header();
    second.frame_id = "1".to_owned();
    validator.validate(&second, 12).unwrap();

    let mut timed = header();
    timed.record_schema_version = 3;
    timed.host_read_completed_elapsed_ns = None;
    timed.capture_timings_ns = Some(CaptureHelperTimingNs {
        appsink_pull_wait_ns: 100,
        rgb_buffer_allocation_ns: 200,
        rgb_mapping_copy_ns: 300,
    });
    let mut timed_validator = FrameBindingValidator::new(binding()).unwrap();
    timed_validator.validate(&timed, 12).unwrap();
}

#[test]
fn binding_rejects_wrong_identity_revision_digest_geometry_format_and_stride() {
    let mutations: Vec<(FrameHeader, FrameBindingError)> = vec![
        (
            {
                let mut value = header();
                value.record_schema_version = 1;
                value
            },
            FrameBindingError::SchemaVersion { actual: 1 },
        ),
        (
            {
                let mut value = header();
                value.record_schema_version = 3;
                value
            },
            FrameBindingError::CaptureTimingSchemaMismatch,
        ),
        (
            {
                let mut value = header();
                value.session_id = "other-session".to_owned();
                value
            },
            FrameBindingError::SessionIdentity,
        ),
        (
            {
                let mut value = header();
                value.configuration_revision += 1;
                value
            },
            FrameBindingError::ConfigurationRevision { expected: 8 },
        ),
        (
            {
                let mut value = header();
                value.camera_profile_id = "other-camera".to_owned();
                value
            },
            FrameBindingError::CameraProfileIdentity,
        ),
        (
            {
                let mut value = header();
                value.capture_settings_sha256 = "b".repeat(64);
                value
            },
            FrameBindingError::CaptureSettingsDigest,
        ),
        (
            {
                let mut value = header();
                value.pixel_format = "bgr8".to_owned();
                value
            },
            FrameBindingError::PixelFormat,
        ),
        (
            {
                let mut value = header();
                value.width_px = 3;
                value
            },
            FrameBindingError::Geometry {
                expected_width: 2,
                expected_height: 2,
            },
        ),
        (
            {
                let mut value = header();
                value.row_stride_bytes = 5;
                value
            },
            FrameBindingError::RowStride { minimum: 6 },
        ),
    ];

    for (bad_header, expected_error) in mutations {
        assert_eq!(validate_one(&bad_header, 12), Err(expected_error));
    }
}

#[test]
fn binding_rejects_wrong_payload_length_or_limit_without_advancing_sequence() {
    let mut validator = FrameBindingValidator::new(binding()).unwrap();
    let frame = header();
    assert_eq!(
        validator.validate(&frame, 11),
        Err(FrameBindingError::PayloadLength {
            expected: 12,
            actual: 11
        })
    );
    validator.validate(&frame, 12).unwrap();

    let mut small_limit = binding();
    small_limit.max_payload_bytes = 11;
    let mut validator = FrameBindingValidator::new(small_limit).unwrap();
    assert_eq!(
        validator.validate(&frame, 12),
        Err(FrameBindingError::ExpectedPayloadExceedsLimit {
            expected: 12,
            limit: 11
        })
    );

    let mut validator = FrameBindingValidator::new(binding()).unwrap();
    assert_eq!(
        validator.validate(&frame, MAX_PAYLOAD_BYTES + 1),
        Err(FrameBindingError::PayloadExceedsLimit {
            actual: MAX_PAYLOAD_BYTES + 1,
            limit: MAX_PAYLOAD_BYTES
        })
    );
}

#[test]
fn binding_uses_checked_arithmetic_for_packed_geometry() {
    let mut large_binding = binding();
    large_binding.width_px = 4_000_000_000;
    large_binding.height_px = 4_000_000_000;
    large_binding.max_payload_bytes = usize::MAX;
    let mut large_header = header();
    large_header.width_px = large_binding.width_px;
    large_header.height_px = large_binding.height_px;
    large_header.row_stride_bytes = 12_000_000_000;
    let mut validator = FrameBindingValidator::new(large_binding).unwrap();
    assert_eq!(
        validator.validate(&large_header, 1),
        Err(FrameBindingError::GeometryOverflow)
    );
}

#[test]
fn binding_rejects_sequence_gaps_repeats_and_noncanonical_decimal_ids() {
    let mut validator = FrameBindingValidator::new(binding()).unwrap();
    let mut frame = header();
    frame.frame_id = "1".to_owned();
    assert_eq!(
        validator.validate(&frame, 12),
        Err(FrameBindingError::FrameSequence { expected: 0 })
    );

    frame.frame_id = "00".to_owned();
    assert_eq!(
        validator.validate(&frame, 12),
        Err(FrameBindingError::FrameSequence { expected: 0 })
    );

    frame.frame_id = "0".to_owned();
    validator.validate(&frame, 12).unwrap();
    assert_eq!(
        validator.validate(&frame, 12),
        Err(FrameBindingError::FrameSequence { expected: 1 })
    );
}

#[test]
fn binding_rejects_zero_expected_geometry_and_zero_payload_limit() {
    let mut invalid = binding();
    invalid.width_px = 0;
    assert!(matches!(
        FrameBindingValidator::new(invalid),
        Err(FrameBindingError::InvalidExpectedGeometry)
    ));

    let mut invalid = binding();
    invalid.max_payload_bytes = 0;
    assert!(matches!(
        FrameBindingValidator::new(invalid),
        Err(FrameBindingError::InvalidPayloadLimit)
    ));
}
