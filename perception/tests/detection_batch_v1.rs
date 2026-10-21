use billiards_perception::detection_batch::{
    decode_detection_batch_v1, encode_detection_batch_v1, BatchOutcome, DetectionBatchV1,
};
use serde_json::Value;

const SUCCESS_FIXTURE: &str =
    include_str!("../contracts/detection-batch/v1/detection-batch-success.json");
const EMPTY_FIXTURE: &str =
    include_str!("../contracts/detection-batch/v1/detection-batch-empty-success.json");
const FAILURE_FIXTURE: &str =
    include_str!("../contracts/detection-batch/v1/detection-batch-failure.json");
const RESTART_FIXTURE: &str =
    include_str!("../contracts/detection-batch/v1/detection-batch-restarted-session.json");

fn parse_fixture(source: &str) -> DetectionBatchV1 {
    decode_detection_batch_v1(source.as_bytes()).expect("fixture must satisfy the contract")
}

#[test]
fn shared_fixtures_decode_validate_and_round_trip() {
    for fixture in [
        SUCCESS_FIXTURE,
        EMPTY_FIXTURE,
        FAILURE_FIXTURE,
        RESTART_FIXTURE,
    ] {
        let batch = parse_fixture(fixture);
        let encoded = encode_detection_batch_v1(&batch).expect("batch must serialize");
        let serialized: Value = serde_json::from_slice(&encoded).unwrap();
        let original: Value = serde_json::from_str(fixture).expect("fixture JSON is valid");
        assert_eq!(serialized, original);
    }
}

#[test]
fn successful_empty_batch_is_distinct_from_inference_failure() {
    let empty = parse_fixture(EMPTY_FIXTURE);
    let failed = parse_fixture(FAILURE_FIXTURE);

    assert_eq!(empty.outcome, BatchOutcome::Success);
    assert!(empty.detections.is_empty());
    assert_eq!(failed.outcome, BatchOutcome::Failed);
    assert!(failed.detections.is_empty());
    assert_eq!(failed.reason.as_deref(), Some("inference_timeout"));
}

#[test]
fn repeated_delivery_preserves_the_batch_and_detection_identity() {
    let batch = parse_fixture(SUCCESS_FIXTURE);
    let first = encode_detection_batch_v1(&batch).unwrap();
    let second = encode_detection_batch_v1(&batch).unwrap();

    assert_eq!(first, second);
    assert_eq!(batch.detections[0].detection_index, 0);
    assert_eq!(batch.source.frame_id, "frame-1042");
}

#[test]
fn restarted_sessions_can_reuse_frame_and_detection_indices_without_collision() {
    let previous = parse_fixture(SUCCESS_FIXTURE);
    let restarted = parse_fixture(RESTART_FIXTURE);

    assert_eq!(previous.source.frame_id, restarted.source.frame_id);
    assert_eq!(
        previous.detections[0].detection_index,
        restarted.detections[0].detection_index
    );
    assert_ne!(previous.source.session_id, restarted.source.session_id);
    assert_ne!(previous.engine_instance_id, restarted.engine_instance_id);
}

#[test]
fn source_time_may_be_unavailable_for_recorded_still_images() {
    let batch = parse_fixture(EMPTY_FIXTURE);
    assert!(batch.source.timestamp.is_none());
}

#[test]
fn missing_nullable_properties_and_unknown_properties_are_rejected() {
    let mut missing: Value = serde_json::from_str(SUCCESS_FIXTURE).unwrap();
    missing["source"]
        .as_object_mut()
        .unwrap()
        .remove("timestamp");
    let error = serde_json::from_value::<DetectionBatchV1>(missing)
        .expect_err("required nullable source time must not be omitted");
    assert!(error.to_string().contains("timestamp"));

    let mut unknown: Value = serde_json::from_str(SUCCESS_FIXTURE).unwrap();
    unknown
        .as_object_mut()
        .unwrap()
        .insert("unexpected".into(), Value::Bool(true));
    let error = serde_json::from_value::<DetectionBatchV1>(unknown)
        .expect_err("unknown fields must fail closed");
    assert!(error.to_string().contains("unknown field"));
}

#[test]
fn unsupported_schema_versions_are_rejected() {
    let mut fixture: Value = serde_json::from_str(SUCCESS_FIXTURE).unwrap();
    fixture["schema_version"] = Value::from(2);
    let bytes = serde_json::to_vec(&fixture).unwrap();

    let error = decode_detection_batch_v1(&bytes).expect_err("only v1 is supported");
    assert!(error.to_string().contains("expected 1"));
}

#[test]
fn semantic_validation_rejects_invalid_spatial_and_class_data() {
    let mut outside = parse_fixture(SUCCESS_FIXTURE);
    outside.detections[0].bbox_image_px.right = 4000.0;
    let error = outside
        .validate()
        .expect_err("box must fit the source frame");
    assert!(error.path.contains("bbox_image_px"));

    let mut class_mismatch = parse_fixture(SUCCESS_FIXTURE);
    class_mismatch.detections[0].class_name = "cue".into();
    let error = class_mismatch
        .validate()
        .expect_err("class name must match model class map");
    assert!(error.path.contains("class_name"));
}

#[test]
fn batch_indices_and_inference_clock_order_are_validated() {
    let mut duplicate_index = parse_fixture(SUCCESS_FIXTURE);
    let mut duplicate = duplicate_index.detections[0].clone();
    duplicate_index.detections.push(duplicate.clone());
    duplicate_index.settings.max_detections = 300;
    duplicate.detection_index = 0;
    duplicate_index.detections[1] = duplicate;
    let error = duplicate_index
        .validate()
        .expect_err("indices must equal their array position");
    assert!(error.path.ends_with("detection_index"));

    let mut out_of_order = parse_fixture(SUCCESS_FIXTURE);
    out_of_order.inference_completed_at.ticks = "1".into();
    let error = out_of_order
        .validate()
        .expect_err("completion must follow inference start");
    assert!(error.path.contains("inference_completed_at"));
}

#[test]
fn failed_batch_cannot_carry_detections() {
    let mut batch = parse_fixture(SUCCESS_FIXTURE);
    batch.outcome = BatchOutcome::Failed;
    batch.reason = Some("inference_timeout".into());

    let error = batch
        .validate()
        .expect_err("failure is not a partial successful detection list");
    assert!(error.path.contains("detections"));
}
