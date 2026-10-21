use billiards_perception::{
    decode_scene_snapshot_v1, encode_scene_snapshot_v1, BallStatus, CalibrationState, Outcome,
    SceneSnapshotV1,
};
use serde_json::Value;

const SUCCESS_FIXTURE: &str = include_str!("../contracts/v1/scene-success.json");
const FAILURE_FIXTURE: &str = include_str!("../contracts/v1/scene-failure.json");
const IDENTITY_FIXTURE: &str = include_str!("../contracts/v1/scene-identity-update.json");
const RESTART_FIXTURE: &str = include_str!("../contracts/v1/scene-restarted-session.json");
const RUST_TRACKER_FIXTURE: &str = include_str!("../contracts/v1/scene-rust-tracked-success.json");

fn parse_fixture(source: &str) -> SceneSnapshotV1 {
    decode_scene_snapshot_v1(source.as_bytes()).expect("fixture must satisfy the contract")
}

#[test]
fn python_contract_fixtures_deserialize_validate_and_round_trip() {
    for fixture in [
        SUCCESS_FIXTURE,
        FAILURE_FIXTURE,
        IDENTITY_FIXTURE,
        RESTART_FIXTURE,
        RUST_TRACKER_FIXTURE,
    ] {
        let snapshot = parse_fixture(fixture);
        let encoded = encode_scene_snapshot_v1(&snapshot).expect("snapshot must serialize");
        let serialized: Value = serde_json::from_slice(&encoded).unwrap();
        let original: Value = serde_json::from_str(fixture).expect("fixture JSON is valid");
        assert_eq!(serialized, original);
    }
}

#[test]
fn successful_empty_scene_is_distinct_from_failed_inference() {
    let mut empty_success: Value = serde_json::from_str(SUCCESS_FIXTURE).unwrap();
    empty_success["balls"] = Value::Array(Vec::new());
    let success_bytes = serde_json::to_vec(&empty_success).unwrap();
    let success = decode_scene_snapshot_v1(&success_bytes).unwrap();
    let failure = parse_fixture(FAILURE_FIXTURE);

    assert_eq!(success.outcome, Outcome::Success);
    assert!(success.balls.is_empty());
    assert_eq!(failure.outcome, Outcome::Failed);
    assert!(failure.balls.is_empty());
}

#[test]
fn identity_annotation_keeps_the_original_ball_observation() {
    let observed = parse_fixture(SUCCESS_FIXTURE);
    let annotated = parse_fixture(IDENTITY_FIXTURE);

    assert_eq!(observed.source, annotated.source);
    assert_eq!(
        observed.balls[0].last_observation,
        annotated.balls[0].last_observation
    );
    assert_eq!(
        observed.balls[0].measurement,
        annotated.balls[0].measurement
    );
    assert_eq!(annotated.balls[0].identity.number, Some(8));
}

#[test]
fn encoder_refuses_semantically_invalid_snapshot_values() {
    let mut snapshot = parse_fixture(SUCCESS_FIXTURE);
    snapshot.balls[0]
        .measurement
        .as_mut()
        .unwrap()
        .image_center_px
        .x = -1.0;

    let error = encode_scene_snapshot_v1(&snapshot).expect_err("invalid data must not be sent");
    assert!(error
        .to_string()
        .contains("inside the declared image dimensions"));
}

#[test]
fn absent_nullable_properties_are_not_treated_as_explicit_nulls() {
    let mut fixture: Value = serde_json::from_str(SUCCESS_FIXTURE).unwrap();
    fixture.as_object_mut().unwrap().remove("reason");

    let error = serde_json::from_value::<SceneSnapshotV1>(fixture)
        .expect_err("the v1 contract requires nullable fields to be present");
    assert!(error.to_string().contains("reason"));
}

#[test]
fn unknown_properties_are_rejected_instead_of_silently_ignored() {
    let mut fixture: Value = serde_json::from_str(SUCCESS_FIXTURE).unwrap();
    fixture
        .as_object_mut()
        .unwrap()
        .insert("unexpected".to_string(), Value::Bool(true));

    let error = serde_json::from_value::<SceneSnapshotV1>(fixture)
        .expect_err("unknown fields must fail closed");
    assert!(error.to_string().contains("unknown field"));
}

#[test]
fn unsupported_schema_versions_are_rejected() {
    let mut fixture: Value = serde_json::from_str(SUCCESS_FIXTURE).unwrap();
    fixture["schema_version"] = Value::from(2);
    let bytes = serde_json::to_vec(&fixture).unwrap();

    let error = decode_scene_snapshot_v1(&bytes).expect_err("unknown schema must fail closed");
    assert!(error.to_string().contains("expected 1"));
}

#[test]
fn source_observed_balls_must_refer_to_the_snapshot_frame() {
    let mut snapshot = parse_fixture(SUCCESS_FIXTURE);
    snapshot.balls[0].last_observation.frame_id = "previous-frame".to_string();

    let error = snapshot
        .validate()
        .expect_err("observation identity must match");
    assert!(error.path.contains("last_observation.frame_id"));
}

#[test]
fn image_locations_are_bounded_and_match_declared_image_space() {
    let mut snapshot = parse_fixture(SUCCESS_FIXTURE);
    let measurement = snapshot.balls[0].measurement.as_mut().unwrap();
    measurement.image_center_px.x = f64::NAN;

    let error = snapshot
        .validate()
        .expect_err("non-finite positions are invalid");
    assert!(error.path.contains("image_center_px"));

    let mut snapshot = parse_fixture(SUCCESS_FIXTURE);
    let measurement = snapshot.balls[0].measurement.as_mut().unwrap();
    measurement.image_center_px.x = f64::from(measurement.image_width_px);
    let error = snapshot
        .validate()
        .expect_err("centers cannot be beyond image width");
    assert!(error
        .message
        .contains("inside the declared image dimensions"));

    let mut snapshot = parse_fixture(SUCCESS_FIXTURE);
    let measurement = snapshot.balls[0].measurement.as_mut().unwrap();
    measurement.image_width_px -= 1;

    let error = snapshot
        .validate()
        .expect_err("raw dimensions must match source");
    assert!(error.message.contains("match the source frame"));
}

#[test]
fn table_coordinates_require_valid_calibration() {
    let mut snapshot = parse_fixture(SUCCESS_FIXTURE);
    snapshot.revisions.calibration_state = CalibrationState::Invalid;

    let error = snapshot
        .validate()
        .expect_err("invalid calibration cannot produce table locations");
    assert!(error.message.contains("require valid calibration"));
}

#[test]
fn track_ids_must_be_unique_inside_a_complete_scene() {
    let mut snapshot = parse_fixture(SUCCESS_FIXTURE);
    let mut duplicate = snapshot.balls[0].clone();
    duplicate.status = BallStatus::Tracked;
    snapshot.balls.push(duplicate);

    let error = snapshot
        .validate()
        .expect_err("duplicate tracks are ambiguous");
    assert!(error.message.contains("unique within one snapshot"));
}

#[test]
fn numbered_identity_requires_bounded_confidence_and_evidence() {
    let mut snapshot = parse_fixture(IDENTITY_FIXTURE);
    snapshot.balls[0].identity.evidence.clear();

    let error = snapshot
        .validate()
        .expect_err("number identity needs evidence");
    assert!(error.message.contains("evidence source"));
}
