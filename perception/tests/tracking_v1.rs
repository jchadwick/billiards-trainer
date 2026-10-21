use std::time::Duration;

use billiards_perception::calibration::{TableCalibrationV1, ValidatedTableCalibrationV1};
use billiards_perception::detection_batch::{decode_detection_batch_v1, DetectionBatchV1};
use billiards_perception::geometry::{calibrate_detection_batch, AppliedCalibrationContext};
use billiards_perception::tracking::{
    AppliedTrackingCalibration, SceneTrackerV1, TrackingConfigV1, TrackingError,
};
use billiards_perception::{
    BallStatus, CalibrationState, ClockKind, CueStatus, ImageSpace, Outcome, TimeUnit, Timestamp,
    TimestampStage,
};
use serde_json::{json, Value};

const SUCCESS_BATCH: &str =
    include_str!("../contracts/detection-batch/v1/detection-batch-success.json");
const TRACKED_SCENE_FIXTURE: &str = include_str!("../contracts/v1/scene-rust-tracked-success.json");
const TABLE_CALIBRATION: &str =
    include_str!("../contracts/calibration/v1/table-calibration-center-origin.json");

fn config() -> TrackingConfigV1 {
    TrackingConfigV1::new(
        1,
        Some(0),
        "camera-profile-test",
        2.5,
        Duration::from_millis(50),
        Duration::from_millis(100),
        Duration::from_millis(500),
        16,
        4,
        7,
        3,
    )
    .unwrap()
}

fn frame(
    batch_sequence: u64,
    source_ticks: u128,
    detections: &[(u32, f64, f64, f64)],
) -> DetectionBatchV1 {
    let mut value: Value = serde_json::from_str(SUCCESS_BATCH).unwrap();
    value["batch_sequence"] = json!(batch_sequence);
    value["source"]["frame_id"] = json!(format!("frame-{batch_sequence:04}"));
    value["source"]["timestamp"]["ticks"] = json!(source_ticks.to_string());
    value["inference_started_at"]["ticks"] = json!((source_ticks + 100).to_string());
    value["inference_completed_at"]["ticks"] = json!((source_ticks + 200).to_string());
    value["published_at"]["ticks"] = json!((source_ticks + 300).to_string());
    value["detections"] = json!(detections
        .iter()
        .enumerate()
        .map(|(index, (class_id, x, y, confidence))| {
            json!({
                "detection_index": index,
                "class_id": class_id,
                "class_name": if *class_id == 0 { "cue" } else { "ball" },
                "confidence": confidence,
                "bbox_image_px": {
                    "left": x - 1.0,
                    "top": y - 1.0,
                    "right": x + 1.0,
                    "bottom": y + 1.0
                },
                "center_image_px": { "x": x, "y": y }
            })
        })
        .collect::<Vec<_>>());
    decode_detection_batch_v1(&serde_json::to_vec(&value).unwrap()).unwrap()
}

fn ball(x: f64, y: f64) -> (u32, f64, f64, f64) {
    (1, x, y, 0.97)
}

fn table_calibration() -> ValidatedTableCalibrationV1 {
    serde_json::from_str::<TableCalibrationV1>(TABLE_CALIBRATION)
        .unwrap()
        .into_validated()
        .unwrap()
}

fn calibrated_frame(
    batch_sequence: u64,
    source_ticks: u128,
    detections: &[(u32, f64, f64, f64)],
) -> DetectionBatchV1 {
    let mut value = serde_json::to_value(frame(batch_sequence, source_ticks, detections)).unwrap();
    value["source"]["width_px"] = json!(1000);
    value["source"]["height_px"] = json!(500);
    value["image_space"] = json!("rectified");
    value["detections"] = json!(detections
        .iter()
        .enumerate()
        .map(|(index, (class_id, x, y, confidence))| {
            json!({
                "detection_index": index,
                "class_id": class_id,
                "class_name": if *class_id == 0 { "cue" } else { "ball" },
                "confidence": confidence,
                "bbox_image_px": { "left": x - 1.0, "top": y - 1.0, "right": x + 1.0, "bottom": y + 1.0 },
                "center_image_px": { "x": x, "y": y }
            })
        })
        .collect::<Vec<_>>());
    decode_detection_batch_v1(&serde_json::to_vec(&value).unwrap()).unwrap()
}

fn process(
    tracker: &mut SceneTrackerV1,
    batch: &DetectionBatchV1,
) -> Result<billiards_perception::SceneSnapshotV1, TrackingError> {
    // This is a distinct synthetic publisher clock. The tracker must not
    // relabel the detector batch's earlier publication timestamp.
    tracker.process(
        batch,
        Timestamp {
            ticks: (2_000_000_000_u64 + batch.batch_sequence).to_string(),
            unit: TimeUnit::Nanoseconds,
            clock: ClockKind::Monotonic,
            domain_id: "scene-writer-clock-a".to_owned(),
            stage: TimestampStage::SnapshotPublication,
        },
    )
}

#[test]
fn first_measurement_is_published_immediately_without_identity_guessing() {
    let mut tracker = SceneTrackerV1::new(config());
    let snapshot = process(
        &mut tracker,
        &frame(
            1,
            1_000_000_000,
            &[(1, 1882.4, 1024.7, 0.97), (0, 500.0, 400.0, 0.88)],
        ),
    )
    .unwrap();

    assert_eq!(snapshot.outcome, Outcome::Success);
    assert_eq!(snapshot.balls.len(), 1);
    assert_eq!(snapshot.balls[0].track_id, "track-1");
    assert_eq!(snapshot.balls[0].status, BallStatus::Observed);
    assert_eq!(
        snapshot.balls[0]
            .measurement
            .as_ref()
            .unwrap()
            .image_center_px
            .x,
        1882.4
    );
    assert_eq!(
        snapshot.balls[0]
            .measurement
            .as_ref()
            .unwrap()
            .table_center_mm,
        None
    );
    assert_eq!(snapshot.balls[0].identity.number, None);
    assert_eq!(snapshot.cue.status, CueStatus::Observed);
    assert_eq!(snapshot.source.timestamp.ticks, "1000000000");
    assert_eq!(snapshot.published_at.ticks, "2000000001");
    assert_eq!(snapshot.published_at.domain_id, "scene-writer-clock-a");
    snapshot.validate().unwrap();
    let encoded = serde_json::to_vec(&snapshot).unwrap();
    let decoded: billiards_perception::SceneSnapshotV1 = serde_json::from_slice(&encoded).unwrap();
    assert_eq!(decoded, snapshot);
    decoded.validate().unwrap();
    assert_eq!(
        serde_json::to_value(&snapshot).unwrap(),
        serde_json::from_str::<Value>(TRACKED_SCENE_FIXTURE).unwrap()
    );
}

#[test]
fn global_assignment_preserves_both_tracks_when_greedy_matching_would_lose_one() {
    let mut tracker = SceneTrackerV1::new(config());
    let first = process(
        &mut tracker,
        &frame(1, 1_000_000_000, &[ball(100.0, 100.0), ball(104.0, 100.0)]),
    )
    .unwrap();
    assert_eq!(first.balls[0].track_id, "track-1");
    assert_eq!(first.balls[1].track_id, "track-2");

    // Track 2 can only match detection 0 within the gate. A global assignment
    // therefore reserves detection 0 for track 2 and assigns track 1 to 1.
    let crossing = process(
        &mut tracker,
        &frame(2, 1_033_000_000, &[ball(102.0, 100.0), ball(98.0, 100.0)]),
    )
    .unwrap();
    assert_eq!(crossing.balls[0].track_id, "track-1");
    assert_eq!(
        crossing.balls[0]
            .measurement
            .as_ref()
            .unwrap()
            .image_center_px
            .x,
        98.0
    );
    assert_eq!(crossing.balls[1].track_id, "track-2");
    assert_eq!(
        crossing.balls[1]
            .measurement
            .as_ref()
            .unwrap()
            .image_center_px
            .x,
        102.0
    );
}

#[test]
fn bounded_velocity_is_used_only_to_associate_after_a_longer_source_gap() {
    let moving_config = TrackingConfigV1::new(
        1,
        Some(0),
        "camera-profile-test",
        11.0,
        Duration::from_millis(100),
        Duration::from_millis(150),
        Duration::from_millis(500),
        16,
        4,
        7,
        3,
    )
    .unwrap();
    let mut tracker = SceneTrackerV1::new(moving_config);
    process(
        &mut tracker,
        &frame(1, 1_000_000_000, &[ball(100.0, 100.0)]),
    )
    .unwrap();
    process(
        &mut tracker,
        &frame(2, 1_033_000_000, &[ball(110.0, 100.0)]),
    )
    .unwrap();
    process(
        &mut tracker,
        &frame(3, 1_066_000_000, &[ball(120.0, 100.0)]),
    )
    .unwrap();

    // The raw displacement from the last center exceeds the gate. The bounded
    // source-time association estimate lands on this detection, but the
    // published measurement must still be the detector's x=150 location.
    let after_gap = process(
        &mut tracker,
        &frame(4, 1_166_000_000, &[ball(150.0, 100.0)]),
    )
    .unwrap();
    assert_eq!(after_gap.balls[0].track_id, "track-1");
    assert_eq!(
        after_gap.balls[0]
            .measurement
            .as_ref()
            .unwrap()
            .image_center_px
            .x,
        150.0
    );
    assert_eq!(after_gap.balls[0].estimate, None);
}

#[test]
fn missed_tracks_age_from_their_last_observation_and_expire_by_source_time() {
    let mut tracker = SceneTrackerV1::new(config());
    let first = process(
        &mut tracker,
        &frame(1, 1_000_000_000, &[ball(100.0, 100.0)]),
    )
    .unwrap();
    let original_observation = first.balls[0].last_observation.clone();

    let occluded = process(&mut tracker, &frame(2, 1_050_000_000, &[])).unwrap();
    assert_eq!(occluded.balls[0].status, BallStatus::Occluded);
    assert_eq!(occluded.balls[0].last_observation, original_observation);
    assert_eq!(
        occluded.balls[0]
            .measurement
            .as_ref()
            .unwrap()
            .image_center_px
            .x,
        100.0
    );
    assert_eq!(occluded.cue.status, CueStatus::NotObserved);

    let lost = process(&mut tracker, &frame(3, 1_110_000_000, &[])).unwrap();
    assert_eq!(lost.balls[0].status, BallStatus::Lost);
    assert_eq!(lost.balls[0].last_observation, original_observation);

    let reacquired = process(
        &mut tracker,
        &frame(4, 1_300_000_000, &[ball(101.0, 100.0)]),
    )
    .unwrap();
    assert_eq!(reacquired.balls[0].track_id, "track-1");
    assert_eq!(reacquired.balls[0].status, BallStatus::Observed);
    assert_eq!(reacquired.balls[0].last_observation.frame_id, "frame-0004");

    let expired = process(&mut tracker, &frame(5, 1_850_000_000, &[])).unwrap();
    assert!(expired.balls.is_empty());
    assert_eq!(
        expired.cue.last_observation.as_ref().unwrap().frame_id,
        "frame-0005"
    );
}

#[test]
fn duplicate_and_out_of_order_observations_are_rejected_without_mutation() {
    let mut tracker = SceneTrackerV1::new(config());
    let first = process(
        &mut tracker,
        &frame(1, 1_000_000_000, &[ball(100.0, 100.0)]),
    )
    .unwrap();
    let duplicate = frame(2, 1_010_000_000, &[ball(101.0, 100.0)]);
    let mut duplicate_value: Value = serde_json::to_value(&duplicate).unwrap();
    duplicate_value["source"]["frame_id"] = json!("frame-0001");
    let duplicate =
        decode_detection_batch_v1(&serde_json::to_vec(&duplicate_value).unwrap()).unwrap();
    assert_eq!(
        process(&mut tracker, &duplicate),
        Err(TrackingError::DuplicateSourceFrame)
    );

    let out_of_order = frame(3, 999_000_000, &[ball(101.0, 100.0)]);
    assert_eq!(
        process(&mut tracker, &out_of_order),
        Err(TrackingError::NonIncreasingSourceTime)
    );

    let next = process(
        &mut tracker,
        &frame(2, 1_020_000_000, &[ball(102.0, 100.0)]),
    )
    .unwrap();
    assert_eq!(next.balls[0].track_id, first.balls[0].track_id);
    assert_eq!(next.snapshot_revision, 2);
}

#[test]
fn detector_failure_and_missing_source_time_do_not_become_empty_scenes() {
    let mut tracker = SceneTrackerV1::new(config());
    let first = process(
        &mut tracker,
        &frame(1, 1_000_000_000, &[ball(100.0, 100.0)]),
    )
    .unwrap();

    let mut failure_value: Value = serde_json::to_value(frame(2, 1_010_000_000, &[])).unwrap();
    failure_value["outcome"] = json!("failed");
    failure_value["reason"] = json!("inference unavailable");
    let failure = decode_detection_batch_v1(&serde_json::to_vec(&failure_value).unwrap()).unwrap();
    assert_eq!(
        process(&mut tracker, &failure),
        Err(TrackingError::UnsuccessfulBatch(
            billiards_perception::detection_batch::BatchOutcome::Failed
        ))
    );

    let mut skipped_value: Value = serde_json::to_value(frame(2, 1_015_000_000, &[])).unwrap();
    skipped_value["outcome"] = json!("skipped");
    skipped_value["reason"] = json!("source frame was intentionally skipped");
    let skipped = decode_detection_batch_v1(&serde_json::to_vec(&skipped_value).unwrap()).unwrap();
    assert_eq!(
        process(&mut tracker, &skipped),
        Err(TrackingError::UnsuccessfulBatch(
            billiards_perception::detection_batch::BatchOutcome::Skipped
        ))
    );

    let mut no_time_value: Value = serde_json::to_value(frame(2, 1_020_000_000, &[])).unwrap();
    no_time_value["source"]["timestamp"] = Value::Null;
    let no_time = decode_detection_batch_v1(&serde_json::to_vec(&no_time_value).unwrap()).unwrap();
    assert_eq!(
        process(&mut tracker, &no_time),
        Err(TrackingError::MissingSourceTimestamp)
    );

    let next = process(
        &mut tracker,
        &frame(2, 1_030_000_000, &[ball(103.0, 100.0)]),
    )
    .unwrap();
    assert_eq!(next.balls[0].track_id, first.balls[0].track_id);
    assert_eq!(next.snapshot_revision, 2);
}

#[test]
fn session_geometry_and_clock_changes_invalidate_association_without_false_continuity() {
    let mut tracker = SceneTrackerV1::new(config());
    let first = process(
        &mut tracker,
        &frame(1, 1_000_000_000, &[ball(100.0, 100.0)]),
    )
    .unwrap();

    let mut resized_value: Value =
        serde_json::to_value(frame(2, 1_020_000_000, &[ball(100.0, 100.0)])).unwrap();
    resized_value["source"]["width_px"] = json!(1920);
    resized_value["source"]["height_px"] = json!(1080);
    resized_value["image_space"] = json!("rectified");
    let resized = decode_detection_batch_v1(&serde_json::to_vec(&resized_value).unwrap()).unwrap();
    let after_geometry_change = process(&mut tracker, &resized).unwrap();
    assert_eq!(after_geometry_change.balls[0].track_id, "track-2");
    assert_ne!(
        after_geometry_change.balls[0].track_id,
        first.balls[0].track_id
    );

    let mut model_value: Value =
        serde_json::to_value(frame(3, 1_040_000_000, &[ball(100.0, 100.0)])).unwrap();
    model_value["model"]["artifact_sha256"] = json!("c".repeat(64));
    let new_model = decode_detection_batch_v1(&serde_json::to_vec(&model_value).unwrap()).unwrap();
    let after_model_change = process(&mut tracker, &new_model).unwrap();
    assert_eq!(after_model_change.balls[0].track_id, "track-3");

    let mut stage_value: Value =
        serde_json::to_value(frame(4, 1_060_000_000, &[ball(100.0, 100.0)])).unwrap();
    stage_value["source"]["timestamp"]["stage"] = json!("frame_arrival");
    let new_stage = decode_detection_batch_v1(&serde_json::to_vec(&stage_value).unwrap()).unwrap();
    let after_stage_change = process(&mut tracker, &new_stage).unwrap();
    assert_eq!(after_stage_change.balls[0].track_id, "track-4");

    let mut clock_value: Value =
        serde_json::to_value(frame(5, 1_000, &[ball(100.0, 100.0)])).unwrap();
    clock_value["source"]["timestamp"]["domain_id"] = json!("new-clock-domain");
    let new_clock = decode_detection_batch_v1(&serde_json::to_vec(&clock_value).unwrap()).unwrap();
    let after_clock_change = process(&mut tracker, &new_clock).unwrap();
    assert_eq!(after_clock_change.balls[0].track_id, "track-5");

    let mut session_value: Value =
        serde_json::to_value(frame(1, 10, &[ball(100.0, 100.0)])).unwrap();
    session_value["source"]["session_id"] = json!("restarted-session");
    let new_session =
        decode_detection_batch_v1(&serde_json::to_vec(&session_value).unwrap()).unwrap();
    let after_session_change = process(&mut tracker, &new_session).unwrap();
    assert_eq!(after_session_change.balls[0].track_id, "track-1");
    assert_eq!(after_session_change.source.session_id, "restarted-session");

    let mut engine_value: Value =
        serde_json::to_value(frame(1, 20, &[ball(100.0, 100.0)])).unwrap();
    engine_value["engine_instance_id"] = json!("engine-rust-stage2-b");
    engine_value["source"]["session_id"] = json!("restarted-session");
    let new_engine =
        decode_detection_batch_v1(&serde_json::to_vec(&engine_value).unwrap()).unwrap();
    let after_engine_restart = process(&mut tracker, &new_engine).unwrap();
    assert_eq!(after_engine_restart.balls[0].track_id, "track-1");
    assert_eq!(after_engine_restart.snapshot_revision, 1);
    assert_eq!(after_engine_restart.transport_sequence, 1);
}

#[test]
fn reconfiguration_resets_association_without_recycling_track_ids() {
    let mut tracker = SceneTrackerV1::new(config());
    let first = process(
        &mut tracker,
        &frame(1, 1_000_000_000, &[ball(100.0, 100.0)]),
    )
    .unwrap();
    assert_eq!(first.balls[0].track_id, "track-1");

    let updated = TrackingConfigV1::new(
        1,
        Some(0),
        "camera-profile-after-change",
        2.5,
        Duration::from_millis(50),
        Duration::from_millis(100),
        Duration::from_millis(500),
        16,
        4,
        8,
        4,
    )
    .unwrap();
    tracker.reconfigure(updated);
    let after_change = process(
        &mut tracker,
        &frame(2, 1_020_000_000, &[ball(100.0, 100.0)]),
    )
    .unwrap();
    assert_eq!(after_change.balls[0].track_id, "track-2");
    assert_eq!(after_change.snapshot_revision, 2);
    assert_eq!(after_change.revisions.configuration, 8);
    assert_eq!(after_change.revisions.rectification, 4);
}

#[test]
fn invalid_tracking_configuration_is_rejected() {
    assert!(TrackingConfigV1::new(
        1,
        Some(1),
        "camera-profile-test",
        2.5,
        Duration::from_millis(50),
        Duration::from_millis(100),
        Duration::from_millis(500),
        16,
        4,
        7,
        3,
    )
    .is_err());
    assert!(TrackingConfigV1::new(
        1,
        Some(0),
        "camera-profile-test",
        f64::NAN,
        Duration::from_millis(50),
        Duration::from_millis(100),
        Duration::from_millis(500),
        16,
        4,
        7,
        3,
    )
    .is_err());
}

#[test]
fn capacity_and_invalid_publication_failures_leave_tracker_state_unchanged() {
    let config = TrackingConfigV1::new(
        1,
        Some(0),
        "camera-profile-test",
        2.5,
        Duration::from_millis(50),
        Duration::from_millis(100),
        Duration::from_millis(500),
        1,
        4,
        7,
        3,
    )
    .unwrap();
    let mut tracker = SceneTrackerV1::new(config);
    let first = process(
        &mut tracker,
        &frame(1, 1_000_000_000, &[ball(100.0, 100.0)]),
    )
    .unwrap();

    let invalid_publication = Timestamp {
        ticks: "2000000002".to_owned(),
        unit: TimeUnit::Nanoseconds,
        clock: ClockKind::Monotonic,
        domain_id: "scene-writer-clock-a".to_owned(),
        stage: TimestampStage::FrameArrival,
    };
    assert_eq!(
        tracker.process(
            &frame(2, 1_010_000_000, &[ball(101.0, 100.0)]),
            invalid_publication
        ),
        Err(TrackingError::InvalidPublicationTimestamp)
    );

    let over_capacity = frame(2, 1_020_000_000, &[ball(101.0, 100.0), ball(200.0, 100.0)]);
    assert_eq!(
        process(&mut tracker, &over_capacity),
        Err(TrackingError::ActiveTrackCapacityExceeded)
    );

    let next = process(
        &mut tracker,
        &frame(2, 1_030_000_000, &[ball(102.0, 100.0)]),
    )
    .unwrap();
    assert_eq!(next.balls[0].track_id, first.balls[0].track_id);
    assert_eq!(next.snapshot_revision, 2);
}

#[test]
fn capacity_pressure_evicts_oldest_unmatched_track_and_publishes_all_current_measurements() {
    let config = TrackingConfigV1::new(
        1,
        Some(0),
        "camera-profile-test",
        2.5,
        Duration::from_millis(50),
        Duration::from_millis(100),
        Duration::from_millis(500),
        2,
        4,
        7,
        3,
    )
    .unwrap();
    let mut tracker = SceneTrackerV1::new(config);
    let first = process(
        &mut tracker,
        &frame(1, 1_000_000_000, &[ball(100.0, 100.0), ball(200.0, 100.0)]),
    )
    .unwrap();
    assert_eq!(
        first
            .balls
            .iter()
            .map(|ball| ball.track_id.as_str())
            .collect::<Vec<_>>(),
        ["track-1", "track-2"]
    );

    let pressure = tracker
        .process_with_precalibrated_batch_report(
            &frame(2, 1_010_000_000, &[ball(201.0, 100.0), ball(400.0, 100.0)]),
            Timestamp {
                ticks: "2000000002".to_owned(),
                unit: TimeUnit::Nanoseconds,
                clock: ClockKind::Monotonic,
                domain_id: "scene-writer-clock-a".to_owned(),
                stage: TimestampStage::SnapshotPublication,
            },
            None,
        )
        .unwrap();
    assert_eq!(pressure.association_capacity_evictions, 1);
    assert_eq!(pressure.snapshot.outcome, Outcome::Degraded);
    assert_eq!(
        pressure.snapshot.reason.as_deref(),
        Some("association_capacity_pressure")
    );
    assert_eq!(pressure.snapshot.balls.len(), 2);
    assert_eq!(
        pressure
            .snapshot
            .balls
            .iter()
            .map(|ball| ball.track_id.as_str())
            .collect::<Vec<_>>(),
        ["track-2", "track-3"]
    );
    assert!(pressure
        .snapshot
        .balls
        .iter()
        .all(|ball| ball.status == BallStatus::Observed));
    assert_eq!(
        pressure
            .snapshot
            .balls
            .iter()
            .map(|ball| { ball.measurement.as_ref().unwrap().image_center_px.x })
            .collect::<Vec<_>>(),
        [201.0, 400.0]
    );
    pressure.snapshot.validate().unwrap();

    let recovered = process(
        &mut tracker,
        &frame(3, 1_020_000_000, &[ball(202.0, 100.0), ball(401.0, 100.0)]),
    )
    .unwrap();
    assert_eq!(recovered.outcome, Outcome::Success);
    assert_eq!(
        recovered
            .balls
            .iter()
            .map(|ball| ball.track_id.as_str())
            .collect::<Vec<_>>(),
        ["track-2", "track-3"]
    );
}

#[test]
fn applied_calibration_adds_table_coordinates_and_revision_to_scene_measurements() {
    let mut tracker = SceneTrackerV1::new(
        TrackingConfigV1::new(
            1,
            Some(0),
            "test-camera-profile",
            10.0,
            Duration::from_millis(50),
            Duration::from_millis(100),
            Duration::from_millis(500),
            16,
            4,
            7,
            2,
        )
        .unwrap(),
    );
    let calibration = table_calibration();
    let snapshot = tracker
        .process_with_calibration(
            &calibrated_frame(1, 1_000_000_000, &[ball(500.0, 250.0)]),
            Timestamp {
                ticks: "2000000000".to_owned(),
                unit: TimeUnit::Nanoseconds,
                clock: ClockKind::Monotonic,
                domain_id: "scene-writer-clock-a".to_owned(),
                stage: TimestampStage::SnapshotPublication,
            },
            Some(AppliedTrackingCalibration {
                calibration: &calibration,
                context: AppliedCalibrationContext {
                    calibration_revision: 3,
                    camera_profile_id: "test-camera-profile",
                    rectification_revision: 2,
                },
            }),
        )
        .unwrap();

    assert_eq!(
        snapshot.revisions.calibration_state,
        CalibrationState::Valid
    );
    assert_eq!(snapshot.revisions.calibration, Some(3));
    assert_eq!(snapshot.outcome, Outcome::Success);
    assert_eq!(
        snapshot.balls[0]
            .measurement
            .as_ref()
            .unwrap()
            .table_center_mm,
        Some(billiards_perception::Point2d { x: 0.0, y: 0.0 })
    );
    snapshot.validate().unwrap();
}

#[test]
fn failed_or_removed_calibration_clears_physical_positions_without_losing_tracks() {
    let mut tracker = SceneTrackerV1::new(
        TrackingConfigV1::new(
            1,
            Some(0),
            "test-camera-profile",
            10.0,
            Duration::from_millis(50),
            Duration::from_millis(100),
            Duration::from_millis(500),
            16,
            4,
            7,
            2,
        )
        .unwrap(),
    );
    let calibration = table_calibration();
    let first = tracker
        .process_with_calibration(
            &calibrated_frame(1, 1_000_000_000, &[ball(500.0, 250.0)]),
            Timestamp {
                ticks: "2000000000".to_owned(),
                unit: TimeUnit::Nanoseconds,
                clock: ClockKind::Monotonic,
                domain_id: "scene-writer-clock-a".to_owned(),
                stage: TimestampStage::SnapshotPublication,
            },
            Some(AppliedTrackingCalibration {
                calibration: &calibration,
                context: AppliedCalibrationContext {
                    calibration_revision: 4,
                    camera_profile_id: "test-camera-profile",
                    rectification_revision: 2,
                },
            }),
        )
        .unwrap();
    assert_eq!(first.revisions.calibration_state, CalibrationState::Invalid);
    assert_eq!(first.outcome, Outcome::Degraded);
    assert_eq!(
        first.balls[0].measurement.as_ref().unwrap().table_center_mm,
        None
    );

    let second = tracker
        .process_with_calibration(
            &calibrated_frame(2, 1_020_000_000, &[]),
            Timestamp {
                ticks: "2020000000".to_owned(),
                unit: TimeUnit::Nanoseconds,
                clock: ClockKind::Monotonic,
                domain_id: "scene-writer-clock-a".to_owned(),
                stage: TimestampStage::SnapshotPublication,
            },
            None,
        )
        .unwrap();
    assert_eq!(
        second.revisions.calibration_state,
        CalibrationState::Unavailable
    );
    assert_eq!(second.balls.len(), 1);
    assert_eq!(second.balls[0].track_id, first.balls[0].track_id);
    assert_eq!(
        second.balls[0]
            .measurement
            .as_ref()
            .unwrap()
            .table_center_mm,
        None
    );
    second.validate().unwrap();
}

#[test]
fn tracking_rejects_applied_camera_geometry_that_disagrees_with_its_revision_identity() {
    let mut tracker = SceneTrackerV1::new(config());
    let batch = frame(1, 1_000_000_000, &[ball(100.0, 100.0)]);
    let calibration = table_calibration();
    let publication = Timestamp {
        ticks: "2000000000".to_owned(),
        unit: TimeUnit::Nanoseconds,
        clock: ClockKind::Monotonic,
        domain_id: "scene-writer-clock-a".to_owned(),
        stage: TimestampStage::SnapshotPublication,
    };

    let error = tracker
        .process_with_calibration(
            &batch,
            publication.clone(),
            Some(AppliedTrackingCalibration {
                calibration: &calibration,
                context: AppliedCalibrationContext {
                    calibration_revision: 3,
                    camera_profile_id: "different-camera-profile",
                    rectification_revision: 3,
                },
            }),
        )
        .unwrap_err();
    assert_eq!(error, TrackingError::AppliedCalibrationContextMismatch);

    // The rejected configuration did not consume the observation or mutate IDs.
    let snapshot = tracker.process(&batch, publication).unwrap();
    assert_eq!(snapshot.balls[0].track_id, "track-1");
}

#[test]
fn precalibrated_results_must_match_the_exact_detection_frame_and_camera_revisions() {
    let batch = calibrated_frame(1, 1_000_000_000, &[ball(500.0, 250.0)]);
    let calibration = table_calibration();
    let context = AppliedCalibrationContext {
        calibration_revision: 3,
        camera_profile_id: "test-camera-profile",
        rectification_revision: 2,
    };
    let calibrated =
        calibrate_detection_batch(&batch, &calibration, context).expect("fixture calibrates");

    #[derive(Clone, Copy)]
    enum Mismatch {
        SourceFrame,
        DetectionMeasurement,
        CameraProfile,
        CalibrationRevision,
        RectificationRevision,
        ImageSpace,
    }

    for mismatch in [
        Mismatch::SourceFrame,
        Mismatch::DetectionMeasurement,
        Mismatch::CameraProfile,
        Mismatch::CalibrationRevision,
        Mismatch::RectificationRevision,
        Mismatch::ImageSpace,
    ] {
        let mut corrupted = calibrated.clone();
        let expected_field = match mismatch {
            Mismatch::SourceFrame => {
                corrupted.source.frame_id = "stale-frame".to_owned();
                "source_frame"
            }
            Mismatch::DetectionMeasurement => {
                corrupted.detections[0].center_image_px.x += 1.0;
                "detection_identity_or_measurement"
            }
            Mismatch::CameraProfile => {
                corrupted.camera_profile_id = "other-camera".to_owned();
                "camera_profile_id"
            }
            Mismatch::CalibrationRevision => {
                corrupted.calibration_revision += 1;
                "calibration_revision"
            }
            Mismatch::RectificationRevision => {
                corrupted.rectification_revision += 1;
                "rectification_revision"
            }
            Mismatch::ImageSpace => {
                corrupted.image_space = billiards_perception::detection_batch::ImageSpace::Raw;
                "image_space"
            }
        };
        let mut tracker = SceneTrackerV1::new(
            TrackingConfigV1::new(
                1,
                Some(0),
                "test-camera-profile",
                10.0,
                Duration::from_millis(50),
                Duration::from_millis(100),
                Duration::from_millis(500),
                16,
                4,
                7,
                2,
            )
            .unwrap(),
        );

        let error = tracker
            .process_with_precalibrated_batch(
                &batch,
                Timestamp {
                    ticks: "2000000000".to_owned(),
                    unit: TimeUnit::Nanoseconds,
                    clock: ClockKind::Monotonic,
                    domain_id: "scene-writer-clock-a".to_owned(),
                    stage: TimestampStage::SnapshotPublication,
                },
                Some((3, Ok(corrupted))),
            )
            .expect_err("mismatched physical coordinates must fail closed");
        assert_eq!(
            error,
            TrackingError::PrecalibratedBatchMismatch(expected_field)
        );

        // A rejected transform must not consume the source observation or
        // change the track-ID epoch.
        let snapshot = process(&mut tracker, &batch).unwrap();
        assert_eq!(snapshot.balls[0].track_id, "track-1");
    }
}

#[test]
fn rectified_measurements_retain_the_declared_space_and_unknown_cue_is_explicit() {
    let mut tracker = SceneTrackerV1::new(
        TrackingConfigV1::new(
            1,
            None,
            "camera-profile-test",
            2.5,
            Duration::from_millis(50),
            Duration::from_millis(100),
            Duration::from_millis(500),
            16,
            4,
            7,
            3,
        )
        .unwrap(),
    );
    let mut value: Value =
        serde_json::to_value(frame(1, 1_000_000_000, &[ball(100.0, 100.0)])).unwrap();
    value["image_space"] = json!("rectified");
    let batch = decode_detection_batch_v1(&serde_json::to_vec(&value).unwrap()).unwrap();
    let snapshot = process(&mut tracker, &batch).unwrap();
    assert_eq!(
        snapshot.balls[0].measurement.as_ref().unwrap().image_space,
        ImageSpace::Rectified
    );
    assert_eq!(snapshot.cue.status, CueStatus::Unavailable);
}
