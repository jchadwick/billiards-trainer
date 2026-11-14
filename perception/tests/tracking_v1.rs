use billiards_perception::Point2d;
use std::time::Duration;

use billiards_perception::ball_localization::{RefinementStatus, ResolvedBallMeasurementBatch};
use billiards_perception::detection_batch::{decode_detection_batch_v1, DetectionBatchV1};
use billiards_perception::tracking::{SceneTrackerV1, TrackingConfigV1, TrackingError};
use billiards_perception::{
    BallStatus, ClockKind, CueStatus, ImageSpace, Outcome, TimeUnit, Timestamp, TimestampStage,
};
use serde_json::{json, Value};

const SUCCESS_BATCH: &str =
    include_str!("../contracts/detection-batch/v1/detection-batch-success.json");
const TRACKED_SCENE_FIXTURE: &str = include_str!("../contracts/v1/scene-rust-tracked-success.json");

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
    let mut batch = frame(
        1,
        1_000_000_000,
        &[(1, 1882.4, 1024.7, 0.97), (0, 500.0, 400.0, 0.88)],
    );
    // This fixture represents a 64 px detector box; match its geometry so the
    // full scene output remains a meaningful differential contract check.
    batch.detections[0].bbox_image_px = billiards_perception::detection_batch::BoundingBox {
        left: 1850.4,
        top: 992.7,
        right: 1914.4,
        bottom: 1056.7,
    };
    let snapshot = process(&mut tracker, &batch).unwrap();

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
            .image_radius_px,
        Some(billiards_perception::Point2d { x: 32.0, y: 32.0 }),
        "the track retains the detector box half-extents"
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
fn clean_full_layout_replaces_absent_tracks_and_accepts_a_valid_empty_result() {
    let mut tracker = SceneTrackerV1::new(config());
    let mut process_full = |batch: &DetectionBatchV1| {
        let resolved = ResolvedBallMeasurementBatch::from_detection_batch_midpoints(
            batch,
            1,
            "camera-profile-test",
            7,
            3,
        );
        tracker
            .process_full_layout_with_resolved_ball_measurements(
                batch,
                Timestamp {
                    ticks: (2_000_000_000_u64 + batch.batch_sequence).to_string(),
                    unit: TimeUnit::Nanoseconds,
                    clock: ClockKind::Monotonic,
                    domain_id: "scene-writer-clock-a".to_owned(),
                    stage: TimestampStage::SnapshotPublication,
                },
                &resolved,
            )
            .unwrap()
            .snapshot
    };
    let initial = process_full(&frame(
        1,
        1_000_000_000,
        &[ball(20.0, 30.0), ball(80.0, 30.0)],
    ));
    assert_eq!(initial.balls.len(), 2);
    let initial_track_id = initial.balls[0].track_id.clone();

    // The periodic full scan is intentionally beyond the legacy one-second
    // active-track age; association still preserves both unambiguous IDs.
    let delayed = process_full(&frame(
        2,
        2_100_000_000,
        &[ball(21.0, 30.0), ball(81.0, 30.0)],
    ));
    assert_eq!(
        delayed
            .balls
            .iter()
            .map(|ball| ball.track_id.as_str())
            .collect::<Vec<_>>(),
        initial
            .balls
            .iter()
            .map(|ball| ball.track_id.as_str())
            .collect::<Vec<_>>()
    );

    let replacement = process_full(&frame(3, 2_200_000_000, &[ball(22.0, 30.0)]));
    assert_eq!(replacement.balls.len(), 1);
    assert_eq!(replacement.balls[0].track_id, initial_track_id);

    let empty = process_full(&frame(4, 2_300_000_000, &[]));
    assert!(
        empty.balls.is_empty(),
        "a clean empty scan removes every active ball"
    );
}

#[test]
fn partial_layout_preserves_unmatched_centers_and_ids_beyond_legacy_expiry() {
    let mut tracker = SceneTrackerV1::new(config());
    let mut scan = |batch: &DetectionBatchV1, partial: bool| {
        let resolved = ResolvedBallMeasurementBatch::from_detection_batch_midpoints(
            batch,
            1,
            "camera-profile-test",
            7,
            3,
        );
        let published_at = Timestamp {
            ticks: (9_000_000_000_u64 + batch.batch_sequence).to_string(),
            unit: TimeUnit::Nanoseconds,
            clock: ClockKind::Monotonic,
            domain_id: "scene-writer-clock-a".to_owned(),
            stage: TimestampStage::SnapshotPublication,
        };
        if partial {
            tracker.process_partial_layout_with_resolved_ball_measurements(
                batch,
                published_at,
                &resolved,
            )
        } else {
            tracker.process_full_layout_with_resolved_ball_measurements(
                batch,
                published_at,
                &resolved,
            )
        }
        .unwrap()
        .snapshot
    };
    let initial = scan(
        &frame(1, 1_000_000_000, &[ball(20.0, 30.0), ball(80.0, 30.0)]),
        false,
    );
    for (sequence, timestamp) in [(2, 2_100_000_000), (3, 5_000_000_000)] {
        let partial = scan(&frame(sequence, timestamp, &[ball(21.0, 30.0)]), true);
        assert_eq!(partial.balls.len(), 2);
        assert_eq!(partial.balls[0].track_id, initial.balls[0].track_id);
        assert_eq!(partial.balls[1].track_id, initial.balls[1].track_id);
        assert_eq!(partial.balls[1].measurement, initial.balls[1].measurement);
        assert_eq!(
            partial.balls[1].last_observation,
            initial.balls[1].last_observation
        );
    }
    let clean = scan(&frame(4, 6_000_000_000, &[ball(22.0, 30.0)]), false);
    assert_eq!(clean.balls.len(), 1);
    assert_eq!(clean.balls[0].track_id, initial.balls[0].track_id);
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

    let pressure_batch = frame(2, 1_010_000_000, &[ball(201.0, 100.0), ball(400.0, 100.0)]);
    let pressure_measurements = ResolvedBallMeasurementBatch::from_detection_batch_midpoints(
        &pressure_batch,
        1,
        "camera-profile-test",
        7,
        3,
    );
    let pressure = tracker
        .process_with_resolved_ball_measurements(
            &pressure_batch,
            Timestamp {
                ticks: "2000000002".to_owned(),
                unit: TimeUnit::Nanoseconds,
                clock: ClockKind::Monotonic,
                domain_id: "scene-writer-clock-a".to_owned(),
                stage: TimestampStage::SnapshotPublication,
            },
            &pressure_measurements,
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
fn resolved_center_drives_tracking_and_xy_while_raw_batch_midpoint_stays_unchanged() {
    let batch = frame(1, 1_000_000_000, &[ball(200.0, 100.0)]);
    let raw_midpoint = batch.detections[0].center_image_px;
    let mut resolved = ResolvedBallMeasurementBatch::from_detection_batch_midpoints(
        &batch,
        1,
        "camera-profile-test",
        7,
        3,
    );
    resolved.optics_profile_id = Some("optics-test".to_owned());
    resolved.optics_revision = Some(4);
    resolved.refiner_revision = Some(2);
    resolved.table_revision = 1;
    resolved.calibration_revision = Some(11);
    resolved.measurements[0].center_image_px = [202.0, 101.0];
    resolved.measurements[0].table_center_mm = Some([123.5, -456.25]);
    resolved.measurements[0].refinement_status = Some(RefinementStatus::Accepted);
    resolved.measurements[0].outline = Some(billiards_perception::ball_localization::BallOutline {
        points_image_px: vec![[180., 101.], [202., 91.], [224., 101.], [202., 111.]],
        half_extent_image_px: [22., 10.],
        local_radius_px: 10.,
    });

    let mut tracker = SceneTrackerV1::new(config());
    let result = tracker
        .process_with_resolved_ball_measurements(
            &batch,
            Timestamp {
                ticks: "2000000000".to_owned(),
                unit: TimeUnit::Nanoseconds,
                clock: ClockKind::Monotonic,
                domain_id: "scene-writer-clock-a".to_owned(),
                stage: TimestampStage::SnapshotPublication,
            },
            &resolved,
        )
        .unwrap();

    assert_eq!(batch.detections[0].center_image_px, raw_midpoint);
    let measurement = result.snapshot.balls[0].measurement.as_ref().unwrap();
    assert_eq!(measurement.image_center_px.x, 202.0);
    assert_eq!(measurement.image_center_px.y, 101.0);
    assert_eq!(measurement.table_center_mm.unwrap().x, 123.5);
    assert_eq!(measurement.table_center_mm.unwrap().y, -456.25);
    assert_eq!(result.snapshot.revisions.calibration, Some(11));
    result.snapshot.validate().unwrap();
    assert_eq!(
        measurement.image_radius_px,
        Some(Point2d { x: 22., y: 10. })
    );
    assert_eq!(
        measurement.image_outline_px.as_ref().unwrap()[0],
        Point2d { x: 180., y: 101. }
    );
    let missing = process(&mut tracker, &frame(2, 1_010_000_000, &[])).unwrap();
    assert!(missing.balls.iter().all(|b| b
        .measurement
        .as_ref()
        .is_none_or(|m| m.image_outline_px.is_none())));
}

#[test]
fn resolved_center_sidecar_rejects_a_different_frame_before_state_changes() {
    let batch = frame(1, 1_000_000_000, &[ball(200.0, 100.0)]);
    let mut resolved = ResolvedBallMeasurementBatch::from_detection_batch_midpoints(
        &batch,
        1,
        "camera-profile-test",
        7,
        3,
    );
    resolved.frame_id = "another-frame".to_owned();

    let mut tracker = SceneTrackerV1::new(config());
    let error = tracker
        .process_with_resolved_ball_measurements(
            &batch,
            Timestamp {
                ticks: "2000000000".to_owned(),
                unit: TimeUnit::Nanoseconds,
                clock: ClockKind::Monotonic,
                domain_id: "scene-writer-clock-a".to_owned(),
                stage: TimestampStage::SnapshotPublication,
            },
            &resolved,
        )
        .unwrap_err();
    assert!(matches!(
        error,
        TrackingError::ResolvedMeasurementMismatch("source_frame")
    ));

    let scene = process(&mut tracker, &batch).unwrap();
    assert_eq!(scene.snapshot_revision, 1);
    assert_eq!(
        scene.balls[0]
            .measurement
            .as_ref()
            .unwrap()
            .image_center_px
            .x,
        200.0
    );
}

#[test]
fn resolved_center_sidecar_rejects_stale_configuration_and_reordered_detections() {
    let batch = frame(1, 1_000_000_000, &[ball(200.0, 100.0), ball(400.0, 120.0)]);
    let base = ResolvedBallMeasurementBatch::from_detection_batch_midpoints(
        &batch,
        1,
        "camera-profile-test",
        7,
        3,
    );
    let mut stale = base.clone();
    stale.configuration_revision += 1;
    let mut reordered = base;
    reordered.measurements.swap(0, 1);

    for (resolved, expected_field) in [
        (stale, "configuration_revision"),
        (reordered, "detection_identity_or_measurement"),
    ] {
        let mut tracker = SceneTrackerV1::new(config());
        let error = tracker
            .process_with_resolved_ball_measurements(
                &batch,
                Timestamp {
                    ticks: "2000000000".to_owned(),
                    unit: TimeUnit::Nanoseconds,
                    clock: ClockKind::Monotonic,
                    domain_id: "scene-writer-clock-a".to_owned(),
                    stage: TimestampStage::SnapshotPublication,
                },
                &resolved,
            )
            .unwrap_err();
        assert_eq!(
            error,
            TrackingError::ResolvedMeasurementMismatch(expected_field)
        );
        let snapshot = process(&mut tracker, &batch).unwrap();
        assert_eq!(snapshot.balls[0].track_id, "track-1");
    }
}

#[test]
fn rejected_refinement_keeps_the_ball_visible_and_clears_its_precise_position() {
    let mut tracker = SceneTrackerV1::new(config());
    let first_batch = frame(1, 1_000_000_000, &[ball(200.0, 100.0)]);
    let mut first_measurement = ResolvedBallMeasurementBatch::from_detection_batch_midpoints(
        &first_batch,
        1,
        "camera-profile-test",
        7,
        3,
    );
    first_measurement.optics_profile_id = Some("optics-test".to_owned());
    first_measurement.optics_revision = Some(4);
    first_measurement.refiner_revision = Some(2);
    first_measurement.table_revision = 1;
    first_measurement.calibration_revision = Some(11);
    first_measurement.measurements[0].center_image_px = [202.0, 101.0];
    first_measurement.measurements[0].table_center_mm = Some([123.5, -456.25]);
    first_measurement.measurements[0].refinement_status = Some(RefinementStatus::Accepted);
    let first = tracker
        .process_with_resolved_ball_measurements(
            &first_batch,
            Timestamp {
                ticks: "2000000000".to_owned(),
                unit: TimeUnit::Nanoseconds,
                clock: ClockKind::Monotonic,
                domain_id: "scene-writer-clock-a".to_owned(),
                stage: TimestampStage::SnapshotPublication,
            },
            &first_measurement,
        )
        .unwrap()
        .snapshot;
    let track_id = first.balls[0].track_id.clone();

    let second_batch = frame(2, 1_020_000_000, &[ball(201.0, 100.0)]);
    let mut second_measurement = ResolvedBallMeasurementBatch::from_detection_batch_midpoints(
        &second_batch,
        1,
        "camera-profile-test",
        7,
        3,
    );
    second_measurement.optics_profile_id = Some("optics-test".to_owned());
    second_measurement.optics_revision = Some(4);
    second_measurement.refiner_revision = Some(2);
    second_measurement.table_revision = 1;
    second_measurement.calibration_revision = Some(11);
    second_measurement.measurements[0].refinement_status = Some(RefinementStatus::Rejected);
    let second = tracker
        .process_with_resolved_ball_measurements(
            &second_batch,
            Timestamp {
                ticks: "2020000000".to_owned(),
                unit: TimeUnit::Nanoseconds,
                clock: ClockKind::Monotonic,
                domain_id: "scene-writer-clock-a".to_owned(),
                stage: TimestampStage::SnapshotPublication,
            },
            &second_measurement,
        )
        .unwrap()
        .snapshot;

    assert_eq!(second.balls[0].track_id, track_id);
    assert_eq!(second.balls[0].status, BallStatus::Observed);
    assert_eq!(
        second.balls[0]
            .measurement
            .as_ref()
            .unwrap()
            .image_center_px
            .x,
        201.0
    );
    assert!(second.balls[0]
        .measurement
        .as_ref()
        .unwrap()
        .table_center_mm
        .is_none());
    second.validate().unwrap();
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
