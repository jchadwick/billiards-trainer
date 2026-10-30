use billiards_perception::{
    decode_scene_snapshot_v1,
    public_state::PublicStateConfig,
    public_state_lifecycle::{
        IngestOutcome, SceneStateLifecycle, SourceStateMetadata, StateFreshness,
        StateLifecycleConfig, StateLifecycleError,
    },
    Outcome, UpdateKind,
};
use serde_json::Value;

fn scene() -> billiards_perception::SceneSnapshotV1 {
    let mut scene = decode_scene_snapshot_v1(include_bytes!(
        "../contracts/v1/scene-rust-tracked-success.json"
    ))
    .unwrap();
    scene.engine_instance_id = "engine-a".into();
    scene.source.session_id = "camera-session-a".into();
    for ball in &mut scene.balls {
        ball.last_observation.session_id = "camera-session-a".into();
    }
    scene
}

fn lifecycle() -> SceneStateLifecycle {
    SceneStateLifecycle::new(
        StateLifecycleConfig {
            stale_after_ns: 100,
            expire_after_ns: 500,
            maximum_balls: 16,
        },
        PublicStateConfig::default(),
    )
    .unwrap()
}

fn activate(lifecycle: &mut SceneStateLifecycle) -> u64 {
    lifecycle
        .activate_source("engine-a", "camera-session-a")
        .unwrap()
}

fn advance(scene: &mut billiards_perception::SceneSnapshotV1, sequence: u64, frame: &str) {
    scene.transport_sequence = sequence;
    scene.snapshot_revision = sequence;
    scene.source.frame_id = frame.into();
    scene.source.timestamp.ticks = (1_000_000_000 + sequence).to_string();
    scene.published_at.ticks = (1_000_000_001 + sequence).to_string();
    for ball in &mut scene.balls {
        if ball.status == billiards_perception::BallStatus::Observed {
            ball.last_observation.frame_id = frame.into();
            ball.last_observation.timestamp.ticks = scene.source.timestamp.ticks.clone();
        }
    }
}

#[test]
fn requires_explicit_identity_and_publishes_latest_fresh_state() {
    let mut store = lifecycle();
    let first = scene();
    assert!(matches!(
        store.ingest(&first, 10, 10, None, None),
        Err(StateLifecycleError::SourceNotActivated)
    ));
    assert_eq!(activate(&mut store), 1);
    assert_eq!(
        store.ingest(&first, 10, 20, None, None).unwrap(),
        IngestOutcome::LocalizationUpdated
    );
    let state = store.current(50).unwrap().unwrap();
    assert_eq!(state.freshness, StateFreshness::Fresh);
    assert_eq!(state.localization_age_ns, Some(40));
    assert_eq!(
        state.payload.observation_status["freshness_status"],
        "fresh"
    );

    let mut second = scene();
    advance(&mut second, 2, "frame-2");
    assert_eq!(
        store.ingest(&second, 60, 60, None, None).unwrap(),
        IngestOutcome::LocalizationUpdated
    );
    assert_eq!(
        store.current(70).unwrap().unwrap().localization_age_ns,
        Some(10)
    );
    assert_eq!(store.stats().accepted_localizations, 2);
    assert!(matches!(
        store.activate_source("engine-a", "camera-session-a"),
        Err(StateLifecycleError::SourceAlreadyActive)
    ));
}

#[test]
fn stale_state_is_marked_cached_then_expired_without_old_coordinates() {
    let mut store = lifecycle();
    activate(&mut store);
    let observation = scene();
    store.ingest(&observation, 10, 10, None, None).unwrap();

    let stale = store.current(110).unwrap().unwrap();
    assert_eq!(stale.freshness, StateFreshness::Stale);
    assert_eq!(
        stale.payload.observation_status["freshness_status"],
        "stale"
    );
    assert_eq!(
        stale.payload.observation_status["ball_positions_status"],
        "cached"
    );
    assert!(stale
        .payload
        .observation_status
        .get("source_timestamp")
        .is_none());
    assert!(stale.payload.balls.iter().all(|ball| {
        !ball.is_observed && ball.confidence == 0.0 && ball.last_observation_confidence.is_some()
    }));

    let expired = store.current(510).unwrap().unwrap();
    assert_eq!(expired.freshness, StateFreshness::Expired);
    assert!(expired.payload.balls.is_empty());
    assert_eq!(
        expired.payload.observation_status["ball_positions_status"],
        "unavailable"
    );
    assert!(expired.ball_age_ns.is_empty());
    assert!(store
        .current(600)
        .unwrap()
        .unwrap()
        .payload
        .balls
        .is_empty());
    assert_eq!(store.stats().stale_transitions, 1);
    assert_eq!(store.stats().expired_transitions, 1);
}

#[test]
fn receiver_age_offsets_preserve_upstream_freshness_without_comparing_clocks() {
    let mut store = lifecycle();
    activate(&mut store);
    let observation = scene();
    let mut ball_age_offsets = std::collections::HashMap::new();
    ball_age_offsets.insert(observation.balls[0].track_id.clone(), 70);

    store
        .ingest_with_source_metadata(
            &observation,
            10,
            10,
            SourceStateMetadata {
                localization_age_offset_ns: 90,
                ball_age_offsets_ns: ball_age_offsets,
                ..SourceStateMetadata::default()
            },
        )
        .unwrap();

    let current = store.current(20).unwrap().unwrap();
    assert_eq!(current.localization_age_ns, Some(100));
    assert_eq!(current.freshness, StateFreshness::Stale);
    assert_eq!(
        current.ball_age_ns.get(&observation.balls[0].track_id),
        Some(&80)
    );
    assert!(current.payload.balls.iter().all(|ball| !ball.is_observed));
}

#[test]
fn rejects_duplicate_and_out_of_order_observations_but_accepts_sequence_gaps() {
    let mut store = lifecycle();
    activate(&mut store);
    let first = scene();
    store.ingest(&first, 10, 10, None, None).unwrap();

    let mut duplicate = first.clone();
    duplicate.transport_sequence = 2;
    duplicate.snapshot_revision = 2;
    assert!(matches!(
        store.ingest(&duplicate, 20, 20, None, None),
        Err(StateLifecycleError::DuplicateObservation)
    ));

    let mut older = scene();
    older.transport_sequence = 0;
    older.snapshot_revision = 0;
    assert!(matches!(
        store.ingest(&older, 30, 30, None, None),
        Err(StateLifecycleError::OutOfOrder)
    ));
    assert_eq!(store.current(30).unwrap().unwrap().updated_monotonic_ns, 10);

    let mut next = scene();
    advance(&mut next, 3, "frame-3");
    assert_eq!(
        store.ingest(&next, 40, 40, None, None).unwrap(),
        IngestOutcome::LocalizationUpdated
    );
    assert_eq!(store.current(40).unwrap().unwrap().updated_monotonic_ns, 40);
}

#[test]
fn health_and_annotation_do_not_refresh_localization_and_restart_clears_old_state() {
    let mut store = lifecycle();
    activate(&mut store);
    let first = scene();
    store.ingest(&first, 10, 10, None, None).unwrap();

    let mut health = scene();
    advance(&mut health, 2, "health-2");
    health.update_kind = UpdateKind::Health;
    health.outcome = Outcome::Failed;
    health.reason = Some("capture_interrupted".into());
    assert_eq!(
        store.ingest(&health, 30, 30, None, None).unwrap(),
        IngestOutcome::HealthUpdated
    );
    let stale = store.current(31).unwrap().unwrap();
    assert_eq!(stale.localization_received_monotonic_ns, Some(10));
    assert_eq!(stale.freshness, StateFreshness::Stale);
    assert_eq!(
        stale.payload.observation_status["health_reason"],
        "capture_interrupted"
    );
    assert_eq!(
        stale.payload.observation_status["ball_detection_status"],
        "failed"
    );

    let mut annotation = scene();
    advance(&mut annotation, 3, "annotation-3");
    annotation.update_kind = UpdateKind::Annotation;
    assert_eq!(
        store.ingest(&annotation, 40, 40, None, None).unwrap(),
        IngestOutcome::AnnotationIgnored
    );
    assert_eq!(
        store
            .current(41)
            .unwrap()
            .unwrap()
            .localization_received_monotonic_ns,
        Some(10)
    );

    let new_epoch = store
        .activate_source("engine-b", "camera-session-b")
        .unwrap();
    assert_eq!(new_epoch, 2);
    assert!(store.current(50).unwrap().is_none());
    assert!(matches!(
        store.ingest(&first, 50, 50, None, None),
        Err(StateLifecycleError::UnexpectedIdentity)
    ));
    let mut restarted = scene();
    restarted.engine_instance_id = "engine-b".into();
    restarted.source.session_id = "camera-session-b".into();
    restarted.balls.clear();
    assert_eq!(
        store.ingest(&restarted, 60, 60, None, None).unwrap(),
        IngestOutcome::LocalizationUpdated
    );
    assert!(store.current(60).unwrap().unwrap().payload.balls.is_empty());
    assert_eq!(store.stats().accepted_health_updates, 1);
    assert_eq!(store.stats().ignored_annotations, 1);
    assert_eq!(store.stats().rejected_identity, 1);
}

#[test]
fn health_before_first_localization_does_not_fabricate_a_localization_receipt() {
    let mut store = lifecycle();
    activate(&mut store);
    let mut health = scene();
    health.update_kind = UpdateKind::Health;
    health.outcome = Outcome::Skipped;
    health.reason = Some("no_frame_available".into());
    assert_eq!(
        store.ingest(&health, 20, 20, None, None).unwrap(),
        IngestOutcome::HealthUpdated
    );
    let state = store.current(30).unwrap().unwrap();
    assert_eq!(state.freshness, StateFreshness::Stale);
    assert_eq!(state.localization_received_monotonic_ns, None);
    assert_eq!(state.localization_age_ns, None);
    assert_eq!(
        state.payload.observation_status["ball_positions_status"],
        "unavailable"
    );
}

#[test]
fn rejects_invalid_clock_policy_and_ball_retention_without_poisoning_sequence() {
    let invalid = StateLifecycleConfig {
        stale_after_ns: 0,
        expire_after_ns: 0,
        maximum_balls: 0,
    };
    assert!(SceneStateLifecycle::new(invalid, PublicStateConfig::default()).is_err());

    let mut store = lifecycle();
    activate(&mut store);
    let observation = scene();
    assert!(matches!(
        store.ingest(&observation, 20, 10, None, None),
        Err(StateLifecycleError::ClockMovedBackwards)
    ));
    let mut too_many = scene();
    let original = too_many.balls[0].clone();
    for index in 0..16 {
        let mut ball = original.clone();
        ball.track_id = format!("extra-{index}");
        too_many.balls.push(ball);
    }
    assert!(matches!(
        store.ingest(&too_many, 10, 10, None, None),
        Err(StateLifecycleError::BallLimitExceeded { .. })
    ));
    // The same source sequence can be retried after correcting the oversized
    // event because failed application did not commit its watermarks.
    let corrected = scene();
    assert_eq!(
        store.ingest(&corrected, 10, 10, None, None).unwrap(),
        IngestOutcome::LocalizationUpdated
    );

    let mut incompatible = scene();
    advance(&mut incompatible, 2, "incompatible");
    incompatible.balls[0]
        .measurement
        .as_mut()
        .unwrap()
        .image_space = billiards_perception::ImageSpace::Rectified;
    assert!(store.ingest(&incompatible, 20, 20, None, None).is_err());
    let mut corrected = scene();
    advance(&mut corrected, 2, "frame-2");
    assert_eq!(
        store.ingest(&corrected, 20, 20, None, None).unwrap(),
        IngestOutcome::LocalizationUpdated
    );
}

#[test]
fn default_lifecycle_policy_is_versioned_and_loaded() {
    let policy: Value = serde_json::from_slice(include_bytes!(
        "../contracts/public-state/lifecycle-policy-v1.json"
    ))
    .unwrap();
    assert_eq!(
        policy["stale_after_ns"].as_u64().unwrap() as u128,
        StateLifecycleConfig::default().stale_after_ns
    );
    assert_eq!(
        policy["expire_after_ns"].as_u64().unwrap() as u128,
        StateLifecycleConfig::default().expire_after_ns
    );
    assert_eq!(StateLifecycleConfig::default().maximum_balls, 64);
}

#[test]
fn native_contour_is_available_only_while_the_source_observation_is_fresh() {
    use billiards_perception::Point2d;
    let mut store = lifecycle();
    activate(&mut store);
    let mut first = scene();
    let m = first.balls[0].measurement.as_mut().unwrap();
    let c = m.image_center_px;
    m.image_outline_px = Some(vec![
        Point2d {
            x: c.x - 2.,
            y: c.y,
        },
        Point2d {
            x: c.x,
            y: c.y - 1.,
        },
        Point2d {
            x: c.x + 2.,
            y: c.y,
        },
        Point2d {
            x: c.x,
            y: c.y + 1.,
        },
    ]);
    store.ingest(&first, 10, 10, None, None).unwrap();
    assert!(store.current(50).unwrap().unwrap().payload.balls[0]
        .outline
        .is_some());
    let stale = store.current(110).unwrap().unwrap();
    assert!(stale
        .payload
        .balls
        .iter()
        .all(|b| !b.is_observed && b.outline.is_none()));
    assert!(store
        .current(510)
        .unwrap()
        .unwrap()
        .payload
        .balls
        .is_empty());
}
