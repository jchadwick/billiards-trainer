use billiards_perception::{
    decode_scene_snapshot_v1,
    public_state::{PublicStateConfig, SceneSnapshotStateAdapter},
    ImageSpace, TimestampStage,
};
use serde_json::{json, Value};
use std::{fs, path::Path};

fn base() -> billiards_perception::SceneSnapshotV1 {
    decode_scene_snapshot_v1(include_bytes!(
        "../contracts/v1/scene-rust-tracked-success.json"
    ))
    .unwrap()
}
fn adapter() -> SceneSnapshotStateAdapter {
    SceneSnapshotStateAdapter::new(PublicStateConfig::default()).unwrap()
}

#[test]
fn matches_checked_in_contract_fixtures() {
    let directory = Path::new(env!("CARGO_MANIFEST_DIR")).join("contracts/public-state");
    let mut count = 0;
    for entry in fs::read_dir(directory).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().and_then(|v| v.to_str()) != Some("json")
            || path
                .file_name()
                .unwrap()
                .to_string_lossy()
                .ends_with("policy-v1.json")
            || path
                .file_name()
                .unwrap()
                .to_string_lossy()
                .ends_with(".schema.json")
        {
            continue;
        }
        let fixture: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        let scene =
            decode_scene_snapshot_v1(&serde_json::to_vec(&fixture["scene"]).unwrap()).unwrap();
        let payload = adapter()
            .adapt(
                &scene,
                Some(fixture["table"].clone()),
                Some(fixture["game_type"].as_str().unwrap().into()),
            )
            .unwrap();
        assert_eq!(
            serde_json::to_value(payload).unwrap(),
            fixture["expected"],
            "fixture {}",
            path.display()
        );
        count += 1;
    }
    assert_eq!(count, 35, "all checked-in contract fixtures must run");
}

#[test]
fn rejects_invalid_and_incompatible_geometry_without_a_fresh_payload() {
    let mut scene = base();
    scene.balls[0].measurement.as_mut().unwrap().image_space = ImageSpace::Rectified;
    assert!(adapter().adapt(&scene, None, None).is_err());
    scene = base();
    scene.source.width_px = 1920;
    assert!(adapter().adapt(&scene, None, None).is_err());
    scene = base();
    scene.balls[0]
        .measurement
        .as_mut()
        .unwrap()
        .image_center_px
        .x = f64::NAN;
    assert!(adapter().adapt(&scene, None, None).is_err());
    scene = base();
    scene.source.timestamp.stage = TimestampStage::InferenceComplete;
    assert!(adapter().adapt(&scene, None, None).is_err());
}

#[test]
fn calibrated_positions_cannot_escape_without_valid_revision_alignment() {
    let mut scene = base();
    scene.balls[0].measurement.as_mut().unwrap().table_center_mm =
        Some(billiards_perception::Point2d { x: 1., y: 2. });
    assert!(adapter().adapt(&scene, None, None).is_err());
    scene.revisions.calibration_state = billiards_perception::CalibrationState::Valid;
    assert!(adapter().adapt(&scene, None, None).is_err());
    scene.revisions.calibration = Some(12);
    let payload = adapter().adapt(&scene, None, None).unwrap().unwrap();
    assert_eq!(payload.observation_status["calibration_revision"], 12);
    assert_eq!(
        payload.balls[0].table_position.as_ref().unwrap().unit,
        "millimeters"
    );
    assert_eq!(payload.balls[0].table_radius_mm, Some(28.575));
    // The contract binds all table coordinates to snapshot.revisions; it has no
    // separate per-measurement revision that an adapter could independently verify.
}

#[test]
fn config_is_validated_and_changes_scaling_and_assumed_radius_explicitly() {
    for config in [
        PublicStateConfig {
            canonical_width_px: 0,
            ..Default::default()
        },
        PublicStateConfig {
            assumed_ball_radius_px: f64::INFINITY,
            ..Default::default()
        },
        PublicStateConfig {
            assumed_ball_radius_px: 0.,
            ..Default::default()
        },
        PublicStateConfig {
            assumed_ball_radius_mm: 0.,
            ..Default::default()
        },
    ] {
        assert!(SceneSnapshotStateAdapter::new(config).is_err());
    }
    let configured = SceneSnapshotStateAdapter::new(PublicStateConfig {
        canonical_width_px: 1920,
        canonical_height_px: 1080,
        assumed_ball_radius_px: 18.,
        ..Default::default()
    })
    .unwrap();
    let payload = configured.adapt(&base(), None, None).unwrap().unwrap();
    assert_eq!(payload.balls[0].position.x, 941.2);
    assert_eq!(payload.balls[0].radius, 18.);
    assert_eq!(
        payload.balls[0].radius_source,
        "legacy_standard_pool_ball_assumption"
    );
}

#[test]
fn maps_measured_ball_box_extent_to_the_canonical_overlay_plane() {
    let mut scene = base();
    scene.balls[0].measurement.as_mut().unwrap().image_radius_px =
        Some(billiards_perception::Point2d { x: 24.0, y: 21.0 });

    let payload = adapter().adapt(&scene, None, None).unwrap().unwrap();
    assert_eq!(
        payload.balls[0].radius_extent,
        Some(billiards_perception::public_state::PublicRadiusExtent {
            x: 24.0,
            y: 21.0,
            scale: [1.0, 1.0],
        }),
        "a source frame on the canonical plane keeps the detector half-extents"
    );
    assert_eq!(
        serde_json::to_value(&payload.balls[0]).unwrap()["radius_extent"],
        json!({ "x": 24.0, "y": 21.0, "scale": [1.0, 1.0] })
    );
}

#[test]
fn preserves_large_decimal_ticks_without_loss_before_subtraction() {
    let mut scene = base();
    scene.source.timestamp.ticks = "99999999999999999998".into();
    scene.published_at.ticks = "99999999999999999999".into();
    scene.published_at.domain_id = scene.source.timestamp.domain_id.clone();
    let payload = adapter().adapt(&scene, None, None).unwrap().unwrap();
    assert_eq!(
        payload.observation_status["frame_publication_age_seconds"],
        json!(1e-9)
    );
}

#[test]
fn repeated_conversion_is_pure_and_never_advances_old_ball_observation() {
    let fixture: Value = serde_json::from_slice(include_bytes!(
        "../contracts/public-state/occluded-extrapolated.json"
    ))
    .unwrap();
    let mut scene =
        decode_scene_snapshot_v1(&serde_json::to_vec(&fixture["scene"]).unwrap()).unwrap();
    let old = adapter().adapt(&scene, None, None).unwrap().unwrap();
    scene.source.frame_id = "later-frame".into();
    scene.snapshot_revision += 1;
    let later = adapter().adapt(&scene, None, None).unwrap().unwrap();
    assert_eq!(
        old.balls[0].source_observation,
        later.balls[0].source_observation
    );
    assert!(!later.balls[0].is_observed);
    assert_eq!(later.balls[0].confidence, 0.);
    assert_eq!(later.observation_status["ball_positions_status"], "cached");
    // Receipt age and duplicate suppression belong to the stateful publisher;
    // a pure compatibility conversion must not manufacture either.
}

#[test]
fn precise_outline_scales_with_canonical_geometry_and_keeps_clipped_vertices() {
    use billiards_perception::Point2d;
    let mut scene = base();
    let m = scene.balls[0].measurement.as_mut().unwrap();
    m.image_center_px = Point2d { x: 2., y: 2. };
    m.image_radius_px = Some(Point2d { x: 6., y: 3. });
    m.image_outline_px = Some(vec![
        Point2d { x: -4., y: 2. },
        Point2d { x: 2., y: -1. },
        Point2d { x: 8., y: 2. },
        Point2d { x: 2., y: 5. },
    ]);
    let scaled = SceneSnapshotStateAdapter::new(PublicStateConfig {
        canonical_width_px: 1920,
        canonical_height_px: 1080,
        ..Default::default()
    })
    .unwrap();
    let payload = scaled.adapt(&scene, None, None).unwrap().unwrap();
    let outline = payload.balls[0].outline.as_ref().unwrap();
    assert_eq!(outline.points[0], Point2d { x: -2., y: 1. });
    assert_eq!(outline.points[1], Point2d { x: 1., y: -0.5 });
    assert_eq!(outline.scale, [1., 1.]);
    assert_eq!(payload.balls[0].radius_extent.as_ref().unwrap().x, 3.);
    assert_eq!(payload.balls[0].radius_source, "native_refined_outline");
    scene.balls[0].status = billiards_perception::BallStatus::Lost;
    assert!(scaled.adapt(&scene, None, None).is_err());
    scene.balls[0]
        .measurement
        .as_mut()
        .unwrap()
        .image_outline_px = None;
    assert!(scaled.adapt(&scene, None, None).unwrap().unwrap().balls[0]
        .outline
        .is_none());
}

#[test]
fn invalid_or_stale_precise_contours_cannot_escape_the_scene_contract() {
    use billiards_perception::Point2d;
    for vertices in [
        vec![Point2d { x: f64::NAN, y: 0. }; 3],
        vec![Point2d { x: 1., y: 1. }; 3],
        vec![Point2d { x: 1., y: 1. }; 513],
        vec![Point2d { x: 1e30, y: 0. }; 3],
    ] {
        let mut scene = base();
        scene.balls[0]
            .measurement
            .as_mut()
            .unwrap()
            .image_outline_px = Some(vertices);
        assert!(adapter().adapt(&scene, None, None).is_err());
    }
    let mut scene = base();
    let c = scene.balls[0].measurement.as_ref().unwrap().image_center_px;
    scene.balls[0]
        .measurement
        .as_mut()
        .unwrap()
        .image_outline_px = Some(vec![
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
    scene.balls[0].last_observation.timestamp.ticks = "999999999".into();
    assert!(adapter().adapt(&scene, None, None).is_err());
}
