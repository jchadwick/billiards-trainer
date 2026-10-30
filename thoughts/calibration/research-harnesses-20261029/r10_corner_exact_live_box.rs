//! Evaluate the exact coarse box reported for the unavailable pocket ball.
//!
//! This uses the exact saved 720p scene upscaled to 4K and target-vaapi-decoded
//! pixels. The box is reconstructed from the R10 live application's public
//! coarse center and half-extents. It is a regression diagnostic, not physical
//! accuracy evidence.

use billiards_perception::ball_localization::{
    refine_ball_batch, BallLocalizationConfig, BallProposal, CameraIntrinsics,
};
use serde_json::{json, Value};
use std::fs;

const SOURCE_RGB_PATH: &str = "/tmp/r10-saved-4k-vaapi-rgb.raw";
const LIVE_STATE_PATH: &str = "thoughts/calibration/refiner-r10-validation-20261029/target-saved-live-r10-scheduler-fix-last-state.json";
const OPTICS_PATH: &str = "/tmp/billiards-gpu-package-20261029/distribution-r10/backend/calibration_data/ball-localization/offline-optics-r10.json";
const CONFIG_PATH: &str =
    "/tmp/billiards-gpu-package-20261029/distribution-r10/config-offline.json";

fn main() {
    let source_rgb = fs::read(SOURCE_RGB_PATH).expect("read saved target-vaapi RGB pixels");
    assert_eq!(source_rgb.len(), 3840 * 2160 * 3);

    let live_state: Value =
        serde_json::from_slice(&fs::read(LIVE_STATE_PATH).expect("read final saved-live state"))
            .expect("parse saved-live state");
    let missing_ball = live_state["balls"]
        .as_array()
        .expect("balls array")
        .iter()
        .find(|ball| ball["table_position"].is_null())
        .expect("one unavailable ball");
    let center_x = missing_ball["position"]["x"].as_f64().expect("center x");
    let center_y = missing_ball["position"]["y"].as_f64().expect("center y");
    let half_width = missing_ball["radius_extent"]["x"]
        .as_f64()
        .expect("coarse half width");
    let half_height = missing_ball["radius_extent"]["y"]
        .as_f64()
        .expect("coarse half height");
    let proposal = BallProposal {
        detection_index: 1,
        bbox: [
            center_x - half_width,
            center_y - half_height,
            center_x + half_width,
            center_y + half_height,
        ],
    };

    let optics: Value =
        serde_json::from_slice(&fs::read(OPTICS_PATH).expect("read optics")).expect("parse optics");
    let intrinsics: CameraIntrinsics =
        serde_json::from_value(optics["intrinsics"].clone()).expect("deserialize intrinsics");
    let runtime_config: Value =
        serde_json::from_slice(&fs::read(CONFIG_PATH).expect("read runtime configuration"))
            .expect("parse runtime configuration");
    let config: BallLocalizationConfig = serde_json::from_value(
        runtime_config["perception"]["rust_live"]["ball_localization"]["parameters"].clone(),
    )
    .expect("deserialize production refiner settings");
    let production_batch_budget_ms = config.batch_budget_ms;
    let production_proposal_budget_ms = config.proposal_budget_ms;
    let observations: Vec<Value> = [1.5, 1.6, 1.75]
        .into_iter()
        .map(|maximum_center_spread_px| {
            let diagnostic_config = BallLocalizationConfig {
                batch_budget_ms: 10_000,
                proposal_budget_ms: 10_000,
                maximum_center_spread_px,
                ..config.clone()
            };
            let result = refine_ball_batch(
                &source_rgb,
                &intrinsics,
                std::slice::from_ref(&proposal),
                &diagnostic_config,
            )
            .expect("run exact production refiner");
            assert_eq!(result.len(), 1);
            let observation = &result[0];
            json!({
                "maximum_center_spread_px": maximum_center_spread_px,
                "batch_budget_ms": diagnostic_config.batch_budget_ms,
                "proposal_budget_ms": diagnostic_config.proposal_budget_ms,
                "status": observation.status,
                "reason": observation.reason,
                "center_image_px": observation.center_image_px,
                "quality": observation.quality,
            })
        })
        .collect();
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "scope": "exact live coarse-box regression on upscaled saved imagery; no physical accuracy claim",
            "ball_id": missing_ball["id"],
            "confidence": missing_ball["confidence"],
            "center_from_live_state_px": [center_x, center_y],
            "half_extent_from_live_state_px": [half_width, half_height],
            "reconstructed_live_box_xyxy_px": proposal.bbox,
            "detection_index": proposal.detection_index,
            "optics_profile_id": optics["optics_profile_id"],
            "refiner_revision": billiards_perception::ball_localization_profile::BALL_REFINER_REVISION,
            "production_batch_budget_ms": production_batch_budget_ms,
            "production_proposal_budget_ms": production_proposal_budget_ms,
            "diagnostic_budget_ms": 10_000,
            "threshold_observations": observations,
        }))
        .expect("serialize result")
    );
}
