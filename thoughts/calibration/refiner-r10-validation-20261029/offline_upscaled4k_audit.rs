//! Re-run the production refiner on the exact saved 4K replay pixels.
//!
//! The image is an upscaled saved 720p frame, so this is a pipeline regression
//! check, not native-4K accuracy evidence. Frozen detector boxes and optics are
//! scaled by exactly 3x to preserve their source-image coordinate convention.

use billiards_perception::ball_localization::{
    refine_ball_batch, BallLocalizationConfig, BallProposal, CameraIntrinsics,
};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::fs;

const CORPUS_PATH: &str = "thoughts/calibration/target-gpu-saved-20261029/corpus.json";
const SOURCE_IMAGE_PATH: &str =
    "thoughts/calibration/refiner-r10-validation-20261029/saved-720-upscaled-to-3840.jpg";
const SOURCE_RGB_PATH: &str = "/tmp/r10-saved-4k-vaapi-rgb.raw";
const CONFIG_PATH: &str = "/tmp/billiards-gpu-package-20261029/distribution-r10/config-offline.json";
const REFINER_PATH: &str = "perception/src/ball_localization.rs";
const HARNESS_PATH: &str =
    "thoughts/calibration/refiner-r10-validation-20261029/offline_upscaled4k_audit.rs";
const OUTPUT_PATH: &str = "/tmp/billiards-r10-upscaled4k-audit.json";

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn finite(value: &Value) -> f64 {
    value.as_f64().filter(|value| value.is_finite()).expect("finite number")
}

fn main() {
    let image_bytes = fs::read(SOURCE_RGB_PATH).expect("read target vaapi RGB frame");
    assert_eq!(image_bytes.len(), 3840 * 2160 * 3);
    let image = image::RgbImage::from_raw(3840, 2160, image_bytes.clone())
        .expect("build RGB image from target vaapi output");
    assert_eq!(image.dimensions(), (3840, 2160));

    let corpus_bytes = fs::read(CORPUS_PATH).expect("read frozen detector corpus");
    let corpus: Value = serde_json::from_slice(&corpus_bytes).expect("parse corpus");
    let measured = corpus["images"]
        .as_array()
        .expect("images")
        .iter()
        .find(|entry| entry["id"] == "measured-grid-720")
        .expect("measured grid image");
    let proposals = measured["ball_proposals"]["proposals"]
        .as_array()
        .expect("proposals");
    assert_eq!(proposals.len(), 15);

    let proposal_records: Vec<Value> = proposals
        .iter()
        .map(|proposal| {
            let source_box = proposal["bbox_xyxy_source_px"]
                .as_array()
                .expect("box");
            let bbox = [
                finite(&source_box[0]) * 3.0,
                finite(&source_box[1]) * 3.0,
                finite(&source_box[2]) * 3.0,
                finite(&source_box[3]) * 3.0,
            ];
            json!({
                "ball_id": proposal["ball_id"],
                "detection_index": proposal["detection_index_in_frozen_audit"],
                "bbox_xyxy_source_px": bbox,
                "source_proposal_sha256": sha256(
                    serde_json::to_vec(proposal).expect("serialize source proposal").as_slice()
                )
            })
        })
        .collect();
    let proposals: Vec<BallProposal> = proposal_records
        .iter()
        .map(|record| {
            let bbox: [f64; 4] = serde_json::from_value(
                record["bbox_xyxy_source_px"].clone(),
            )
            .expect("scaled proposal box");
            BallProposal {
                detection_index: record["detection_index"].as_u64().expect("detection index"),
                bbox,
            }
        })
        .collect();

    let matrix = &corpus["optics_models"]["frozen_zero_radial_720"]["camera_matrix"];
    let intrinsics = CameraIntrinsics {
        source_width_px: 3840,
        source_height_px: 2160,
        fx: finite(&matrix[0][0]) * 3.0,
        fy: finite(&matrix[1][1]) * 3.0,
        cx: finite(&matrix[0][2]) * 3.0,
        cy: finite(&matrix[1][2]) * 3.0,
    };
    let runtime_config: Value =
        serde_json::from_slice(&fs::read(CONFIG_PATH).expect("read app qualification config"))
            .expect("parse app qualification config");
    let config: BallLocalizationConfig = serde_json::from_value(
        runtime_config["perception"]["rust_live"]["ball_localization"]["parameters"]
            .clone(),
    )
    .expect("deserialize exact application localization settings");
    let production_batch_budget_ms = config.batch_budget_ms;
    let production_proposal_budget_ms = config.proposal_budget_ms;
    // Isolate source-pixel and fit-selection behavior from diagnostic-host
    // scheduling. Production values are retained in the report.
    let config = BallLocalizationConfig {
        batch_budget_ms: 10_000,
        proposal_budget_ms: 10_000,
        ..config
    };
    let started = std::time::Instant::now();
    let refinements = refine_ball_batch(
        image.as_raw(),
        &intrinsics,
        &proposals,
        &config,
    )
    .expect("run production estimator");
    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;

    let results: Vec<Value> = refinements
        .iter()
        .zip(proposal_records.iter())
        .map(|(result, proposal)| {
            json!({
                "ball_id": proposal["ball_id"],
                "detection_index": proposal["detection_index"],
                "status": result.status,
                "reason": result.reason,
                "center_image_px": result.center_image_px,
                "quality": result.quality,
            })
        })
        .collect();
    let output = json!({
        "scope": "upscaled saved-image refiner regression only; not native-4K accuracy evidence",
        "source_image_path": SOURCE_IMAGE_PATH,
        "source_rgb8_sha256": sha256(&image_bytes),
        "source_jpeg_sha256": sha256(
            &fs::read(SOURCE_IMAGE_PATH).expect("saved JPEG provenance input")
        ),
        "source_image_dimensions_px": [image.width(), image.height()],
        "frozen_720_corpus_sha256": sha256(&corpus_bytes),
        "refiner_source_sha256": sha256(&fs::read(REFINER_PATH).expect("refiner source")),
        "harness_sha256": sha256(&fs::read(HARNESS_PATH).expect("harness source")),
        "refiner_revision": billiards_perception::ball_localization_profile::BALL_REFINER_REVISION,
        "config_sha256": sha256(&fs::read(CONFIG_PATH).expect("app config")),
        "batch_budget_ms": config.batch_budget_ms,
        "proposal_budget_ms": config.proposal_budget_ms,
        "production_batch_budget_ms": production_batch_budget_ms,
        "production_proposal_budget_ms": production_proposal_budget_ms,
        "batch_elapsed_ms_diagnostic_host": elapsed_ms,
        "accepted_count": results.iter().filter(|result| result["status"] == "accepted").count(),
        "proposal_count": results.len(),
        "results": results,
        "physical_accuracy_claim": false
    });
    fs::write(OUTPUT_PATH, serde_json::to_vec_pretty(&output).expect("serialize output"))
        .expect("write output");
    println!("{}", serde_json::to_string_pretty(&output).expect("print output"));
}
