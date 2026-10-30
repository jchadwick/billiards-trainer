use billiards_perception::ball_localization::{
    refine_ball_batch, BallLocalizationConfig, BallProposal, CameraIntrinsics,
};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::fs;

fn sha256(path: &str) -> String {
    sha256_bytes(&fs::read(path).expect("read hash input"))
}

fn sha256_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn main() {
    let corpus_path = "thoughts/calibration/native-localization-corpus-v1.json";
    let baseline_path = "thoughts/calibration/offline-refiner-identity-check.json";
    let source_path = "thoughts/calibration/table-grid-20261026-114241.jpg";
    let refiner_path = "perception/src/ball_localization.rs";
    let harness_path = "thoughts/calibration/offline-refiner-identity-audit-20261027.rs";
    let corpus: Value =
        serde_json::from_slice(&fs::read(corpus_path).expect("read corpus")).expect("parse corpus");
    let baseline: Value =
        serde_json::from_slice(&fs::read(baseline_path).expect("read original audit"))
            .expect("parse original audit");

    let image_record = corpus["images"]
        .as_array()
        .expect("image records")
        .iter()
        .find(|record| record["id"] == "measured-grid-720")
        .expect("frozen measured image record");
    let optics = &corpus["optics_models"]["frozen_zero_radial_720"]["camera_matrix"];
    let source_image = image::open(source_path)
        .expect("read frozen measured image")
        .to_rgb8();
    let intrinsics = CameraIntrinsics {
        source_width_px: source_image.width(),
        source_height_px: source_image.height(),
        fx: optics[0][0].as_f64().expect("fx"),
        fy: optics[1][1].as_f64().expect("fy"),
        cx: optics[0][2].as_f64().expect("cx"),
        cy: optics[1][2].as_f64().expect("cy"),
    };
    let config = BallLocalizationConfig {
        batch_budget_ms: 5_000,
        proposal_budget_ms: 400,
        ..BallLocalizationConfig::default()
    };

    let baseline_results = baseline["results"]
        .as_array()
        .expect("old identity results");
    let mut results = Vec::new();
    for frozen in image_record["ball_proposals"]["proposals"]
        .as_array()
        .expect("frozen manual proposals")
    {
        let bbox = frozen["bbox_xyxy_source_px"]
            .as_array()
            .expect("proposal bbox")
            .iter()
            .map(|value| value.as_f64().expect("finite proposal coordinate"))
            .collect::<Vec<_>>();
        let template = BallProposal {
            detection_index: 0,
            bbox: [bbox[0], bbox[1], bbox[2], bbox[3]],
        };
        let mut outcomes = Vec::with_capacity(16);
        for detection_index in 0..16 {
            let proposal = BallProposal {
                detection_index,
                ..template.clone()
            };
            let result = refine_ball_batch(
                source_image.as_raw(),
                &intrinsics,
                std::slice::from_ref(&proposal),
                &config,
            )
            .expect("production estimator result")
            .into_iter()
            .next()
            .expect("one result per proposal");
            outcomes.push(serde_json::to_value(result).expect("serialize result"));
        }

        let normalized = outcomes
            .iter()
            .map(|outcome| {
                let mut semantic = outcome.clone();
                semantic
                    .as_object_mut()
                    .expect("serialized refinement object")
                    .remove("detection_index");
                semantic
            })
            .collect::<Vec<_>>();
        let invariant = normalized.iter().all(|value| value == &normalized[0]);
        let outcome_fingerprints = normalized
            .iter()
            .enumerate()
            .map(|(detection_index, semantic)| {
                let encoded = serde_json::to_vec(semantic).expect("encode semantic result");
                json!({
                    "detection_index": detection_index,
                    "semantic_output_sha256": sha256_bytes(&encoded),
                    "status": semantic["status"],
                    "reason": semantic["reason"],
                    "center_image_px": semantic["center_image_px"]
                })
            })
            .collect::<Vec<_>>();
        let accepted_indices = outcomes
            .iter()
            .filter(|outcome| outcome["center_image_px"].is_array())
            .map(|outcome| outcome["detection_index"].as_u64().expect("index"))
            .collect::<Vec<_>>();
        let baseline_ball = baseline_results
            .iter()
            .find(|record| record["id"] == frozen["ball_id"])
            .expect("matching baseline ball");
        let baseline_accepted = baseline_ball["accepted_indices"].as_array().unwrap();
        let baseline_accepted = baseline_accepted
            .iter()
            .map(|value| value.as_u64().unwrap())
            .collect::<Vec<_>>();

        results.push(json!({
            "ball_id": frozen["ball_id"],
            "physical_reference_id": frozen["physical_reference_id"],
            "physical_reference_present": frozen["physical_reference_present"],
            "proposal_origin": frozen["origin"],
            "bbox_xyxy_source_px": frozen["bbox_xyxy_source_px"],
            "same_semantic_result_for_indices_0_through_15": invariant,
            "accepted_indices": accepted_indices,
            "baseline_accepted_indices": baseline_accepted,
            "acceptance_set_changed_from_baseline": baseline_accepted != accepted_indices,
            "outcome_fingerprints": outcome_fingerprints,
            "representative_outcome": outcomes[0],
        }));
    }

    let measured_changed = results
        .iter()
        .filter(|result| result["physical_reference_present"] == true)
        .filter(|result| result["acceptance_set_changed_from_baseline"] == true)
        .count();
    let report = json!({
        "schema_version": 1,
        "experiment": "detection-identity invariance on frozen saved image and manual proposals",
        "date": "2026-10-27",
        "not_detector_output": true,
        "not_physical_accuracy_validation": true,
        "input_image": {
            "path": source_path,
            "sha256": sha256(source_path),
            "dimensions_px": [source_image.width(), source_image.height()],
            "corpus_image_record": image_record,
        },
        "corpus": {"path": corpus_path, "sha256": sha256(corpus_path)},
        "original_identity_baseline": {"path": baseline_path, "sha256": sha256(baseline_path)},
        "refiner_source": {"path": refiner_path, "sha256": sha256(refiner_path)},
        "experiment_harness": {"path": harness_path, "sha256": sha256(harness_path)},
        "intrinsics": intrinsics,
        "runtime_parameters": config,
        "detection_indices_tested": {"first": 0, "last": 15, "count": 16},
        "all_fifteen_manual_proposals_invariant": results.iter().all(|result| result["same_semantic_result_for_indices_0_through_15"] == true),
        "measured_proposals_with_acceptance_set_changed_from_baseline": measured_changed,
        "measured_proposal_count": results.iter().filter(|result| result["physical_reference_present"] == true).count(),
        "limitations": [
            "This tests repeated runs on one frozen still, not independent placements.",
            "Proposals are manual annotation-derived boxes, not detector output.",
            "Identity invariance does not establish that a chosen image boundary is the true ball center.",
            "Acceptance changes compare different estimator source revisions; they are not a direct accuracy comparison."
        ],
        "results": results,
    });
    println!(
        "{}",
        serde_json::to_string_pretty(&report).expect("serialize report")
    );
}
