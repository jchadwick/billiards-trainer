//! Run actual saved detector proposals through the production refiner without
//! trace overhead. Native RGB is decoded once per scene and borrowed by every
//! batch. Development-host timings do not qualify target-device performance.
use billiards_perception::ball_localization::{
    refine_ball_batch, BallLocalizationConfig, BallProposal, CameraIntrinsics,
};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{fs, time::Instant};
fn main() {
    let input_path = "thoughts/calibration/actual-detector-replay-20261028/results.json";
    let raw = fs::read(input_path).unwrap();
    let source: Value = serde_json::from_slice(&raw).unwrap();
    let optics: Value = serde_json::from_slice(
        &fs::read("thoughts/calibration/checkerboard-eight-view-baseline.json").unwrap(),
    )
    .unwrap();
    let matrix = &optics["models"]["zero"]["full_fit"]["camera_matrix"];
    let mut scenes = Vec::new();
    for scene in source["scenes"].as_array().unwrap() {
        let image = image::open(scene["source_path"].as_str().unwrap())
            .unwrap()
            .to_rgb8();
        let scale = f64::from(image.width()) / 1280.;
        let k = CameraIntrinsics {
            source_width_px: image.width(),
            source_height_px: image.height(),
            fx: matrix[0][0].as_f64().unwrap() * scale,
            fy: matrix[1][1].as_f64().unwrap() * scale,
            cx: matrix[0][2].as_f64().unwrap() * scale,
            cy: matrix[1][2].as_f64().unwrap() * scale,
        };
        let proposals: Vec<_> = scene["detections"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|d| d["class_name"] == "ball")
            .map(|d| BallProposal {
                detection_index: d["index"].as_u64().unwrap(),
                bbox: std::array::from_fn(|i| d["bbox_xyxy_source_px"][i].as_f64().unwrap()),
            })
            .collect();
        let mut profiles = Vec::new();
        for (label, cfg) in [(
            "geometry_diagnostic",
            BallLocalizationConfig {
                batch_budget_ms: 10_000,
                proposal_budget_ms: 10_000,
                ..Default::default()
            },
        )] {
            for _ in 0..2 {
                refine_ball_batch(image.as_raw(), &k, &proposals, &cfg).unwrap();
            }
            let mut runs = Vec::new();
            for _ in 0..12 {
                let start = Instant::now();
                let result = refine_ball_batch(image.as_raw(), &k, &proposals, &cfg).unwrap();
                let elapsed = start.elapsed().as_secs_f64() * 1000.;
                let results = serde_json::to_value(result).unwrap();
                runs.push(json!({"elapsed_ms":elapsed,"observations":results}));
            }
            let mut times: Vec<_> = runs
                .iter()
                .map(|r| r["elapsed_ms"].as_f64().unwrap())
                .collect();
            times.sort_by(f64::total_cmp);
            profiles.push(json!({"label":label,"config":cfg,"median_ms":(times[5]+times[6])/2.,"p95_ms":times[11],"runs":runs}));
        }
        scenes.push(json!({"scene_id":scene["id"],"source_sha256":scene["source_sha256"],"proposal_count":proposals.len(),"profiles":profiles}));
    }
    let report = json!({"scope":"Production no-trace batch replay on development host; no camera or independent physical accuracy claim","refiner_revision":billiards_perception::ball_localization_profile::BALL_REFINER_REVISION,"release_build":!cfg!(debug_assertions),"proposal_origin":"frozen source-checkpoint PyTorch CPU detector outputs; not backend parity", "refiner_source_sha256":format!("{:x}",Sha256::digest(fs::read("perception/src/ball_localization.rs").unwrap())), "actual_proposals_sha256":format!("{:x}",Sha256::digest(raw)),"scenes":scenes});
    let output = std::env::args()
        .nth(1)
        .expect("explicit report output path");
    fs::write(output, serde_json::to_vec_pretty(&report).unwrap()).unwrap();
}
