use billiards_perception::ball_localization::{
    refine_ball_batch, BallLocalizationConfig, BallProposal, CameraIntrinsics,
};

fn main() {
    let source = image::open("thoughts/calibration/table-grid-20261026-114241.jpg")
        .expect("read measured 15-ball source image")
        .to_rgb8();
    let annotations: serde_json::Value = serde_json::from_slice(
        &std::fs::read("thoughts/calibration/ball-outline-consistency.json")
            .expect("read saved source annotations"),
    )
    .expect("parse saved source annotations");
    let balls = annotations["balls"].as_array().expect("annotated balls");
    let mut proposals = Vec::with_capacity(balls.len());
    for (index, ball) in balls.iter().enumerate() {
        let center = ball["segmentation_bbox_midpoint_px"]
            .as_array()
            .expect("coarse center");
        let gradient_diameters = ball["gradient_ellipse"]["diameters_px"]
            .as_array()
            .expect("gradient extent");
        let side = gradient_diameters
            .iter()
            .map(|value| value.as_f64().expect("finite diameter"))
            .fold(0.0, f64::max)
            + 4.0;
        let x = center[0].as_f64().expect("center x");
        let y = center[1].as_f64().expect("center y");
        proposals.push(BallProposal {
            detection_index: index as u64,
            bbox: [x - side / 2.0, y - side / 2.0, x + side / 2.0, y + side / 2.0],
        });
    }
    let intrinsics = CameraIntrinsics {
        source_width_px: source.width(),
        source_height_px: source.height(),
        fx: 696.1731580029823,
        fy: 695.3611772093032,
        cx: 644.1755624688407,
        cy: 349.67152391690104,
    };
    let config = BallLocalizationConfig {
        batch_budget_ms: 5_000,
        proposal_budget_ms: 400,
        ..BallLocalizationConfig::default()
    };
    eprintln!("AUDIT_METADATA:{}", serde_json::json!({"intrinsics": &intrinsics, "config": &config}));
    for (ball, original) in balls.iter().zip(proposals) {
        let mut accepted = Vec::new();
        let mut reasons = std::collections::BTreeMap::<String, usize>::new();
        for detection_index in 0..16 {
            let proposal = BallProposal { detection_index, ..original.clone() };
            let result = refine_ball_batch(source.as_raw(), &intrinsics, &[proposal], &config)
                .expect("refine identical crop")
                .remove(0);
            if let Some(center) = result.center_image_px {
                accepted.push((detection_index, center));
            } else {
                *reasons.entry(format!("{:?}", result.reason)).or_default() += 1;
            }
        }
        let mut spread = 0.0_f64;
        for (_, a) in &accepted {
            for (_, b) in &accepted {
                spread = spread.max((a[0] - b[0]).hypot(a[1] - b[1]));
            }
        }
        println!("{}", serde_json::json!({
            "id": ball["id"],
            "physical_reference": ball["physical_ground_truth_available"],
            "accepted_of_16": accepted.len(),
            "accepted_indices": accepted.iter().map(|(index, _)| index).collect::<Vec<_>>(),
            "maximum_accepted_center_separation_px": spread,
            "rejections": reasons,
        }));
    }
}
