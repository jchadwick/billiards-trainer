use billiards_perception::ball_localization::*;
use serde_json::json;
fn intrinsics() -> CameraIntrinsics {
    CameraIntrinsics {
        source_width_px: 480,
        source_height_px: 300,
        fx: 260.0,
        fy: 273.0,
        cx: 240.0,
        cy: 150.0,
    }
}
fn config() -> BallLocalizationConfig {
    BallLocalizationConfig {
        reference_width_px: 480,
        batch_budget_ms: 2000,
        proposal_budget_ms: 1000,
        ..Default::default()
    }
}
/// Render a 2D contour by the same documented camera-ray coordinate rotation;
/// estimator receives only RGB and a coarse bounding proposal, never depth.
fn image(center: [f64; 2], striped: bool, shadow: bool) -> (Vec<u8>, BallProposal) {
    image_with_split_background(center, striped, shadow, false)
}

fn image_with_split_background(
    center: [f64; 2],
    striped: bool,
    shadow: bool,
    split: bool,
) -> (Vec<u8>, BallProposal) {
    let k = intrinsics();
    let mut rgb = vec![0; 480 * 300 * 3];
    let radius = 19.0;
    let d = [(center[0] - k.cx) / k.fx, (center[1] - k.cy) / k.fy, 1.0];
    let norm = (d.iter().map(|v| v * v).sum::<f64>()).sqrt();
    let axis = [d[0] / norm, d[1] / norm, d[2] / norm];
    let v = [axis[1], -axis[0], 0.0];
    let skew = [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]];
    let mut rot = [[0.; 3]; 3];
    for (i, row) in rot.iter_mut().enumerate() {
        for (j, value) in row.iter_mut().enumerate() {
            *value = f64::from(i == j) + skew[i][j];
            for (t, skew_row) in skew.iter().enumerate() {
                *value += skew[i][t] * skew_row[j] / (1.0 + axis[2]);
            }
        }
    }
    let mut bounds = [
        f64::INFINITY,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NEG_INFINITY,
    ];
    for y in 0..300 {
        for x in 0..480 {
            let ray = [(x as f64 - k.cx) / k.fx, (y as f64 - k.cy) / k.fy, 1.0];
            let mut local = [0.; 3];
            for i in 0..3 {
                local[i] = (0..3).map(|j| rot[i][j] * ray[j]).sum();
            }
            let q = [k.fx * local[0] / local[2], k.fy * local[1] / local[2]];
            let distance = q[0].hypot(q[1]);
            let alpha = (radius + 0.5 - distance).clamp(0., 1.);
            let bg = if split && q[0] < 0.0 {
                [18., 20., 22.]
            } else {
                [24., 145., 118.]
            };
            let color = if striped && q[1].abs() < 4. {
                [235., 235., 235.]
            } else {
                [225., 62., 35.]
            };
            let shade = if shadow && (q[0] - 9.).hypot(q[1] - 6.) < radius + 3. {
                0.7
            } else {
                1.
            };
            for ch in 0..3 {
                rgb[(y * 480 + x) * 3 + ch] =
                    (bg[ch] * shade * (1. - alpha) + color[ch] * alpha) as u8;
            }
            if distance < radius {
                bounds[0] = bounds[0].min(x as f64);
                bounds[1] = bounds[1].min(y as f64);
                bounds[2] = bounds[2].max(x as f64);
                bounds[3] = bounds[3].max(y as f64);
            }
        }
    }
    (
        rgb,
        BallProposal {
            detection_index: 7,
            bbox: bounds,
        },
    )
}

fn main() {
    let mut observations = Vec::new();
    for center in [[240.0, 150.0], [75.0, 145.0], [405.0, 145.0]] {
        for scene in ["cloth", "rail", "neighbor"] {
            let (mut rgb, proposal) = if scene == "rail" {
                image_with_split_background(center, true, true, true)
            } else {
                image(center, true, false)
            };
            let neighbor = [center[0] + 38.5, center[1]];
            if scene == "neighbor" {
                let (other, _) = image(neighbor, true, false);
                let (rgb_pixels, _) = rgb.as_chunks_mut::<3>();
                let (other_pixels, _) = other.as_chunks::<3>();
                for (a, b) in rgb_pixels.iter_mut().zip(other_pixels.iter()) {
                    if *b != [24, 145, 118] {
                        a.copy_from_slice(b);
                    }
                }
            }
            for maximum_center_spread_px in [1.5, 1.6] {
                for prior in [0.2, 0.3, 0.5] {
                    let cfg = BallLocalizationConfig {
                        maximum_center_spread_px,
                        maximum_center_shift_fraction: prior,
                        batch_budget_ms: 10_000,
                        proposal_budget_ms: 10_000,
                        ..config()
                    };
                    for shift in [-8.0, -6.0, -4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0, 6.0, 8.0] {
                        let p = BallProposal {
                            detection_index: 0,
                            bbox: [
                                proposal.bbox[0] + shift,
                                proposal.bbox[1],
                                proposal.bbox[2] + shift,
                                proposal.bbox[3],
                            ],
                        };
                        let result = refine_ball_batch(&rgb, &intrinsics(), &[p], &cfg)
                            .unwrap()
                            .remove(0);
                        let error = result
                            .center_image_px
                            .map(|p| (p[0] - center[0]).hypot(p[1] - center[1]));
                        let wrong_neighbor = result.center_image_px.map(|p| {
                            (p[0] - neighbor[0]).hypot(p[1] - neighbor[1]) < error.unwrap()
                        });
                        observations.push(json!({"truth":center,"scene":scene,"coarse_shift_px":shift,"maximum_center_shift_fraction":prior,"maximum_center_spread_px":maximum_center_spread_px,"known_center_error_px":error,"wrong_neighbor":wrong_neighbor,"observation":result}));
                    }
                }
            }
        }
    }
    let report = json!({"purpose":"Known-center coarse-proposal prior and adjacent-ball association diagnostics; source synthetic pixels, not physical accuracy qualification", "refiner_revision":billiards_perception::ball_localization_profile::BALL_REFINER_REVISION, "observations":observations});
    std::fs::write(
        "thoughts/calibration/refiner-r10-validation-20261029/paired-center-spread-synthetic-audit-r10.json",
        serde_json::to_vec_pretty(&report).unwrap(),
    )
    .unwrap();
}
