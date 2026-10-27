use billiards_perception::ball_localization::{
    BallLocalizationAuditLimits, BallLocalizationConfig, BallProposal, CameraIntrinsics,
    refine_ball_batch, refine_ball_batch_with_audit,
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

const CORPUS_PATH: &str = "thoughts/calibration/native-localization-corpus-v1.json";
const PROPOSALS_PATH: &str = "thoughts/calibration/reserved-scene-coarse-proposals-20261027.json";
const MEASURED_IMAGE: &str = "thoughts/calibration/table-grid-20261026-114241.jpg";
const REFINER_PATH: &str = "perception/src/ball_localization.rs";
const HARNESS_PATH: &str = "thoughts/calibration/offline-refiner-robustness-audit-20261027.rs";
const OUTPUT_ROOT: &str = "thoughts/calibration/robustness-audit-20261027";

fn hash_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn hash_file(path: &str) -> String {
    hash_bytes(&fs::read(path).expect("read hash input"))
}

fn number(value: &Value) -> f64 {
    value
        .as_f64()
        .filter(|value| value.is_finite())
        .expect("finite number")
}

fn string<'a>(value: &'a Value) -> &'a str {
    value.as_str().expect("string")
}

fn intrinsics_for(corpus: &Value, model_id: &str, width: u32, height: u32) -> CameraIntrinsics {
    let model = &corpus["optics_models"][model_id];
    let matrix = &model["camera_matrix"];
    CameraIntrinsics {
        source_width_px: width,
        source_height_px: height,
        fx: number(&matrix[0][0]),
        fy: number(&matrix[1][1]),
        cx: number(&matrix[0][2]),
        cy: number(&matrix[1][2]),
    }
}

fn proposal_from_box(detection_index: u64, bbox: [f64; 4]) -> BallProposal {
    BallProposal {
        detection_index,
        bbox,
    }
}

fn bbox_values(value: &Value) -> [f64; 4] {
    [
        number(&value[0]),
        number(&value[1]),
        number(&value[2]),
        number(&value[3]),
    ]
}

fn one_result(
    pixels: &[u8],
    intrinsics: &CameraIntrinsics,
    proposal: &BallProposal,
    config: &BallLocalizationConfig,
) -> Value {
    let start = Instant::now();
    let results = refine_ball_batch(pixels, intrinsics, std::slice::from_ref(proposal), config)
        .expect("run production refiner");
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
    let result = serde_json::to_value(&results[0]).expect("serialize refinement");
    json!({
        "detection_index": proposal.detection_index,
        "bbox_xyxy_source_px": proposal.bbox,
        "status": result["status"],
        "reason": result["reason"],
        "center_image_px": result["center_image_px"],
        "quality": result["quality"],
        "diagnostic_wall_time_ms": elapsed_ms,
    })
}

fn fresh_geometry_config() -> BallLocalizationConfig {
    BallLocalizationConfig {
        batch_budget_ms: 10_000,
        proposal_budget_ms: 10_000,
        ..BallLocalizationConfig::default()
    }
}

fn center_and_size_sensitivity(
    pixels: &[u8],
    intrinsics: &CameraIntrinsics,
    frozen: &Value,
) -> Vec<Value> {
    let mut rows = Vec::new();
    let bbox = bbox_values(&frozen["bbox_xyxy_source_px"]);
    let center = [(bbox[0] + bbox[2]) * 0.5, (bbox[1] + bbox[3]) * 0.5];
    let width = bbox[2] - bbox[0];
    let height = bbox[3] - bbox[1];
    let detection_index = frozen["detection_index_in_frozen_audit"]
        .as_u64()
        .expect("detection index");
    let cfg = fresh_geometry_config();

    for axis in ["x", "y"] {
        for offset in [-2.0, -1.0, 0.0, 1.0, 2.0] {
            let mut perturbed_center = center;
            if axis == "x" {
                perturbed_center[0] += offset;
            } else {
                perturbed_center[1] += offset;
            }
            let proposal = proposal_from_box(
                detection_index,
                [
                    perturbed_center[0] - width * 0.5,
                    perturbed_center[1] - height * 0.5,
                    perturbed_center[0] + width * 0.5,
                    perturbed_center[1] + height * 0.5,
                ],
            );
            let mut row = one_result(pixels, intrinsics, &proposal, &cfg);
            row["experiment"] = json!("center_sensitivity");
            row["axis"] = json!(axis);
            row["offset_px"] = json!(offset);
            row["ball_id"] = frozen["ball_id"].clone();
            rows.push(row);
        }
    }

    for axis in ["width", "height"] {
        for factor in [0.90, 0.95, 1.00, 1.05, 1.10] {
            let candidate_width = if axis == "width" {
                width * factor
            } else {
                width
            };
            let candidate_height = if axis == "height" {
                height * factor
            } else {
                height
            };
            let proposal = proposal_from_box(
                detection_index,
                [
                    center[0] - candidate_width * 0.5,
                    center[1] - candidate_height * 0.5,
                    center[0] + candidate_width * 0.5,
                    center[1] + candidate_height * 0.5,
                ],
            );
            let mut row = one_result(pixels, intrinsics, &proposal, &cfg);
            row["experiment"] = json!("size_sensitivity");
            row["axis"] = json!(axis);
            row["factor"] = json!(factor);
            row["contains_original_reviewed_region"] = json!(factor >= 1.0);
            row["may_clip_original_silhouette"] = json!(factor < 1.0);
            row["ball_id"] = frozen["ball_id"].clone();
            rows.push(row);
        }
    }
    rows
}

fn seed_sensitivity(pixels: &[u8], intrinsics: &CameraIntrinsics, frozen: &Value) -> Vec<Value> {
    let bbox = bbox_values(&frozen["bbox_xyxy_source_px"]);
    let detection_index = frozen["detection_index_in_frozen_audit"]
        .as_u64()
        .expect("detection index");
    let proposal = proposal_from_box(detection_index, bbox);
    let mut rows = Vec::new();
    let mut seeds = vec![716u64];
    seeds.extend(0u64..15);
    for seed in seeds {
        let mut config = fresh_geometry_config();
        config.seed = seed;
        let mut result = one_result(pixels, intrinsics, &proposal, &config);
        result["experiment"] = json!("fitting_start_seed");
        result["seed"] = json!(seed);
        result["ball_id"] = frozen["ball_id"].clone();
        rows.push(result);
    }
    rows
}

fn run_work_budgets(
    pixels: &[u8],
    intrinsics: &CameraIntrinsics,
    frozen_proposals: &[Value],
) -> Vec<Value> {
    let proposals = frozen_proposals
        .iter()
        .map(|frozen| {
            proposal_from_box(
                frozen["detection_index_in_frozen_audit"]
                    .as_u64()
                    .expect("index"),
                bbox_values(&frozen["bbox_xyxy_source_px"]),
            )
        })
        .collect::<Vec<_>>();
    let settings = [
        (5, 5),
        (10, 10),
        (25, 10),
        (50, 25),
        (100, 50),
        (1000, 1000),
        (10_000, 10_000),
    ];
    settings
        .into_iter()
        .map(|(batch_budget_ms, proposal_budget_ms)| {
            let config = BallLocalizationConfig {
                batch_budget_ms,
                proposal_budget_ms,
                ..BallLocalizationConfig::default()
            };
            let start = Instant::now();
            let results = refine_ball_batch(pixels, intrinsics, &proposals, &config)
                .expect("run bounded workload");
            let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
            let accepted = results
                .iter()
                .filter(|result| result.status == billiards_perception::ball_localization::RefinementStatus::Accepted)
                .count();
            let rejected = results.len() - accepted;
            let deadlines = results
                .iter()
                .filter(|result| result.reason == Some(billiards_perception::ball_localization::RejectionReason::DeadlineExceeded))
                .count();
            json!({
                "batch_budget_ms": batch_budget_ms,
                "proposal_budget_ms": proposal_budget_ms,
                "elapsed_wall_ms_development_host": elapsed_ms,
                "proposal_count": results.len(),
                "accepted_count": accepted,
                "rejected_count": rejected,
                "deadline_rejections": deadlines,
                "results": results,
            })
        })
        .collect()
}

fn xml(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

fn base64(bytes: &[u8]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut encoded = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let a = chunk[0];
        let b = chunk.get(1).copied().unwrap_or(0);
        let c = chunk.get(2).copied().unwrap_or(0);
        encoded.push(ALPHABET[(a >> 2) as usize] as char);
        encoded.push(ALPHABET[(((a & 3) << 4) | (b >> 4)) as usize] as char);
        encoded.push(if chunk.len() > 1 {
            ALPHABET[(((b & 15) << 2) | (c >> 6)) as usize] as char
        } else {
            '='
        });
        encoded.push(if chunk.len() > 2 {
            ALPHABET[(c & 63) as usize] as char
        } else {
            '='
        });
    }
    encoded
}

fn candidate_color(reason: Option<&str>) -> &'static str {
    match reason {
        None => "#31e875",
        Some("ambiguous_boundary") => "#ff43d1",
        Some("deadline_exceeded") => "#ff2424",
        Some(_) => "#ffb638",
    }
}

fn trace_overlay_fragment(case: &Value, proposal: &Value, show_label: bool) -> String {
    let mut body = String::new();
    let center = [
        number(&proposal["center_px_approx"][0]),
        number(&proposal["center_px_approx"][1]),
    ];
    let side = number(&proposal["bbox_side_px"]);
    let left = center[0] - side * 0.5;
    let top = center[1] - side * 0.5;
    body.push_str(&format!(
        "<rect x=\"{left:.2}\" y=\"{top:.2}\" width=\"{side:.2}\" height=\"{side:.2}\" fill=\"none\" stroke=\"#fff\" stroke-width=\"1\"/>\n"
    ));
    if show_label {
        body.push_str(&format!(
            "<text x=\"{left:.2}\" y=\"{:.2}\" font-size=\"4\" fill=\"white\" stroke=\"#111\" stroke-width=\"0.3\" paint-order=\"stroke\">{}: {}</text>\n",
            top - 2.0,
            xml(string(&proposal["id"])),
            xml(case["refinement"]["reason"].as_str().unwrap_or("none")),
        ));
    }
    for event in case["trace"]["events"].as_array().expect("trace events") {
        let data = &event["data"];
        let kind = data["kind"].as_str().unwrap_or_default();
        if kind == "selected_inlier" {
            if let Some(point) = data["point_image_px"].as_array() {
                if point.len() == 2 {
                    body.push_str(&format!(
                        "<circle cx=\"{:.3}\" cy=\"{:.3}\" r=\"1.2\" fill=\"#ffe85b\"/>\n",
                        number(&point[0]),
                        number(&point[1])
                    ));
                }
            }
        } else if kind == "candidate_evaluation" {
            let reason = data["rejection"].as_str();
            let color = candidate_color(reason);
            if let Some(points) = data["outline_image_px"].as_array() {
                let coordinates = points
                    .iter()
                    .filter_map(|point| {
                        let values = point.as_array()?;
                        if values.len() != 2 || !values[0].is_number() || !values[1].is_number() {
                            return None;
                        }
                        Some(format!(
                            "{:.3},{:.3}",
                            number(&values[0]),
                            number(&values[1])
                        ))
                    })
                    .collect::<Vec<_>>();
                if coordinates.len() >= 3 {
                    body.push_str(&format!(
                        "<polyline points=\"{}\" fill=\"none\" stroke=\"{color}\" stroke-width=\"1.15\"/>\n",
                        coordinates.join(" ")
                    ));
                }
            }
        }
    }
    if let Some(center) = case["refinement"]["center_image_px"].as_array() {
        if center.len() == 2 {
            body.push_str(&format!(
                "<circle cx=\"{:.3}\" cy=\"{:.3}\" r=\"3\" fill=\"none\" stroke=\"#00ff54\" stroke-width=\"1.5\"/>\n",
                number(&center[0]), number(&center[1])
            ));
        }
    }
    body
}

fn audited_reserved_scenes(corpus: &Value, proposal_file: &Value, output: &Path) -> Vec<Value> {
    let mut summaries = Vec::new();
    let case_dir = output.join("reserved-cases");
    let overlay_dir = output.join("reserved-overlays");
    fs::create_dir_all(&case_dir).expect("make reserved trace directory");
    fs::create_dir_all(&overlay_dir).expect("make reserved overlay directory");

    for set in proposal_file["proposal_sets"]
        .as_array()
        .expect("proposal sets")
    {
        let image_id = string(&set["image_id"]);
        let path = string(&set["path"]);
        let source_bytes = fs::read(path).expect("read reserved image");
        assert_eq!(
            hash_bytes(&source_bytes),
            string(&set["sha256"]),
            "image hash"
        );
        let image = image::open(path).expect("decode reserved image").to_rgb8();
        assert_eq!(
            image.width(),
            set["dimensions_px"][0].as_u64().unwrap() as u32
        );
        assert_eq!(
            image.height(),
            set["dimensions_px"][1].as_u64().unwrap() as u32
        );
        let intrinsics = intrinsics_for(
            corpus,
            string(&set["optics_model_id"]),
            image.width(),
            image.height(),
        );
        let config = fresh_geometry_config();
        let proposals = set["proposals"]
            .as_array()
            .expect("manual proposals")
            .iter()
            .map(|item| {
                let center = [
                    number(&item["center_px_approx"][0]),
                    number(&item["center_px_approx"][1]),
                ];
                let side = number(&item["bbox_side_px"]);
                proposal_from_box(
                    item["detection_index"].as_u64().unwrap(),
                    [
                        center[0] - side * 0.5,
                        center[1] - side * 0.5,
                        center[0] + side * 0.5,
                        center[1] + side * 0.5,
                    ],
                )
            })
            .collect::<Vec<_>>();
        let mut full_svg = format!(
            "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{}\" height=\"{}\" viewBox=\"0 0 {} {}\"><image href=\"data:image/jpeg;base64,{}\" x=\"0\" y=\"0\" width=\"{}\" height=\"{}\"/>\n",
            image.width(),
            image.height(),
            image.width(),
            image.height(),
            base64(&source_bytes),
            image.width(),
            image.height()
        );
        let mut case_summaries = Vec::new();
        for item in set["proposals"].as_array().unwrap() {
            let index = item["detection_index"].as_u64().unwrap();
            let proposal = proposals
                .iter()
                .find(|p| p.detection_index == index)
                .unwrap();
            let (mut refinements, trace) = refine_ball_batch_with_audit(
                image.as_raw(),
                &intrinsics,
                std::slice::from_ref(proposal),
                &config,
                BallLocalizationAuditLimits::default(),
            )
            .expect("audit production refiner");
            let refinement = refinements.pop().unwrap();
            let case = json!({
                "schema_version": 1,
                "image_id": image_id,
                "image_path": path,
                "optics_model_id": set["optics_model_id"],
                "optics_status": set["optics_status"],
                "intrinsics": intrinsics,
                "refiner_revision": billiards_perception::ball_localization_profile::BALL_REFINER_REVISION,
                "proposal_file_sha256": hash_file(PROPOSALS_PATH),
                "proposal": item,
                "bbox_xyxy_source_px": proposal.bbox,
                "refinement": refinement,
                "trace": trace,
                "diagnostic_wall_time_scope": "embedded in refinement trace API; development host only",
            });
            let stem = string(&item["id"]);
            let json_path = case_dir.join(format!("{stem}.json"));
            fs::write(&json_path, serde_json::to_vec_pretty(&case).unwrap()).unwrap();
            full_svg.push_str(&trace_overlay_fragment(&case, item, false));
            let center = [
                number(&item["center_px_approx"][0]),
                number(&item["center_px_approx"][1]),
            ];
            let margin = number(&item["bbox_side_px"]) * 1.75;
            let x = (center[0] - margin).max(0.0);
            let y = (center[1] - margin).max(0.0);
            let right = (center[0] + margin).min(f64::from(image.width()));
            let bottom = (center[1] + margin).min(f64::from(image.height()));
            let crop_path = overlay_dir.join(format!("{stem}.svg"));
            let body = trace_overlay_fragment(&case, item, true);
            let svg = format!(
                "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"960\" height=\"960\" viewBox=\"{x:.3} {y:.3} {:.3} {:.3}\"><image href=\"data:image/jpeg;base64,{}\" x=\"0\" y=\"0\" width=\"{}\" height=\"{}\"/>{body}</svg>\n",
                right - x,
                bottom - y,
                base64(&source_bytes),
                image.width(),
                image.height()
            );
            fs::write(&crop_path, svg).unwrap();
            case_summaries.push(json!({
                "id": stem,
                "visibility": item["visibility"],
                "status": case["refinement"]["status"],
                "reason": case["refinement"]["reason"],
                "center_image_px": case["refinement"]["center_image_px"],
                "quality": case["refinement"]["quality"],
                "trace_dropped_events": case["trace"]["dropped_events"],
                "trace_retained_bytes": case["trace"]["retained_bytes"],
                "trace_path": format!("reserved-cases/{stem}.json"),
                "overlay_path": format!("reserved-overlays/{stem}.svg"),
                "trace_sha256": hash_bytes(&fs::read(json_path).unwrap()),
                "overlay_sha256": hash_bytes(&fs::read(crop_path).unwrap()),
            }));
        }
        full_svg.push_str("</svg>\n");
        let full_path = overlay_dir.join(format!("{image_id}-full.svg"));
        fs::write(&full_path, full_svg).unwrap();
        summaries.push(json!({
            "image_id": image_id,
            "image_path": path,
            "image_sha256": string(&set["sha256"]),
            "dimensions_px": set["dimensions_px"],
            "optics_model_id": set["optics_model_id"],
            "optics_status": set["optics_status"],
            "proposal_count": case_summaries.len(),
            "accepted_count": case_summaries.iter().filter(|r| r["status"] == "accepted").count(),
            "rejected_count": case_summaries.iter().filter(|r| r["status"] == "rejected").count(),
            "trace_drop_cases": case_summaries.iter().filter(|r| r["trace_dropped_events"] != 0).count(),
            "full_overlay_path": format!("reserved-overlays/{image_id}-full.svg"),
            "full_overlay_sha256": hash_bytes(&fs::read(full_path).unwrap()),
            "results": case_summaries,
        }));
    }
    summaries
}

fn preserve_accepted_variant_traces(
    pixels: &[u8],
    source_jpeg: &[u8],
    intrinsics: &CameraIntrinsics,
    center_size_rows: &[Value],
    seed_rows: &[Value],
    output: &Path,
) -> Vec<Value> {
    let mut selected = center_size_rows
        .iter()
        .filter(|row| row["status"] == "accepted")
        .cloned()
        .collect::<Vec<_>>();
    selected.extend(
        seed_rows
            .iter()
            .filter(|row| row["status"] == "accepted")
            .cloned(),
    );

    let case_dir = output.join("accepted-variant-traces");
    let overlay_dir = output.join("accepted-variant-overlays");
    fs::create_dir_all(&case_dir).unwrap();
    fs::create_dir_all(&overlay_dir).unwrap();
    let mut summaries = Vec::new();
    for row in selected {
        let id = if row["experiment"] == "fitting_start_seed" {
            format!("seed-{}-{}", row["seed"], string(&row["ball_id"]))
        } else {
            format!(
                "center-{}-{}-{}-{}",
                string(&row["ball_id"]),
                string(&row["axis"]),
                row["offset_px"].as_f64().unwrap_or_default() as i32,
                "px"
            )
        };
        let bbox = bbox_values(&row["bbox_xyxy_source_px"]);
        let proposal = proposal_from_box(row["detection_index"].as_u64().unwrap(), bbox);
        let mut config = fresh_geometry_config();
        if let Some(seed) = row["seed"].as_u64() {
            config.seed = seed;
        }
        let (mut refinements, trace) = refine_ball_batch_with_audit(
            pixels,
            intrinsics,
            std::slice::from_ref(&proposal),
            &config,
            BallLocalizationAuditLimits::default(),
        )
        .expect("trace accepted sensitivity result");
        let refinement = refinements.pop().unwrap();
        let center = [(bbox[0] + bbox[2]) * 0.5, (bbox[1] + bbox[3]) * 0.5];
        let side = (bbox[2] - bbox[0]).max(bbox[3] - bbox[1]);
        let proposal_json = json!({
            "id": id,
            "center_px_approx": center,
            "bbox_side_px": side,
            "visibility": "sensitivity_variant",
        });
        let case = json!({
            "schema_version": 1,
            "source_image": MEASURED_IMAGE,
            "source_image_sha256": hash_file(MEASURED_IMAGE),
            "experiment": row["experiment"],
            "seed": config.seed,
            "sensitivity_result": row,
            "proposal": proposal_json,
            "bbox_xyxy_source_px": proposal.bbox,
            "refiner_revision": billiards_perception::ball_localization_profile::BALL_REFINER_REVISION,
            "refinement": refinement,
            "trace": trace,
        });
        let json_path = case_dir.join(format!("{id}.json"));
        fs::write(&json_path, serde_json::to_vec_pretty(&case).unwrap()).unwrap();
        let margin = side * 1.75;
        let left = (center[0] - margin).max(0.0);
        let top = (center[1] - margin).max(0.0);
        let right = (center[0] + margin).min(f64::from(intrinsics.source_width_px));
        let bottom = (center[1] + margin).min(f64::from(intrinsics.source_height_px));
        let overlay_path = overlay_dir.join(format!("{id}.svg"));
        let body = trace_overlay_fragment(&case, &proposal_json, true);
        let svg = format!(
            "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"960\" height=\"960\" viewBox=\"{left:.3} {top:.3} {:.3} {:.3}\"><image href=\"data:image/jpeg;base64,{}\" x=\"0\" y=\"0\" width=\"{}\" height=\"{}\"/>{body}</svg>\n",
            right - left,
            bottom - top,
            base64(source_jpeg),
            intrinsics.source_width_px,
            intrinsics.source_height_px,
        );
        fs::write(&overlay_path, svg).unwrap();
        summaries.push(json!({
            "id": id,
            "experiment": row["experiment"],
            "seed": config.seed,
            "ball_id": row["ball_id"],
            "center_image_px": case["refinement"]["center_image_px"],
            "reason": case["refinement"]["reason"],
            "trace_dropped_events": case["trace"]["dropped_events"],
            "trace_path": format!("accepted-variant-traces/{id}.json"),
            "overlay_path": format!("accepted-variant-overlays/{id}.svg"),
            "trace_sha256": hash_bytes(&fs::read(json_path).unwrap()),
            "overlay_sha256": hash_bytes(&fs::read(overlay_path).unwrap()),
        }));
    }
    summaries
}

fn main() {
    let corpus: Value = serde_json::from_slice(&fs::read(CORPUS_PATH).unwrap()).unwrap();
    let proposal_file: Value = serde_json::from_slice(&fs::read(PROPOSALS_PATH).unwrap()).unwrap();
    let measured = corpus["images"]
        .as_array()
        .unwrap()
        .iter()
        .find(|image| image["id"] == "measured-grid-720")
        .unwrap();
    let source = image::open(MEASURED_IMAGE).unwrap().to_rgb8();
    let source_jpeg = fs::read(MEASURED_IMAGE).unwrap();
    assert_eq!(hash_bytes(&source_jpeg), string(&measured["sha256"]));
    let intrinsics = intrinsics_for(
        &corpus,
        "frozen_zero_radial_720",
        source.width(),
        source.height(),
    );
    let proposals = measured["ball_proposals"]["proposals"].as_array().unwrap();
    let output = PathBuf::from(OUTPUT_ROOT);
    fs::create_dir_all(&output).unwrap();
    let mut center_size = Vec::new();
    let mut seeds = Vec::new();
    for frozen in proposals {
        center_size.extend(center_and_size_sensitivity(
            source.as_raw(),
            &intrinsics,
            frozen,
        ));
        seeds.extend(seed_sensitivity(source.as_raw(), &intrinsics, frozen));
    }
    let budgets = run_work_budgets(source.as_raw(), &intrinsics, proposals);
    let accepted_traces = preserve_accepted_variant_traces(
        source.as_raw(),
        &source_jpeg,
        &intrinsics,
        &center_size,
        &seeds,
        &output,
    );
    let reserved = audited_reserved_scenes(&corpus, &proposal_file, &output);
    let report = json!({
        "schema_version": 1,
        "date": "2026-10-27",
        "experiment": "controlled proposal and fitting-seed sensitivity plus frozen reserved-scene baseline",
        "purpose": "measure implementation sensitivity and contour rejection behavior; not physical accuracy validation",
        "corpus_path": CORPUS_PATH,
        "corpus_sha256": hash_file(CORPUS_PATH),
        "measured_image": {
            "path": MEASURED_IMAGE,
            "sha256": hash_file(MEASURED_IMAGE),
            "dimensions_px": [source.width(), source.height()],
            "proposal_origin": measured["ball_proposals"]["status"],
            "proposals_are_detector_output": measured["ball_proposals"]["detector_generated"],
        },
        "reserved_proposal_file": {
            "path": PROPOSALS_PATH,
            "sha256": hash_file(PROPOSALS_PATH),
            "proposals_are_detector_output": false,
        },
        "refiner_source": {"path": REFINER_PATH, "sha256": hash_file(REFINER_PATH)},
        "harness": {"path": HARNESS_PATH, "sha256": hash_file(HARNESS_PATH)},
        "refiner_revision": billiards_perception::ball_localization_profile::BALL_REFINER_REVISION,
        "runtime_parameters": fresh_geometry_config(),
        "center_and_size_sensitivity": {
            "source_pixel_offsets": [-2, -1, 0, 1, 2],
            "size_factors": [0.90, 0.95, 1.00, 1.05, 1.10],
            "center_then_size_change_axes_separately": true,
            "run_count": center_size.len(),
            "results": center_size,
        },
        "fitting_seed_sensitivity": {
            "seeds": [716, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
            "run_count": seeds.len(),
            "results": seeds,
        },
        "work_budget_sensitivity": {
            "description": "Same frozen 15 proposals, with bounded batch/proposal budgets varied; elapsed wall time is from this development host and is not target-device performance.",
            "results": budgets,
        },
        "accepted_variant_traces": accepted_traces,
        "reserved_scene_baselines": reserved,
        "limitations": [
            "Repeated fits and proposal perturbations on one still are not independent placements.",
            "Saved manual boxes are approximate crop seeds and do not establish image-center truth.",
            "The alternate-grid image has no matching physical references and uses exploratory 720p intrinsics.",
            "The 4K image has only three balls, derived diagnostic intrinsics, and no verified table-wide calibration.",
            "This audit does not qualify the detector, target device, FPS, physical X/Y accuracy, or guidance availability."
        ]
    });
    fs::write(
        output.join("summary.json"),
        serde_json::to_vec_pretty(&report).unwrap(),
    )
    .unwrap();
    println!("{}", serde_json::to_string_pretty(&report).unwrap());
}
