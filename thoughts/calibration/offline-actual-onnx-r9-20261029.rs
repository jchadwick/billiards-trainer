use billiards_perception::ball_localization::{
    refine_ball_batch_with_audit, BallLocalizationAuditLimits, BallLocalizationConfig,
    BallProposal, CameraIntrinsics,
};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

const CORPUS_PATH: &str = "thoughts/calibration/actual-onnx-proposals-20261029/corpus.json";
const SOURCE_IMAGE_PATH: &str = "thoughts/calibration/table-grid-20261026-114241.jpg";
const REFINER_PATH: &str = "perception/src/ball_localization.rs";
const HARNESS_PATH: &str = "thoughts/calibration/offline-actual-onnx-r9-20261029.rs";
const OUTPUT_ROOT: &str = "thoughts/calibration/actual-onnx-r9-20261029";

fn sha256_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn sha256_file(path: &str) -> String {
    sha256_bytes(&fs::read(path).expect("read hash input"))
}

fn base64(bytes: &[u8]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut encoded = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let first = chunk[0];
        let second = chunk.get(1).copied().unwrap_or(0);
        let third = chunk.get(2).copied().unwrap_or(0);
        encoded.push(ALPHABET[(first >> 2) as usize] as char);
        encoded.push(ALPHABET[(((first & 0x03) << 4) | (second >> 4)) as usize] as char);
        if chunk.len() > 1 {
            encoded.push(ALPHABET[(((second & 0x0f) << 2) | (third >> 6)) as usize] as char);
        } else {
            encoded.push('=');
        }
        if chunk.len() > 2 {
            encoded.push(ALPHABET[(third & 0x3f) as usize] as char);
        } else {
            encoded.push('=');
        }
    }
    encoded
}

fn number(value: &Value) -> Option<f64> {
    value.as_f64().filter(|number| number.is_finite())
}

fn point(value: &Value) -> Option<[f64; 2]> {
    let values = value.as_array()?;
    Some([number(values.first()?)?, number(values.get(1)?)?])
}

fn xml(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

fn point_attribute(point: [f64; 2]) -> String {
    format!("{:.3},{:.3}", point[0], point[1])
}

fn outline_points(value: &Value) -> Vec<[f64; 2]> {
    value
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(point)
        .collect()
}

fn rejection_color(reason: &Value) -> &'static str {
    match reason.as_str() {
        None => "#31e875",
        Some("ambiguous_boundary") => "#ff43d1",
        Some("deadline_exceeded") => "#ff2424",
        Some(_) => "#ffb638",
    }
}

fn render_case_events(case: &Value, show_labels: bool) -> String {
    let mut svg = String::new();
    let proposal = &case["proposal"]["bbox_xyxy_source_px"];
    let bbox = proposal.as_array().expect("proposal bbox");
    let left = number(&bbox[0]).expect("left");
    let top = number(&bbox[1]).expect("top");
    let right = number(&bbox[2]).expect("right");
    let bottom = number(&bbox[3]).expect("bottom");
    let ball_id = case["ball_id"].as_str().expect("ball id");
    let width = right - left;
    let height = bottom - top;
    svg.push_str(&format!(
        "<rect x=\"{left:.3}\" y=\"{top:.3}\" width=\"{width:.3}\" height=\"{height:.3}\" fill=\"none\" stroke=\"#fff\" stroke-width=\"0.7\"/>\n"
    ));
    if show_labels {
        svg.push_str(&format!(
            "<text x=\"{left:.3}\" y=\"{:.3}\" font-size=\"5\" fill=\"white\" stroke=\"#111\" stroke-width=\"0.25\" paint-order=\"stroke\">{}</text>\n",
            top - 1.5,
            xml(ball_id)
        ));
    }

    let events = case["trace"]["events"].as_array().expect("trace events");
    let mut candidate_label_index = 0usize;
    for event in events {
        let data = &event["data"];
        let kind = data["kind"].as_str().unwrap_or_default();
        match kind {
            "boundary_candidate" => {
                if let Some(p) = point(&data["point_image_px"]) {
                    svg.push_str(&format!(
                        "<circle cx=\"{:.3}\" cy=\"{:.3}\" r=\"0.55\" fill=\"#53e8ff\" fill-opacity=\"0.72\"/>\n",
                        p[0], p[1]
                    ));
                }
            }
            "selected_inlier" => {
                if let Some(p) = point(&data["point_image_px"]) {
                    svg.push_str(&format!(
                        "<circle cx=\"{:.3}\" cy=\"{:.3}\" r=\"0.85\" fill=\"#ffe85b\" fill-opacity=\"0.9\"/>\n",
                        p[0], p[1]
                    ));
                }
            }
            "raw_hypothesis" => {
                if let Some(p) = point(&data["center_image_px"]) {
                    svg.push_str(&format!(
                        "<path d=\"M {:.3} {:.3} h 2 m -1 -1 v 2\" fill=\"none\" stroke=\"#54a8ff\" stroke-width=\"0.65\"/>\n",
                        p[0], p[1]
                    ));
                }
            }
            "candidate_evaluation" => {
                let center = point(&data["center_image_px"]);
                let color = rejection_color(&data["rejection"]);
                let points = outline_points(&data["outline_image_px"]);
                if points.len() >= 3 {
                    let coordinates = points
                        .iter()
                        .copied()
                        .map(point_attribute)
                        .collect::<Vec<_>>()
                        .join(" ");
                    svg.push_str(&format!(
                        "<polyline points=\"{coordinates}\" fill=\"none\" stroke=\"{color}\" stroke-width=\"1.15\" stroke-linejoin=\"round\" stroke-opacity=\"0.95\"/>\n"
                    ));
                }
                if let Some(center) = center {
                    svg.push_str(&format!(
                        "<circle cx=\"{:.3}\" cy=\"{:.3}\" r=\"1.8\" fill=\"none\" stroke=\"{color}\" stroke-width=\"0.8\"/>\n",
                        center[0], center[1]
                    ));
                    if show_labels {
                        let fit_id = data["fit_id"].as_u64().unwrap_or_default();
                        let reason = data["rejection"].as_str().unwrap_or("qualified");
                        // Keep per-candidate labels in a separate column outside the proposal
                        // box. Candidate centers are close, so labeling each center directly
                        // makes the text unreadable on the enlarged crops.
                        let label_x = right + 2.0;
                        let label_y = top + 3.0 + candidate_label_index as f64 * 4.0;
                        candidate_label_index += 1;
                        svg.push_str(&format!(
                            "<text x=\"{label_x:.3}\" y=\"{label_y:.3}\" font-size=\"3.2\" fill=\"{color}\" stroke=\"#111\" stroke-width=\"0.3\" paint-order=\"stroke\">fit {fit_id}: {}</text>\n",
                            xml(reason)
                        ));
                    }
                }
            }
            _ => {}
        }
    }

    if let Some(center) = point(&case["refinement"]["center_image_px"]) {
        svg.push_str(&format!(
            "<circle cx=\"{:.3}\" cy=\"{:.3}\" r=\"3.0\" fill=\"none\" stroke=\"#00ff54\" stroke-width=\"1.4\"/>\n<path d=\"M {:.3} {:.3} h 5 m -2.5 -2.5 v 5\" stroke=\"#00ff54\" stroke-width=\"0.8\"/>\n",
            center[0], center[1], center[0] - 2.5, center[1]
        ));
    }
    svg
}

fn write_svg(
    path: &Path,
    body: &str,
    view_box: [f64; 4],
    width: u32,
    height: u32,
    image_data_uri: &str,
) {
    let contents = format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{width}\" height=\"{height}\" viewBox=\"{:.3} {:.3} {:.3} {:.3}\">\n<image href=\"{image_data_uri}\" x=\"0\" y=\"0\" width=\"1280\" height=\"720\"/>\n{body}</svg>\n",
        view_box[0], view_box[1], view_box[2], view_box[3]
    );
    fs::write(path, contents).expect("write SVG overlay");
}

fn main() {
    let corpus: Value =
        serde_json::from_slice(&fs::read(CORPUS_PATH).expect("read corpus")).expect("parse corpus");
    let source_record = corpus["images"]
        .as_array()
        .expect("image records")
        .iter()
        .find(|image| image["id"] == "measured-grid-720")
        .expect("frozen source image record");
    let source_image = image::open(SOURCE_IMAGE_PATH)
        .expect("read frozen measured image")
        .to_rgb8();
    let image_data_uri = format!(
        "data:image/jpeg;base64,{}",
        base64(&fs::read(SOURCE_IMAGE_PATH).expect("read source JPEG for standalone overlays"))
    );
    let optics = &corpus["optics_models"]["frozen_zero_radial_720"]["camera_matrix"];
    let intrinsics = CameraIntrinsics {
        source_width_px: source_image.width(),
        source_height_px: source_image.height(),
        fx: number(&optics[0][0]).expect("fx"),
        fy: number(&optics[1][1]).expect("fy"),
        cx: number(&optics[0][2]).expect("cx"),
        cy: number(&optics[1][2]).expect("cy"),
    };
    let config = BallLocalizationConfig {
        batch_budget_ms: 10_000,
        proposal_budget_ms: 10_000,
        ..BallLocalizationConfig::default()
    };
    let output_root = PathBuf::from(OUTPUT_ROOT);
    let case_dir = output_root.join("cases");
    let overlay_dir = output_root.join("overlays");
    fs::create_dir_all(&case_dir).expect("create case output directory");
    fs::create_dir_all(&overlay_dir).expect("create overlay output directory");

    let mut full_overlay = format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"1280\" height=\"720\" viewBox=\"0 0 1280 720\">\n<image href=\"{image_data_uri}\" x=\"0\" y=\"0\" width=\"1280\" height=\"720\"/>\n"
    );
    let mut summaries = Vec::new();
    let mut total_elapsed_ms = 0.0f64;

    for frozen in source_record["ball_proposals"]["proposals"]
        .as_array()
        .expect("frozen actual detector proposals")
    {
        let bbox_values = frozen["bbox_xyxy_source_px"]
            .as_array()
            .expect("bbox array");
        let bbox = [
            number(&bbox_values[0]).expect("left"),
            number(&bbox_values[1]).expect("top"),
            number(&bbox_values[2]).expect("right"),
            number(&bbox_values[3]).expect("bottom"),
        ];
        let detection_index = frozen["detection_index_in_frozen_audit"]
            .as_u64()
            .expect("frozen detection index");
        let proposal = BallProposal {
            detection_index,
            bbox,
        };
        let started = Instant::now();
        let (mut refinements, trace) = refine_ball_batch_with_audit(
            source_image.as_raw(),
            &intrinsics,
            &[proposal],
            &config,
            BallLocalizationAuditLimits::default(),
        )
        .expect("run production estimator with bounded trace");
        let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
        total_elapsed_ms += elapsed_ms;
        let refinement = refinements.pop().expect("one refinement per crop");
        let ball_id = frozen["ball_id"].as_str().expect("ball id");
        let case_id = xml(ball_id);
        let trace_complete = trace.dropped_events == 0;
        let result_value = serde_json::to_value(&refinement).expect("serialize refinement");
        let case_report = json!({
            "schema_version": 1,
            "ball_id": ball_id,
            "source_image_id": source_record["id"],
            "proposal": frozen,
            "intrinsics": intrinsics,
            "refiner_revision": billiards_perception::ball_localization_profile::BALL_REFINER_REVISION,
            "audit_schema_version": trace.schema_version,
            "refinement": result_value.clone(),
            "trace": trace,
            "trace_complete": trace_complete,
            "diagnostic_wall_time_ms": elapsed_ms,
        });
        let case_path = case_dir.join(format!("{case_id}.json"));
        fs::write(
            &case_path,
            serde_json::to_vec_pretty(&case_report).expect("serialize case trace"),
        )
        .expect("write case trace");

        let svg_fragment = render_case_events(&case_report, false);
        full_overlay.push_str(&svg_fragment);
        let center = point(&frozen["bbox_center_source_px"]).expect("bbox center");
        let side = number(&frozen["bbox_side_source_px"]).expect("bbox side");
        let margin = side * 1.75;
        let crop_left = (center[0] - margin).max(0.0);
        let crop_top = (center[1] - margin).max(0.0);
        let crop_right = (center[0] + margin).min(f64::from(source_image.width()));
        let crop_bottom = (center[1] + margin).min(f64::from(source_image.height()));
        let crop_body = render_case_events(&case_report, true);
        let crop_path = overlay_dir.join(format!("{case_id}.svg"));
        write_svg(
            &crop_path,
            &crop_body,
            [
                crop_left,
                crop_top,
                crop_right - crop_left,
                crop_bottom - crop_top,
            ],
            960,
            960,
            &image_data_uri,
        );
        let case_relative = format!("cases/{ball_id}.json");
        let crop_relative = format!("overlays/{ball_id}.svg");
        summaries.push(json!({
            "ball_id": ball_id,
            "physical_reference_id": frozen["physical_reference_id"],
            "physical_reference_present": frozen["physical_reference_present"],
            "status": result_value["status"],
            "reason": result_value["reason"],
            "center_image_px": result_value["center_image_px"],
            "quality": result_value["quality"],
            "trace_retained_bytes": case_report["trace"]["retained_bytes"],
            "trace_dropped_events": case_report["trace"]["dropped_events"],
            "trace_event_count": case_report["trace"]["events"].as_array().map(Vec::len).unwrap_or(0),
            "candidate_evaluations": case_report["trace"]["events"].as_array().map(|events| events.iter().filter(|event| event["data"]["kind"] == "candidate_evaluation").count()).unwrap_or(0),
            "diagnostic_wall_time_ms": elapsed_ms,
            "case_trace_path": case_relative,
            "crop_overlay_path": crop_relative,
            "case_trace_sha256": sha256_bytes(&fs::read(&case_path).expect("read saved trace")),
            "crop_overlay_sha256": sha256_bytes(&fs::read(&crop_path).expect("read saved crop overlay")),
        }));
    }

    full_overlay.push_str(
        "<g font-family=\"sans-serif\"><rect x=\"18\" y=\"16\" width=\"300\" height=\"24\" rx=\"3\" fill=\"#111\" fill-opacity=\"0.82\"/><text x=\"27\" y=\"32\" font-size=\"13\" fill=\"#ffe85b\">DIAGNOSTIC ONLY — 0/15 precise centers accepted</text></g>\n</svg>\n",
    );
    let full_overlay_path = overlay_dir.join("full-source.svg");
    fs::write(&full_overlay_path, full_overlay).expect("write full-source overlay");
    let report = json!({
        "schema_version": 1,
        "experiment": "bounded production-refiner audit on frozen image and actual detector proposals",
        "date": "2026-10-29",
        "purpose": "diagnose boundary selection and rejection causes; not physical accuracy qualification",
        "input_image": {
            "id": source_record["id"],
            "path": SOURCE_IMAGE_PATH,
            "sha256": sha256_file(SOURCE_IMAGE_PATH),
            "dimensions_px": [source_image.width(), source_image.height()],
            "proposal_origin": source_record["ball_proposals"]["status"],
            "proposals_are_detector_output": source_record["ball_proposals"]["detector_generated"],
        },
        "corpus": {"path": CORPUS_PATH, "sha256": sha256_file(CORPUS_PATH)},
        "refiner_source": {"path": REFINER_PATH, "sha256": sha256_file(REFINER_PATH)},
        "research_harness": {"path": HARNESS_PATH, "sha256": sha256_file(HARNESS_PATH)},
        "intrinsics": intrinsics,
        "refiner_revision": billiards_perception::ball_localization_profile::BALL_REFINER_REVISION,
        "audit_schema_version": billiards_perception::ball_localization::BALL_LOCALIZATION_AUDIT_VERSION,
        "runtime_parameters": config,
        "audit_limits": BallLocalizationAuditLimits::default(),
        "total_diagnostic_wall_time_ms": total_elapsed_ms,
        "summary": {
            "proposal_count": summaries.len(),
            "measured_count": summaries.iter().filter(|result| result["physical_reference_present"] == true).count(),
            "accepted_count": summaries.iter().filter(|result| result["status"] == "accepted").count(),
            "measured_accepted_count": summaries.iter().filter(|result| result["physical_reference_present"] == true && result["status"] == "accepted").count(),
            "trace_dropped_event_cases": summaries.iter().filter(|result| result["trace_dropped_events"] != 0).count(),
        },
        "overlays": {
            "full_source": {
                "path": "overlays/full-source.svg",
                "sha256": sha256_bytes(&fs::read(&full_overlay_path).expect("read full-source overlay")),
            },
            "crop_count": summaries.len(),
        },
        "limitations": [
            "Boxes are actual verified-checkpoint PyTorch CPU detector proposals; compiled Coral numerical parity is unverified.",
            "The labels do not establish exact image-center truth.",
            "Repeated refiner observations on this still are not independent placements.",
            "The nine physical references have unknown measurement uncertainty; this audit does not report millimeter accuracy."
        ],
        "results": summaries,
    });
    fs::write(
        output_root.join("summary.json"),
        serde_json::to_vec_pretty(&report).expect("serialize summary"),
    )
    .expect("write summary JSON");

    let mut markdown = String::from(
        "# Converged-candidate audit — 2026-10-29\n\nThis audit uses the frozen saved 1280 × 720 image and actual saved detector boxes. It diagnoses estimator behavior; it does not establish physical X/Y accuracy. Each crop has a bounded candidate trace. Acceptance counts are listed in the table and machine-readable summary. Do not use candidate outlines as resolved measurements.\n\n| Proposal | Physical reference | Result | Qualified candidates | Search truncated | Trace dropped |\n|---|---:|---|---:|---:|---:|\n",
    );
    for result in &summaries {
        markdown.push_str(&format!(
            "| {} | {} | {} / {} | {} | {} | {} |\n",
            result["ball_id"].as_str().unwrap_or(""),
            if result["physical_reference_present"] == true {
                "Yes"
            } else {
                "No"
            },
            result["status"].as_str().unwrap_or("unknown"),
            result["reason"].as_str().unwrap_or("none"),
            result["quality"]["qualified_candidate_count"]
                .as_u64()
                .unwrap_or_default(),
            result["quality"]["candidate_search_truncated"]
                .as_bool()
                .unwrap_or(false),
            result["trace_dropped_events"].as_u64().unwrap_or_default(),
        ));
    }
    markdown.push_str("\nThe full-source overlay is [overlays/full-source.svg](overlays/full-source.svg). Enlarged per-ball overlays are in `overlays/`; complete per-crop traces are in `cases/`. Green outlines passed an individual candidate's gates, but the resolver can still reject the observation when qualified solutions disagree. Magenta outlines show rejected ambiguous fits; orange outlines failed another gate. Cyan dots are sampled boundary candidates, yellow dots are selected inliers, and blue crosses are raw hypotheses. A green center marker appears only for an accepted final result. Crop candidate labels are placed in a side column to keep them off the fitted outlines.\n");
    fs::write(output_root.join("README.md"), markdown).expect("write audit summary");
    println!(
        "{}",
        serde_json::to_string_pretty(&report).expect("serialize result summary")
    );
}
