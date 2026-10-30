use billiards_perception::ball_localization::{
    refine_ball_batch_with_audit, BallLocalizationAuditEventData, BallLocalizationAuditLimits,
    BallLocalizationConfig, BallProposal, CameraIntrinsics,
};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

const CORPUS_PATH: &str = "thoughts/calibration/native-localization-corpus-v1.json";
const IMAGE_PATH: &str = "thoughts/calibration/table-grid-20261026-114241.jpg";
const REFINER_PATH: &str = "perception/src/ball_localization.rs";
const HARNESS_PATH: &str =
    "thoughts/calibration/offline-refiner-full-aiming-sampling-audit-20261027.rs";
const OUTPUT_ROOT: &str = "thoughts/calibration/full-aiming-sampling-audit-20261027";
const BALL_ID: &str = "interior-purple-second-left-middle-row-ball";

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn file_sha256(path: &str) -> String {
    sha256(&fs::read(path).expect("read hash input"))
}

fn number(value: &Value) -> f64 {
    value
        .as_f64()
        .filter(|value| value.is_finite())
        .expect("finite numeric value")
}

fn text(value: &Value) -> &str {
    value.as_str().expect("string value")
}

fn bbox(value: &Value) -> [f64; 4] {
    [
        number(&value[0]),
        number(&value[1]),
        number(&value[2]),
        number(&value[3]),
    ]
}

fn point(value: &[f64; 2]) -> Value {
    json!([value[0], value[1]])
}

fn hausdorff(a: &[[f64; 2]], b: &[[f64; 2]]) -> f64 {
    fn directed_max(a: &[[f64; 2]], b: &[[f64; 2]]) -> f64 {
        a.iter()
            .map(|p| {
                b.iter()
                    .map(|q| (p[0] - q[0]).hypot(p[1] - q[1]))
                    .fold(f64::INFINITY, f64::min)
            })
            .fold(0.0, f64::max)
    }
    directed_max(a, b).max(directed_max(b, a))
}

fn run_case(
    pixels: &[u8],
    intrinsics: &CameraIntrinsics,
    proposal: &BallProposal,
    seed: u64,
    ransac_iterations: usize,
    case_dir: &Path,
) -> Value {
    let config = BallLocalizationConfig {
        batch_budget_ms: 10_000,
        proposal_budget_ms: 10_000,
        seed,
        ransac_iterations,
        ..BallLocalizationConfig::default()
    };
    let started = Instant::now();
    let (refinements, trace) = refine_ball_batch_with_audit(
        pixels,
        intrinsics,
        std::slice::from_ref(proposal),
        &config,
        BallLocalizationAuditLimits::default(),
    )
    .expect("run production refiner with audit trace");
    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
    assert_eq!(refinements.len(), 1, "one result per proposal");
    assert_eq!(trace.dropped_events, 0, "no audit trace event loss");
    assert!(
        trace.events.len() < trace.maximum_events,
        "trace reached event cap"
    );

    let mut sampled_hypotheses = Vec::new();
    let mut refined_fits = std::collections::BTreeMap::<(usize, usize), Value>::new();
    let mut candidate_evaluations = Vec::new();
    let mut qualified = Vec::<(usize, usize, Vec<[f64; 2]>)>::new();

    for event in &trace.events {
        match &event.data {
            BallLocalizationAuditEventData::RawHypothesis {
                fit_id,
                sampled_candidate_indices,
                sampled_candidate_rays,
                center_local_px,
                center_image_px,
                radius_local_px,
                support,
                squared_error_local_px,
            } => sampled_hypotheses.push(json!({
                "iteration": event.iteration,
                "fit_id": fit_id,
                "sampled_candidate_indices": sampled_candidate_indices,
                "sampled_candidate_rays": sampled_candidate_rays,
                "center_local_px": point(center_local_px),
                "center_image_px": center_image_px,
                "radius_local_px": radius_local_px,
                "support": support,
                "squared_error_local_px": squared_error_local_px,
            })),
            BallLocalizationAuditEventData::RefinedFit {
                fit_id,
                center_local_px,
                center_image_px,
                radius_local_px,
                outline_image_px: _,
            } => {
                refined_fits.insert(
                    (event.iteration.unwrap_or_default(), *fit_id),
                    json!({
                        "center_local_px": point(center_local_px),
                        "center_image_px": center_image_px,
                        "radius_local_px": radius_local_px,
                    }),
                );
            }
            BallLocalizationAuditEventData::CandidateEvaluation {
                fit_id,
                center_image_px,
                supported_rays,
                residual_rms_px,
                source_residual_rms_px,
                rejection,
                quality,
                outline_image_px,
            } => {
                let outline: Vec<[f64; 2]> = outline_image_px.iter().filter_map(|p| *p).collect();
                let is_qualified =
                    rejection.is_none() && center_image_px.is_some() && outline.len() >= 4;
                if is_qualified {
                    qualified.push((
                        event.iteration.unwrap_or_default(),
                        *fit_id,
                        outline.clone(),
                    ));
                }
                candidate_evaluations.push(json!({
                    "iteration": event.iteration,
                    "fit_id": fit_id,
                    "center_image_px": center_image_px,
                    "supported_rays": supported_rays,
                    "residual_rms_px": residual_rms_px,
                    "source_residual_rms_px": source_residual_rms_px,
                    "rejection": rejection,
                    "is_fully_qualified": is_qualified,
                    "outline_point_count": outline.len(),
                    "quality": quality,
                    "refined_fit": refined_fits.get(&(event.iteration.unwrap_or_default(), *fit_id)),
                }));
            }
            _ => {}
        }
    }

    let mut pairwise_outline_distances = Vec::new();
    for left in 0..qualified.len() {
        for right in left + 1..qualified.len() {
            if qualified[left].0 != qualified[right].0 {
                continue;
            }
            pairwise_outline_distances.push(json!({
                "iteration": qualified[left].0,
                "fit_id_a": qualified[left].1,
                "fit_id_b": qualified[right].1,
                "hausdorff_source_px": hausdorff(&qualified[left].2, &qualified[right].2),
            }));
        }
    }

    let refinement = serde_json::to_value(&refinements[0]).expect("serialize refinement");
    let case_id = format!("seed-{seed}-ransac-{ransac_iterations}");
    let trace_value = serde_json::to_value(&trace).expect("serialize audit trace");
    let case = json!({
        "schema_version": 1,
        "case_id": case_id,
        "diagnostics_only": true,
        "production_behavior_changed": false,
        "maximum_aiming_iterations": config.max_iterations,
        "refiner_revision": billiards_perception::ball_localization_profile::BALL_REFINER_REVISION,
        "audit_schema_version": trace.schema_version,
        "ball_id": BALL_ID,
        "source_image": IMAGE_PATH,
        "source_image_sha256": file_sha256(IMAGE_PATH),
        "proposal_bbox_xyxy_source_px": proposal.bbox,
        "intrinsics": intrinsics,
        "config": config,
        "elapsed_with_audit_trace_ms_development_host": elapsed_ms,
        "refinement": refinement,
        "raw_hypothesis_count": sampled_hypotheses.len(),
        "sampled_hypotheses": sampled_hypotheses,
        "candidate_evaluations": candidate_evaluations,
        "fully_qualified_fit_count": qualified.len(),
        "pairwise_source_outline_hausdorff_distances": pairwise_outline_distances,
        "trace": trace_value,
        "limitations": [
            "Only valid RawHypothesis samples are recorded; random draws rejected before a raw hypothesis do not emit a triplet event.",
            "One frozen saved image and one manual annotation-derived proposal do not establish physical image-center truth.",
            "The optics model is exploratory and not qualified for runtime use.",
            "Trace-enabled elapsed time is diagnostic overhead on the development host, not target or production latency."
        ]
    });
    let case_path = case_dir.join(format!("{case_id}.json"));
    fs::write(&case_path, serde_json::to_vec_pretty(&case).unwrap()).expect("write case trace");

    json!({
        "case_id": case_id,
        "seed": seed,
        "ransac_iterations": ransac_iterations,
        "max_aiming_iterations": config.max_iterations,
        "aiming_iterations_reached": refinement["quality"]["iterations"],
        "status": refinement["status"],
        "reason": refinement["reason"],
        "center_image_px": refinement["center_image_px"],
        "raw_hypothesis_count": sampled_hypotheses.len(),
        "sampled_hypotheses": sampled_hypotheses,
        "candidate_evaluation_count": candidate_evaluations.len(),
        "candidate_evaluations": candidate_evaluations,
        "fully_qualified_fit_count": qualified.len(),
        "pairwise_source_outline_hausdorff_distances": pairwise_outline_distances,
        "trace_event_count": trace.events.len(),
        "trace_dropped_events": trace.dropped_events,
        "trace_retained_bytes": trace.retained_bytes,
        "trace_path": format!("cases/{case_id}.json"),
        "trace_sha256": file_sha256(case_path.to_str().expect("utf-8 path")),
        "elapsed_with_audit_trace_ms_development_host": elapsed_ms,
    })
}

fn main() {
    let corpus_bytes = fs::read(CORPUS_PATH).expect("read frozen corpus");
    let corpus: Value = serde_json::from_slice(&corpus_bytes).expect("parse frozen corpus");
    let source_record = corpus["images"]
        .as_array()
        .expect("corpus images")
        .iter()
        .find(|image| image["id"] == "measured-grid-720")
        .expect("frozen measured-grid record");
    let source_jpeg = fs::read(IMAGE_PATH).expect("read source JPEG");
    assert_eq!(sha256(&source_jpeg), text(&source_record["sha256"]));
    let source = image::open(IMAGE_PATH)
        .expect("decode source JPEG")
        .to_rgb8();
    assert_eq!(
        [source.width(), source.height()],
        [1280, 720],
        "expected exact frozen 720p image"
    );
    let camera = &corpus["optics_models"]["frozen_zero_radial_720"]["camera_matrix"];
    let intrinsics = CameraIntrinsics {
        source_width_px: source.width(),
        source_height_px: source.height(),
        fx: number(&camera[0][0]),
        fy: number(&camera[1][1]),
        cx: number(&camera[0][2]),
        cy: number(&camera[1][2]),
    };
    let frozen = source_record["ball_proposals"]["proposals"]
        .as_array()
        .expect("saved proposal array")
        .iter()
        .find(|proposal| proposal["ball_id"] == BALL_ID)
        .expect("purple reference proposal");
    let proposal = BallProposal {
        detection_index: frozen["detection_index_in_frozen_audit"]
            .as_u64()
            .expect("detection identity"),
        bbox: bbox(&frozen["bbox_xyxy_source_px"]),
    };

    let output_root = PathBuf::from(OUTPUT_ROOT);
    let case_dir = output_root.join("cases");
    fs::create_dir_all(&case_dir).expect("create case output directory");
    let settings = [(716, 128), (716, 256), (716, 512), (716, 1024)];
    let results = settings
        .into_iter()
        .map(|(seed, iterations)| {
            run_case(
                source.as_raw(),
                &intrinsics,
                &proposal,
                seed,
                iterations,
                &case_dir,
            )
        })
        .collect::<Vec<_>>();
    let report = json!({
        "schema_version": 1,
        "date": "2026-10-27",
        "experiment": "full-aiming RANSAC iteration-count sensitivity",
        "purpose": "determine whether first-iteration candidate sampling differences persist through normal repeated aiming updates",
        "diagnostics_only": true,
        "production_behavior_changed": false,
        "corpus": { "path": CORPUS_PATH, "sha256": sha256(&corpus_bytes) },
        "source_image": {
            "path": IMAGE_PATH,
            "sha256": sha256(&source_jpeg),
            "dimensions_px": [source.width(), source.height()],
            "proposal_origin": source_record["ball_proposals"]["status"],
            "proposals_are_detector_output": source_record["ball_proposals"]["detector_generated"]
        },
        "proposal": {
            "ball_id": BALL_ID,
            "detection_index": proposal.detection_index,
            "bbox_xyxy_source_px": proposal.bbox,
            "origin": frozen["origin"],
            "physical_reference_present": frozen["physical_reference_present"],
            "localization_review_flags": frozen["localization_review_flags"]
        },
        "intrinsics": {
            "profile_id": "frozen_zero_radial_720",
            "status": corpus["optics_models"]["frozen_zero_radial_720"]["status"],
            "camera_intrinsics": intrinsics
        },
        "refiner_revision": billiards_perception::ball_localization_profile::BALL_REFINER_REVISION,
        "refiner_source": { "path": REFINER_PATH, "sha256": file_sha256(REFINER_PATH) },
        "harness": { "path": HARNESS_PATH, "sha256": file_sha256(HARNESS_PATH) },
        "audit_schema_version": billiards_perception::ball_localization::BALL_LOCALIZATION_AUDIT_VERSION,
        "fixed_parameters": {
            "max_aiming_iterations": BallLocalizationConfig::default().max_iterations,
            "batch_budget_ms": 10000,
            "proposal_budget_ms": 10000,
            "production_seed_reference": 716,
            "production_ransac_iterations_reference": 128,
            "other_config": "BallLocalizationConfig::default()"
        },
        "settings": settings.iter().map(|(seed, iterations)| json!({"seed": seed, "ransac_iterations": iterations})).collect::<Vec<_>>(),
        "run_count": results.len(),
        "results": results,
        "limitations": [
            "This one-image study measures estimator sensitivity, not physical accuracy.",
            "The selected proposal is manual annotation-derived and not detector output or exact center truth.",
            "The frozen zero-radial intrinsics are exploratory and not a qualified runtime optics profile.",
            "Only valid raw hypotheses retain sampled triplets; pre-hypothesis rejected draws are not traced.",
            "A larger RANSAC search may reveal more alternatives and reject more observations; acceptance count is not the objective.",
            "Trace-enabled timing is from the development host and is not target performance."
        ]
    });
    fs::write(
        output_root.join("summary.json"),
        serde_json::to_vec_pretty(&report).expect("serialize summary"),
    )
    .expect("write summary");
    println!(
        "Full-aiming sampling audit complete: {} cases, {} raw hypotheses, results in {OUTPUT_ROOT}/summary.json",
        report["results"].as_array().unwrap().len(),
        report["results"]
            .as_array()
            .unwrap()
            .iter()
            .map(|result| result["raw_hypothesis_count"].as_u64().unwrap())
            .sum::<u64>()
    );
}
