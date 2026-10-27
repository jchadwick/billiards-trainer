use billiards_perception::ball_localization::{
    refine_ball_batch_with_audit, BallLocalizationAuditEventData, BallLocalizationAuditLimits,
    BallLocalizationConfig, BallProposal, CameraIntrinsics,
};
use image::RgbImage;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

const CORPUS_PATH: &str = "thoughts/calibration/native-localization-corpus-v1.json";
const RESERVED_PATH: &str = "thoughts/calibration/reserved-scene-coarse-proposals-20261027.json";
const REFINER_PATH: &str = "perception/src/ball_localization.rs";
const HARNESS_PATH: &str = "thoughts/calibration/offline-refiner-contour-color-audit-20261027.rs";
const OUTPUT_ROOT: &str = "thoughts/calibration/contour-color-audit-20261027";

#[derive(Clone)]
struct ProposalCase {
    id: String,
    detection_index: u64,
    bbox: [f64; 4],
    physical_reference_present: bool,
    visibility: Option<String>,
}

struct ImageSet {
    id: String,
    path: String,
    sha256: String,
    optics_id: String,
    optics_status: String,
    proposals_are_detector_output: bool,
    proposals: Vec<ProposalCase>,
}

#[derive(Clone, Copy)]
struct View {
    rotation: [[f64; 3]; 3],
    focal: [f64; 2],
}

impl View {
    fn new(k: &CameraIntrinsics, center: [f64; 2]) -> Self {
        let mut direction = [(center[0] - k.cx) / k.fx, (center[1] - k.cy) / k.fy, 1.0];
        let norm = direction.iter().map(|v| v * v).sum::<f64>().sqrt();
        for value in &mut direction {
            *value /= norm;
        }
        let cross = [direction[1], -direction[0], 0.0];
        let skew = [
            [0.0, -cross[2], cross[1]],
            [cross[2], 0.0, -cross[0]],
            [-cross[1], cross[0], 0.0],
        ];
        let mut rotation = [[0.0; 3]; 3];
        for (i, row) in rotation.iter_mut().enumerate() {
            for (j, value) in row.iter_mut().enumerate() {
                *value = f64::from(i == j) + skew[i][j];
            }
            for (q, skew_row) in skew.iter().enumerate() {
                let scale = skew[i][q] / (1.0 + direction[2]);
                for (j, value) in row.iter_mut().enumerate() {
                    *value += scale * skew_row[j];
                }
            }
        }
        Self {
            rotation,
            focal: [k.fx, k.fy],
        }
    }

    fn local(self, k: &CameraIntrinsics, source: [f64; 2]) -> Option<[f64; 2]> {
        let direction = [(source[0] - k.cx) / k.fx, (source[1] - k.cy) / k.fy, 1.0];
        let mut projected = [0.0; 3];
        for (i, value) in projected.iter_mut().enumerate() {
            *value = (0..3).map(|j| self.rotation[i][j] * direction[j]).sum();
        }
        (projected[2] > 0.0).then_some([
            self.focal[0] * projected[0] / projected[2],
            self.focal[1] * projected[1] / projected[2],
        ])
    }

    fn source(self, k: &CameraIntrinsics, local: [f64; 2]) -> Option<[f64; 2]> {
        let direction = [local[0] / self.focal[0], local[1] / self.focal[1], 1.0];
        let mut projected = [0.0; 3];
        for (i, value) in projected.iter_mut().enumerate() {
            *value = (0..3).map(|j| self.rotation[j][i] * direction[j]).sum();
        }
        if projected[2] <= 0.0 {
            return None;
        }
        let source = [
            k.fx * projected[0] / projected[2] + k.cx,
            k.fy * projected[1] / projected[2] + k.cy,
        ];
        source
            .iter()
            .all(|value| value.is_finite())
            .then_some(source)
    }
}

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
        .expect("finite JSON number")
}

fn text(value: &Value) -> String {
    value.as_str().expect("JSON string").to_owned()
}

fn pair(value: &Value) -> [f64; 2] {
    [number(&value[0]), number(&value[1])]
}

fn bbox(value: &Value) -> [f64; 4] {
    [
        number(&value[0]),
        number(&value[1]),
        number(&value[2]),
        number(&value[3]),
    ]
}

fn distance(a: [f64; 2], b: [f64; 2]) -> f64 {
    (a[0] - b[0]).hypot(a[1] - b[1])
}

fn color_distance(a: [f64; 3], b: [f64; 3]) -> f64 {
    ((0..3).map(|i| (a[i] - b[i]).powi(2)).sum::<f64>() / 3.0).sqrt()
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    let n = values.len();
    if n.is_multiple_of(2) {
        (values[n / 2 - 1] + values[n / 2]) / 2.0
    } else {
        values[n / 2]
    }
}

fn sample(
    image: &RgbImage,
    k: &CameraIntrinsics,
    view: View,
    local: [f64; 2],
    roi: [f64; 4],
) -> Option<[f64; 3]> {
    let [x, y] = view.source(k, local)?;
    if x < roi[0]
        || y < roi[1]
        || x < 0.0
        || y < 0.0
        || x >= f64::from(k.source_width_px - 1)
        || y >= f64::from(k.source_height_px - 1)
        || x > roi[2]
        || y > roi[3]
    {
        return None;
    }
    let x0 = x.floor() as u32;
    let y0 = y.floor() as u32;
    let tx = x - f64::from(x0);
    let ty = y - f64::from(y0);
    let p00 = image.get_pixel(x0, y0).0;
    let p10 = image.get_pixel(x0 + 1, y0).0;
    let p01 = image.get_pixel(x0, y0 + 1).0;
    let p11 = image.get_pixel(x0 + 1, y0 + 1).0;
    let mut rgb = [0.0; 3];
    for channel in 0..3 {
        let top = f64::from(p00[channel]) * (1.0 - tx) + f64::from(p10[channel]) * tx;
        let bottom = f64::from(p01[channel]) * (1.0 - tx) + f64::from(p11[channel]) * tx;
        rgb[channel] = (top * (1.0 - ty) + bottom * ty) / f64::from(u8::MAX);
    }
    Some(rgb)
}

fn normalized_config(mut config: BallLocalizationConfig, width: u32) -> BallLocalizationConfig {
    let scale = f64::from(width) / f64::from(config.reference_width_px);
    config.background_probe_px *= scale;
    config
}

fn estimated_background(
    image: &RgbImage,
    k: &CameraIntrinsics,
    proposal: &BallProposal,
    config: &BallLocalizationConfig,
    view: View,
) -> Option<([f64; 3], f64)> {
    let [left, top, right, bottom] = proposal.bbox;
    let initial = [(left + right) / 2.0, (top + bottom) / 2.0];
    let half = (right - left).max(bottom - top) * config.roi_padding_factor / 2.0;
    let roi = [
        initial[0] - half,
        initial[1] - half,
        initial[0] + half,
        initial[1] + half,
    ];
    let edges = [
        [left, initial[1]],
        [right, initial[1]],
        [initial[0], top],
        [initial[0], bottom],
    ];
    let radii: Vec<_> = edges
        .iter()
        .filter_map(|point| view.local(k, *point))
        .collect();
    if radii.len() != 4 {
        return None;
    }
    let mut radius_values: Vec<_> = radii.iter().map(|p| distance(*p, [0.0, 0.0])).collect();
    let radius = median(&mut radius_values);
    let mut colors = [Vec::new(), Vec::new(), Vec::new()];
    for ray in 0..config.angular_samples {
        let angle = std::f64::consts::TAU * ray as f64 / config.angular_samples as f64;
        for factor in [config.annulus_inner_factor, config.annulus_outer_factor] {
            if let Some(rgb) = sample(
                image,
                k,
                view,
                [angle.cos() * radius * factor, angle.sin() * radius * factor],
                roi,
            ) {
                for ch in 0..3 {
                    colors[ch].push(rgb[ch]);
                }
            }
        }
    }
    if colors[0].is_empty() {
        return None;
    }
    let background = [
        median(&mut colors[0]),
        median(&mut colors[1]),
        median(&mut colors[2]),
    ];
    Some((background, radius))
}

fn image_set_from_manifest(corpus: &Value, reserved: &Value, image_id: &str) -> ImageSet {
    if image_id == "measured-grid-720" {
        let image = corpus["images"]
            .as_array()
            .unwrap()
            .iter()
            .find(|image| image["id"] == image_id)
            .unwrap();
        let proposals = image["ball_proposals"]["proposals"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|proposal| proposal["physical_reference_present"] == true)
            .map(|proposal| ProposalCase {
                id: text(&proposal["ball_id"]),
                detection_index: proposal["detection_index_in_frozen_audit"]
                    .as_u64()
                    .unwrap(),
                bbox: bbox(&proposal["bbox_xyxy_source_px"]),
                physical_reference_present: true,
                visibility: None,
            })
            .collect();
        return ImageSet {
            id: image_id.to_owned(),
            path: text(&image["path"]),
            sha256: text(&image["sha256"]),
            optics_id: "frozen_zero_radial_720".to_owned(),
            optics_status: text(&corpus["optics_models"]["frozen_zero_radial_720"]["status"]),
            proposals_are_detector_output: false,
            proposals,
        };
    }

    let proposal_set = reserved["proposal_sets"]
        .as_array()
        .unwrap()
        .iter()
        .find(|set| set["image_id"] == image_id)
        .unwrap();
    let proposals = proposal_set["proposals"]
        .as_array()
        .unwrap()
        .iter()
        .map(|proposal| {
            let center = pair(&proposal["center_px_approx"]);
            let side = number(&proposal["bbox_side_px"]);
            ProposalCase {
                id: text(&proposal["id"]),
                detection_index: proposal["detection_index"].as_u64().unwrap(),
                bbox: [
                    center[0] - side / 2.0,
                    center[1] - side / 2.0,
                    center[0] + side / 2.0,
                    center[1] + side / 2.0,
                ],
                physical_reference_present: false,
                visibility: proposal.get("visibility").map(text),
            }
        })
        .collect();
    ImageSet {
        id: image_id.to_owned(),
        path: text(&proposal_set["path"]),
        sha256: text(&proposal_set["sha256"]),
        optics_id: text(&proposal_set["optics_model_id"]),
        optics_status: text(&proposal_set["optics_status"]),
        proposals_are_detector_output: proposal_set["proposals_are_detector_output"] == true,
        proposals,
    }
}

fn intrinsics(corpus: &Value, model_id: &str, width: u32, height: u32) -> CameraIntrinsics {
    let model = &corpus["optics_models"][model_id];
    let camera = &model["camera_matrix"];
    let k = CameraIntrinsics {
        source_width_px: width,
        source_height_px: height,
        fx: number(&camera[0][0]),
        fy: number(&camera[1][1]),
        cx: number(&camera[0][2]),
        cy: number(&camera[1][2]),
    };
    k
}

fn json_point(point: [f64; 2]) -> Value {
    json!([point[0], point[1]])
}

fn color_delta(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn fit_color_evidence(
    image: &RgbImage,
    k: &CameraIntrinsics,
    proposal: &BallProposal,
    config: &BallLocalizationConfig,
    trace: &billiards_perception::ball_localization::BallLocalizationAuditTrace,
    candidate_evaluations: &[Value],
) -> Vec<Value> {
    let mut axis_by_iteration = BTreeMap::new();
    let mut candidate_points: BTreeMap<usize, BTreeMap<usize, Vec<(usize, [f64; 2], f64)>>> =
        BTreeMap::new();
    let mut selected_points: BTreeMap<(usize, usize), Vec<(usize, [f64; 2])>> = BTreeMap::new();
    let mut refined_fits: BTreeMap<(usize, usize), ([f64; 2], f64)> = BTreeMap::new();

    for event in &trace.events {
        let iteration = event.iteration.unwrap_or_default();
        match &event.data {
            BallLocalizationAuditEventData::IterationStarted { axis_image_px } => {
                axis_by_iteration.insert(iteration, *axis_image_px);
            }
            BallLocalizationAuditEventData::BoundaryCandidate {
                ray,
                point_local_px,
                strength,
                ..
            } => candidate_points
                .entry(iteration)
                .or_default()
                .entry(*ray)
                .or_default()
                .push((0, *point_local_px, *strength)),
            BallLocalizationAuditEventData::SelectedInlier {
                fit_id,
                ray,
                point_local_px,
                ..
            } => selected_points
                .entry((iteration, *fit_id))
                .or_default()
                .push((*ray, *point_local_px)),
            BallLocalizationAuditEventData::RefinedFit {
                fit_id,
                center_local_px,
                radius_local_px,
                ..
            } => {
                refined_fits.insert((iteration, *fit_id), (*center_local_px, *radius_local_px));
            }
            _ => {}
        }
    }

    // BoundaryCandidate trace events are emitted in production order: angular
    // ray order, then strongest distinct transition first within each ray.
    for rays in candidate_points.values_mut() {
        for candidates in rays.values_mut() {
            for (rank, candidate) in candidates.iter_mut().enumerate() {
                candidate.0 = rank;
            }
        }
    }

    let scaled = normalized_config(config.clone(), k.source_width_px);
    let mut qualified = Vec::new();
    for evaluation in candidate_evaluations {
        if evaluation["is_qualified"] != true {
            continue;
        }
        let iteration = evaluation["iteration"].as_u64().unwrap() as usize;
        let fit_id = evaluation["fit_id"].as_u64().unwrap() as usize;
        let Some(axis) = axis_by_iteration.get(&iteration).copied() else {
            continue;
        };
        let view = View::new(k, axis);
        for point in [[0.0, 0.0], [12.0, 0.0], [0.0, -12.0], [8.0, 9.0]] {
            let round_trip = view
                .source(k, point)
                .and_then(|source| view.local(k, source))
                .expect("local-view projection round trip");
            assert!(
                distance(point, round_trip) < 1e-8,
                "local-view projection failed round trip"
            );
        }
        let Some((background, estimated_radius)) =
            estimated_background(image, k, proposal, &scaled, view)
        else {
            continue;
        };
        let initial = [
            (proposal.bbox[0] + proposal.bbox[2]) / 2.0,
            (proposal.bbox[1] + proposal.bbox[3]) / 2.0,
        ];
        let half = (proposal.bbox[2] - proposal.bbox[0]).max(proposal.bbox[3] - proposal.bbox[1])
            * scaled.roi_padding_factor
            / 2.0;
        let roi = [
            initial[0] - half,
            initial[1] - half,
            initial[0] + half,
            initial[1] + half,
        ];
        let selected = selected_points
            .get(&(iteration, fit_id))
            .cloned()
            .unwrap_or_default();
        let mut per_ray = Vec::new();
        let mut rank_counts = BTreeMap::<usize, usize>::new();
        let mut inner_distances = Vec::new();
        let mut outer_distances = Vec::new();
        let mut strengths = Vec::new();
        for (ray, selected_point) in selected {
            let Some(candidates) = candidate_points
                .get(&iteration)
                .and_then(|rays| rays.get(&ray))
            else {
                continue;
            };
            let Some((rank, point_local, strength)) = candidates
                .iter()
                .min_by(|a, b| {
                    distance(a.1, selected_point).total_cmp(&distance(b.1, selected_point))
                })
                .copied()
                .filter(|candidate| distance(candidate.1, selected_point) < 1e-6)
            else {
                continue;
            };
            let radial_distance = distance(point_local, [0.0, 0.0]);
            if radial_distance <= scaled.background_probe_px {
                continue;
            }
            let direction = [
                point_local[0] / radial_distance,
                point_local[1] / radial_distance,
            ];
            let inside_local = [
                (radial_distance - scaled.background_probe_px) * direction[0],
                (radial_distance - scaled.background_probe_px) * direction[1],
            ];
            let outside_local = [
                (radial_distance + scaled.background_probe_px) * direction[0],
                (radial_distance + scaled.background_probe_px) * direction[1],
            ];
            let (Some(inside_rgb), Some(outside_rgb)) = (
                sample(image, k, view, inside_local, roi),
                sample(image, k, view, outside_local, roi),
            ) else {
                continue;
            };
            let inside_background_distance = color_distance(inside_rgb, background);
            let outside_background_distance = color_distance(outside_rgb, background);
            rank_counts
                .entry(rank)
                .and_modify(|count| *count += 1)
                .or_insert(1);
            inner_distances.push(inside_background_distance);
            outer_distances.push(outside_background_distance);
            strengths.push(strength);
            per_ray.push(json!({
                "ray": ray,
                "candidate_rank_by_gradient_strength": rank,
                "transition_local_px": json_point(point_local),
                "transition_source_px": view.source(k, point_local),
                "transition_radius_from_aiming_axis_px": radial_distance,
                "boundary_strength_rgb_rms_normalized": strength,
                "sample_offset_px": scaled.background_probe_px,
                "inside_rgb_normalized": inside_rgb,
                "outside_rgb_normalized": outside_rgb,
                "estimated_annulus_background_rgb_normalized": background,
                "inside_to_background_rgb_rms_normalized": inside_background_distance,
                "outside_to_background_rgb_rms_normalized": outside_background_distance,
                "inside_minus_outside_rgb_normalized": color_delta(inside_rgb, outside_rgb)
            }));
        }
        let Some((center_local, radius_local)) = refined_fits.get(&(iteration, fit_id)).copied()
        else {
            continue;
        };
        let mean = |values: &[f64]| {
            if values.is_empty() {
                None
            } else {
                Some(values.iter().sum::<f64>() / values.len() as f64)
            }
        };
        qualified.push(json!({
            "iteration": iteration,
            "fit_id": fit_id,
            "center_local_px": json_point(center_local),
            "center_image_px": evaluation["center_image_px"],
            "radius_local_px": radius_local,
            "initial_proposal_radius_local_px": estimated_radius,
            "support": evaluation["support"],
            "source_residual_rms_px": evaluation["source_residual_rms_px"],
            "rank_counts": rank_counts,
            "sampled_inlier_rays": per_ray.len(),
            "mean_inside_background_distance": mean(&inner_distances),
            "mean_outside_background_distance": mean(&outer_distances),
            "mean_transition_strength": mean(&strengths),
            "inlier_color_profiles_by_ray": per_ray
        }));
    }
    qualified
}

fn run_proposal(
    source: &RgbImage,
    intrinsics: &CameraIntrinsics,
    set: &ImageSet,
    proposal_case: &ProposalCase,
    case_dir: &Path,
) -> Value {
    let proposal = BallProposal {
        detection_index: proposal_case.detection_index,
        bbox: proposal_case.bbox,
    };
    let config = BallLocalizationConfig {
        batch_budget_ms: 10_000,
        proposal_budget_ms: 10_000,
        seed: 716,
        ransac_iterations: 128,
        maximum_refined_candidates: 16,
        ..BallLocalizationConfig::default()
    };
    let started = Instant::now();
    let (refinements, trace) = refine_ball_batch_with_audit(
        source.as_raw(),
        intrinsics,
        std::slice::from_ref(&proposal),
        &config,
        BallLocalizationAuditLimits::default(),
    )
    .expect("run the production revision-2 refiner with audit trace");
    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
    assert_eq!(refinements.len(), 1);
    assert_eq!(trace.dropped_events, 0);
    assert!(trace.events.len() < trace.maximum_events);

    let mut candidate_evaluations = Vec::new();
    let mut representative_rank = BTreeMap::<usize, usize>::new();
    for event in &trace.events {
        if let BallLocalizationAuditEventData::CandidateEvaluation {
            fit_id,
            center_image_px,
            supported_rays,
            residual_rms_px,
            source_residual_rms_px,
            rejection,
            quality,
            outline_image_px,
        } = &event.data
        {
            let iteration = event.iteration.unwrap_or_default();
            let rank = representative_rank
                .entry(iteration)
                .and_modify(|rank| *rank += 1)
                .or_insert(1);
            let outline: Vec<[f64; 2]> =
                outline_image_px.iter().filter_map(|point| *point).collect();
            candidate_evaluations.push(json!({
                "iteration": iteration,
                "representative_evaluation_rank": rank,
                "fit_id": fit_id,
                "center_image_px": center_image_px,
                "support": supported_rays,
                "residual_rms_px": residual_rms_px,
                "source_residual_rms_px": source_residual_rms_px,
                "rejection": rejection,
                "is_qualified": rejection.is_none() && center_image_px.is_some() && outline.len() >= 4,
                "outline_image_px": outline,
                "quality": quality
            }));
        }
    }
    let color_evidence = fit_color_evidence(
        source,
        intrinsics,
        &proposal,
        &config,
        &trace,
        &candidate_evaluations,
    );
    let refinement = serde_json::to_value(&refinements[0]).expect("serialize result");
    let case_id = format!("{}--{}", set.id, proposal_case.id);
    let trace_value = serde_json::to_value(&trace).expect("serialize trace");
    let case = json!({
        "schema_version": 1,
        "case_id": case_id,
        "image_id": set.id,
        "source_image": set.path,
        "source_image_sha256": set.sha256,
        "source_dimensions_px": [source.width(), source.height()],
        "proposal_id": proposal_case.id,
        "proposal_bbox_xyxy_source_px": proposal.bbox,
        "proposal_origin": if proposal_case.physical_reference_present { "saved_manual_reference_scene" } else { "saved_manual_coarse_region" },
        "physical_reference_present": proposal_case.physical_reference_present,
        "visibility": proposal_case.visibility,
        "proposals_are_detector_output": set.proposals_are_detector_output,
        "optics_model_id": set.optics_id,
        "optics_status": set.optics_status,
        "intrinsics": intrinsics,
        "refiner_revision": billiards_perception::ball_localization_profile::BALL_REFINER_REVISION,
        "audit_schema_version": trace.schema_version,
        "production_settings": {
            "seed": config.seed,
            "ransac_iterations": config.ransac_iterations,
            "candidate_cap": config.maximum_refined_candidates,
            "max_aiming_iterations": config.max_iterations,
            "quality_gates": "BallLocalizationConfig::default()"
        },
        "diagnostic_budgets_ms": {
            "batch": config.batch_budget_ms,
            "proposal": config.proposal_budget_ms
        },
        "development_host_trace_enabled_elapsed_ms": elapsed_ms,
        "refinement": refinement,
        "candidate_evaluations": candidate_evaluations,
        "qualified_fit_color_evidence": color_evidence,
        "trace": trace_value,
        "limitations": [
            "No new image, camera capture, user measurement, or positioning jig was used.",
            "All proposals are saved manual regions, not detector outputs.",
            "The optics models are diagnostic and not qualified for runtime.",
            "Color profiles reconstruct the current estimator's local-view transform and bilinear RGB sampling from saved JPEGs; they do not label a true physical silhouette.",
            "User-reported physical-reference uncertainty is unknown and is not used to select a fit.",
            "Trace-enabled development-host elapsed time is diagnostic only."
        ]
    });
    let path = case_dir.join(format!("{case_id}.json"));
    fs::write(&path, serde_json::to_vec_pretty(&case).unwrap()).expect("write case file");
    json!({
        "case_id": case_id,
        "image_id": set.id,
        "proposal_id": proposal_case.id,
        "physical_reference_present": proposal_case.physical_reference_present,
        "status": refinement["status"],
        "reason": refinement["reason"],
        "candidate_evaluation_count": candidate_evaluations.len(),
        "qualified_fit_count": color_evidence.len(),
        "qualified_fit_rank_profiles": color_evidence.iter().map(|fit| json!({
            "iteration": fit["iteration"],
            "fit_id": fit["fit_id"],
            "support": fit["support"],
            "rank_counts": fit["rank_counts"],
            "mean_inside_background_distance": fit["mean_inside_background_distance"],
            "mean_outside_background_distance": fit["mean_outside_background_distance"],
            "mean_transition_strength": fit["mean_transition_strength"]
        })).collect::<Vec<_>>(),
        "trace_event_count": trace.events.len(),
        "trace_dropped_events": trace.dropped_events,
        "trace_sha256": file_sha256(path.to_str().unwrap()),
        "case_path": format!("cases/{}.json", path.file_stem().unwrap().to_string_lossy()),
        "elapsed_with_trace_ms_development_host": elapsed_ms
    })
}

fn load_set(set: &ImageSet, corpus: &Value) -> (RgbImage, CameraIntrinsics) {
    let jpeg = fs::read(&set.path).expect("read saved input JPEG");
    assert_eq!(sha256(&jpeg), set.sha256, "source image hash changed");
    let image = image::load_from_memory(&jpeg)
        .expect("decode saved JPEG with the project image codec")
        .to_rgb8();
    let intrinsics = intrinsics(corpus, &set.optics_id, image.width(), image.height());
    (image, intrinsics)
}

fn base64(bytes: &[u8]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut encoded = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let a = chunk[0];
        let b = chunk.get(1).copied().unwrap_or(0);
        let c = chunk.get(2).copied().unwrap_or(0);
        encoded.push(ALPHABET[(a >> 2) as usize] as char);
        encoded.push(ALPHABET[(((a & 0x03) << 4) | (b >> 4)) as usize] as char);
        encoded.push(if chunk.len() > 1 {
            ALPHABET[(((b & 0x0f) << 2) | (c >> 6)) as usize] as char
        } else {
            '='
        });
        encoded.push(if chunk.len() > 2 {
            ALPHABET[(c & 0x3f) as usize] as char
        } else {
            '='
        });
    }
    encoded
}

fn xml_escape(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

fn write_case_overlays(sets: &[ImageSet], case_summaries: &[Value], output_root: &Path) {
    let overlay_dir = output_root.join("overlays");
    fs::create_dir_all(&overlay_dir).expect("create overlay folder");
    let palette = [
        "#ff3b30", "#ff9500", "#ffcc00", "#34c759", "#00c7be", "#32ade6", "#007aff", "#5856d6",
        "#af52de", "#ff2d55", "#8e8e93", "#a2845e",
    ];
    const PANEL: f64 = 360.0;
    const COLUMNS: usize = 3;
    for set in sets {
        let jpeg = fs::read(&set.path).expect("read source image for overlay");
        let source = image::load_from_memory(&jpeg)
            .expect("decode overlay background")
            .to_rgb8();
        let data_uri = format!("data:image/jpeg;base64,{}", base64(&jpeg));
        let cases: Vec<_> = case_summaries
            .iter()
            .filter(|row| row["image_id"] == set.id)
            .collect();
        let rows = cases.len().div_ceil(COLUMNS);
        let mut montage = format!(
            "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"1080\" height=\"{}\" viewBox=\"0 0 1080 {}\"><defs><image id=\"source\" href=\"{}\" x=\"0\" y=\"0\" width=\"{}\" height=\"{}\"/></defs><rect width=\"100%\" height=\"100%\" fill=\"#111\"/>",
            rows as f64 * PANEL,
            rows as f64 * PANEL,
            data_uri,
            source.width(),
            source.height()
        );
        for (case_index, summary) in cases.into_iter().enumerate() {
            let case_path = output_root.join(summary["case_path"].as_str().unwrap());
            let case: Value =
                serde_json::from_slice(&fs::read(case_path).expect("read case output"))
                    .expect("parse case output");
            let bbox = bbox(&case["proposal_bbox_xyxy_source_px"]);
            let panel_x = (case_index % COLUMNS) as f64 * PANEL;
            let panel_y = (case_index / COLUMNS) as f64 * PANEL;
            let crop_height = PANEL - 28.0;
            let mut crop = format!(
                "<svg x=\"{}\" y=\"{}\" width=\"{}\" height=\"{}\" viewBox=\"{} {} {} {}\"><use href=\"#source\"/><rect x=\"{}\" y=\"{}\" width=\"{}\" height=\"{}\" fill=\"none\" stroke=\"#ffffff\" stroke-width=\"0.5\" stroke-dasharray=\"2 2\"/>",
                panel_x,
                panel_y,
                PANEL,
                crop_height,
                bbox[0] - 12.0,
                bbox[1] - 12.0,
                bbox[2] - bbox[0] + 24.0,
                bbox[3] - bbox[1] + 24.0,
                bbox[0],
                bbox[1],
                bbox[2] - bbox[0],
                bbox[3] - bbox[1]
            );
            for (index, evaluation) in case["candidate_evaluations"]
                .as_array()
                .unwrap()
                .iter()
                .filter(|evaluation| evaluation["is_qualified"] == true)
                .enumerate()
            {
                let color = palette[index % palette.len()];
                let center = pair(&evaluation["center_image_px"]);
                let points = evaluation["outline_image_px"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|point| {
                        let point = pair(point);
                        format!("{:.4},{:.4}", point[0], point[1])
                    })
                    .collect::<Vec<_>>()
                    .join(" ");
                crop.push_str(&format!(
                    "<polyline points=\"{}\" fill=\"none\" stroke=\"{}\" stroke-width=\"0.35\" opacity=\"0.9\"/>",
                    points, color
                ));
                crop.push_str(&format!(
                    "<path d=\"M {:.3} {:.3} m -1.5 -1.5 l 3 3 m 0 -3 l -3 3\" stroke=\"{}\" stroke-width=\"0.45\"/>",
                    center[0], center[1], color
                ));
            }
            crop.push_str("</svg>");
            montage.push_str(&crop);
            let full_id = summary["proposal_id"].as_str().unwrap();
            let short_id = if full_id.chars().count() > 20 {
                format!("{}…", full_id.chars().take(19).collect::<String>())
            } else {
                full_id.to_owned()
            };
            montage.push_str(&format!(
                "<text x=\"{}\" y=\"{}\" font-family=\"sans-serif\" font-size=\"13\" fill=\"white\">{} {}: {} fits</text>",
                panel_x + 6.0,
                panel_y + PANEL - 7.0,
                case_index + 1,
                xml_escape(&short_id),
                case["qualified_fit_color_evidence"].as_array().unwrap().len()
            ));
        }
        montage.push_str("</svg>");
        fs::write(overlay_dir.join(format!("{}-crops.svg", set.id)), montage)
            .expect("write crop montage overlay");
    }
}

fn main() {
    let corpus_bytes = fs::read(CORPUS_PATH).expect("read frozen corpus");
    let corpus: Value = serde_json::from_slice(&corpus_bytes).expect("parse corpus");
    let reserved_bytes = fs::read(RESERVED_PATH).expect("read saved coarse proposals");
    let reserved: Value = serde_json::from_slice(&reserved_bytes).expect("parse proposal file");
    let sets = [
        image_set_from_manifest(&corpus, &reserved, "measured-grid-720"),
        image_set_from_manifest(&corpus, &reserved, "alternate-grid-720"),
        image_set_from_manifest(&corpus, &reserved, "framing-after-4k"),
    ];
    let output_root = PathBuf::from(OUTPUT_ROOT);
    let case_dir = output_root.join("cases");
    fs::create_dir_all(&case_dir).expect("create audit output directory");
    let mut results = Vec::new();
    let mut input_records = Vec::new();
    for set in &sets {
        let (source, intrinsics) = load_set(set, &corpus);
        input_records.push(json!({
            "image_id": set.id,
            "path": set.path,
            "sha256": set.sha256,
            "dimensions_px": [source.width(), source.height()],
            "optics_model_id": set.optics_id,
            "optics_status": set.optics_status,
            "proposal_count": set.proposals.len(),
            "proposals_are_detector_output": set.proposals_are_detector_output
        }));
        for proposal in &set.proposals {
            results.push(run_proposal(&source, &intrinsics, set, proposal, &case_dir));
        }
    }
    write_case_overlays(&sets, &results, &output_root);
    let report = json!({
        "schema_version": 1,
        "date": "2026-10-27",
        "purpose": "Characterize raw RGB transitions and per-ray gradient ranks across qualified competing revision-2 ball-boundary fits.",
        "diagnostics_only": true,
        "production_behavior_changed": false,
        "corpus": { "path": CORPUS_PATH, "sha256": sha256(&corpus_bytes) },
        "reserved_proposals": { "path": RESERVED_PATH, "sha256": sha256(&reserved_bytes) },
        "refiner_source": { "path": REFINER_PATH, "sha256": file_sha256(REFINER_PATH) },
        "harness": { "path": HARNESS_PATH, "sha256": file_sha256(HARNESS_PATH) },
        "audit_schema_version": billiards_perception::ball_localization::BALL_LOCALIZATION_AUDIT_VERSION,
        "refiner_revision": billiards_perception::ball_localization_profile::BALL_REFINER_REVISION,
        "settings": {
            "seed": 716,
            "ransac_iterations": 128,
            "candidate_cap": 16,
            "max_aiming_iterations": BallLocalizationConfig::default().max_iterations,
            "quality_gates": "BallLocalizationConfig::default()",
            "batch_and_proposal_diagnostic_budgets_ms": 10000,
            "colorspace": "RGB normalized to [0,1], matching the refiner source buffer",
            "profile_sample_offsets": "transition radius plus/minus the scaled production background_probe_px"
        },
        "inputs": input_records,
        "run_count": results.len(),
        "results": results,
        "interpretation_limits": [
            "Qualified candidate fits are a competing-fit set, not true-center labels.",
            "The nine physical positions were not used to select any fit and have unknown uncertainty.",
            "The held-out scenes were previously seen, have manually chosen crop regions, and have no independent X/Y truth.",
            "A similar pixel profile does not establish which outline is the physical ball silhouette.",
            "The audit cannot establish millimeter accuracy or production availability."
        ]
    });
    fs::write(
        output_root.join("summary.json"),
        serde_json::to_vec_pretty(&report).unwrap(),
    )
    .expect("write summary");
    println!(
        "Contour-color audit complete: {} proposals, {} qualified fit profiles.",
        report["run_count"].as_u64().unwrap(),
        report["results"]
            .as_array()
            .unwrap()
            .iter()
            .map(|case| case["qualified_fit_count"].as_u64().unwrap())
            .sum::<u64>()
    );
}
