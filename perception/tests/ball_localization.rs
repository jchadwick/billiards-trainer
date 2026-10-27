use billiards_perception::ball_center_calibration::{
    solve_ball_center_calibration, BallCenterCalibrationSolverInput,
    BallCenterCalibrationSolverPolicy, BallCenterCorrespondence,
};
use billiards_perception::ball_localization::*;
use billiards_perception::ball_localization_profile::{
    BallLocalizationSettings, CameraOpticsProfileV1, LoadedBallLocalizationProfile,
    BALL_REFINER_REVISION, CAMERA_OPTICS_PROFILE_VERSION,
};
use billiards_perception::calibration::{ImagePointPx, TablePointMm};
use billiards_perception::detection_batch::{
    decode_detection_batch_v1, ImageSpace as BatchImageSpace,
};
use billiards_perception::ImageSpace as CalibrationImageSpace;

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
            let bg = [24., 145., 118.];
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
#[test]
fn recovers_native_source_centers_for_perspective_circles() {
    for center in [[240., 150.], [75., 145.], [405., 145.]] {
        let (rgb, p) = image(center, false, false);
        let result =
            refine_ball_batch(&rgb, &intrinsics(), std::slice::from_ref(&p), &config()).unwrap();
        let result = &result[0];
        assert_eq!(
            result.status,
            RefinementStatus::Accepted,
            "{center:?}: {result:?}"
        );
        let observed = result.center_image_px.unwrap();
        assert!(
            (observed[0] - center[0]).hypot(observed[1] - center[1]) < 1.,
            "{result:?}"
        );
        assert!(result.outline.as_ref().unwrap().local_radius_px > 15.);
    }
}
#[test]
fn stripe_and_soft_shadow_do_not_become_the_boundary() {
    let (rgb, p) = image([240., 150.], true, true);
    let result = refine_ball_batch(&rgb, &intrinsics(), &[p], &config()).unwrap();
    assert_eq!(
        result[0].status,
        RefinementStatus::Accepted,
        "{:?}",
        result[0]
    );
    let c = result[0].center_image_px.unwrap();
    assert!((c[0] - 240.).hypot(c[1] - 150.) < 1.5);
}
#[test]
fn highlight_fragment_is_rejected() {
    let mut rgb = vec![0; 480 * 300 * 3];
    for p in rgb.as_mut_slice().as_chunks_mut::<3>().0 {
        p.copy_from_slice(&[24, 145, 118]);
    }
    for y in 147..153 {
        for x in 237..243 {
            rgb[(y * 480 + x) * 3..(y * 480 + x) * 3 + 3].copy_from_slice(&[255, 255, 255]);
        }
    }
    let p = BallProposal {
        detection_index: 0,
        bbox: [220., 130., 260., 170.],
    };
    let r = refine_ball_batch(&rgb, &intrinsics(), &[p], &config()).unwrap();
    assert_eq!(r[0].status, RefinementStatus::Rejected);
    assert!(r[0].center_image_px.is_none());
}
#[test]
fn validates_buffer_configuration_intrinsics_and_duplicate_identity() {
    let (rgb, p) = image([240., 150.], false, false);
    let mut cfg = config();
    cfg.minimum_coverage = f64::NAN;
    assert_eq!(
        cfg.validate(),
        Err(BallLocalizationError::InvalidConfiguration)
    );
    let mut cfg = config();
    cfg.maximum_refined_candidates = 17;
    assert_eq!(
        cfg.validate(),
        Err(BallLocalizationError::InvalidConfiguration)
    );
    let mut cfg = config();
    cfg.maximum_inlier_reselection_rounds = 4;
    assert_eq!(
        cfg.validate(),
        Err(BallLocalizationError::InvalidConfiguration)
    );
    assert_eq!(
        refine_ball_batch(
            &rgb[..10],
            &intrinsics(),
            std::slice::from_ref(&p),
            &config()
        ),
        Err(BallLocalizationError::InvalidRgbBuffer)
    );
    assert_eq!(
        refine_ball_batch(&rgb, &intrinsics(), &[p.clone(), p], &config()),
        Err(BallLocalizationError::DuplicateDetectionIndex)
    );
    let mut k = intrinsics();
    k.fx = 0.;
    assert_eq!(k.validate(), Err(BallLocalizationError::InvalidIntrinsics));
    let parsed: BallLocalizationConfig = serde_json::from_str("{}").unwrap();
    assert_eq!(parsed, BallLocalizationConfig::default());
}

#[test]
fn checked_in_defaults_match_typed_candidate_search_configuration() {
    let defaults: serde_json::Value =
        serde_json::from_str(include_str!("../contracts/configuration/v1/defaults.json")).unwrap();
    let parameters: BallLocalizationConfig = serde_json::from_value(
        defaults["perception"]["rust_live"]["ball_localization"]["parameters"].clone(),
    )
    .unwrap();
    assert_eq!(parameters, BallLocalizationConfig::default());
}
#[test]
fn exhausted_deadline_retains_one_rejection_per_proposal() {
    let (rgb, p) = image([240., 150.], false, false);
    let cfg = BallLocalizationConfig {
        batch_budget_ms: 0,
        ..config()
    };
    let r = refine_ball_batch(&rgb, &intrinsics(), &[p], &cfg).unwrap();
    assert_eq!(r.len(), 1);
    assert_eq!(r[0].reason, Some(RejectionReason::DeadlineExceeded));
}

#[test]
fn audit_traces_the_same_fitting_results_with_bounded_events() {
    let (rgb, proposal) = image([240.0, 150.0], true, false);
    let baseline = refine_ball_batch(
        &rgb,
        &intrinsics(),
        std::slice::from_ref(&proposal),
        &config(),
    )
    .unwrap();
    let (observations, trace) = refine_ball_batch_with_audit(
        &rgb,
        &intrinsics(),
        std::slice::from_ref(&proposal),
        &config(),
        BallLocalizationAuditLimits::default(),
    )
    .unwrap();
    assert_eq!(observations, baseline);
    assert_eq!(BALL_LOCALIZATION_AUDIT_VERSION, 3);
    assert_eq!(trace.schema_version, 3);
    assert_eq!(trace.dropped_events, 0);
    assert!(trace.retained_bytes <= trace.maximum_bytes);
    assert_eq!(trace.maximum_bytes, MAX_BALL_LOCALIZATION_AUDIT_BYTES);
    assert!(trace
        .events
        .iter()
        .all(|e| e.detection_index == proposal.detection_index));
    assert!(trace.events.iter().any(|e| matches!(
        e.data,
        BallLocalizationAuditEventData::BoundaryCandidate {
            point_image_px: Some(_),
            ..
        }
    )));
    assert!(trace.events.iter().any(|e| matches!(
        e.data,
        BallLocalizationAuditEventData::RawHypothesis {
            center_image_px: Some(_),
            ..
        }
    )));
    for hypothesis in &trace.events {
        let BallLocalizationAuditEventData::RawHypothesis {
            sampled_candidate_indices,
            sampled_candidate_rays,
            ..
        } = &hypothesis.data
        else {
            continue;
        };
        let candidate_rays = trace
            .events
            .iter()
            .filter(|candidate| candidate.iteration == hypothesis.iteration)
            .filter_map(|candidate| match &candidate.data {
                BallLocalizationAuditEventData::BoundaryCandidate { ray, .. } => Some(*ray),
                _ => None,
            })
            .collect::<Vec<_>>();
        for slot in 0..3 {
            assert_eq!(
                candidate_rays[sampled_candidate_indices[slot]],
                sampled_candidate_rays[slot]
            );
        }
    }
    assert!(trace.events.iter().any(|e| matches!(
        e.data,
        BallLocalizationAuditEventData::RefinedFit {
            center_image_px: Some(_),
            ..
        }
    )));
    assert!(trace.events.iter().any(|e| matches!(
        e.data,
        BallLocalizationAuditEventData::SelectedInlier {
            point_image_px: Some(_),
            ..
        }
    )));
    assert!(trace.events.iter().any(|event| matches!(
        &event.data,
        BallLocalizationAuditEventData::CandidateEvaluation {
            source_residual_rms_px: Some(_),
            outline_image_px,
            ..
        } if !outline_image_px.is_empty()
    )));
    let outcome = trace.events.last().unwrap();
    assert_eq!(outcome.stage, observations[0].quality.stage);
    assert!(matches!(
        outcome.data,
        BallLocalizationAuditEventData::Outcome { .. }
    ));

    let limits = BallLocalizationAuditLimits {
        maximum_events: 1,
        ..BallLocalizationAuditLimits::default()
    };
    let (truncated, trace) =
        refine_ball_batch_with_audit(&rgb, &intrinsics(), &[proposal], &config(), limits).unwrap();
    assert_eq!(truncated, baseline);
    assert_eq!(trace.events.len(), 1);
    assert!(trace.dropped_events > 0);
}

#[test]
fn detection_identity_and_batch_order_do_not_change_localization_geometry() {
    let (rgb, template) = image([240.0, 150.0], true, false);
    let mut cfg = config();
    cfg.batch_budget_ms = 5_000;
    cfg.proposal_budget_ms = 400;

    let proposals = (0..16)
        .map(|detection_index| BallProposal {
            detection_index,
            ..template.clone()
        })
        .collect::<Vec<_>>();
    let individual = proposals
        .iter()
        .map(|proposal| {
            refine_ball_batch(&rgb, &intrinsics(), std::slice::from_ref(proposal), &cfg)
                .unwrap()
                .into_iter()
                .next()
                .unwrap()
        })
        .collect::<Vec<_>>();
    let expected = &individual[0];
    for result in &individual[1..] {
        assert_eq!(result.center_image_px, expected.center_image_px);
        assert_eq!(result.status, expected.status);
        assert_eq!(result.reason, expected.reason);
        assert_eq!(result.quality, expected.quality);
        assert_eq!(result.outline, expected.outline);
    }

    let reversed = proposals.iter().rev().cloned().collect::<Vec<_>>();
    let batched = refine_ball_batch(&rgb, &intrinsics(), &reversed, &cfg).unwrap();
    for result in batched {
        let expected = &individual[result.detection_index as usize];
        assert_eq!(result.center_image_px, expected.center_image_px);
        assert_eq!(result.status, expected.status);
        assert_eq!(result.reason, expected.reason);
        assert_eq!(result.quality, expected.quality);
        assert_eq!(result.outline, expected.outline);
    }
}

#[test]
fn audit_distinguishes_uncomputed_invalid_proposal_and_capacity_rejection() {
    let (rgb, proposal) = image([240.0, 150.0], false, false);
    let invalid = BallProposal {
        detection_index: 12,
        bbox: [-1.0, 0.0, 10.0, 10.0],
    };
    let mut cfg = config();
    cfg.max_proposals = 1;
    let limits = BallLocalizationAuditLimits {
        maximum_events: 20,
        ..BallLocalizationAuditLimits::default()
    };
    let (results, trace) =
        refine_ball_batch_with_audit(&rgb, &intrinsics(), &[invalid, proposal], &cfg, limits)
            .unwrap();
    assert_eq!(results[0].reason, Some(RejectionReason::InvalidProposal));
    assert_eq!(results[0].quality.stage, BallFitStage::ProposalValidation);
    assert_eq!(results[0].quality.iterations, 0);
    assert_eq!(results[0].quality.boundary_candidates, None);
    assert_eq!(results[0].quality.supported_ray_fraction, None);
    assert_eq!(
        results[1].reason,
        Some(RejectionReason::ProposalLimitExceeded)
    );
    assert_eq!(results[1].quality.stage, BallFitStage::ProposalCapacity);
    assert_eq!(trace.events.len(), 2);
    assert_eq!(trace.events[0].iteration, None);
    assert_eq!(trace.events[1].stage, BallFitStage::ProposalCapacity);
    assert_eq!(
        refine_ball_batch_with_audit(
            &rgb,
            &intrinsics(),
            &[],
            &config(),
            BallLocalizationAuditLimits {
                maximum_events: 0,
                ..limits
            },
        ),
        Err(BallLocalizationError::InvalidAuditLimits)
    );
    assert_eq!(
        refine_ball_batch_with_audit(
            &rgb,
            &intrinsics(),
            &[],
            &config(),
            BallLocalizationAuditLimits {
                maximum_events: MAX_BALL_LOCALIZATION_AUDIT_EVENTS + 1,
                ..limits
            },
        ),
        Err(BallLocalizationError::InvalidAuditLimits)
    );
    assert_eq!(
        refine_ball_batch_with_audit(
            &rgb,
            &intrinsics(),
            &[],
            &config(),
            BallLocalizationAuditLimits {
                maximum_bytes: 1,
                ..limits
            },
        ),
        Err(BallLocalizationError::InvalidAuditLimits)
    );
    assert_eq!(
        refine_ball_batch_with_audit(
            &rgb,
            &intrinsics(),
            &[],
            &config(),
            BallLocalizationAuditLimits {
                maximum_bytes: MAX_BALL_LOCALIZATION_AUDIT_BYTES + 1,
                ..limits
            },
        ),
        Err(BallLocalizationError::InvalidAuditLimits)
    );
}

#[test]
fn proposal_capacity_keeps_visible_excess_results_rejected() {
    let (rgb, proposal) = image([240.0, 150.0], false, false);
    let mut second = proposal.clone();
    second.detection_index += 1;
    let proposals = [proposal, second];
    let mut cfg = config();
    cfg.max_proposals = 1;
    let results = refine_ball_batch(&rgb, &intrinsics(), &proposals, &cfg).unwrap();
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].status, RefinementStatus::Accepted);
    assert_eq!(results[1].status, RefinementStatus::Rejected);
    assert_eq!(
        results[1].reason,
        Some(RejectionReason::ProposalLimitExceeded)
    );
}

#[test]
fn a_half_occluded_ball_fails_closed_instead_of_fitting_a_full_circle() {
    let (mut rgb, p) = image([240., 150.], false, false);
    // Hide one half of the ball in cloth. A fitted circle through the remaining
    // bright semicircle alone is insufficient evidence of a complete boundary.
    for y in 125..175 {
        for x in 215..240 {
            rgb[(y * 480 + x) * 3..(y * 480 + x) * 3 + 3].copy_from_slice(&[24, 145, 118]);
        }
    }
    let result = refine_ball_batch(&rgb, &intrinsics(), &[p], &config()).unwrap();
    assert_eq!(
        result[0].status,
        RefinementStatus::Rejected,
        "{:?}",
        result[0]
    );
    assert!(result[0].center_image_px.is_none());
}

#[test]
fn oversized_and_invalid_proposals_have_structured_individual_failures() {
    let (rgb, _) = image([240., 150.], false, false);
    let proposals = [
        BallProposal {
            detection_index: 1,
            bbox: [-1., 0., 10., 10.],
        },
        BallProposal {
            detection_index: 2,
            bbox: [0., 0., 480., 300.],
        },
    ];
    let mut cfg = config();
    cfg.max_roi_side_px = 100;
    let result = refine_ball_batch(&rgb, &intrinsics(), &proposals, &cfg).unwrap();
    assert_eq!(result[0].reason, Some(RejectionReason::InvalidProposal));
    assert_eq!(result[1].reason, Some(RejectionReason::RoiTooLarge));
}

#[test]
fn absolute_resource_ceilings_reject_pathological_settings_before_allocation() {
    let defaults = BallLocalizationConfig::default();
    let mut cases = Vec::new();
    let mut c = defaults.clone();
    c.max_proposals = usize::MAX;
    cases.push(c);
    let mut c = defaults.clone();
    c.angular_samples = usize::MAX;
    cases.push(c);
    let mut c = defaults.clone();
    c.radial_samples = usize::MAX;
    cases.push(c);
    let mut c = defaults.clone();
    c.candidates_per_ray = usize::MAX;
    cases.push(c);
    let mut c = defaults.clone();
    c.ransac_iterations = usize::MAX;
    cases.push(c);
    let mut c = defaults.clone();
    c.irls_iterations = usize::MAX;
    cases.push(c);
    let mut c = defaults.clone();
    c.outline_samples = usize::MAX;
    cases.push(c);
    let mut c = defaults.clone();
    c.angular_sectors = usize::MAX;
    cases.push(c);
    let mut c = defaults.clone();
    c.max_roi_side_px = u32::MAX;
    cases.push(c);
    let mut c = defaults.clone();
    c.patch_size_px = u32::MAX;
    cases.push(c);
    let mut c = defaults.clone();
    c.patch_size_px = 194;
    cases.push(c);
    let mut c = defaults.clone();
    c.batch_budget_ms = u64::MAX;
    cases.push(c);
    let mut c = defaults.clone();
    c.proposal_budget_ms = u64::MAX;
    cases.push(c);
    for c in cases {
        assert_eq!(
            c.validate(),
            Err(BallLocalizationError::InvalidConfiguration)
        );
    }
    assert!(defaults.validate().is_ok());
}

fn resolver_batch() -> billiards_perception::detection_batch::DetectionBatchV1 {
    decode_detection_batch_v1(
        include_str!("../contracts/detection-batch/v1/detection-batch-success.json").as_bytes(),
    )
    .unwrap()
}

fn resolver_profile() -> LoadedBallLocalizationProfile {
    LoadedBallLocalizationProfile {
        settings: BallLocalizationSettings::default(),
        optics: Some(CameraOpticsProfileV1 {
            schema_version: CAMERA_OPTICS_PROFILE_VERSION,
            optics_profile_id: "optics-test".to_owned(),
            camera_profile_id: "camera-profile-test".to_owned(),
            optics_revision: 2,
            refiner_revision: BALL_REFINER_REVISION,
            image_space: CalibrationImageSpace::Raw,
            intrinsics: CameraIntrinsics {
                source_width_px: 3840,
                source_height_px: 2160,
                fx: 2100.0,
                fy: 2100.0,
                cx: 1920.0,
                cy: 1080.0,
            },
            provenance: "resolver unit test".to_owned(),
        }),
        calibration: None,
        calibration_error: None,
    }
}

fn accepted_refinement(
    batch: &billiards_perception::detection_batch::DetectionBatchV1,
) -> BallRefinementBatch {
    BallRefinementBatch {
        source_width_px: batch.source.width_px,
        source_height_px: batch.source.height_px,
        camera_profile_id: "camera-profile-test".to_owned(),
        image_space: BatchImageSpace::Raw,
        configuration_revision: 7,
        optics_profile_id: "optics-test".to_owned(),
        optics_revision: 2,
        refiner_revision: BALL_REFINER_REVISION,
        table_revision: 1,
        refinement_elapsed_ms: 1.0,
        identity: Some(BallRefinementIdentity {
            source_session_id: batch.source.session_id.clone(),
            frame_id: batch.source.frame_id.clone(),
            configuration_revision: 7,
            calibration_revision: 4,
            motion_generation: 12,
        }),
        observations: vec![BallRefinement {
            detection_index: batch.detections[0].detection_index,
            center_image_px: Some([1880.25, 1022.75]),
            status: RefinementStatus::Accepted,
            reason: None,
            quality: BallLocalizationQuality::default(),
            outline: None,
        }],
    }
}

fn resolver_calibration(
) -> billiards_perception::ball_center_calibration::ValidatedBallCenterCalibrationV1 {
    let table_width_mm = 2336.8;
    let table_height_mm = 1155.7;
    let correspondences = [
        (120.0, 100.0),
        (1300.0, 100.0),
        (2500.0, 100.0),
        (3700.0, 100.0),
        (120.0, 1000.0),
        (1300.0, 1000.0),
        (2500.0, 1000.0),
        (3700.0, 1000.0),
        (120.0, 2000.0),
        (1300.0, 2000.0),
        (2500.0, 2000.0),
        (3700.0, 2000.0),
    ]
    .into_iter()
    .map(|(x_px, y_px)| BallCenterCorrespondence {
        image_point_px: ImagePointPx { x_px, y_px },
        center_table_mm: TablePointMm {
            x_mm: (x_px / 3839.0 - 0.5) * table_width_mm,
            y_mm: (y_px / 2159.0 - 0.5) * table_height_mm,
        },
        provenance: format!("synthetic independent locator point {x_px},{y_px}"),
    })
    .collect();
    solve_ball_center_calibration(BallCenterCalibrationSolverInput {
        calibration_revision: 4,
        camera_profile_id: "camera-profile-test".to_owned(),
        optics_profile_id: "optics-test".to_owned(),
        optics_revision: 2,
        refiner_revision: BALL_REFINER_REVISION,
        table_revision: 1,
        rectification_revision: 0,
        image_space: CalibrationImageSpace::Raw,
        source_width_px: 3840,
        source_height_px: 2160,
        table_width_mm,
        table_height_mm,
        ball_diameter_mm: 57.15,
        provenance: "synthetic test calibration, not physical accuracy evidence".to_owned(),
        solver_policy: BallCenterCalibrationSolverPolicy::default(),
        correspondences,
    })
    .unwrap()
    .into_validated()
    .unwrap()
}

#[test]
fn resolver_selects_only_matching_refinement_and_never_invents_table_xy() {
    let batch = resolver_batch();
    let profile = resolver_profile();
    let expected_identity = BallRefinementIdentity {
        source_session_id: batch.source.session_id.clone(),
        frame_id: batch.source.frame_id.clone(),
        configuration_revision: 7,
        calibration_revision: 4,
        motion_generation: 99,
    };
    let refinements = accepted_refinement(&batch);
    let selected = resolve_ball_measurements_for_batch(
        &batch,
        BallMeasurementResolutionContext {
            ball_class_id: 1,
            camera_profile_id: "camera-profile-test",
            configuration_revision: 7,
            rectification_revision: 0,
            profile: Some(&profile),
            refinements: Some(&refinements),
            expected_identity: Some(&expected_identity),
            require_identity: true,
        },
    );
    assert_eq!(selected.measurements[0].center_image_px, [1880.25, 1022.75]);
    assert_eq!(
        selected.measurements[0].refinement_status,
        Some(RefinementStatus::Accepted)
    );
    assert_eq!(selected.measurements[0].table_center_mm, None);
    assert_eq!(selected.calibration_revision, None);

    let mut stale = refinements;
    stale.identity.as_mut().unwrap().frame_id = "another-frame".to_owned();
    let fallback = resolve_ball_measurements_for_batch(
        &batch,
        BallMeasurementResolutionContext {
            ball_class_id: 1,
            camera_profile_id: "camera-profile-test",
            configuration_revision: 7,
            rectification_revision: 0,
            profile: Some(&profile),
            refinements: Some(&stale),
            expected_identity: Some(&expected_identity),
            require_identity: true,
        },
    );
    assert_eq!(
        fallback.measurements[0].center_image_px,
        [
            batch.detections[0].center_image_px.x,
            batch.detections[0].center_image_px.y
        ]
    );
    assert_eq!(fallback.measurements[0].refinement_status, None);
    assert_eq!(fallback.measurements[0].table_center_mm, None);
}

#[test]
fn accepted_refined_center_uses_its_matching_ball_center_calibration_for_xy() {
    let batch = resolver_batch();
    let mut profile = resolver_profile();
    profile.calibration = Some(resolver_calibration());
    let expected_identity = BallRefinementIdentity {
        source_session_id: batch.source.session_id.clone(),
        frame_id: batch.source.frame_id.clone(),
        configuration_revision: 7,
        calibration_revision: 4,
        motion_generation: 99,
    };
    let refinements = accepted_refinement(&batch);
    let selected = resolve_ball_measurements_for_batch(
        &batch,
        BallMeasurementResolutionContext {
            ball_class_id: 1,
            camera_profile_id: "camera-profile-test",
            configuration_revision: 7,
            rectification_revision: 0,
            profile: Some(&profile),
            refinements: Some(&refinements),
            expected_identity: Some(&expected_identity),
            require_identity: true,
        },
    );
    let expected_x = (1880.25 / 3839.0 - 0.5) * 2336.8;
    let expected_y = (1022.75 / 2159.0 - 0.5) * 1155.7;
    let actual = selected.measurements[0].table_center_mm.unwrap();
    assert!((actual[0] - expected_x).abs() < 0.01);
    assert!((actual[1] - expected_y).abs() < 0.01);
    assert_eq!(selected.calibration_revision, Some(4));
}
