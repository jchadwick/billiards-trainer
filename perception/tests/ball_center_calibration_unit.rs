use billiards_perception::ball_center_calibration::*;
use billiards_perception::{calibration, ImageSpace};
use calibration::{CalibrationContext, ImagePointPx, TablePointMm};

fn true_projection(p: ImagePointPx) -> TablePointMm {
    let d = 0.00001 * p.x_px - 0.00002 * p.y_px + 1.0;
    TablePointMm {
        x_mm: (0.55 * p.x_px + 0.02 * p.y_px - 1050.0) / d,
        y_mm: (-0.01 * p.x_px + 0.48 * p.y_px - 480.0) / d,
    }
}

fn input() -> BallCenterCalibrationSolverInput {
    let mut correspondences = Vec::new();
    for y in [120.0, 1000.0, 1880.0] {
        for x in [150.0, 1300.0, 2450.0, 3600.0] {
            let image_point_px = ImagePointPx { x_px: x, y_px: y };
            correspondences.push(BallCenterCorrespondence {
                image_point_px,
                center_table_mm: true_projection(image_point_px),
                provenance: format!("independent synthetic locator source-frame-{x}-{y}"),
            });
        }
    }
    BallCenterCalibrationSolverInput {
        calibration_revision: 8,
        camera_profile_id: "native-4k-camera".into(),
        optics_profile_id: "verified-fixed-focus".into(),
        optics_revision: 3,
        refiner_revision: 4,
        table_revision: 2,
        rectification_revision: 0,
        image_space: ImageSpace::Raw,
        source_width_px: 3840,
        source_height_px: 2160,
        table_width_mm: 2500.0,
        table_height_mm: 1300.0,
        ball_diameter_mm: 57.15,
        provenance: "Known synthetic 2D projective correspondences; not physical accuracy evidence"
            .into(),
        solver_policy: BallCenterCalibrationSolverPolicy::default(),
        correspondences,
    }
}

fn context() -> BallCenterCalibrationContext<'static> {
    BallCenterCalibrationContext {
        frame: CalibrationContext {
            calibration_revision: 8,
            camera_profile_id: "native-4k-camera",
            rectification_revision: 0,
            image_space: ImageSpace::Raw,
            source_width_px: 3840,
            source_height_px: 2160,
        },
        optics_profile_id: "verified-fixed-focus",
        optics_revision: 3,
        refiner_revision: 4,
        table_revision: 2,
    }
}

#[test]
fn exact_overdetermined_fit_projects_unseen_corners_in_center_origin_millimeters() {
    let asset = solve_ball_center_calibration(input()).unwrap();
    asset.validate().unwrap();
    assert_eq!(asset.fit_report.residuals.len(), 12);
    assert!(asset.fit_report.rms_mm < 1e-8);
    for p in [
        ImagePointPx {
            x_px: 0.0,
            y_px: 0.0,
        },
        ImagePointPx {
            x_px: 3839.0,
            y_px: 2159.0,
        },
        ImagePointPx {
            x_px: 1920.0,
            y_px: 1080.0,
        },
    ] {
        let expected = true_projection(p);
        let actual = asset.project(p, context()).unwrap();
        assert!((expected.x_mm - actual.x_mm).abs() < 1e-8);
        assert!((expected.y_mm - actual.y_mm).abs() < 1e-8);
    }
    assert_eq!(asset.ball_diameter_mm, 57.15);
    assert_eq!(asset.table_width_mm, 2500.0);
    let json = serde_json::to_string(&asset).unwrap();
    let decoded: BallCenterCalibrationV1 = serde_json::from_str(&json).unwrap();
    decoded.validate().unwrap();
    let validated = decoded.into_validated().unwrap();
    let p = ImagePointPx {
        x_px: 1750.0,
        y_px: 970.0,
    };
    assert_eq!(
        validated.project(p, context()).unwrap(),
        asset.project(p, context()).unwrap()
    );
    assert_eq!(validated.as_asset(), &asset);
    let mut mismatch = context();
    mismatch.refiner_revision += 1;
    assert!(validated.validate_context(mismatch).is_err());
    assert!(validated.project(p, mismatch).is_err());
}

#[test]
fn perturbed_fit_reports_every_observation_and_configurable_acceptance_gate() {
    let mut data = input();
    for (index, p) in data.correspondences.iter_mut().enumerate() {
        p.center_table_mm.x_mm += if index % 2 == 0 { 0.15 } else { -0.15 };
        p.center_table_mm.y_mm += if index % 3 == 0 { 0.1 } else { -0.1 };
    }
    data.correspondences[5].center_table_mm.x_mm += 5.0;
    let asset = solve_ball_center_calibration(data.clone()).unwrap();
    assert_eq!(asset.correspondences, data.correspondences);
    assert_eq!(asset.fit_report.residuals.len(), 12);
    assert!(asset.fit_report.maximum_residual_mm > 2.0);
    for (index, r) in asset.fit_report.residuals.iter().enumerate() {
        assert_eq!(index, r.correspondence_index);
        assert!((r.error_x_mm.hypot(r.error_y_mm) - r.distance_mm).abs() < 1e-10);
    }
    data.solver_policy.maximum_fit_rms_mm = Some(0.05);
    assert_eq!(
        solve_ball_center_calibration(data).unwrap_err(),
        BallCenterCalibrationError::FitErrorExceedsLimit
    );
}

#[test]
fn insufficient_unbounded_nonfinite_and_collinear_samples_are_rejected() {
    let mut data = input();
    data.correspondences.truncate(5);
    assert_eq!(
        solve_ball_center_calibration(data).unwrap_err(),
        BallCenterCalibrationError::TooFewCorrespondences
    );
    let mut data = input();
    data.solver_policy.minimum_correspondences = 4;
    assert_eq!(
        solve_ball_center_calibration(data).unwrap_err(),
        BallCenterCalibrationError::InvalidPolicy
    );
    let mut data = input();
    data.solver_policy.maximum_correspondences = usize::MAX;
    assert_eq!(
        solve_ball_center_calibration(data).unwrap_err(),
        BallCenterCalibrationError::InvalidPolicy
    );
    let mut data = input();
    data.solver_policy.numerical_tolerance = f64::NAN;
    assert_eq!(
        solve_ball_center_calibration(data).unwrap_err(),
        BallCenterCalibrationError::InvalidPolicy
    );
    let mut data = input();
    data.correspondences[0].image_point_px.x_px = f64::INFINITY;
    assert_eq!(
        solve_ball_center_calibration(data).unwrap_err(),
        BallCenterCalibrationError::PointOutsideImage
    );
    let mut data = input();
    for (index, p) in data.correspondences.iter_mut().enumerate() {
        p.image_point_px = ImagePointPx {
            x_px: 100.0 + index as f64 * 50.0,
            y_px: 100.0,
        };
    }
    assert_eq!(
        solve_ball_center_calibration(data).unwrap_err(),
        BallCenterCalibrationError::DegenerateGeometry
    );
    let mut data = input();
    data.solver_policy.maximum_condition_number = 1.0;
    assert_eq!(
        solve_ball_center_calibration(data).unwrap_err(),
        BallCenterCalibrationError::IllConditioned
    );
}

#[test]
fn every_revision_identity_and_geometry_binding_is_enforced() {
    let asset = solve_ball_center_calibration(input()).unwrap();
    let point = ImagePointPx {
        x_px: 100.0,
        y_px: 100.0,
    };
    let mut c = context();
    c.frame.calibration_revision += 1;
    assert!(matches!(
        asset.project(point, c),
        Err(BallCenterCalibrationError::ContextMismatch(
            "calibration_revision"
        ))
    ));
    let mut c = context();
    c.frame.camera_profile_id = "changed";
    assert!(asset.project(point, c).is_err());
    let mut c = context();
    c.frame.rectification_revision += 1;
    assert!(asset.project(point, c).is_err());
    let mut c = context();
    c.frame.image_space = ImageSpace::Rectified;
    assert!(asset.project(point, c).is_err());
    let mut c = context();
    c.frame.source_width_px = 1280;
    c.frame.source_height_px = 720;
    assert!(asset.project(point, c).is_err());
    let mut c = context();
    c.optics_profile_id = "other-lens";
    assert!(asset.project(point, c).is_err());
    let mut c = context();
    c.optics_revision += 1;
    assert!(asset.project(point, c).is_err());
    let mut c = context();
    c.refiner_revision += 1;
    assert!(asset.project(point, c).is_err());
    let mut c = context();
    c.table_revision += 1;
    assert!(asset.project(point, c).is_err());
    assert_eq!(
        asset
            .project(
                ImagePointPx {
                    x_px: 3840.0,
                    y_px: 0.0
                },
                context()
            )
            .unwrap_err(),
        BallCenterCalibrationError::PointOutsideImage
    );
}

#[test]
fn malformed_provenance_report_schema_and_projective_poles_fail_closed() {
    let asset = solve_ball_center_calibration(input()).unwrap();
    let mut bad = asset.clone();
    bad.schema_version = 2;
    assert!(bad.validate().is_err());
    let mut bad = asset.clone();
    bad.correspondences[0].provenance = " ".into();
    assert_eq!(
        bad.validate().unwrap_err(),
        BallCenterCalibrationError::InvalidProvenance
    );
    let mut bad = asset.clone();
    bad.fit_report.residuals[0].distance_mm += 10.0;
    assert_eq!(
        bad.validate().unwrap_err(),
        BallCenterCalibrationError::InvalidFitReport
    );
    let mut bad = asset.clone();
    bad.fit_report.residuals.pop();
    assert_eq!(
        bad.validate().unwrap_err(),
        BallCenterCalibrationError::InvalidFitReport
    );
    let mut bad = asset.clone();
    bad.image_to_table_mm[2] = [0.001, 0.0, -1.0];
    assert!(bad.validate().is_err());
    let mut bad = asset.clone();
    bad.image_to_table_mm[1] = bad.image_to_table_mm[0];
    assert_eq!(
        bad.validate().unwrap_err(),
        BallCenterCalibrationError::InvalidHomography
    );
    let mut value = serde_json::to_value(&asset).unwrap();
    value["unrecognized_parameter"] = true.into();
    assert!(serde_json::from_value::<BallCenterCalibrationV1>(value).is_err());
    let mut value = serde_json::to_value(&asset).unwrap();
    value["correspondences"][0]["image_point_px"]["z_px"] = 1.into();
    assert!(serde_json::from_value::<BallCenterCalibrationV1>(value).is_err());
}

#[test]
fn six_and_nine_points_require_overdetermined_well_conditioned_holdouts() {
    for indices in [vec![0, 3, 4, 7, 8, 11], vec![0, 1, 3, 4, 5, 7, 8, 9, 11]] {
        let mut data = input();
        data.correspondences = indices
            .iter()
            .map(|&i| data.correspondences[i].clone())
            .collect();
        let asset = solve_ball_center_calibration(data).unwrap();
        assert_eq!(asset.fit_report.leave_one_out.len(), indices.len());
        assert!(asset.fit_report.leave_one_out_rms_mm < 1e-7);
        assert!(
            asset.fit_report.source_hull_area_fraction > 0.
                && asset.fit_report.source_hull_area_fraction < 1.
        );
        assert!(
            asset.fit_report.table_hull_area_fraction > 0.
                && asset.fit_report.table_hull_area_fraction < 1.
        );
        assert_eq!(
            asset.fit_report.source_image_corners_extrapolated,
            [true; 4]
        );
        for (i, fold) in asset.fit_report.leave_one_out.iter().enumerate() {
            assert_eq!(fold.held_out_index, i);
            assert_eq!(fold.training_correspondences, indices.len() - 1);
            assert!(fold.normalized_design_condition_number >= 1.);
        }
        let restored: BallCenterCalibrationV1 =
            serde_json::from_str(&serde_json::to_string(&asset).unwrap()).unwrap();
        restored.validate().unwrap();
    }
}

#[test]
fn duplicate_image_or_physical_reference_cannot_increase_independent_count() {
    for physical in [false, true] {
        let mut data = input();
        if physical {
            data.correspondences[1].center_table_mm = data.correspondences[0].center_table_mm;
        } else {
            data.correspondences[1].image_point_px = data.correspondences[0].image_point_px;
        }
        assert_eq!(
            solve_ball_center_calibration(data).unwrap_err(),
            BallCenterCalibrationError::DuplicateCorrespondence
        );
    }
}

#[test]
fn loader_recomputes_condition_holdouts_coverage_and_the_fitted_matrix() {
    let asset = solve_ball_center_calibration(input()).unwrap();
    let mut forged = asset.clone();
    forged.fit_report.normalized_design_condition_number = 1.;
    assert_eq!(
        forged.validate().unwrap_err(),
        BallCenterCalibrationError::InvalidFitReport
    );
    let mut forged = asset.clone();
    forged.fit_report.leave_one_out[0].normalized_design_condition_number = 1.;
    assert_eq!(
        forged.validate().unwrap_err(),
        BallCenterCalibrationError::InvalidFitReport
    );
    let mut forged = asset.clone();
    forged.fit_report.source_hull_area_fraction = 1.;
    assert_eq!(
        forged.validate().unwrap_err(),
        BallCenterCalibrationError::InvalidFitReport
    );
    let mut forged = asset.clone();
    forged.fit_report.leave_one_out[0].distance_mm += 1.;
    assert_eq!(
        forged.validate().unwrap_err(),
        BallCenterCalibrationError::InvalidFitReport
    );
    let mut forged = asset;
    forged.image_to_table_mm[0][2] += 1.;
    assert_eq!(
        forged.validate().unwrap_err(),
        BallCenterCalibrationError::InvalidFitReport
    );
}

#[test]
fn a_single_critical_reference_cannot_make_a_degenerate_layout_eligible() {
    let mut data = input();
    data.correspondences = [
        [100., 100.],
        [600., 100.],
        [1100., 100.],
        [1600., 100.],
        [100., 600.],
        [1600., 600.],
    ]
    .map(|[x_px, y_px]| {
        let image_point_px = ImagePointPx { x_px, y_px };
        BallCenterCorrespondence {
            image_point_px,
            center_table_mm: true_projection(image_point_px),
            provenance: format!("known critical-layout point {x_px}/{y_px}"),
        }
    })
    .to_vec();
    assert_eq!(
        solve_ball_center_calibration(data).unwrap_err(),
        BallCenterCalibrationError::DegenerateGeometry
    );
}
