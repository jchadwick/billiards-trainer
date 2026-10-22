use billiards_perception::calibration::{
    solve_table_calibration, CalibrationContext, CalibrationError, ImagePointPx,
    TableCalibrationSolverInput, TableCalibrationV1, TablePointMm,
};
use billiards_perception::ImageSpace;

const CENTER_ORIGIN_FIXTURE: &str =
    include_str!("../contracts/calibration/v1/table-calibration-center-origin.json");
const PYTHON_EXPORT_FIXTURE: &str =
    include_str!("../contracts/calibration/v1/table-calibration-python-export.json");

fn calibration() -> TableCalibrationV1 {
    serde_json::from_str(CENTER_ORIGIN_FIXTURE).expect("calibration fixture must decode")
}

fn context() -> CalibrationContext<'static> {
    CalibrationContext {
        calibration_revision: 3,
        camera_profile_id: "test-camera-profile",
        rectification_revision: 2,
        image_space: ImageSpace::Rectified,
        source_width_px: 1000,
        source_height_px: 500,
    }
}

fn solver_input(corners_px: [ImagePointPx; 4]) -> TableCalibrationSolverInput {
    TableCalibrationSolverInput {
        calibration_revision: 3,
        camera_profile_id: "test-camera-profile".to_owned(),
        rectification_revision: 2,
        image_space: ImageSpace::Rectified,
        source_width_px: 1000,
        source_height_px: 500,
        table_width_mm: 2000.0,
        table_height_mm: 1000.0,
        corners_px,
    }
}

fn assert_point(actual: TablePointMm, expected_x_mm: f64, expected_y_mm: f64) {
    assert!((actual.x_mm - expected_x_mm).abs() < 1e-9);
    assert!((actual.y_mm - expected_y_mm).abs() < 1e-9);
}

#[test]
fn center_origin_fixture_maps_image_corners_and_center_to_millimeters() {
    let validated = calibration().into_validated().unwrap();

    assert_point(
        validated
            .project(
                ImagePointPx {
                    x_px: 0.0,
                    y_px: 0.0,
                },
                context(),
            )
            .unwrap(),
        -1000.0,
        -500.0,
    );
    assert_point(
        validated
            .project(
                ImagePointPx {
                    x_px: 500.0,
                    y_px: 250.0,
                },
                context(),
            )
            .unwrap(),
        0.0,
        0.0,
    );
    assert_point(
        validated
            .project(
                ImagePointPx {
                    x_px: 0.0,
                    y_px: 499.0,
                },
                context(),
            )
            .unwrap(),
        -1000.0,
        498.0,
    );
}

#[test]
fn python_exported_perspective_fixture_maps_the_declared_table_corners() {
    let validated: billiards_perception::calibration::ValidatedTableCalibrationV1 =
        serde_json::from_str::<TableCalibrationV1>(PYTHON_EXPORT_FIXTURE)
            .expect("Python exporter fixture must decode")
            .into_validated()
            .expect("Python exporter fixture must validate");
    let expected_corners = [
        (30.0, 20.0, -1000.0, -500.0),
        (970.0, 45.0, 1000.0, -500.0),
        (930.0, 480.0, 1000.0, 500.0),
        (55.0, 455.0, -1000.0, 500.0),
    ];

    for (x_px, y_px, expected_x_mm, expected_y_mm) in expected_corners {
        let point = validated
            .project(ImagePointPx { x_px, y_px }, context())
            .expect("the exact source context must permit projection");
        assert_point(point, expected_x_mm, expected_y_mm);
    }
}

#[test]
fn solver_creates_revision_bound_calibration_with_exact_center_origin_mapping() {
    let corners = [
        ImagePointPx {
            x_px: 100.0,
            y_px: 50.0,
        },
        ImagePointPx {
            x_px: 900.0,
            y_px: 50.0,
        },
        ImagePointPx {
            x_px: 900.0,
            y_px: 450.0,
        },
        ImagePointPx {
            x_px: 100.0,
            y_px: 450.0,
        },
    ];
    let solved = solve_table_calibration(solver_input(corners)).unwrap();

    assert_eq!(solved.calibration_revision, 3);
    assert_eq!(solved.camera_profile_id, "test-camera-profile");
    assert_eq!(solved.rectification_revision, 2);
    assert_eq!(solved.image_space, ImageSpace::Rectified);
    assert_eq!(
        (solved.source_width_px, solved.source_height_px),
        (1000, 500)
    );
    assert_eq!(
        (solved.table_width_mm, solved.table_height_mm),
        (2000.0, 1000.0)
    );
    solved.validate().unwrap();

    let validated = solved.into_validated().unwrap();
    for (corner, expected_x, expected_y) in [
        (corners[0], -1000.0, -500.0),
        (corners[1], 1000.0, -500.0),
        (corners[2], 1000.0, 500.0),
        (corners[3], -1000.0, 500.0),
    ] {
        assert_point(
            validated.project(corner, context()).unwrap(),
            expected_x,
            expected_y,
        );
    }
    assert_point(
        validated
            .project(
                ImagePointPx {
                    x_px: 500.0,
                    y_px: 250.0,
                },
                context(),
            )
            .unwrap(),
        0.0,
        0.0,
    );
}

#[test]
fn solver_matches_python_perspective_export_for_skewed_table_corners() {
    let corners = [
        ImagePointPx {
            x_px: 30.0,
            y_px: 20.0,
        },
        ImagePointPx {
            x_px: 970.0,
            y_px: 45.0,
        },
        ImagePointPx {
            x_px: 930.0,
            y_px: 480.0,
        },
        ImagePointPx {
            x_px: 55.0,
            y_px: 455.0,
        },
    ];
    let solved = solve_table_calibration(solver_input(corners)).unwrap();
    let rust_projection = solved.clone().into_validated().unwrap();
    let python_projection = serde_json::from_str::<TableCalibrationV1>(PYTHON_EXPORT_FIXTURE)
        .unwrap()
        .into_validated()
        .unwrap();

    for point in [
        ImagePointPx {
            x_px: 30.0,
            y_px: 20.0,
        },
        ImagePointPx {
            x_px: 970.0,
            y_px: 45.0,
        },
        ImagePointPx {
            x_px: 930.0,
            y_px: 480.0,
        },
        ImagePointPx {
            x_px: 55.0,
            y_px: 455.0,
        },
        ImagePointPx {
            x_px: 500.0,
            y_px: 250.0,
        },
        ImagePointPx {
            x_px: 400.0,
            y_px: 300.0,
        },
    ] {
        let rust_point = rust_projection.project(point, context()).unwrap();
        let python_point = python_projection.project(point, context()).unwrap();
        assert!((rust_point.x_mm - python_point.x_mm).abs() < 1.0e-7);
        assert!((rust_point.y_mm - python_point.y_mm).abs() < 1.0e-7);
    }
}

#[test]
fn solver_rejects_reversed_or_misordered_corner_sequence() {
    let corners = [
        ImagePointPx {
            x_px: 100.0,
            y_px: 50.0,
        },
        ImagePointPx {
            x_px: 100.0,
            y_px: 450.0,
        },
        ImagePointPx {
            x_px: 900.0,
            y_px: 450.0,
        },
        ImagePointPx {
            x_px: 900.0,
            y_px: 50.0,
        },
    ];
    assert_eq!(
        solve_table_calibration(solver_input(corners)),
        Err(CalibrationError::InvalidCalibrationCorners)
    );
}

#[test]
fn solver_rejects_degenerate_and_nearly_degenerate_quadrilaterals() {
    let degenerate = [
        ImagePointPx {
            x_px: 100.0,
            y_px: 100.0,
        },
        ImagePointPx {
            x_px: 300.0,
            y_px: 100.0,
        },
        ImagePointPx {
            x_px: 700.0,
            y_px: 100.0,
        },
        ImagePointPx {
            x_px: 900.0,
            y_px: 100.0,
        },
    ];
    assert!(matches!(
        solve_table_calibration(solver_input(degenerate)),
        Err(CalibrationError::InvalidCalibrationCorners)
            | Err(CalibrationError::DegenerateCalibrationGeometry)
    ));

    let nearly_degenerate = [
        ImagePointPx {
            x_px: 100.0,
            y_px: 100.0,
        },
        ImagePointPx {
            x_px: 900.0,
            y_px: 100.0,
        },
        ImagePointPx {
            x_px: 900.0,
            y_px: 100.00000001,
        },
        ImagePointPx {
            x_px: 100.0,
            y_px: 100.00000001,
        },
    ];
    assert_eq!(
        solve_table_calibration(solver_input(nearly_degenerate)),
        Err(CalibrationError::DegenerateCalibrationGeometry)
    );
}

#[test]
fn solver_rejects_non_finite_and_out_of_bounds_corner_observations() {
    let valid = [
        ImagePointPx {
            x_px: 100.0,
            y_px: 50.0,
        },
        ImagePointPx {
            x_px: 900.0,
            y_px: 50.0,
        },
        ImagePointPx {
            x_px: 900.0,
            y_px: 450.0,
        },
        ImagePointPx {
            x_px: 100.0,
            y_px: 450.0,
        },
    ];
    let mut non_finite = valid;
    non_finite[2].x_px = f64::NAN;
    assert_eq!(
        solve_table_calibration(solver_input(non_finite)),
        Err(CalibrationError::NonFiniteImagePoint)
    );

    let mut outside = valid;
    outside[3].y_px = 500.0;
    assert_eq!(
        solve_table_calibration(solver_input(outside)),
        Err(CalibrationError::ImagePointOutsideSource)
    );
}

#[test]
fn projective_mapping_is_homogeneous_scale_invariant() {
    let mut first = calibration();
    first.pixel_to_table_mm = [[2.0, 0.0, -1000.0], [0.0, 2.0, -500.0], [0.001, 0.002, 1.0]];
    let mut second = first.clone();
    for row in &mut second.pixel_to_table_mm {
        for value in row {
            *value *= 17.0;
        }
    }
    let point = ImagePointPx {
        x_px: 100.0,
        y_px: 50.0,
    };
    let first_result = first
        .into_validated()
        .unwrap()
        .project(point, context())
        .unwrap();
    let second_result = second
        .into_validated()
        .unwrap()
        .project(point, context())
        .unwrap();

    assert_point(first_result, -800.0 / 1.2, -400.0 / 1.2);
    assert_point(second_result, first_result.x_mm, first_result.y_mm);
}

#[test]
fn projection_requires_exact_source_profile_and_applied_revisions() {
    let validated = calibration().into_validated().unwrap();

    let mut mismatched = context();
    mismatched.calibration_revision = 4;
    assert_eq!(
        validated.project(
            ImagePointPx {
                x_px: 5.0,
                y_px: 5.0
            },
            mismatched
        ),
        Err(CalibrationError::CalibrationRevisionMismatch)
    );

    let mut mismatched = context();
    mismatched.camera_profile_id = "another-camera";
    assert_eq!(
        validated.project(
            ImagePointPx {
                x_px: 5.0,
                y_px: 5.0
            },
            mismatched
        ),
        Err(CalibrationError::CameraProfileMismatch)
    );

    let mut mismatched = context();
    mismatched.rectification_revision = 3;
    assert_eq!(
        validated.project(
            ImagePointPx {
                x_px: 5.0,
                y_px: 5.0
            },
            mismatched
        ),
        Err(CalibrationError::RectificationRevisionMismatch)
    );

    let mut mismatched = context();
    mismatched.image_space = ImageSpace::Raw;
    assert_eq!(
        validated.project(
            ImagePointPx {
                x_px: 5.0,
                y_px: 5.0
            },
            mismatched
        ),
        Err(CalibrationError::ImageSpaceMismatch)
    );

    let mut mismatched = context();
    mismatched.source_width_px = 1280;
    assert_eq!(
        validated.project(
            ImagePointPx {
                x_px: 5.0,
                y_px: 5.0
            },
            mismatched
        ),
        Err(CalibrationError::SourceResolutionMismatch)
    );
}

#[test]
fn invalid_or_numerically_undefined_calibration_never_produces_positions() {
    let mut invalid = calibration();
    invalid.schema_version = 2;
    assert_eq!(
        invalid.into_validated().unwrap_err(),
        CalibrationError::UnsupportedSchemaVersion
    );

    let mut invalid = calibration();
    invalid.pixel_to_table_mm = [[1.0, 0.0, 0.0]; 3];
    assert_eq!(
        invalid.into_validated().unwrap_err(),
        CalibrationError::InvalidHomography
    );

    let mut invalid = calibration();
    invalid.pixel_to_table_mm[0][0] = f64::NAN;
    assert_eq!(
        invalid.into_validated().unwrap_err(),
        CalibrationError::InvalidHomography
    );

    let validated = calibration().into_validated().unwrap();
    assert_eq!(
        validated.project(
            ImagePointPx {
                x_px: f64::INFINITY,
                y_px: 10.0,
            },
            context(),
        ),
        Err(CalibrationError::NonFiniteImagePoint)
    );
    assert_eq!(
        validated.project(
            ImagePointPx {
                x_px: 1000.0,
                y_px: 10.0
            },
            context()
        ),
        Err(CalibrationError::ImagePointOutsideSource)
    );

    let mut horizon = calibration();
    horizon.pixel_to_table_mm = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, -100.0]];
    let horizon = horizon.into_validated().unwrap();
    assert_eq!(
        horizon.project(
            ImagePointPx {
                x_px: 100.0,
                y_px: 10.0
            },
            context()
        ),
        Err(CalibrationError::ProjectiveSingularity)
    );
}
