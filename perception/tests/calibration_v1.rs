use billiards_perception::calibration::{
    CalibrationContext, CalibrationError, ImagePointPx, TableCalibrationV1, TablePointMm,
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
