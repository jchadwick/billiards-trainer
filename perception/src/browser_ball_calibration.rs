//! Fit browser-collected ball centers against measured physical references.
//!
//! The table calibration identifies the physical playing area and source
//! geometry. Its cloth-plane homography is never used to create a ball-center
//! correspondence. The caller supplies measured positions and associates the
//! selected image points with exact native frames before fitting.

use anyhow::{ensure, Context, Result};
use serde::Deserialize;

use crate::ball_center_calibration::{
    solve_ball_center_calibration, BallCenterCalibrationSolverInput,
    BallCenterCalibrationSolverPolicy, BallCenterCalibrationV1, BallCenterCorrespondence,
};
use crate::ball_localization::REGULATION_BALL_DIAMETER_MM;
use crate::ball_localization_profile::{CameraOpticsProfileV1, BALL_REFINER_REVISION};
use crate::calibration::{ImagePointPx, TablePointMm, ValidatedTableCalibrationV1};

pub(crate) const MINIMUM_BROWSER_SAMPLES: usize = 12;
// Every fit performs leave-one-out validation, making this a work limit as
// well as a request limit. The offline solver supports larger measurement sets.
pub(crate) const MAXIMUM_BROWSER_SAMPLES: usize = 24;
const MAXIMUM_FRAME_ID_BYTES: usize = 128;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct BrowserBallCalibrationSample {
    pub(crate) image_center_px: [f64; 2],
    pub(crate) table_center_mm: [f64; 2],
    pub(crate) frame_id: String,
    pub(crate) frame_sha256: String,
}

pub(crate) fn fit(
    samples: Vec<BrowserBallCalibrationSample>,
    active_table: &ValidatedTableCalibrationV1,
    optics: &CameraOpticsProfileV1,
    table_revision: u64,
) -> Result<BallCenterCalibrationV1> {
    ensure!(
        (MINIMUM_BROWSER_SAMPLES..=MAXIMUM_BROWSER_SAMPLES).contains(&samples.len()),
        "ball calibration requires {MINIMUM_BROWSER_SAMPLES} to {MAXIMUM_BROWSER_SAMPLES} measured placements"
    );

    let camera_profile_id = active_table.camera_profile_id();
    let image_space = active_table.image_space();
    let (source_width_px, source_height_px) = active_table.source_dimensions_px();
    let (table_width_mm, table_height_mm) = active_table.playing_surface_dimensions_mm();
    optics
        .validate_frame(
            camera_profile_id,
            image_space,
            source_width_px,
            source_height_px,
        )
        .context("camera optics do not match the active table calibration")?;
    ensure!(
        optics.refiner_revision == BALL_REFINER_REVISION,
        "camera optics use a different native ball refiner revision"
    );

    let correspondences = samples
        .into_iter()
        .map(|sample| {
            ensure!(
                !sample.frame_id.trim().is_empty()
                    && sample.frame_id.len() <= MAXIMUM_FRAME_ID_BYTES
                    && !sample.frame_id.chars().any(char::is_control),
                "ball calibration sample requires a valid source frame ID"
            );
            ensure!(
                sample.frame_sha256.len() == 64
                    && sample
                        .frame_sha256
                        .bytes()
                        .all(|byte| byte.is_ascii_hexdigit()),
                "ball calibration sample requires a SHA-256 frame digest"
            );
            let [x_px, y_px] = sample.image_center_px;
            let [x_mm, y_mm] = sample.table_center_mm;
            Ok(BallCenterCorrespondence {
                image_point_px: ImagePointPx { x_px, y_px },
                center_table_mm: TablePointMm { x_mm, y_mm },
                provenance: format!(
                    "Browser measurement, frame {:?}, SHA-256 {}",
                    sample.frame_id,
                    sample.frame_sha256.to_ascii_lowercase(),
                ),
            })
        })
        .collect::<Result<Vec<_>>>()?;

    solve_ball_center_calibration(BallCenterCalibrationSolverInput {
        calibration_revision: active_table.calibration_revision(),
        camera_profile_id: camera_profile_id.to_owned(),
        optics_profile_id: optics.optics_profile_id.clone(),
        optics_revision: optics.optics_revision,
        refiner_revision: optics.refiner_revision,
        table_revision,
        rectification_revision: active_table.rectification_revision(),
        image_space,
        source_width_px,
        source_height_px,
        table_width_mm,
        table_height_mm,
        ball_diameter_mm: REGULATION_BALL_DIAMETER_MM,
        provenance: "Browser calibration using operator-measured ball placements".to_owned(),
        solver_policy: BallCenterCalibrationSolverPolicy {
            minimum_correspondences: MINIMUM_BROWSER_SAMPLES,
            maximum_correspondences: MAXIMUM_BROWSER_SAMPLES,
            ..BallCenterCalibrationSolverPolicy::default()
        },
        correspondences,
    })
    .context("could not fit ball calibration to the measured placements")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ball_localization::CameraIntrinsics;
    use crate::ball_localization_profile::CAMERA_OPTICS_PROFILE_VERSION;
    use crate::calibration::TableCalibrationV1;
    use crate::ImageSpace;

    #[test]
    fn fits_independent_centers_and_binds_asset_to_active_table_and_optics() {
        let mut table: TableCalibrationV1 = serde_json::from_str(include_str!(
            "../contracts/calibration/v1/table-calibration-perspective.json"
        ))
        .unwrap();
        table.image_space = ImageSpace::Raw;
        let active_table = table.into_validated().unwrap();
        let (source_width_px, source_height_px) = active_table.source_dimensions_px();
        let mut optics = CameraOpticsProfileV1 {
            schema_version: CAMERA_OPTICS_PROFILE_VERSION,
            optics_profile_id: "browser-optics".to_owned(),
            camera_profile_id: active_table.camera_profile_id().to_owned(),
            optics_revision: 4,
            refiner_revision: BALL_REFINER_REVISION,
            image_space: ImageSpace::Raw,
            intrinsics: CameraIntrinsics {
                source_width_px,
                source_height_px,
                fx: 700.0,
                fy: 700.0,
                cx: 500.0,
                cy: 250.0,
            },
            provenance: "Synthetic optics for calibration test".to_owned(),
        };
        // An affine ball mapping intentionally distinct from the fixture's
        // cloth-plane perspective mapping, with coverage in both axes.
        let samples = || {
            (0..3)
                .flat_map(|row| (0..4).map(move |column| (row, column)))
                .map(|(row, column)| {
                    let x = 140.0 + f64::from(column) * 240.0;
                    let y = 90.0 + f64::from(row) * 150.0;
                    BrowserBallCalibrationSample {
                        image_center_px: [x, y],
                        table_center_mm: [(x - 500.0) * 2.1, (y - 250.0) * 2.0],
                        frame_id: format!("frame-{row}-{column}"),
                        frame_sha256: format!("{:064x}", row * 4 + column),
                    }
                })
                .collect::<Vec<_>>()
        };

        let asset = fit(samples(), &active_table, &optics, 5).unwrap();
        assert_eq!(
            asset.calibration_revision,
            active_table.calibration_revision()
        );
        assert_eq!(asset.optics_revision, optics.optics_revision);
        assert_eq!(asset.table_revision, 5);
        assert_eq!(
            (asset.table_width_mm, asset.table_height_mm),
            (2000.0, 1000.0)
        );
        assert_eq!(
            asset.fit_report.leave_one_out.len(),
            MINIMUM_BROWSER_SAMPLES
        );
        assert!(asset.fit_report.leave_one_out_maximum_residual_mm < 1e-6);

        let mut bad_samples = samples();
        bad_samples[0].frame_sha256 = "not a frame digest".to_owned();
        assert!(fit(bad_samples, &active_table, &optics, 5).is_err());
        optics.intrinsics.source_width_px -= 1;
        assert!(fit(samples(), &active_table, &optics, 5).is_err());
    }
}
