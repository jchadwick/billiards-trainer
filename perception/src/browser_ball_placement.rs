//! Live calibration guidance uses a recent producer RGB frame and returns
//! coordinates only. The projected target is approximate placement guidance.

use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};

use crate::ball_localization::{
    refine_ball_batch, BallLocalizationConfig, BallProposal, CameraIntrinsics, RefinementStatus,
    REGULATION_BALL_DIAMETER_MM,
};
use crate::calibration::{TablePointMm, ValidatedTableCalibrationV1};
use crate::frame_preview::FramePreviewDescriptorV1;

pub(crate) const BALL_PLACEMENT_REQUEST_V1: &[u8; 8] = b"BTBP0001";
pub(crate) const MAXIMUM_PLACEMENT_MESSAGE_BYTES: usize = 8 * 1024;

#[derive(Clone, Copy, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct BallSearchTarget {
    seed: [f64; 2],
    pub(crate) center: [f64; 2],
    pub(crate) tolerance: [f64; 2],
    ball_radius: f64,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct BallMark {
    pub(crate) image_center_px: [f64; 2],
    radius_px: f64,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct PlacementRequest {
    pub(crate) target: BallSearchTarget,
    pub(crate) intrinsics: CameraIntrinsics,
    pub(crate) parameters: BallLocalizationConfig,
    pub(crate) camera_profile_id: String,
    pub(crate) configuration_revision: u64,
    pub(crate) calibration_revision: u64,
    pub(crate) rectification_revision: u64,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct PlacementObservation {
    pub(crate) frame: FramePreviewDescriptorV1,
    pub(crate) candidate: Option<BallMark>,
}

impl PlacementRequest {
    pub(crate) fn matches(&self, frame: &FramePreviewDescriptorV1) -> bool {
        frame.image_space == crate::detection_batch::ImageSpace::Raw
            && frame.camera_profile_id == self.camera_profile_id
            && frame.configuration_revision == self.configuration_revision
            && frame.calibration_revision == self.calibration_revision
            && frame.rectification_revision == self.rectification_revision
            && (frame.width_px, frame.height_px)
                == (
                    self.intrinsics.source_width_px,
                    self.intrinsics.source_height_px,
                )
    }
}

impl BallSearchTarget {
    pub(crate) fn project(
        table: &ValidatedTableCalibrationV1,
        intrinsics: &CameraIntrinsics,
        parameters: &BallLocalizationConfig,
        [x_mm, y_mm]: [f64; 2],
    ) -> Result<Self> {
        intrinsics.validate()?;
        parameters.validate()?;
        ensure!(
            table.contains_playing_surface_point(TablePointMm { x_mm, y_mm }),
            "Choose a target inside the playing surface."
        );
        let seed = table
            .image_search_seed([x_mm, y_mm])
            .context("The position cannot be located in the camera image.")?;
        let physical_radius = REGULATION_BALL_DIAMETER_MM / 2.0;
        // One physical ball radius is the placement tolerance. Project each
        // axis independently for an oblique view.
        let projected_radius = |axis: usize| -> Result<f64> {
            let mut low = [x_mm, y_mm];
            let mut high = low;
            low[axis] -= physical_radius;
            high[axis] += physical_radius;
            let low = table
                .image_search_seed(low)
                .context("Invalid target boundary.")?;
            let high = table
                .image_search_seed(high)
                .context("Invalid target boundary.")?;
            Ok((high[0] - low[0]).hypot(high[1] - low[1]) / 2.0)
        };
        let tolerance = [projected_radius(0)?, projected_radius(1)?];
        let scale =
            f64::from(intrinsics.source_width_px) / f64::from(parameters.reference_width_px);
        let ball_radius = (tolerance.into_iter().sum::<f64>() / 2.0).clamp(
            parameters.minimum_radius_px * scale,
            parameters.maximum_radius_px * scale,
        );
        // Approximate the center's elevation above the cloth. This affects
        // target guidance only, never the physical fit correspondence.
        let center = [
            seed[0] + (seed[0] - intrinsics.cx) * ball_radius / intrinsics.fx,
            seed[1] + (seed[1] - intrinsics.cy) * ball_radius / intrinsics.fy,
        ];
        ensure!(
            tolerance.into_iter().all(|v| v.is_finite() && v > 0.0)
                && (0.0..f64::from(intrinsics.source_width_px)).contains(&center[0])
                && (0.0..f64::from(intrinsics.source_height_px)).contains(&center[1]),
            "The placement target is outside the camera view."
        );
        Ok(Self {
            seed,
            center,
            tolerance,
            ball_radius,
        })
    }

    pub(crate) fn contains(&self, point: [f64; 2]) -> bool {
        let [dx, dy] =
            std::array::from_fn(|axis| (point[axis] - self.center[axis]) / self.tolerance[axis]);
        dx.hypot(dy) <= 1.0
    }

    pub(crate) fn refine(
        &self,
        rgb: &[u8],
        intrinsics: &CameraIntrinsics,
        parameters: &BallLocalizationConfig,
    ) -> Result<Option<BallMark>> {
        let radius = self.ball_radius;
        ensure!(radius.is_finite() && radius > 0.0, "Invalid search radius.");
        let [cx, cy] = self.center;
        let proposals: Vec<_> = [self.center, self.seed]
            .into_iter()
            .chain([-0.5_f64, 0.0, 0.5].into_iter().flat_map(|dx| {
                [-0.5_f64, 0.0, 0.5].into_iter().filter_map(move |dy| {
                    (dx != 0.0 || dy != 0.0).then_some([cx + dx * radius, cy + dy * radius])
                })
            }))
            .filter(|[x, y]| {
                *x >= radius
                    && *y >= radius
                    && *x + radius <= f64::from(intrinsics.source_width_px)
                    && *y + radius <= f64::from(intrinsics.source_height_px)
            })
            .enumerate()
            .map(|(index, [x, y])| BallProposal {
                detection_index: index as u64,
                bbox: [x - radius, y - radius, x + radius, y + radius],
            })
            .collect();
        let distance = |[x, y]: [f64; 2]| (x - cx).hypot(y - cy);
        Ok(refine_ball_batch(rgb, intrinsics, &proposals, parameters)?
            .into_iter()
            .filter(|fit| fit.status == RefinementStatus::Accepted)
            .filter_map(|fit| {
                Some(BallMark {
                    image_center_px: fit.center_image_px?,
                    radius_px: fit.outline?.half_extent_image_px.into_iter().sum::<f64>() / 2.0,
                })
            })
            .min_by(|left, right| {
                distance(left.image_center_px).total_cmp(&distance(right.image_center_px))
            }))
    }
}
