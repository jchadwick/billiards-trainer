//! Revision-bound image-to-table geometry.
//!
//! The calibration is immutable once validated. A frame must match its camera
//! profile, source image space and dimensions, calibration revision, and
//! rectification revision exactly; callers must first map detector coordinates
//! back into the declared source-image coordinate space.

use std::error::Error;
use std::fmt::{Display, Formatter};

use serde::{Deserialize, Serialize};

use crate::ImageSpace;

pub const TABLE_CALIBRATION_SCHEMA_VERSION: u8 = 1;
const MAX_SAFE_JSON_INTEGER: u64 = 9_007_199_254_740_991;
const MAX_IDENTIFIER_CHARACTERS: usize = 128;
const SINGULARITY_GUARD_MULTIPLIER: f64 = 64.0;

/// A complete immutable homography and the exact frame to which it applies.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TableCalibrationV1 {
    pub schema_version: u8,
    pub calibration_revision: u64,
    pub camera_profile_id: String,
    pub rectification_revision: u64,
    pub image_space: ImageSpace,
    pub source_width_px: u32,
    pub source_height_px: u32,
    pub coordinate_frame: TableCoordinateFrame,
    pub table_width_mm: f64,
    pub table_height_mm: f64,
    /// Maps homogeneous source-image pixel coordinates to table millimeters.
    pub pixel_to_table_mm: [[f64; 3]; 3],
}

/// The v1 table axes are tied to the order used by the calibration corners.
///
/// Corner 0 is the image-space top-left corner; corners 1, 2, and 3 continue
/// clockwise. The origin is the center of the playing surface. X increases
/// from corner 0 toward corner 1; Y increases from corner 0 toward corner 3.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TableCoordinateFrame {
    CenterOriginXCorner0To1YCorner0To3,
}

/// Per-frame properties that must agree with the applied calibration.
#[derive(Debug, Clone, Copy)]
pub struct CalibrationContext<'a> {
    pub calibration_revision: u64,
    pub camera_profile_id: &'a str,
    pub rectification_revision: u64,
    pub image_space: ImageSpace,
    pub source_width_px: u32,
    pub source_height_px: u32,
}

/// Pixel coordinates in the original source image, before detector resizing.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ImagePointPx {
    pub x_px: f64,
    pub y_px: f64,
}

/// Physical coordinates on the calibrated playing-surface plane.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TablePointMm {
    pub x_mm: f64,
    pub y_mm: f64,
}

/// A bundle that has passed structural and numerical validation.
#[derive(Debug, Clone, PartialEq)]
pub struct ValidatedTableCalibrationV1 {
    calibration: TableCalibrationV1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CalibrationError {
    UnsupportedSchemaVersion,
    InvalidRevision,
    InvalidCameraProfileId,
    InvalidSourceDimensions,
    InvalidTableDimensions,
    InvalidHomography,
    CalibrationRevisionMismatch,
    CameraProfileMismatch,
    RectificationRevisionMismatch,
    ImageSpaceMismatch,
    SourceResolutionMismatch,
    NonFiniteImagePoint,
    ImagePointOutsideSource,
    ProjectiveSingularity,
    NonFiniteTablePoint,
}

impl Display for CalibrationError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "table calibration error: {self:?}")
    }
}

impl Error for CalibrationError {}

impl TableCalibrationV1 {
    /// Validate a decoded calibration bundle without applying it.
    pub fn validate(&self) -> Result<(), CalibrationError> {
        if self.schema_version != TABLE_CALIBRATION_SCHEMA_VERSION {
            return Err(CalibrationError::UnsupportedSchemaVersion);
        }
        if self.calibration_revision > MAX_SAFE_JSON_INTEGER
            || self.rectification_revision > MAX_SAFE_JSON_INTEGER
        {
            return Err(CalibrationError::InvalidRevision);
        }
        let profile_length = self.camera_profile_id.chars().count();
        if profile_length == 0
            || profile_length > MAX_IDENTIFIER_CHARACTERS
            || self.camera_profile_id.chars().any(char::is_control)
        {
            return Err(CalibrationError::InvalidCameraProfileId);
        }
        if self.source_width_px == 0 || self.source_height_px == 0 {
            return Err(CalibrationError::InvalidSourceDimensions);
        }
        if !self.table_width_mm.is_finite()
            || !self.table_height_mm.is_finite()
            || self.table_width_mm <= 0.0
            || self.table_height_mm <= 0.0
        {
            return Err(CalibrationError::InvalidTableDimensions);
        }
        validate_homography(&self.pixel_to_table_mm)
    }

    /// Convert this decoded bundle to a type that can safely project points.
    pub fn into_validated(self) -> Result<ValidatedTableCalibrationV1, CalibrationError> {
        self.validate()?;
        Ok(ValidatedTableCalibrationV1 { calibration: self })
    }
}

impl ValidatedTableCalibrationV1 {
    pub fn calibration_revision(&self) -> u64 {
        self.calibration.calibration_revision
    }

    pub fn camera_profile_id(&self) -> &str {
        &self.calibration.camera_profile_id
    }

    pub fn rectification_revision(&self) -> u64 {
        self.calibration.rectification_revision
    }

    pub fn image_space(&self) -> ImageSpace {
        self.calibration.image_space
    }

    pub fn source_dimensions_px(&self) -> (u32, u32) {
        (
            self.calibration.source_width_px,
            self.calibration.source_height_px,
        )
    }

    /// Reject a frame before iterating its detections, including empty frames.
    pub fn validate_frame_context(
        &self,
        context: CalibrationContext<'_>,
    ) -> Result<(), CalibrationError> {
        self.validate_context(context)
    }

    /// Project one source-image point into the calibrated table plane.
    ///
    /// Output is not clamped to the table dimensions: an out-of-table result
    /// remains measurable evidence for a later quality policy. The method
    /// returns errors for incompatible revisions and numerically undefined
    /// projective points instead of fabricating coordinates.
    pub fn project(
        &self,
        point: ImagePointPx,
        context: CalibrationContext<'_>,
    ) -> Result<TablePointMm, CalibrationError> {
        self.validate_context(context)?;
        if !point.x_px.is_finite() || !point.y_px.is_finite() {
            return Err(CalibrationError::NonFiniteImagePoint);
        }
        if point.x_px < 0.0
            || point.y_px < 0.0
            || point.x_px >= f64::from(self.calibration.source_width_px)
            || point.y_px >= f64::from(self.calibration.source_height_px)
        {
            return Err(CalibrationError::ImagePointOutsideSource);
        }

        let scale = self
            .calibration
            .pixel_to_table_mm
            .iter()
            .flatten()
            .fold(0.0_f64, |largest, value| largest.max(value.abs()));
        let matrix = self
            .calibration
            .pixel_to_table_mm
            .map(|row| row.map(|value| value / scale));
        let x_term = matrix[0][0] * point.x_px;
        let y_term = matrix[0][1] * point.y_px;
        let offset = matrix[0][2];
        let projected_x = x_term + y_term + offset;
        let x_term_y = matrix[1][0] * point.x_px;
        let y_term_y = matrix[1][1] * point.y_px;
        let offset_y = matrix[1][2];
        let projected_y = x_term_y + y_term_y + offset_y;
        let denominator_terms = [
            matrix[2][0] * point.x_px,
            matrix[2][1] * point.y_px,
            matrix[2][2],
        ];
        let denominator = denominator_terms.iter().sum::<f64>();
        let denominator_scale = denominator_terms.iter().map(|term| term.abs()).sum::<f64>();
        if !denominator.is_finite()
            || denominator.abs() <= f64::EPSILON * SINGULARITY_GUARD_MULTIPLIER * denominator_scale
        {
            return Err(CalibrationError::ProjectiveSingularity);
        }

        let result = TablePointMm {
            x_mm: projected_x / denominator,
            y_mm: projected_y / denominator,
        };
        if !result.x_mm.is_finite() || !result.y_mm.is_finite() {
            return Err(CalibrationError::NonFiniteTablePoint);
        }
        Ok(result)
    }

    fn validate_context(&self, context: CalibrationContext<'_>) -> Result<(), CalibrationError> {
        if context.calibration_revision != self.calibration.calibration_revision {
            return Err(CalibrationError::CalibrationRevisionMismatch);
        }
        if context.camera_profile_id != self.calibration.camera_profile_id {
            return Err(CalibrationError::CameraProfileMismatch);
        }
        if context.rectification_revision != self.calibration.rectification_revision {
            return Err(CalibrationError::RectificationRevisionMismatch);
        }
        if context.image_space != self.calibration.image_space {
            return Err(CalibrationError::ImageSpaceMismatch);
        }
        if context.source_width_px != self.calibration.source_width_px
            || context.source_height_px != self.calibration.source_height_px
        {
            return Err(CalibrationError::SourceResolutionMismatch);
        }
        Ok(())
    }
}

fn validate_homography(matrix: &[[f64; 3]; 3]) -> Result<(), CalibrationError> {
    if matrix.iter().flatten().any(|value| !value.is_finite()) {
        return Err(CalibrationError::InvalidHomography);
    }
    let scale = matrix
        .iter()
        .flatten()
        .fold(0.0_f64, |largest, value| largest.max(value.abs()));
    if scale == 0.0 {
        return Err(CalibrationError::InvalidHomography);
    }

    // A homography is invariant under nonzero scalar multiplication. Normalize
    // before the determinant check so calibration authors can serialize any
    // equivalent matrix scale without changing validity.
    let normalized = matrix.map(|row| row.map(|value| value / scale));
    let determinant = determinant(&normalized);
    let frobenius_norm = normalized
        .iter()
        .flatten()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();
    let relative_guard = f64::EPSILON * SINGULARITY_GUARD_MULTIPLIER * frobenius_norm.powi(3);
    if !determinant.is_finite() || determinant.abs() <= relative_guard {
        return Err(CalibrationError::InvalidHomography);
    }
    Ok(())
}

fn determinant(matrix: &[[f64; 3]; 3]) -> f64 {
    matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
        - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
        + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
}
