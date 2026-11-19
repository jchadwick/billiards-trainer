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
// The four-corner solve is performed in normalized source coordinates. This
// dimensionless area floor rejects quadrilaterals whose table plane is too
// close to a line for a useful, stable calibration.
const MIN_NORMALIZED_QUADRILATERAL_AREA: f64 = 1.0e-10;
const MIN_RELATIVE_PIVOT: f64 = 1.0e-12;
const MAX_CORNER_MAPPING_ERROR: f64 = 1.0e-8;

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

/// All operator-supplied inputs needed to solve a revision-bound calibration.
///
/// `corners_px` must be ordered clockwise in image coordinates, beginning at
/// the table's declared corner 0. That order defines the positive table axes;
/// the solver never guesses or silently reorders the operator's points.
#[derive(Debug, Clone, PartialEq)]
pub struct TableCalibrationSolverInput {
    pub calibration_revision: u64,
    pub camera_profile_id: String,
    pub rectification_revision: u64,
    pub image_space: ImageSpace,
    pub source_width_px: u32,
    pub source_height_px: u32,
    pub table_width_mm: f64,
    pub table_height_mm: f64,
    pub corners_px: [ImagePointPx; 4],
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
    InvalidCalibrationCorners,
    DegenerateCalibrationGeometry,
    UnstableHomography,
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
        self.validate_metadata()?;
        validate_homography(&self.pixel_to_table_mm)
    }

    fn validate_metadata(&self) -> Result<(), CalibrationError> {
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
        Ok(())
    }

    /// Convert this decoded bundle to a type that can safely project points.
    pub fn into_validated(self) -> Result<ValidatedTableCalibrationV1, CalibrationError> {
        self.validate()?;
        Ok(ValidatedTableCalibrationV1 { calibration: self })
    }
}

/// Solve the source-pixel to center-origin table-plane homography from four
/// ordered corner observations.
///
/// Source coordinates are centered and uniformly scaled before solving to
/// reduce conditioning problems. The resulting matrix is transformed back to
/// the original source image coordinates and passed through the same validator
/// and projector used by production inference before it is returned.
pub fn solve_table_calibration(
    input: TableCalibrationSolverInput,
) -> Result<TableCalibrationV1, CalibrationError> {
    let mut calibration = TableCalibrationV1 {
        schema_version: TABLE_CALIBRATION_SCHEMA_VERSION,
        calibration_revision: input.calibration_revision,
        camera_profile_id: input.camera_profile_id,
        rectification_revision: input.rectification_revision,
        image_space: input.image_space,
        source_width_px: input.source_width_px,
        source_height_px: input.source_height_px,
        coordinate_frame: TableCoordinateFrame::CenterOriginXCorner0To1YCorner0To3,
        table_width_mm: input.table_width_mm,
        table_height_mm: input.table_height_mm,
        // This finite nonsingular placeholder permits metadata validation
        // before attempting any geometry work; it is replaced by the solve.
        pixel_to_table_mm: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    };
    calibration.validate_metadata()?;
    validate_calibration_corners(
        &input.corners_px,
        input.source_width_px,
        input.source_height_px,
    )?;

    let geometry = normalize_source_corners(&input.corners_px)?;
    let target_corners = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)];
    let normalized_homography = solve_normalized_homography(&geometry.corners, &target_corners)?;
    calibration.pixel_to_table_mm = normalized_homography_to_table_mm(
        normalized_homography,
        geometry.center_x_px,
        geometry.center_y_px,
        geometry.scale_px,
        input.table_width_mm,
        input.table_height_mm,
    );

    let validated = calibration
        .clone()
        .into_validated()
        .map_err(|_| CalibrationError::UnstableHomography)?;
    let context = CalibrationContext {
        calibration_revision: calibration.calibration_revision,
        camera_profile_id: &calibration.camera_profile_id,
        rectification_revision: calibration.rectification_revision,
        image_space: calibration.image_space,
        source_width_px: calibration.source_width_px,
        source_height_px: calibration.source_height_px,
    };
    for (corner, (target_x, target_y)) in input.corners_px.iter().zip(target_corners) {
        let projected = validated
            .project(*corner, context)
            .map_err(|_| CalibrationError::UnstableHomography)?;
        let expected_x = target_x * input.table_width_mm / 2.0;
        let expected_y = target_y * input.table_height_mm / 2.0;
        let x_error = (projected.x_mm - expected_x).abs() / input.table_width_mm.max(1.0);
        let y_error = (projected.y_mm - expected_y).abs() / input.table_height_mm.max(1.0);
        if x_error > MAX_CORNER_MAPPING_ERROR || y_error > MAX_CORNER_MAPPING_ERROR {
            return Err(CalibrationError::UnstableHomography);
        }
    }

    Ok(calibration)
}

#[derive(Debug, Clone, Copy)]
struct NormalizedSourceCorners {
    center_x_px: f64,
    center_y_px: f64,
    scale_px: f64,
    corners: [(f64, f64); 4],
}

fn validate_calibration_corners(
    corners: &[ImagePointPx; 4],
    source_width_px: u32,
    source_height_px: u32,
) -> Result<(), CalibrationError> {
    for corner in corners {
        if !corner.x_px.is_finite() || !corner.y_px.is_finite() {
            return Err(CalibrationError::NonFiniteImagePoint);
        }
        if corner.x_px < 0.0
            || corner.y_px < 0.0
            || corner.x_px >= f64::from(source_width_px)
            || corner.y_px >= f64::from(source_height_px)
        {
            return Err(CalibrationError::ImagePointOutsideSource);
        }
    }

    let extent_x = corners
        .iter()
        .map(|corner| corner.x_px)
        .fold(f64::NEG_INFINITY, f64::max)
        - corners
            .iter()
            .map(|corner| corner.x_px)
            .fold(f64::INFINITY, f64::min);
    let extent_y = corners
        .iter()
        .map(|corner| corner.y_px)
        .fold(f64::NEG_INFINITY, f64::max)
        - corners
            .iter()
            .map(|corner| corner.y_px)
            .fold(f64::INFINITY, f64::min);
    let scale = extent_x.max(extent_y);
    if !scale.is_finite() || scale == 0.0 {
        return Err(CalibrationError::DegenerateCalibrationGeometry);
    }

    // In image coordinates (y grows down), a correctly ordered top-left,
    // top-right, bottom-right, bottom-left polygon has positive cross products.
    // Requiring every turn to clear a relative area floor rejects reversed,
    // self-intersecting, collinear and numerically near-degenerate inputs.
    let minimum_cross = MIN_NORMALIZED_QUADRILATERAL_AREA * scale * scale;
    for index in 0..4 {
        let first = corners[index];
        let second = corners[(index + 1) % 4];
        let third = corners[(index + 2) % 4];
        let first_x = second.x_px - first.x_px;
        let first_y = second.y_px - first.y_px;
        let second_x = third.x_px - second.x_px;
        let second_y = third.y_px - second.y_px;
        let cross = first_x * second_y - first_y * second_x;
        if !cross.is_finite() || cross <= 0.0 {
            return Err(CalibrationError::InvalidCalibrationCorners);
        }
        if cross <= minimum_cross {
            return Err(CalibrationError::DegenerateCalibrationGeometry);
        }
    }

    // Translate before scaling to avoid cancellation when a small table
    // polygon sits far from the source image origin.
    let origin = corners[0];
    let normalized_corners = corners.map(|corner| {
        (
            (corner.x_px - origin.x_px) / scale,
            (corner.y_px - origin.y_px) / scale,
        )
    });
    let area_twice = normalized_corners
        .iter()
        .zip(normalized_corners.iter().cycle().skip(1))
        .take(4)
        .map(|(first, second)| first.0 * second.1 - second.0 * first.1)
        .sum::<f64>();
    if !area_twice.is_finite() || area_twice <= MIN_NORMALIZED_QUADRILATERAL_AREA {
        return Err(CalibrationError::DegenerateCalibrationGeometry);
    }
    Ok(())
}

fn normalize_source_corners(
    corners: &[ImagePointPx; 4],
) -> Result<NormalizedSourceCorners, CalibrationError> {
    let center_x_px = corners.iter().map(|corner| corner.x_px).sum::<f64>() / 4.0;
    let center_y_px = corners.iter().map(|corner| corner.y_px).sum::<f64>() / 4.0;
    let scale_px = corners
        .iter()
        .map(|corner| {
            (corner.x_px - center_x_px)
                .abs()
                .max((corner.y_px - center_y_px).abs())
        })
        .fold(0.0_f64, f64::max);
    if !center_x_px.is_finite()
        || !center_y_px.is_finite()
        || !scale_px.is_finite()
        || scale_px == 0.0
    {
        return Err(CalibrationError::DegenerateCalibrationGeometry);
    }
    let normalized = corners.map(|corner| {
        (
            (corner.x_px - center_x_px) / scale_px,
            (corner.y_px - center_y_px) / scale_px,
        )
    });
    Ok(NormalizedSourceCorners {
        center_x_px,
        center_y_px,
        scale_px,
        corners: normalized,
    })
}

fn solve_normalized_homography(
    source: &[(f64, f64); 4],
    target: &[(f64, f64); 4],
) -> Result<[[f64; 3]; 3], CalibrationError> {
    // Unknowns are h00..h21 with h22 fixed to 1. Centering at the mean of the
    // four convex source corners places the normalization origin inside the
    // table quadrilateral, where a valid finite projective map has a nonzero
    // denominator and can safely use this scale.
    let mut augmented = [[0.0_f64; 9]; 8];
    for index in 0..4 {
        let (x, y) = source[index];
        let (u, v) = target[index];
        let x_row = index * 2;
        let y_row = x_row + 1;
        augmented[x_row] = [x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y, u];
        augmented[y_row] = [0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y, v];
    }
    let solution = solve_linear_system_8x8(augmented)?;
    let homography = [
        [solution[0], solution[1], solution[2]],
        [solution[3], solution[4], solution[5]],
        [solution[6], solution[7], 1.0],
    ];
    if homography.iter().flatten().any(|value| !value.is_finite()) {
        return Err(CalibrationError::UnstableHomography);
    }
    Ok(homography)
}

fn solve_linear_system_8x8(mut augmented: [[f64; 9]; 8]) -> Result<[f64; 8], CalibrationError> {
    let mut largest_initial_pivot = 0.0_f64;
    let mut smallest_pivot = f64::INFINITY;
    for pivot_column in 0..8 {
        let pivot_row = (pivot_column..8)
            .max_by(|left, right| {
                augmented[*left][pivot_column]
                    .abs()
                    .total_cmp(&augmented[*right][pivot_column].abs())
            })
            .ok_or(CalibrationError::UnstableHomography)?;
        let pivot = augmented[pivot_row][pivot_column].abs();
        if !pivot.is_finite() || pivot <= f64::MIN_POSITIVE {
            return Err(CalibrationError::UnstableHomography);
        }
        if pivot_column == 0 {
            largest_initial_pivot = pivot;
        }
        smallest_pivot = smallest_pivot.min(pivot);
        augmented.swap(pivot_column, pivot_row);

        let pivot_value = augmented[pivot_column][pivot_column];
        for value in augmented[pivot_column].iter_mut().skip(pivot_column) {
            *value /= pivot_value;
        }
        let normalized_pivot_row = augmented[pivot_column];
        for (row_index, row_values) in augmented.iter_mut().enumerate() {
            if row_index == pivot_column {
                continue;
            }
            let factor = row_values[pivot_column];
            for (column, value) in row_values.iter_mut().enumerate().skip(pivot_column) {
                *value -= factor * normalized_pivot_row[column];
            }
        }
    }
    if !smallest_pivot.is_finite() || smallest_pivot / largest_initial_pivot <= MIN_RELATIVE_PIVOT {
        return Err(CalibrationError::UnstableHomography);
    }
    let solution = std::array::from_fn(|row| augmented[row][8]);
    if solution.iter().any(|value| !value.is_finite()) {
        return Err(CalibrationError::UnstableHomography);
    }
    Ok(solution)
}

fn normalized_homography_to_table_mm(
    homography: [[f64; 3]; 3],
    center_x_px: f64,
    center_y_px: f64,
    scale_px: f64,
    table_width_mm: f64,
    table_height_mm: f64,
) -> [[f64; 3]; 3] {
    let normalized_to_source = [
        [1.0 / scale_px, 0.0, -center_x_px / scale_px],
        [0.0, 1.0 / scale_px, -center_y_px / scale_px],
        [0.0, 0.0, 1.0],
    ];
    let pixel_to_normalized_table = multiply_3x3(homography, normalized_to_source);
    let mut pixel_to_table_mm = pixel_to_normalized_table;
    for value in &mut pixel_to_table_mm[0] {
        *value *= table_width_mm / 2.0;
    }
    for value in &mut pixel_to_table_mm[1] {
        *value *= table_height_mm / 2.0;
    }
    pixel_to_table_mm
}

fn multiply_3x3(left: [[f64; 3]; 3], right: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    std::array::from_fn(|row| {
        std::array::from_fn(|column| {
            (0..3)
                .map(|index| left[row][index] * right[index][column])
                .sum()
        })
    })
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

    pub fn playing_surface_dimensions_mm(&self) -> (f64, f64) {
        (
            self.calibration.table_width_mm,
            self.calibration.table_height_mm,
        )
    }

    /// Source-image pixels to center-origin playing-surface millimeters.
    /// Published with its revision so clients can project table-space paths
    /// through the exact homography used for production ball coordinates.
    pub fn pixel_to_table_mm(&self) -> [[f64; 3]; 3] {
        self.calibration.pixel_to_table_mm
    }

    pub fn source_dimensions_px(&self) -> (u32, u32) {
        (
            self.calibration.source_width_px,
            self.calibration.source_height_px,
        )
    }

    /// Include the actual calibration geometry without its save revision.
    pub(crate) fn capture_geometry_key(&self) -> impl Serialize + '_ {
        (
            &self.calibration.coordinate_frame,
            self.playing_surface_dimensions_mm(),
            &self.calibration.pixel_to_table_mm,
        )
    }

    /// Locate a table-plane point in the image for an approximate search seed.
    /// A ball center lies above this plane, so this is not a correspondence.
    pub(crate) fn image_search_seed(&self, [x, y]: [f64; 2]) -> Option<[f64; 2]> {
        let [a, b, c] = self.calibration.pixel_to_table_mm;
        let a = [a[0] - x * c[0], a[1] - x * c[1], x * c[2] - a[2]];
        let b = [b[0] - y * c[0], b[1] - y * c[1], y * c[2] - b[2]];
        let determinant = a[0] * b[1] - a[1] * b[0];
        let point = [
            (a[2] * b[1] - a[1] * b[2]) / determinant,
            (a[0] * b[2] - a[2] * b[0]) / determinant,
        ];
        point.into_iter().all(f64::is_finite).then_some(point)
    }

    /// Whether a projected point lies on the calibrated playing surface.
    /// The calibration maps the cloth boundary to a center-origin rectangle;
    /// points exactly on its edge count as inside so an endpoint on the rail
    /// line can still identify the cue tip side.
    pub fn contains_playing_surface_point(&self, point: TablePointMm) -> bool {
        point.x_mm.is_finite()
            && point.y_mm.is_finite()
            && point.x_mm.abs() <= self.calibration.table_width_mm / 2.0
            && point.y_mm.abs() <= self.calibration.table_height_mm / 2.0
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
