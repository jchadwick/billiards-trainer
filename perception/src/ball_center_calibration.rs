//! Versioned, revision-bound calibration of refined 2D ball centers.
//!
//! This asset is deliberately separate from the four-corner table calibration:
//! its correspondences describe measured ball centers, not cloth corners or
//! detector bounding-box midpoints. Every observation is retained. A normalized
//! overdetermined homography is fitted with pivoted, reorthogonalized QR; normal
//! equations (which square the condition number) are never formed.

use crate::ball_localization::REGULATION_BALL_DIAMETER_MM;
use crate::calibration::{CalibrationContext, ImagePointPx, TableCoordinateFrame, TablePointMm};
use crate::ImageSpace;
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::fmt::{Display, Formatter};

pub const BALL_CENTER_CALIBRATION_SCHEMA_VERSION: u8 = 1;
// Wire/resource bounds, not assumptions about the camera or physical table.
pub const MAX_BALL_CENTER_CORRESPONDENCES: usize = 4096;
const MAX_SAFE_JSON_INTEGER: u64 = 9_007_199_254_740_991;
const MAX_PROVENANCE_BYTES: usize = 4096;
const PARAMETER_COUNT: usize = 8;
// Each leave-one-out fit must itself be overdetermined: 2*(N-1)>8.
const MINIMUM_WITH_OVERDETERMINED_HOLDOUT: usize = PARAMETER_COUNT / 2 + 2;
type Matrix3 = [[f64; 3]; 3];

struct NormalizedPointSet {
    points: Vec<[f64; 2]>,
    transform: Matrix3,
    inverse: Matrix3,
}

/// All numeric acceptance settings travel with the fitted asset for auditability.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BallCenterCalibrationSolverPolicy {
    pub minimum_correspondences: usize,
    pub maximum_correspondences: usize,
    pub relative_rank_tolerance: f64,
    pub maximum_condition_number: f64,
    pub numerical_tolerance: f64,
    /// Optional application acceptance gate; None makes no accuracy assertion.
    pub maximum_fit_rms_mm: Option<f64>,
}

impl Default for BallCenterCalibrationSolverPolicy {
    fn default() -> Self {
        Self {
            minimum_correspondences: MINIMUM_WITH_OVERDETERMINED_HOLDOUT,
            maximum_correspondences: 1024,
            relative_rank_tolerance: 1e-10,
            maximum_condition_number: 1e8,
            numerical_tolerance: 1e-10,
            maximum_fit_rms_mm: None,
        }
    }
}

impl BallCenterCalibrationSolverPolicy {
    pub fn validate(&self) -> Result<(), BallCenterCalibrationError> {
        if self.minimum_correspondences < MINIMUM_WITH_OVERDETERMINED_HOLDOUT
            || self.maximum_correspondences < self.minimum_correspondences
            || self.maximum_correspondences > MAX_BALL_CENTER_CORRESPONDENCES
            || !self.relative_rank_tolerance.is_finite()
            || self.relative_rank_tolerance <= 0.0
            || self.relative_rank_tolerance >= 1.0
            || !self.maximum_condition_number.is_finite()
            || self.maximum_condition_number < 1.0
            || !self.numerical_tolerance.is_finite()
            || self.numerical_tolerance <= 0.0
            || self.numerical_tolerance >= 1.0
            || self
                .maximum_fit_rms_mm
                .is_some_and(|v| !v.is_finite() || v <= 0.0)
        {
            return Err(BallCenterCalibrationError::InvalidPolicy);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BallCenterCorrespondence {
    #[serde(deserialize_with = "deserialize_image_point")]
    pub image_point_px: ImagePointPx,
    /// Center-origin table coordinates in millimeters, with the table's axes.
    #[serde(deserialize_with = "deserialize_table_point")]
    pub center_table_mm: TablePointMm,
    /// Measurement method/reference and source-frame identity; never inferred
    /// solely from apparent contact with a pocket jaw or from a detector box.
    pub provenance: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BallCenterCalibrationResidual {
    pub correspondence_index: usize,
    #[serde(deserialize_with = "deserialize_table_point")]
    pub projected_table_mm: TablePointMm,
    pub error_x_mm: f64,
    pub error_y_mm: f64,
    pub distance_mm: f64,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BallCenterCalibrationFitReport {
    /// Frobenius condition number of the normalized, pivoted QR triangular
    /// factor. It is a conservative diagnostic, not a physical uncertainty.
    pub normalized_design_condition_number: f64,
    pub rms_mm: f64,
    pub maximum_residual_mm: f64,
    pub residuals: Vec<BallCenterCalibrationResidual>,
    pub leave_one_out_rms_mm: f64,
    pub leave_one_out_maximum_residual_mm: f64,
    pub leave_one_out: Vec<BallCenterCalibrationHoldout>,
    /// Fractions of the entire source-image and physical-table rectangles.
    /// Coverage is reported, not converted into an arbitrary pass threshold.
    pub source_hull_area_fraction: f64,
    pub table_hull_area_fraction: f64,
    pub source_reference_hull_px: Vec<[f64; 2]>,
    pub table_reference_hull_mm: Vec<[f64; 2]>,
    pub source_image_corners_extrapolated: [bool; 4],
    pub physical_table_corners_extrapolated: [bool; 4],
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BallCenterCalibrationV1 {
    pub schema_version: u8,
    pub calibration_revision: u64,
    pub camera_profile_id: String,
    pub optics_profile_id: String,
    pub optics_revision: u64,
    pub refiner_revision: u64,
    pub table_revision: u64,
    pub rectification_revision: u64,
    pub image_space: ImageSpace,
    pub source_width_px: u32,
    pub source_height_px: u32,
    pub coordinate_frame: TableCoordinateFrame,
    pub table_width_mm: f64,
    pub table_height_mm: f64,
    pub ball_diameter_mm: f64,
    pub image_to_table_mm: [[f64; 3]; 3],
    pub provenance: String,
    pub solver_policy: BallCenterCalibrationSolverPolicy,
    pub correspondences: Vec<BallCenterCorrespondence>,
    pub fit_report: BallCenterCalibrationFitReport,
}

/// An observation excluded from both parameter fitting and conditioning checks.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BallCenterCalibrationHoldout {
    pub held_out_index: usize,
    pub training_correspondences: usize,
    pub normalized_design_condition_number: f64,
    pub projected_table_mm: [f64; 2],
    pub error_mm: [f64; 2],
    pub distance_mm: f64,
    pub source_point_extrapolated: bool,
}

/// Immutable load-validated asset for allocation-free per-ball projection.
/// Keeping the underlying asset private prevents invalidating that guarantee.
#[derive(Debug, Clone, PartialEq)]
pub struct ValidatedBallCenterCalibrationV1 {
    asset: BallCenterCalibrationV1,
}

impl ValidatedBallCenterCalibrationV1 {
    pub fn as_asset(&self) -> &BallCenterCalibrationV1 {
        &self.asset
    }

    pub fn validate_context(
        &self,
        context: BallCenterCalibrationContext<'_>,
    ) -> Result<(), BallCenterCalibrationError> {
        self.asset.validate_context(context)
    }

    pub fn project(
        &self,
        point: ImagePointPx,
        context: BallCenterCalibrationContext<'_>,
    ) -> Result<TablePointMm, BallCenterCalibrationError> {
        self.asset.validate_context(context)?;
        self.asset.validate_image_point(point)?;
        project_matrix(
            self.asset.image_to_table_mm,
            point,
            self.asset.solver_policy.numerical_tolerance,
        )
    }
}

#[derive(Debug, Clone)]
pub struct BallCenterCalibrationSolverInput {
    pub calibration_revision: u64,
    pub camera_profile_id: String,
    pub optics_profile_id: String,
    pub optics_revision: u64,
    pub refiner_revision: u64,
    pub table_revision: u64,
    pub rectification_revision: u64,
    pub image_space: ImageSpace,
    pub source_width_px: u32,
    pub source_height_px: u32,
    pub table_width_mm: f64,
    pub table_height_mm: f64,
    pub ball_diameter_mm: f64,
    pub provenance: String,
    pub solver_policy: BallCenterCalibrationSolverPolicy,
    pub correspondences: Vec<BallCenterCorrespondence>,
}

/// All identities that can change the meaning of a refined image center.
#[derive(Debug, Clone, Copy)]
pub struct BallCenterCalibrationContext<'a> {
    pub frame: CalibrationContext<'a>,
    pub optics_profile_id: &'a str,
    pub optics_revision: u64,
    pub refiner_revision: u64,
    pub table_revision: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BallCenterCalibrationError {
    UnsupportedSchemaVersion,
    InvalidPolicy,
    InvalidMetadata,
    InvalidProvenance,
    TooFewCorrespondences,
    TooManyCorrespondences,
    InvalidCorrespondence,
    DuplicateCorrespondence,
    DegenerateGeometry,
    IllConditioned,
    InvalidHomography,
    InvalidFitReport,
    FitErrorExceedsLimit,
    ContextMismatch(&'static str),
    PointOutsideImage,
    ProjectiveSingularity,
    NonFiniteProjection,
}

impl Display for BallCenterCalibrationError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "ball-center calibration error: {self:?}")
    }
}
impl Error for BallCenterCalibrationError {}

// Existing shared point types permit unknown fields. This new asset's strict
// wire contract must also reject unexpected nested coordinates (including Z).
fn deserialize_image_point<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> Result<ImagePointPx, D::Error> {
    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Point {
        x_px: f64,
        y_px: f64,
    }
    let point = Point::deserialize(deserializer)?;
    Ok(ImagePointPx {
        x_px: point.x_px,
        y_px: point.y_px,
    })
}

fn deserialize_table_point<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> Result<TablePointMm, D::Error> {
    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Point {
        x_mm: f64,
        y_mm: f64,
    }
    let point = Point::deserialize(deserializer)?;
    Ok(TablePointMm {
        x_mm: point.x_mm,
        y_mm: point.y_mm,
    })
}

fn valid_text(value: &str) -> bool {
    !value.trim().is_empty()
        && value.len() <= MAX_PROVENANCE_BYTES
        && !value.chars().any(char::is_control)
}

fn finite_point(point: TablePointMm) -> bool {
    point.x_mm.is_finite() && point.y_mm.is_finite()
}

impl BallCenterCalibrationV1 {
    pub fn into_validated(
        self,
    ) -> Result<ValidatedBallCenterCalibrationV1, BallCenterCalibrationError> {
        self.validate()?;
        Ok(ValidatedBallCenterCalibrationV1 { asset: self })
    }
    fn validate_metadata(&self) -> Result<(), BallCenterCalibrationError> {
        self.solver_policy.validate()?;
        if self.schema_version != BALL_CENTER_CALIBRATION_SCHEMA_VERSION {
            return Err(BallCenterCalibrationError::UnsupportedSchemaVersion);
        }
        if [
            self.calibration_revision,
            self.optics_revision,
            self.refiner_revision,
            self.table_revision,
            self.rectification_revision,
        ]
        .iter()
        .any(|r| *r > MAX_SAFE_JSON_INTEGER)
            || !valid_text(&self.camera_profile_id)
            || !valid_text(&self.optics_profile_id)
            || self.source_width_px == 0
            || self.source_height_px == 0
            || ![
                self.table_width_mm,
                self.table_height_mm,
                self.ball_diameter_mm,
            ]
            .iter()
            .all(|v| v.is_finite() && *v > 0.0)
            || self.ball_diameter_mm != REGULATION_BALL_DIAMETER_MM
            || self.ball_diameter_mm >= self.table_width_mm.min(self.table_height_mm)
        {
            return Err(BallCenterCalibrationError::InvalidMetadata);
        }
        if !valid_text(&self.provenance) {
            return Err(BallCenterCalibrationError::InvalidProvenance);
        }
        if self.correspondences.len() < self.solver_policy.minimum_correspondences {
            return Err(BallCenterCalibrationError::TooFewCorrespondences);
        }
        if self.correspondences.len() > self.solver_policy.maximum_correspondences {
            return Err(BallCenterCalibrationError::TooManyCorrespondences);
        }
        let key = |x: f64, y: f64| {
            (
                if x == 0.0 { 0 } else { x.to_bits() },
                if y == 0.0 { 0 } else { y.to_bits() },
            )
        };
        let mut source_seen = std::collections::BTreeSet::new();
        let mut target_seen = std::collections::BTreeSet::new();
        for point in &self.correspondences {
            self.validate_image_point(point.image_point_px)?;
            if !finite_point(point.center_table_mm)
                || point.center_table_mm.x_mm.abs() > self.table_width_mm / 2.0
                || point.center_table_mm.y_mm.abs() > self.table_height_mm / 2.0
            {
                return Err(BallCenterCalibrationError::InvalidCorrespondence);
            }
            if !source_seen.insert(key(point.image_point_px.x_px, point.image_point_px.y_px))
                || !target_seen.insert(key(point.center_table_mm.x_mm, point.center_table_mm.y_mm))
            {
                return Err(BallCenterCalibrationError::DuplicateCorrespondence);
            }
            if !valid_text(&point.provenance) {
                return Err(BallCenterCalibrationError::InvalidProvenance);
            }
        }
        Ok(())
    }

    fn validate_image_point(&self, point: ImagePointPx) -> Result<(), BallCenterCalibrationError> {
        if !point.x_px.is_finite()
            || !point.y_px.is_finite()
            || point.x_px < 0.0
            || point.y_px < 0.0
            || point.x_px >= f64::from(self.source_width_px)
            || point.y_px >= f64::from(self.source_height_px)
        {
            return Err(BallCenterCalibrationError::PointOutsideImage);
        }
        Ok(())
    }

    /// Validate decoded assets, including provenance and recomputed residuals.
    /// This does not turn training residuals into independent physical accuracy.
    pub fn validate(&self) -> Result<(), BallCenterCalibrationError> {
        self.validate_metadata()?;
        validate_matrix(
            self.image_to_table_mm,
            self.source_width_px,
            self.source_height_px,
            self.solver_policy.numerical_tolerance,
        )?;
        // A collinear correspondence set is invalid even if an externally
        // supplied nonsingular matrix happens to map those points exactly.
        let source: Vec<_> = self
            .correspondences
            .iter()
            .map(|p| [p.image_point_px.x_px, p.image_point_px.y_px])
            .collect();
        let target: Vec<_> = self
            .correspondences
            .iter()
            .map(|p| [p.center_table_mm.x_mm, p.center_table_mm.y_mm])
            .collect();
        normalize_points(&source, self.solver_policy.relative_rank_tolerance)?;
        normalize_points(&target, self.solver_policy.relative_rank_tolerance)?;
        let report = &self.fit_report;
        if !report.normalized_design_condition_number.is_finite()
            || report.normalized_design_condition_number < 1.0
            || report.normalized_design_condition_number
                > self.solver_policy.maximum_condition_number
            || report.residuals.len() != self.correspondences.len()
        {
            return Err(BallCenterCalibrationError::InvalidFitReport);
        }
        let expected = complete_report(self)?;
        let close = |a: f64, b: f64| {
            a.is_finite()
                && b.is_finite()
                && (a - b).abs()
                    <= self.solver_policy.numerical_tolerance * (1.0 + a.abs().max(b.abs()))
        };
        if !report_values_match(
            &serde_json::to_value(report)
                .map_err(|_| BallCenterCalibrationError::InvalidFitReport)?,
            &serde_json::to_value(&expected)
                .map_err(|_| BallCenterCalibrationError::InvalidFitReport)?,
            &close,
        ) {
            return Err(BallCenterCalibrationError::InvalidFitReport);
        }
        if self
            .solver_policy
            .maximum_fit_rms_mm
            .is_some_and(|limit| report.rms_mm > limit)
        {
            return Err(BallCenterCalibrationError::FitErrorExceedsLimit);
        }
        Ok(())
    }

    pub fn validate_context(
        &self,
        context: BallCenterCalibrationContext<'_>,
    ) -> Result<(), BallCenterCalibrationError> {
        let frame = context.frame;
        for (matches, name) in [
            (
                frame.calibration_revision == self.calibration_revision,
                "calibration_revision",
            ),
            (
                frame.camera_profile_id == self.camera_profile_id,
                "camera_profile_id",
            ),
            (
                frame.rectification_revision == self.rectification_revision,
                "rectification_revision",
            ),
            (frame.image_space == self.image_space, "image_space"),
            (
                frame.source_width_px == self.source_width_px
                    && frame.source_height_px == self.source_height_px,
                "source_geometry",
            ),
            (
                context.optics_profile_id == self.optics_profile_id,
                "optics_profile_id",
            ),
            (
                context.optics_revision == self.optics_revision,
                "optics_revision",
            ),
            (
                context.refiner_revision == self.refiner_revision,
                "refiner_revision",
            ),
            (
                context.table_revision == self.table_revision,
                "table_revision",
            ),
        ] {
            if !matches {
                return Err(BallCenterCalibrationError::ContextMismatch(name));
            }
        }
        Ok(())
    }

    /// Safe entry point for an untrusted, decoded asset. Callers processing
    /// many balls can validate once upstream; this public API remains fail-closed.
    pub fn project(
        &self,
        point: ImagePointPx,
        context: BallCenterCalibrationContext<'_>,
    ) -> Result<TablePointMm, BallCenterCalibrationError> {
        self.validate()?;
        self.validate_context(context)?;
        self.validate_image_point(point)?;
        project_matrix(
            self.image_to_table_mm,
            point,
            self.solver_policy.numerical_tolerance,
        )
    }
}

fn project_matrix(
    h: [[f64; 3]; 3],
    p: ImagePointPx,
    tolerance: f64,
) -> Result<TablePointMm, BallCenterCalibrationError> {
    let denominator = h[2][0] * p.x_px + h[2][1] * p.y_px + h[2][2];
    let scale = (h[2][0] * p.x_px).abs() + (h[2][1] * p.y_px).abs() + h[2][2].abs();
    if !denominator.is_finite() || denominator.abs() <= tolerance * scale {
        return Err(BallCenterCalibrationError::ProjectiveSingularity);
    }
    let result = TablePointMm {
        x_mm: (h[0][0] * p.x_px + h[0][1] * p.y_px + h[0][2]) / denominator,
        y_mm: (h[1][0] * p.x_px + h[1][1] * p.y_px + h[1][2]) / denominator,
    };
    if !finite_point(result) {
        return Err(BallCenterCalibrationError::NonFiniteProjection);
    }
    Ok(result)
}

fn validate_matrix(
    h: [[f64; 3]; 3],
    width: u32,
    height: u32,
    tolerance: f64,
) -> Result<(), BallCenterCalibrationError> {
    if !h.iter().flatten().all(|x| x.is_finite()) {
        return Err(BallCenterCalibrationError::InvalidHomography);
    }
    let scale = h.iter().flatten().fold(0.0_f64, |a, b| a.max(b.abs()));
    if scale == 0.0 {
        return Err(BallCenterCalibrationError::InvalidHomography);
    }
    let a = h.map(|row| row.map(|v| v / scale));
    let det = a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
    // Relative to the sum of determinant terms, rather than raw pixel/mm
    // matrix norm: large coordinate translations alone are not degeneracy.
    let det_scale = (a[0][0] * a[1][1] * a[2][2]).abs()
        + (a[0][0] * a[1][2] * a[2][1]).abs()
        + (a[0][1] * a[1][0] * a[2][2]).abs()
        + (a[0][1] * a[1][2] * a[2][0]).abs()
        + (a[0][2] * a[1][0] * a[2][1]).abs()
        + (a[0][2] * a[1][1] * a[2][0]).abs();
    if !det.is_finite() || det.abs() <= tolerance * det_scale || det_scale == 0.0 {
        return Err(BallCenterCalibrationError::InvalidHomography);
    }
    let mut sign = None;
    for x in [0.0, f64::from(width - 1)] {
        for y in [0.0, f64::from(height - 1)] {
            let p = ImagePointPx { x_px: x, y_px: y };
            project_matrix(h, p, tolerance)?;
            let positive = h[2][0] * x + h[2][1] * y + h[2][2] > 0.0;
            if sign.is_some_and(|previous| previous != positive) {
                return Err(BallCenterCalibrationError::InvalidHomography);
            }
            sign = Some(positive);
        }
    }
    Ok(())
}

fn residuals(
    asset: &BallCenterCalibrationV1,
) -> Result<BallCenterCalibrationFitReport, BallCenterCalibrationError> {
    let mut values = Vec::with_capacity(asset.correspondences.len());
    let mut squares = 0.0;
    let mut maximum = 0.0_f64;
    for (index, point) in asset.correspondences.iter().enumerate() {
        let projected = project_matrix(
            asset.image_to_table_mm,
            point.image_point_px,
            asset.solver_policy.numerical_tolerance,
        )?;
        let dx = projected.x_mm - point.center_table_mm.x_mm;
        let dy = projected.y_mm - point.center_table_mm.y_mm;
        let distance = dx.hypot(dy);
        squares += distance * distance;
        maximum = maximum.max(distance);
        values.push(BallCenterCalibrationResidual {
            correspondence_index: index,
            projected_table_mm: projected,
            error_x_mm: dx,
            error_y_mm: dy,
            distance_mm: distance,
        });
    }
    let rms = (squares / values.len() as f64).sqrt();
    if !rms.is_finite() {
        return Err(BallCenterCalibrationError::NonFiniteProjection);
    }
    Ok(BallCenterCalibrationFitReport {
        normalized_design_condition_number: asset.fit_report.normalized_design_condition_number,
        rms_mm: rms,
        maximum_residual_mm: maximum,
        residuals: values,
        ..BallCenterCalibrationFitReport::default()
    })
}

fn multiply(a: [[f64; 3]; 3], b: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut result = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                result[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    result
}

/// Hartley normalization with an explicit 2D spread check. Centering avoids
/// dependence of conditioning on either image resolution or millimeter units.
fn normalize_points(
    points: &[[f64; 2]],
    tolerance: f64,
) -> Result<NormalizedPointSet, BallCenterCalibrationError> {
    let n = points.len() as f64;
    let center = [
        points.iter().map(|p| p[0] / n).sum::<f64>(),
        points.iter().map(|p| p[1] / n).sum::<f64>(),
    ];
    let mut xx = 0.0;
    let mut yy = 0.0;
    let mut xy = 0.0;
    for p in points {
        let x = p[0] - center[0];
        let y = p[1] - center[1];
        xx += x * x / n;
        yy += y * y / n;
        xy += x * y / n;
    }
    let spread = xx + yy;
    if !spread.is_finite() || spread <= 0.0 || xx * yy - xy * xy <= tolerance * spread * spread {
        return Err(BallCenterCalibrationError::DegenerateGeometry);
    }
    let scale = (2.0 / spread).sqrt();
    let transform = [
        [scale, 0.0, -scale * center[0]],
        [0.0, scale, -scale * center[1]],
        [0.0, 0.0, 1.0],
    ];
    let inverse = [
        [1.0 / scale, 0.0, center[0]],
        [0.0, 1.0 / scale, center[1]],
        [0.0, 0.0, 1.0],
    ];
    let normalized = points
        .iter()
        .map(|p| [(p[0] - center[0]) * scale, (p[1] - center[1]) * scale])
        .collect();
    Ok(NormalizedPointSet {
        points: normalized,
        transform,
        inverse,
    })
}

fn back_substitute(
    r: &[[f64; PARAMETER_COUNT]; PARAMETER_COUNT],
    b: [f64; PARAMETER_COUNT],
) -> [f64; PARAMETER_COUNT] {
    let mut x = [0.0; PARAMETER_COUNT];
    for i in (0..PARAMETER_COUNT).rev() {
        x[i] = (b[i]
            - (i + 1..PARAMETER_COUNT)
                .map(|j| r[i][j] * x[j])
                .sum::<f64>())
            / r[i][i];
    }
    x
}

fn qr_solve(
    rows: &[[f64; PARAMETER_COUNT]],
    target: &[f64],
    policy: &BallCenterCalibrationSolverPolicy,
) -> Result<([f64; PARAMETER_COUNT], f64), BallCenterCalibrationError> {
    let mut columns: Vec<Vec<f64>> = (0..PARAMETER_COUNT)
        .map(|j| rows.iter().map(|row| row[j]).collect())
        .collect();
    let mut permutation: Vec<_> = (0..PARAMETER_COUNT).collect();
    let mut r = [[0.0; PARAMETER_COUNT]; PARAMETER_COUNT];
    let mut rhs = [0.0; PARAMETER_COUNT];
    let mut first_norm = 0.0;
    for k in 0..PARAMETER_COUNT {
        let pivot = (k..PARAMETER_COUNT)
            .max_by(|a, b| {
                let norm = |j: usize| columns[j].iter().map(|v| v * v).sum::<f64>();
                norm(*a).total_cmp(&norm(*b))
            })
            .ok_or(BallCenterCalibrationError::DegenerateGeometry)?;
        columns.swap(k, pivot);
        permutation.swap(k, pivot);
        for row in r.iter_mut().take(k) {
            row.swap(k, pivot);
        }
        let norm = columns[k].iter().map(|v| v * v).sum::<f64>().sqrt();
        if k == 0 {
            first_norm = norm;
        }
        if !norm.is_finite() || norm <= first_norm * policy.relative_rank_tolerance || norm == 0.0 {
            return Err(BallCenterCalibrationError::DegenerateGeometry);
        }
        r[k][k] = norm;
        let q: Vec<_> = columns[k].iter().map(|v| v / norm).collect();
        rhs[k] = q.iter().zip(target).map(|(a, b)| a * b).sum();
        for j in k + 1..PARAMETER_COUNT {
            // Two passes avoid loss of orthogonality in near-dependent views.
            for _ in 0..2 {
                let dot: f64 = q.iter().zip(&columns[j]).map(|(a, b)| a * b).sum();
                r[k][j] += dot;
                for (v, qv) in columns[j].iter_mut().zip(&q) {
                    *v -= dot * qv;
                }
            }
        }
    }
    let norm_r = r.iter().flatten().map(|v| v * v).sum::<f64>().sqrt();
    let mut inverse_squared = 0.0;
    for col in 0..PARAMETER_COUNT {
        let mut unit = [0.0; PARAMETER_COUNT];
        unit[col] = 1.0;
        inverse_squared += back_substitute(&r, unit).iter().map(|v| v * v).sum::<f64>();
    }
    let condition = norm_r * inverse_squared.sqrt();
    if !condition.is_finite() || condition > policy.maximum_condition_number {
        return Err(BallCenterCalibrationError::IllConditioned);
    }
    let permuted = back_substitute(&r, rhs);
    let mut result = [0.0; PARAMETER_COUNT];
    for i in 0..PARAMETER_COUNT {
        result[permutation[i]] = permuted[i];
    }
    if !result.iter().all(|v| v.is_finite()) {
        return Err(BallCenterCalibrationError::IllConditioned);
    }
    Ok((result, condition))
}

fn fit_homography(
    correspondences: &[BallCenterCorrespondence],
    policy: &BallCenterCalibrationSolverPolicy,
) -> Result<(Matrix3, f64), BallCenterCalibrationError> {
    let source: Vec<_> = correspondences
        .iter()
        .map(|p| [p.image_point_px.x_px, p.image_point_px.y_px])
        .collect();
    let target: Vec<_> = correspondences
        .iter()
        .map(|p| [p.center_table_mm.x_mm, p.center_table_mm.y_mm])
        .collect();
    let source = normalize_points(&source, policy.relative_rank_tolerance)?;
    let target = normalize_points(&target, policy.relative_rank_tolerance)?;
    let mut rows = Vec::with_capacity(source.points.len() * 2);
    let mut rhs = Vec::with_capacity(source.points.len() * 2);
    for ([x, y], [u, v]) in source
        .points
        .iter()
        .copied()
        .zip(target.points.iter().copied())
    {
        rows.push([x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y]);
        rhs.push(u);
        rows.push([0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y]);
        rhs.push(v);
    }
    let (h, condition) = qr_solve(&rows, &rhs, policy)?;
    let normalized = [[h[0], h[1], h[2]], [h[3], h[4], h[5]], [h[6], h[7], 1.0]];
    Ok((
        multiply(target.inverse, multiply(normalized, source.transform)),
        condition,
    ))
}

pub fn solve_ball_center_calibration(
    input: BallCenterCalibrationSolverInput,
) -> Result<BallCenterCalibrationV1, BallCenterCalibrationError> {
    let mut asset = BallCenterCalibrationV1 {
        schema_version: BALL_CENTER_CALIBRATION_SCHEMA_VERSION,
        calibration_revision: input.calibration_revision,
        camera_profile_id: input.camera_profile_id,
        optics_profile_id: input.optics_profile_id,
        optics_revision: input.optics_revision,
        refiner_revision: input.refiner_revision,
        table_revision: input.table_revision,
        rectification_revision: input.rectification_revision,
        image_space: input.image_space,
        source_width_px: input.source_width_px,
        source_height_px: input.source_height_px,
        coordinate_frame: TableCoordinateFrame::CenterOriginXCorner0To1YCorner0To3,
        table_width_mm: input.table_width_mm,
        table_height_mm: input.table_height_mm,
        ball_diameter_mm: input.ball_diameter_mm,
        image_to_table_mm: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        provenance: input.provenance,
        solver_policy: input.solver_policy,
        correspondences: input.correspondences,
        fit_report: BallCenterCalibrationFitReport::default(),
    };
    asset.validate_metadata()?;
    asset.image_to_table_mm = fit_homography(&asset.correspondences, &asset.solver_policy)?.0;
    asset.fit_report = complete_report(&asset)?;
    asset.validate()?;
    Ok(asset)
}

fn report_values_match(
    actual: &serde_json::Value,
    expected: &serde_json::Value,
    close: &impl Fn(f64, f64) -> bool,
) -> bool {
    use serde_json::Value;
    match (actual, expected) {
        (Value::Number(a), Value::Number(b)) => {
            if a.is_u64() && b.is_u64() {
                a.as_u64() == b.as_u64()
            } else {
                a.as_f64().zip(b.as_f64()).is_some_and(|(a, b)| close(a, b))
            }
        }
        (Value::Array(a), Value::Array(b)) => {
            a.len() == b.len()
                && a.iter()
                    .zip(b)
                    .all(|(a, b)| report_values_match(a, b, close))
        }
        (Value::Object(a), Value::Object(b)) => {
            a.len() == b.len()
                && a.iter().all(|(key, value)| {
                    b.get(key)
                        .is_some_and(|b| report_values_match(value, b, close))
                })
        }
        _ => actual == expected,
    }
}

fn cross(a: [f64; 2], b: [f64; 2], c: [f64; 2]) -> f64 {
    (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
}

/// A deterministic counter-clockwise hull; collinear edge points add no area.
fn convex_hull(points: &[[f64; 2]]) -> Vec<[f64; 2]> {
    let mut sorted = points.to_vec();
    sorted.sort_by(|a, b| a[0].total_cmp(&b[0]).then_with(|| a[1].total_cmp(&b[1])));
    sorted.dedup();
    if sorted.len() < 3 {
        return sorted;
    }
    let mut lower = Vec::new();
    for &point in &sorted {
        while lower.len() >= 2 && cross(lower[lower.len() - 2], lower[lower.len() - 1], point) <= 0.
        {
            lower.pop();
        }
        lower.push(point);
    }
    let mut upper = Vec::new();
    for &point in sorted.iter().rev() {
        while upper.len() >= 2 && cross(upper[upper.len() - 2], upper[upper.len() - 1], point) <= 0.
        {
            upper.pop();
        }
        upper.push(point);
    }
    lower.pop();
    upper.pop();
    lower.extend(upper);
    lower
}
fn hull_area(hull: &[[f64; 2]]) -> f64 {
    (0..hull.len())
        .map(|i| {
            hull[i][0] * hull[(i + 1) % hull.len()][1] - hull[i][1] * hull[(i + 1) % hull.len()][0]
        })
        .sum::<f64>()
        .abs()
        / 2.
}
fn inside_hull(hull: &[[f64; 2]], point: [f64; 2], tolerance: f64) -> bool {
    if hull.len() < 3 {
        return false;
    }
    let scale = hull
        .iter()
        .map(|p| (p[0] - point[0]).abs().max((p[1] - point[1]).abs()))
        .fold(1., f64::max);
    (0..hull.len())
        .all(|i| cross(hull[i], hull[(i + 1) % hull.len()], point) >= -tolerance * scale * scale)
}

/// Recompute all numerical eligibility and diagnostics from the source data.
/// The serialized condition number is never trusted to establish rank/quality.
fn complete_report(
    asset: &BallCenterCalibrationV1,
) -> Result<BallCenterCalibrationFitReport, BallCenterCalibrationError> {
    let policy = &asset.solver_policy;
    let (fitted, condition) = fit_homography(&asset.correspondences, policy)?;
    // Homogeneous scale is arbitrary. Compare canonical matrices so externally
    // supplied assets cannot claim this solver's report for a different map.
    let close = |a: f64, b: f64| {
        a.is_finite()
            && b.is_finite()
            && (a - b).abs() <= policy.numerical_tolerance * (1. + a.abs().max(b.abs()))
    };
    for (a, b) in asset
        .image_to_table_mm
        .iter()
        .flatten()
        .zip(fitted.iter().flatten())
    {
        if !close(a / asset.image_to_table_mm[2][2], b / fitted[2][2]) {
            return Err(BallCenterCalibrationError::InvalidFitReport);
        }
    }
    let mut report = residuals(asset)?;
    report.normalized_design_condition_number = condition;
    let source: Vec<_> = asset
        .correspondences
        .iter()
        .map(|p| [p.image_point_px.x_px, p.image_point_px.y_px])
        .collect();
    let target: Vec<_> = asset
        .correspondences
        .iter()
        .map(|p| [p.center_table_mm.x_mm, p.center_table_mm.y_mm])
        .collect();
    report.source_reference_hull_px = convex_hull(&source);
    report.table_reference_hull_mm = convex_hull(&target);
    report.source_hull_area_fraction = hull_area(&report.source_reference_hull_px)
        / (f64::from(asset.source_width_px) * f64::from(asset.source_height_px));
    report.table_hull_area_fraction =
        hull_area(&report.table_reference_hull_mm) / (asset.table_width_mm * asset.table_height_mm);
    let image_corners = [
        [0., 0.],
        [f64::from(asset.source_width_px - 1), 0.],
        [
            f64::from(asset.source_width_px - 1),
            f64::from(asset.source_height_px - 1),
        ],
        [0., f64::from(asset.source_height_px - 1)],
    ];
    let w = asset.table_width_mm / 2.;
    let h = asset.table_height_mm / 2.;
    let table_corners = [[-w, -h], [w, -h], [w, h], [-w, h]];
    report.source_image_corners_extrapolated = image_corners.map(|p| {
        !inside_hull(
            &report.source_reference_hull_px,
            p,
            policy.numerical_tolerance,
        )
    });
    report.physical_table_corners_extrapolated = table_corners.map(|p| {
        !inside_hull(
            &report.table_reference_hull_mm,
            p,
            policy.numerical_tolerance,
        )
    });
    let mut squares = 0.;
    let mut maximum: f64 = 0.;
    for (held_out_index, held_out) in asset.correspondences.iter().enumerate() {
        let training: Vec<_> = asset
            .correspondences
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != held_out_index)
            .map(|(_, p)| p.clone())
            .collect();
        let (matrix, fold_condition) = fit_homography(&training, policy)?;
        validate_matrix(
            matrix,
            asset.source_width_px,
            asset.source_height_px,
            policy.numerical_tolerance,
        )?;
        let predicted =
            project_matrix(matrix, held_out.image_point_px, policy.numerical_tolerance)?;
        let error = [
            predicted.x_mm - held_out.center_table_mm.x_mm,
            predicted.y_mm - held_out.center_table_mm.y_mm,
        ];
        let distance_mm = error[0].hypot(error[1]);
        let train_source: Vec<_> = training
            .iter()
            .map(|p| [p.image_point_px.x_px, p.image_point_px.y_px])
            .collect();
        report.leave_one_out.push(BallCenterCalibrationHoldout {
            held_out_index,
            training_correspondences: training.len(),
            normalized_design_condition_number: fold_condition,
            projected_table_mm: [predicted.x_mm, predicted.y_mm],
            error_mm: error,
            distance_mm,
            source_point_extrapolated: !inside_hull(
                &convex_hull(&train_source),
                source[held_out_index],
                policy.numerical_tolerance,
            ),
        });
        squares += distance_mm * distance_mm;
        maximum = maximum.max(distance_mm);
    }
    report.leave_one_out_rms_mm = (squares / asset.correspondences.len() as f64).sqrt();
    report.leave_one_out_maximum_residual_mm = maximum;
    if !report.leave_one_out_rms_mm.is_finite()
        || !report.source_hull_area_fraction.is_finite()
        || !report.table_hull_area_fraction.is_finite()
    {
        return Err(BallCenterCalibrationError::NonFiniteProjection);
    }
    Ok(report)
}

#[cfg(test)]
mod eligibility_tests {
    use super::*;
    #[test]
    fn full_rank_does_not_imply_every_withheld_fit_has_rank() {
        let points = [
            [100., 100.],
            [600., 100.],
            [1100., 100.],
            [1600., 100.],
            [100., 600.],
            [1600., 600.],
        ];
        let observations: Vec<_> = points
            .iter()
            .map(|&[x, y]| BallCenterCorrespondence {
                image_point_px: ImagePointPx { x_px: x, y_px: y },
                center_table_mm: TablePointMm {
                    x_mm: x / 2. - 500.,
                    y_mm: y / 2. - 300.,
                },
                provenance: "synthetic known planar reference".into(),
            })
            .collect();
        let policy = BallCenterCalibrationSolverPolicy::default();
        assert!(fit_homography(&observations, &policy).is_ok());
        let training: Vec<_> = observations
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != 4)
            .map(|(_, p)| p.clone())
            .collect();
        assert!(matches!(
            fit_homography(&training, &policy),
            Err(BallCenterCalibrationError::DegenerateGeometry)
        ));
    }
}
