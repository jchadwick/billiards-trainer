//! Bounded table-presence checks against an operator-confirmed calibration frame.
//!
//! The reference retains only a bounded set of RGB8 samples from the playing
//! surface and rail profiles. It never retains or needs to copy a full frame.
//! Callers must treat every result except [`TableVisibilityStatus::Visible`]
//! as ineligible for guidance.

use std::fmt;

use serde::{Deserialize, Serialize};

/// Tunable sampling and classification settings for table visibility checks.
///
/// Defaults are provisional and should be qualified against the target camera,
/// lighting, and table. Every threshold used to accept/reject image evidence
/// lives here so the live configuration layer can own production policy.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TableVisibilityConfig {
    /// Largest accepted frame edge, bounding validation and pixel addressing.
    pub maximum_frame_dimension: u32,
    /// Hard cap on the total stored RGB feature samples in a reference.
    pub maximum_feature_samples: usize,
    /// Number of cloth samples along each playing-surface axis.
    pub cloth_grid_width: u8,
    pub cloth_grid_height: u8,
    /// Fraction inset from each table-region edge before sampling cloth.
    pub cloth_inset_fraction: f32,
    /// Number of profiles sampled along each of the four calibrated edges.
    pub rail_profiles_per_side: u8,
    /// Number of RGB samples in each profile crossing the rail boundary.
    pub rail_profile_depth_samples: u8,
    /// Profile half-width as a fraction of the shorter table dimension.
    pub rail_profile_half_width_fraction: f32,
    /// Lowest accepted median cloth luminance in a confirmed reference.
    pub minimum_reference_cloth_luminance: u8,
    /// Highest accepted median cloth luminance in a confirmed reference.
    pub maximum_reference_cloth_luminance: u8,
    /// Highest accepted median absolute luminance deviation within cloth.
    pub maximum_reference_cloth_luminance_mad: u8,
    /// Minimum luminance range within a rail-normal profile to count as useful.
    pub minimum_rail_profile_contrast: u8,
    /// Minimum fraction of profiles with enough contrast at calibration time.
    pub minimum_reference_rail_contrast_fraction: f32,
    /// Minimum fraction of profiles that must remain legible in a live frame.
    pub minimum_live_rail_contrast_fraction: f32,
    /// Largest global luminance shift accepted as ordinary lighting variation.
    pub maximum_global_luminance_shift: u8,
    /// Largest per-channel global shift accepted as ordinary color variation.
    pub maximum_global_channel_shift: u8,
    /// Largest channel deviation from the global luminance shift. This keeps
    /// chromatic occluders from being normalized as ordinary exposure changes.
    pub maximum_global_chroma_shift: u8,
    /// Per-channel sample difference that marks a cloth sample as changed.
    pub maximum_cloth_channel_difference: u8,
    /// Per-channel sample difference that marks a rail sample as changed.
    pub maximum_rail_channel_difference: u8,
    /// Largest changed cloth fraction allowed while the table remains visible.
    pub maximum_changed_cloth_fraction: f32,
    /// Largest changed rail fraction allowed while the table remains visible.
    pub maximum_changed_rail_fraction: f32,
    /// Number of consecutive visible samples required before the motion/layout
    /// scheduler may begin reacquiring a map after table loss.
    pub reacquisition_consensus_count: usize,
}

impl Default for TableVisibilityConfig {
    fn default() -> Self {
        Self {
            maximum_frame_dimension: 8192,
            maximum_feature_samples: 1024,
            cloth_grid_width: 12,
            cloth_grid_height: 8,
            cloth_inset_fraction: 0.14,
            rail_profiles_per_side: 8,
            rail_profile_depth_samples: 7,
            rail_profile_half_width_fraction: 0.04,
            minimum_reference_cloth_luminance: 35,
            maximum_reference_cloth_luminance: 245,
            maximum_reference_cloth_luminance_mad: 22,
            minimum_rail_profile_contrast: 18,
            minimum_reference_rail_contrast_fraction: 0.35,
            minimum_live_rail_contrast_fraction: 0.25,
            maximum_global_luminance_shift: 52,
            maximum_global_channel_shift: 64,
            maximum_global_chroma_shift: 28,
            maximum_cloth_channel_difference: 34,
            maximum_rail_channel_difference: 38,
            maximum_changed_cloth_fraction: 0.20,
            maximum_changed_rail_fraction: 0.24,
            reacquisition_consensus_count: 3,
        }
    }
}

impl TableVisibilityConfig {
    /// Validate tunables and ensure the resulting sample layout fits its cap.
    pub fn validate(&self) -> Result<(), TableVisibilityError> {
        if !(32..=16_384).contains(&self.maximum_frame_dimension) {
            return Err(TableVisibilityError::InvalidConfig(
                "maximum_frame_dimension must be between 32 and 16384".to_owned(),
            ));
        }
        if !(128..=16_384).contains(&self.maximum_feature_samples) {
            return Err(TableVisibilityError::InvalidConfig(
                "maximum_feature_samples must be between 128 and 16384".to_owned(),
            ));
        }
        if !(4..=64).contains(&self.cloth_grid_width) || !(4..=64).contains(&self.cloth_grid_height)
        {
            return Err(TableVisibilityError::InvalidConfig(
                "cloth grid dimensions must be between 4 and 64".to_owned(),
            ));
        }
        if !self.cloth_inset_fraction.is_finite()
            || !(0.0..0.45).contains(&self.cloth_inset_fraction)
        {
            return Err(TableVisibilityError::InvalidConfig(
                "cloth_inset_fraction must be finite and in [0, 0.45)".to_owned(),
            ));
        }
        if !(2..=64).contains(&self.rail_profiles_per_side)
            || !(3..=31).contains(&self.rail_profile_depth_samples)
        {
            return Err(TableVisibilityError::InvalidConfig(
                "rail profile counts are outside their supported bounds".to_owned(),
            ));
        }
        if !self.rail_profile_half_width_fraction.is_finite()
            || !(0.0..=0.20).contains(&self.rail_profile_half_width_fraction)
            || self.rail_profile_half_width_fraction == 0.0
        {
            return Err(TableVisibilityError::InvalidConfig(
                "rail_profile_half_width_fraction must be finite and in (0, 0.20]".to_owned(),
            ));
        }
        if self.minimum_reference_cloth_luminance >= self.maximum_reference_cloth_luminance {
            return Err(TableVisibilityError::InvalidConfig(
                "reference cloth luminance bounds must be increasing".to_owned(),
            ));
        }
        if self.maximum_reference_cloth_luminance_mad == 0
            || self.minimum_rail_profile_contrast == 0
            || self.maximum_global_luminance_shift == 0
            || self.maximum_global_channel_shift == 0
            || self.maximum_global_chroma_shift == 0
            || self.maximum_cloth_channel_difference == 0
            || self.maximum_rail_channel_difference == 0
        {
            return Err(TableVisibilityError::InvalidConfig(
                "difference and contrast thresholds must be positive".to_owned(),
            ));
        }
        for (name, value) in [
            (
                "minimum_reference_rail_contrast_fraction",
                self.minimum_reference_rail_contrast_fraction,
            ),
            (
                "minimum_live_rail_contrast_fraction",
                self.minimum_live_rail_contrast_fraction,
            ),
            (
                "maximum_changed_cloth_fraction",
                self.maximum_changed_cloth_fraction,
            ),
            (
                "maximum_changed_rail_fraction",
                self.maximum_changed_rail_fraction,
            ),
        ] {
            if !value.is_finite() || !(0.0..=1.0).contains(&value) || value == 0.0 {
                return Err(TableVisibilityError::InvalidConfig(format!(
                    "{name} must be finite and in (0, 1]"
                )));
            }
        }
        if self.minimum_reference_rail_contrast_fraction < self.minimum_live_rail_contrast_fraction
        {
            return Err(TableVisibilityError::InvalidConfig(
                "live rail contrast requirement cannot exceed reference requirement".to_owned(),
            ));
        }
        if !(1..=16).contains(&self.reacquisition_consensus_count) {
            return Err(TableVisibilityError::InvalidConfig(
                "reacquisition_consensus_count must be between 1 and 16".to_owned(),
            ));
        }

        let sample_count = self.feature_sample_count()?;
        if sample_count > self.maximum_feature_samples {
            return Err(TableVisibilityError::InvalidConfig(format!(
                "sample layout requires {sample_count} features, above the configured cap of {}",
                self.maximum_feature_samples
            )));
        }
        Ok(())
    }

    fn feature_sample_count(&self) -> Result<usize, TableVisibilityError> {
        let cloth = usize::from(self.cloth_grid_width)
            .checked_mul(usize::from(self.cloth_grid_height))
            .ok_or_else(|| {
                TableVisibilityError::InvalidConfig("cloth sample count overflow".to_owned())
            })?;
        let rail = 4_usize
            .checked_mul(usize::from(self.rail_profiles_per_side))
            .and_then(|value| value.checked_mul(usize::from(self.rail_profile_depth_samples)))
            .ok_or_else(|| {
                TableVisibilityError::InvalidConfig("rail sample count overflow".to_owned())
            })?;
        cloth.checked_add(rail).ok_or_else(|| {
            TableVisibilityError::InvalidConfig("feature sample count overflow".to_owned())
        })
    }

    fn feature_layout(&self) -> FeatureLayout {
        FeatureLayout {
            cloth_grid_width: self.cloth_grid_width,
            cloth_grid_height: self.cloth_grid_height,
            cloth_inset_fraction: self.cloth_inset_fraction,
            rail_profiles_per_side: self.rail_profiles_per_side,
            rail_profile_depth_samples: self.rail_profile_depth_samples,
            rail_profile_half_width_fraction: self.rail_profile_half_width_fraction,
        }
    }
}

/// Borrowed RGB8 frame; construction is cheap and does not copy image data.
#[derive(Debug, Clone, Copy)]
pub struct Rgb8Frame<'a> {
    pub width: u32,
    pub height: u32,
    pub pixels: &'a [u8],
}

/// Normalized image-plane coordinate in the inclusive range 0 through 1.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NormalizedPoint {
    pub x: f32,
    pub y: f32,
}

/// Calibrated table playing-surface boundary in top-left, top-right,
/// bottom-right, bottom-left order. The quadrilateral must be convex.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TableRegion {
    pub corners: [NormalizedPoint; 4],
}

impl TableRegion {
    pub fn new(corners: [NormalizedPoint; 4]) -> Self {
        Self { corners }
    }

    fn is_valid(self) -> bool {
        if self.corners.iter().any(|point| {
            !point.x.is_finite()
                || !point.y.is_finite()
                || !(0.0..=1.0).contains(&point.x)
                || !(0.0..=1.0).contains(&point.y)
        }) {
            return false;
        }
        let mut sign = 0.0_f32;
        let mut area_twice = 0.0_f32;
        for index in 0..4 {
            let current = self.corners[index];
            let next = self.corners[(index + 1) % 4];
            let following = self.corners[(index + 2) % 4];
            area_twice += current.x * next.y - next.x * current.y;
            let cross = (next.x - current.x) * (following.y - next.y)
                - (next.y - current.y) * (following.x - next.x);
            if cross.abs() <= f32::EPSILON {
                return false;
            }
            if sign == 0.0 {
                sign = cross.signum();
            } else if cross.signum() != sign {
                return false;
            }
        }
        area_twice.abs() > f32::EPSILON
    }
}

/// RGB value retained for one bounded reference feature location.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RgbFeature {
    pub red: u8,
    pub green: u8,
    pub blue: u8,
}

impl RgbFeature {
    fn channel(self, index: usize) -> i16 {
        match index {
            0 => i16::from(self.red),
            1 => i16::from(self.green),
            _ => i16::from(self.blue),
        }
    }
}

/// Sampling parameters embedded in a reference so incompatible feature
/// layouts fail closed if runtime configuration changes after confirmation.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FeatureLayout {
    pub cloth_grid_width: u8,
    pub cloth_grid_height: u8,
    pub cloth_inset_fraction: f32,
    pub rail_profiles_per_side: u8,
    pub rail_profile_depth_samples: u8,
    pub rail_profile_half_width_fraction: f32,
}

/// Bounded calibration evidence. This stores sampled features, never a frame.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TableVisibilityReference {
    pub schema_version: u8,
    pub source_width: u32,
    pub source_height: u32,
    pub table_region: TableRegion,
    pub feature_layout: FeatureLayout,
    pub cloth_features: Vec<RgbFeature>,
    /// Profiles are flattened in side-major, along-edge, depth order.
    pub rail_features: Vec<RgbFeature>,
}

/// Why a calibration frame cannot safely become a table-presence reference.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReferenceIssue {
    InvalidTableRegion,
    ClothTooDark,
    ClothTooBright,
    ClothColorInconsistent,
    InsufficientRailContrast,
    RailProfileOutsideFrame,
}

/// Quality decision and bounded diagnostics from reference derivation.
#[derive(Debug, Clone, PartialEq)]
pub struct ReferenceValidity {
    pub valid: bool,
    /// Unique issue codes; the number of possible entries is fixed by the enum.
    pub reasons: Vec<ReferenceIssue>,
    pub median_cloth_luminance: Option<u8>,
    pub cloth_luminance_mad: Option<u8>,
    pub rail_contrast_profile_fraction: Option<f32>,
}

/// Result of deriving reference samples from a frame the operator confirmed.
#[derive(Debug, Clone, PartialEq)]
pub struct ReferenceBuildResult {
    pub reference: Option<TableVisibilityReference>,
    pub validity: ReferenceValidity,
}

/// Explicit assertion that the operator confirmed this calibration frame and
/// table boundary. Constructing this value is the caller's confirmation step.
#[derive(Debug, Clone, Copy)]
pub struct OperatorConfirmedFrame<'a> {
    pub frame: Rgb8Frame<'a>,
    pub table_region: TableRegion,
}

impl<'a> OperatorConfirmedFrame<'a> {
    pub fn new(frame: Rgb8Frame<'a>, table_region: TableRegion) -> Self {
        Self {
            frame,
            table_region,
        }
    }
}

/// Whether current pixels establish that the calibrated table is present.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TableVisibilityStatus {
    /// Evidence agrees with the confirmed table reference.
    Visible,
    /// An image was present, but it conflicts with or cannot satisfy reference evidence.
    NotVisible,
    /// Evidence is unavailable, such as a missing frame or reference.
    Unknown,
}

/// Stable reason for the visibility result, suitable for logs and scheduler policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TableVisibilityReason {
    ReferenceMatch,
    MissingReference,
    MissingFrame,
    InvalidReference,
    SamplingLayoutChanged,
    ResolutionChanged,
    ExtremeBrightnessShift,
    ExtremeColorShift,
    LostRailContrast,
    ClothMismatch,
    RailMismatch,
}

/// Visibility decision with bounded mismatch metrics for diagnostics.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TableVisibilityAssessment {
    pub status: TableVisibilityStatus,
    pub reason: TableVisibilityReason,
    pub global_luminance_shift: Option<i16>,
    pub cloth_changed_fraction: Option<f32>,
    pub rail_changed_fraction: Option<f32>,
    pub live_rail_contrast_profile_fraction: Option<f32>,
}

impl TableVisibilityAssessment {
    fn unavailable(reason: TableVisibilityReason) -> Self {
        Self {
            status: TableVisibilityStatus::Unknown,
            reason,
            global_luminance_shift: None,
            cloth_changed_fraction: None,
            rail_changed_fraction: None,
            live_rail_contrast_profile_fraction: None,
        }
    }

    fn rejected(reason: TableVisibilityReason) -> Self {
        Self {
            status: TableVisibilityStatus::NotVisible,
            ..Self::unavailable(reason)
        }
    }
}

/// Invalid settings or malformed frame data encountered by this pure module.
#[derive(Debug, Clone, PartialEq, Eq)]
// Keep the `Invalid` prefix to clearly distinguish malformed input from a
// valid assessment that reports the table as not visible.
#[allow(clippy::enum_variant_names)]
pub enum TableVisibilityError {
    InvalidConfig(String),
    InvalidDimensions { width: u32, height: u32 },
    InvalidFrameLength { expected: usize, actual: usize },
}

impl fmt::Display for TableVisibilityError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(message) => {
                write!(
                    formatter,
                    "invalid table visibility configuration: {message}"
                )
            }
            Self::InvalidDimensions { width, height } => {
                write!(formatter, "invalid RGB8 frame dimensions {width}x{height}")
            }
            Self::InvalidFrameLength { expected, actual } => write!(
                formatter,
                "RGB8 frame has {actual} bytes; expected {expected}"
            ),
        }
    }
}

impl std::error::Error for TableVisibilityError {}

/// Derive a bounded sampled reference after explicit operator confirmation.
///
/// Poor but structurally valid evidence returns `reference: None` with reasons;
/// invalid dimensions, byte lengths, and configuration return an error. The
/// caller should persist a reference only when `validity.valid` is true.
pub fn derive_reference_from_confirmed_frame(
    confirmed: OperatorConfirmedFrame<'_>,
    config: &TableVisibilityConfig,
) -> Result<ReferenceBuildResult, TableVisibilityError> {
    config.validate()?;
    validate_frame(confirmed.frame, config)?;

    let mut reasons = Vec::new();
    if !confirmed.table_region.is_valid() {
        reasons.push(ReferenceIssue::InvalidTableRegion);
        return Ok(ReferenceBuildResult {
            reference: None,
            validity: ReferenceValidity {
                valid: false,
                reasons,
                median_cloth_luminance: None,
                cloth_luminance_mad: None,
                rail_contrast_profile_fraction: None,
            },
        });
    }

    let layout = config.feature_layout();
    let cloth_features = sample_cloth(confirmed.frame, confirmed.table_region, layout)?;
    let rail_features = match sample_rails(confirmed.frame, confirmed.table_region, layout) {
        Ok(samples) => samples,
        Err(SamplingError::OutsideFrame) => {
            reasons.push(ReferenceIssue::RailProfileOutsideFrame);
            return Ok(ReferenceBuildResult {
                reference: None,
                validity: ReferenceValidity {
                    valid: false,
                    reasons,
                    median_cloth_luminance: None,
                    cloth_luminance_mad: None,
                    rail_contrast_profile_fraction: None,
                },
            });
        }
        Err(SamplingError::Frame(error)) => return Err(error),
    };

    let mut cloth_luminance: Vec<u8> = cloth_features.iter().copied().map(luminance).collect();
    let median_cloth_luminance = median_u8(&mut cloth_luminance);
    let mut deviations: Vec<u8> = cloth_luminance
        .iter()
        .map(|value| value.abs_diff(median_cloth_luminance))
        .collect();
    let cloth_luminance_mad = median_u8(&mut deviations);

    if median_cloth_luminance < config.minimum_reference_cloth_luminance {
        reasons.push(ReferenceIssue::ClothTooDark);
    }
    if median_cloth_luminance > config.maximum_reference_cloth_luminance {
        reasons.push(ReferenceIssue::ClothTooBright);
    }
    if cloth_luminance_mad > config.maximum_reference_cloth_luminance_mad {
        reasons.push(ReferenceIssue::ClothColorInconsistent);
    }

    let rail_contrast_profile_fraction = rail_contrast_fraction(&rail_features, layout, config);
    if rail_contrast_profile_fraction < config.minimum_reference_rail_contrast_fraction {
        reasons.push(ReferenceIssue::InsufficientRailContrast);
    }

    let valid = reasons.is_empty();
    let reference = valid.then_some(TableVisibilityReference {
        schema_version: 1,
        source_width: confirmed.frame.width,
        source_height: confirmed.frame.height,
        table_region: confirmed.table_region,
        feature_layout: layout,
        cloth_features,
        rail_features,
    });
    Ok(ReferenceBuildResult {
        reference,
        validity: ReferenceValidity {
            valid,
            reasons,
            median_cloth_luminance: Some(median_cloth_luminance),
            cloth_luminance_mad: Some(cloth_luminance_mad),
            rail_contrast_profile_fraction: Some(rail_contrast_profile_fraction),
        },
    })
}

/// Compare a current frame with an optional reference. Only `Visible` is a
/// positive result; missing evidence, incompatible data, and detected changes
/// fail closed for downstream guidance.
pub fn assess_table_visibility(
    reference: Option<&TableVisibilityReference>,
    frame: Option<Rgb8Frame<'_>>,
    config: &TableVisibilityConfig,
) -> Result<TableVisibilityAssessment, TableVisibilityError> {
    config.validate()?;
    let Some(reference) = reference else {
        return Ok(TableVisibilityAssessment::unavailable(
            TableVisibilityReason::MissingReference,
        ));
    };
    let Some(frame) = frame else {
        return Ok(TableVisibilityAssessment::unavailable(
            TableVisibilityReason::MissingFrame,
        ));
    };
    validate_frame(frame, config)?;

    if !reference_is_valid(reference, config) {
        return Ok(TableVisibilityAssessment::rejected(
            TableVisibilityReason::InvalidReference,
        ));
    }
    if reference.source_width != frame.width || reference.source_height != frame.height {
        return Ok(TableVisibilityAssessment::rejected(
            TableVisibilityReason::ResolutionChanged,
        ));
    }
    if reference.feature_layout != config.feature_layout() {
        return Ok(TableVisibilityAssessment::rejected(
            TableVisibilityReason::SamplingLayoutChanged,
        ));
    }

    let live_cloth = sample_cloth(frame, reference.table_region, reference.feature_layout)?;
    let live_rails = match sample_rails(frame, reference.table_region, reference.feature_layout) {
        Ok(samples) => samples,
        Err(SamplingError::OutsideFrame) => {
            return Ok(TableVisibilityAssessment::rejected(
                TableVisibilityReason::ResolutionChanged,
            ));
        }
        Err(SamplingError::Frame(error)) => return Err(error),
    };

    let channel_shift = estimate_channel_shift(&reference.cloth_features, &live_cloth);
    let luminance_shift = luminance_from_channels(channel_shift);
    let mut assessment = TableVisibilityAssessment {
        status: TableVisibilityStatus::Visible,
        reason: TableVisibilityReason::ReferenceMatch,
        global_luminance_shift: Some(luminance_shift),
        cloth_changed_fraction: None,
        rail_changed_fraction: None,
        live_rail_contrast_profile_fraction: None,
    };

    if luminance_shift.unsigned_abs() > u16::from(config.maximum_global_luminance_shift) {
        assessment.status = TableVisibilityStatus::NotVisible;
        assessment.reason = TableVisibilityReason::ExtremeBrightnessShift;
        return Ok(assessment);
    }
    if channel_shift
        .iter()
        .any(|shift| shift.unsigned_abs() > u16::from(config.maximum_global_channel_shift))
        || channel_shift.iter().any(|shift| {
            (*shift - luminance_shift).unsigned_abs()
                > u16::from(config.maximum_global_chroma_shift)
        })
    {
        assessment.status = TableVisibilityStatus::NotVisible;
        assessment.reason = TableVisibilityReason::ExtremeColorShift;
        return Ok(assessment);
    }

    let cloth_changed = changed_fraction(
        &reference.cloth_features,
        &live_cloth,
        channel_shift,
        config.maximum_cloth_channel_difference,
    );
    let rail_changed = changed_fraction(
        &reference.rail_features,
        &live_rails,
        channel_shift,
        config.maximum_rail_channel_difference,
    );
    let live_rail_contrast = rail_contrast_fraction(&live_rails, reference.feature_layout, config);
    assessment.cloth_changed_fraction = Some(cloth_changed);
    assessment.rail_changed_fraction = Some(rail_changed);
    assessment.live_rail_contrast_profile_fraction = Some(live_rail_contrast);

    if live_rail_contrast < config.minimum_live_rail_contrast_fraction {
        assessment.status = TableVisibilityStatus::NotVisible;
        assessment.reason = TableVisibilityReason::LostRailContrast;
    } else if cloth_changed > config.maximum_changed_cloth_fraction {
        assessment.status = TableVisibilityStatus::NotVisible;
        assessment.reason = TableVisibilityReason::ClothMismatch;
    } else if rail_changed > config.maximum_changed_rail_fraction {
        assessment.status = TableVisibilityStatus::NotVisible;
        assessment.reason = TableVisibilityReason::RailMismatch;
    }
    Ok(assessment)
}

fn reference_is_valid(
    reference: &TableVisibilityReference,
    config: &TableVisibilityConfig,
) -> bool {
    if reference.schema_version != 1
        || reference.source_width < 2
        || reference.source_height < 2
        || reference.source_width > config.maximum_frame_dimension
        || reference.source_height > config.maximum_frame_dimension
        || !reference.table_region.is_valid()
        || reference.feature_layout != config.feature_layout()
    {
        return false;
    }
    let expected_cloth =
        usize::from(config.cloth_grid_width).checked_mul(usize::from(config.cloth_grid_height));
    let expected_rails = 4_usize
        .checked_mul(usize::from(config.rail_profiles_per_side))
        .and_then(|value| value.checked_mul(usize::from(config.rail_profile_depth_samples)));
    expected_cloth == Some(reference.cloth_features.len())
        && expected_rails == Some(reference.rail_features.len())
        && reference.cloth_features.len() + reference.rail_features.len()
            <= config.maximum_feature_samples
}

/// Return whether a stored reference is structurally compatible with the
/// active sampling policy. Call this after deserialization and before the
/// reference is made available to a capture runtime.
pub fn validate_reference(
    reference: &TableVisibilityReference,
    config: &TableVisibilityConfig,
) -> bool {
    config.validate().is_ok() && reference_is_valid(reference, config)
}

/// Validate a persisted feature reference without assuming the runtime's
/// matching thresholds. The file reader supplies hard memory and dimension
/// bounds; later, the runtime also requires an exact sampling-policy match.
pub fn validate_reference_structure(
    reference: &TableVisibilityReference,
    maximum_frame_dimension: u32,
    maximum_feature_samples: usize,
) -> bool {
    let config = TableVisibilityConfig {
        maximum_frame_dimension,
        maximum_feature_samples,
        cloth_grid_width: reference.feature_layout.cloth_grid_width,
        cloth_grid_height: reference.feature_layout.cloth_grid_height,
        cloth_inset_fraction: reference.feature_layout.cloth_inset_fraction,
        rail_profiles_per_side: reference.feature_layout.rail_profiles_per_side,
        rail_profile_depth_samples: reference.feature_layout.rail_profile_depth_samples,
        rail_profile_half_width_fraction: reference.feature_layout.rail_profile_half_width_fraction,
        ..TableVisibilityConfig::default()
    };
    config.validate().is_ok() && reference_is_valid(reference, &config)
}

fn validate_frame(
    frame: Rgb8Frame<'_>,
    config: &TableVisibilityConfig,
) -> Result<(), TableVisibilityError> {
    if frame.width < 2
        || frame.height < 2
        || frame.width > config.maximum_frame_dimension
        || frame.height > config.maximum_frame_dimension
    {
        return Err(TableVisibilityError::InvalidDimensions {
            width: frame.width,
            height: frame.height,
        });
    }
    let expected = usize::try_from(frame.width)
        .ok()
        .and_then(|width| {
            usize::try_from(frame.height)
                .ok()
                .and_then(|height| width.checked_mul(height))
        })
        .and_then(|pixels| pixels.checked_mul(3))
        .ok_or(TableVisibilityError::InvalidDimensions {
            width: frame.width,
            height: frame.height,
        })?;
    if frame.pixels.len() != expected {
        return Err(TableVisibilityError::InvalidFrameLength {
            expected,
            actual: frame.pixels.len(),
        });
    }
    Ok(())
}

#[derive(Debug)]
enum SamplingError {
    OutsideFrame,
    Frame(TableVisibilityError),
}

impl From<TableVisibilityError> for SamplingError {
    fn from(value: TableVisibilityError) -> Self {
        Self::Frame(value)
    }
}

fn sample_cloth(
    frame: Rgb8Frame<'_>,
    region: TableRegion,
    layout: FeatureLayout,
) -> Result<Vec<RgbFeature>, TableVisibilityError> {
    let capacity = usize::from(layout.cloth_grid_width) * usize::from(layout.cloth_grid_height);
    let mut features = Vec::with_capacity(capacity);
    let usable = 1.0 - 2.0 * layout.cloth_inset_fraction;
    for row in 0..layout.cloth_grid_height {
        let v = layout.cloth_inset_fraction
            + (f32::from(row) + 0.5) * usable / f32::from(layout.cloth_grid_height);
        for column in 0..layout.cloth_grid_width {
            let u = layout.cloth_inset_fraction
                + (f32::from(column) + 0.5) * usable / f32::from(layout.cloth_grid_width);
            let point = bilinear_point(region, u, v);
            features.push(sample_pixel(frame, point)?);
        }
    }
    Ok(features)
}

fn sample_rails(
    frame: Rgb8Frame<'_>,
    region: TableRegion,
    layout: FeatureLayout,
) -> Result<Vec<RgbFeature>, SamplingError> {
    let capacity = 4_usize
        * usize::from(layout.rail_profiles_per_side)
        * usize::from(layout.rail_profile_depth_samples);
    let mut features = Vec::with_capacity(capacity);
    let width = frame.width as f32;
    let height = frame.height as f32;
    let corners = region
        .corners
        .map(|point| (point.x * (width - 1.0), point.y * (height - 1.0)));
    let center = (
        corners.iter().map(|point| point.0).sum::<f32>() / 4.0,
        corners.iter().map(|point| point.1).sum::<f32>() / 4.0,
    );
    let horizontal_size =
        (distance(corners[0], corners[1]) + distance(corners[3], corners[2])) / 2.0;
    let vertical_size = (distance(corners[0], corners[3]) + distance(corners[1], corners[2])) / 2.0;
    let half_width = horizontal_size.min(vertical_size) * layout.rail_profile_half_width_fraction;

    for side in 0..4 {
        let start = corners[side];
        let end = corners[(side + 1) % 4];
        let edge = (end.0 - start.0, end.1 - start.1);
        let edge_length = edge.0.hypot(edge.1);
        if !edge_length.is_finite() || edge_length <= f32::EPSILON {
            return Err(SamplingError::OutsideFrame);
        }
        let mut normal = (-edge.1 / edge_length, edge.0 / edge_length);
        let midpoint = ((start.0 + end.0) / 2.0, (start.1 + end.1) / 2.0);
        if normal.0 * (center.0 - midpoint.0) + normal.1 * (center.1 - midpoint.1) < 0.0 {
            normal = (-normal.0, -normal.1);
        }
        for along_index in 0..layout.rail_profiles_per_side {
            let along = (f32::from(along_index) + 0.5) / f32::from(layout.rail_profiles_per_side);
            let edge_point = (start.0 + edge.0 * along, start.1 + edge.1 * along);
            for depth_index in 0..layout.rail_profile_depth_samples {
                let depth_fraction =
                    f32::from(depth_index) / f32::from(layout.rail_profile_depth_samples - 1);
                let offset = (depth_fraction * 2.0 - 1.0) * half_width;
                let point = (
                    edge_point.0 + normal.0 * offset,
                    edge_point.1 + normal.1 * offset,
                );
                if !point_is_inside_frame(frame, point.0, point.1) {
                    return Err(SamplingError::OutsideFrame);
                }
                features
                    .push(sample_pixel_xy(frame, point.0, point.1).map_err(SamplingError::Frame)?);
            }
        }
    }
    Ok(features)
}

fn sample_pixel(
    frame: Rgb8Frame<'_>,
    point: (f32, f32),
) -> Result<RgbFeature, TableVisibilityError> {
    sample_pixel_xy(
        frame,
        point.0 * (frame.width as f32 - 1.0),
        point.1 * (frame.height as f32 - 1.0),
    )
}

fn sample_pixel_xy(
    frame: Rgb8Frame<'_>,
    x: f32,
    y: f32,
) -> Result<RgbFeature, TableVisibilityError> {
    if !point_is_inside_frame(frame, x, y) {
        return Err(TableVisibilityError::InvalidDimensions {
            width: frame.width,
            height: frame.height,
        });
    }
    let x = x.round() as usize;
    let y = y.round() as usize;
    let index = (y * frame.width as usize + x) * 3;
    let pixel =
        frame
            .pixels
            .get(index..index + 3)
            .ok_or(TableVisibilityError::InvalidFrameLength {
                expected: (frame.width as usize) * (frame.height as usize) * 3,
                actual: frame.pixels.len(),
            })?;
    Ok(RgbFeature {
        red: pixel[0],
        green: pixel[1],
        blue: pixel[2],
    })
}

fn point_is_inside_frame(frame: Rgb8Frame<'_>, x: f32, y: f32) -> bool {
    x.is_finite()
        && y.is_finite()
        && x >= 0.0
        && y >= 0.0
        && x <= frame.width as f32 - 1.0
        && y <= frame.height as f32 - 1.0
}

fn bilinear_point(region: TableRegion, u: f32, v: f32) -> (f32, f32) {
    let [top_left, top_right, bottom_right, bottom_left] = region.corners;
    let top = (1.0 - u) * (1.0 - v);
    let right = u * (1.0 - v);
    let bottom = u * v;
    let left = (1.0 - u) * v;
    (
        top * top_left.x + right * top_right.x + bottom * bottom_right.x + left * bottom_left.x,
        top * top_left.y + right * top_right.y + bottom * bottom_right.y + left * bottom_left.y,
    )
}

fn distance(first: (f32, f32), second: (f32, f32)) -> f32 {
    (second.0 - first.0).hypot(second.1 - first.1)
}

fn luminance(feature: RgbFeature) -> u8 {
    // Integer approximation of Rec. 709 luma; coefficients are a color-space
    // conversion, not a policy threshold.
    ((54_u16 * u16::from(feature.red)
        + 183_u16 * u16::from(feature.green)
        + 19_u16 * u16::from(feature.blue)
        + 128)
        / 256) as u8
}

fn median_u8(values: &mut [u8]) -> u8 {
    values.sort_unstable();
    values[values.len() / 2]
}

fn median_i16(values: &mut [i16]) -> i16 {
    values.sort_unstable();
    values[values.len() / 2]
}

fn estimate_channel_shift(reference: &[RgbFeature], live: &[RgbFeature]) -> [i16; 3] {
    let mut shifts = [0_i16; 3];
    for (channel, shift) in shifts.iter_mut().enumerate() {
        let mut differences: Vec<i16> = reference
            .iter()
            .zip(live)
            .map(|(expected, actual)| actual.channel(channel) - expected.channel(channel))
            .collect();
        *shift = median_i16(&mut differences);
    }
    shifts
}

fn luminance_from_channels(shifts: [i16; 3]) -> i16 {
    (54_i32 * i32::from(shifts[0]) + 183_i32 * i32::from(shifts[1]) + 19_i32 * i32::from(shifts[2]))
        .div_euclid(256) as i16
}

fn changed_fraction(
    reference: &[RgbFeature],
    live: &[RgbFeature],
    channel_shift: [i16; 3],
    threshold: u8,
) -> f32 {
    let changed = reference
        .iter()
        .zip(live)
        .filter(|(expected, actual)| {
            (0..3).any(|channel| {
                let adjusted = actual.channel(channel) - channel_shift[channel];
                (adjusted - expected.channel(channel)).unsigned_abs() > u16::from(threshold)
            })
        })
        .count();
    changed as f32 / reference.len() as f32
}

fn rail_contrast_fraction(
    features: &[RgbFeature],
    layout: FeatureLayout,
    config: &TableVisibilityConfig,
) -> f32 {
    let profiles = 4_usize * usize::from(layout.rail_profiles_per_side);
    let depth = usize::from(layout.rail_profile_depth_samples);
    let useful = features
        .chunks_exact(depth)
        .take(profiles)
        .filter(|profile| {
            let (minimum, maximum) = profile
                .iter()
                .copied()
                .map(luminance)
                .fold((u8::MAX, u8::MIN), |(minimum, maximum), value| {
                    (minimum.min(value), maximum.max(value))
                });
            maximum - minimum >= config.minimum_rail_profile_contrast
        })
        .count();
    useful as f32 / profiles as f32
}
