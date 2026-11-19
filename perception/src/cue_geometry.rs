//! Bounded image-based refinement for cue-shaft geometry.
//!
//! This deliberately does not classify balls or invoke a neural model. It
//! extracts RGB gradient samples once, then fits a narrow, long axis first in
//! recent proposal regions and finally across the image for reacquisition.
//! The result is useful geometry, not a guarantee that the detected line is a
//! cue: table rails, seams, and other long objects can still be confounders.

use std::time::Duration;

use crate::calibration::{CalibrationContext, ImagePointPx, ValidatedTableCalibrationV1};

const RGB_CHANNELS: usize = 3;
const ABSOLUTE_MAX_DIMENSION: u32 = 8192;
const ABSOLUTE_MAX_PIXELS: usize = 16_777_216;
const ABSOLUTE_MAX_EDGE_SAMPLES: usize = 500_000;
const ABSOLUTE_MAX_TEMPORARY_BYTES: usize = 32 * 1024 * 1024;
const ABSOLUTE_MAX_POINT_VISITS: usize = 50_000_000;
const MAX_CENTERLINE_SECTIONS: usize = 128;
const MAX_EDGE_PEAKS_PER_SECTION: usize = 16;
const CENTERLINE_HISTOGRAM_CAPACITY: usize = 2048;

/// Limits for image work, temporary storage, and acceptable cue-like fits.
///
/// The maxima are intentionally validated as a group. A caller can tune these
/// values, but cannot accidentally turn an ordinary camera frame into an
/// unbounded allocation or multiply ROI fitting work without limit.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CueGeometryConfig {
    pub max_width_px: u32,
    pub max_height_px: u32,
    pub max_pixels: usize,
    pub max_edge_samples: usize,
    pub max_proposals: usize,
    pub max_refinements: usize,
    pub max_point_visits: usize,
    pub max_temporary_bytes: usize,
    pub minimum_gradient: f64,
    pub maximum_fit_residual_px: f64,
    pub minimum_length_fraction: f64,
    /// Lines this long are likely to be table rails or image-wide structures.
    pub maximum_length_fraction_of_longest_dimension: f64,
    pub minimum_eigenvalue_ratio: f64,
    pub minimum_support_points: usize,
    pub minimum_occupied_length_fraction: f64,
    /// Cue width limits scale with the shorter image dimension so the same
    /// estimator works on the supported camera resolutions.
    pub minimum_shaft_width_fraction: f64,
    pub maximum_shaft_width_fraction: f64,
    pub minimum_shaft_width_px: f64,
    /// Pixel fallback for the maximum width when a thumbnail is too small for
    /// the dimension-scaled shaft-width bound to resolve both sides.
    pub minimum_shaft_maximum_width_px: f64,
    /// Minimum spacing between longitudinal cross-sections before a short cue
    /// is handled as one section with several profiles along its span.
    pub minimum_centerline_section_spacing_px: f64,
    /// Maximum cross-sections used to recover the shaft center from paired
    /// side edges. Shorter observed spans use wider sections, and missing
    /// sections remain allowed for partial occlusion.
    pub centerline_sections: usize,
    /// Longitudinal image profiles sampled within each cross-section. This
    /// recovers an edge even when the sparse gradient sample misses it.
    pub cross_section_profile_samples: usize,
    pub minimum_paired_section_fraction: f64,
    /// Maximum ratio between robust low and high shaft-width percentiles.
    pub maximum_shaft_taper_ratio: f64,
    pub minimum_pair_edge_support: usize,
    pub maximum_edge_peaks_per_section: usize,
    pub pair_histogram_bin_width_px: f64,
    pub pair_peak_radius_px: f64,
    pub minimum_cross_section_contrast: f64,
    /// A dark endpoint must differ by this many luminance levels to label it
    /// as the likely cue tip. Weak or ambiguous asymmetry leaves direction unset.
    pub minimum_tip_contrast: f64,
}

impl Default for CueGeometryConfig {
    fn default() -> Self {
        Self {
            // Match the supported canonical camera ceiling while keeping
            // edge samples, fitting passes, and temporary point storage
            // independently bounded below.
            max_width_px: 3840,
            max_height_px: 2160,
            max_pixels: 8_294_400,
            max_edge_samples: 80_000,
            max_proposals: 4,
            max_refinements: 2,
            // Includes point selection and every bounded robust-fit pass.
            max_point_visits: 2_500_000,
            // The global edge samples plus one reusable ROI inlier array.
            max_temporary_bytes: 3_000_000,
            minimum_gradient: 24.0,
            maximum_fit_residual_px: 2.5,
            minimum_length_fraction: 0.12,
            maximum_length_fraction_of_longest_dimension: 0.78,
            minimum_eigenvalue_ratio: 7.0,
            minimum_support_points: 18,
            minimum_occupied_length_fraction: 0.65,
            minimum_shaft_width_fraction: 0.002,
            maximum_shaft_width_fraction: 0.05,
            minimum_shaft_width_px: 0.5,
            minimum_shaft_maximum_width_px: 2.0,
            minimum_centerline_section_spacing_px: 8.0,
            centerline_sections: 16,
            cross_section_profile_samples: 3,
            minimum_paired_section_fraction: 0.5,
            maximum_shaft_taper_ratio: 4.0,
            minimum_pair_edge_support: 1,
            maximum_edge_peaks_per_section: 8,
            pair_histogram_bin_width_px: 1.0,
            pair_peak_radius_px: 0.25,
            minimum_cross_section_contrast: 10.0,
            minimum_tip_contrast: 55.0,
        }
    }
}

impl CueGeometryConfig {
    pub fn validate(self) -> Result<Self, CueGeometryError> {
        let valid_float = |value: f64| value.is_finite();
        if self.max_width_px < 3
            || self.max_height_px < 3
            || self.max_width_px > ABSOLUTE_MAX_DIMENSION
            || self.max_height_px > ABSOLUTE_MAX_DIMENSION
            || self.max_pixels < 9
            || self.max_pixels > ABSOLUTE_MAX_PIXELS
            || self.max_edge_samples < 16
            || self.max_edge_samples > ABSOLUTE_MAX_EDGE_SAMPLES
            || self.max_edge_samples > self.max_pixels
            || self.max_proposals > 16
            || !(1..=8).contains(&self.max_refinements)
            || self.max_point_visits > ABSOLUTE_MAX_POINT_VISITS
            || self.max_temporary_bytes == 0
            || self.max_temporary_bytes > ABSOLUTE_MAX_TEMPORARY_BYTES
            || !valid_float(self.minimum_gradient)
            || self.minimum_gradient <= 0.0
            || !valid_float(self.maximum_fit_residual_px)
            || self.maximum_fit_residual_px <= 0.0
            || !valid_float(self.minimum_length_fraction)
            || !valid_float(self.maximum_length_fraction_of_longest_dimension)
            || !(0.0..1.0).contains(&self.minimum_length_fraction)
            || !(self.minimum_length_fraction..=1.0)
                .contains(&self.maximum_length_fraction_of_longest_dimension)
            || !valid_float(self.minimum_eigenvalue_ratio)
            || self.minimum_eigenvalue_ratio <= 1.0
            || self.minimum_support_points < 4
            || !valid_float(self.minimum_occupied_length_fraction)
            || !(0.0..=1.0).contains(&self.minimum_occupied_length_fraction)
            || !valid_float(self.minimum_shaft_width_fraction)
            || !valid_float(self.maximum_shaft_width_fraction)
            || self.minimum_shaft_width_fraction <= 0.0
            || self.maximum_shaft_width_fraction > 0.05
            || self.minimum_shaft_width_fraction > self.maximum_shaft_width_fraction
            || !valid_float(self.minimum_shaft_width_px)
            || self.minimum_shaft_width_px <= 0.0
            || !valid_float(self.minimum_shaft_maximum_width_px)
            || self.minimum_shaft_maximum_width_px < self.minimum_shaft_width_px
            || !valid_float(self.minimum_centerline_section_spacing_px)
            || self.minimum_centerline_section_spacing_px <= 0.0
            || self.minimum_centerline_section_spacing_px > f64::from(ABSOLUTE_MAX_DIMENSION)
            || !(8..=MAX_CENTERLINE_SECTIONS).contains(&self.centerline_sections)
            || !(1..=8).contains(&self.cross_section_profile_samples)
            || !valid_float(self.minimum_paired_section_fraction)
            || !(0.0..=1.0).contains(&self.minimum_paired_section_fraction)
            || self.minimum_paired_section_fraction == 0.0
            || !valid_float(self.maximum_shaft_taper_ratio)
            || self.maximum_shaft_taper_ratio < 1.0
            || self.maximum_shaft_taper_ratio > 16.0
            || self.minimum_pair_edge_support == 0
            || self.minimum_pair_edge_support > self.max_edge_samples
            || !(2..=MAX_EDGE_PEAKS_PER_SECTION).contains(&self.maximum_edge_peaks_per_section)
            || !valid_float(self.pair_histogram_bin_width_px)
            || !(0.5..=2.0).contains(&self.pair_histogram_bin_width_px)
            || !valid_float(self.pair_peak_radius_px)
            || self.pair_peak_radius_px <= 0.0
            || self.pair_peak_radius_px > 8.0
            || !valid_float(self.minimum_cross_section_contrast)
            || !(0.0..=255.0).contains(&self.minimum_cross_section_contrast)
            || !valid_float(self.minimum_tip_contrast)
            || !(0.0..=255.0).contains(&self.minimum_tip_contrast)
        {
            return Err(CueGeometryError::InvalidConfiguration);
        }

        let max_image_bytes = (self.max_width_px as usize)
            .checked_mul(self.max_height_px as usize)
            .and_then(|pixels| pixels.checked_mul(RGB_CHANNELS))
            .ok_or(CueGeometryError::InvalidConfiguration)?;
        if self.max_pixels > (self.max_width_px as usize) * (self.max_height_px as usize)
            || max_image_bytes == 0
        {
            return Err(CueGeometryError::InvalidConfiguration);
        }
        let point_bytes = std::mem::size_of::<EdgePoint>();
        let required_temporary_bytes = self
            .max_edge_samples
            .checked_mul(point_bytes)
            .and_then(|bytes| bytes.checked_mul(2))
            .ok_or(CueGeometryError::InvalidConfiguration)?;
        if required_temporary_bytes > self.max_temporary_bytes {
            return Err(CueGeometryError::InvalidConfiguration);
        }
        let region_count = self
            .max_proposals
            .checked_add(1)
            .ok_or(CueGeometryError::InvalidConfiguration)?;
        let edge_visits_per_region = self
            .max_refinements
            .checked_add(2)
            .and_then(|passes| self.max_edge_samples.checked_mul(passes))
            .ok_or(CueGeometryError::InvalidConfiguration)?;
        let profile_visits_per_region =
            profile_visit_budget(self.max_width_px, self.max_height_px, self)
                .ok_or(CueGeometryError::InvalidConfiguration)?;
        let visits_per_region = edge_visits_per_region
            .checked_add(profile_visits_per_region)
            .ok_or(CueGeometryError::InvalidConfiguration)?;
        let worst_case_visits = visits_per_region
            .checked_mul(region_count)
            .ok_or(CueGeometryError::InvalidConfiguration)?;
        if self.max_point_visits < worst_case_visits {
            return Err(CueGeometryError::InvalidConfiguration);
        }
        Ok(self)
    }
}

fn profile_visit_budget(width_px: u32, height_px: u32, config: CueGeometryConfig) -> Option<usize> {
    let maximum_width = (f64::from(width_px.min(height_px)) * config.maximum_shaft_width_fraction)
        .max(config.minimum_shaft_maximum_width_px);
    let half_range = maximum_width + config.pair_peak_radius_px;
    let histogram_len =
        ((2.0 * half_range / config.pair_histogram_bin_width_px).ceil() as usize).checked_add(1)?;
    if histogram_len > CENTERLINE_HISTOGRAM_CAPACITY {
        return None;
    }
    config
        .centerline_sections
        .checked_mul(config.cross_section_profile_samples)?
        .checked_mul(histogram_len)
}

/// Integer image-space proposal, with right and bottom exclusive.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CueProposal {
    pub left: u32,
    pub top: u32,
    pub right: u32,
    pub bottom: u32,
}

/// Endpoint on the cue-tip side of the shaft. When a compatible table
/// calibration is available, the endpoint on the playing surface is the tip
/// side and the endpoint outside the surface is the butt side. Appearance is
/// retained as a fallback when the calibrated boundary cannot orient the line.
/// This is not a motion vector or a complete shot-direction estimate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SignedCueDirection {
    TowardStart,
    TowardEnd,
}

/// Shaft geometry refined from the current image.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CueGeometry {
    pub shaft_start_px: [f64; 2],
    pub shaft_end_px: [f64; 2],
    /// Unit vector pointing from canonical start toward canonical end.
    pub axis_unit: [f64; 2],
    /// Confidence in the fitted image line only, in [0, 1].
    pub confidence: f64,
    /// Age is supplied by the caller because this module has no clock authority.
    pub observation_age_ms: u64,
    /// `None` means neither calibrated table geometry nor endpoint appearance
    /// could confidently distinguish the tip side from the butt side.
    pub signed_direction: Option<SignedCueDirection>,
    /// `None` means the line came from bounded whole-image reacquisition.
    pub proposal_index: Option<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CueGeometryError {
    InvalidConfiguration,
    InvalidDimensions { width_px: u32, height_px: u32 },
    ImageTooLarge { width_px: u32, height_px: u32 },
    PixelLengthMismatch { expected: usize, actual: usize },
    AllocationFailed,
}

/// Estimate cue-shaft geometry from one tightly packed RGB8 image.
///
/// Recent detector regions are tested first. A whole-image candidate is also
/// evaluated so a cue that has moved outside a previous proposal can be
/// reacquired without asking a neural detector to run at cue-update cadence.
/// All image dimensions, input bytes, samples, proposal count, temporary bytes,
/// and point visits are bounded by `config`.
pub fn estimate_cue_geometry(
    rgb8: &[u8],
    width_px: u32,
    height_px: u32,
    proposals: &[CueProposal],
    observation_age: Duration,
    config: CueGeometryConfig,
) -> Result<Option<CueGeometry>, CueGeometryError> {
    let config = config.validate()?;
    validate_image(rgb8, width_px, height_px, config)?;

    let mut edge_points = Vec::new();
    edge_points
        .try_reserve_exact(config.max_edge_samples)
        .map_err(|_| CueGeometryError::AllocationFailed)?;
    extract_edge_samples(
        rgb8,
        width_px as usize,
        height_px as usize,
        config,
        &mut edge_points,
    );
    if edge_points.len() < config.minimum_support_points {
        return Ok(None);
    }

    // Reuse one bounded temporary point array across ROIs. The point-visit
    // budget covers proposal filtering and each robust fitting pass, so a list
    // of broad boxes cannot create unbounded repeated work.
    let mut inliers = Vec::new();
    inliers
        .try_reserve_exact(config.max_edge_samples)
        .map_err(|_| CueGeometryError::AllocationFailed)?;
    let allocated_point_bytes = edge_points
        .capacity()
        .checked_add(inliers.capacity())
        .and_then(|capacity| capacity.checked_mul(std::mem::size_of::<EdgePoint>()))
        .ok_or(CueGeometryError::AllocationFailed)?;
    if edge_points.capacity() > config.max_edge_samples
        || inliers.capacity() > config.max_edge_samples
        || allocated_point_bytes > config.max_temporary_bytes
    {
        return Err(CueGeometryError::AllocationFailed);
    }
    let mut point_visits = 0_usize;
    let mut best: Option<(Fit, Option<usize>)> = None;

    for (proposal_index, proposal) in proposals.iter().take(config.max_proposals).enumerate() {
        let Some(region) = clip_region(*proposal, width_px, height_px) else {
            continue;
        };
        if let Some(fit) = fit_region(
            &edge_points,
            &mut inliers,
            region,
            width_px,
            height_px,
            config,
            &mut point_visits,
            rgb8,
        ) {
            select_best(&mut best, fit, Some(proposal_index));
        }
    }

    // This bounded fallback is what allows the shaft to reappear elsewhere
    // after the old detector region has become stale.
    let full_image = Region {
        left: 0.0,
        top: 0.0,
        right: width_px as f64,
        bottom: height_px as f64,
    };
    if let Some(fit) = fit_region(
        &edge_points,
        &mut inliers,
        full_image,
        width_px,
        height_px,
        config,
        &mut point_visits,
        rgb8,
    ) {
        select_best(&mut best, fit, None);
    }

    let Some((fit, proposal_index)) = best else {
        return Ok(None);
    };
    let (shaft_start_px, shaft_end_px, axis_unit) = canonical_endpoints(fit);
    let signed_direction = infer_dark_tip_endpoint(
        rgb8,
        width_px as usize,
        height_px as usize,
        shaft_start_px,
        shaft_end_px,
        axis_unit,
        config.minimum_tip_contrast,
    );
    Ok(Some(CueGeometry {
        shaft_start_px,
        shaft_end_px,
        axis_unit,
        confidence: fit.confidence,
        observation_age_ms: observation_age.as_millis().min(u64::MAX as u128) as u64,
        signed_direction,
        proposal_index,
    }))
}

/// Prefer the calibrated playing-surface boundary to orient an already fitted
/// cue. A cue shaft that crosses the cloth boundary has one endpoint on the
/// table (tip side) and one outside (butt side), which resolves the otherwise
/// arbitrary sign of the fitted line. If calibration is incompatible, either
/// endpoint cannot be projected, or both endpoints occupy the same side of
/// the boundary, this leaves the estimator's appearance-based fallback intact.
pub fn orient_cue_geometry_from_playing_surface(
    geometry: &mut CueGeometry,
    calibration: &ValidatedTableCalibrationV1,
    context: CalibrationContext<'_>,
) -> bool {
    let Ok(start) = calibration.project(
        ImagePointPx {
            x_px: geometry.shaft_start_px[0],
            y_px: geometry.shaft_start_px[1],
        },
        context,
    ) else {
        return false;
    };
    let Ok(end) = calibration.project(
        ImagePointPx {
            x_px: geometry.shaft_end_px[0],
            y_px: geometry.shaft_end_px[1],
        },
        context,
    ) else {
        return false;
    };

    let start_on_table = calibration.contains_playing_surface_point(start);
    let end_on_table = calibration.contains_playing_surface_point(end);
    match (start_on_table, end_on_table) {
        (true, false) => geometry.signed_direction = Some(SignedCueDirection::TowardStart),
        (false, true) => geometry.signed_direction = Some(SignedCueDirection::TowardEnd),
        // A fit fully over the cloth or fully outside the table is not enough
        // evidence to tell which shaft end is the tip.
        _ => return false,
    }
    true
}

#[derive(Debug, Clone, Copy)]
struct EdgePoint {
    x: f64,
    y: f64,
}

#[derive(Debug, Clone, Copy)]
struct Region {
    left: f64,
    top: f64,
    right: f64,
    bottom: f64,
}

#[derive(Debug, Clone, Copy)]
struct Fit {
    center: [f64; 2],
    axis: [f64; 2],
    min_projection: f64,
    max_projection: f64,
    confidence: f64,
}

fn validate_image(
    rgb8: &[u8],
    width_px: u32,
    height_px: u32,
    config: CueGeometryConfig,
) -> Result<(), CueGeometryError> {
    if width_px < 3 || height_px < 3 {
        return Err(CueGeometryError::InvalidDimensions {
            width_px,
            height_px,
        });
    }
    if width_px > config.max_width_px || height_px > config.max_height_px {
        return Err(CueGeometryError::ImageTooLarge {
            width_px,
            height_px,
        });
    }
    let pixels = (width_px as usize).checked_mul(height_px as usize).ok_or(
        CueGeometryError::ImageTooLarge {
            width_px,
            height_px,
        },
    )?;
    let expected = pixels
        .checked_mul(RGB_CHANNELS)
        .ok_or(CueGeometryError::ImageTooLarge {
            width_px,
            height_px,
        })?;
    if pixels > config.max_pixels {
        return Err(CueGeometryError::ImageTooLarge {
            width_px,
            height_px,
        });
    }
    if rgb8.len() != expected {
        return Err(CueGeometryError::PixelLengthMismatch {
            expected,
            actual: rgb8.len(),
        });
    }
    Ok(())
}

/// Sample all pixels in the configured image bound but retain at most the
/// configured number of edge points using deterministic reservoir sampling.
/// This preserves image-wide coverage better than filling a capped vector from
/// the top-left corner, while keeping storage independent of image texture.
fn extract_edge_samples(
    rgb8: &[u8],
    width: usize,
    height: usize,
    config: CueGeometryConfig,
    output: &mut Vec<EdgePoint>,
) {
    let mut edge_count = 0_u64;
    let mut random_state = ((width as u64) << 32) ^ height as u64 ^ 0x9e37_79b9_7f4a_7c15;
    for y in 1..height - 1 {
        for x in 1..width - 1 {
            let gx = luminance(rgb8, (y * width + x + 1) * 3) as f64
                - luminance(rgb8, (y * width + x - 1) * 3) as f64;
            let gy = luminance(rgb8, ((y + 1) * width + x) * 3) as f64
                - luminance(rgb8, ((y - 1) * width + x) * 3) as f64;
            if gx.mul_add(gx, gy * gy).sqrt() < config.minimum_gradient {
                continue;
            }
            edge_count = edge_count.saturating_add(1);
            let point = EdgePoint {
                x: x as f64,
                y: y as f64,
            };
            if output.len() < config.max_edge_samples {
                output.push(point);
            } else {
                random_state ^= random_state << 13;
                random_state ^= random_state >> 7;
                random_state ^= random_state << 17;
                let slot = random_state % edge_count;
                if slot < config.max_edge_samples as u64 {
                    output[slot as usize] = point;
                }
            }
        }
    }
}

fn luminance(rgb8: &[u8], offset: usize) -> u8 {
    let red = rgb8[offset] as u32;
    let green = rgb8[offset + 1] as u32;
    let blue = rgb8[offset + 2] as u32;
    ((red * 77 + green * 150 + blue * 29) >> 8) as u8
}

fn clip_region(proposal: CueProposal, width: u32, height: u32) -> Option<Region> {
    let left = proposal.left.min(width) as f64;
    let top = proposal.top.min(height) as f64;
    let right = proposal.right.min(width) as f64;
    let bottom = proposal.bottom.min(height) as f64;
    (right - left >= 3.0 && bottom - top >= 3.0).then_some(Region {
        left,
        top,
        right,
        bottom,
    })
}

// Keep geometry inputs explicit here: the caller shares one bounded point
// visit counter across all candidate regions, while only proposal regions use
// the detector-box seed path.
#[allow(clippy::too_many_arguments)]
fn fit_region(
    points: &[EdgePoint],
    inliers: &mut Vec<EdgePoint>,
    region: Region,
    width: u32,
    height: u32,
    config: CueGeometryConfig,
    point_visits: &mut usize,
    rgb8: &[u8],
) -> Option<Fit> {
    inliers.clear();
    for point in points {
        *point_visits = point_visits.checked_add(1)?;
        if *point_visits > config.max_point_visits {
            return None;
        }
        if in_region(*point, region) {
            inliers.push(*point);
        }
    }
    if inliers.len() < config.minimum_support_points {
        return None;
    }

    let mut line = principal_line(inliers, config.minimum_eigenvalue_ratio)?;
    for _ in 0..config.max_refinements {
        inliers.clear();
        for point in points {
            *point_visits = point_visits.checked_add(1)?;
            if *point_visits > config.max_point_visits {
                return None;
            }
            if in_region(*point, region)
                && perpendicular_distance(*point, line.center, line.axis)
                    <= config.maximum_fit_residual_px
            {
                inliers.push(*point);
            }
        }
        if inliers.len() < config.minimum_support_points {
            return None;
        }
        line = principal_line(inliers, config.minimum_eigenvalue_ratio)?;
    }

    let mut occupied = [false; 16];
    let mut max_residual = 0.0_f64;
    let mut min_projection = f64::INFINITY;
    let mut max_projection = f64::NEG_INFINITY;
    for point in inliers.iter().copied() {
        let projection = dot(
            [point.x - line.center[0], point.y - line.center[1]],
            line.axis,
        );
        min_projection = min_projection.min(projection);
        max_projection = max_projection.max(projection);
        max_residual = max_residual.max(perpendicular_distance(point, line.center, line.axis));
    }
    let side_length = max_projection - min_projection;
    let longest_dimension = width.max(height) as f64;
    if side_length < longest_dimension * config.minimum_length_fraction
        || side_length > longest_dimension * config.maximum_length_fraction_of_longest_dimension
        || max_residual > config.maximum_fit_residual_px * 1.5
    {
        return None;
    }
    for point in inliers.iter().copied() {
        *point_visits = point_visits.checked_add(1)?;
        if *point_visits > config.max_point_visits {
            return None;
        }
        let projection = dot(
            [point.x - line.center[0], point.y - line.center[1]],
            line.axis,
        );
        let fraction = (projection - min_projection) / side_length;
        let bin = ((fraction * occupied.len() as f64) as usize).min(occupied.len() - 1);
        occupied[bin] = true;
    }
    let occupied_fraction =
        occupied.iter().filter(|value| **value).count() as f64 / occupied.len() as f64;
    if occupied_fraction < config.minimum_occupied_length_fraction {
        return None;
    }
    let residual_ratio = (max_residual / config.maximum_fit_residual_px).min(1.0);
    let length_score = ((side_length / longest_dimension) / 0.45).clamp(0.0, 1.0);
    let support_score = ((inliers.len() as f64) / 80.0).clamp(0.0, 1.0);
    let confidence = (occupied_fraction * 0.45 + length_score * 0.30 + support_score * 0.25)
        * (1.0 - residual_ratio * 0.25);
    paired_centerline_fit(
        line,
        min_projection,
        max_projection,
        width,
        height,
        config,
        point_visits,
        rgb8,
        confidence.clamp(0.0, 1.0),
    )
}

#[derive(Debug, Clone, Copy)]
struct EdgePeak {
    offset: f64,
    support: u32,
}

#[derive(Debug, Clone, Copy)]
struct PairedEdges {
    left: f64,
    right: f64,
    support: u32,
    contrast: f64,
}

/// Recover the central shaft axis from edge pairs in each cross-section.
///
/// The initial robust line identifies a cue-like orientation and longitudinal
/// span. It is deliberately not returned directly: that line can be either
/// visible side of the shaft, and its position can therefore flip when edge
/// contrast changes. This pass finds two supported edges around that proposal,
/// verifies that they enclose a coherent image region, and fits their midpoint
/// samples. Missing or ambiguous cross-sections are skipped instead of
/// allowing one edge to masquerade as the shaft center.
#[allow(clippy::too_many_arguments)]
fn paired_centerline_fit(
    edge_line: PrincipalLine,
    min_projection: f64,
    max_projection: f64,
    width: u32,
    height: u32,
    config: CueGeometryConfig,
    point_visits: &mut usize,
    rgb8: &[u8],
    side_confidence: f64,
) -> Option<Fit> {
    let side_length = max_projection - min_projection;
    if side_length <= f64::EPSILON {
        return None;
    }

    let shorter_dimension = f64::from(width.min(height));
    let minimum_width = config
        .minimum_shaft_width_px
        .max(shorter_dimension * config.minimum_shaft_width_fraction);
    // At thumbnail-sized inputs the relative cap can be narrower than a
    // resolvable shaft. The configured pixel fallback keeps both edges in the
    // sampling window, while normal camera resolutions remain governed by the
    // dimension-scaled maximum above.
    let maximum_width = (shorter_dimension * config.maximum_shaft_width_fraction)
        .max(config.minimum_shaft_maximum_width_px);
    if maximum_width < minimum_width {
        return None;
    }

    let bin_width = config.pair_histogram_bin_width_px;
    let half_range = maximum_width + config.pair_peak_radius_px;
    let histogram_len = ((2.0 * half_range / bin_width).ceil() as usize).saturating_add(1);
    if histogram_len > CENTERLINE_HISTOGRAM_CAPACITY {
        return None;
    }

    let normal = [-edge_line.axis[1], edge_line.axis[0]];
    let mut center_points = [EdgePoint { x: 0.0, y: 0.0 }; MAX_CENTERLINE_SECTIONS];
    let mut widths = [0.0_f64; MAX_CENTERLINE_SECTIONS];
    let mut section_indices = [0_usize; MAX_CENTERLINE_SECTIONS];
    let mut center_count = 0_usize;
    // Keep cross-sections about eight pixels apart so their profiles observe
    // distinct raster phases. On thumbnail inputs this can collapse the work
    // to one section whose longitudinal profiles still yield paired edges.
    let section_count = config.centerline_sections.min(
        (side_length / config.minimum_centerline_section_spacing_px)
            .floor()
            .max(1.0) as usize,
    );
    let section_width = side_length / section_count as f64;
    let peak_radius_bins = (config.pair_peak_radius_px / bin_width).floor() as usize;
    let mut profile_histogram = [0_u32; CENTERLINE_HISTOGRAM_CAPACITY];
    let mut smoothed = [0_u32; CENTERLINE_HISTOGRAM_CAPACITY];

    for section_index in 0..section_count {
        profile_histogram[..histogram_len].fill(0);

        // Sparse edge extraction can miss one side of a thin diagonal shaft
        // because of pixel-grid phase. Re-sample short RGB profiles at several
        // positions within this section so both physical boundaries vote even
        // when the global edge reservoir contains only one of them.
        let profile_gradient_threshold = config.minimum_gradient * bin_width / 2.0;
        for profile_index in 0..config.cross_section_profile_samples {
            let along = min_projection
                + (section_index as f64
                    + (profile_index as f64 + 0.5) / config.cross_section_profile_samples as f64)
                    * section_width;
            let anchor = [
                edge_line.center[0] + edge_line.axis[0] * along,
                edge_line.center[1] + edge_line.axis[1] * along,
            ];
            for (bin, votes) in profile_histogram[..histogram_len].iter_mut().enumerate() {
                *point_visits = point_visits.checked_add(1)?;
                if *point_visits > config.max_point_visits {
                    return None;
                }
                let first_offset = -half_range + bin as f64 * bin_width;
                let second_offset = first_offset + bin_width;
                let first = sample_luminance_bilinear(
                    rgb8,
                    width as usize,
                    height as usize,
                    [
                        anchor[0] + normal[0] * first_offset,
                        anchor[1] + normal[1] * first_offset,
                    ],
                );
                let second = sample_luminance_bilinear(
                    rgb8,
                    width as usize,
                    height as usize,
                    [
                        anchor[0] + normal[0] * second_offset,
                        anchor[1] + normal[1] * second_offset,
                    ],
                );
                if let (Some(first), Some(second)) = (first, second) {
                    if (second - first).abs() >= profile_gradient_threshold {
                        *votes = votes.saturating_add(1);
                    }
                }
            }
        }
        for (bin, smoothed_support) in smoothed[..histogram_len].iter_mut().enumerate() {
            let start = bin.saturating_sub(peak_radius_bins);
            let end = bin
                .saturating_add(peak_radius_bins)
                .saturating_add(1)
                .min(histogram_len);
            *smoothed_support = profile_histogram[start..end]
                .iter()
                .fold(0_u32, |total, count| total.saturating_add(*count));
        }

        let mut peaks = [EdgePeak {
            offset: 0.0,
            support: 0,
        }; MAX_EDGE_PEAKS_PER_SECTION];
        let mut peak_count = 0_usize;
        for bin in 0..histogram_len {
            let support = smoothed[bin];
            if support < config.minimum_pair_edge_support as u32
                || (bin > 0 && support < smoothed[bin - 1])
                || (bin + 1 < histogram_len && support <= smoothed[bin + 1])
            {
                continue;
            }

            let start = bin.saturating_sub(peak_radius_bins);
            let end = bin
                .saturating_add(peak_radius_bins)
                .saturating_add(1)
                .min(histogram_len);
            let mut weighted_offset = 0.0;
            let mut total_weight = 0_u32;
            for (neighbor, weight) in profile_histogram.iter().enumerate().take(end).skip(start) {
                weighted_offset +=
                    (-half_range + (neighbor as f64 + 0.5) * bin_width) * f64::from(*weight);
                total_weight = total_weight.saturating_add(*weight);
            }
            if total_weight == 0 {
                continue;
            }
            let peak = EdgePeak {
                offset: weighted_offset / f64::from(total_weight),
                support,
            };
            let insert_at = peaks[..peak_count]
                .iter()
                .position(|current| peak.support > current.support)
                .unwrap_or(peak_count);
            if insert_at >= config.maximum_edge_peaks_per_section {
                continue;
            }
            let new_count = peak_count
                .saturating_add(1)
                .min(config.maximum_edge_peaks_per_section);
            for index in (insert_at + 1..new_count).rev() {
                peaks[index] = peaks[index - 1];
            }
            peaks[insert_at] = peak;
            peak_count = new_count;
        }

        let section_projection = min_projection + (section_index as f64 + 0.5) * section_width;
        let anchor = [
            edge_line.center[0] + edge_line.axis[0] * section_projection,
            edge_line.center[1] + edge_line.axis[1] * section_projection,
        ];
        let mut best_pair: Option<PairedEdges> = None;
        for left_index in 0..peak_count {
            for right_index in left_index + 1..peak_count {
                let first = peaks[left_index];
                let second = peaks[right_index];
                let (left, right) = if first.offset <= second.offset {
                    (first, second)
                } else {
                    (second, first)
                };
                let shaft_width = right.offset - left.offset;
                if shaft_width < minimum_width || shaft_width > maximum_width {
                    continue;
                }
                let support = left.support.min(right.support);
                if support < config.minimum_pair_edge_support as u32 {
                    continue;
                }
                let Some(contrast) = cross_section_contrast(
                    rgb8,
                    width as usize,
                    height as usize,
                    anchor,
                    normal,
                    left.offset,
                    right.offset,
                    config.maximum_fit_residual_px,
                ) else {
                    continue;
                };
                if contrast < config.minimum_cross_section_contrast {
                    continue;
                }
                let candidate = PairedEdges {
                    left: left.offset,
                    right: right.offset,
                    support,
                    contrast,
                };
                let candidate_center_distance = ((candidate.left + candidate.right) * 0.5).abs();
                let replaces = best_pair.is_none_or(|current| {
                    candidate.contrast > current.contrast
                        || (candidate.contrast == current.contrast
                            && (candidate.support > current.support
                                || (candidate.support == current.support
                                    && candidate_center_distance
                                        < ((current.left + current.right) * 0.5).abs())))
                });
                if replaces {
                    best_pair = Some(candidate);
                }
            }
        }

        if let Some(pair) = best_pair {
            center_points[center_count] = EdgePoint {
                x: anchor[0] + normal[0] * (pair.left + pair.right) * 0.5,
                y: anchor[1] + normal[1] * (pair.left + pair.right) * 0.5,
            };
            widths[center_count] = pair.right - pair.left;
            section_indices[center_count] = section_index;
            center_count += 1;
        }
    }

    let minimum_sections =
        (config.minimum_paired_section_fraction * section_count as f64).ceil() as usize;
    if center_count < minimum_sections {
        return None;
    }

    let residual_limit = config.maximum_fit_residual_px * 1.5;
    let (centerline, retained) = if center_count == 1 {
        // A short but clear shaft can provide only one resolvable cross-
        // section. Its paired edges determine the center offset, while the
        // robust image-edge fit supplies the unsigned line direction.
        let point = center_points[0];
        let center_offset = dot(
            [point.x - edge_line.center[0], point.y - edge_line.center[1]],
            normal,
        );
        (
            PrincipalLine {
                center: [
                    edge_line.center[0] + normal[0] * center_offset,
                    edge_line.center[1] + normal[1] * center_offset,
                ],
                axis: edge_line.axis,
            },
            1,
        )
    } else {
        let initial_centerline = principal_line(
            &center_points[..center_count],
            config.minimum_eigenvalue_ratio,
        )?;
        let mut retained = 0_usize;
        for index in 0..center_count {
            if perpendicular_distance(
                center_points[index],
                initial_centerline.center,
                initial_centerline.axis,
            ) <= residual_limit
            {
                center_points[retained] = center_points[index];
                widths[retained] = widths[index];
                section_indices[retained] = section_indices[index];
                retained += 1;
            }
        }
        if retained < minimum_sections {
            return None;
        }
        let centerline =
            principal_line(&center_points[..retained], config.minimum_eigenvalue_ratio)?;
        (centerline, retained)
    };

    let mut ordered_widths = widths;
    ordered_widths[..retained].sort_unstable_by(f64::total_cmp);
    let lower_width = ordered_widths[retained / 4];
    let upper_width = ordered_widths[(retained * 3 / 4).min(retained - 1)];
    if lower_width <= 0.0 || upper_width / lower_width > config.maximum_shaft_taper_ratio {
        return None;
    }

    let mut occupied = [false; 16];
    let occupied_bin_count = section_count.min(occupied.len());
    let mut max_residual = 0.0_f64;
    let (mut min_center_projection, mut max_center_projection) = if retained == 1 {
        (min_projection, max_projection)
    } else {
        let mut minimum = f64::INFINITY;
        let mut maximum = f64::NEG_INFINITY;
        for point in center_points[..retained].iter().copied() {
            let projection = dot(
                [
                    point.x - centerline.center[0],
                    point.y - centerline.center[1],
                ],
                centerline.axis,
            );
            minimum = minimum.min(projection);
            maximum = maximum.max(projection);
            max_residual = max_residual.max(perpendicular_distance(
                point,
                centerline.center,
                centerline.axis,
            ));
        }
        (minimum, maximum)
    };
    let centerline_length = max_center_projection - min_center_projection;
    let longest_dimension = f64::from(width.max(height));
    if centerline_length < longest_dimension * config.minimum_length_fraction
        || centerline_length
            > longest_dimension * config.maximum_length_fraction_of_longest_dimension
        || max_residual > residual_limit
    {
        return None;
    }
    for point in center_points[..retained].iter().copied() {
        let projection = dot(
            [
                point.x - centerline.center[0],
                point.y - centerline.center[1],
            ],
            centerline.axis,
        );
        let fraction = (projection - min_center_projection) / centerline_length;
        let bin = ((fraction * occupied_bin_count as f64) as usize).min(occupied_bin_count - 1);
        occupied[bin] = true;
    }
    let occupied_fraction = occupied[..occupied_bin_count]
        .iter()
        .filter(|value| **value)
        .count() as f64
        / occupied_bin_count as f64;
    if occupied_fraction < config.minimum_paired_section_fraction {
        return None;
    }

    let first_section = section_indices[..retained].iter().min().copied()?;
    let last_section = section_indices[..retained].iter().max().copied()?;
    let end_extension = section_width * 0.5;
    if section_count > 1 && first_section == 0 {
        min_center_projection -= end_extension;
    }
    if section_count > 1 && last_section + 1 == section_count {
        max_center_projection += end_extension;
    }
    let pairing_fraction = retained as f64 / section_count as f64;
    let length_score = ((centerline_length / longest_dimension) / 0.45).clamp(0.0, 1.0);
    let paired_confidence =
        (pairing_fraction * 0.45 + occupied_fraction * 0.30 + length_score * 0.25).clamp(0.0, 1.0);
    Some(Fit {
        center: centerline.center,
        axis: centerline.axis,
        min_projection: min_center_projection,
        max_projection: max_center_projection,
        confidence: (side_confidence * 0.35 + paired_confidence * 0.65).clamp(0.0, 1.0),
    })
}

#[allow(clippy::too_many_arguments)]
fn cross_section_contrast(
    rgb8: &[u8],
    width: usize,
    height: usize,
    anchor: [f64; 2],
    normal: [f64; 2],
    left: f64,
    right: f64,
    sample_margin: f64,
) -> Option<f64> {
    let shaft_width = right - left;
    let interior_offsets = [
        left + shaft_width * 0.25,
        left + shaft_width * 0.5,
        left + shaft_width * 0.75,
    ];
    let exterior_offsets = [left - sample_margin, right + sample_margin];
    let sample = |offset: f64| {
        sample_luminance_bilinear(
            rgb8,
            width,
            height,
            [
                anchor[0] + normal[0] * offset,
                anchor[1] + normal[1] * offset,
            ],
        )
    };
    let mut interior_samples = [0.0; 3];
    let mut interior_count = 0;
    for offset in interior_offsets {
        if let Some(value) = sample(offset) {
            interior_samples[interior_count] = value;
            interior_count += 1;
        }
    }
    let mut exterior_samples = [0.0; 2];
    let mut exterior_count = 0;
    for offset in exterior_offsets {
        if let Some(value) = sample(offset) {
            exterior_samples[exterior_count] = value;
            exterior_count += 1;
        }
    }
    // Near an image boundary one exterior sample can fall outside the frame.
    // Keep the valid-side contrast evidence instead of dropping a real cue
    // that enters or leaves at the edge of a thumbnail.
    if interior_count < 2 || exterior_count == 0 {
        return None;
    }
    let interior_mean =
        interior_samples[..interior_count].iter().sum::<f64>() / interior_count as f64;
    let exterior_mean =
        exterior_samples[..exterior_count].iter().sum::<f64>() / exterior_count as f64;
    let interior_deviation = interior_samples[..interior_count]
        .iter()
        .map(|sample| (sample - interior_mean).abs())
        .sum::<f64>()
        / interior_count as f64;
    let exterior_deviation = exterior_samples[..exterior_count]
        .iter()
        .map(|sample| (sample - exterior_mean).abs())
        .sum::<f64>()
        / exterior_count as f64;
    Some(((interior_mean - exterior_mean).abs() - interior_deviation - exterior_deviation).max(0.0))
}

fn sample_luminance_bilinear(
    rgb8: &[u8],
    width: usize,
    height: usize,
    point: [f64; 2],
) -> Option<f64> {
    if !point[0].is_finite()
        || !point[1].is_finite()
        || point[0] < 0.0
        || point[1] < 0.0
        || point[0] > (width - 1) as f64
        || point[1] > (height - 1) as f64
    {
        return None;
    }
    let x0 = point[0].floor() as usize;
    let y0 = point[1].floor() as usize;
    let x1 = (x0 + 1).min(width - 1);
    let y1 = (y0 + 1).min(height - 1);
    let x_fraction = point[0] - x0 as f64;
    let y_fraction = point[1] - y0 as f64;
    let sample = |x: usize, y: usize| luminance(rgb8, (y * width + x) * RGB_CHANNELS) as f64;
    let top = sample(x0, y0) * (1.0 - x_fraction) + sample(x1, y0) * x_fraction;
    let bottom = sample(x0, y1) * (1.0 - x_fraction) + sample(x1, y1) * x_fraction;
    Some(top * (1.0 - y_fraction) + bottom * y_fraction)
}

fn in_region(point: EdgePoint, region: Region) -> bool {
    point.x >= region.left
        && point.x < region.right
        && point.y >= region.top
        && point.y < region.bottom
}

fn select_best(best: &mut Option<(Fit, Option<usize>)>, candidate: Fit, source: Option<usize>) {
    // A proposal gets a small tie-breaking preference because it came from
    // recent cue evidence. A substantially stronger whole-image fit can still
    // win, which is useful when an old proposal has become stale.
    let candidate_score = candidate.confidence + if source.is_some() { 0.06 } else { 0.0 };
    let should_replace = best.is_none_or(|(current, current_source)| {
        let current_score = current.confidence + if current_source.is_some() { 0.06 } else { 0.0 };
        candidate_score > current_score
    });
    if should_replace {
        *best = Some((candidate, source));
    }
}

#[derive(Debug, Clone, Copy)]
struct PrincipalLine {
    center: [f64; 2],
    axis: [f64; 2],
}

/// Principal-component line fit. Repeated residual trimming in `fit_region`
/// makes the final estimate robust to isolated edges and modest texture, while
/// the eigenvalue ratio rejects broad blobs and unstructured clutter.
fn principal_line(points: &[EdgePoint], minimum_eigenvalue_ratio: f64) -> Option<PrincipalLine> {
    let count = points.len() as f64;
    if count < 2.0 {
        return None;
    }
    let (sum_x, sum_y) = points
        .iter()
        .fold((0.0, 0.0), |(sx, sy), point| (sx + point.x, sy + point.y));
    let center = [sum_x / count, sum_y / count];
    let (xx, xy, yy) = points.iter().fold((0.0, 0.0, 0.0), |(xx, xy, yy), point| {
        let dx = point.x - center[0];
        let dy = point.y - center[1];
        (xx + dx * dx, xy + dx * dy, yy + dy * dy)
    });
    let xx = xx / count;
    let xy = xy / count;
    let yy = yy / count;
    let trace = xx + yy;
    let discriminant = ((xx - yy).mul_add(xx - yy, 4.0 * xy * xy)).sqrt();
    let major = (trace + discriminant) / 2.0;
    let minor = (trace - discriminant) / 2.0;
    // Zero minor variance is the strongest possible line fit, not a degenerate
    // blob. This matters for paired side midpoints that lie on the exact same
    // pixel-space axis after subpixel averaging.
    if major <= f64::EPSILON || (minor > f64::EPSILON && major / minor < minimum_eigenvalue_ratio) {
        return None;
    }
    let mut axis = if xy.abs() > 1e-9 {
        let x = major - yy;
        normalize([x, xy])?
    } else if xx >= yy {
        [1.0, 0.0]
    } else {
        [0.0, 1.0]
    };
    if axis[0] < 0.0 || (axis[0].abs() <= 1e-9 && axis[1] < 0.0) {
        axis = [-axis[0], -axis[1]];
    }
    Some(PrincipalLine { center, axis })
}

fn normalize(vector: [f64; 2]) -> Option<[f64; 2]> {
    let norm = vector[0].hypot(vector[1]);
    (norm > f64::EPSILON).then_some([vector[0] / norm, vector[1] / norm])
}

fn perpendicular_distance(point: EdgePoint, center: [f64; 2], axis: [f64; 2]) -> f64 {
    let dx = point.x - center[0];
    let dy = point.y - center[1];
    (dx * -axis[1] + dy * axis[0]).abs()
}

fn dot(left: [f64; 2], right: [f64; 2]) -> f64 {
    left[0].mul_add(right[0], left[1] * right[1])
}

fn canonical_endpoints(fit: Fit) -> ([f64; 2], [f64; 2], [f64; 2]) {
    let mut axis = fit.axis;
    if axis[0] < 0.0 || (axis[0].abs() <= 1e-9 && axis[1] < 0.0) {
        axis = [-axis[0], -axis[1]];
    }
    let start = [
        fit.center[0] + axis[0] * fit.min_projection,
        fit.center[1] + axis[1] * fit.min_projection,
    ];
    let end = [
        fit.center[0] + axis[0] * fit.max_projection,
        fit.center[1] + axis[1] * fit.max_projection,
    ];
    (start, end, axis)
}

/// Label a likely tip only when an endpoint has a strong dark contrast against
/// both the opposite endpoint and the shaft's middle. A uniformly colored cue
/// or a partially occluded line has no defensible sign, so returns `None`.
fn infer_dark_tip_endpoint(
    rgb8: &[u8],
    width: usize,
    height: usize,
    start: [f64; 2],
    end: [f64; 2],
    axis: [f64; 2],
    minimum_contrast: f64,
) -> Option<SignedCueDirection> {
    let length = (end[0] - start[0]).hypot(end[1] - start[1]);
    if length < 30.0 {
        return None;
    }
    let start_luma = endpoint_luminance(rgb8, width, height, start, axis, length, 0.06)?;
    let end_luma =
        endpoint_luminance(rgb8, width, height, end, [-axis[0], -axis[1]], length, 0.06)?;
    let middle = [(start[0] + end[0]) / 2.0, (start[1] + end[1]) / 2.0];
    let middle_luma = endpoint_luminance(rgb8, width, height, middle, axis, length, 0.10)?;
    if start_luma + minimum_contrast <= end_luma && start_luma + minimum_contrast <= middle_luma {
        Some(SignedCueDirection::TowardStart)
    } else if end_luma + minimum_contrast <= start_luma
        && end_luma + minimum_contrast <= middle_luma
    {
        Some(SignedCueDirection::TowardEnd)
    } else {
        None
    }
}

fn endpoint_luminance(
    rgb8: &[u8],
    width: usize,
    height: usize,
    anchor: [f64; 2],
    inward_axis: [f64; 2],
    shaft_length: f64,
    span_fraction: f64,
) -> Option<f64> {
    let normal = [-inward_axis[1], inward_axis[0]];
    let longitudinal = (shaft_length * span_fraction).max(3.0);
    let mut total = 0.0;
    let mut count = 0_usize;
    for step in 0..6 {
        let distance = longitudinal * (step as f64 + 1.0) / 7.0;
        let center_x = anchor[0] + inward_axis[0] * distance;
        let center_y = anchor[1] + inward_axis[1] * distance;
        for lateral in -2..=2 {
            let x = (center_x + normal[0] * lateral as f64).round() as isize;
            let y = (center_y + normal[1] * lateral as f64).round() as isize;
            if x >= 0 && y >= 0 && (x as usize) < width && (y as usize) < height {
                total += luminance(rgb8, (y as usize * width + x as usize) * 3) as f64;
                count += 1;
            }
        }
    }
    (count > 0).then_some(total / count as f64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::calibration::{
        TableCalibrationV1, TableCoordinateFrame, TABLE_CALIBRATION_SCHEMA_VERSION,
    };
    use crate::ImageSpace;

    fn calibration() -> ValidatedTableCalibrationV1 {
        TableCalibrationV1 {
            schema_version: TABLE_CALIBRATION_SCHEMA_VERSION,
            calibration_revision: 1,
            camera_profile_id: "camera-test".to_owned(),
            rectification_revision: 2,
            image_space: ImageSpace::Raw,
            source_width_px: 100,
            source_height_px: 100,
            coordinate_frame: TableCoordinateFrame::CenterOriginXCorner0To1YCorner0To3,
            table_width_mm: 100.0,
            table_height_mm: 100.0,
            // Synthetic identity mapping: the cloth occupies [-50, 50] mm on
            // each axis in this 100x100 test image.
            pixel_to_table_mm: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        }
        .into_validated()
        .expect("synthetic calibration should be valid")
    }

    fn calibration_context() -> CalibrationContext<'static> {
        CalibrationContext {
            calibration_revision: 1,
            camera_profile_id: "camera-test",
            rectification_revision: 2,
            image_space: ImageSpace::Raw,
            source_width_px: 100,
            source_height_px: 100,
        }
    }

    fn cue(start: [f64; 2], end: [f64; 2]) -> CueGeometry {
        CueGeometry {
            shaft_start_px: start,
            shaft_end_px: end,
            axis_unit: [1.0, 0.0],
            confidence: 0.9,
            observation_age_ms: 0,
            // Deliberately conflict with the table boundary to verify that
            // calibrated table geometry is the primary direction signal.
            signed_direction: Some(SignedCueDirection::TowardEnd),
            proposal_index: None,
        }
    }

    #[test]
    fn table_boundary_orients_tip_toward_playing_surface() {
        let mut geometry = cue([20.0, 0.0], [80.0, 0.0]);
        assert!(orient_cue_geometry_from_playing_surface(
            &mut geometry,
            &calibration(),
            calibration_context(),
        ));
        assert_eq!(
            geometry.signed_direction,
            Some(SignedCueDirection::TowardStart)
        );

        let mut reversed = cue([80.0, 0.0], [20.0, 0.0]);
        reversed.axis_unit = [-1.0, 0.0];
        assert!(orient_cue_geometry_from_playing_surface(
            &mut reversed,
            &calibration(),
            calibration_context(),
        ));
        assert_eq!(
            reversed.signed_direction,
            Some(SignedCueDirection::TowardEnd)
        );
    }

    #[test]
    fn table_boundary_keeps_fallback_when_it_cannot_orient_the_shaft() {
        let mut geometry = cue([10.0, 0.0], [40.0, 0.0]);
        let original_direction = geometry.signed_direction;
        assert!(!orient_cue_geometry_from_playing_surface(
            &mut geometry,
            &calibration(),
            calibration_context(),
        ));
        assert_eq!(geometry.signed_direction, original_direction);

        let mut incompatible = cue([20.0, 0.0], [80.0, 0.0]);
        let original_direction = incompatible.signed_direction;
        let mut context = calibration_context();
        context.calibration_revision = 3;
        assert!(!orient_cue_geometry_from_playing_surface(
            &mut incompatible,
            &calibration(),
            context,
        ));
        assert_eq!(incompatible.signed_direction, original_direction);
    }
}
