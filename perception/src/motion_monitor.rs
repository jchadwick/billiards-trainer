//! Lightweight, bounded motion assessment for RGB8 billiards frames.
//!
//! The monitor reduces each input to a small luminance grid, then compares it
//! with both the immediately preceding frame and an explicitly verified
//! stationary reference. A robust median brightness offset removes global
//! exposure changes before local differences are scored. The stable reference
//! is only replaced when the caller confirms a settled, verified layout.

use std::fmt;
use std::time::Duration;

/// Configures the bounded sampling and motion classification behavior.
#[derive(Debug, Clone, PartialEq)]
pub struct MotionMonitorConfig {
    /// Maximum width or height of the luminance grid. The source aspect ratio
    /// is preserved, so the largest retained grid has at most this squared
    /// number of samples.
    pub max_sample_dimension: u32,
    /// Number of evenly spaced samples to take along each axis inside every
    /// downsampled cell. More samples improve anti-aliasing at bounded cost.
    pub samples_per_cell_axis: u8,
    /// Minimum compensated luminance difference for a sample to count as
    /// changed.
    pub luminance_difference_threshold: u8,
    /// A localized changed fraction or changed-mask movement above this value
    /// indicates motion. Keep this small enough to detect a pool ball.
    pub movement_fraction_threshold: f64,
    /// A changed fraction at or above this value is considered broad scene
    /// change or occlusion and is reported as uncertain.
    pub broad_change_fraction: f64,
    /// Changes spanning this fraction of the grid's bounding rectangle are
    /// also considered broad when they affect enough samples.
    pub broad_change_extent_fraction: f64,
    /// Minimum changed fraction before the broad bounding-rectangle test is
    /// applied. This avoids treating one distant noisy pixel as occlusion.
    pub broad_change_min_fraction: f64,
    /// A global brightness shift larger than this value is too extreme to
    /// trust as a simple lighting change.
    pub maximum_global_brightness_shift: u8,
    /// Once the current scene differs from the verified reference, it can be
    /// called quiet only after its localized difference mask remains stable
    /// for this long. This allows recovery after a ball has stopped at a new
    /// location without hiding slow movement.
    pub reference_settle_dwell: Duration,
    /// Minimum changed neighbors in a 3-by-3 sample-grid neighborhood for a
    /// reference-difference sample to participate in temporal drift tracking.
    /// This suppresses isolated camera/compression noise while retaining
    /// spatially coherent ball and occlusion boundaries.
    pub reference_mask_minimum_neighbors: u8,
}

impl Default for MotionMonitorConfig {
    fn default() -> Self {
        Self {
            max_sample_dimension: 160,
            samples_per_cell_axis: 3,
            luminance_difference_threshold: 14,
            movement_fraction_threshold: 0.0005,
            broad_change_fraction: 0.20,
            broad_change_extent_fraction: 0.28,
            broad_change_min_fraction: 0.04,
            maximum_global_brightness_shift: 96,
            reference_settle_dwell: Duration::from_millis(350),
            reference_mask_minimum_neighbors: 2,
        }
    }
}

impl MotionMonitorConfig {
    /// Validate settings that affect allocation bounds and classification.
    pub fn validate(&self) -> Result<(), MotionMonitorError> {
        if !(8..=512).contains(&self.max_sample_dimension) {
            return Err(MotionMonitorError::InvalidConfig(
                "max_sample_dimension must be between 8 and 512".into(),
            ));
        }
        if !(1..=8).contains(&self.samples_per_cell_axis) {
            return Err(MotionMonitorError::InvalidConfig(
                "samples_per_cell_axis must be between 1 and 8".into(),
            ));
        }
        if self.luminance_difference_threshold == 0 {
            return Err(MotionMonitorError::InvalidConfig(
                "luminance_difference_threshold must be positive".into(),
            ));
        }
        if !(1..=9).contains(&self.reference_mask_minimum_neighbors) {
            return Err(MotionMonitorError::InvalidConfig(
                "reference_mask_minimum_neighbors must be between 1 and 9".into(),
            ));
        }
        for (name, value) in [
            (
                "movement_fraction_threshold",
                self.movement_fraction_threshold,
            ),
            ("broad_change_fraction", self.broad_change_fraction),
            (
                "broad_change_extent_fraction",
                self.broad_change_extent_fraction,
            ),
            ("broad_change_min_fraction", self.broad_change_min_fraction),
        ] {
            if !value.is_finite() || !(0.0..=1.0).contains(&value) {
                return Err(MotionMonitorError::InvalidConfig(format!(
                    "{name} must be finite and between 0 and 1"
                )));
            }
        }
        if self.movement_fraction_threshold >= self.broad_change_fraction {
            return Err(MotionMonitorError::InvalidConfig(
                "movement threshold must be below broad-change threshold".into(),
            ));
        }
        if self.movement_fraction_threshold == 0.0 || self.broad_change_fraction == 0.0 {
            return Err(MotionMonitorError::InvalidConfig(
                "movement and broad-change fractions must be positive".into(),
            ));
        }
        Ok(())
    }
}

/// Coarse assessment suitable for mapping to the phase scheduler's evidence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MotionClassification {
    /// No detectable motion is present and the changed layout has been stable
    /// for the configured dwell.
    Quiet,
    /// Localized movement is visible between frames or is still changing
    /// relative to the last verified reference.
    Moving,
    /// Broad changes, an extreme brightness shift, or insufficient context
    /// make a quiet/moving decision unsafe.
    OccludedOrUncertain,
}

/// Why the monitor returned its classification, with rates useful for logs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MotionAssessment {
    pub classification: MotionClassification,
    pub reason: MotionReason,
    pub consecutive_changed_fraction: f64,
    pub reference_changed_fraction: Option<f64>,
    pub reference_change_delta_fraction: Option<f64>,
    pub consecutive_brightness_shift: i16,
    pub reference_brightness_shift: Option<i16>,
    pub sample_width: u32,
    pub sample_height: u32,
}

/// Classification detail for diagnostics and deterministic evaluation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MotionReason {
    Initializing,
    Stable,
    ConsecutiveChange,
    ReferenceDrift,
    BroadChange,
    ExtremeBrightnessShift,
    ResolutionChanged,
}

/// Errors raised for invalid configuration, frame data, or time ordering.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MotionMonitorError {
    InvalidConfig(String),
    InvalidDimensions {
        width: u32,
        height: u32,
    },
    InvalidFrameLength {
        expected: usize,
        actual: usize,
    },
    MissingCurrentFrame,
    TimestampRegression {
        previous: Duration,
        current: Duration,
    },
}

impl fmt::Display for MotionMonitorError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(message) => {
                write!(formatter, "invalid motion monitor configuration: {message}")
            }
            Self::InvalidDimensions { width, height } => {
                write!(formatter, "invalid RGB8 frame dimensions {width}x{height}")
            }
            Self::InvalidFrameLength { expected, actual } => {
                write!(
                    formatter,
                    "RGB8 frame has {actual} bytes; expected {expected}"
                )
            }
            Self::MissingCurrentFrame => {
                formatter.write_str("motion reference cannot be promoted before a frame is sampled")
            }
            Self::TimestampRegression { previous, current } => write!(
                formatter,
                "motion observation time regressed from {previous:?} to {current:?}"
            ),
        }
    }
}

impl std::error::Error for MotionMonitorError {}

#[derive(Debug, Clone, Copy, Default)]
struct DifferenceMetrics {
    fraction: f64,
    brightness_shift: i16,
    min_x: u32,
    min_y: u32,
    max_x: u32,
    max_y: u32,
    changed_count: usize,
}

impl DifferenceMetrics {
    fn extent_fraction(self, width: usize, height: usize) -> f64 {
        if self.changed_count == 0 || width == 0 || height == 0 {
            return 0.0;
        }
        let extent_width = (self.max_x - self.min_x + 1) as f64;
        let extent_height = (self.max_y - self.min_y + 1) as f64;
        (extent_width * extent_height) / (width as f64 * height as f64)
    }
}

/// A bounded monitor for the low-cost motion path in the perception process.
pub struct MotionMonitor {
    config: MotionMonitorConfig,
    source_width: Option<u32>,
    source_height: Option<u32>,
    sample_width: usize,
    sample_height: usize,
    previous: Vec<u8>,
    reference: Vec<u8>,
    current: Vec<u8>,
    delta_scratch: Vec<i16>,
    reference_mask: Vec<u8>,
    filtered_reference_mask: Vec<u8>,
    previous_reference_mask: Vec<u8>,
    last_observed_at: Option<Duration>,
    last_reference_drift_at: Option<Duration>,
    reference_mask_initialized: bool,
    reference_valid: bool,
}

impl MotionMonitor {
    pub fn new(config: MotionMonitorConfig) -> Result<Self, MotionMonitorError> {
        config.validate()?;
        Ok(Self {
            config,
            source_width: None,
            source_height: None,
            sample_width: 0,
            sample_height: 0,
            previous: Vec::new(),
            reference: Vec::new(),
            current: Vec::new(),
            delta_scratch: Vec::new(),
            reference_mask: Vec::new(),
            filtered_reference_mask: Vec::new(),
            previous_reference_mask: Vec::new(),
            last_observed_at: None,
            last_reference_drift_at: None,
            reference_mask_initialized: false,
            reference_valid: false,
        })
    }

    pub fn config(&self) -> &MotionMonitorConfig {
        &self.config
    }

    /// Compare a frame with its predecessor and verified reference.
    ///
    /// `observed_at` must be monotonic within the source session. The first
    /// valid frame seeds both the preceding and reference images, allowing the
    /// scheduler to begin its quiet dwell while it obtains the first layout.
    pub fn observe_rgb8(
        &mut self,
        observed_at: Duration,
        width: u32,
        height: u32,
        rgb8: &[u8],
    ) -> Result<MotionAssessment, MotionMonitorError> {
        self.validate_frame(width, height, rgb8)?;
        if let Some(previous_time) = self.last_observed_at {
            if observed_at < previous_time {
                return Err(MotionMonitorError::TimestampRegression {
                    previous: previous_time,
                    current: observed_at,
                });
            }
        }

        let dimension_changed = self.source_width.is_some_and(|old| old != width)
            || self.source_height.is_some_and(|old| old != height);
        self.resize_sampling_grid(width, height);
        self.sample_luminance(width, height, rgb8);

        if self.previous.is_empty() || dimension_changed {
            self.previous.clone_from(&self.current);
            if dimension_changed {
                self.reference.clear();
                self.reference_valid = false;
            } else if !self.reference_valid {
                self.reference.clone_from(&self.current);
                self.reference_valid = true;
            }
            self.source_width = Some(width);
            self.source_height = Some(height);
            self.last_observed_at = Some(observed_at);
            self.last_reference_drift_at = None;
            self.reference_mask_initialized = false;
            return Ok(self.assessment(
                MotionClassification::OccludedOrUncertain,
                if dimension_changed {
                    MotionReason::ResolutionChanged
                } else {
                    MotionReason::Initializing
                },
                DifferenceMetrics::default(),
                None,
                None,
            ));
        }

        let count = self.current.len();
        self.delta_scratch.resize(count, 0);
        self.reference_mask.resize(count, 0);
        self.previous_reference_mask.resize(count, 0);

        let consecutive = compare_grids(
            &self.current,
            &self.previous,
            self.sample_width,
            self.sample_height,
            self.config.luminance_difference_threshold,
            &mut self.delta_scratch,
        );
        let (reference, reference_delta_fraction) = if self.reference_valid {
            let reference = compare_grids(
                &self.current,
                &self.reference,
                self.sample_width,
                self.sample_height,
                self.config.luminance_difference_threshold,
                &mut self.delta_scratch,
            );
            fill_difference_mask(
                &self.current,
                &self.reference,
                self.config.luminance_difference_threshold,
                reference.brightness_shift,
                &mut self.reference_mask,
            );
            filter_reference_mask(
                &self.reference_mask,
                &mut self.filtered_reference_mask,
                self.sample_width,
                self.sample_height,
                self.config.reference_mask_minimum_neighbors,
            );
            let delta_fraction = if self.reference_mask_initialized {
                let changed = self
                    .filtered_reference_mask
                    .iter()
                    .zip(&self.previous_reference_mask)
                    .filter(|(current, previous)| current != previous)
                    .count();
                Some(changed as f64 / count as f64)
            } else {
                self.reference_mask_initialized = true;
                None
            };
            std::mem::swap(
                &mut self.filtered_reference_mask,
                &mut self.previous_reference_mask,
            );
            (Some(reference), delta_fraction)
        } else {
            (None, None)
        };

        // Broad frame-to-frame changes usually indicate a hand, lighting
        // artifact, or a large event. A large but stationary difference from
        // the old verified reference may instead be a settled new layout, so
        // it must pass through the same drift dwell and detector consensus.
        let is_broad = is_broad_change(
            consecutive,
            self.sample_width,
            self.sample_height,
            &self.config,
        );
        let extreme_brightness = consecutive.brightness_shift.unsigned_abs()
            > u16::from(self.config.maximum_global_brightness_shift)
            || reference.is_some_and(|metrics| {
                metrics.brightness_shift.unsigned_abs()
                    > u16::from(self.config.maximum_global_brightness_shift)
            });
        let changed_since_reference = reference
            .is_some_and(|metrics| metrics.fraction >= self.config.movement_fraction_threshold);
        let reference_mask_moved = reference_delta_fraction
            .is_some_and(|fraction| fraction >= self.config.movement_fraction_threshold);
        let consecutive_motion = consecutive.fraction >= self.config.movement_fraction_threshold;
        let layout_drift = changed_since_reference && reference_mask_moved;

        let (classification, reason) = if is_broad {
            self.last_reference_drift_at = Some(observed_at);
            (
                MotionClassification::OccludedOrUncertain,
                MotionReason::BroadChange,
            )
        } else if extreme_brightness {
            (
                MotionClassification::OccludedOrUncertain,
                MotionReason::ExtremeBrightnessShift,
            )
        } else if consecutive_motion || layout_drift {
            self.last_reference_drift_at = Some(observed_at);
            (
                MotionClassification::Moving,
                if layout_drift {
                    MotionReason::ReferenceDrift
                } else {
                    MotionReason::ConsecutiveChange
                },
            )
        } else if !self.reference_valid {
            (
                MotionClassification::OccludedOrUncertain,
                MotionReason::Initializing,
            )
        } else if changed_since_reference {
            let drift_age = self
                .last_reference_drift_at
                .map(|last| observed_at.saturating_sub(last))
                .unwrap_or_default();
            if drift_age < self.config.reference_settle_dwell {
                (MotionClassification::Moving, MotionReason::ReferenceDrift)
            } else {
                (MotionClassification::Quiet, MotionReason::Stable)
            }
        } else {
            self.last_reference_drift_at = None;
            (MotionClassification::Quiet, MotionReason::Stable)
        };

        self.previous.clone_from(&self.current);
        self.source_width = Some(width);
        self.source_height = Some(height);
        self.last_observed_at = Some(observed_at);
        Ok(self.assessment(
            classification,
            reason,
            consecutive,
            reference,
            reference_delta_fraction,
        ))
    }

    /// Replace the stationary comparison image after the caller has accepted
    /// a verified layout. This does not alter observation timestamps.
    pub fn set_verified_reference(
        &mut self,
        width: u32,
        height: u32,
        rgb8: &[u8],
    ) -> Result<(), MotionMonitorError> {
        self.validate_frame(width, height, rgb8)?;
        if self.source_width.is_some_and(|old| old != width)
            || self.source_height.is_some_and(|old| old != height)
        {
            self.previous.clear();
            self.reference_mask_initialized = false;
        }
        self.resize_sampling_grid(width, height);
        self.sample_luminance(width, height, rgb8);
        self.reference.clone_from(&self.current);
        self.source_width = Some(width);
        self.source_height = Some(height);
        self.last_reference_drift_at = None;
        self.reference_mask_initialized = false;
        self.reference_valid = true;
        Ok(())
    }

    /// Promote the latest lightweight sample to the verified stationary
    /// reference after the scheduler accepts a consistent full-table layout.
    /// This avoids retaining or copying a full-resolution source frame.
    pub fn set_current_as_verified_reference(&mut self) -> Result<(), MotionMonitorError> {
        if self.current.is_empty() || self.source_width.is_none() || self.source_height.is_none() {
            return Err(MotionMonitorError::MissingCurrentFrame);
        }
        self.reference.clone_from(&self.current);
        self.last_reference_drift_at = None;
        self.reference_mask_initialized = false;
        self.reference_valid = true;
        Ok(())
    }

    /// Drop both temporal comparisons after a source or calibration reset.
    pub fn reset(&mut self) {
        self.source_width = None;
        self.source_height = None;
        self.sample_width = 0;
        self.sample_height = 0;
        self.previous.clear();
        self.reference.clear();
        self.current.clear();
        self.delta_scratch.clear();
        self.reference_mask.clear();
        self.filtered_reference_mask.clear();
        self.previous_reference_mask.clear();
        self.last_observed_at = None;
        self.last_reference_drift_at = None;
        self.reference_mask_initialized = false;
        self.reference_valid = false;
    }

    /// Maximum number of bytes retained across all monitor-owned buffers.
    /// This is primarily useful for telemetry and bounded-memory tests.
    pub fn retained_buffer_bytes(&self) -> usize {
        self.previous.capacity()
            + self.reference.capacity()
            + self.current.capacity()
            + self.delta_scratch.capacity() * std::mem::size_of::<i16>()
            + self.reference_mask.capacity()
            + self.filtered_reference_mask.capacity()
            + self.previous_reference_mask.capacity()
    }

    fn validate_frame(
        &self,
        width: u32,
        height: u32,
        rgb8: &[u8],
    ) -> Result<(), MotionMonitorError> {
        if width == 0 || height == 0 {
            return Err(MotionMonitorError::InvalidDimensions { width, height });
        }
        let expected = usize::try_from(width)
            .ok()
            .and_then(|width| {
                usize::try_from(height)
                    .ok()
                    .and_then(|height| width.checked_mul(height))
            })
            .and_then(|pixels| pixels.checked_mul(3))
            .ok_or(MotionMonitorError::InvalidDimensions { width, height })?;
        if rgb8.len() != expected {
            return Err(MotionMonitorError::InvalidFrameLength {
                expected,
                actual: rgb8.len(),
            });
        }
        Ok(())
    }

    fn resize_sampling_grid(&mut self, source_width: u32, source_height: u32) {
        let max_dimension = self.config.max_sample_dimension as f64;
        let scale = (max_dimension / f64::from(source_width.max(source_height))).min(1.0);
        let sample_width = ((f64::from(source_width) * scale).round() as usize).max(1);
        let sample_height = ((f64::from(source_height) * scale).round() as usize).max(1);
        if sample_width != self.sample_width || sample_height != self.sample_height {
            self.previous.clear();
            self.reference.clear();
            self.reference_mask_initialized = false;
            self.reference_valid = false;
        }
        self.sample_width = sample_width;
        self.sample_height = sample_height;
        let sample_count = self.sample_width * self.sample_height;
        self.current.resize(sample_count, 0);
        self.delta_scratch.resize(sample_count, 0);
        self.reference_mask.resize(sample_count, 0);
        self.filtered_reference_mask.resize(sample_count, 0);
        self.previous_reference_mask.resize(sample_count, 0);
    }

    fn sample_luminance(&mut self, width: u32, height: u32, rgb8: &[u8]) {
        let axis_samples = usize::from(self.config.samples_per_cell_axis);
        let width = width as usize;
        let height = height as usize;
        for sample_y in 0..self.sample_height {
            let start_y = sample_y * height / self.sample_height;
            let end_y = ((sample_y + 1) * height / self.sample_height).max(start_y + 1);
            for sample_x in 0..self.sample_width {
                let start_x = sample_x * width / self.sample_width;
                let end_x = ((sample_x + 1) * width / self.sample_width).max(start_x + 1);
                let mut luminance_sum = 0u64;
                let mut samples = 0u64;
                for sample_row in 0..axis_samples {
                    let source_y = stratified_coordinate(start_y, end_y, sample_row, axis_samples);
                    for sample_column in 0..axis_samples {
                        let source_x =
                            stratified_coordinate(start_x, end_x, sample_column, axis_samples);
                        let offset = (source_y * width + source_x) * 3;
                        let red = u32::from(rgb8[offset]);
                        let green = u32::from(rgb8[offset + 1]);
                        let blue = u32::from(rgb8[offset + 2]);
                        // Integer BT.601 luminance keeps this path deterministic
                        // and avoids floating-point work for every source pixel.
                        luminance_sum += u64::from((77 * red + 150 * green + 29 * blue) >> 8);
                        samples += 1;
                    }
                }
                self.current[sample_y * self.sample_width + sample_x] =
                    (luminance_sum / samples.max(1)) as u8;
            }
        }
    }

    fn assessment(
        &self,
        classification: MotionClassification,
        reason: MotionReason,
        consecutive: DifferenceMetrics,
        reference: Option<DifferenceMetrics>,
        reference_delta_fraction: Option<f64>,
    ) -> MotionAssessment {
        MotionAssessment {
            classification,
            reason,
            consecutive_changed_fraction: consecutive.fraction,
            reference_changed_fraction: reference.map(|metrics| metrics.fraction),
            reference_change_delta_fraction: reference_delta_fraction,
            consecutive_brightness_shift: consecutive.brightness_shift,
            reference_brightness_shift: reference.map(|metrics| metrics.brightness_shift),
            sample_width: self.sample_width as u32,
            sample_height: self.sample_height as u32,
        }
    }
}

fn stratified_coordinate(start: usize, end: usize, sample: usize, sample_count: usize) -> usize {
    let span = end.saturating_sub(start).max(1);
    let position = ((2 * sample + 1) * span) / (2 * sample_count);
    start + position.min(span - 1)
}

fn compare_grids(
    current: &[u8],
    baseline: &[u8],
    width: usize,
    height: usize,
    threshold: u8,
    delta_scratch: &mut [i16],
) -> DifferenceMetrics {
    debug_assert_eq!(current.len(), baseline.len());
    debug_assert_eq!(current.len(), delta_scratch.len());
    for ((delta, current), baseline) in delta_scratch.iter_mut().zip(current).zip(baseline) {
        *delta = i16::from(*current) - i16::from(*baseline);
    }
    let middle = delta_scratch.len() / 2;
    let brightness_shift = if delta_scratch.is_empty() {
        0
    } else {
        let (_, median, _) = delta_scratch.select_nth_unstable(middle);
        *median
    };
    let mut metrics = DifferenceMetrics {
        min_x: width as u32,
        min_y: height as u32,
        ..DifferenceMetrics::default()
    };
    let threshold = i16::from(threshold);
    for (index, delta) in delta_scratch.iter().enumerate() {
        if (*delta - brightness_shift).abs() >= threshold {
            let x = index % width;
            let y = index / width;
            metrics.changed_count += 1;
            metrics.min_x = metrics.min_x.min(x as u32);
            metrics.min_y = metrics.min_y.min(y as u32);
            metrics.max_x = metrics.max_x.max(x as u32);
            metrics.max_y = metrics.max_y.max(y as u32);
        }
    }
    metrics.fraction = metrics.changed_count as f64 / current.len().max(1) as f64;
    metrics.brightness_shift = brightness_shift;
    metrics
}

fn fill_difference_mask(
    current: &[u8],
    reference: &[u8],
    threshold: u8,
    brightness_shift: i16,
    output: &mut [u8],
) {
    let threshold = i16::from(threshold);
    for ((value, reference), mask) in current.iter().zip(reference).zip(output) {
        *mask = u8::from(
            (i16::from(*value) - i16::from(*reference) - brightness_shift).abs() >= threshold,
        );
    }
}

/// Remove isolated reference-difference samples before comparing successive
/// masks. Inter-frame comparisons remain unfiltered and continue to detect
/// actual movement directly; this spatial consistency check only prevents
/// sensor/compression speckles from resetting quiet continuity indefinitely.
fn filter_reference_mask(
    input: &[u8],
    output: &mut [u8],
    width: usize,
    height: usize,
    minimum_neighbors: u8,
) {
    debug_assert_eq!(input.len(), width * height);
    debug_assert_eq!(output.len(), input.len());
    for y in 0..height {
        for x in 0..width {
            let mut changed_neighbors = 0_u8;
            let y_start = y.saturating_sub(1);
            let y_end = (y + 1).min(height.saturating_sub(1));
            let x_start = x.saturating_sub(1);
            let x_end = (x + 1).min(width.saturating_sub(1));
            for neighbor_y in y_start..=y_end {
                let row_offset = neighbor_y * width;
                for neighbor_x in x_start..=x_end {
                    changed_neighbors += input[row_offset + neighbor_x];
                }
            }
            output[y * width + x] = u8::from(changed_neighbors >= minimum_neighbors);
        }
    }
}

fn is_broad_change(
    consecutive: DifferenceMetrics,
    width: usize,
    height: usize,
    config: &MotionMonitorConfig,
) -> bool {
    consecutive.fraction >= config.broad_change_fraction
        || (consecutive.fraction >= config.broad_change_min_fraction
            && consecutive.extent_fraction(width, height) >= config.broad_change_extent_fraction)
}
