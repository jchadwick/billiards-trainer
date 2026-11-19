#[path = "../src/motion_monitor.rs"]
mod motion_monitor;

use motion_monitor::{
    MotionClassification, MotionMonitor, MotionMonitorConfig, MotionMonitorError, MotionReason,
};
use std::time::Duration;

fn test_config() -> MotionMonitorConfig {
    MotionMonitorConfig {
        max_sample_dimension: 32,
        samples_per_cell_axis: 1,
        luminance_difference_threshold: 12,
        movement_fraction_threshold: 0.001,
        broad_change_fraction: 0.20,
        broad_change_extent_fraction: 0.28,
        broad_change_min_fraction: 0.04,
        maximum_global_brightness_shift: 96,
        reference_settle_dwell: Duration::from_millis(300),
        reference_mask_minimum_neighbors: 2,
    }
}

fn gray_frame(width: u32, height: u32, value: u8) -> Vec<u8> {
    let mut pixels = Vec::with_capacity(width as usize * height as usize * 3);
    for _ in 0..width * height {
        pixels.extend_from_slice(&[value, value, value]);
    }
    pixels
}

fn paint_rect(
    frame: &mut [u8],
    width: u32,
    x: u32,
    y: u32,
    rect_width: u32,
    rect_height: u32,
    value: u8,
) {
    for row in y..(y + rect_height) {
        for column in x..(x + rect_width) {
            let offset = ((row * width + column) * 3) as usize;
            frame[offset..offset + 3].fill(value);
        }
    }
}

#[test]
fn identical_frames_are_quiet_after_initialization() {
    let mut monitor = MotionMonitor::new(test_config()).unwrap();
    let frame = gray_frame(64, 48, 75);

    let first = monitor
        .observe_rgb8(Duration::ZERO, 64, 48, &frame)
        .unwrap();
    let second = monitor
        .observe_rgb8(Duration::from_millis(67), 64, 48, &frame)
        .unwrap();

    assert_eq!(
        first.classification,
        MotionClassification::OccludedOrUncertain
    );
    assert_eq!(first.reason, MotionReason::Initializing);
    assert_eq!(second.classification, MotionClassification::Quiet);
    assert_eq!(second.consecutive_changed_fraction, 0.0);
    assert_eq!(second.reference_changed_fraction, Some(0.0));
}

#[test]
fn verified_reference_requires_and_promotes_a_current_sample() {
    let mut monitor = MotionMonitor::new(test_config()).unwrap();
    assert_eq!(
        monitor.set_current_as_verified_reference(),
        Err(MotionMonitorError::MissingCurrentFrame)
    );

    let frame = gray_frame(64, 48, 75);
    monitor
        .observe_rgb8(Duration::ZERO, 64, 48, &frame)
        .unwrap();
    monitor.set_current_as_verified_reference().unwrap();

    let unchanged = monitor
        .observe_rgb8(Duration::from_millis(67), 64, 48, &frame)
        .unwrap();
    assert_eq!(unchanged.reference_changed_fraction, Some(0.0));
}

#[test]
fn global_lighting_shift_is_compensated() {
    let mut monitor = MotionMonitor::new(test_config()).unwrap();
    let baseline = gray_frame(64, 48, 50);
    let brighter = gray_frame(64, 48, 103);
    monitor
        .observe_rgb8(Duration::ZERO, 64, 48, &baseline)
        .unwrap();

    let assessment = monitor
        .observe_rgb8(Duration::from_millis(67), 64, 48, &brighter)
        .unwrap();

    assert_eq!(assessment.classification, MotionClassification::Quiet);
    assert_eq!(assessment.consecutive_changed_fraction, 0.0);
    assert_eq!(assessment.consecutive_brightness_shift, 53);
}

#[test]
fn slow_ball_sized_displacement_stays_moving_until_it_settles() {
    let mut monitor = MotionMonitor::new(test_config()).unwrap();
    let background = gray_frame(64, 48, 55);
    monitor
        .observe_rgb8(Duration::ZERO, 64, 48, &background)
        .unwrap();

    let mut ball_left = background.clone();
    paint_rect(&mut ball_left, 64, 16, 16, 10, 10, 225);
    let first_move = monitor
        .observe_rgb8(Duration::from_millis(100), 64, 48, &ball_left)
        .unwrap();
    assert_eq!(first_move.classification, MotionClassification::Moving);

    let mut ball_right = background.clone();
    paint_rect(&mut ball_right, 64, 22, 16, 10, 10, 225);
    let continued_move = monitor
        .observe_rgb8(Duration::from_millis(200), 64, 48, &ball_right)
        .unwrap();
    assert_eq!(continued_move.classification, MotionClassification::Moving);
    assert_eq!(continued_move.reason, MotionReason::ReferenceDrift);

    let settled = monitor
        .observe_rgb8(Duration::from_millis(550), 64, 48, &ball_right)
        .unwrap();
    assert_eq!(settled.classification, MotionClassification::Quiet);
    assert!(settled.reference_changed_fraction.unwrap() > 0.0);

    monitor.set_verified_reference(64, 48, &ball_right).unwrap();
    let after_verification = monitor
        .observe_rgb8(Duration::from_millis(617), 64, 48, &ball_right)
        .unwrap();
    assert_eq!(
        after_verification.classification,
        MotionClassification::Quiet
    );
    assert_eq!(after_verification.reference_changed_fraction, Some(0.0));
}

#[test]
fn gradual_reference_mismatch_starts_a_finite_settle_dwell() {
    let mut monitor = MotionMonitor::new(test_config()).unwrap();
    let baseline = gray_frame(64, 48, 100);
    monitor
        .observe_rgb8(Duration::ZERO, 64, 48, &baseline)
        .unwrap();

    // This one sample cell accumulates a larger reference difference through
    // sub-threshold steps. It remains isolated in the spatial mask, so neither
    // the consecutive-motion nor temporal-mask path starts the settle timer.
    let mut first_shift = baseline.clone();
    paint_rect(&mut first_shift, 64, 16, 16, 2, 2, 110);
    let gradual = monitor
        .observe_rgb8(Duration::from_millis(100), 64, 48, &first_shift)
        .unwrap();
    assert_eq!(gradual.classification, MotionClassification::Quiet);

    let mut settled_layout = baseline.clone();
    paint_rect(&mut settled_layout, 64, 16, 16, 2, 2, 120);
    let mismatch = monitor
        .observe_rgb8(Duration::from_millis(200), 64, 48, &settled_layout)
        .unwrap();
    assert_eq!(mismatch.classification, MotionClassification::Moving);
    assert_eq!(mismatch.reason, MotionReason::ReferenceDrift);

    let before_dwell = monitor
        .observe_rgb8(Duration::from_millis(499), 64, 48, &settled_layout)
        .unwrap();
    assert_eq!(before_dwell.classification, MotionClassification::Moving);

    let after_dwell = monitor
        .observe_rgb8(Duration::from_millis(500), 64, 48, &settled_layout)
        .unwrap();
    assert_eq!(after_dwell.classification, MotionClassification::Quiet);
    assert_eq!(after_dwell.reason, MotionReason::Stable);
    assert!(after_dwell.reference_changed_fraction.unwrap() > 0.0);
}

#[test]
fn stable_extreme_lighting_shift_settles_after_finite_dwell() {
    let mut monitor = MotionMonitor::new(test_config()).unwrap();
    let baseline = gray_frame(64, 48, 80);
    monitor
        .observe_rgb8(Duration::ZERO, 64, 48, &baseline)
        .unwrap();

    let lit = gray_frame(64, 48, 200);
    let lighting_transition = monitor
        .observe_rgb8(Duration::from_millis(100), 64, 48, &lit)
        .unwrap();
    assert_eq!(
        lighting_transition.classification,
        MotionClassification::OccludedOrUncertain
    );
    assert_eq!(
        lighting_transition.reason,
        MotionReason::ExtremeBrightnessShift
    );

    let stable_lighting = monitor
        .observe_rgb8(Duration::from_millis(200), 64, 48, &lit)
        .unwrap();
    assert_eq!(stable_lighting.classification, MotionClassification::Moving);
    assert_eq!(stable_lighting.reason, MotionReason::ReferenceDrift);

    let before_dwell = monitor
        .observe_rgb8(Duration::from_millis(399), 64, 48, &lit)
        .unwrap();
    assert_eq!(before_dwell.classification, MotionClassification::Moving);

    let settled = monitor
        .observe_rgb8(Duration::from_millis(400), 64, 48, &lit)
        .unwrap();
    assert_eq!(settled.classification, MotionClassification::Quiet);
    assert_eq!(settled.reason, MotionReason::Stable);
    assert_eq!(settled.consecutive_brightness_shift, 0);
    assert_eq!(settled.reference_brightness_shift, Some(120));
}

#[test]
fn repeated_extreme_lighting_shift_restarts_the_settle_dwell() {
    let mut monitor = MotionMonitor::new(test_config()).unwrap();
    let baseline = gray_frame(64, 48, 10);
    monitor
        .observe_rgb8(Duration::ZERO, 64, 48, &baseline)
        .unwrap();

    let first_lighting_level = gray_frame(64, 48, 250);
    let first_shift = monitor
        .observe_rgb8(Duration::from_millis(100), 64, 48, &first_lighting_level)
        .unwrap();
    assert_eq!(first_shift.reason, MotionReason::ExtremeBrightnessShift);
    let stable_first_shift = monitor
        .observe_rgb8(Duration::from_millis(200), 64, 48, &first_lighting_level)
        .unwrap();
    assert_eq!(
        stable_first_shift.classification,
        MotionClassification::Moving
    );

    // The second change arrives after the original reference mismatch has
    // already outlasted the settle dwell. It remains far from that reference
    // but changes abruptly relative to the prior stable lighting level.
    let second_lighting_level = gray_frame(64, 48, 150);
    let second_shift = monitor
        .observe_rgb8(Duration::from_millis(550), 64, 48, &second_lighting_level)
        .unwrap();
    assert_eq!(second_shift.reason, MotionReason::ExtremeBrightnessShift);

    let before_new_dwell = monitor
        .observe_rgb8(Duration::from_millis(600), 64, 48, &second_lighting_level)
        .unwrap();
    assert_eq!(
        before_new_dwell.classification,
        MotionClassification::Moving
    );
    assert_eq!(before_new_dwell.reason, MotionReason::ReferenceDrift);
    let at_new_dwell = monitor
        .observe_rgb8(Duration::from_millis(900), 64, 48, &second_lighting_level)
        .unwrap();
    assert_eq!(at_new_dwell.classification, MotionClassification::Quiet);
}

#[test]
fn abrupt_return_to_verified_brightness_still_waits_for_settle_dwell() {
    let mut monitor = MotionMonitor::new(test_config()).unwrap();
    let baseline = gray_frame(64, 48, 80);
    monitor
        .observe_rgb8(Duration::ZERO, 64, 48, &baseline)
        .unwrap();

    let shifted = gray_frame(64, 48, 200);
    let lighting_on = monitor
        .observe_rgb8(Duration::from_millis(100), 64, 48, &shifted)
        .unwrap();
    assert_eq!(lighting_on.reason, MotionReason::ExtremeBrightnessShift);

    let lighting_off = monitor
        .observe_rgb8(Duration::from_millis(150), 64, 48, &baseline)
        .unwrap();
    assert_eq!(lighting_off.reason, MotionReason::ExtremeBrightnessShift);

    let before_dwell = monitor
        .observe_rgb8(Duration::from_millis(200), 64, 48, &baseline)
        .unwrap();
    assert_eq!(before_dwell.classification, MotionClassification::Moving);
    assert_eq!(before_dwell.reason, MotionReason::ReferenceDrift);

    let last_unsettled = monitor
        .observe_rgb8(Duration::from_millis(449), 64, 48, &baseline)
        .unwrap();
    assert_eq!(last_unsettled.classification, MotionClassification::Moving);
    let settled = monitor
        .observe_rgb8(Duration::from_millis(450), 64, 48, &baseline)
        .unwrap();
    assert_eq!(settled.classification, MotionClassification::Quiet);
    assert_eq!(settled.reason, MotionReason::Stable);
}

#[test]
fn broad_hand_or_occlusion_change_is_uncertain() {
    let mut monitor = MotionMonitor::new(test_config()).unwrap();
    let baseline = gray_frame(64, 48, 70);
    monitor
        .observe_rgb8(Duration::ZERO, 64, 48, &baseline)
        .unwrap();
    let mut occluded = baseline.clone();
    paint_rect(&mut occluded, 64, 4, 6, 48, 30, 180);

    let assessment = monitor
        .observe_rgb8(Duration::from_millis(67), 64, 48, &occluded)
        .unwrap();

    assert_eq!(
        assessment.classification,
        MotionClassification::OccludedOrUncertain
    );
    assert_eq!(assessment.reason, MotionReason::BroadChange);
    assert!(assessment.consecutive_changed_fraction >= test_config().broad_change_fraction);
}

#[test]
fn rejects_invalid_configuration_frame_sizes_and_timestamp_regression() {
    let mut config = test_config();
    config.max_sample_dimension = 4;
    assert!(matches!(
        MotionMonitor::new(config),
        Err(MotionMonitorError::InvalidConfig(_))
    ));

    let mut monitor = MotionMonitor::new(test_config()).unwrap();
    assert!(matches!(
        monitor.observe_rgb8(Duration::ZERO, 64, 48, &[0; 3]),
        Err(MotionMonitorError::InvalidFrameLength { .. })
    ));
    assert!(matches!(
        monitor.observe_rgb8(Duration::ZERO, 0, 48, &[]),
        Err(MotionMonitorError::InvalidDimensions { .. })
    ));
    let frame = gray_frame(64, 48, 40);
    monitor
        .observe_rgb8(Duration::from_millis(10), 64, 48, &frame)
        .unwrap();
    assert!(matches!(
        monitor.observe_rgb8(Duration::from_millis(9), 64, 48, &frame),
        Err(MotionMonitorError::TimestampRegression { .. })
    ));
}

#[test]
fn sample_grid_and_retained_buffers_remain_bounded() {
    let mut config = test_config();
    config.max_sample_dimension = 96;
    let mut monitor = MotionMonitor::new(config).unwrap();
    let frame = gray_frame(1280, 720, 90);
    let assessment = monitor
        .observe_rgb8(Duration::ZERO, 1280, 720, &frame)
        .unwrap();

    assert!(assessment.sample_width <= 96);
    assert!(assessment.sample_height <= 96);
    let maximum_grid = 96usize * 96;
    // Three luminance grids, one i16 delta buffer, and three one-byte masks.
    assert!(monitor.retained_buffer_bytes() <= maximum_grid * 8);
}

#[test]
fn isolated_reference_mask_flicker_does_not_prevent_quiet_recovery() {
    let mut monitor = MotionMonitor::new(test_config()).unwrap();
    let reference = gray_frame(64, 48, 100);
    monitor
        .observe_rgb8(Duration::ZERO, 64, 48, &reference)
        .unwrap();

    let mut latest = None;
    for sample in 1..=6_u32 {
        let mut frame = gray_frame(64, 48, 111);
        // Each changed sample differs from the reference enough to toggle the
        // reference mask, but by only two levels from the preceding frame.
        // The sparse positions model low-level camera/compression flicker.
        for y in (1..24_u32).step_by(3) {
            for x in (1..32_u32).step_by(3) {
                if (x / 3 + y / 3) % 2 == sample % 2 {
                    let offset = (((y * 2 + 1) * 64 + (x * 2 + 1)) * 3) as usize;
                    frame[offset..offset + 3].fill(113);
                }
            }
        }
        latest = Some(
            monitor
                .observe_rgb8(
                    Duration::from_millis(u64::from(sample) * 100),
                    64,
                    48,
                    &frame,
                )
                .unwrap(),
        );
    }

    let settled = latest.unwrap();
    assert_eq!(settled.consecutive_changed_fraction, 0.0);
    assert_eq!(settled.classification, MotionClassification::Quiet);
}

#[test]
fn stable_reference_boundary_flicker_does_not_rearm_shot_motion() {
    let mut monitor = MotionMonitor::new(test_config()).unwrap();
    let reference = gray_frame(64, 48, 100);
    monitor
        .observe_rgb8(Duration::ZERO, 64, 48, &reference)
        .unwrap();

    let mut latest = None;
    for sample in 1..=6_u64 {
        // The broad changed region represents a settled object layout that
        // differs from the last verified reference. The background stays at
        // its reference brightness so this is not mistaken for a global
        // exposure shift.
        let mut frame = gray_frame(64, 48, 100);
        for sample_y in 6..15 {
            for sample_x in 6..15 {
                let x = sample_x * 2 + 1;
                let y = sample_y * 2 + 1;
                let offset = ((y * 64 + x) * 3) as usize;
                frame[offset..offset + 3].fill(130);
            }
        }

        // One boundary sample toggles just across the reference-difference
        // threshold (100 + 12) while changing only two luminance levels from
        // the prior frame. Its diagonal stable neighbor keeps it in the
        // per-frame filtered mask, but the isolated temporal-mask delta must
        // not count as scene motion.
        let flicker_value = if sample % 2 == 1 { 113 } else { 111 };
        let flicker_x = 5 * 2 + 1;
        let flicker_y = 5 * 2 + 1;
        let flicker_offset = ((flicker_y * 64 + flicker_x) * 3) as usize;
        frame[flicker_offset..flicker_offset + 3].fill(flicker_value);

        let assessment = monitor
            .observe_rgb8(Duration::from_millis(sample * 100), 64, 48, &frame)
            .unwrap();
        if sample > 1 {
            assert_eq!(assessment.consecutive_changed_fraction, 0.0);
            assert_eq!(assessment.reference_change_delta_fraction, Some(0.0));
        }
        latest = Some(assessment);
    }

    let settled = latest.unwrap();
    assert_eq!(settled.classification, MotionClassification::Quiet);
    assert_eq!(settled.reason, MotionReason::Stable);
    assert!(settled.reference_changed_fraction.unwrap() > 0.0);
}

#[test]
fn connected_threshold_band_flicker_does_not_rearm_shot_motion() {
    let mut monitor = MotionMonitor::new(test_config()).unwrap();
    let reference = gray_frame(64, 48, 100);
    monitor
        .observe_rgb8(Duration::ZERO, 64, 48, &reference)
        .unwrap();

    let mut latest = None;
    for sample in 1..=6_u64 {
        let mut frame = reference.clone();
        // A stable changed region models an object layout that differs from
        // the last verified reference without changing between these frames.
        for sample_y in 6..15 {
            for sample_x in 6..15 {
                paint_rect(&mut frame, 64, sample_x * 2, sample_y * 2, 2, 2, 130);
            }
        }

        // A connected patch of camera noise alternates between 5 and 15
        // luminance levels above the reference. The ten-level frame-to-frame
        // change stays below the direct threshold of 12, while the patch
        // repeatedly crosses the reference threshold and has enough spatial
        // support to evade an isolated-sample filter. Per-sample hysteresis
        // derived from its own temporal variation should admit the patch once
        // and keep its membership stable until it returns close to the
        // reference.
        let flicker_value = if sample % 2 == 1 { 115 } else { 105 };
        for sample_y in 7..15 {
            for sample_x in 20..28 {
                paint_rect(
                    &mut frame,
                    64,
                    sample_x * 2,
                    sample_y * 2,
                    2,
                    2,
                    flicker_value,
                );
            }
        }

        let assessment = monitor
            .observe_rgb8(Duration::from_millis(sample * 100), 64, 48, &frame)
            .unwrap();
        if sample > 1 {
            assert_eq!(assessment.consecutive_changed_fraction, 0.0);
            assert_eq!(assessment.reference_change_delta_fraction, Some(0.0));
        }
        latest = Some(assessment);
    }

    let settled = latest.unwrap();
    assert_eq!(settled.classification, MotionClassification::Quiet);
    assert_eq!(settled.reason, MotionReason::Stable);
}

#[test]
fn resolution_change_is_uncertain_until_reference_is_reverified() {
    let mut monitor = MotionMonitor::new(test_config()).unwrap();
    let old_frame = gray_frame(64, 48, 60);
    monitor
        .observe_rgb8(Duration::ZERO, 64, 48, &old_frame)
        .unwrap();

    let new_frame = gray_frame(80, 48, 60);
    let resolution_change = monitor
        .observe_rgb8(Duration::from_millis(67), 80, 48, &new_frame)
        .unwrap();
    assert_eq!(
        resolution_change.classification,
        MotionClassification::OccludedOrUncertain
    );
    assert_eq!(resolution_change.reason, MotionReason::ResolutionChanged);

    let still_unverified = monitor
        .observe_rgb8(Duration::from_millis(134), 80, 48, &new_frame)
        .unwrap();
    assert_eq!(
        still_unverified.classification,
        MotionClassification::OccludedOrUncertain
    );
    assert_eq!(still_unverified.reference_changed_fraction, None);

    monitor.set_verified_reference(80, 48, &new_frame).unwrap();
    let verified = monitor
        .observe_rgb8(Duration::from_millis(201), 80, 48, &new_frame)
        .unwrap();
    assert_eq!(verified.classification, MotionClassification::Quiet);
    assert_eq!(verified.reference_changed_fraction, Some(0.0));

    assert_eq!(monitor.config().max_sample_dimension, 32);
    monitor.reset();
    let reset = monitor
        .observe_rgb8(Duration::from_millis(202), 80, 48, &new_frame)
        .unwrap();
    assert_eq!(reset.reason, MotionReason::Initializing);
}
