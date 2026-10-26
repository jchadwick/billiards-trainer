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
    // Three luminance grids, one i16 delta buffer, and two one-byte masks.
    assert!(monitor.retained_buffer_bytes() <= maximum_grid * 7);
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
