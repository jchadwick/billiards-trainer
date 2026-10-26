use billiards_perception::cue_geometry::{
    estimate_cue_geometry, CueGeometryConfig, CueGeometryError, CueProposal, SignedCueDirection,
};
use std::time::Duration;

const WIDTH: u32 = 220;
const HEIGHT: u32 = 140;

fn test_config() -> CueGeometryConfig {
    CueGeometryConfig {
        max_width_px: WIDTH,
        max_height_px: HEIGHT,
        max_pixels: (WIDTH * HEIGHT) as usize,
        max_edge_samples: 12_000,
        max_point_visits: 300_000,
        max_temporary_bytes: 12_000 * std::mem::size_of::<[f64; 2]>() * 2,
        ..CueGeometryConfig::default()
    }
}

fn felt_image() -> Vec<u8> {
    let mut pixels = vec![0_u8; (WIDTH * HEIGHT * 3) as usize];
    for pixel in pixels.as_chunks_mut::<3>().0 {
        pixel.copy_from_slice(&[30, 100, 30]);
    }
    pixels
}

fn draw_line(pixels: &mut [u8], from: [f64; 2], to: [f64; 2], half_width: f64, color: [u8; 3]) {
    let dx = to[0] - from[0];
    let dy = to[1] - from[1];
    let length_squared = dx * dx + dy * dy;
    for y in 0..HEIGHT as usize {
        for x in 0..WIDTH as usize {
            let t = (((x as f64 - from[0]) * dx + (y as f64 - from[1]) * dy) / length_squared)
                .clamp(0.0, 1.0);
            let nearest = [from[0] + t * dx, from[1] + t * dy];
            if (x as f64 - nearest[0]).hypot(y as f64 - nearest[1]) <= half_width {
                let offset = (y * WIDTH as usize + x) * 3;
                pixels[offset..offset + 3].copy_from_slice(&color);
            }
        }
    }
}

fn diagonal_cue() -> Vec<u8> {
    let mut pixels = felt_image();
    draw_line(
        &mut pixels,
        [30.0, 108.0],
        [166.0, 35.0],
        2.5,
        [235, 235, 235],
    );
    pixels
}

#[test]
fn default_configuration_satisfies_its_own_memory_and_work_bounds() {
    assert_eq!(
        CueGeometryConfig::default().validate(),
        Ok(CueGeometryConfig::default())
    );
}

#[test]
fn fits_a_clear_long_cue_and_returns_canonical_unit_axis() {
    let pixels = diagonal_cue();
    let result = estimate_cue_geometry(
        &pixels,
        WIDTH,
        HEIGHT,
        &[],
        Duration::from_millis(37),
        test_config(),
    )
    .unwrap()
    .expect("cue shaft should be recovered from the whole image");

    assert!(
        result.confidence > 0.5,
        "confidence was {}",
        result.confidence
    );
    assert_eq!(result.observation_age_ms, 37);
    assert!(result.shaft_start_px[0] < result.shaft_end_px[0]);
    assert!(result.axis_unit[0] > 0.0);
    assert!((result.axis_unit[0].hypot(result.axis_unit[1]) - 1.0).abs() < 1e-9);
    let expected_angle = (73.0_f64 / 136.0).atan();
    let actual_angle = result.axis_unit[1].abs().atan2(result.axis_unit[0]);
    assert!((actual_angle - expected_angle).abs() < 0.10);
}

#[test]
fn recent_proposal_is_used_for_cue_refinement() {
    let pixels = diagonal_cue();
    let proposal = CueProposal {
        left: 20,
        top: 20,
        right: 180,
        bottom: 120,
    };
    let result = estimate_cue_geometry(
        &pixels,
        WIDTH,
        HEIGHT,
        &[proposal],
        Duration::ZERO,
        test_config(),
    )
    .unwrap()
    .expect("the cue is inside the proposed region");
    assert_eq!(result.proposal_index, Some(0));
}

#[test]
fn whole_image_reacquisition_finds_cue_outside_stale_proposal() {
    let pixels = diagonal_cue();
    let stale_proposal = CueProposal {
        left: 4,
        top: 4,
        right: 55,
        bottom: 45,
    };
    let result = estimate_cue_geometry(
        &pixels,
        WIDTH,
        HEIGHT,
        &[stale_proposal],
        Duration::from_millis(80),
        test_config(),
    )
    .unwrap()
    .expect("whole-image fallback should reacquire a moved cue");
    assert_eq!(result.proposal_index, None);
}

#[test]
fn texture_and_small_clutter_do_not_look_like_a_cue() {
    let mut pixels = felt_image();
    // A deterministic checker texture creates many edges but no dominant long
    // narrow axis. A few short blocks exercise the minimum-length rejection.
    for y in 0..HEIGHT as usize {
        for x in 0..WIDTH as usize {
            if ((x / 5) + (y / 5)) % 2 == 0 {
                let offset = (y * WIDTH as usize + x) * 3;
                pixels[offset..offset + 3].copy_from_slice(&[180, 180, 180]);
            }
        }
    }
    draw_line(
        &mut pixels,
        [20.0, 20.0],
        [36.0, 20.0],
        1.0,
        [240, 240, 240],
    );
    assert!(
        estimate_cue_geometry(&pixels, WIDTH, HEIGHT, &[], Duration::ZERO, test_config(),)
            .unwrap()
            .is_none()
    );
}

#[test]
fn direction_is_absent_without_strong_endpoint_asymmetry() {
    let pixels = diagonal_cue();
    let result = estimate_cue_geometry(&pixels, WIDTH, HEIGHT, &[], Duration::ZERO, test_config())
        .unwrap()
        .expect("the shaft itself remains detectable");
    assert_eq!(result.signed_direction, None);
}

#[test]
fn strong_dark_tip_asymmetry_can_supply_a_signed_endpoint() {
    let mut pixels = felt_image();
    draw_line(
        &mut pixels,
        [30.0, 108.0],
        [166.0, 35.0],
        2.5,
        [235, 235, 235],
    );
    draw_line(&mut pixels, [30.0, 108.0], [48.0, 98.0], 2.5, [5, 5, 5]);
    let result = estimate_cue_geometry(&pixels, WIDTH, HEIGHT, &[], Duration::ZERO, test_config())
        .unwrap()
        .expect("the shaft should remain visible around its dark endpoint");
    assert_eq!(
        result.signed_direction,
        Some(SignedCueDirection::TowardStart)
    );
}

#[test]
fn invalid_dimensions_and_rgb_lengths_are_rejected() {
    let pixels = felt_image();
    assert!(matches!(
        estimate_cue_geometry(&pixels, 2, HEIGHT, &[], Duration::ZERO, test_config()),
        Err(CueGeometryError::InvalidDimensions { .. })
    ));
    assert!(matches!(
        estimate_cue_geometry(
            &pixels[..pixels.len() - 1],
            WIDTH,
            HEIGHT,
            &[],
            Duration::ZERO,
            test_config()
        ),
        Err(CueGeometryError::PixelLengthMismatch { .. })
    ));
    let bounded = CueGeometryConfig {
        max_width_px: 100,
        max_pixels: 100 * HEIGHT as usize,
        ..test_config()
    };
    assert!(matches!(
        estimate_cue_geometry(&pixels, WIDTH, HEIGHT, &[], Duration::ZERO, bounded),
        Err(CueGeometryError::ImageTooLarge { .. })
    ));
}

#[test]
fn configuration_rejects_unbounded_memory_and_roi_work() {
    let too_little_memory = CueGeometryConfig {
        max_temporary_bytes: 64,
        ..test_config()
    };
    assert_eq!(
        too_little_memory.validate(),
        Err(CueGeometryError::InvalidConfiguration)
    );
    let too_little_work = CueGeometryConfig {
        max_point_visits: 100,
        ..test_config()
    };
    assert_eq!(
        too_little_work.validate(),
        Err(CueGeometryError::InvalidConfiguration)
    );
    let excessive_roi_count = CueGeometryConfig {
        max_proposals: 17,
        ..test_config()
    };
    assert_eq!(
        excessive_roi_count.validate(),
        Err(CueGeometryError::InvalidConfiguration)
    );
}
