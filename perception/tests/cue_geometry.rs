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
        max_point_visits: 380_000,
        max_temporary_bytes: 12_000 * std::mem::size_of::<[f64; 2]>() * 2 + 16_384,
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

fn tapered_cue_frame(
    reverse_edge_contrast: bool,
    occlude_one_edge: bool,
    add_parallel_texture: bool,
    reverse_slope: bool,
) -> (Vec<u8>, [f64; 2], [f64; 2]) {
    let mut pixels = vec![0_u8; (WIDTH * HEIGHT * 3) as usize];
    let start: [f64; 2] = if reverse_slope {
        [35.0, 34.0]
    } else {
        [35.0, 106.0]
    };
    let end = if reverse_slope {
        [183.0, 106.0]
    } else {
        [183.0, 34.0]
    };
    let delta = [end[0] - start[0], end[1] - start[1]];
    let length = delta[0].hypot(delta[1]);
    let axis = [delta[0] / length, delta[1] / length];
    let normal = [-axis[1], axis[0]];
    let contrast_sign = if reverse_edge_contrast { -1.0 } else { 1.0 };

    for y in 0..HEIGHT as usize {
        for x in 0..WIDTH as usize {
            let relative = [x as f64 - start[0], y as f64 - start[1]];
            let along = relative[0] * axis[0] + relative[1] * axis[1];
            let across = relative[0] * normal[0] + relative[1] * normal[1];
            // A slow cloth gradient makes one shaft edge stronger while
            // keeping the background itself below the edge extraction limit.
            let background = (86.0 + contrast_sign * across.clamp(-12.0, 12.0) * 4.0)
                .round()
                .clamp(0.0, 255.0) as u8;
            let mut value = background;
            if (0.0..=length).contains(&along) {
                let fraction = along / length;
                let half_width = 3.1 - fraction * 0.8;
                if across.abs() <= half_width {
                    value = if fraction < 0.12 {
                        58
                    } else if fraction > 0.94 {
                        245
                    } else {
                        210
                    };
                    if occlude_one_edge && (0.34..0.49).contains(&fraction) && across > 0.0 {
                        value = background;
                    }
                }
            }
            if add_parallel_texture
                && (0.20..0.74).contains(&(along / length))
                && (across - 9.0).abs() <= 0.55
            {
                value = 245;
            }
            let offset = (y * WIDTH as usize + x) * 3;
            pixels[offset..offset + 3].fill(value);
        }
    }
    (pixels, start, end)
}

fn small_guidance_cue_frame() -> Vec<u8> {
    const SMALL_WIDTH: usize = 16;
    const SMALL_HEIGHT: usize = 16;
    let mut pixels = vec![0_u8; SMALL_WIDTH * SMALL_HEIGHT * 3];
    for y in 2..14 {
        for x in 2..14 {
            let offset = (y * SMALL_WIDTH + x) * 3;
            pixels[offset..offset + 3].copy_from_slice(&[30, 100, 30]);
        }
    }
    let from = [3.0, 12.0];
    let to = [11.0, 4.0];
    let dx = to[0] - from[0];
    let dy = to[1] - from[1];
    let length_squared = dx * dx + dy * dy;
    for y in 0..SMALL_HEIGHT {
        for x in 0..SMALL_WIDTH {
            let fraction = (((x as f64 - from[0]) * dx + (y as f64 - from[1]) * dy)
                / length_squared)
                .clamp(0.0, 1.0);
            let nearest = [from[0] + fraction * dx, from[1] + fraction * dy];
            if (x as f64 - nearest[0]).hypot(y as f64 - nearest[1]) <= 1.2 {
                let offset = (y * SMALL_WIDTH + x) * 3;
                pixels[offset..offset + 3].copy_from_slice(&[235, 235, 235]);
            }
        }
    }
    pixels
        .into_iter()
        .map(|value| value.saturating_add(48))
        .collect()
}

fn centerline_error(
    geometry: &billiards_perception::cue_geometry::CueGeometry,
    point: [f64; 2],
) -> f64 {
    let normal = [-geometry.axis_unit[1], geometry.axis_unit[0]];
    ((point[0] - geometry.shaft_start_px[0]) * normal[0]
        + (point[1] - geometry.shaft_start_px[1]) * normal[1])
        .abs()
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
fn thumbnail_sized_cue_still_recovers_its_paired_centerline() {
    const SMALL_WIDTH: u32 = 16;
    const SMALL_HEIGHT: u32 = 16;
    let pixels = small_guidance_cue_frame();
    let config = CueGeometryConfig {
        max_width_px: SMALL_WIDTH,
        max_height_px: SMALL_HEIGHT,
        max_pixels: (SMALL_WIDTH * SMALL_HEIGHT) as usize,
        max_edge_samples: (SMALL_WIDTH * SMALL_HEIGHT) as usize,
        ..CueGeometryConfig::default()
    };
    let proposal = CueProposal {
        left: 3,
        top: 3,
        right: 12,
        bottom: 13,
    };
    let geometry = estimate_cue_geometry(
        &pixels,
        SMALL_WIDTH,
        SMALL_HEIGHT,
        &[proposal],
        Duration::ZERO,
        config,
    )
    .unwrap()
    .expect("the rendered cue should remain visible at thumbnail resolution");
    assert!(centerline_error(&geometry, [7.0, 8.0]) < 0.75);
    assert_eq!(geometry.observation_age_ms, 0);
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
fn paired_sides_keep_the_centerline_when_edge_contrast_reverses() {
    let proposal = CueProposal {
        left: 12,
        top: 12,
        right: 208,
        bottom: 128,
    };
    let mut errors = Vec::new();
    let mut axes = Vec::new();
    for reverse_contrast in [false, true] {
        let (pixels, start, end) = tapered_cue_frame(reverse_contrast, false, false, false);
        let midpoint = [(start[0] + end[0]) * 0.5, (start[1] + end[1]) * 0.5];
        let geometry = estimate_cue_geometry(
            &pixels,
            WIDTH,
            HEIGHT,
            &[proposal],
            Duration::ZERO,
            test_config(),
        )
        .unwrap()
        .expect("both visible shaft sides should define a centerline");
        errors.push(centerline_error(&geometry, midpoint));
        axes.push(geometry.axis_unit);
    }

    assert!(
        errors.iter().all(|error| *error < 1.25),
        "axis offsets: {errors:?}"
    );
    assert!(
        (errors[0] - errors[1]).abs() < 0.75,
        "changing which edge has stronger contrast moved the centerline: {errors:?}"
    );
    let axis_alignment = axes[0][0] * axes[1][0] + axes[0][1] * axes[1][1];
    assert!(
        axis_alignment > 0.999,
        "the axis flipped with exposure: {axes:?}"
    );
}

#[test]
fn tapered_shaft_survives_partial_occlusion_and_parallel_texture() {
    let (pixels, start, end) = tapered_cue_frame(true, true, true, true);
    let midpoint = [(start[0] + end[0]) * 0.5, (start[1] + end[1]) * 0.5];
    let proposal = CueProposal {
        left: 12,
        top: 12,
        right: 208,
        bottom: 128,
    };
    let geometry = estimate_cue_geometry(
        &pixels,
        WIDTH,
        HEIGHT,
        &[proposal],
        Duration::ZERO,
        test_config(),
    )
    .unwrap()
    .expect("paired sections should remain available around a partial occlusion");
    assert!(centerline_error(&geometry, midpoint) < 1.5);
    assert!(
        geometry.axis_unit[0] > 0.0,
        "axis should use canonical ordering"
    );
    assert!(
        geometry.axis_unit[1] > 0.0,
        "reversed slope was lost: {geometry:?}"
    );
}

#[test]
fn each_frame_is_fresh_when_a_previously_visible_cue_disappears() {
    let pixels = diagonal_cue();
    assert!(
        estimate_cue_geometry(&pixels, WIDTH, HEIGHT, &[], Duration::ZERO, test_config(),)
            .unwrap()
            .is_some()
    );

    let absent = felt_image();
    assert!(
        estimate_cue_geometry(&absent, WIDTH, HEIGHT, &[], Duration::ZERO, test_config(),)
            .unwrap()
            .is_none()
    );
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
    let excessive_sections = CueGeometryConfig {
        centerline_sections: 129,
        ..test_config()
    };
    assert_eq!(
        excessive_sections.validate(),
        Err(CueGeometryError::InvalidConfiguration)
    );
    let excessive_profile_samples = CueGeometryConfig {
        cross_section_profile_samples: 9,
        ..test_config()
    };
    assert_eq!(
        excessive_profile_samples.validate(),
        Err(CueGeometryError::InvalidConfiguration)
    );
    let excessive_width = CueGeometryConfig {
        maximum_shaft_width_fraction: 0.051,
        ..test_config()
    };
    assert_eq!(
        excessive_width.validate(),
        Err(CueGeometryError::InvalidConfiguration)
    );
    let invalid_width_fallback = CueGeometryConfig {
        minimum_shaft_maximum_width_px: 0.25,
        ..test_config()
    };
    assert_eq!(
        invalid_width_fallback.validate(),
        Err(CueGeometryError::InvalidConfiguration)
    );
    let invalid_section_spacing = CueGeometryConfig {
        minimum_centerline_section_spacing_px: 0.0,
        ..test_config()
    };
    assert_eq!(
        invalid_section_spacing.validate(),
        Err(CueGeometryError::InvalidConfiguration)
    );
}
