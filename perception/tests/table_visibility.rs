#[path = "../src/table_visibility.rs"]
mod table_visibility;

use table_visibility::{
    assess_table_visibility, derive_reference_from_confirmed_frame, validate_reference,
    validate_reference_structure, NormalizedPoint, OperatorConfirmedFrame, ReferenceIssue,
    Rgb8Frame, TableRegion, TableVisibilityAssessment, TableVisibilityConfig,
    TableVisibilityReason, TableVisibilityStatus,
};

const WIDTH: u32 = 320;
const HEIGHT: u32 = 240;

fn table_region() -> TableRegion {
    TableRegion::new([
        point(40.0 / WIDTH as f32, 35.0 / HEIGHT as f32),
        point(280.0 / WIDTH as f32, 35.0 / HEIGHT as f32),
        point(280.0 / WIDTH as f32, 205.0 / HEIGHT as f32),
        point(40.0 / WIDTH as f32, 205.0 / HEIGHT as f32),
    ])
}

fn point(x: f32, y: f32) -> NormalizedPoint {
    NormalizedPoint { x, y }
}

fn render_table(offset_x: i32, cloth: [u8; 3], background: [u8; 3]) -> Vec<u8> {
    let mut pixels = vec![0; WIDTH as usize * HEIGHT as usize * 3];
    for y in 0..HEIGHT as i32 {
        for x in 0..WIDTH as i32 {
            let shifted_x = x - offset_x;
            let color = if (40..=280).contains(&shifted_x) && (35..=205).contains(&y) {
                cloth
            } else if (33..=287).contains(&shifted_x) && (28..=212).contains(&y) {
                [106, 72, 42]
            } else {
                background
            };
            let index = (y as usize * WIDTH as usize + x as usize) * 3;
            pixels[index..index + 3].copy_from_slice(&color);
        }
    }
    pixels
}

fn frame(pixels: &[u8]) -> Rgb8Frame<'_> {
    Rgb8Frame {
        width: WIDTH,
        height: HEIGHT,
        pixels,
    }
}

fn reference(
    config: &TableVisibilityConfig,
) -> (Vec<u8>, table_visibility::TableVisibilityReference) {
    let pixels = render_table(0, [42, 132, 54], [24, 22, 20]);
    let result = derive_reference_from_confirmed_frame(
        OperatorConfirmedFrame::new(frame(&pixels), table_region()),
        config,
    )
    .expect("valid synthetic frame should be sampled");
    assert!(result.validity.valid, "{:?}", result.validity.reasons);
    (pixels, result.reference.expect("valid reference"))
}

fn assess(
    reference: &table_visibility::TableVisibilityReference,
    pixels: &[u8],
    config: &TableVisibilityConfig,
) -> TableVisibilityAssessment {
    assess_table_visibility(Some(reference), Some(frame(pixels)), config).expect("assessment")
}

#[test]
fn matching_reference_and_ordinary_brightness_shift_are_visible() {
    let config = TableVisibilityConfig::default();
    let (source, reference) = reference(&config);
    let exact = assess(&reference, &source, &config);
    assert_eq!(exact.status, TableVisibilityStatus::Visible);
    assert_eq!(exact.reason, TableVisibilityReason::ReferenceMatch);

    let brighter: Vec<u8> = source
        .iter()
        .map(|value| value.saturating_add(20))
        .collect();
    let shifted = assess(&reference, &brighter, &config);
    assert_eq!(shifted.status, TableVisibilityStatus::Visible);
    assert!(shifted.global_luminance_shift.unwrap() > 0);
}

#[test]
fn persisted_reference_must_match_the_active_sampling_layout() {
    let config = TableVisibilityConfig::default();
    let (_pixels, reference) = reference(&config);
    assert!(validate_reference(&reference, &config));
    assert!(validate_reference_structure(&reference, 16_384, 16_384));

    let mut incompatible = config.clone();
    incompatible.cloth_grid_width += 1;
    assert!(!validate_reference(&reference, &incompatible));
}

#[test]
fn dark_or_covered_scene_fails_closed() {
    let config = TableVisibilityConfig::default();
    let (source, reference) = reference(&config);
    let dark: Vec<u8> = source
        .iter()
        .map(|value| value.saturating_sub(80))
        .collect();
    let dark_result = assess(&reference, &dark, &config);
    assert_eq!(dark_result.status, TableVisibilityStatus::NotVisible);
    assert_eq!(
        dark_result.reason,
        TableVisibilityReason::ExtremeBrightnessShift
    );

    let mut covered = source.clone();
    for y in 44..196 {
        for x in 50..271 {
            let index = (y * WIDTH as usize + x) * 3;
            covered[index..index + 3].copy_from_slice(&[80, 80, 80]);
        }
    }
    let covered_result = assess(&reference, &covered, &config);
    assert_eq!(covered_result.status, TableVisibilityStatus::NotVisible);
    assert!(matches!(
        covered_result.reason,
        TableVisibilityReason::ExtremeColorShift
            | TableVisibilityReason::ClothMismatch
            | TableVisibilityReason::LostRailContrast
    ));
}

#[test]
fn extreme_exposure_does_not_hide_lost_rail_structure() {
    let config = TableVisibilityConfig::default();
    let (source, reference) = reference(&config);
    let shifted: Vec<u8> = source
        .iter()
        .map(|value| value.saturating_add(100))
        .collect();
    let legible = assess(&reference, &shifted, &config);
    assert_eq!(
        legible.reason,
        TableVisibilityReason::ExtremeBrightnessShift
    );
    assert!(
        legible.live_rail_contrast_profile_fraction.unwrap()
            >= config.minimum_live_rail_contrast_fraction
    );

    let covered = vec![200; source.len()];
    let missing_structure = assess(&reference, &covered, &config);
    assert_eq!(missing_structure.status, TableVisibilityStatus::NotVisible);
    assert_eq!(
        missing_structure.reason,
        TableVisibilityReason::LostRailContrast
    );
}

#[test]
fn low_quality_reference_reports_dark_surface_and_missing_rail_contrast() {
    let config = TableVisibilityConfig::default();
    let dark = vec![8; WIDTH as usize * HEIGHT as usize * 3];
    let dark_result = derive_reference_from_confirmed_frame(
        OperatorConfirmedFrame::new(frame(&dark), table_region()),
        &config,
    )
    .expect("well-formed dark frame");
    assert!(!dark_result.validity.valid);
    assert!(dark_result.reference.is_none());
    assert!(dark_result
        .validity
        .reasons
        .contains(&ReferenceIssue::ClothTooDark));
    assert!(dark_result
        .validity
        .reasons
        .contains(&ReferenceIssue::InsufficientRailContrast));

    let flat = vec![70; WIDTH as usize * HEIGHT as usize * 3];
    let flat_result = derive_reference_from_confirmed_frame(
        OperatorConfirmedFrame::new(frame(&flat), table_region()),
        &config,
    )
    .expect("well-formed low-contrast frame");
    assert!(!flat_result.validity.valid);
    assert!(flat_result
        .validity
        .reasons
        .contains(&ReferenceIssue::InsufficientRailContrast));
}

#[test]
fn displaced_table_is_rejected_by_rail_profiles() {
    let config = TableVisibilityConfig::default();
    let (_, reference) = reference(&config);
    let displaced = render_table(16, [42, 132, 54], [24, 22, 20]);
    let result = assess(&reference, &displaced, &config);
    assert_eq!(result.status, TableVisibilityStatus::NotVisible);
    assert!(matches!(
        result.reason,
        TableVisibilityReason::RailMismatch
            | TableVisibilityReason::ClothMismatch
            | TableVisibilityReason::LostRailContrast
    ));
}

#[test]
fn missing_or_malformed_evidence_never_reports_visible() {
    let config = TableVisibilityConfig::default();
    let (source, reference) = reference(&config);
    let missing_reference = assess_table_visibility(None, Some(frame(&source)), &config).unwrap();
    assert_eq!(missing_reference.status, TableVisibilityStatus::Unknown);
    assert_eq!(
        missing_reference.reason,
        TableVisibilityReason::MissingReference
    );

    let missing_frame = assess_table_visibility(Some(&reference), None, &config).unwrap();
    assert_eq!(missing_frame.status, TableVisibilityStatus::Unknown);
    assert_eq!(missing_frame.reason, TableVisibilityReason::MissingFrame);

    let mut malformed = reference.clone();
    malformed.rail_features.pop();
    let malformed_result = assess(&malformed, &source, &config);
    assert_eq!(malformed_result.status, TableVisibilityStatus::NotVisible);
    assert_eq!(
        malformed_result.reason,
        TableVisibilityReason::InvalidReference
    );
}

#[test]
fn malformed_frame_and_unbounded_configuration_are_rejected() {
    let config = TableVisibilityConfig::default();
    let (_, reference) = reference(&config);
    let short_frame = Rgb8Frame {
        width: WIDTH,
        height: HEIGHT,
        pixels: &[],
    };
    let error = assess_table_visibility(Some(&reference), Some(short_frame), &config).unwrap_err();
    assert!(matches!(
        error,
        table_visibility::TableVisibilityError::InvalidFrameLength { .. }
    ));

    let mut too_many_samples = config.clone();
    too_many_samples.maximum_feature_samples = 128;
    assert!(matches!(
        too_many_samples.validate(),
        Err(table_visibility::TableVisibilityError::InvalidConfig(_))
    ));
}

#[test]
fn invalid_calibrated_boundary_returns_a_bounded_quality_reason() {
    let config = TableVisibilityConfig::default();
    let pixels = render_table(0, [42, 132, 54], [24, 22, 20]);
    let invalid_region = TableRegion::new([
        point(0.2, 0.2),
        point(0.8, 0.8),
        point(0.8, 0.2),
        point(0.2, 0.8),
    ]);
    let result = derive_reference_from_confirmed_frame(
        OperatorConfirmedFrame::new(frame(&pixels), invalid_region),
        &config,
    )
    .unwrap();
    assert!(!result.validity.valid);
    assert!(result.reference.is_none());
    assert_eq!(
        result.validity.reasons,
        vec![ReferenceIssue::InvalidTableRegion]
    );
}
