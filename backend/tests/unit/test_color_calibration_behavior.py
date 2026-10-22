"""Deterministic behavioral tests for color calibration and profile handling."""

from __future__ import annotations

import json
from datetime import datetime
from typing import TYPE_CHECKING, Any, cast

import cv2
import numpy as np
import pytest

from backend.vision.calibration.color import (
    ColorCalibrator,
    ColorProfile,
    ColorSample,
    ColorThresholds,
)

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray


@pytest.fixture
def calibrator(tmp_path: Path) -> ColorCalibrator:
    """Keep profile reads and writes within the pytest temporary directory."""
    return ColorCalibrator(str(tmp_path / "profiles"))


def _thresholds() -> ColorThresholds:
    return ColorThresholds(20, 40, 80, 220, 30, 240)


def _profile() -> ColorProfile:
    return ColorProfile(
        name="practice",
        table_color=ColorThresholds(50, 70, 100, 255, 40, 210),
        ball_colors={"cue": ColorThresholds(0, 179, 0, 40, 180, 255)},
        lighting_condition="overhead-led",
        creation_date="2026-10-21T12:30:00",
        ambient_light_level=128.0,
    )


def _patch_config(monkeypatch: pytest.MonkeyPatch, values: dict[str, Any]) -> None:
    """Supply only explicit test settings and preserve requested defaults."""

    def get_config(key: str, default: Any = None) -> Any:
        return values.get(key, default)

    monkeypatch.setattr(
        "backend.vision.calibration.color.config_manager.get", get_config
    )


def _bgr_from_hsv(hsv: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Convert synthetic HSV fixtures while accounting for OpenCV stub unions."""
    return cast("NDArray[np.uint8]", cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR))


def _hsv_from_bgr(frame: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Convert fixtures back to HSV using the runtime's uint8 image contract."""
    return cast("NDArray[np.uint8]", cv2.cvtColor(frame, cv2.COLOR_BGR2HSV))


def test_thresholds_round_trip_and_apply_inclusive_hsv_bounds() -> None:
    thresholds = ColorThresholds.from_dict(_thresholds().to_dict())
    hsv = np.array(
        [
            [
                [19, 100, 100],
                [20, 80, 30],
                [30, 150, 100],
                [40, 220, 240],
                [41, 100, 100],
            ]
        ],
        dtype=np.uint8,
    )

    mask = thresholds.apply_mask(hsv)

    np.testing.assert_array_equal(mask, [[0, 255, 255, 255, 0]])


def test_threshold_mask_wraps_hue_across_zero_and_still_checks_saturation_value() -> (
    None
):
    thresholds = ColorThresholds(175, 5, 100, 255, 50, 200)
    hsv = np.array(
        [[[179, 200, 180], [0, 200, 180], [5, 100, 50], [6, 200, 180], [2, 99, 180]]],
        dtype=np.uint8,
    )

    np.testing.assert_array_equal(thresholds.apply_mask(hsv), [[255, 255, 255, 0, 0]])


def test_thresholds_reject_an_invalid_hsv_array_through_opencv() -> None:
    with pytest.raises(cv2.error):
        _thresholds().apply_mask(np.zeros((4, 4), dtype=np.uint8))


def test_profile_round_trip_preserves_nested_thresholds() -> None:
    profile = _profile()

    assert ColorProfile.from_dict(profile.to_dict()) == profile
    assert profile.to_dict()["ball_colors"]["cue"] == {
        "hue_min": 0,
        "hue_max": 179,
        "saturation_min": 0,
        "saturation_max": 40,
        "value_min": 180,
        "value_max": 255,
    }


def test_color_sample_collects_roi_pixels_and_reports_statistics() -> None:
    sample = ColorSample((1, 0, 2, 2), "felt")
    sample.add_pixels(
        np.array(
            [
                [[10, 20, 30], [40, 50, 60], [70, 80, 90]],
                [[100, 110, 120], [130, 140, 150], [160, 170, 180]],
            ],
            dtype=np.uint8,
        )
    )

    stats = sample.get_statistics()

    assert sample.label == "felt"
    assert sample.pixels == [
        (40, 50, 60),
        (70, 80, 90),
        (130, 140, 150),
        (160, 170, 180),
    ]
    assert stats["count"] == 4
    assert stats["mean"] == [100.0, 110.0, 120.0]
    assert stats["min"] == [40, 50, 60]
    assert stats["max"] == [160, 170, 180]
    np.testing.assert_allclose(stats["std"], np.std(sample.pixels, axis=0))


def test_color_sample_empty_roi_and_invalid_array_behavior() -> None:
    sample = ColorSample((10, 10, 2, 2), "outside")
    image = np.zeros((3, 3, 3), dtype=np.uint8)

    sample.add_pixels(image)
    assert sample.get_statistics() == {}

    malformed = ColorSample((0, 0, 2, 2), "malformed")
    with pytest.raises(ValueError):
        malformed.add_pixels(np.zeros((2, 2), dtype=np.uint8))


def test_calibrator_loads_configured_default_ball_colors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configured = {
        "eight": {
            "hue_min": 10,
            "hue_max": 20,
            "saturation_min": 30,
            "saturation_max": 40,
            "value_min": 50,
            "value_max": 60,
        }
    }
    _patch_config(
        monkeypatch,
        {"vision.calibration.color.default_ball_colors": configured},
    )

    calibrator = ColorCalibrator(str(tmp_path / "configured"))

    assert calibrator.default_ball_colors == {
        "eight": ColorThresholds(10, 20, 30, 40, 50, 60)
    }


def test_table_calibration_selects_largest_cluster_and_respects_given_mask(
    calibrator: ColorCalibrator, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_config(
        monkeypatch,
        {
            "vision.calibration.color.auto_calibration.kmeans_clusters": 2,
            "vision.calibration.color.auto_calibration.kmeans_random_state": 7,
            "vision.calibration.color.auto_calibration.kmeans_n_init": 1,
            "vision.calibration.color.auto_calibration.std_deviation_multiplier": 2,
        },
    )
    hsv = np.empty((4, 4, 3), dtype=np.uint8)
    hsv[:3, :, :] = (60, 180, 150)
    hsv[3, :, :] = (0, 180, 150)
    frame = _bgr_from_hsv(hsv)
    table_mask = np.zeros((4, 4), dtype=np.uint8)
    table_mask[:, :] = 255

    result = calibrator.auto_calibrate_table_color(frame, table_mask)

    assert result == ColorThresholds(60, 60, 180, 180, 150, 150)


def test_table_calibration_builds_center_mask_using_configured_margin(
    calibrator: ColorCalibrator, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_config(
        monkeypatch,
        {
            "vision.calibration.color.auto_calibration.table_mask_margin_ratio": 0.2,
            "vision.calibration.color.auto_calibration.kmeans_clusters": 1,
            "vision.calibration.color.auto_calibration.kmeans_n_init": 1,
        },
    )
    hsv = np.zeros((10, 10, 3), dtype=np.uint8)
    hsv[:, :] = (0, 200, 150)
    hsv[2:8, 2:8] = (60, 180, 140)
    frame = _bgr_from_hsv(hsv)

    result = calibrator.auto_calibrate_table_color(frame)

    assert result.hue_min == result.hue_max == 60
    assert result.saturation_min == result.saturation_max == 180
    assert result.value_min == result.value_max == 140


def test_table_calibration_empty_mask_returns_configured_fallback(
    calibrator: ColorCalibrator, monkeypatch: pytest.MonkeyPatch
) -> None:
    fallback = {
        "hue_min": 44,
        "hue_max": 76,
        "saturation_min": 35,
        "saturation_max": 240,
        "value_min": 45,
        "value_max": 205,
    }
    _patch_config(
        monkeypatch,
        {"vision.calibration.color.default_table_color": fallback},
    )
    frame = np.zeros((8, 8, 3), dtype=np.uint8)

    result = calibrator.auto_calibrate_table_color(
        frame, np.zeros((8, 8), dtype=np.uint8)
    )

    assert result == ColorThresholds(44, 76, 35, 240, 45, 205)


def test_table_calibration_rejects_too_few_pixels_for_configured_cluster_count(
    calibrator: ColorCalibrator, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_config(
        monkeypatch,
        {"vision.calibration.color.auto_calibration.kmeans_clusters": 3},
    )
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    mask = np.zeros((2, 2), dtype=np.uint8)
    mask[0, :2] = 255

    with pytest.raises(ValueError, match="n_samples=2 should be >= n_clusters=3"):
        calibrator.auto_calibrate_table_color(frame, mask)


def test_table_calibration_rejects_malformed_frame_and_mask_shapes(
    calibrator: ColorCalibrator,
) -> None:
    with pytest.raises(cv2.error):
        calibrator.auto_calibrate_table_color(np.zeros((4, 4), dtype=np.uint8))

    with pytest.raises(IndexError):
        calibrator.auto_calibrate_table_color(
            np.zeros((4, 4, 3), dtype=np.uint8),
            np.ones((2, 2), dtype=np.uint8),
        )


def test_ball_calibration_uses_configured_pixel_filter_and_percentiles(
    calibrator: ColorCalibrator, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_config(
        monkeypatch,
        {
            "vision.calibration.color.auto_calibration.pixel_filter_min_value": 30,
            "vision.calibration.color.auto_calibration.pixel_filter_max_value": 220,
            "vision.calibration.color.auto_calibration.min_pixels_for_calibration": 4,
            "vision.calibration.color.ball_calibration.percentiles.hue": [0, 100],
            "vision.calibration.color.ball_calibration.percentiles.saturation": [
                0,
                100,
            ],
            "vision.calibration.color.ball_calibration.percentiles.value": [0, 100],
        },
    )
    hsv = np.array(
        [[[30, 100, 80], [32, 110, 100]], [[34, 130, 120], [36, 140, 180]]],
        dtype=np.uint8,
    )
    frame = _bgr_from_hsv(hsv)

    result = calibrator.calibrate_ball_colors(frame, {"stripe": [(0, 0, 2, 2)]})

    assert result == {"stripe": ColorThresholds(30, 36, 99, 140, 80, 180)}


def test_ball_calibration_filters_dark_highlighted_and_unknown_samples_to_defaults(
    calibrator: ColorCalibrator, monkeypatch: pytest.MonkeyPatch
) -> None:
    configured_default = ColorThresholds(10, 30, 40, 200, 35, 210)
    calibrator.default_ball_colors["cue"] = configured_default
    _patch_config(
        monkeypatch,
        {
            "vision.calibration.color.auto_calibration.pixel_filter_min_value": 30,
            "vision.calibration.color.auto_calibration.pixel_filter_max_value": 220,
            "vision.calibration.color.auto_calibration.min_pixels_for_calibration": 2,
        },
    )
    hsv = np.array([[[60, 180, 20], [60, 180, 240]]], dtype=np.uint8)
    frame = _bgr_from_hsv(hsv)

    result = calibrator.calibrate_ball_colors(
        frame, {"cue": [(0, 0, 2, 1)], "unknown": [(0, 0, 2, 1)]}
    )

    assert result == {
        "cue": configured_default,
        "unknown": ColorThresholds(0, 180, 0, 255, 0, 255),
    }


def test_ball_calibration_detects_red_hue_wraparound(
    calibrator: ColorCalibrator, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_config(
        monkeypatch,
        {
            "vision.calibration.color.auto_calibration.pixel_filter_min_value": 0,
            "vision.calibration.color.auto_calibration.pixel_filter_max_value": 255,
            "vision.calibration.color.auto_calibration.min_pixels_for_calibration": 4,
            "vision.calibration.color.ball_calibration.percentiles.hue": [0, 100],
            "vision.calibration.color.ball_calibration.percentiles.saturation": [
                0,
                100,
            ],
            "vision.calibration.color.ball_calibration.percentiles.value": [0, 100],
            "vision.calibration.color.ball_calibration.hue_wraparound_threshold": 90,
            "vision.calibration.color.hue_shift_threshold": 90,
        },
    )
    hsv = np.array(
        [
            [
                [178, 200, 180],
                [179, 200, 180],
                [0, 200, 180],
                [2, 200, 180],
                [4, 200, 180],
            ]
        ],
        dtype=np.uint8,
    )
    frame = _bgr_from_hsv(hsv)
    recovered_hsv = _hsv_from_bgr(frame)

    result = calibrator.calibrate_ball_colors(frame, {"red": [(0, 0, 5, 1)]})["red"]

    assert result.hue_min > result.hue_max
    assert result.apply_mask(recovered_hsv).tolist() == [[255, 255, 255, 255, 255]]


def test_lighting_estimate_tracks_grayscale_brightness(
    calibrator: ColorCalibrator,
) -> None:
    assert calibrator._estimate_ambient_light(np.zeros((3, 4, 3), dtype=np.uint8)) == 0
    assert (
        calibrator._estimate_ambient_light(np.full((3, 4, 3), 100, dtype=np.uint8))
        == 100
    )


def test_lighting_adaptation_brightens_thresholds_and_clamps_values(
    calibrator: ColorCalibrator, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_config(monkeypatch, {})
    reference = ColorProfile(
        name="reference",
        table_color=ColorThresholds(50, 70, 80, 220, 100, 200),
        ball_colors={"solid": ColorThresholds(0, 15, 90, 240, 80, 180)},
        lighting_condition="dim",
        creation_date="2026-01-01",
        ambient_light_level=50,
    )

    adapted = calibrator.adapt_to_lighting(
        np.full((4, 4, 3), 200, dtype=np.uint8), reference
    )

    assert adapted.name == "reference_adapted"
    assert adapted.lighting_condition == "adapted_from_dim"
    assert adapted.ambient_light_level == 200
    assert adapted.table_color == ColorThresholds(50, 70, 60, 220, 255, 255)
    assert adapted.ball_colors["solid"] == ColorThresholds(0, 15, 70, 240, 255, 255)
    datetime.fromisoformat(adapted.creation_date)


def test_lighting_adaptation_darkens_thresholds_and_zero_reference_uses_unit_ratio(
    calibrator: ColorCalibrator, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_config(monkeypatch, {})
    reference = _profile()

    darkened = calibrator.adapt_to_lighting(
        np.full((4, 4, 3), 20, dtype=np.uint8), reference
    )
    zero_reference = ColorProfile(**{**reference.__dict__, "ambient_light_level": 0.0})
    neutral = calibrator.adapt_to_lighting(
        np.full((4, 4, 3), 128, dtype=np.uint8), zero_reference
    )

    assert darkened.table_color == ColorThresholds(50, 70, 90, 255, 7, 36)
    assert darkened.ball_colors["cue"] == ColorThresholds(0, 179, 0, 40, 33, 43)
    assert neutral.ambient_light_level == 128
    assert neutral.table_color == reference.table_color
    assert neutral.ball_colors == reference.ball_colors


def test_dark_lighting_adaptation_keeps_uint8_value_thresholds_in_range(
    calibrator: ColorCalibrator, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_config(monkeypatch, {})

    adapted = calibrator._adapt_thresholds(
        ColorThresholds(20, 40, 50, 200, 100, 255), 0.99
    )

    assert 0 <= adapted.value_min <= adapted.value_max <= 255


def test_ambient_light_empty_histogram_uses_midpoint_fallback(
    calibrator: ColorCalibrator, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "backend.vision.calibration.color.cv2.calcHist",
        lambda *_args, **_kwargs: np.zeros((256, 1), dtype=np.float32),
    )

    assert (
        calibrator._estimate_ambient_light(np.zeros((2, 2, 3), dtype=np.uint8)) == 128.0
    )


def test_threshold_optimization_skips_objects_without_usable_positive_regions(
    calibrator: ColorCalibrator, caplog: pytest.LogCaptureFixture
) -> None:
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    objects: list[dict[str, Any]] = [
        {"name": "no-positive", "positive_samples": []},
        {"name": "outside-frame", "positive_samples": [(8, 8, 2, 2)]},
    ]

    assert calibrator.optimize_thresholds(frame, objects) == {}
    assert "No positive samples" in caplog.text
    assert "Positive samples contain no pixels" in caplog.text


def test_threshold_optimization_uses_positive_percentiles_without_negatives(
    calibrator: ColorCalibrator,
) -> None:
    hsv = np.array(
        [
            [[10, 100, 100], [20, 120, 120]],
            [[30, 140, 140], [40, 160, 160]],
        ],
        dtype=np.uint8,
    )
    frame = _bgr_from_hsv(hsv)
    recovered_hsv = _hsv_from_bgr(frame)

    result = calibrator.optimize_thresholds(
        frame, [{"name": "ball", "positive_samples": [(0, 0, 2, 2)]}]
    )["ball"]

    assert result.hue_min == int(np.percentile(recovered_hsv[:, :, 0], 5))
    assert result.hue_max == int(np.percentile(recovered_hsv[:, :, 0], 95))
    assert result.saturation_min == int(np.percentile(recovered_hsv[:, :, 1], 10))
    assert result.saturation_max == int(np.percentile(recovered_hsv[:, :, 1], 90))
    assert result.value_min == int(np.percentile(recovered_hsv[:, :, 2], 10))
    assert result.value_max == int(np.percentile(recovered_hsv[:, :, 2], 90))


def test_threshold_optimization_tightens_bounds_when_negative_pixels_match(
    calibrator: ColorCalibrator,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_config(
        monkeypatch,
        {
            "vision.calibration.color.threshold_optimization.max_iterations": 2,
            "vision.calibration.color.threshold_optimization.range_tightening_factor": 0.25,
        },
    )
    hsv = np.empty((5, 5, 3), dtype=np.uint8)
    hsv[:4, :, :] = np.array(
        [
            [
                [
                    100 + ((x + y) % 2) * 20,
                    100 + ((x + y) % 2) * 40,
                    100 + ((x + y) % 2) * 40,
                ]
                for x in range(5)
            ]
            for y in range(4)
        ],
        dtype=np.uint8,
    )
    hsv[4, 0] = (110, 120, 120)
    hsv[4, 1:] = (0, 0, 0)
    frame = _bgr_from_hsv(hsv)

    result = calibrator.optimize_thresholds(
        frame,
        [
            {
                "name": "cue",
                "positive_samples": [(0, 0, 5, 4)],
                "negative_samples": [(0, 4, 1, 1)],
            }
        ],
    )["cue"]

    assert result.hue_min > 100
    assert result.hue_max < 120
    assert result.saturation_min > 100
    assert result.saturation_max < 140
    assert result.value_min > 100
    assert result.value_max < 140


def test_threshold_optimization_stops_when_negative_sample_is_already_excluded(
    calibrator: ColorCalibrator,
) -> None:
    hsv = np.array(
        [
            [[60, 180, 180], [60, 180, 180], [60, 180, 180]],
            [[60, 180, 180], [60, 180, 180], [0, 0, 0]],
        ],
        dtype=np.uint8,
    )
    frame = _bgr_from_hsv(hsv)

    result = calibrator.optimize_thresholds(
        frame,
        [
            {
                "name": "felt-ball",
                "positive_samples": [(0, 0, 2, 2)],
                "negative_samples": [(2, 1, 1, 1)],
            }
        ],
    )["felt-ball"]

    assert result == ColorThresholds(60, 60, 180, 180, 180, 180)


def test_profile_save_load_and_listing_round_trip(
    calibrator: ColorCalibrator,
) -> None:
    profile = _profile()

    assert calibrator.save_profile(profile)
    filename = "practice_2026-10-21.json"
    assert calibrator.list_profiles() == [filename]
    loaded = calibrator.load_profile(filename)

    assert loaded == profile
    assert calibrator.current_profile == profile


def test_profile_save_uses_explicit_filename_and_reports_write_failure(
    calibrator: ColorCalibrator,
) -> None:
    profile = _profile()

    assert calibrator.save_profile(profile, "custom.json")
    assert (calibrator.cache_dir / "custom.json").is_file()
    assert not calibrator.save_profile(profile, "missing-parent/profile.json")


def test_profile_load_reports_missing_and_malformed_files_without_replacing_current(
    calibrator: ColorCalibrator,
) -> None:
    original = _profile()
    calibrator.current_profile = original
    (calibrator.cache_dir / "broken.json").write_text("{invalid json", encoding="utf-8")

    assert calibrator.load_profile("missing.json") is None
    assert calibrator.load_profile("broken.json") is None
    assert calibrator.current_profile is original


def test_default_profile_uses_calibrated_table_defaults_and_measured_light(
    calibrator: ColorCalibrator,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_config(
        monkeypatch,
        {
            "vision.calibration.color.auto_calibration.kmeans_clusters": 1,
            "vision.calibration.color.auto_calibration.kmeans_n_init": 1,
        },
    )
    calibrator.default_ball_colors = {"cue": ColorThresholds(0, 179, 0, 40, 180, 255)}
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    frame[:, :] = (0, 120, 0)

    profile = calibrator.create_default_profile(frame, "my-table")

    assert profile.name == "my-table"
    assert profile.lighting_condition == "auto_detected"
    assert profile.ambient_light_level == calibrator._estimate_ambient_light(frame)
    assert profile.ball_colors == calibrator.default_ball_colors
    assert calibrator.current_profile is profile
    datetime.fromisoformat(profile.creation_date)


def test_invalid_profile_data_is_reported_as_load_failure(
    calibrator: ColorCalibrator,
) -> None:
    path = calibrator.cache_dir / "wrong-shape.json"
    path.write_text(json.dumps({"name": "incomplete"}), encoding="utf-8")

    assert calibrator.load_profile(path.name) is None
