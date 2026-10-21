"""Regression coverage for canonical Core coordinate boundaries."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any

import pytest

from backend.config import config
from backend.core.coordinates import Vector2D
from backend.core.game_state import GameStateManager
from backend.core.models import BallState, CoordinateMetadata, TableState

if TYPE_CHECKING:
    from pathlib import Path


def _scaled_point(x: float, y: float, factor: float) -> dict[str, Any]:
    return {"x": x, "y": y, "scale": [factor, factor]}


def _table_payload(scale: float) -> dict[str, Any]:
    corners = [
        _scaled_point(320 / scale, 280 / scale, scale),
        _scaled_point(3520 / scale, 280 / scale, scale),
        _scaled_point(3520 / scale, 1880 / scale, scale),
        _scaled_point(320 / scale, 1880 / scale, scale),
    ]
    pockets = [
        corners[0],
        _scaled_point(1920 / scale, 280 / scale, scale),
        corners[1],
        corners[3],
        _scaled_point(1920 / scale, 1880 / scale, scale),
        corners[2],
    ]
    return {
        "width": 3200 / scale,
        "height": 1600 / scale,
        "dimensions_scale": [scale, scale],
        "dimensions_unit": "pixels",
        "playing_area_corners": corners,
        "pocket_positions": pockets,
        "pocket_radius": 72 / scale,
        "pocket_radius_scale": [scale, scale],
        "pocket_radius_unit": "pixels",
        "real_world_dimensions": {"width": 2.54, "height": 1.27, "unit": "meters"},
        "coordinate_space": "pixel",
    }


def _state_payload(scale: float) -> dict[str, Any]:
    return {
        "source_resolution": [1920, 1080] if scale == 2 else [3840, 2160],
        "coordinate_space": "4k_canonical",
        "balls": [
            {
                "id": "cue",
                "number": 0,
                "is_cue_ball": True,
                "position": _scaled_point(1920 / scale, 1080 / scale, scale),
                "velocity": _scaled_point(4 / scale, 6 / scale, scale),
                "radius": 36 / scale,
                "radius_scale": [scale, scale],
            }
        ],
        "cue": {
            "tip_position": _scaled_point(1000 / scale, 1000 / scale, scale),
            "angle": 0,
            "length": 1.47,
            "length_unit": "meters",
        },
        "table": _table_payload(scale),
    }


def test_vector_geometry_uses_canonical_space_across_scales() -> None:
    source = Vector2D(960, 540, scale=(2, 2))
    canonical = Vector2D.from_4k(1920, 1080)
    source_neighbor = Vector2D(962, 540, scale=(2, 2))
    canonical_neighbor = Vector2D.from_4k(1924, 1080)

    assert source == canonical
    assert source_neighbor == canonical_neighbor
    assert source.distance_to(source_neighbor) == pytest.approx(4)
    assert canonical.distance_to(source_neighbor) == pytest.approx(4)
    assert source.angle_to(source_neighbor) == pytest.approx(0)
    assert source.magnitude() == pytest.approx(canonical.magnitude())
    assert hash(source) == hash(canonical)


def test_vector_rejects_non_finite_geometry_and_scale() -> None:
    with pytest.raises(ValueError, match="finite"):
        Vector2D(float("nan"), 0, scale=(1, 1))
    with pytest.raises(ValueError, match="finite"):
        Vector2D(0, 0, scale=(float("inf"), 1))


def test_vector_dict_rejects_non_pixel_space_even_with_scale() -> None:
    with pytest.raises(ValueError, match="Unsupported pixel coordinate space"):
        Vector2D.from_dict(
            {
                "x": 1.0,
                "y": 2.0,
                "scale": [1.0, 1.0],
                "coordinate_space": "world_meters",
            }
        )


def test_ball_state_normalizes_positions_velocities_and_radius() -> None:
    ball = BallState(
        id="source",
        position=Vector2D(960, 540, scale=(2, 2)),
        velocity=Vector2D(2, 3, scale=(2, 2)),
        radius=18,
        radius_scale=(2, 2),
    )

    assert ball.position == Vector2D.from_4k(1920, 1080)
    assert ball.velocity == Vector2D.from_4k(4, 6)
    assert ball.radius == pytest.approx(36)
    assert ball.radius_scale == (1, 1)
    assert ball.to_dict()["radius_unit"] == "canonical_pixels"
    assert ball.to_dict()["mass_unit"] == "kg"


def test_ball_rejects_anisotropic_radius_scale() -> None:
    with pytest.raises(ValueError, match="anisotropic"):
        BallState(
            id="bad-radius",
            position=Vector2D.from_4k(100, 100),
            radius=10,
            radius_scale=(2, 3),
        )


def test_table_normalizes_geometry_extents_and_radius_and_round_trips() -> None:
    table = TableState(
        width=1600,
        height=800,
        dimensions_scale=(2, 2),
        pocket_positions=[
            Vector2D(160, 140, scale=(2, 2)),
            Vector2D(960, 140, scale=(2, 2)),
            Vector2D(1760, 140, scale=(2, 2)),
            Vector2D(160, 940, scale=(2, 2)),
            Vector2D(960, 940, scale=(2, 2)),
            Vector2D(1760, 940, scale=(2, 2)),
        ],
        pocket_radius=36,
        pocket_radius_scale=(2, 2),
        playing_area_corners=[
            Vector2D(160, 140, scale=(2, 2)),
            Vector2D(1760, 140, scale=(2, 2)),
            Vector2D(1760, 940, scale=(2, 2)),
            Vector2D(160, 940, scale=(2, 2)),
        ],
        real_width_meters=2.54,
        real_height_meters=1.27,
    )

    assert table.width == pytest.approx(3200)
    assert table.height == pytest.approx(1600)
    assert table.pocket_radius == pytest.approx(72)
    assert table.pocket_positions[0] == Vector2D.from_4k(320, 280)
    assert table.playing_area_corners[0] == Vector2D.from_4k(320, 280)
    assert table.real_width_meters == pytest.approx(2.54)
    assert table.real_height_meters == pytest.approx(1.27)
    payload = table.to_dict()
    assert payload["dimensions_scale"] == [1, 1]
    assert payload["dimensions_unit"] == "canonical_pixels"
    assert payload["pocket_radius_scale"] == [1, 1]
    assert payload["pocket_radius_unit"] == "canonical_pixels"
    restored = TableState.from_dict(payload)
    assert restored.to_dict() == payload


def test_table_dict_uses_declared_resolution_for_points_extents_and_radius() -> None:
    source_table = _table_payload(scale=2)
    points = [
        {key: value for key, value in point.items() if key != "scale"}
        for point in source_table["playing_area_corners"]
    ]
    pocket_points = [
        {key: value for key, value in point.items() if key != "scale"}
        for point in source_table["pocket_positions"]
    ]
    payload = {
        "width": 1600,
        "height": 800,
        "dimensions_unit": "pixels",
        "pocket_radius": 36,
        "pocket_radius_unit": "pixels",
        "playing_area_corners": points,
        "pocket_positions": pocket_points,
        "real_world_dimensions": source_table["real_world_dimensions"],
        "coordinate_space": "pixel",
        "source_resolution": [1920, 1080],
    }

    table = TableState.from_dict(payload)

    assert table.width == pytest.approx(3200)
    assert table.height == pytest.approx(1600)
    assert table.pocket_radius == pytest.approx(72)
    assert table.playing_area_corners == [
        Vector2D.from_4k(320, 280),
        Vector2D.from_4k(3520, 280),
        Vector2D.from_4k(3520, 1880),
        Vector2D.from_4k(320, 1880),
    ]


def test_table_uses_explicit_units_instead_of_magnitude_heuristics() -> None:
    pockets = [Vector2D.from_4k(i, i) for i in range(6)]
    table = TableState(
        width=2.54,
        height=1.27,
        pocket_positions=pockets,
        real_width_meters=2.54,
        real_height_meters=1.27,
    )

    assert table.width == pytest.approx(2.54)
    assert table.height == pytest.approx(1.27)
    with pytest.raises(ValueError, match="pixel units"):
        TableState.from_dict(
            {
                "width": 2.54,
                "height": 1.27,
                "dimensions_unit": "meters",
                "pocket_positions": [point.to_dict() for point in pockets],
            }
        )


def test_calibration_produces_equivalent_4k_geometry_and_keeps_meters() -> None:
    corners_1080 = [(160, 140), (1760, 140), (1760, 940), (160, 940)]
    corners_4k = [(320, 280), (3520, 280), (3520, 1880), (320, 1880)]
    table_1080 = TableState.from_calibration(
        corners_1080,
        (1920, 1080),
        table_dimensions_real=(2.54, 1.27),
    )
    table_4k = TableState.from_calibration(
        corners_4k,
        (3840, 2160),
        table_dimensions_real=(2.54, 1.27),
    )

    assert table_1080.width == pytest.approx(table_4k.width)
    assert table_1080.height == pytest.approx(table_4k.height)
    assert table_1080.pocket_radius == pytest.approx(table_4k.pocket_radius)
    assert table_1080.playing_area_corners == table_4k.playing_area_corners
    assert table_1080.real_width_meters == pytest.approx(2.54)
    assert table_1080.real_height_meters == pytest.approx(1.27)


def test_game_state_manager_normalizes_all_incoming_geometry(tmp_path) -> None:
    states = []
    for scale in (2, 1):
        manager = GameStateManager(persistence_path=tmp_path / str(scale))
        states.append(manager.update_state(_state_payload(scale)))

    low_resolution, canonical = states
    low_ball, canonical_ball = low_resolution.balls[0], canonical.balls[0]
    assert low_ball.position == canonical_ball.position
    assert low_ball.velocity == canonical_ball.velocity
    assert low_ball.radius == canonical_ball.radius
    assert low_resolution.cue.tip_position == canonical.cue.tip_position
    assert low_resolution.table.width == canonical.table.width
    assert low_resolution.table.height == canonical.table.height
    assert low_resolution.table.pocket_positions == canonical.table.pocket_positions
    assert low_resolution.table.pocket_radius == canonical.table.pocket_radius
    assert (
        low_resolution.table.playing_area_corners
        == canonical.table.playing_area_corners
    )
    assert low_resolution.coordinate_metadata.table_bounds == (
        320,
        280,
        3520,
        1880,
    )
    metadata_payload = low_resolution.coordinate_metadata.to_dict()
    assert metadata_payload["table_bounds_scale"] == [1, 1]
    assert metadata_payload["table_bounds_unit"] == "canonical_pixels"
    assert metadata_payload["coordinate_space"] == "4k_canonical"


def test_legacy_payload_without_resolution_uses_configured_camera_size(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setattr(
        config,
        "get",
        lambda key, default=None: (
            [1920, 1080] if key == "vision.camera.resolution" else default
        ),
    )
    payload = _state_payload(1)
    payload.pop("source_resolution")
    payload["table"]["coordinate_space"] = "4k_canonical"
    payload["table"]["dimensions_unit"] = "canonical_pixels"

    manager = GameStateManager(persistence_path=tmp_path)
    state = manager.update_state(payload)

    assert state.coordinate_metadata is not None
    assert state.coordinate_metadata.camera_resolution == (1920, 1080)


@pytest.mark.parametrize("ball_status", ["failed", "unavailable", "model_unavailable"])
def test_incomplete_ball_observation_retains_authoritative_core_balls(
    tmp_path: Path, ball_status: str
) -> None:
    manager = GameStateManager(persistence_path=tmp_path)
    previous = manager.update_state(_state_payload(2))
    previous_ball = previous.balls[0]

    incomplete = deepcopy(_state_payload(2))
    incomplete["balls"] = []
    incomplete["observation_status"] = {"ball_detection_status": ball_status}
    # Ball failure must not prevent independent cue/table data from updating.
    incomplete["cue"] = None
    incomplete["table"]["pocket_radius"] = 40

    updated = manager.update_state(incomplete)

    assert len(updated.balls) == 1
    retained_ball = updated.balls[0]
    assert retained_ball.id == previous_ball.id
    assert retained_ball.position == previous_ball.position
    assert retained_ball.last_update == previous_ball.last_update
    assert retained_ball is not previous_ball
    assert updated.cue is None
    assert updated.table.pocket_radius == pytest.approx(80)


def test_successful_empty_ball_observation_clears_core_balls(tmp_path: Path) -> None:
    manager = GameStateManager(persistence_path=tmp_path)
    manager.update_state(_state_payload(2))

    observed_empty = deepcopy(_state_payload(2))
    observed_empty["balls"] = []
    observed_empty["observation_status"] = {"ball_detection_status": "observed"}

    updated = manager.update_state(observed_empty)

    assert updated.balls == []


def test_game_state_manager_rejects_unscaled_unknown_points(tmp_path) -> None:
    manager = GameStateManager(persistence_path=tmp_path)
    with pytest.raises(ValueError, match="missing coordinate scale metadata"):
        manager._extract_ball_states(
            {"balls": [{"id": "unknown", "position": {"x": 100, "y": 200}}]}
        )


def test_manager_uses_exact_resolution_for_pixel_geometry_missing_point_scales(
    tmp_path,
) -> None:
    manager = GameStateManager(persistence_path=tmp_path)
    manager._default_table = None
    detection_data = {
        "source_resolution": [1920, 1080],
        "coordinate_space": "pixel",
        "balls": [
            {
                "id": "ball",
                "position": {"x": 960, "y": 540},
                "velocity": {"x": 0, "y": 0},
                "radius": 18,
                "radius_scale": [2, 2],
                "radius_unit": "pixels",
            }
        ],
        "table": _table_payload(2),
    }
    table_payload = detection_data["table"]
    table_payload.pop("dimensions_scale")
    table_payload["dimensions_unit"] = "pixels"
    table_payload["pocket_radius_unit"] = "pixels"
    table_payload["coordinate_space"] = "pixel"
    table_payload["source_resolution"] = [1920, 1080]

    ball = manager._extract_ball_states(detection_data)[0]
    table = manager._extract_table_state(detection_data)

    assert ball.position == Vector2D.from_4k(1920, 1080)
    assert ball.radius == pytest.approx(36)
    assert table.width == pytest.approx(3200)
    assert table.height == pytest.approx(1600)
    assert table.pocket_radius == pytest.approx(72)


def test_table_bounds_require_explicit_scale_or_source_resolution() -> None:
    metadata = CoordinateMetadata.from_dict(
        {
            "camera_resolution": [1920, 1080],
            "table_bounds": [100, 100, 1820, 980],
            "table_bounds_scale": [2, 2],
            "table_bounds_unit": "pixels",
            "coordinate_space": "pixel",
        }
    )
    assert metadata.table_bounds == (200, 200, 3640, 1960)
    with pytest.raises(ValueError, match="scale metadata or a source resolution"):
        CoordinateMetadata.from_dict(
            {
                "table_bounds": [100, 100, 1820, 980],
                "table_bounds_unit": "pixels",
                "coordinate_space": "pixel",
            }
        )


def test_pixels_per_meter_requires_calibrated_real_dimensions() -> None:
    pockets = [Vector2D.from_4k(i, i) for i in range(6)]
    table = TableState(width=3200, height=1600, pocket_positions=pockets)
    assert table.calculate_pixels_per_meter() is None
    calibrated = TableState(
        width=3200,
        height=1600,
        pocket_positions=pockets,
        real_width_meters=2.54,
        real_height_meters=1.27,
    )
    assert calibrated.calculate_pixels_per_meter() == pytest.approx(
        (3200 / 2.54, 1600 / 1.27)
    )


def test_coordinate_metadata_normalizes_and_serializes_boundary_scale() -> None:
    metadata = CoordinateMetadata(
        table_bounds=(50, 25, 900, 450),
        table_bounds_scale=(2, 2),
        coordinate_space="4k_canonical",
    )

    assert metadata.table_bounds == (100, 50, 1800, 900)
    payload = metadata.to_dict()
    assert payload["table_bounds_scale"] == [1, 1]
    restored = CoordinateMetadata.from_dict(payload)
    assert restored.table_bounds == metadata.table_bounds
