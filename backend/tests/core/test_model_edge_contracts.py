"""Behavioral edge contracts for the core state and prediction models.

The larger model suite covers the ordinary construction paths. This file
concentrates on round-trip boundaries, optional fields, and malformed payloads
that can otherwise make old or partial state look valid.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from backend.core.coordinates import Vector2D
from backend.core.models import (
    BallState,
    Collision,
    CoordinateMetadata,
    CueState,
    GameEvent,
    GameState,
    GameType,
    ShotAnalysis,
    ShotType,
    TableState,
    Trajectory,
    calculate_ball_separation,
    create_standard_ball_set,
    deserialize_from_json,
    find_closest_ball,
    serialize_to_json,
    validate_ball_physics,
)


def _pockets() -> list[Vector2D]:
    return [
        Vector2D.from_4k(x, y)
        for x, y in (
            (200, 200),
            (1920, 200),
            (3640, 200),
            (200, 1960),
            (1920, 1960),
            (3640, 1960),
        )
    ]


def _table(**kwargs: Any) -> TableState:
    return TableState(
        width=3840,
        height=2160,
        pocket_positions=_pockets(),
        **kwargs,
    )


def _ball_payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": "cue",
        "position": {"x": 1000.0, "y": 1000.0, "scale": [1.0, 1.0]},
        "velocity": {"x": 0.0, "y": 0.0, "scale": [1.0, 1.0]},
        "spin": None,
        "radius": 36.0,
        "radius_scale": [1.0, 1.0],
        "radius_unit": "canonical_pixels",
        "mass": 0.17,
        "mass_unit": "kg",
        "last_update": 12.5,
    }
    payload.update(overrides)
    return payload


def _table_payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "width": 3840.0,
        "height": 2160.0,
        "dimensions_scale": [1.0, 1.0],
        "dimensions_unit": "canonical_pixels",
        "pocket_positions": [pocket.to_dict() for pocket in _pockets()],
        "pocket_radius": 72.0,
        "pocket_radius_scale": [1.0, 1.0],
        "pocket_radius_unit": "canonical_pixels",
        "cushion_height": 0.025,
        "cushion_height_unit": "meters",
    }
    payload.update(overrides)
    return payload


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"last_seen": 1.0}, "timestamps must use last_update"),
        ({"mass_unit": "grams"}, "mass must use kg"),
        ({"radius_scale": None}, "two scale factors"),
        ({"radius_unit": "pixels"}, "radius must use canonical_pixels"),
        ({"velocity": None}, "no attribute 'get'"),
    ],
)
def test_ball_deserialization_rejects_legacy_or_incomplete_physics_payloads(
    changes: dict[str, Any], message: str
) -> None:
    payload = _ball_payload(**changes)

    with pytest.raises((ValueError, TypeError, AttributeError), match=message):
        BallState.from_dict(payload)


def test_ball_deserialization_defaults_optional_spin_and_observation_flags() -> None:
    ball = BallState.from_dict(_ball_payload())

    assert ball.spin == Vector2D.from_4k(0, 0)
    assert ball.is_observed is True
    assert ball.radius_scale == (1.0, 1.0)


def test_ball_copy_preserves_optional_spin_and_is_deep_for_vectors() -> None:
    original = BallState.from_4k("object", 1200, 900, spin=None, is_observed=False)
    copied = original.copy()

    assert copied is not original
    assert copied.position is not original.position
    assert copied.velocity is not original.velocity
    assert copied.spin is None
    assert copied.is_observed is False


def test_ball_factory_scales_explicit_radius_but_keeps_default_radius_canonical() -> (
    None
):
    scaled = BallState.from_resolution("scaled", 480, 270, (1920, 1080), radius=18)
    defaulted = BallState.from_resolution("default", 480, 270, (1920, 1080))

    assert scaled.position == Vector2D.from_4k(960, 540)
    assert scaled.radius == pytest.approx(36)
    assert defaulted.radius == pytest.approx(36)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("rail_height", 0.04, "Unsupported legacy table fields: rail_height"),
        ("dimensions_unit", "meters", "Table dimensions must use pixel units"),
        ("coordinate_space", "normalized", "Unsupported table coordinate space"),
        ("cushion_height_unit", "pixels", "Cushion height must use meters"),
        ("pocket_radius_unit", "meters", "Pocket radius must use pixel units"),
        ("pocket_radius_scale", None, "Pocket radius scale metadata is required"),
    ],
)
def test_table_deserialization_rejects_incompatible_units_and_legacy_fields(
    field: str, value: Any, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        TableState.from_dict(_table_payload(**{field: value}))


def test_table_deserialization_converts_pixel_geometry_from_source_resolution() -> None:
    pixel_pockets = [{"x": p.x / 2, "y": p.y / 2} for p in _pockets()]
    table = TableState.from_dict(
        _table_payload(
            width=1920,
            height=1080,
            dimensions_scale=None,
            dimensions_unit="pixels",
            coordinate_space="pixel",
            source_resolution=[1920, 1080],
            pocket_positions=pixel_pockets,
            pocket_radius=36,
            pocket_radius_scale=None,
            pocket_radius_unit="pixels",
        )
    )

    assert (table.width, table.height) == (3840, 2160)
    assert table.pocket_radius == pytest.approx(72)
    assert [(point.x, point.y) for point in table.pocket_positions] == [
        (point.x, point.y) for point in _pockets()
    ]


def test_table_deserialization_requires_explicit_scale_for_canonical_dimensions() -> (
    None
):
    with pytest.raises(ValueError, match="Table dimensions scale metadata is required"):
        TableState.from_dict(
            _table_payload(dimensions_scale=None, dimensions_unit="canonical_pixels")
        )


def test_table_deserialization_handles_real_dimensions_and_empty_playing_area() -> None:
    table = TableState.from_dict(
        _table_payload(
            real_world_dimensions={"width": 2.54, "height": 1.27, "unit": "meters"},
            playing_area_corners=[],
        )
    )

    assert table.calculate_pixels_per_meter() == pytest.approx(
        (3840 / 2.54, 2160 / 1.27)
    )
    assert table.playing_area_corners is None
    assert table.to_dict()["real_world_dimensions"] == {
        "width": 2.54,
        "height": 1.27,
        "unit": "meters",
    }


def test_table_deserialization_accepts_explicit_real_dimension_tuple() -> None:
    table = TableState.from_dict(
        _table_payload(
            real_world_dimensions=[2.0, 1.0],
            real_world_dimensions_unit="meters",
        )
    )

    assert table.calculate_pixels_per_meter() == (1920, 2160)


@pytest.mark.parametrize(
    ("real_dimensions", "extra", "message"),
    [
        ({"width": 2, "height": 1, "unit": "feet"}, {}, "must use meters"),
        ([2, 1], {}, "require an explicit meters unit"),
        ([2], {"real_world_dimensions_unit": "meters"}, "not enough values"),
    ],
)
def test_table_deserialization_does_not_guess_real_world_units(
    real_dimensions: Any, extra: dict[str, Any], message: str
) -> None:
    payload = _table_payload(real_world_dimensions=real_dimensions, **extra)

    with pytest.raises((ValueError, TypeError), match=message):
        TableState.from_dict(payload)


def test_table_calibration_uses_supplied_pockets_and_real_dimensions() -> None:
    corners = [(100.0, 100.0), (1700.0, 100.0), (1700.0, 900.0), (100.0, 900.0)]
    pockets = [
        (float(x), float(y))
        for x, y in (
            (100, 100),
            (900, 100),
            (1700, 100),
            (100, 900),
            (900, 900),
            (1700, 900),
        )
    ]
    table = TableState.from_calibration(
        corners,
        (1920, 1080),
        table_dimensions_real=(2.0, 1.0),
        pocket_positions_pixel=pockets,
    )

    assert len(table.pocket_positions) == 6
    assert table.playing_area_corners is not None
    assert table.width == pytest.approx(3200)
    assert table.height == pytest.approx(1600)
    assert table.calculate_pixels_per_meter() == pytest.approx((1600, 1600))


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (([(0, 0)] * 3, (1920, 1080), None, None), "exactly four table corners"),
        (([(0, 0)] * 4, (1920,), None, None), "positive width and height"),
        (([(0, 0)] * 4, (1920, 1080), (2,), None), "width/height pair"),
        (([(0, 0)] * 4, (1920, 1080), (-2, 1), None), "positive meters"),
    ],
)
def test_table_calibration_rejects_incomplete_metadata(
    args: tuple[Any, Any, Any, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        TableState.from_calibration(*args)


def test_playing_area_handles_clockwise_order_degenerate_edges_and_rectangular_fallback() -> (
    None
):
    clockwise = [
        Vector2D.from_4k(x, y)
        for x, y in ((100, 100), (100, 1500), (3100, 1500), (3100, 100))
    ]
    table = _table(playing_area_corners=clockwise)
    assert table.is_point_in_playing_area(Vector2D.from_4k(1600, 800), margin=100)

    degenerate = _table(
        playing_area_corners=[
            Vector2D.from_4k(100, 100),
            Vector2D.from_4k(100, 100),
            Vector2D.from_4k(100, 100),
            Vector2D.from_4k(100, 100),
        ]
    )
    assert not degenerate.is_point_in_playing_area(Vector2D.from_4k(100, 100))
    assert _table().is_point_in_playing_area(Vector2D.from_4k(1000, 1000))


@pytest.mark.parametrize(
    ("target", "message"),
    [((0, 720), "dimensions must be positive"), ((1920, 1080.5), "whole pixels")],
)
def test_table_corner_rescaling_rejects_invalid_target_resolution(
    target: tuple[float, float], message: str
) -> None:
    table = _table(
        playing_area_corners=[
            Vector2D.from_4k(x, y)
            for x, y in ((0, 0), (1000, 0), (1000, 500), (0, 500))
        ]
    )

    with pytest.raises(ValueError, match=message):
        table.scale_playing_area_corners(3840, 2160, *target)


def test_closest_cushion_uses_calibrated_bounds_and_returns_inward_normal() -> None:
    table = _table(
        playing_area_corners=[
            Vector2D.from_4k(x, y)
            for x, y in ((100, 100), (3500, 100), (3500, 2000), (100, 2000))
        ]
    )

    side, distance, normal = table.get_closest_cushion(Vector2D.from_4k(500, 120))

    assert side == "top"
    assert distance == pytest.approx(20)
    assert normal == Vector2D.from_4k(0, -1)


def test_coordinate_metadata_infers_pixel_scale_from_camera_resolution() -> None:
    metadata = CoordinateMetadata.from_dict(
        {
            "camera_resolution": [1920, 1080],
            "table_bounds": [10, 20, 900, 500],
            "table_bounds_unit": "pixels",
            "coordinate_space": "pixel",
        }
    )

    assert metadata.camera_resolution == (1920, 1080)
    assert metadata.table_bounds == (20, 40, 1800, 1000)
    assert metadata.to_dict()["table_bounds_unit"] == "canonical_pixels"


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"table_bounds": [0, 0, 1, 1]}, "declared coordinate space"),
        (
            {"coordinate_space": "normalized", "table_bounds": [0, 0, 1, 1]},
            "Unsupported table-bound coordinate space",
        ),
        (
            {
                "coordinate_space": "pixel",
                "table_bounds": [0, 0, 1, 1],
                "table_bounds_unit": "meters",
                "table_bounds_scale": [1, 1],
            },
            "Table bounds must use pixel units",
        ),
        (
            {
                "coordinate_space": "4k_canonical",
                "table_bounds": [0, 0, 1, 1],
                "table_bounds_unit": "canonical_pixels",
            },
            "scale metadata or a source resolution",
        ),
    ],
)
def test_coordinate_metadata_requires_explicit_bounds_semantics(
    payload: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        CoordinateMetadata.from_dict(payload)


def test_coordinate_metadata_rejects_reversed_bounds_and_bad_resolution() -> None:
    with pytest.raises(ValueError, match="minimums must not exceed maximums"):
        CoordinateMetadata(table_bounds=(20, 0, 10, 5), table_bounds_scale=(1, 1))
    with pytest.raises(ValueError):
        CoordinateMetadata(camera_resolution=(0, 1080))


def test_cue_payload_preserves_optional_butt_and_impact_points() -> None:
    cue = CueState.from_dict(
        {
            "tip_position": {"x": 100, "y": 200, "scale": [1, 1]},
            "butt_position": {"x": 500, "y": 300, "scale": [1, 1]},
            "impact_point": {"x": 120, "y": 200, "scale": [1, 1]},
            "angle": 0,
            "estimated_force": 20,
        }
    )

    assert cue.to_dict()["butt_position"]["x"] == 500
    assert cue.impact_point == Vector2D.from_4k(120, 200)
    assert cue.calculate_impact_velocity() == Vector2D.from_4k(2, 0)
    assert cue.get_aim_line(length=10)[1] == Vector2D.from_4k(110, 200)


def test_game_event_defaults_and_state_round_trip_nested_optional_records() -> None:
    event = GameEvent.from_dict(
        {"timestamp": 1.0, "event_type": "shot", "description": "break"}
    )
    shot = ShotAnalysis(
        shot_type=ShotType.BANK,
        difficulty=0.4,
        success_probability=0.7,
        recommended_force=12,
        recommended_angle=30,
        cue_ball_final_position=Vector2D.from_4k(400, 500),
        alternative_shots=[
            ShotAnalysis(
                shot_type=ShotType.SAFETY,
                difficulty=0.6,
                success_probability=0.8,
                recommended_force=5,
                recommended_angle=10,
            )
        ],
    )
    state = GameState(
        timestamp=1,
        frame_number=4,
        balls=[BallState.from_4k("cue", 1000, 1000, is_cue_ball=True)],
        table=_table(),
        cue=CueState.from_4k(900, 1000),
        game_type=GameType.NINE_BALL,
        last_shot=shot,
        events=[event],
        coordinate_metadata=CoordinateMetadata(camera_resolution=(3840, 2160)),
    )

    restored = GameState.from_dict(json.loads(serialize_to_json(state)))

    assert restored.last_shot is not None
    assert len(restored.last_shot.alternative_shots) == 1
    assert restored.last_shot.alternative_shots[0].alternative_shots == []
    assert restored.last_shot.cue_ball_final_position == Vector2D.from_4k(400, 500)
    assert restored.events[0].frame_number == 0
    assert restored.cue is not None
    assert restored.coordinate_metadata is not None


def test_state_copy_isolates_mutable_ball_events_scores_and_errors() -> None:
    state = GameState(
        timestamp=1,
        frame_number=1,
        balls=[BallState.from_4k("cue", 1000, 1000, is_cue_ball=True)],
        table=_table(),
        scores={1: 2},
        events=[GameEvent(1, "shot", "one")],
        validation_errors=["initial"],
    )

    duplicate = state.copy()
    duplicate.balls[0].position.x = 2000
    duplicate.scores[1] = 9
    duplicate.events[0].description = "changed"
    duplicate.validation_errors.append("new")

    assert state.balls[0].position.x == 1000
    assert state.scores == {1: 2}
    assert state.events[0].description == "one"
    assert state.validation_errors == ["initial"]


def test_collision_round_trip_covers_velocity_map_and_non_ball_collision() -> None:
    collision = Collision(
        time=0.5,
        position=Vector2D.from_4k(100, 200),
        ball1_id="cue",
        type="cushion",
        resulting_velocities={"cue": Vector2D.from_4k(3, 4)},
    )
    restored = Collision.from_dict(collision.to_dict())

    assert restored.is_cushion_collision()
    assert not restored.is_ball_collision()
    assert restored.get_involved_balls() == ["cue"]
    assert restored.resulting_velocities == {"cue": Vector2D.from_4k(3, 4)}
    assert Collision.from_dict(
        {
            "time": 0,
            "position": Vector2D.from_4k(0, 0).to_dict(),
            "ball1_id": "cue",
            "type": "pocket",
        }
    ).is_pocket_collision()


def test_trajectory_queries_cover_empty_single_point_and_collision_ordering() -> None:
    start = Vector2D.from_4k(0, 0)
    middle = Vector2D.from_4k(10, 0)
    end = Vector2D.from_4k(20, 0)
    past = Collision(-1, start, "cue", type="pocket")
    soon = Collision(0.5, middle, "cue", type="pocket")
    later = Collision(1.5, end, "cue", type="pocket")

    trajectory = Trajectory(
        ball_id="cue",
        points=[start, middle, end],
        collisions=[later, past, soon],
        final_position=end,
        final_velocity=Vector2D.from_4k(0, 0),
        time_to_rest=2,
        will_be_pocketed=False,
    )
    assert trajectory.total_distance == pytest.approx(20)
    assert trajectory.get_position_at_time(-1) is None
    assert trajectory.get_position_at_time(3) is None
    assert trajectory.get_position_at_time(1) == middle
    assert trajectory.get_position_at_time(2) is end
    assert trajectory.get_collisions_before_time(0.5) == [past, soon]
    assert trajectory.get_next_collision() is soon

    no_points = Trajectory("empty", [], [], end, start, 1, False)
    one_point = Trajectory("still", [middle], [], end, start, 0, False)
    assert no_points.get_position_at_time(0) is None
    assert one_point.get_position_at_time(0) is end
    assert Trajectory("none", [], [], end, start, 1, False).get_next_collision() is None


def test_shot_analysis_bounds_risk_and_deduplicates_problems() -> None:
    shot = ShotAnalysis(ShotType.DIRECT, 0.5, 0.2, 8, 0)
    shot.add_problem("blocked")
    shot.add_problem("blocked")
    shot.add_risk("scratch", 2.0)
    shot.add_risk("miss", -1.0)

    assert shot.potential_problems == ["blocked"]
    assert shot.risk_assessment == {"scratch": 1.0, "miss": 0.0}
    assert shot.is_high_risk()
    assert shot.get_risk_score() == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"difficulty": -0.1}, "Difficulty must be between"),
        ({"success_probability": 1.1}, "Success probability must be between"),
        ({"shot_confidence": -0.1}, "Shot confidence must be between"),
        ({"recommended_force": -1}, "Recommended force must be non-negative"),
    ],
)
def test_shot_analysis_rejects_values_outside_documented_ranges(
    changes: dict[str, float], message: str
) -> None:
    options: dict[str, Any] = {
        "shot_type": ShotType.DIRECT,
        "difficulty": 0.5,
        "success_probability": 0.5,
        "recommended_force": 1,
        "recommended_angle": 0,
    }
    options.update(changes)

    with pytest.raises(ValueError, match=message):
        ShotAnalysis(**options)


def test_serialized_game_state_round_trip_keeps_empty_optional_sections() -> None:
    state = GameState(
        timestamp=0,
        frame_number=0,
        balls=[],
        table=_table(),
    )
    restored = deserialize_from_json(serialize_to_json(state), GameState)

    assert restored.cue is None
    assert restored.last_shot is None
    assert restored.events == []
    assert restored.coordinate_metadata is None


def test_utility_functions_ignore_self_and_pocketed_balls_and_report_physics_errors() -> (
    None
):
    target = BallState.from_4k("target", 1000, 1000)
    pocketed = BallState.from_4k("pocketed", 1001, 1000, is_pocketed=True)
    farther = BallState.from_4k("farther", 1100, 1000)

    assert find_closest_ball(target, []) is None
    assert find_closest_ball(target, [target, pocketed, farther]) is farther
    assert calculate_ball_separation(target, farther) == pytest.approx(28)
    target.velocity = Vector2D.from_4k(11, 0)
    target.mass = 0
    target.radius = 0
    assert validate_ball_physics(target) == [
        "Ball target velocity 11.00 exceeds maximum 10.0",
        "Ball target has invalid mass: 0",
        "Ball target has invalid radius: 0",
    ]


def test_standard_ball_set_contains_one_cue_and_all_numbered_object_balls() -> None:
    balls = create_standard_ball_set()

    assert len(balls) == 16
    assert [ball.number for ball in balls if ball.is_cue_ball] == [None]
    assert {ball.number for ball in balls if not ball.is_cue_ball} == set(range(1, 16))
