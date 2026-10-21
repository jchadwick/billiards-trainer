"""Coordinate contract tests for trajectory WebSocket broadcasts."""

from unittest.mock import AsyncMock

import pytest

from backend.api.websocket.broadcaster import MessageBroadcaster


def _point(x: float, y: float, scale: tuple[float, float] = (1.0, 1.0)):
    return {"x": x, "y": y, "scale": list(scale)}


def _lines():
    return [
        {
            "start": _point(10.0, 20.0),
            "end": _point(30.0, 40.0),
            "type": "primary",
        },
        {"start": _point(30.0, 40.0), "end": _point(50.0, 60.0)},
    ]


@pytest.mark.asyncio
async def test_broadcast_accepts_scaled_trajectory_endpoints_and_collisions():
    broadcaster = MessageBroadcaster()
    broadcaster._broadcast_to_stream = AsyncMock()
    collisions = [
        {
            "type": "ball_ball",
            "position": _point(30.0, 40.0, (0.5, 0.25)),
            "ball1_id": "cue",
            "ball2_id": "2",
        }
    ]

    await broadcaster.broadcast_trajectory(_lines(), collisions)

    broadcaster._broadcast_to_stream.assert_awaited_once()
    payload = broadcaster._broadcast_to_stream.await_args.args[1]
    assert payload["lines"] == _lines()
    assert payload["collisions"] == collisions
    assert "is_reference" not in payload
    assert broadcaster.broadcast_stats.validation_failures == 0


@pytest.mark.parametrize(
    "bad_point",
    [
        {"x": 1.0, "y": 2.0},
        {"x": float("nan"), "y": 2.0, "scale": [1.0, 1.0]},
        {"x": 1.0, "y": float("inf"), "scale": [1.0, 1.0]},
        {"x": 1.0, "y": 2.0, "scale": [0.0, 1.0]},
        {"x": 1.0, "y": 2.0, "scale": [1.0, float("inf")]},
        {"x": True, "y": 2.0, "scale": [1.0, 1.0]},
        {"x": 1.0, "y": 2.0, "scale": [1.0]},
    ],
)
@pytest.mark.asyncio
async def test_broadcast_rejects_invalid_line_endpoint_scale(bad_point):
    broadcaster = MessageBroadcaster()
    broadcaster._broadcast_to_stream = AsyncMock()
    lines = _lines()
    lines[0]["end"] = bad_point

    await broadcaster.broadcast_trajectory(lines)

    broadcaster._broadcast_to_stream.assert_not_awaited()
    assert broadcaster.broadcast_stats.validation_failures == 1


@pytest.mark.parametrize(
    "bad_position",
    [
        [30.0, 40.0],
        {"x": 30.0, "y": 40.0},
        {"x": 30.0, "y": 40.0, "scale": [1.0, -1.0]},
    ],
)
@pytest.mark.asyncio
async def test_broadcast_rejects_invalid_collision_position(bad_position):
    broadcaster = MessageBroadcaster()
    broadcaster._broadcast_to_stream = AsyncMock()
    collisions = [{"type": "ball_ball", "position": bad_position}]

    await broadcaster.broadcast_trajectory(_lines(), collisions)

    broadcaster._broadcast_to_stream.assert_not_awaited()
    assert broadcaster.broadcast_stats.validation_failures == 1


@pytest.mark.asyncio
async def test_broadcast_rejects_collision_without_nested_scaled_position():
    broadcaster = MessageBroadcaster()
    broadcaster._broadcast_to_stream = AsyncMock()
    collisions = [
        {
            "type": "ball_ball",
            "x": 30.0,
            "y": 40.0,
        }
    ]

    await broadcaster.broadcast_trajectory(_lines(), collisions)

    broadcaster._broadcast_to_stream.assert_not_awaited()
    assert broadcaster.broadcast_stats.validation_failures == 1


@pytest.mark.asyncio
async def test_broadcast_rejects_line_without_both_scaled_endpoints():
    broadcaster = MessageBroadcaster()
    broadcaster._broadcast_to_stream = AsyncMock()

    await broadcaster.broadcast_trajectory(
        [
            {"start": _point(0.0, 0.0)},
            {"start": _point(1.0, 1.0), "end": _point(2.0, 2.0)},
        ]
    )

    broadcaster._broadcast_to_stream.assert_not_awaited()
    assert broadcaster.broadcast_stats.validation_failures == 1
