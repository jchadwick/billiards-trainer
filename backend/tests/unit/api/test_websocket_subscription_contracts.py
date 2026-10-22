"""Focused contracts for subscription filtering, freshness, and lifecycle."""

from datetime import datetime, timezone

import pytest

from backend.api.websocket.manager import StreamType
from backend.api.websocket.schemas import QualityLevel
from backend.api.websocket.subscriptions import (
    FilterCondition,
    FilterOperator,
    SubscriptionManager,
)


@pytest.mark.parametrize(
    ("operator", "actual", "expected", "matches"),
    [
        (FilterOperator.EQUALS, 4, 4, True),
        (FilterOperator.NOT_EQUALS, 4, 5, True),
        (FilterOperator.GREATER_THAN, 5, 4, True),
        (FilterOperator.GREATER_EQUAL, 4, 4, True),
        (FilterOperator.LESS_THAN, 3, 4, True),
        (FilterOperator.LESS_EQUAL, 4, 4, True),
        (FilterOperator.IN, "eight", ["eight", "nine"], True),
        (FilterOperator.NOT_IN, "eight", ["nine"], True),
        (FilterOperator.CONTAINS, "8-ball", "ball", True),
        (FilterOperator.MATCHES, "ball-8", r"^ball-\d$", True),
        (FilterOperator.EQUALS, 4, 5, False),
    ],
)
def test_filter_operators_evaluate_nested_fields(operator, actual, expected, matches):
    manager = SubscriptionManager()
    condition = FilterCondition("balls.0.label", operator, expected)

    assert (
        manager._evaluate_filter_condition({"balls": [{"label": actual}]}, condition)
        is matches
    )


@pytest.mark.parametrize(
    ("data", "condition"),
    [
        ({"cue": {}}, FilterCondition("cue.angle", FilterOperator.EQUALS, 0)),
        (
            {"balls": [{"position": "unknown"}]},
            FilterCondition("balls.0.position.x", FilterOperator.GREATER_THAN, 0),
        ),
        (
            {"score": "not-a-number"},
            FilterCondition("score", FilterOperator.GREATER_THAN, 0),
        ),
    ],
)
def test_filter_rejects_missing_nested_fields_and_incompatible_values(data, condition):
    assert not SubscriptionManager()._evaluate_filter_condition(data, condition)


@pytest.mark.asyncio
async def test_create_subscription_recovers_from_malformed_optional_configuration():
    manager = SubscriptionManager()

    assert not await manager.create_subscription("client", "not-a-stream")
    assert await manager.create_subscription(
        "client",
        "state",
        quality_level="not-a-quality",
        sample_rate=3.0,
        filter_conditions=[
            {"field": "score", "operator": "unknown", "value": 1},
            {"field": "balls.0.visible", "operator": "eq", "value": True},
            {"operator": "eq", "value": 1},
        ],
    )

    subscription = manager.subscriptions["client"][StreamType.STATE]
    assert subscription.quality_level is QualityLevel.AUTO
    assert subscription.sample_rate == 1.0
    assert len(subscription.filter_conditions) == 1
    assert await manager.process_message(
        StreamType.STATE, {"score": 2, "balls": [{"visible": True}]}
    ) == {"client": {"score": 2, "balls": [{"visible": True}]}}


@pytest.mark.parametrize("configured_capacity", [True, False, 0, -2, 2.5, "4"])
def test_invalid_aggregation_capacity_uses_safe_default(
    monkeypatch, configured_capacity
):
    from backend.api.websocket import subscriptions

    monkeypatch.setattr(
        subscriptions.config,
        "get",
        lambda _key, default=None: configured_capacity,
    )

    assert SubscriptionManager().aggregation_buffer_capacity == 32


@pytest.mark.asyncio
async def test_sampling_is_per_client_and_keeps_each_clients_current_sample():
    manager = SubscriptionManager()
    assert await manager.create_subscription(
        "first", StreamType.STATE, sample_rate=0.5, sample_offset=0.0
    )
    assert await manager.create_subscription(
        "second", StreamType.STATE, sample_rate=0.5, sample_offset=0.5
    )

    outputs = [
        await manager.process_message(StreamType.STATE, {"frame": frame})
        for frame in range(6)
    ]

    assert outputs == [
        {"first": {"frame": 0}},
        {"second": {"frame": 1}},
        {"first": {"frame": 2}},
        {"second": {"frame": 3}},
        {"first": {"frame": 4}},
        {"second": {"frame": 5}},
    ]
    # Offsets partition the cadence while each client receives only the
    # current input value; skipped observations are never buffered for replay.
    assert manager.subscriptions["first"][StreamType.STATE].messages_filtered == 3
    assert manager.subscriptions["second"][StreamType.STATE].messages_filtered == 3
    assert manager.sample_counters["first"][StreamType.STATE] == 6
    assert manager.sample_counters["second"][StreamType.STATE] == 6


@pytest.mark.asyncio
async def test_zero_sample_rate_filters_every_observation_without_retaining_history():
    manager = SubscriptionManager()
    assert await manager.create_subscription("client", StreamType.STATE, sample_rate=0)

    assert await manager.process_message(StreamType.STATE, {"frame": 1}) == {}
    assert await manager.process_message(StreamType.STATE, {"frame": 2}) == {}

    subscription = manager.subscriptions["client"][StreamType.STATE]
    assert subscription.messages_sent == 0
    assert subscription.messages_filtered == 2
    assert not manager.aggregation_buffers


@pytest.mark.asyncio
async def test_frame_quality_is_adjusted_per_subscriber_without_mutating_input():
    manager = SubscriptionManager()
    for client, quality in (
        ("low", "low"),
        ("medium", "medium"),
        ("high", "high"),
        ("auto", "auto"),
    ):
        assert await manager.create_subscription(
            client, StreamType.FRAME, quality_level=quality
        )
    source = {"width": 1920, "height": 1080, "quality": 85}

    result = await manager.process_message(StreamType.FRAME, source)

    assert result == {
        "low": {"width": 1280, "height": 720, "quality": 60},
        "medium": {"width": 1600, "height": 900, "quality": 75},
        "high": {"width": 1920, "height": 1080, "quality": 90},
        "auto": {"width": 1920, "height": 1080, "quality": 85},
    }
    assert source == {"width": 1920, "height": 1080, "quality": 85}


@pytest.mark.asyncio
async def test_include_fields_takes_precedence_over_exclude_fields():
    manager = SubscriptionManager()
    assert await manager.create_subscription(
        "client",
        StreamType.STATE,
        include_fields=["balls", "private"],
        exclude_fields=["private"],
    )

    assert await manager.process_message(
        StreamType.STATE, {"balls": [], "private": True, "debug": 1}
    ) == {"client": {"balls": [], "private": True}}


@pytest.mark.asyncio
async def test_preprocessor_failure_preserves_filtered_message_and_other_clients():
    manager = SubscriptionManager()
    assert await manager.create_subscription("broken", StreamType.STATE)
    assert await manager.create_subscription("healthy", StreamType.STATE)
    manager.subscriptions["broken"][StreamType.STATE].preprocessor = lambda _data: (
        _ for _ in ()
    ).throw(RuntimeError("bad extension"))
    manager.subscriptions["healthy"][StreamType.STATE].preprocessor = lambda data: {
        **data,
        "source": "healthy",
    }

    assert await manager.process_message(StreamType.STATE, {"frame": 8}) == {
        "broken": {"frame": 8},
        "healthy": {"frame": 8, "source": "healthy"},
    }


@pytest.mark.asyncio
async def test_pause_resume_info_and_stats_track_only_active_deliveries():
    manager = SubscriptionManager()
    assert await manager.create_subscription("client", StreamType.STATE)
    timestamp = datetime(2026, 1, 2, tzinfo=timezone.utc)

    await manager.pause_subscription("client", StreamType.STATE)
    assert manager.get_performance_stats()["active_subscriptions"] == 0
    assert (
        await manager.process_message(StreamType.STATE, {"frame": 1}, timestamp) == {}
    )
    await manager.resume_subscription("client", StreamType.STATE)
    assert manager.get_performance_stats()["active_subscriptions"] == 1
    delivered = await manager.process_message(StreamType.STATE, {"frame": 2}, timestamp)

    assert delivered == {"client": {"frame": 2}}
    info = await manager.get_subscription_info("client")
    subscription_info = info["subscriptions"][0]
    assert subscription_info["active"] is True
    assert subscription_info["messages_sent"] == 1
    assert subscription_info["messages_filtered"] == 0
    assert subscription_info["last_message_time"] == timestamp.isoformat()
    assert await manager.get_subscription_info("absent") == {
        "client_id": "absent",
        "subscriptions": [],
    }
    assert manager.get_performance_stats() == {
        "total_subscriptions": 1,
        "active_subscriptions": 1,
        "messages_processed": 1,
        "messages_filtered": 0,
        "aggregation_hit_rate": 0.0,
        "aggregation_dropped_messages": 0,
    }


@pytest.mark.asyncio
async def test_inactive_subscription_does_not_advance_sampling_or_stats():
    manager = SubscriptionManager()
    assert await manager.create_subscription(
        "client", StreamType.STATE, sample_rate=0.5
    )
    await manager.pause_subscription("client", StreamType.STATE)

    assert await manager.process_message(StreamType.STATE, {"frame": 1}) == {}
    subscription = manager.subscriptions["client"][StreamType.STATE]
    assert subscription.messages_filtered == 0
    assert manager.sample_counters["client"][StreamType.STATE] == 0
    assert manager.performance_stats["messages_processed"] == 0


@pytest.mark.asyncio
async def test_aggregation_retains_latest_state_and_caps_buffer_capacity():
    manager = SubscriptionManager(aggregation_buffer_capacity=2)
    assert await manager.create_subscription(
        "client",
        StreamType.STATE,
        enable_aggregation=True,
        aggregation_window_ms=60_000,
    )
    timestamp = datetime(2026, 1, 2, tzinfo=timezone.utc)

    for frame in range(4):
        await manager._add_to_aggregation_buffer(
            "client", StreamType.STATE, {"frame": frame}, timestamp
        )
    buffer = manager.aggregation_buffers["client"][StreamType.STATE]

    assert [entry["data"]["frame"] for entry in buffer] == [2, 3]
    assert await manager._aggregate_messages(buffer, StreamType.STATE) == {"frame": 3}
    assert manager.performance_stats["aggregation_dropped_messages"] == 2
    assert await manager.remove_subscription("client", StreamType.STATE)


@pytest.mark.asyncio
async def test_aggregation_uses_latest_frame_and_combines_trajectory_segments():
    manager = SubscriptionManager()
    entries = [
        {"data": {"frame": 1, "lines": ["a"], "collisions": [1]}},
        {"data": {"frame": 2, "lines": ["b"], "collisions": [2]}},
    ]

    assert await manager._aggregate_messages(entries, StreamType.FRAME) == {
        "frame": 2,
        "lines": ["b"],
        "collisions": [2],
    }
    assert await manager._aggregate_messages(entries, StreamType.TRAJECTORY) == {
        "lines": ["a", "b"],
        "collisions": [1, 2],
        "aggregated": True,
        "message_count": 2,
    }
    assert await manager._aggregate_messages([], StreamType.STATE) is None
    assert (
        await manager._aggregate_messages(entries, StreamType.ALERT)
        == entries[-1]["data"]
    )


@pytest.mark.asyncio
async def test_remove_all_cancels_each_aggregation_task_and_clears_client():
    manager = SubscriptionManager()
    for stream in (StreamType.STATE, StreamType.FRAME):
        assert await manager.create_subscription(
            "client", stream, enable_aggregation=True, aggregation_window_ms=60_000
        )
    tasks = list(manager.aggregation_tasks.values())

    await manager.remove_all_subscriptions("client")
    await manager.remove_all_subscriptions("missing")

    assert all(task.done() for task in tasks)
    assert not manager.aggregation_tasks
    assert "client" not in manager.subscriptions
    assert "client" not in manager.aggregation_buffers
    assert "client" not in manager.sample_counters
    assert all(
        "client" not in subscribers
        for subscribers in manager.stream_subscribers.values()
    )
