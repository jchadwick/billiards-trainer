"""System-level checks for subscription teardown and background task cleanup."""

import pytest

from backend.api.websocket.manager import StreamType
from backend.api.websocket.subscriptions import SubscriptionManager


@pytest.mark.system
@pytest.mark.asyncio
async def test_unsubscribe_cancels_aggregation_and_cleans_client_state():
    """Removing an aggregated stream stops its task and drops buffered state."""
    manager = SubscriptionManager()
    assert await manager.create_subscription(
        "client",
        StreamType.STATE,
        enable_aggregation=True,
        aggregation_window_ms=10_000,
    )
    task_key = "client:state"
    task = manager.aggregation_tasks[task_key]
    manager.aggregation_buffers["client"][StreamType.STATE].append(
        {"data": {"score": 1}}
    )
    manager.sample_counters["client"][StreamType.STATE] = 4

    assert await manager.remove_subscription("client", StreamType.STATE)

    assert task.done()
    assert task_key not in manager.aggregation_tasks
    assert "client" not in manager.subscriptions
    assert "client" not in manager.aggregation_buffers
    assert "client" not in manager.sample_counters
    assert "client" not in manager.stream_subscribers[StreamType.STATE]


@pytest.mark.system
@pytest.mark.asyncio
async def test_aggregation_buffer_keeps_only_the_newest_configured_samples():
    """Aggregation storage stays bounded while retaining current observations."""
    from datetime import datetime, timezone

    manager = SubscriptionManager(aggregation_buffer_capacity=2)
    now = datetime.now(timezone.utc)

    for value in range(4):
        await manager._add_to_aggregation_buffer(
            "client", StreamType.STATE, {"value": value}, now
        )

    buffered = manager.aggregation_buffers["client"][StreamType.STATE]
    assert len(buffered) == 2
    assert [entry["data"]["value"] for entry in buffered] == [2, 3]
    assert manager.performance_stats["aggregation_dropped_messages"] == 2
