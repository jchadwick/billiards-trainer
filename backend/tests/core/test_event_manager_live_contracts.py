"""Behavioral contracts for the Core event manager's sync and async paths."""

from __future__ import annotations

import asyncio
import threading
from collections import deque
from typing import Any

import pytest

from backend.core.events.manager import Event, EventManager, EventPriority, EventType


@pytest.fixture
def event_manager() -> Any:
    manager = EventManager(max_history_size=8, enable_logging=False)
    try:
        yield manager
    finally:
        manager.stop_enhanced_processing()


def test_sync_dispatch_preserves_registration_snapshot_and_isolates_errors(
    event_manager: EventManager,
) -> None:
    calls: list[str] = []
    subscription_ids: dict[str, str] = {}

    def first(_event_type: str, _data: dict[str, Any]) -> None:
        calls.append("first")
        # Callbacks run outside the subscriber lock. Removing a later handler
        # is safe, and the current dispatch uses its already-copied snapshot.
        assert event_manager.unsubscribe(subscription_ids["second"])

    def second(_event_type: str, _data: dict[str, Any]) -> None:
        calls.append("second")

    def failing(_event_type: str, _data: dict[str, Any]) -> None:
        calls.append("failing")
        raise RuntimeError("one subscriber failed")

    def last(_event_type: str, _data: dict[str, Any]) -> None:
        calls.append("last")

    subscription_ids["first"] = event_manager.subscribe_to_events("live", first)
    subscription_ids["second"] = event_manager.subscribe_to_events("live", second)
    subscription_ids["failing"] = event_manager.subscribe_to_events("live", failing)
    subscription_ids["last"] = event_manager.subscribe_to_events("live", last)

    event_manager.emit_event("live", {"frame": 12})

    assert calls == ["first", "second", "failing", "last"]
    assert event_manager.get_subscriber_count("live") == 3
    assert event_manager.stats["errors"] == 1
    assert not event_manager.unsubscribe(subscription_ids["second"])

    calls.clear()
    event_manager.emit_event("live", {"frame": 13})
    assert calls == ["first", "failing", "last"]


def test_subscription_clear_is_scoped_and_reports_counts(
    event_manager: EventManager,
) -> None:
    def callback(_event_type: str, _data: dict[str, Any]) -> None:
        return None

    live_ids = [event_manager.subscribe_to_events("live", callback) for _ in range(2)]
    other_id = event_manager.subscribe_to_events("other", callback)

    assert len(set(live_ids)) == 2
    assert event_manager.get_subscriber_count("live") == 2
    assert event_manager.get_subscriber_count() == 5  # includes coordinator handlers

    event_manager.clear_subscribers("live")
    assert event_manager.get_subscriber_count("live") == 0
    assert event_manager.get_subscriber_count("other") == 1
    event_manager.emit_event("live", {})
    assert event_manager.unsubscribe(other_id)

    event_manager.clear_subscribers()
    assert event_manager.get_subscriber_count() == 0


@pytest.mark.asyncio
async def test_async_callbacks_run_on_current_loop_and_report_task_errors(
    event_manager: EventManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    received: list[tuple[str, dict[str, Any]]] = []
    completions: list[str] = []
    all_done = asyncio.Event()

    async def successful(event_type: str, data: dict[str, Any]) -> None:
        received.append((event_type, data))

    async def failing(_event_type: str, _data: dict[str, Any]) -> None:
        raise RuntimeError("async subscriber failed")

    event_manager.subscribe_to_events("async", successful)
    event_manager.subscribe_to_events("async", failing)
    original_done_handler = event_manager._handle_async_callback_result

    def observe_done(task: asyncio.Task[Any], subscription_id: str) -> None:
        original_done_handler(task, subscription_id)
        completions.append(subscription_id)
        if len(completions) == 2:
            all_done.set()

    monkeypatch.setattr(event_manager, "_handle_async_callback_result", observe_done)
    payload = {"sequence": 29}
    event_manager.emit_event("async", payload)

    await asyncio.wait_for(all_done.wait(), timeout=1)

    assert received == [("async", payload)]
    assert event_manager.stats["async_callback_success"] == 2
    assert event_manager.stats["async_callback_errors"] == 1


def test_async_callbacks_are_skipped_without_a_running_loop(
    event_manager: EventManager,
) -> None:
    async def callback(_event_type: str, _data: dict[str, Any]) -> None:
        return None

    event_manager.subscribe_to_events("sync-context", callback)
    event_manager.emit_event("sync-context", {})

    assert event_manager.stats["async_callbacks_skipped_no_loop"] == 1
    assert event_manager.stats["async_callback_success"] == 0


def test_enhanced_queue_applies_filters_targets_and_records_in_order(
    event_manager: EventManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    processed = threading.Event()
    delivered = threading.Event()
    processed_ids: list[str] = []
    callback_payloads: list[dict[str, Any]] = []

    def record_callback(_event_type: str, data: dict[str, Any]) -> None:
        callback_payloads.append(data)
        delivered.set()

    event_manager.subscribe_to_events("custom", record_callback)
    event_manager.add_event_filter(
        "keep-current", lambda event: event.data.get("keep", False)
    )

    original_handler = event_manager._handle_enhanced_event

    def observe_processing(event: Event) -> None:
        original_handler(event)
        processed_ids.append(event.id)
        if len(processed_ids) == 3:
            processed.set()

    monkeypatch.setattr(event_manager, "_handle_enhanced_event", observe_processing)

    kept_id = event_manager.emit_enhanced_event(
        EventType.CUSTOM,
        {"keep": True, "frame": 1},
        source_module="vision",
        priority="high",
    )
    filtered_id = event_manager.emit_enhanced_event(
        "custom", {"keep": False, "frame": 2}, source_module="vision"
    )
    targeted_id = event_manager.emit_targeted_event(
        "custom", {"keep": True, "frame": 3}, target_modules=["projector"]
    )

    assert processed.wait(timeout=1)
    assert delivered.wait(timeout=1)
    assert processed_ids == [kept_id, filtered_id, targeted_id]
    assert callback_payloads == [{"keep": True, "frame": 1}]

    history = event_manager.get_event_history()
    assert {event.id for event in history} == {kept_id, filtered_id, targeted_id}
    assert (
        next(event for event in history if event.id == kept_id).priority
        is EventPriority.HIGH
    )
    assert event_manager.get_event_history(event_type=EventType.CUSTOM)
    assert event_manager.get_event_history(source_module="vision")
    assert event_manager.get_event_history(time_range=(-1, float("inf")))
    assert len(event_manager.get_event_history(limit=1)) == 1
    assert event_manager.remove_event_filter("keep-current")
    assert not event_manager.remove_event_filter("keep-current")


def test_filter_failure_does_not_drop_event_and_history_is_bounded(
    event_manager: EventManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    event_manager.event_history = deque(maxlen=2)
    event_manager.add_event_filter(
        "broken", lambda _event: (_ for _ in ()).throw(ValueError("bad filter"))
    )
    processed = threading.Event()
    callback_condition = threading.Condition()
    seen: list[dict[str, Any]] = []

    def callback(_event_type: str, data: dict[str, Any]) -> None:
        with callback_condition:
            seen.append(data)
            callback_condition.notify_all()

    event_manager.subscribe_to_events("custom", callback)
    original_handler = event_manager._handle_enhanced_event
    handled_count = 0

    def observe_processing(event: Event) -> None:
        nonlocal handled_count
        original_handler(event)
        handled_count += 1
        if handled_count == 4:
            processed.set()

    monkeypatch.setattr(event_manager, "_handle_enhanced_event", observe_processing)

    event_manager.emit_enhanced_event("custom", {"id": 1})
    event_manager.emit_enhanced_event("custom", {"id": 2})
    event_manager.emit_enhanced_event("custom", {"id": 3})
    event_manager.emit_enhanced_event("custom", {"id": 4})

    assert processed.wait(timeout=1)
    with callback_condition:
        assert callback_condition.wait_for(lambda: len(seen) == 4, timeout=1)

    assert [event.data["id"] for event in event_manager.event_history] == [3, 4]
    assert {item["id"] for item in seen} == {1, 2, 3, 4}


def test_event_serialization_round_trips_known_and_custom_types() -> None:
    known = Event(
        id="known",
        type=EventType.STATE_UPDATED,
        data={"frame": 42},
        timestamp=123.5,
        source_module="core",
        priority=EventPriority.CRITICAL,
        target_modules=["api"],
        correlation_id="request-1",
        metadata={"origin": "test"},
    )
    restored_known = Event.from_dict(known.to_dict())
    assert restored_known == known

    custom = Event(
        id="custom",
        type="vendor_event",
        data={},
        timestamp=456.0,
        source_module="extension",
    )
    restored_custom = Event.from_dict(custom.to_dict())
    assert restored_custom.type == "vendor_event"
    assert restored_custom.priority is EventPriority.NORMAL


def test_processing_start_is_idempotent_and_stop_releases_worker(
    event_manager: EventManager,
) -> None:
    original_thread = event_manager.processing_thread
    assert original_thread is not None
    assert original_thread.is_alive()

    event_manager.start_enhanced_processing()
    assert event_manager.processing_thread is original_thread

    event_manager.stop_enhanced_processing()
    event_manager.stop_enhanced_processing()

    assert not event_manager.is_running
    assert not original_thread.is_alive()
