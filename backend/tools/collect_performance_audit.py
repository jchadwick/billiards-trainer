#!/usr/bin/env python3
"""Run the off-device state and Rust live-camera qualification tools.

This script is a measurement harness, not an application server. The
``--rust-live-qualification`` mode starts the Rust live process with explicit
camera settings. The generic WebSocket audit mode is retained for compatibility
testing and requires its corresponding audit endpoints; it does not instrument
the supported Rust gateway. Repeated runs reconnect to avoid carrying
connection state from one sample window into the next.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import ipaddress
import json
import math
import os
import platform
import shutil
import socket
import subprocess
import sys
import threading
import time
import uuid
from collections import Counter, OrderedDict, deque
from dataclasses import fields, is_dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlsplit, urlunsplit
from urllib.request import urlopen

import websockets

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.config import Config, config
from backend.performance_audit import AsyncAuditRecorder


def percentile(values: list[float], rank: float) -> float | None:
    """Return a nearest-rank percentile in milliseconds."""
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(rank * len(ordered)) - 1))
    return ordered[index]


def observation_from_message(message: dict[str, Any]) -> dict[str, Any]:
    """Extract state identity from Python and native Rust state envelopes."""
    data = message.get("data")
    if not isinstance(data, dict):
        data = {}
    status = data.get("observation_status")
    if not isinstance(status, dict):
        status = {}
    identity = status.get("observation_identity")
    if not isinstance(identity, dict):
        identity = {}

    engine_instance_id = identity.get("engine_instance_id") or status.get(
        "engine_instance_id"
    )
    source_session_id = identity.get("session_id") or status.get("source_session_id")
    source_frame_id = identity.get("frame_id") or status.get("source_frame_id")
    source_frame_number = status.get("source_frame_number")
    if source_frame_number is None:
        source_frame_number = _numeric_frame_sequence(source_frame_id)
    source_timestamp = status.get("source_timestamp")
    source_timestamp_kind = status.get("source_timestamp_kind")
    source_timestamp_stage = None
    if isinstance(source_timestamp_kind, str) and ":" in source_timestamp_kind:
        source_timestamp_stage = source_timestamp_kind.rsplit(":", 1)[-1]
    elif isinstance(status.get("source_timestamp"), dict):
        source_timestamp = status["source_timestamp"].get("ticks")
        source_timestamp_stage = status["source_timestamp"].get("stage")
    has_scene_identity = all(
        isinstance(value, str) and value
        for value in (engine_instance_id, source_session_id, source_frame_id)
    )
    source_observation_key = (
        (str(engine_instance_id), str(source_session_id), str(source_frame_id))
        if has_scene_identity
        else None
    )
    stream_sequence = data.get("sequence", message.get("sequence"))
    state_complete = (
        isinstance(stream_sequence, int)
        and not isinstance(stream_sequence, bool)
        and isinstance(data.get("balls"), list)
        and isinstance(data.get("observation_status"), dict)
        and (has_scene_identity or isinstance(source_frame_number, int))
    )
    component_status = status.get("component_observation_status")
    component_status = (
        dict(component_status) if isinstance(component_status, dict) else {}
    )
    # The Python live pipeline currently publishes these established top-level
    # fields alongside an empty component_observation_status object. Merge the
    # fields when individual component entries are absent so an empty envelope
    # does not silently turn observed frames into unknown freshness.
    legacy_component_statuses = {
        "balls": status.get("ball_positions_status")
        or status.get("ball_detection_status"),
        "cue": status.get("cue_detection_status"),
        "motion": status.get("motion_monitor_status"),
    }
    for component, component_state in legacy_component_statuses.items():
        if component not in component_status and isinstance(component_state, str):
            component_status[component] = component_state
    if not component_status:
        component_status = None
    return {
        "stream_sequence": stream_sequence,
        "source_frame_number": source_frame_number,
        "source_frame_id": source_frame_id,
        "source_session_id": source_session_id,
        "source_observation_key": source_observation_key,
        "state_complete": state_complete,
        "source_timestamp": source_timestamp,
        "source_timestamp_stage": source_timestamp_stage,
        "source_timestamp_kind": source_timestamp_kind,
        "source_process_id": status.get("source_process_id"),
        "source_publication_monotonic_ns": status.get(
            "source_publication_monotonic_ns"
        ),
        "vision_instance_id": status.get("vision_instance_id") or engine_instance_id,
        "vision_process_id": status.get("vision_process_id"),
        "map_revision": status.get("map_revision"),
        "calibration_revision": status.get("calibration_revision"),
        "ball_count": data.get("ball_count"),
        "cue_present": data.get("cue") is not None,
        "component_status": component_status,
    }


def observation_age_ms(
    source_timestamp: Any, received_wall_time: float
) -> float | None:
    """Compute local publication age only for a finite epoch timestamp."""
    if isinstance(source_timestamp, bool) or not isinstance(
        source_timestamp, (int, float)
    ):
        return None
    source_timestamp_seconds = float(source_timestamp)
    if not math.isfinite(source_timestamp_seconds) or source_timestamp_seconds <= 0:
        return None
    return max(0.0, received_wall_time - source_timestamp_seconds) * 1000


def trajectory_from_message(message: dict[str, Any]) -> dict[str, Any]:
    """Extract trajectory source identity and lifecycle for delivery analysis."""
    data = message.get("data")
    if not isinstance(data, dict):
        data = {}
    source_observations = data.get("source_observations")
    if not isinstance(source_observations, dict):
        source_observations = {}
    return {
        "stream_sequence": data.get("sequence", message.get("sequence")),
        "source_frame_number": data.get("source_frame_number"),
        "source_timestamp": data.get("source_timestamp"),
        "source_timestamp_kind": data.get("source_timestamp_kind"),
        "source_process_id": source_observations.get("source_process_id"),
        "source_publication_monotonic_ns": source_observations.get(
            "source_publication_monotonic_ns"
        ),
        "vision_instance_id": source_observations.get("vision_instance_id"),
        "map_revision": data.get("map_revision"),
        "calibration_revision": data.get("calibration_revision"),
        "prediction_status": data.get("prediction_status"),
        "spatial_valid": data.get("spatial_valid"),
        "line_count": data.get("line_count"),
        "collision_count": data.get("collision_count"),
    }


def collect_run_summary(
    events: list[dict[str, Any]], duration: float
) -> dict[str, Any]:
    """Summarize received states with bounded percentile sample storage."""
    accumulator = StateAccumulator()
    for event in events:
        accumulator.add(event)
    return accumulator.summary(duration)


class _BoundedIdentityWindow:
    """Retain recent source high-water marks without growing across restarts.

    The collector cannot preserve duplicate and ordering checks for identities
    that have aged out of this LRU window. The summary reports both the
    capacity and eviction count so that limitation remains visible in long
    audits rather than silently consuming unbounded memory.
    """

    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError("identity window capacity must be positive")
        self.capacity = capacity
        self._values: OrderedDict[str, Any] = OrderedDict()
        self.evictions = 0

    def get(self, identity: str) -> Any | None:
        """Return and refresh one retained identity's recency."""
        value = self._values.get(identity)
        if value is not None:
            self._values.move_to_end(identity)
        return value

    def set(self, identity: str, value: Any) -> None:
        """Insert a high-water mark, evicting the least-recent identity."""
        if identity in self._values:
            self._values[identity] = value
            self._values.move_to_end(identity)
            return
        if len(self._values) >= self.capacity:
            self._values.popitem(last=False)
            self.evictions += 1
        self._values[identity] = value

    def summary(self) -> dict[str, int | str]:
        """Expose the bounded identity window and lost deduplication history."""
        return {
            "policy": "least-recently-seen source identities",
            "capacity": self.capacity,
            "retained": len(self._values),
            "evictions": self.evictions,
        }


class StateAccumulator:
    """Online counters with bounded latency samples for long audit runs."""

    def __init__(self, maximum_samples: int = 10000) -> None:
        self.message_count = 0
        self.unique_observations = 0
        self.unique_source_frames = 0
        self.cached_or_failed_source_updates = 0
        self.unknown_freshness_source_updates = 0
        self.repeated_observations = 0
        self.uncorrelated_messages = 0
        self.source_sequence_gaps = 0
        self.source_sequence_restarts = 0
        self.source_out_of_order = 0
        self.stream_sequence_gaps = 0
        self.stream_sequence_duplicates = 0
        self.stream_sequence_out_of_order = 0
        self.complete_messages = 0
        self.incomplete_messages = 0
        # Keep source publication time with the high-water frame. A camera may
        # restart inside the same capture process and reset its frame counter;
        # a newer shared-memory publication timestamp distinguishes that new
        # sequence epoch from a delayed observation from the previous epoch.
        self._last_source = _BoundedIdentityWindow(maximum_samples)
        self._last_stream_sequence: int | None = None
        self._ages: deque[float] = deque(maxlen=maximum_samples)
        self._source_to_receipt_ms: deque[float] = deque(maxlen=maximum_samples)
        self._latency_samples: deque[dict[str, Any]] = deque(maxlen=maximum_samples)
        self._age_sample_count = 0
        self._source_to_receipt_sample_count = 0
        self._latency_sample_count = 0
        self._component_status_counts: dict[str, dict[str, int]] = {}

    def add(self, event: dict[str, Any]) -> None:
        self.message_count += 1
        source = event.get("source_frame_number")
        instance = "|".join(
            str(event.get(field) or "unknown")
            for field in (
                "vision_instance_id",
                "source_process_id",
                "source_session_id",
            )
        )
        if isinstance(source, int) and not isinstance(source, bool):
            previous = self._last_source.get(instance)
            source_timestamp = event.get("source_timestamp")
            if (
                isinstance(source_timestamp, bool)
                or not isinstance(source_timestamp, (int, float))
                or not math.isfinite(source_timestamp)
            ):
                source_timestamp = None
            else:
                source_timestamp = float(source_timestamp)
            has_publication_timestamp = (
                event.get("source_timestamp_kind") == "shared_memory_publication"
                and source_timestamp is not None
            )

            if previous is None:
                self._record_unique_source_frame(event)
                self._last_source.set(instance, (source, source_timestamp))
            elif (
                has_publication_timestamp
                and previous[1] is not None
                and source_timestamp is not None
                and source_timestamp < previous[1]
            ):
                # A message from an older source-time epoch arrived after a
                # newer one. Do not let it poison the active frame high-water.
                self.source_out_of_order += 1
            elif source > previous[0]:
                self._record_unique_source_frame(event)
                if source > previous[0] + 1:
                    self.source_sequence_gaps += source - previous[0] - 1
                self._last_source.set(instance, (source, source_timestamp))
            elif (
                has_publication_timestamp
                and previous[1] is not None
                and source_timestamp is not None
                and source_timestamp > previous[1]
            ):
                # A newer publication paired with a non-increasing frame
                # number is a source restart, including a reset to the same
                # first frame number. Start a new observation sequence epoch.
                self._record_unique_source_frame(event)
                self.source_sequence_restarts += 1
                self._last_source.set(instance, (source, source_timestamp))
            else:
                self.repeated_observations += int(source == previous[0])
                self.source_out_of_order += int(source < previous[0])
        else:
            self.uncorrelated_messages += 1

        if isinstance(event.get("state_complete"), bool):
            state_complete = event["state_complete"]
        else:
            state_complete = (
                isinstance(event.get("stream_sequence"), int)
                and not isinstance(event.get("stream_sequence"), bool)
                and isinstance(source, int)
                and not isinstance(source, bool)
                and isinstance(event.get("component_status"), dict)
            )
        if state_complete:
            self.complete_messages += 1
        else:
            self.incomplete_messages += 1

        sequence = event.get("stream_sequence")
        if isinstance(sequence, int) and not isinstance(sequence, bool):
            if self._last_stream_sequence is not None:
                if sequence > self._last_stream_sequence + 1:
                    self.stream_sequence_gaps += (
                        sequence - self._last_stream_sequence - 1
                    )
                elif sequence == self._last_stream_sequence:
                    self.stream_sequence_duplicates += 1
                elif sequence < self._last_stream_sequence:
                    self.stream_sequence_out_of_order += 1
            if (
                self._last_stream_sequence is None
                or sequence > self._last_stream_sequence
            ):
                self._last_stream_sequence = sequence
        age = event.get("source_publication_wall_age_estimate_ms")
        if age is None:
            age = event.get("source_publication_age_ms")
        if isinstance(age, (int, float)) and math.isfinite(age):
            self._ages.append(float(age))
            self._age_sample_count += 1
        receive_ns = event.get("receive_monotonic_ns")
        if isinstance(receive_ns, int) and not isinstance(receive_ns, bool):
            publication_ns = event.get("source_publication_monotonic_ns")
            if (
                event.get("clock_comparable") is True
                and isinstance(publication_ns, int)
                and not isinstance(publication_ns, bool)
                and publication_ns > 0
                and receive_ns >= publication_ns
            ):
                self._source_to_receipt_ms.append((receive_ns - publication_ns) / 1e6)
                self._source_to_receipt_sample_count += 1
            self._latency_samples.append(
                {
                    "stream_sequence": sequence,
                    "receive_monotonic_ns": receive_ns,
                    "source_frame_number": source,
                    "source_process_id": event.get("source_process_id"),
                }
            )
            self._latency_sample_count += 1
        component_status = event.get("component_status")
        if isinstance(component_status, dict):
            for component, status in component_status.items():
                counts = self._component_status_counts.setdefault(str(component), {})
                name = str(status)
                counts[name] = counts.get(name, 0) + 1

    def _record_unique_source_frame(self, event: dict[str, Any]) -> None:
        """Count one new source frame and only call observed balls fresh."""
        self.unique_source_frames += 1
        component_status = event.get("component_status")
        ball_status = (
            component_status.get("balls")
            if isinstance(component_status, dict)
            else None
        )
        if ball_status == "observed":
            self.unique_observations += 1
        elif ball_status in {"cached", "failed", "skipped", "unavailable"}:
            self.cached_or_failed_source_updates += 1
        else:
            self.unknown_freshness_source_updates += 1

    def summary(self, duration: float) -> dict[str, Any]:
        return {
            "duration_seconds": duration,
            "state_messages_received": self.message_count,
            "state_messages_per_second": (
                self.message_count / duration if duration else 0
            ),
            "unique_source_observations": self.unique_observations,
            "unique_observations_per_second": (
                self.unique_observations / duration if duration else 0
            ),
            "unique_source_frame_updates": self.unique_source_frames,
            "unique_source_frame_updates_per_second": (
                self.unique_source_frames / duration if duration else 0
            ),
            "cached_or_failed_source_updates": self.cached_or_failed_source_updates,
            "unknown_freshness_source_updates": self.unknown_freshness_source_updates,
            "repeated_source_observations": self.repeated_observations,
            "uncorrelated_state_messages": self.uncorrelated_messages,
            "complete_state_messages": self.complete_messages,
            "incomplete_state_messages": self.incomplete_messages,
            "source_sequence_gaps": self.source_sequence_gaps,
            "source_sequence_restarts": self.source_sequence_restarts,
            "source_out_of_order_messages": self.source_out_of_order,
            "source_identity_deduplication_window": self._last_source.summary(),
            "stream_sequence_gaps": self.stream_sequence_gaps,
            "stream_sequence_duplicates": self.stream_sequence_duplicates,
            "stream_sequence_out_of_order_messages": self.stream_sequence_out_of_order,
            "source_publication_wall_age_estimate_ms": {
                "count": self._age_sample_count,
                "retained_count": len(self._ages),
                "sample_limit": self._ages.maxlen,
                "truncated": self._age_sample_count > len(self._ages),
                "p50": percentile(list(self._ages), 0.50),
                "p95": percentile(list(self._ages), 0.95),
                "p99": percentile(list(self._ages), 0.99),
            },
            "source_publication_to_client_receipt_ms": {
                "count": self._source_to_receipt_sample_count,
                "retained_count": len(self._source_to_receipt_ms),
                "sample_limit": self._source_to_receipt_ms.maxlen,
                "truncated": self._source_to_receipt_sample_count
                > len(self._source_to_receipt_ms),
                "p50": percentile(list(self._source_to_receipt_ms), 0.50),
                "p95": percentile(list(self._source_to_receipt_ms), 0.95),
                "p99": percentile(list(self._source_to_receipt_ms), 0.99),
            },
            "client_latency_sample_count": self._latency_sample_count,
            "retained_client_latency_sample_count": len(self._latency_samples),
            "client_latency_samples_truncated": (
                self._latency_sample_count > len(self._latency_samples)
            ),
            "component_observation_status_counts": self._component_status_counts,
            "accuracy_evidence": "unavailable for unannotated live streams",
        }


def _numeric_frame_sequence(frame_id: Any) -> int | None:
    """Extract an ordered frame counter from the Rust frame-N identity form."""
    if not isinstance(frame_id, str):
        return None
    suffix = frame_id.rsplit("-", 1)[-1]
    return int(suffix) if suffix.isdecimal() else None


class RustLiveStateAccumulator:
    """Bounded summary of state snapshots emitted by the local Rust publisher."""

    def __init__(self, maximum_samples: int) -> None:
        if maximum_samples < 1:
            raise ValueError("maximum_samples must be positive")
        self.message_count = 0
        self.unique_observations = 0
        self.repeated_observations = 0
        self.uncorrelated_observations = 0
        self.source_sequence_gaps = 0
        self.source_out_of_order = 0
        self.complete_messages = 0
        self.exposure_timestamp_claims = 0
        self.observed_ball_count = 0
        self.identified_ball_count = 0
        self.table_position_count = 0
        self._last_source: dict[tuple[str, str], int] = {}
        self._identities: set[tuple[str, str, str]] = set()
        self._identity_order: deque[tuple[str, str, str]] = deque(
            maxlen=maximum_samples
        )
        self._source_order: deque[tuple[str, str]] = deque(maxlen=maximum_samples)
        self._receipt_intervals_ms: deque[float] = deque(maxlen=maximum_samples)
        self._last_receipt_ns: int | None = None
        self._timestamp_stages: Counter[str] = Counter()
        self._trace: deque[dict[str, Any]] = deque(maxlen=maximum_samples)

    @property
    def retained_trace(self) -> list[dict[str, Any]]:
        return list(self._trace)

    def add(self, event: dict[str, Any]) -> None:
        """Update streaming counts and keep only a bounded state-only trace."""
        self.message_count += 1
        engine_id = event.get("engine_instance_id")
        session_id = event.get("source_session_id")
        frame_id = event.get("source_frame_id")
        receipt_ns = event.get("state_receipt_monotonic_ns")
        if isinstance(receipt_ns, int) and not isinstance(receipt_ns, bool):
            if (
                self._last_receipt_ns is not None
                and receipt_ns >= self._last_receipt_ns
            ):
                self._receipt_intervals_ms.append(
                    (receipt_ns - self._last_receipt_ns) / 1_000_000
                )
            self._last_receipt_ns = receipt_ns

        if (
            isinstance(engine_id, str)
            and engine_id
            and isinstance(session_id, str)
            and session_id
            and isinstance(frame_id, str)
            and frame_id
        ):
            identity = (engine_id, session_id, frame_id)
            source_identity = (engine_id, session_id)
            if identity in self._identities:
                self.repeated_observations += 1
            else:
                if len(self._identity_order) == self._identity_order.maxlen:
                    expired_identity = self._identity_order[0]
                    self._identities.discard(expired_identity)
                self._identities.add(identity)
                self._identity_order.append(identity)
                self.unique_observations += 1
                sequence = _numeric_frame_sequence(frame_id)
                previous = self._last_source.get(source_identity)
                if (
                    previous is None
                    and len(self._source_order) == self._source_order.maxlen
                ):
                    expired_source = self._source_order[0]
                    self._last_source.pop(expired_source, None)
                if sequence is not None:
                    if previous is not None:
                        if sequence > previous + 1:
                            self.source_sequence_gaps += sequence - previous - 1
                        elif sequence < previous:
                            self.source_out_of_order += 1
                    if previous is None or sequence > previous:
                        if source_identity not in self._last_source:
                            self._source_order.append(source_identity)
                        self._last_source[source_identity] = sequence
        else:
            self.uncorrelated_observations += 1

        if event.get("state_complete") is True:
            self.complete_messages += 1
        if event.get("source_timestamp_is_exposure") is True:
            self.exposure_timestamp_claims += 1
        timestamp_stage = event.get("source_timestamp_stage")
        if isinstance(timestamp_stage, str) and timestamp_stage:
            self._timestamp_stages[timestamp_stage] += 1
        balls = event.get("balls")
        if isinstance(balls, list):
            for ball in balls:
                if not isinstance(ball, dict):
                    continue
                if ball.get("is_observed") is True:
                    self.observed_ball_count += 1
                if ball.get("identity_status") == "identified":
                    self.identified_ball_count += 1
                if isinstance(ball.get("table_position"), dict):
                    self.table_position_count += 1
        self._trace.append(event)

    def summary(self, duration_seconds: float) -> dict[str, Any]:
        intervals = list(self._receipt_intervals_ms)
        return {
            "duration_seconds": duration_seconds,
            "state_messages_received": self.message_count,
            "state_messages_per_second": (
                self.message_count / duration_seconds if duration_seconds > 0 else 0
            ),
            "unique_source_observations": self.unique_observations,
            "unique_observations_per_second": (
                self.unique_observations / duration_seconds
                if duration_seconds > 0
                else 0
            ),
            "repeated_source_observations": self.repeated_observations,
            "deduplication_window_observations": self._identity_order.maxlen,
            "repeats_are_exact_within_window": True,
            "repeats_older_than_window_may_be_counted_as_new": True,
            "uncorrelated_state_messages": self.uncorrelated_observations,
            "source_sequence_gaps": self.source_sequence_gaps,
            "source_out_of_order_messages": self.source_out_of_order,
            "complete_state_messages": self.complete_messages,
            "observed_ball_measurements": self.observed_ball_count,
            "identified_ball_measurements": self.identified_ball_count,
            "table_coordinate_measurements": self.table_position_count,
            "state_receipt_interval_ms": {
                "count": len(intervals),
                "sample_limit": self._receipt_intervals_ms.maxlen,
                "p50": percentile(intervals, 0.50),
                "p95": percentile(intervals, 0.95),
                "p99": percentile(intervals, 0.99),
            },
            "source_timestamp_stages": dict(self._timestamp_stages),
            "source_exposure_timestamp_claims": self.exposure_timestamp_claims,
            "accuracy_evidence": "unavailable without temporal ground-truth labels",
            "retained_state_trace_count": len(self._trace),
            "retained_state_trace_limit": self._trace.maxlen,
            "state_trace_truncated": self.message_count > len(self._trace),
        }


class RustLiveTimingSummaryBridge:
    """Forward bounded timing JSONL to the Rust report accumulator.

    The callback path only enqueues a Python object with ``put_nowait``. JSON
    serialization and pipe writes run on a separate task; a full queue or a
    failed child is recorded as audit loss and can never stall state handling.
    Collector ``perf_counter_ns`` values cross the bridge unchanged because
    the Rust process must not substitute a different clock domain.
    """

    _EXECUTABLE_NAME = "billiards-performance-audit-timing-summary"
    _MAXIMUM_LINE_BYTES = 16 * 1024 * 1024
    _FINISH_SENTINEL = object()

    def __init__(
        self,
        *,
        maximum_samples: int,
        queue_capacity: int,
        maximum_line_bytes: int,
        write_timeout_seconds: float,
        executable: str | Path | None = None,
    ) -> None:
        if maximum_samples < 1 or queue_capacity < 1:
            raise ValueError("timing bridge sample and queue limits must be positive")
        if not 1 <= maximum_line_bytes <= self._MAXIMUM_LINE_BYTES:
            raise ValueError("timing bridge maximum line size is outside Rust limits")
        if not math.isfinite(write_timeout_seconds) or write_timeout_seconds <= 0:
            raise ValueError("timing bridge write timeout must be positive")
        self.maximum_samples = maximum_samples
        self.queue_capacity = queue_capacity
        self.maximum_line_bytes = maximum_line_bytes
        self.write_timeout_seconds = write_timeout_seconds
        self.configured_executable = executable
        self.executable_path: Path | None = None
        self._queue: asyncio.Queue[dict[str, Any] | object] = asyncio.Queue(
            maxsize=queue_capacity
        )
        self._process: asyncio.subprocess.Process | None = None
        self._writer_task: asyncio.Task[None] | None = None
        self._result: dict[str, Any] | None = None
        self._failure: str | None = None
        self._dropped_records = 0
        self._callback_rejected_records = 0
        self._upstream_dropped_records = 0
        self._closed = False
        self._finished = False
        self._return_code: int | None = None
        self._stderr_tail = ""

    @classmethod
    def resolve_executable(cls, configured: str | Path | None = None) -> Path:
        """Resolve an override, PATH installation, or existing Cargo artifact."""
        executable_name = cls._EXECUTABLE_NAME
        if os.name == "nt":
            executable_name += ".exe"
        if configured is not None:
            path = Path(configured).expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(
                    f"timing summary executable does not exist: {path}"
                )
            if os.name != "nt" and not os.access(path, os.X_OK):
                raise PermissionError(
                    f"timing summary executable is not executable: {path}"
                )
            return path

        configured_target_dir = os.environ.get("CARGO_TARGET_DIR")
        target_directories = [
            Path(configured_target_dir).expanduser() if configured_target_dir else None,
            ROOT / "perception" / "target",
            ROOT / "target",
        ]
        candidates: list[Path] = []
        for target_directory in target_directories:
            if target_directory is None:
                continue
            if not target_directory.is_absolute():
                target_directory = Path.cwd() / target_directory
            candidates.extend(
                target_directory / profile / executable_name
                for profile in ("release", "debug")
            )
        for candidate in candidates:
            if candidate.is_file() and (
                os.name == "nt" or os.access(candidate, os.X_OK)
            ):
                return candidate.resolve()
        on_path = shutil.which(executable_name)
        if on_path:
            return Path(on_path).resolve()
        searched = ", ".join(str(candidate) for candidate in candidates)
        raise FileNotFoundError(
            f"could not find {executable_name} on PATH or in Cargo target directories; "
            f"build it with `cargo build --manifest-path {ROOT / 'perception' / 'Cargo.toml'} "
            f"--bin {cls._EXECUTABLE_NAME}` (searched: {searched})"
        )

    async def start(self) -> None:
        """Start the native summarizer without making failure fatal to capture."""
        if self._process is not None or self._closed:
            return
        try:
            self.executable_path = self.resolve_executable(self.configured_executable)
            self._process = await asyncio.create_subprocess_exec(
                str(self.executable_path),
                "--maximum-samples",
                str(self.maximum_samples),
                "--maximum-line-bytes",
                str(self.maximum_line_bytes),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except (OSError, ValueError) as error:
            self._record_failure(f"{type(error).__name__}: {error}")
            return
        self._writer_task = asyncio.create_task(
            self._write_messages(), name="rust-live-timing-summary-writer"
        )

    def try_record(self, record: dict[str, Any]) -> bool:
        """Nonblocking ingress callback; return false when audit data is lost."""
        if (
            self._process is None
            or self._failure is not None
            or self._closed
            or (self._writer_task is not None and self._writer_task.done())
        ):
            self._dropped_records += 1
            self._callback_rejected_records += 1
            return False
        try:
            self._queue.put_nowait({"kind": "record", "record": record})
        except asyncio.QueueFull:
            self._dropped_records += 1
            self._callback_rejected_records += 1
            return False
        return True

    async def begin_measurement(self, monotonic_ns: int) -> None:
        await self._enqueue_control(
            {"kind": "begin_measurement", "monotonic_ns": monotonic_ns}
        )

    async def end_measurement(self, monotonic_ns: int) -> None:
        await self._enqueue_control(
            {"kind": "end_measurement", "monotonic_ns": monotonic_ns}
        )

    async def _enqueue_control(self, envelope: dict[str, Any]) -> None:
        if (
            self._process is None
            or self._failure is not None
            or self._closed
            or (self._writer_task is not None and self._writer_task.done())
        ):
            return
        try:
            await asyncio.wait_for(
                self._queue.put(envelope), timeout=self.write_timeout_seconds
            )
        except asyncio.TimeoutError:
            self._record_failure("timed out enqueueing timing summary control")
            return
        if self._failure is not None or (
            self._writer_task is not None and self._writer_task.done()
        ):
            self._record_failure(
                "timing summary writer stopped while enqueueing control"
            )
            self._discard_queued_records()

    async def _write_messages(self) -> None:
        process = self._process
        if process is None or process.stdin is None:
            return
        envelope: dict[str, Any] | object | None = None
        try:
            while True:
                envelope = await self._queue.get()
                if envelope is self._FINISH_SENTINEL:
                    self._queue.task_done()
                    break
                try:
                    encoded = (
                        json.dumps(envelope, separators=(",", ":"), allow_nan=False)
                        + "\n"
                    ).encode("utf-8")
                    if len(encoded) > self.maximum_line_bytes:
                        raise ValueError(
                            "timing bridge JSONL record exceeds configured maximum "
                            f"of {self.maximum_line_bytes} bytes"
                        )
                    process.stdin.write(encoded)
                    await asyncio.wait_for(
                        process.stdin.drain(), timeout=self.write_timeout_seconds
                    )
                    envelope = None
                finally:
                    self._queue.task_done()
        except asyncio.CancelledError:
            if isinstance(envelope, dict) and envelope.get("kind") == "record":
                self._dropped_records += 1
            self._record_failure("Rust timing summary writer was cancelled")
            self._discard_queued_records()
            raise
        except Exception as error:
            if isinstance(envelope, dict) and envelope.get("kind") == "record":
                self._dropped_records += 1
            self._record_failure(f"{type(error).__name__}: {error}")
            self._discard_queued_records()
        finally:
            try:
                process.stdin.close()
            except (BrokenPipeError, ConnectionError, OSError):
                pass

    def _discard_queued_records(self) -> None:
        while True:
            try:
                queued = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            if isinstance(queued, dict) and queued.get("kind") == "record":
                self._dropped_records += 1
            self._queue.task_done()

    def _record_failure(self, message: str) -> None:
        if self._failure is None:
            self._failure = message

    async def finish(
        self,
        state_trace: list[dict[str, Any]],
        *,
        state_trace_truncated: bool,
        upstream_dropped_records: int = 0,
    ) -> dict[str, Any]:
        """Send retained identity-only state entries and return Rust summaries."""
        if self._finished:
            return self.summary()
        self._finished = True
        # The ingress loss count includes callback refusals. Use the larger of
        # the caller's total and our local refusals to avoid counting those
        # same records twice while still making a directly used bridge honest.
        upstream_drops = max(
            max(0, upstream_dropped_records), self._callback_rejected_records
        )
        self._upstream_dropped_records = upstream_drops
        if self._process is None or self._writer_task is None:
            return self.summary()
        for event in state_trace:
            identity = self._state_identity(event)
            if identity is not None and self._failure is None:
                await self._enqueue_control({"kind": "state", "state": identity})
        # Synchronous callback refusals are already included in ingress's
        # consumer loss count. Count later writer-side failures separately to
        # avoid double-counting the same callback rejection.
        writer_drops = max(0, self._dropped_records - self._callback_rejected_records)
        total_drops = upstream_drops + writer_drops
        if self._failure is None:
            await self._enqueue_control(
                {
                    "kind": "finish",
                    "state_trace_truncated": bool(state_trace_truncated),
                    "bridge_dropped_records": total_drops,
                }
            )
        if self._writer_task is not None and not self._writer_task.done():
            try:
                await asyncio.wait_for(
                    self._queue.put(self._FINISH_SENTINEL),
                    timeout=self.write_timeout_seconds,
                )
            except asyncio.TimeoutError:
                self._record_failure("timed out stopping Rust timing summary writer")
                await self._stop_process()
        try:
            await asyncio.wait_for(
                self._writer_task,
                timeout=max(
                    self.write_timeout_seconds,
                    float(
                        config.get(
                            "performance_audit.export_drain_timeout_seconds", 5.0
                        )
                    ),
                ),
            )
            await self._collect_process_result()
        except (asyncio.TimeoutError, OSError, RuntimeError) as error:
            self._record_failure(f"{type(error).__name__}: {error}")
            await self._stop_process()
        if self._process is not None and self._return_code not in (None, 0):
            self._record_failure(
                f"Rust timing summarizer exited with status {self._return_code}: "
                f"{self._stderr_tail.strip()}"
            )
        if self._failure is None and self._result is None:
            self._record_failure("Rust timing summarizer returned no JSON result")
        if self._failure is not None:
            return self.summary()
        return self._result or self.summary()

    @staticmethod
    def _state_identity(event: dict[str, Any]) -> dict[str, Any] | None:
        engine = event.get("engine_instance_id")
        session = event.get("source_session_id")
        frame = event.get("source_frame_id")
        sequence = event.get("source_transport_sequence")
        if (
            not isinstance(engine, str)
            or not engine
            or not isinstance(session, str)
            or not session
            or not isinstance(frame, str)
            or not frame
            or not isinstance(sequence, int)
            or isinstance(sequence, bool)
            or sequence < 0
        ):
            return None
        return {
            "engine_instance_id": engine,
            "source_session_id": session,
            "source_frame_id": frame,
            "source_transport_sequence": sequence,
        }

    async def _collect_process_result(self) -> None:
        process = self._process
        if process is None:
            return
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=self.write_timeout_seconds
            )
        except asyncio.TimeoutError:
            await self._stop_process()
            raise
        self._return_code = process.returncode
        self._stderr_tail = stderr.decode("utf-8", errors="replace")[-4096:]
        if stdout:
            try:
                result = json.loads(stdout)
                if not isinstance(result, dict) or not isinstance(
                    result.get("summary"), dict
                ):
                    raise ValueError("timing summary output has an invalid shape")
                self._result = result
            except (json.JSONDecodeError, ValueError) as error:
                self._record_failure(f"invalid Rust timing summary output: {error}")
        elif self._return_code == 0:
            self._record_failure("Rust timing summarizer produced empty output")

    async def _stop_process(self) -> None:
        process = self._process
        if process is None or process.returncode is not None:
            return
        try:
            process.terminate()
        except ProcessLookupError:
            pass
        try:
            await asyncio.wait_for(process.wait(), timeout=self.write_timeout_seconds)
        except asyncio.TimeoutError:
            try:
                process.kill()
            except ProcessLookupError:
                pass
            await process.wait()
        self._return_code = process.returncode

    def summary(self) -> dict[str, Any]:
        """Return the native result or an explicit partial diagnostic result."""
        if self._result is not None:
            result = dict(self._result)
            writer_drops = max(
                0, self._dropped_records - self._callback_rejected_records
            )
            bridge_drops = self._upstream_dropped_records + writer_drops
            reported_bridge_drops = result.get("bridge_dropped_records", 0)
            if isinstance(reported_bridge_drops, int) and not isinstance(
                reported_bridge_drops, bool
            ):
                bridge_drops = max(bridge_drops, reported_bridge_drops)
            result["bridge_dropped_records"] = bridge_drops
            summary = result.get("summary")
            if isinstance(summary, dict) and (
                self._failure is not None or bridge_drops > 0
            ):
                summary = dict(summary)
                summary["status"] = "partial"
                if bridge_drops > 0:
                    summary["bridge_dropped_records"] = bridge_drops
                if self._failure is not None:
                    summary["status"] = "partial"
                    summary["bridge_failure"] = self._failure
                result["summary"] = summary
            return result
        return {
            "summary": {
                "status": "partial",
                "bridge_failure": self._failure or "summary has not completed",
                "bridge_dropped_records": self._upstream_dropped_records
                + max(0, self._dropped_records - self._callback_rejected_records),
                "retention_limit": self.maximum_samples,
            },
            "stale_inference_timing_diagnostic": {
                "status": "unavailable",
                "reason": self._failure or "summary has not completed",
            },
            "bridge_dropped_records": self._upstream_dropped_records
            + max(0, self._dropped_records - self._callback_rejected_records),
        }

    def failure_metadata(self) -> dict[str, Any]:
        writer_dropped_records = max(
            0, self._dropped_records - self._callback_rejected_records
        )
        return {
            "status": (
                "partial"
                if self._failure is not None
                or self._dropped_records > 0
                or self._upstream_dropped_records > 0
                else "complete"
            ),
            "executable": str(self.executable_path) if self.executable_path else None,
            "failure": self._failure,
            "dropped_records": self._dropped_records,
            "callback_rejected_records": self._callback_rejected_records,
            "writer_dropped_records": writer_dropped_records,
            "upstream_dropped_records": self._upstream_dropped_records,
            "queue_capacity": self.queue_capacity,
            "maximum_samples": self.maximum_samples,
            "maximum_line_bytes": self.maximum_line_bytes,
            "write_timeout_seconds": self.write_timeout_seconds,
            "return_code": self._return_code,
            "stderr_tail": self._stderr_tail,
        }

    async def close(self) -> None:
        """Close a started bridge even when capture exits before finalization."""
        if self._closed:
            return
        self._closed = True
        if self._writer_task is not None and not self._writer_task.done():
            try:
                await asyncio.wait_for(
                    self._queue.put(self._FINISH_SENTINEL),
                    timeout=self.write_timeout_seconds,
                )
                await asyncio.wait_for(
                    self._writer_task,
                    timeout=max(
                        self.write_timeout_seconds,
                        float(
                            config.get(
                                "performance_audit.export_drain_timeout_seconds", 5.0
                            )
                        ),
                    ),
                )
            except asyncio.TimeoutError:
                self._record_failure("timed out closing Rust timing summary writer")
                await self._stop_process()
        if self._process is not None and self._process.returncode is None:
            await self._stop_process()


class RustTrajectorySummaryBridge:
    """Forward trajectory receipts to a bounded native summary process.

    The receive callback only enqueues an object with ``put_nowait``. JSON
    encoding, pipe writes and child-process waits happen on a separate task so
    a slow audit consumer cannot delay state delivery or the WebSocket reader.
    """

    _EXECUTABLE_NAME = "billiards-performance-audit-trajectory-summary"
    _MAXIMUM_SAMPLES = 1_000_000
    _MAXIMUM_RECORDS = 5_000_000
    _MAXIMUM_LINE_BYTES = 16 * 1024 * 1024
    _DEFAULT_LINE_BYTES = 1_048_576
    _MAXIMUM_QUEUE_CAPACITY = 100_000

    def __init__(
        self,
        *,
        maximum_samples: int,
        maximum_records: int,
        queue_capacity: int,
        write_timeout_seconds: float,
        executable: str | Path | None = None,
    ) -> None:
        if maximum_samples < 1:
            raise ValueError("trajectory summary sample limit must be positive")
        if maximum_records < 2:
            raise ValueError("trajectory summary record limit must be at least two")
        if queue_capacity < 1:
            raise ValueError("trajectory summary queue capacity must be positive")
        if not math.isfinite(write_timeout_seconds) or write_timeout_seconds <= 0:
            raise ValueError("trajectory summary write timeout must be positive")
        self.requested_maximum_samples = maximum_samples
        self.maximum_samples = min(maximum_samples, self._MAXIMUM_SAMPLES)
        self.requested_maximum_records = maximum_records
        self.maximum_records = min(maximum_records, self._MAXIMUM_RECORDS)
        self.queue_capacity = min(
            queue_capacity, self.maximum_records, self._MAXIMUM_QUEUE_CAPACITY
        )
        self.maximum_line_bytes = self._DEFAULT_LINE_BYTES
        self.write_timeout_seconds = write_timeout_seconds
        self.configured_executable = executable
        self.executable_path: Path | None = None
        self._queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(
            maxsize=self.queue_capacity
        )
        self._process: asyncio.subprocess.Process | None = None
        self._writer_task: asyncio.Task[None] | None = None
        self._result: dict[str, Any] | None = None
        self._failure: str | None = None
        self._callback_rejected_records = 0
        self._writer_dropped_records = 0
        self._submitted_records = 0
        self._exporter_loss_count: int | None = None
        self._return_code: int | None = None
        self._stderr_tail = ""
        self._prepared = False
        self._finished = False
        self._closed = False

    async def prepare(self) -> None:
        """Resolve or build the reducer before a measured socket is opened."""
        if self._prepared or self._closed:
            return
        self._prepared = True
        try:
            # Cargo build or path discovery can take time; keep it off the
            # collector event loop and finish it before the measured socket
            # exists, so a cold developer build cannot perturb delivery.
            self.executable_path = await asyncio.to_thread(
                _rust_audit_summary_executable,
                self._EXECUTABLE_NAME,
                (
                    str(self.configured_executable)
                    if self.configured_executable is not None
                    else os.environ.get(
                        "PERFORMANCE_AUDIT_TRAJECTORY_SUMMARY_EXECUTABLE"
                    )
                ),
            )
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            self._record_failure(f"{type(error).__name__}: {error}")

    async def start(self) -> None:
        """Launch the prepared reducer and send its bounded retention setting."""
        if self._process is not None or self._closed:
            return
        if not self._prepared:
            await self.prepare()
        if self._failure is not None or self.executable_path is None:
            return
        try:
            self._process = await asyncio.create_subprocess_exec(
                str(self.executable_path),
                "--maximum-samples",
                str(self.maximum_samples),
                "--maximum-records",
                str(self.maximum_records),
                "--maximum-line-bytes",
                str(self.maximum_line_bytes),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            self._record_failure(f"{type(error).__name__}: {error}")
            return
        self._writer_task = asyncio.create_task(
            self._write_messages(), name="rust-trajectory-summary-writer"
        )
        await self._enqueue_control(
            {"type": "config", "maximum_samples": self.maximum_samples}
        )

    def try_record(self, event: dict[str, Any]) -> bool:
        """Enqueue one trajectory receipt without waiting for the child."""
        if (
            self._process is None
            or self._failure is not None
            or self._closed
            or self._finished
            or (self._writer_task is not None and self._writer_task.done())
        ):
            self._callback_rejected_records += 1
            return False
        # Reserve one protocol record for the final summary control message.
        if self._submitted_records >= self.maximum_records - 1:
            self._callback_rejected_records += 1
            return False
        try:
            self._queue.put_nowait({"type": "trajectory_receipt", "event": event})
        except asyncio.QueueFull:
            self._callback_rejected_records += 1
            return False
        self._submitted_records += 1
        return True

    async def _enqueue_control(self, record: dict[str, Any]) -> None:
        if (
            self._process is None
            or self._failure is not None
            or self._closed
            or (self._writer_task is not None and self._writer_task.done())
        ):
            return
        try:
            await asyncio.wait_for(
                self._queue.put(record), timeout=self.write_timeout_seconds
            )
        except asyncio.TimeoutError:
            self._record_failure("timed out enqueueing trajectory summary control")
            return
        self._submitted_records += 1

    async def _write_messages(self) -> None:
        process = self._process
        if process is None or process.stdin is None:
            return
        current: dict[str, Any] | None = None
        try:
            while True:
                current = await self._queue.get()
                try:
                    encoded = (
                        json.dumps(
                            _finite_audit_json_value(current),
                            separators=(",", ":"),
                            allow_nan=False,
                        )
                        + "\n"
                    ).encode("utf-8")
                    if len(encoded) > self.maximum_line_bytes:
                        raise ValueError(
                            "trajectory summary record exceeds the configured "
                            f"{self.maximum_line_bytes}-byte line limit"
                        )
                    process.stdin.write(encoded)
                    await asyncio.wait_for(
                        process.stdin.drain(), timeout=self.write_timeout_seconds
                    )
                    should_finish = current.get("type") == "finish"
                    current = None
                finally:
                    self._queue.task_done()
                if should_finish:
                    break
        except asyncio.CancelledError:
            if current is not None and current.get("type") == "trajectory_receipt":
                self._writer_dropped_records += 1
            self._record_failure("Rust trajectory summary writer was cancelled")
            self._discard_queued_records()
            raise
        except Exception as error:
            if current is not None and current.get("type") == "trajectory_receipt":
                self._writer_dropped_records += 1
            self._record_failure(f"{type(error).__name__}: {error}")
            self._discard_queued_records()
        finally:
            try:
                process.stdin.close()
            except (BrokenPipeError, ConnectionError, OSError):
                pass

    def _discard_queued_records(self) -> None:
        while True:
            try:
                queued = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            if queued.get("type") == "trajectory_receipt":
                self._writer_dropped_records += 1
            self._queue.task_done()

    def _record_failure(self, message: str) -> None:
        if self._failure is None:
            self._failure = message

    async def finish(
        self,
        duration_seconds: float,
        *,
        exporter_loss_count: int | None = None,
    ) -> dict[str, Any] | None:
        """Close the native measurement window and collect its one summary."""
        if self._finished:
            return self._result
        self._finished = True
        self._exporter_loss_count = exporter_loss_count
        if self._process is None or self._writer_task is None:
            return None
        await self._enqueue_control(
            {
                "type": "finish",
                "duration_seconds": duration_seconds,
                "exporter_loss_count": exporter_loss_count,
                "bridge_loss_count": self._callback_rejected_records,
            }
        )
        if self._failure is None:
            try:
                await asyncio.wait_for(
                    self._writer_task,
                    timeout=max(
                        self.write_timeout_seconds,
                        float(
                            config.get(
                                "performance_audit.export_drain_timeout_seconds", 5.0
                            )
                        ),
                    ),
                )
                await self._collect_process_result()
            except (asyncio.TimeoutError, OSError, RuntimeError) as error:
                self._record_failure(f"{type(error).__name__}: {error}")
                await self._stop_process()
        if self._process is not None and self._return_code not in (None, 0):
            self._record_failure(
                f"Rust trajectory summarizer exited with status {self._return_code}: "
                f"{self._stderr_tail.strip()}"
            )
        if self._failure is None and self._result is None:
            self._record_failure("Rust trajectory summarizer returned no JSON result")
        return self._result

    async def _collect_process_result(self) -> None:
        process = self._process
        if process is None:
            return
        stdout, stderr = await asyncio.wait_for(
            process.communicate(), timeout=self.write_timeout_seconds
        )
        self._return_code = process.returncode
        self._stderr_tail = stderr.decode("utf-8", errors="replace")[-4096:]
        if stdout:
            try:
                result = json.loads(stdout)
                if not isinstance(result, dict):
                    raise ValueError("trajectory summary output must be an object")
                self._result = result
            except (json.JSONDecodeError, ValueError) as error:
                self._record_failure(f"invalid Rust trajectory summary output: {error}")
        elif self._return_code == 0:
            self._record_failure("Rust trajectory summarizer produced empty output")

    async def _stop_process(self) -> None:
        process = self._process
        if process is None or process.returncode is not None:
            return
        process.terminate()
        try:
            await asyncio.wait_for(process.wait(), timeout=self.write_timeout_seconds)
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
        self._return_code = process.returncode

    def failure_metadata(self) -> dict[str, Any]:
        """Disclose partial native summaries and every bounded bridge loss."""
        return {
            "status": (
                "partial"
                if self._failure is not None or self._callback_rejected_records > 0
                else "complete"
            ),
            "executable": str(self.executable_path) if self.executable_path else None,
            "failure": self._failure,
            "callback_rejected_records": self._callback_rejected_records,
            "writer_dropped_records": self._writer_dropped_records,
            "exporter_loss_count": self._exporter_loss_count,
            "maximum_samples_requested": self.requested_maximum_samples,
            "maximum_samples": self.maximum_samples,
            "maximum_records_requested": self.requested_maximum_records,
            "maximum_records": self.maximum_records,
            "queue_capacity": self.queue_capacity,
            "maximum_line_bytes": self.maximum_line_bytes,
            "return_code": self._return_code,
            "stderr_tail": self._stderr_tail,
        }

    async def close(self) -> None:
        """Stop an unfinished child after an interrupted measurement."""
        if self._closed:
            return
        self._closed = True
        if self._writer_task is not None and not self._writer_task.done():
            self._writer_task.cancel()
            await asyncio.gather(self._writer_task, return_exceptions=True)
        await self._stop_process()


class RustLiveChildResourceSampler:
    """Control the native bounded sampler for one supervised Rust child."""

    _EXECUTABLE_NAME = "billiards-performance-audit-resources"
    _PROTOCOL_SCHEMA_VERSION = 2
    _MAXIMUM_LINE_BYTES = 16 * 1024 * 1024
    _MAXIMUM_RETAINED_SAMPLES = 50_000

    def __init__(
        self,
        *,
        process_id: int,
        interval_seconds: float,
        maximum_samples: int,
        startup_timeout_seconds: float,
        write_timeout_seconds: float,
        executable: str | Path | None = None,
    ) -> None:
        if (
            isinstance(process_id, bool)
            or not isinstance(process_id, int)
            or process_id < 1
        ):
            raise ValueError("process_id must be a positive integer")
        if (
            isinstance(interval_seconds, bool)
            or not isinstance(interval_seconds, (int, float))
            or not math.isfinite(interval_seconds)
            or interval_seconds <= 0
        ):
            raise ValueError("resource sample interval must be positive")
        if (
            isinstance(maximum_samples, bool)
            or not isinstance(maximum_samples, int)
            or maximum_samples < 1
        ):
            raise ValueError("maximum resource samples must be positive")
        if not math.isfinite(startup_timeout_seconds) or startup_timeout_seconds <= 0:
            raise ValueError("resource sampler startup timeout must be positive")
        if not math.isfinite(write_timeout_seconds) or write_timeout_seconds <= 0:
            raise ValueError("resource sampler write timeout must be positive")
        self._process_id = process_id
        self._interval_seconds = interval_seconds
        self._requested_maximum_samples = maximum_samples
        self._maximum_samples = min(maximum_samples, self._MAXIMUM_RETAINED_SAMPLES)
        self._startup_timeout_seconds = startup_timeout_seconds
        self._write_timeout_seconds = write_timeout_seconds
        self._configured_executable = executable
        self._process: asyncio.subprocess.Process | None = None
        self._started = False
        self._closed = False
        self._failure: str | None = None
        self._native_summary: dict[str, Any] | None = None
        self._started_process_start_time: int | None = None

    @classmethod
    def resolve_executable(cls, configured: str | Path | None = None) -> Path:
        """Find an explicit executable, current Cargo artifact, or PATH entry."""
        executable_name = cls._EXECUTABLE_NAME + (".exe" if os.name == "nt" else "")
        if configured is not None:
            path = Path(configured).expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(
                    f"resource sampler executable does not exist: {path}"
                )
            if os.name != "nt" and not os.access(path, os.X_OK):
                raise PermissionError(
                    f"resource sampler executable is not executable: {path}"
                )
            return path
        configured_target_dir = os.environ.get("CARGO_TARGET_DIR")
        target_directories = [
            Path(configured_target_dir).expanduser() if configured_target_dir else None,
            ROOT / "perception" / "target",
            ROOT / "target",
        ]
        for target_directory in target_directories:
            if target_directory is None:
                continue
            if not target_directory.is_absolute():
                target_directory = Path.cwd() / target_directory
            for profile in ("release", "debug"):
                candidate = target_directory / profile / executable_name
                if candidate.is_file() and (
                    os.name == "nt" or os.access(candidate, os.X_OK)
                ):
                    return candidate.resolve()
        on_path = shutil.which(executable_name)
        if on_path:
            return Path(on_path).resolve()
        raise FileNotFoundError(
            f"could not find {executable_name} on PATH or in Cargo target directories; "
            f"build it with `cargo build --locked --manifest-path "
            f"{ROOT / 'perception' / 'Cargo.toml'} --bin {cls._EXECUTABLE_NAME}`"
        )

    async def start(self) -> None:
        """Start the native helper and begin sampling after its ready handshake."""
        if self._process is not None or self._closed:
            return
        try:
            executable = self.resolve_executable(self._configured_executable)
            process = await asyncio.create_subprocess_exec(
                str(executable),
                "--process-id",
                str(self._process_id),
                "--sample-interval-ms",
                str(max(1, round(self._interval_seconds * 1000))),
                "--maximum-samples",
                str(self._maximum_samples),
                "--startup-timeout-ms",
                str(max(1, round(self._startup_timeout_seconds * 1000))),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                limit=self._MAXIMUM_LINE_BYTES,
            )
            self._process = process
            await self._read_message("ready", self._startup_timeout_seconds)
            await self._send_control("start")
            started = await self._read_message("started", self._startup_timeout_seconds)
            if started.get("process_id") != self._process_id:
                raise ValueError("native sampler started with a different process ID")
            process_start_time = started.get("process_start_time_unix_seconds")
            if process_start_time is not None and (
                isinstance(process_start_time, bool)
                or not isinstance(process_start_time, int)
            ):
                raise ValueError(
                    "native sampler returned an invalid process start time"
                )
            self._started_process_start_time = process_start_time
            self._started = True
        except asyncio.CancelledError:
            await self._stop_process_shielded()
            raise
        except (OSError, ValueError, RuntimeError, asyncio.TimeoutError) as error:
            self._record_failure(f"{type(error).__name__}: {error}")
            await self._stop_process()

    async def close(self) -> None:
        """Stop sampling and retain the bounded native result."""
        if self._closed:
            return
        self._closed = True
        process = self._process
        if process is None:
            return
        try:
            if self._started and self._failure is None:
                await self._send_control("stop")
                message = await self._read_message(
                    "summary", self._write_timeout_seconds
                )
                summary = message.get("summary")
                if not isinstance(summary, dict):
                    raise ValueError("native resource summary has an invalid shape")
                if summary.get("process_id") != self._process_id:
                    raise ValueError("native sampler summarized a different process ID")
                if (
                    summary.get("process_start_time_unix_seconds")
                    != self._started_process_start_time
                ):
                    raise ValueError(
                        "native sampler process identity changed during sampling"
                    )
                self._native_summary = summary
            await asyncio.wait_for(process.wait(), timeout=self._write_timeout_seconds)
            stderr = b""
            if process.stderr is not None:
                stderr = await process.stderr.read()
            if process.returncode != 0:
                detail = stderr.decode("utf-8", errors="replace")[-4096:]
                raise RuntimeError(
                    f"native resource sampler exited with status {process.returncode}: {detail}"
                )
        except asyncio.CancelledError:
            await self._stop_process_shielded()
            raise
        except (OSError, ValueError, RuntimeError, asyncio.TimeoutError) as error:
            self._record_failure(f"{type(error).__name__}: {error}")
            await self._stop_process_shielded()

    async def _send_control(self, kind: str) -> None:
        process = self._process
        if process is None or process.stdin is None:
            raise RuntimeError("native resource sampler control pipe is unavailable")
        message = json.dumps({"kind": kind}, separators=(",", ":")).encode() + b"\n"
        process.stdin.write(message)
        await asyncio.wait_for(
            process.stdin.drain(), timeout=self._write_timeout_seconds
        )

    async def _read_message(
        self, expected_kind: str, timeout_seconds: float
    ) -> dict[str, Any]:
        process = self._process
        if process is None or process.stdout is None:
            raise RuntimeError("native resource sampler output pipe is unavailable")
        line = await asyncio.wait_for(
            process.stdout.readline(), timeout=timeout_seconds
        )
        if not line:
            try:
                await asyncio.wait_for(process.wait(), timeout=timeout_seconds)
            except asyncio.TimeoutError:
                raise RuntimeError(
                    "native resource sampler closed its output before the "
                    f"{expected_kind!r} message"
                ) from None
            detail = ""
            if process.stderr is not None:
                detail = (await process.stderr.read()).decode(
                    "utf-8", errors="replace"
                )[-4096:]
            raise RuntimeError(
                "native resource sampler exited before the "
                f"{expected_kind!r} message"
                + (f": {detail.strip()}" if detail.strip() else "")
            )
        if len(line) > self._MAXIMUM_LINE_BYTES:
            raise RuntimeError("native resource sampler returned an oversized message")
        message = json.loads(line)
        if (
            not isinstance(message, dict)
            or message.get("schema_version") != self._PROTOCOL_SCHEMA_VERSION
            or message.get("kind") != expected_kind
        ):
            raise ValueError(
                f"native resource sampler did not return {expected_kind!r}"
            )
        return message

    async def _stop_process(self) -> None:
        process = self._process
        if process is None or process.returncode is not None:
            return
        process.terminate()
        try:
            await asyncio.wait_for(process.wait(), timeout=self._write_timeout_seconds)
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()

    async def _stop_process_shielded(self) -> None:
        cleanup = asyncio.create_task(self._stop_process())
        try:
            await asyncio.shield(cleanup)
        except asyncio.CancelledError:
            await cleanup
            raise

    def _record_failure(self, message: str) -> None:
        if self._failure is None:
            self._failure = message

    def summary(self) -> dict[str, Any]:
        """Adapt native resource metrics to the established report shape."""
        native = self._native_summary
        if native is None:
            reason = self._failure or "native resource sampler returned no summary"
            return {
                "status": "unavailable",
                "unavailable_reason": reason,
                "process_role": "rust_live_inference_child",
                "process_id": self._process_id,
                "process_identity_resolution": "unavailable",
                "process_identity_resolution_note": None,
                "sample_count": 0,
                "available_sample_count": 0,
                "unavailable_sample_count": 0,
                "retained_sample_count": 0,
                "retention_limit": self._maximum_samples,
                "requested_retention_limit": self._requested_maximum_samples,
                "samples_truncated": False,
                "statistics_window": "unavailable",
                "cpu_percent": {
                    "count": 0,
                    "p50": None,
                    "p95": None,
                    "p99": None,
                    "maximum": None,
                },
                "cpu_time_seconds": {"first": None, "last": None, "delta": None},
                "resident_memory_bytes": {
                    "count": 0,
                    "first": None,
                    "last": None,
                    "minimum": None,
                    "maximum": None,
                    "delta": None,
                },
                "retained_samples": [],
            }

        cpu_time = native.get("accumulated_cpu_time_ms", {})
        if not isinstance(cpu_time, dict):
            cpu_time = {}
        memory = native.get("resident_memory_bytes", {})
        if not isinstance(memory, dict):
            memory = {}
        samples = native.get("retained_samples", [])
        if not isinstance(samples, list):
            samples = []

        def milliseconds_to_seconds(value: Any) -> float | None:
            if isinstance(value, int) and not isinstance(value, bool):
                return value / 1000
            return None

        retained_samples = [
            {
                "elapsed_ms": sample.get("elapsed_ms"),
                "status": sample.get("status"),
                "unavailable_reason": sample.get("unavailable_reason"),
                "cpu_percent": sample.get("cpu_percent_per_core"),
                "cpu_time_seconds": milliseconds_to_seconds(
                    sample.get("accumulated_cpu_time_ms")
                ),
                "resident_memory_bytes": sample.get("resident_memory_bytes"),
                "clock": "resource_sampler_process_monotonic",
            }
            for sample in samples
            if isinstance(sample, dict)
        ]
        return {
            "status": native.get("status", "unavailable"),
            "unavailable_reason": native.get("unavailable_reason"),
            "process_role": "rust_live_inference_child",
            "process_id": native.get("process_id", self._process_id),
            "process_start_time_unix_seconds": native.get(
                "process_start_time_unix_seconds"
            ),
            "process_identity_resolution": native.get(
                "process_identity_resolution", "unavailable"
            ),
            "process_identity_resolution_note": native.get(
                "process_identity_resolution_note"
            ),
            "sample_interval_ms": native.get(
                "sample_interval_ms", round(self._interval_seconds * 1000)
            ),
            "sample_count": native.get("sample_count", 0),
            "available_sample_count": native.get("available_sample_count", 0),
            "unavailable_sample_count": native.get("unavailable_sample_count", 0),
            "retained_sample_count": native.get("retained_sample_count", 0),
            "retention_limit": native.get("retention_limit", self._maximum_samples),
            "requested_retention_limit": self._requested_maximum_samples,
            "samples_truncated": native.get("samples_truncated", False),
            "statistics_window": native.get("statistics_window", "unavailable"),
            "cpu_percent": native.get(
                "cpu_percent_per_core",
                {"count": 0, "p50": None, "p95": None, "p99": None, "maximum": None},
            ),
            "cpu_time_seconds": {
                "first": milliseconds_to_seconds(cpu_time.get("first")),
                "last": milliseconds_to_seconds(cpu_time.get("last")),
                "delta": milliseconds_to_seconds(cpu_time.get("delta")),
            },
            "resident_memory_bytes": {
                "count": sum(
                    sample.get("resident_memory_bytes") is not None
                    for sample in samples
                    if isinstance(sample, dict)
                ),
                "first": memory.get("first"),
                "last": memory.get("last"),
                "minimum": memory.get("minimum"),
                "maximum": memory.get("maximum"),
                "delta": memory.get("delta"),
            },
            "retained_samples": retained_samples,
        }


class RustLiveQualificationBroadcaster:
    """Record adapted game states in bounded memory and export metadata only."""

    def __init__(
        self,
        *,
        repetition: int,
        recorder: AsyncAuditRecorder,
        history_size: int,
    ) -> None:
        self._repetition = repetition
        self._recorder = recorder
        self._history_size = history_size
        self._warmup_messages = 0
        self._measurement: RustLiveStateAccumulator | None = None
        self._measurement_open = False

    @property
    def warmup_messages(self) -> int:
        return self._warmup_messages

    @property
    def measurement(self) -> RustLiveStateAccumulator:
        if self._measurement is None:
            raise RuntimeError("measurement has not started")
        return self._measurement

    def begin_measurement(self) -> int:
        """Start a measured window after source readiness and configured warm-up."""
        self._measurement = RustLiveStateAccumulator(self._history_size)
        self._measurement_open = True
        return time.perf_counter_ns()

    def end_measurement(self) -> int:
        self._measurement_open = False
        return time.perf_counter_ns()

    async def broadcast_game_state(
        self,
        balls: object,
        cue: dict[str, Any] | None = None,
        table: dict[str, Any] | None = None,
        timestamp: Any = None,
        observation_status: dict[str, Any] | None = None,
        game_type: str | None = None,
    ) -> None:
        """Capture state locations and identities without retaining any image."""
        del table, timestamp, game_type
        if not self._measurement_open:
            self._warmup_messages += 1
            return
        observation = observation_status if isinstance(observation_status, dict) else {}
        identity = observation.get("observation_identity")
        identity = identity if isinstance(identity, dict) else {}
        balls_value = balls
        balls = balls_value if isinstance(balls_value, list) else []
        compact_balls = [
            {
                "id": ball.get("id"),
                "x": (
                    (ball.get("position") or {}).get("x")
                    if isinstance(ball.get("position"), dict)
                    else None
                ),
                "y": (
                    (ball.get("position") or {}).get("y")
                    if isinstance(ball.get("position"), dict)
                    else None
                ),
                "table_position": ball.get("table_position"),
                "is_observed": ball.get("is_observed"),
                "tracking_status": ball.get("tracking_status"),
                "identity_status": ball.get("identity_status"),
                "number": ball.get("number"),
                "confidence": ball.get("confidence"),
            }
            for ball in balls
            if isinstance(ball, dict)
        ]
        frame_id = observation.get("source_frame_id")
        timestamp_kind = observation.get("source_timestamp_kind")
        timestamp_stage = (
            timestamp_kind.rsplit(":", 1)[-1]
            if isinstance(timestamp_kind, str)
            else None
        )
        event = {
            "event_type": "rust_live_state_received",
            "repetition": self._repetition,
            "engine_instance_id": identity.get("engine_instance_id")
            or observation.get("engine_instance_id"),
            "source_session_id": identity.get("session_id")
            or observation.get("source_session_id"),
            "source_frame_id": identity.get("frame_id") or frame_id,
            "source_snapshot_revision": identity.get("snapshot_revision")
            or observation.get("snapshot_revision"),
            "source_transport_sequence": observation.get("transport_sequence"),
            "source_timestamp_ticks": observation.get("source_timestamp"),
            "source_timestamp_unit": observation.get("source_timestamp_unit"),
            "source_timestamp_kind": timestamp_kind,
            "source_timestamp_stage": timestamp_stage,
            "source_timestamp_is_exposure": observation.get(
                "source_timestamp_is_exposure"
            ),
            "configuration_revision": observation.get("configuration_revision"),
            "rectification_revision": observation.get("rectification_revision"),
            "calibration_revision": observation.get("calibration_revision"),
            "state_receipt_monotonic_ns": time.perf_counter_ns(),
            "state_complete": bool(
                frame_id
                and observation.get("source_timestamp") is not None
                and isinstance(balls_value, list)
            ),
            "ball_count": len(compact_balls),
            "cue_present": cue is not None,
            "balls": compact_balls,
        }
        self.measurement.add(event)
        self._recorder.record(event)


class TrajectoryAccumulator:
    """Bounded counters for trajectory publication and source freshness."""

    def __init__(self, maximum_samples: int = 10000) -> None:
        self.messages = 0
        self.repeated_source_observations = 0
        self.unique_source_observations = 0
        self.invalidated_messages = 0
        self._last_source = _BoundedIdentityWindow(maximum_samples)
        self._ages: deque[float] = deque(maxlen=maximum_samples)
        self.source_sequence_gaps = 0
        self.source_out_of_order = 0
        self._last_stream_sequence: int | None = None
        self.stream_sequence_gaps = 0
        self.stream_sequence_duplicates = 0
        self.stream_sequence_out_of_order = 0
        self.incomplete_messages = 0
        self._source_to_receipt_ms: deque[float] = deque(maxlen=maximum_samples)

    def add(self, event: dict[str, Any]) -> None:
        self.messages += 1
        sequence = event.get("stream_sequence")
        source = event.get("source_frame_number")
        if not (
            isinstance(sequence, int)
            and not isinstance(sequence, bool)
            and isinstance(source, int)
            and not isinstance(source, bool)
        ):
            self.incomplete_messages += 1
        if isinstance(sequence, int) and not isinstance(sequence, bool):
            if self._last_stream_sequence is not None:
                if sequence > self._last_stream_sequence + 1:
                    self.stream_sequence_gaps += (
                        sequence - self._last_stream_sequence - 1
                    )
                elif sequence == self._last_stream_sequence:
                    self.stream_sequence_duplicates += 1
                elif sequence < self._last_stream_sequence:
                    self.stream_sequence_out_of_order += 1
            if (
                self._last_stream_sequence is None
                or sequence > self._last_stream_sequence
            ):
                self._last_stream_sequence = sequence
        source_identity = "|".join(
            str(event.get(field) or "unknown")
            for field in ("vision_instance_id", "source_process_id")
        )
        if isinstance(source, int) and not isinstance(source, bool):
            previous = self._last_source.get(source_identity)
            if previous is None:
                self.unique_source_observations += 1
                self._last_source.set(source_identity, source)
            elif source == previous:
                self.repeated_source_observations += 1
            elif source > previous:
                self.unique_source_observations += 1
                self.source_sequence_gaps += max(0, source - previous - 1)
                self._last_source.set(source_identity, source)
            else:
                self.source_out_of_order += 1
        if (
            event.get("prediction_status") == "invalidated"
            or event.get("spatial_valid") is False
        ):
            self.invalidated_messages += 1
        age = event.get("source_publication_wall_age_estimate_ms")
        if age is None:
            age = event.get("source_publication_age_ms")
        if isinstance(age, (int, float)) and math.isfinite(age):
            self._ages.append(float(age))
        source_publication_to_receipt_ms = event.get(
            "source_publication_to_client_receipt_ms"
        )
        if (
            event.get("clock_comparable") is True
            and isinstance(source_publication_to_receipt_ms, (int, float))
            and math.isfinite(source_publication_to_receipt_ms)
        ):
            self._source_to_receipt_ms.append(float(source_publication_to_receipt_ms))

    def summary(self, duration: float) -> dict[str, Any]:
        return {
            "trajectory_messages_received": self.messages,
            "trajectory_messages_per_second": (
                self.messages / duration if duration else 0
            ),
            "incomplete_trajectory_messages": self.incomplete_messages,
            "trajectory_sequence_gaps": self.stream_sequence_gaps,
            "trajectory_sequence_duplicates": self.stream_sequence_duplicates,
            "trajectory_sequence_out_of_order_messages": self.stream_sequence_out_of_order,
            "trajectory_unique_source_observations": self.unique_source_observations,
            "trajectory_repeated_source_observations": self.repeated_source_observations,
            "trajectory_source_sequence_gaps": self.source_sequence_gaps,
            "trajectory_source_out_of_order_messages": self.source_out_of_order,
            "trajectory_source_identity_deduplication_window": (
                self._last_source.summary()
            ),
            "trajectory_invalidated_or_spatially_invalid_messages": self.invalidated_messages,
            "trajectory_source_publication_wall_age_estimate_ms": {
                "count": len(self._ages),
                "sample_limit": self._ages.maxlen,
                "p50": percentile(list(self._ages), 0.50),
                "p95": percentile(list(self._ages), 0.95),
                "p99": percentile(list(self._ages), 0.99),
            },
            "trajectory_source_publication_to_client_receipt_ms": {
                "count": len(self._source_to_receipt_ms),
                "sample_limit": self._source_to_receipt_ms.maxlen,
                "p50": percentile(list(self._source_to_receipt_ms), 0.50),
                "p95": percentile(list(self._source_to_receipt_ms), 0.95),
                "p99": percentile(list(self._source_to_receipt_ms), 0.99),
            },
        }


def _project_metadata() -> dict[str, Any]:
    def git(*args: str) -> str | None:
        try:
            return subprocess.check_output(
                ["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    git_revision = git("rev-parse", "HEAD")
    configured_revision = config.get("metadata.source_revision") or os.getenv(
        "APP_REVISION"
    )
    revision = git_revision or configured_revision
    if git_revision is not None:
        revision_status = "git_checkout"
    elif configured_revision:
        revision_status = "configured_build_revision"
    else:
        revision_status = "unavailable"

    configured_output = config.get(
        "performance_audit.output_directory", "thoughts/performance-audits"
    )
    output_root = Path(str(configured_output)).expanduser()
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    try:
        output_relative = output_root.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        output_relative = None

    changed_paths: set[str] = set()
    tracked_changes = git("diff", "--name-only", "-z", "HEAD", "--")
    if tracked_changes:
        changed_paths.update(path for path in tracked_changes.split("\0") if path)
    untracked_changes = git("ls-files", "--others", "--exclude-standard", "-z")
    if untracked_changes:
        changed_paths.update(path for path in untracked_changes.split("\0") if path)

    def is_audit_artifact(relative_path: str) -> bool:
        return bool(
            output_relative
            and (
                relative_path == output_relative
                or relative_path.startswith(output_relative + "/")
            )
        )

    relevant_paths = sorted(
        path for path in changed_paths if not is_audit_artifact(path)
    )
    dirty_digest = hashlib.sha256()
    for relative_path in relevant_paths:
        changed_path = ROOT / relative_path
        dirty_digest.update(relative_path.encode("utf-8"))
        dirty_digest.update(b"\0")
        if changed_path.is_file():
            dirty_digest.update(_hash_file(changed_path).encode("ascii"))
        elif changed_path.is_symlink():
            dirty_digest.update(str(changed_path.readlink()).encode("utf-8"))
        else:
            dirty_digest.update(b"<deleted-or-non-file>")

    selected = {
        "performance_audit": config.get("performance_audit", {}),
        "perception": config.get("perception", {}),
        "camera": config.get("vision.camera", {}),
        "detection": config.get("vision.detection", {}),
        "processing": config.get("vision.processing", {}),
        "vision_performance": config.get("vision.performance", {}),
        "video": config.get("video", {}),
        "calibration": config.get("table", {}),
        "websocket_delivery": config.get("api.websocket.delivery", {}),
        "stream_quality": config.get("api.stream.quality", {}),
    }
    try:
        import psutil

        hardware = {
            "logical_cpu_count": psutil.cpu_count(logical=True),
            "physical_cpu_count": psutil.cpu_count(logical=False),
            "system_memory_bytes": psutil.virtual_memory().total,
        }
    except Exception:
        hardware = {
            "logical_cpu_count": None,
            "physical_cpu_count": None,
            "system_memory_bytes": None,
        }
    perception_provider = config.get("perception.provider")
    rust_replay = config.get("perception.rust_replay", {})
    if not isinstance(rust_replay, dict):
        rust_replay = {}
    if perception_provider == "rust":
        model_value = rust_replay.get("onnx_path")
        input_value = rust_replay.get("input_path")
        executable_value = rust_replay.get("executable_path")
        source_model_value = rust_replay.get("source_checkpoint_path")
    else:
        model_value = config.get("vision.detection.yolo_model_path")
        input_value = config.get("vision.camera.video_file_path")
        executable_value = None
        source_model_value = None
    model_path = _resolve_configured_path(model_value)
    input_path = _resolve_configured_path(input_value)
    executable_path = _resolve_configured_path(executable_value)
    source_model_path = _resolve_configured_path(source_model_value)
    model_identity = {
        "configured_path": str(model_value) if model_value else None,
        "exists": bool(model_path and model_path.exists()),
        "sha256": None,
    }
    if model_path is not None and (model_path.is_file() or model_path.is_dir()):
        model_identity["sha256"] = _hash_path(model_path)
    source_model_identity = {
        "configured_path": str(source_model_value) if source_model_value else None,
        "exists": bool(source_model_path and source_model_path.exists()),
        "sha256": (
            _hash_path(source_model_path)
            if source_model_path is not None
            and (source_model_path.is_file() or source_model_path.is_dir())
            else None
        ),
    }
    executable_identity = {
        "configured_path": str(executable_value) if executable_value else None,
        "exists": bool(executable_path and executable_path.is_file()),
        "sha256": (
            _hash_file(executable_path)
            if executable_path is not None and executable_path.is_file()
            else None
        ),
    }
    return {
        "revision": revision,
        "revision_status": revision_status,
        "dirty_worktree_sha256": (
            dirty_digest.hexdigest() if git_revision is not None else None
        ),
        "dirty_worktree_files": relevant_paths if git_revision is not None else None,
        "host": platform.node(),
        "platform": platform.platform(),
        "processor": platform.processor() or None,
        "hardware": hardware,
        "python": platform.python_version(),
        "perception_provider": perception_provider,
        "model": model_identity,
        "source_model": source_model_identity,
        "inference_executable": executable_identity,
        "configured_video_input": {
            "path": str(input_value) if input_value else None,
            "exists": bool(input_path and input_path.exists()),
            "sha256": (
                _hash_path(input_path)
                if input_path is not None
                and (input_path.is_file() or input_path.is_dir())
                else None
            ),
            "kind": (
                "recorded_frame_directory"
                if input_path is not None and input_path.is_dir()
                else "file" if input_path is not None and input_path.is_file() else None
            ),
        },
        "effective_configuration": selected,
        "runtime_measurement_capabilities": {
            "cpu_percent": "psutil process sample when supported",
            "resident_memory_bytes": "RSS sample; allocation counts unavailable",
            "accelerator_metrics": "unavailable unless separately exposed by the detector backend",
            "browser_render_and_physical_display_latency": "not measured",
        },
        "configuration_note": "Sensitive or unrelated settings are omitted.",
    }


def _resolve_configured_path(value: Any) -> Path | None:
    """Resolve an optional configured path relative to the repository root."""
    if not isinstance(value, str) or not value.strip():
        return None
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_path(path: Path) -> str:
    """Hash a model file or a directory tree deterministically."""
    if path.is_file():
        return _hash_file(path)
    digest = hashlib.sha256()
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(str(child.relative_to(path)).encode("utf-8"))
        digest.update(_hash_file(child).encode("ascii"))
    return digest.hexdigest()


def _http_status_url(websocket_url: str, route: str) -> str:
    """Map the local WebSocket origin to its configured API status route."""
    parsed = urlsplit(websocket_url)
    scheme = "https" if parsed.scheme == "wss" else "http"
    prefix = str(config.get("api.api_prefix", "/api/v1")).rstrip("/")
    return urlunsplit((scheme, parsed.netloc, f"{prefix}{route}", "", ""))


def _read_json_url(url: str, timeout_seconds: float = 2.0) -> dict[str, Any] | None:
    try:
        with urlopen(url, timeout=timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return payload if isinstance(payload, dict) else None
    except Exception:
        return None


async def _respond_to_server_ping(websocket: Any, message: dict[str, Any]) -> bool:
    """Answer the service's JSON heartbeat, separate from WebSocket control pings."""
    if message.get("type") != "ping":
        return False
    # The application handler tracks JSON pings to decide whether a subscriber is
    # alive. The websockets library's RFC 6455 ping_interval uses control frames
    # and does not refresh that application-level watchdog.
    await websocket.send(json.dumps({"type": "ping", "data": {}}))
    return True


async def _send_client_heartbeat(websocket: Any) -> None:
    """Send the application heartbeat independently of server-initiated pings."""
    await websocket.send(json.dumps({"type": "ping", "data": {}}))


def _wait_for_audit_export_drain(url: str) -> dict[str, Any] | None:
    """Wait briefly for accepted server events to reach the JSONL file."""
    timeout_seconds = float(
        config.get("performance_audit.export_drain_timeout_seconds", 5.0)
    )
    deadline = time.monotonic() + timeout_seconds
    status: dict[str, Any] | None = None
    while True:
        remaining = deadline - time.monotonic()
        status = _read_json_url(url, timeout_seconds=max(0.01, min(0.25, remaining)))
        export = status.get("event_export") if status else None
        if not isinstance(export, dict) or export.get("queue_depth", 0) == 0:
            return status
        if remaining <= 0:
            return status
        time.sleep(min(0.05, remaining))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read complete JSONL events while ignoring a partial final write."""
    events: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as source:
            for line in source:
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(value, dict):
                    events.append(value)
    except OSError:
        return []
    return events


def _delivery_latency_summary(
    client_events: list[dict[str, Any]],
    server_events: list[dict[str, Any]],
    *,
    deadline_budget_ms: float = 50.0,
    server_clock_domain_id: str | None = None,
) -> dict[str, Any]:
    """Delegate bounded delivery correlation and statistics to the Rust reducer."""
    maximum_events = 200_000
    maximum_input_bytes = 134_217_728
    events = [("client", client_events), ("server", server_events)]
    event_count = sum(len(group) for _, group in events)
    if event_count > maximum_events:
        raise ValueError(
            f"delivery audit input has {event_count} events; Rust limit is {maximum_events}"
        )

    clock_name = time.get_clock_info("perf_counter").implementation
    config_record = {
        "kind": "config",
        "deadline_budget_ms": deadline_budget_ms,
        "server_clock_domain_id": server_clock_domain_id,
        "clock_name": clock_name,
    }
    payload = bytearray()

    def append_record(record: dict[str, Any]) -> None:
        payload.extend(
            json.dumps(
                record,
                allow_nan=False,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        payload.extend(b"\n")
        if len(payload) > maximum_input_bytes:
            raise ValueError(
                "delivery audit event input exceeds the Rust summarizer's "
                f"{maximum_input_bytes}-byte limit"
            )

    append_record(config_record)
    for kind, group in events:
        for index, event in enumerate(group, start=1):
            if not isinstance(event, dict):
                raise ValueError(
                    f"delivery audit {kind} event {index} must be an object"
                )
            append_record({"kind": kind, "event": _finite_audit_json_value(event)})

    result = subprocess.run(
        [
            str(
                _rust_audit_summary_executable(
                    "billiards-performance-audit-delivery-summary",
                    os.environ.get("PERFORMANCE_AUDIT_DELIVERY_SUMMARY_EXECUTABLE"),
                )
            )
        ],
        input=bytes(payload),
        capture_output=True,
        timeout=120,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(
            "Rust delivery audit summarizer failed"
            + (f": {detail}" if detail else f" with exit status {result.returncode}")
        )
    try:
        summary = json.loads(result.stdout)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(
            "Rust delivery audit summarizer returned invalid JSON"
        ) from error
    if not isinstance(summary, dict):
        raise RuntimeError(
            "Rust delivery audit summarizer returned a non-object summary"
        )
    return summary


def _finite_audit_json_value(value: Any) -> Any:
    """Convert Python's non-standard NaN values to JSON null for Rust input."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _finite_audit_json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_finite_audit_json_value(item) for item in value]
    return value


@lru_cache(maxsize=2)
def _rust_audit_summary_executable(
    binary_name: str, configured_executable: str | None
) -> Path:
    """Build or locate one Rust audit reducer, honoring a test/deployment override."""
    executable_name = binary_name
    if os.name == "nt":
        executable_name += ".exe"
    if configured_executable:
        executable = Path(configured_executable).expanduser().resolve()
        if not executable.is_file() or (
            os.name != "nt" and not os.access(executable, os.X_OK)
        ):
            raise FileNotFoundError(
                f"configured {binary_name} executable is missing or not executable: "
                f"{executable}"
            )
        return executable

    configured_target_dir = os.environ.get("CARGO_TARGET_DIR")
    target_directory = (
        Path(configured_target_dir).expanduser()
        if configured_target_dir
        else ROOT / "perception" / "target"
    )
    if not target_directory.is_absolute():
        target_directory = ROOT / target_directory
    subprocess.run(
        [
            "cargo",
            "build",
            "--locked",
            "--manifest-path",
            str(ROOT / "perception" / "Cargo.toml"),
            "--bin",
            binary_name,
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    executable = target_directory / "debug" / executable_name
    if not executable.is_file() or (
        os.name != "nt" and not os.access(executable, os.X_OK)
    ):
        raise FileNotFoundError(
            f"Rust {binary_name} build completed without the expected executable: "
            f"{executable}"
        )
    return executable.resolve()


def _rust_server_summary_executable() -> Path:
    """Build and locate the Rust retained server-event reducer."""
    return _rust_audit_summary_executable(
        "billiards-performance-audit-server-summary",
        os.environ.get("PERFORMANCE_AUDIT_SERVER_SUMMARY_EXECUTABLE"),
    )


def _server_measurement_summary(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Delegate retained-event reduction to the bounded Rust implementation."""
    maximum_events = 200_000
    maximum_input_bytes = 134_217_728
    if len(events) > maximum_events:
        raise ValueError(
            f"server audit input has {len(events)} events; Rust limit is {maximum_events}"
        )

    payload = bytearray()
    for index, event in enumerate(events, start=1):
        if not isinstance(event, dict):
            raise ValueError(f"server audit event {index} must be an object")
        line = (
            json.dumps(
                _finite_audit_json_value(event),
                allow_nan=False,
                separators=(",", ":"),
            ).encode("utf-8")
            + b"\n"
        )
        payload.extend(line)
        if len(payload) > maximum_input_bytes:
            raise ValueError(
                "server audit event input exceeds the Rust summarizer's "
                f"{maximum_input_bytes}-byte limit"
            )

    result = subprocess.run(
        [str(_rust_server_summary_executable())],
        input=bytes(payload),
        capture_output=True,
        timeout=120,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(
            "Rust server audit summarizer failed"
            + (f": {detail}" if detail else f" with exit status {result.returncode}")
        )
    try:
        summary = json.loads(result.stdout)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(
            "Rust server audit summarizer returned invalid JSON"
        ) from error
    if not isinstance(summary, dict):
        raise RuntimeError("Rust server audit summarizer returned a non-object summary")
    return summary


def _event_export_status_by_process(resource_summary: Any) -> dict[str, Any]:
    """Extract per-process exporter status from the nested resource summary."""
    if not isinstance(resource_summary, dict):
        return {}
    by_process = resource_summary.get("by_process")
    if not isinstance(by_process, dict):
        return {}
    return {
        process: resource["event_export_status"]
        for process, resource in by_process.items()
        if isinstance(resource, dict)
        and resource.get("event_export_status") is not None
    }


def _merge_final_export_status(
    resources: dict[str, Any], final_status: dict[str, Any] | None
) -> None:
    """Replace sampled exporter snapshots with the final status when available."""
    if not isinstance(final_status, dict):
        return
    process_id = final_status.get("process_id")
    host_id = final_status.get("host_id")
    if process_id is None or not isinstance(host_id, str) or not host_id:
        return
    by_process = resources.setdefault("by_process", {})
    if not isinstance(by_process, dict):
        by_process = {}
        resources["by_process"] = by_process
    process_key = f"{host_id}:{process_id}"
    process_summary = by_process.setdefault(process_key, {})
    if not isinstance(process_summary, dict):
        process_summary = {}
        by_process[process_key] = process_summary
    process_summary["event_export_status"] = final_status


def _compare_overhead(
    baseline_path: Path | None,
    repetitions: list[dict[str, Any]],
    metadata: dict[str, Any],
    budget_percent: float,
    *,
    minimum_repetitions: int | None = None,
    minimum_delivery_samples: int | None = None,
) -> dict[str, Any] | None:
    """Load the baseline in Python and delegate comparison decisions to Rust."""
    if baseline_path is None:
        return None
    try:
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read baseline summary {baseline_path}: {error}")
    if not isinstance(baseline, dict):
        raise ValueError(f"baseline summary {baseline_path} must contain an object")
    baseline_metadata = baseline.get("metadata", {})
    baseline_repetitions = baseline.get("repetitions", [])
    if not isinstance(baseline_metadata, dict):
        raise ValueError(f"baseline summary {baseline_path} metadata must be an object")
    if not isinstance(baseline_repetitions, list) or not all(
        isinstance(run, dict) for run in baseline_repetitions
    ):
        raise ValueError(
            f"baseline summary {baseline_path} repetitions must be objects"
        )
    if minimum_repetitions is None:
        minimum_repetitions = int(
            config.get("performance_audit.minimum_overhead_repetitions", 3)
        )
    if minimum_delivery_samples is None:
        minimum_delivery_samples = int(
            config.get("performance_audit.minimum_overhead_delivery_samples", 100)
        )

    comparison = {
        "baseline_summary": str(baseline_path),
        "current_metadata": metadata,
        "current_repetitions": repetitions,
        "baseline_metadata": baseline_metadata,
        "baseline_repetitions": baseline_repetitions,
        "budget_percent": budget_percent,
        "minimum_repetitions": minimum_repetitions,
        "minimum_delivery_samples": minimum_delivery_samples,
    }
    payload = json.dumps(
        _finite_audit_json_value(comparison),
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")
    maximum_input_bytes = 134_217_728
    if len(payload) > maximum_input_bytes:
        raise ValueError(
            "overhead comparison input exceeds the Rust summarizer's "
            f"{maximum_input_bytes}-byte limit"
        )

    result = subprocess.run(
        [
            str(
                _rust_audit_summary_executable(
                    "billiards-performance-audit-overhead-summary",
                    os.environ.get("PERFORMANCE_AUDIT_OVERHEAD_SUMMARY_EXECUTABLE"),
                )
            )
        ],
        input=payload,
        capture_output=True,
        timeout=120,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(
            "Rust audit overhead summarizer failed"
            + (f": {detail}" if detail else f" with exit status {result.returncode}")
        )
    try:
        summary = json.loads(result.stdout)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(
            "Rust audit overhead summarizer returned invalid JSON"
        ) from error
    if not isinstance(summary, dict):
        raise RuntimeError(
            "Rust audit overhead summarizer returned a non-object summary"
        )
    return summary


class VideoLoad:
    """Optional bounded-time MJPEG consumer used only in viewer scenarios."""

    def __init__(self, url: str) -> None:
        self.url = url
        self.stop_event = threading.Event()
        self.bytes_received = 0
        self.chunks_received = 0
        self.frames_received = 0
        self.error: str | None = None
        self._boundary_tail = b""
        self._counter_lock = threading.Lock()
        self.thread = threading.Thread(target=self._consume, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=2.0)

    def snapshot(self) -> dict[str, int]:
        """Return a consistent counter snapshot for warm-up subtraction."""
        with self._counter_lock:
            return {
                "bytes_received": self.bytes_received,
                "chunks_received": self.chunks_received,
                "frames_received": self.frames_received,
            }

    def _consume(self) -> None:
        try:
            with urlopen(self.url, timeout=2.0) as response:
                while not self.stop_event.is_set():
                    chunk = response.read(65536)
                    if not chunk:
                        break
                    combined = self._boundary_tail + chunk
                    with self._counter_lock:
                        self.bytes_received += len(chunk)
                        self.chunks_received += 1
                        self.frames_received += combined.count(b"--frame")
                    self._boundary_tail = combined[-len(b"--frame") + 1 :]
        except Exception as error:
            self.error = f"{type(error).__name__}: {error}"


def _event_trace_export_loss_count(status: dict[str, Any]) -> int:
    """Count only lost trace records; deliberate sampling is not loss."""
    return sum(
        value
        for key in ("dropped_events", "export_limit_dropped_events", "export_errors")
        if isinstance((value := status.get(key)), int)
        and not isinstance(value, bool)
        and value > 0
    )


def _trajectory_summary_parity_mismatches(
    reference: dict[str, Any], native: dict[str, Any]
) -> list[str]:
    """Compare the retained Python oracle's stable fields with the Rust result."""
    mismatches: list[str] = []
    metric_fields = {"count", "sample_limit", "p50", "p95", "p99"}
    percentile_fields = {
        "trajectory_source_publication_wall_age_estimate_ms",
        "trajectory_source_publication_to_client_receipt_ms",
    }
    for key, expected in reference.items():
        actual = native.get(key)
        if isinstance(expected, dict):
            actual_projection = (
                {field: actual.get(field) for field in expected}
                if isinstance(actual, dict)
                else None
            )
            if actual_projection != expected:
                # Rust intentionally adds all-sample counts and truncation
                # disclosure to percentile records. Those additive keys are
                # not part of the reference's stable parity contract.
                if key in percentile_fields and isinstance(actual, dict):
                    if {field: actual.get(field) for field in metric_fields} == {
                        field: expected.get(field) for field in metric_fields
                    }:
                        continue
                mismatches.append(key)
        elif actual != expected:
            mismatches.append(key)
    return mismatches


async def _receive_until_type(
    websocket: Any, message_type: str, timeout_seconds: float = 10.0
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        remaining = max(0.001, deadline - time.monotonic())
        raw = await asyncio.wait_for(websocket.recv(), timeout=remaining)
        message = json.loads(raw)
        if isinstance(message, dict) and message.get("type") == message_type:
            return message
    raise TimeoutError(f"timed out waiting for WebSocket {message_type!r} message")


async def collect_repetition(
    url: str,
    *,
    streams: list[str],
    warmup_seconds: float,
    measured_seconds: float,
    repetition: int,
    recorder: AsyncAuditRecorder,
    video_url: str | None = None,
    progress: bool = True,
    history_size: int = 10000,
    subscriber_delay_ms: float = 0.0,
    heartbeat_interval_seconds: float = 20.0,
    trajectory_queue_capacity: int = 2048,
    trajectory_maximum_records: int = 200_002,
) -> dict[str, Any]:
    """Connect, warm up, and collect one state stream measurement window."""
    accumulator = StateAccumulator(maximum_samples=history_size)
    trajectory_bridge = (
        RustTrajectorySummaryBridge(
            maximum_samples=history_size,
            maximum_records=trajectory_maximum_records,
            queue_capacity=trajectory_queue_capacity,
            write_timeout_seconds=float(
                config.get("performance_audit.export_drain_timeout_seconds", 5.0)
            ),
        )
        if "trajectory" in streams
        else None
    )
    trajectory_accumulator = TrajectoryAccumulator(
        maximum_samples=(
            trajectory_bridge.maximum_samples if trajectory_bridge else history_size
        )
    )
    native_trajectory_summary: dict[str, Any] | None = None
    trajectory_bridge_metadata: dict[str, Any] | None = None
    trajectory_exporter_loss_count = 0
    video_load = VideoLoad(video_url) if video_url else None
    video_counts_after_measurement: dict[str, int] = {}
    application_pings_answered = 0
    client_application_heartbeats_sent = 0
    performance_url = _http_status_url(url, "/vision/performance")
    audit_url = _http_status_url(url, "/vision/performance/audit")
    server_start = await asyncio.to_thread(_read_json_url, performance_url)
    audit_status = await asyncio.to_thread(_read_json_url, audit_url)
    collector_host_id = socket.gethostname()
    server_host_id = audit_status.get("host_id") if audit_status else None
    collector_clock_domain_id = recorder.clock_domain_id
    server_clock_domain_id = (
        audit_status.get("clock_domain_id") if audit_status else None
    )
    clock_comparable = bool(
        collector_clock_domain_id
        and server_clock_domain_id
        and collector_clock_domain_id == server_clock_domain_id
    )
    if trajectory_bridge is not None:
        await trajectory_bridge.prepare()
    async with websockets.connect(url, max_size=None, ping_interval=20) as websocket:
        connection_message = await _receive_until_type(websocket, "connection")
        connection_data = connection_message.get("data")
        client_id = (
            connection_data.get("client_id")
            if isinstance(connection_data, dict)
            else None
        )
        await websocket.send(
            json.dumps({"type": "subscribe", "data": {"streams": streams}})
        )
        await _receive_until_type(websocket, "subscribed")
        if trajectory_bridge is not None:
            await trajectory_bridge.start()
        if video_load is not None:
            video_load.start()
        try:
            warmup_until = time.monotonic() + warmup_seconds
            next_heartbeat = time.monotonic() + heartbeat_interval_seconds
            while time.monotonic() < warmup_until:
                try:
                    raw = await asyncio.wait_for(
                        websocket.recv(),
                        timeout=min(
                            1.0,
                            warmup_until - time.monotonic(),
                            max(0.001, next_heartbeat - time.monotonic()),
                        ),
                    )
                    try:
                        message = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(message, dict):
                        application_pings_answered += await _respond_to_server_ping(
                            websocket, message
                        )
                except asyncio.TimeoutError:
                    pass
                now = time.monotonic()
                if now >= next_heartbeat:
                    await _send_client_heartbeat(websocket)
                    client_application_heartbeats_sent += 1
                    next_heartbeat = now + heartbeat_interval_seconds
        except BaseException:
            if video_load is not None:
                video_load.stop()
            if trajectory_bridge is not None:
                await asyncio.shield(trajectory_bridge.close())
            raise

        video_counts_before_measurement = (
            video_load.snapshot() if video_load is not None else None
        )
        trace_export_losses_before = _event_trace_export_loss_count(recorder.status())
        started_ns = time.perf_counter_ns()
        started_wall = time.time()
        deadline = time.monotonic() + measured_seconds
        next_heartbeat = time.monotonic() + heartbeat_interval_seconds
        last_progress = 0.0
        try:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                receive_timeout = min(
                    remaining,
                    max(0.001, next_heartbeat - time.monotonic()),
                )
                try:
                    raw = await asyncio.wait_for(
                        websocket.recv(), timeout=receive_timeout
                    )
                except asyncio.TimeoutError:
                    now = time.monotonic()
                    if now >= deadline:
                        break
                    if now >= next_heartbeat:
                        await _send_client_heartbeat(websocket)
                        client_application_heartbeats_sent += 1
                        next_heartbeat = now + heartbeat_interval_seconds
                    continue
                now = time.monotonic()
                if now >= next_heartbeat:
                    await _send_client_heartbeat(websocket)
                    client_application_heartbeats_sent += 1
                    next_heartbeat = now + heartbeat_interval_seconds
                received_ns = time.perf_counter_ns()
                received_wall = time.time()
                try:
                    message = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if not isinstance(message, dict):
                    continue
                if await _respond_to_server_ping(websocket, message):
                    application_pings_answered += 1
                    continue
                if message.get("type") == "state":
                    observed = observation_from_message(message)
                    event = {
                        "event_type": "state_received",
                        "repetition": repetition,
                        **observed,
                        "receive_elapsed_ns": received_ns - started_ns,
                        "receive_monotonic_ns": received_ns,
                        "client_id": client_id,
                        "source_publication_wall_age_estimate_ms": (
                            observation_age_ms(
                                observed.get("source_timestamp"), received_wall
                            )
                        ),
                        "source_publication_to_client_receipt_ms": (
                            (received_ns - observed["source_publication_monotonic_ns"])
                            / 1e6
                            if clock_comparable
                            and isinstance(
                                observed.get("source_publication_monotonic_ns"), int
                            )
                            and observed["source_publication_monotonic_ns"] > 0
                            and received_ns
                            >= observed["source_publication_monotonic_ns"]
                            else None
                        ),
                        "clock_comparable": clock_comparable,
                    }
                    accumulator.add(event)
                elif message.get("type") == "trajectory":
                    observed = trajectory_from_message(message)
                    event = {
                        "event_type": "trajectory_received",
                        "repetition": repetition,
                        **observed,
                        "receive_elapsed_ns": received_ns - started_ns,
                        "receive_monotonic_ns": received_ns,
                        "client_id": client_id,
                        "source_publication_wall_age_estimate_ms": (
                            observation_age_ms(
                                observed.get("source_timestamp"), received_wall
                            )
                            if clock_comparable
                            else None
                        ),
                        "source_publication_to_client_receipt_ms": (
                            (received_ns - observed["source_publication_monotonic_ns"])
                            / 1e6
                            if clock_comparable
                            and isinstance(
                                observed.get("source_publication_monotonic_ns"), int
                            )
                            and observed["source_publication_monotonic_ns"] > 0
                            and received_ns
                            >= observed["source_publication_monotonic_ns"]
                            else None
                        ),
                        "clock_comparable": clock_comparable,
                    }
                    trajectory_accumulator.add(event)
                    if trajectory_bridge is not None:
                        trajectory_bridge.try_record(event)
                else:
                    continue
                recorder.record(event)
                if subscriber_delay_ms > 0:
                    await asyncio.sleep(subscriber_delay_ms / 1000)
                now = time.monotonic()
                if progress and now - last_progress >= 1.0:
                    print(
                        f"\rRun {repetition}: {accumulator.message_count} states, "
                        f"{accumulator.unique_observations} unique observations, "
                        f"{trajectory_accumulator.messages} trajectories",
                        end="",
                        flush=True,
                    )
                    last_progress = now
        finally:
            measured_finished_ns = time.perf_counter_ns()
            if video_load is not None:
                video_load.stop()
                video_counts_after_measurement = video_load.snapshot()
            if trajectory_bridge is not None:
                trace_export_losses_after = _event_trace_export_loss_count(
                    recorder.status()
                )
                trajectory_exporter_loss_count = max(
                    0, trace_export_losses_after - trace_export_losses_before
                )
                elapsed_for_summary = max(
                    0.0, (measured_finished_ns - started_ns) / 1_000_000_000
                )
                try:
                    native_trajectory_summary = await asyncio.shield(
                        trajectory_bridge.finish(
                            elapsed_for_summary,
                            exporter_loss_count=trajectory_exporter_loss_count,
                        )
                    )
                finally:
                    trajectory_bridge_metadata = trajectory_bridge.failure_metadata()
                    await asyncio.shield(trajectory_bridge.close())

    elapsed = max(0.0, (measured_finished_ns - started_ns) / 1_000_000_000)
    server_end = await asyncio.to_thread(_read_json_url, performance_url)
    audit_end = await asyncio.to_thread(_wait_for_audit_export_drain, audit_url)
    summary = accumulator.summary(elapsed)
    python_trajectory_summary = trajectory_accumulator.summary(elapsed)
    if native_trajectory_summary is not None:
        parity_mismatches = _trajectory_summary_parity_mismatches(
            python_trajectory_summary, native_trajectory_summary
        )
        summary.update(
            python_trajectory_summary
            if parity_mismatches
            else native_trajectory_summary
        )
        summary["trajectory_summary_engine"] = (
            "python_reference_fallback" if parity_mismatches else "rust"
        )
        summary["trajectory_summary_differential"] = {
            "status": "mismatch" if parity_mismatches else "matched",
            "mismatched_fields": parity_mismatches,
        }
        if parity_mismatches and trajectory_bridge_metadata is not None:
            trajectory_bridge_metadata["status"] = "partial"
            trajectory_bridge_metadata["differential_mismatch_fields"] = (
                parity_mismatches
            )
    else:
        summary.update(python_trajectory_summary)
        if trajectory_bridge is not None:
            summary["trajectory_summary_engine"] = "python_reference_fallback"
            summary["trajectory_summary_differential"] = {
                "status": "unavailable",
                "mismatched_fields": [],
            }
    if trajectory_bridge is not None:
        if native_trajectory_summary is None:
            summary["trajectory_exporter_dropped_records"] = (
                trajectory_exporter_loss_count
            )
            bridge_losses = trajectory_bridge_metadata or {}
            summary["trajectory_bridge_dropped_records"] = int(
                bridge_losses.get("callback_rejected_records", 0)
            ) + int(bridge_losses.get("writer_dropped_records", 0))
        summary["trajectory_summary_bridge"] = trajectory_bridge_metadata
    summary.update(
        {
            "repetition": repetition,
            "streams": streams,
            "warmup_seconds": warmup_seconds,
            "requested_measured_seconds": measured_seconds,
            "subscriber_delay_ms": subscriber_delay_ms,
            "server_application_pings_answered": application_pings_answered,
            "client_application_heartbeats_sent": client_application_heartbeats_sent,
            "measurement_started_monotonic_ns": started_ns,
            "measurement_ended_monotonic_ns": measured_finished_ns,
            "start_wall_time": datetime.fromtimestamp(started_wall)
            .astimezone()
            .isoformat(),
            "video_viewer": (
                {
                    "url": video_url,
                    "bytes_received": video_counts_after_measurement.get(
                        "bytes_received", 0
                    )
                    - (video_counts_before_measurement or {}).get("bytes_received", 0),
                    "chunks_received": video_counts_after_measurement.get(
                        "chunks_received", 0
                    )
                    - (video_counts_before_measurement or {}).get("chunks_received", 0),
                    "frames_received": video_counts_after_measurement.get(
                        "frames_received", 0
                    )
                    - (video_counts_before_measurement or {}).get("frames_received", 0),
                    "error": video_load.error,
                }
                if video_load is not None
                else None
            ),
            "server_audit_domains": (
                (audit_end or audit_status or {}).get("audit_domains")
            ),
            "server_host_id": server_host_id,
            "collector_host_id": collector_host_id,
            "server_clock_domain_id": server_clock_domain_id,
            "collector_clock_domain_id": collector_clock_domain_id,
            "monotonic_clocks_comparable": clock_comparable,
            "server_audit_run_id": (
                ((audit_end or audit_status or {}).get("event_export") or {}).get(
                    "run_id"
                )
            ),
            "server_event_export": (audit_end or audit_status or {}).get(
                "event_export"
            ),
            "server_profiler_frame_count_before": (
                server_start.get("total_frames") if server_start else None
            ),
            "server_profiler_frame_count_after": (
                server_end.get("total_frames") if server_end else None
            ),
            "server_profiler_sampling_after": (
                server_end.get("profile_sampling") if server_end else None
            ),
            "server_profiler_measurement": (
                "post-window process totals, including warm-up; unavailable when state audit is disabled"
            ),
        }
    )
    return summary


def _markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Real-time state performance audit",
        "",
        f"Run: `{report['audit_run_id']}`",
        f"Scenario: `{report['scenario']}`",
        f"WebSocket: `{report['websocket_url']}`",
        "",
        "| Repetition | Streams | State messages/s | Unique observations/s | Source publication-to-receipt p95 (ms) | Source gaps / restarts / out-of-order |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for run in report["repetitions"]:
        lines.append(
            "| {repetition} | {streams} | {messages:.2f} | {observations:.2f} | {age} | {source_order} |".format(
                repetition=run["repetition"],
                streams=", ".join(run["streams"]),
                messages=run["state_messages_per_second"],
                observations=run["unique_observations_per_second"],
                age=(
                    run["source_publication_to_client_receipt_ms"]["p95"]
                    if run["source_publication_to_client_receipt_ms"]["p95"] is not None
                    else "unavailable"
                ),
                source_order=(
                    f"{run.get('source_sequence_gaps', 0)} / "
                    f"{run.get('source_sequence_restarts', 0)} / "
                    f"{run.get('source_out_of_order_messages', 0)}"
                ),
            )
        )
    server_measurements = report.get("server_measurements")
    vision_measurements = (
        server_measurements.get("vision")
        if isinstance(server_measurements, dict)
        else None
    )
    ranked_stages = (
        vision_measurements.get("exclusive_stage_work_ranked", [])
        if isinstance(vision_measurements, dict)
        else []
    )
    lines.extend(["", "## Measured vision-stage work", ""])
    if ranked_stages:
        lines.extend(
            [
                "Stages are ranked by cumulative exclusive time in the retained frame records. This is workload attribution, not a claim that removing a stage yields the same end-to-end speedup.",
                "",
                "| Rank | Stage | Count | Exclusive mean (ms) | Exclusive p95 (ms) | Share of measured exclusive stage time |",
                "|---:|---|---:|---:|---:|---:|",
            ]
        )
        for rank, stage in enumerate(ranked_stages, start=1):
            exclusive = stage.get("exclusive", {})
            lines.append(
                "| {rank} | {name} | {count} | {mean} | {p95} | {share} |".format(
                    rank=rank,
                    name=stage.get("stage", "unknown"),
                    count=exclusive.get("count", 0),
                    mean=(
                        f"{exclusive['mean'] * 1000:.3f}"
                        if isinstance(exclusive.get("mean"), (int, float))
                        else "unavailable"
                    ),
                    p95=(
                        f"{exclusive['p95'] * 1000:.3f}"
                        if isinstance(exclusive.get("p95"), (int, float))
                        else "unavailable"
                    ),
                    share=(
                        f"{stage['share_of_observed_exclusive_stage_seconds']:.1%}"
                        if isinstance(
                            stage.get("share_of_observed_exclusive_stage_seconds"),
                            (int, float),
                        )
                        else "unavailable"
                    ),
                )
            )
    else:
        lines.append(
            "Vision-stage timing is unavailable in the retained server trace. The run cannot rank detector, preprocessing, tracking, or state-assembly work."
        )
    capture_transport = report.get("capture_transport_benchmark")
    if isinstance(capture_transport, dict):
        lines.extend(
            [
                "",
                "## Capture transport microbenchmark",
                "",
                "This isolated deterministic RGB8 experiment compares in-process processing with a supervised stdout pipe.",
                "It does not include a camera, detector, capture mailbox, or downstream client.",
                "Camera source-to-parent latency, process CPU time, and resident memory were not measured.",
                "The in-process producer uses a capacity-one channel; the supervised helper uses the operating-system stdout pipe. The operating-system pipe capacity is not queried.",
                "Logical payload MiB/s is configured frame volume divided by elapsed time. The in-process arm moves Vec ownership and does not copy frame bytes, so its value is not memory bandwidth.",
                f"Producer/consumer model: `{capture_transport['producer_consumer_model']}`.",
                f"In-process data movement: `{capture_transport['repetitions'][0]['in_process']['data_movement_model']}`.",
                f"Pipe data movement: `{capture_transport['repetitions'][0]['supervised_stdout_pipe']['data_movement_model']}`.",
                f"Rust build profile: `{capture_transport['build_profile']}`.",
            ]
        )
        if capture_transport["build_profile"] == "debug":
            lines.append(
                "**Diagnostic only:** this binary was built with the debug profile; do not use its throughput or latency values to rank production transports."
            )
        lines.extend(
            [
                "",
                "| Repetition | In-process frames/s | In-process logical MiB/s | In-process receipt p95 (ms) | Supervised pipe frames/s | Pipe logical MiB/s | Pipe receipt p95 (ms) |",
                "|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for repetition in capture_transport["repetitions"]:
            in_process = repetition["in_process"]
            pipe = repetition["supervised_stdout_pipe"]
            lines.append(
                "| {repetition} | {in_fps:.2f} | {in_mib:,.2f} | {in_p95} | {pipe_fps:.2f} | {pipe_mib:,.2f} | {pipe_p95} |".format(
                    repetition=repetition["repetition"],
                    in_fps=in_process["frames_per_second"],
                    in_mib=in_process["logical_payload_mib_per_second"],
                    in_p95=(
                        f"{in_process['parent_frame_completion_p95_ns'] / 1e6:.3f}"
                        if in_process["parent_frame_completion_p95_ns"] is not None
                        else "unavailable"
                    ),
                    pipe_fps=pipe["frames_per_second"],
                    pipe_mib=pipe["logical_payload_mib_per_second"],
                    pipe_p95=(
                        f"{pipe['parent_frame_completion_p95_ns'] / 1e6:.3f}"
                        if pipe["parent_frame_completion_p95_ns"] is not None
                        else "unavailable"
                    ),
                )
            )
        lines.extend(
            [
                "",
                "Receipt p95 is sampled from parent frame receipt and identity validation; it is not camera source-to-parent latency. The same setup-generated template is repeated, and its SHA-256 is checked outside measured loops; individual timed payloads are not hashed.",
                "",
            ]
        )
    validity = report.get("audit_validity")
    if isinstance(validity, dict) and validity.get("status") == "invalid":
        reason = validity.get("reason") or "This run failed its validity checks."
        lines.extend(["", f"**INVALID RUN:** {reason}", ""])
    exporter_statuses = report.get("server_event_export_status_by_process", {})
    exporter_status_summary = (
        "; ".join(
            f"{process}: dropped={status.get('dropped_events')}, "
            f"sampled={status.get('sampled_out_events')}, "
            f"export-limit-dropped={status.get('export_limit_dropped_events')}, "
            f"errors={status.get('export_errors')}"
            for process, status in exporter_statuses.items()
            if isinstance(status, dict)
        )
        or "unavailable"
    )
    detailed_event_sampling_summary = []
    for process, status in exporter_statuses.items():
        if not isinstance(status, dict):
            continue
        categories = status.get("detailed_sampling")
        if not isinstance(categories, dict):
            continue
        for category, counts in categories.items():
            if not isinstance(counts, dict):
                continue
            detailed_event_sampling_summary.append(
                "{process} {category}: selected={selected}/{candidates}, "
                "sampled-out={sampled_out}".format(
                    process=process,
                    category=category,
                    selected=counts.get("selected", "unknown"),
                    candidates=counts.get("candidates", "unknown"),
                    sampled_out=counts.get("sampled_out", "unknown"),
                )
            )
    detailed_event_sampling_summary_text = (
        "; ".join(detailed_event_sampling_summary) or "unavailable"
    )
    profiler_sampling_summary = []
    for repetition in report.get("repetitions", []):
        sampling = repetition.get("server_profiler_sampling_after")
        if not isinstance(sampling, dict):
            continue
        profiler_sampling_summary.append(
            "run {run}: configured detail rate={rate}, process frames={frames}, "
            "detailed={detailed}, sampled-out={sampled_out}, failed={failed}, "
            "retained in history={retained}/{capacity}".format(
                run=repetition.get("repetition", "?"),
                rate=sampling.get("configured_detailed_sample_rate", "unknown"),
                frames=sampling.get("processed_frames", "unknown"),
                detailed=sampling.get("detailed_frames", "unknown"),
                sampled_out=sampling.get("sampled_out_frames", "unknown"),
                failed=sampling.get("failed_frames", "unknown"),
                retained=sampling.get("retained_detailed_frames", "unknown"),
                capacity=sampling.get("history_capacity", "unknown"),
            )
        )
    profiler_sampling_summary_text = (
        "; ".join(profiler_sampling_summary)
        if profiler_sampling_summary
        else "unavailable"
    )
    lines.extend(
        [
            "",
            "Accuracy against labels: unavailable for this live receiver run. Use the recorded-video benchmark and annotated still set for accuracy evidence.",
            "Source timestamps are labeled by the application and indicate shared-memory publication time where available; they do not represent camera exposure time.",
            "Same-host publication-to-receipt timing requires retained server events and a shared perf-counter clock domain.",
            f"State publication-to-receipt p95 (ms): {report['delivery_latency']['state_publication_to_receipt_ms']['p95'] if report['delivery_latency']['state_publication_to_receipt_ms']['p95'] is not None else 'unavailable'}.",
            f"Source publication-to-state p95 (ms): {report['delivery_latency']['source_publication_to_state_ms']['p95'] if report['delivery_latency']['source_publication_to_state_ms']['p95'] is not None else 'unavailable'}.",
            f"Deadline misses above {report['delivery_latency']['deadline_budget_ms']:.1f} ms: {report['delivery_latency']['deadline_misses']}.",
            f"Client trace losses: dropped={report['event_export'].get('dropped_events')}, sampled={report['event_export'].get('sampled_out_events')}, export-limit-dropped={report['event_export'].get('export_limit_dropped_events')}, errors={report['event_export'].get('export_errors')}.",
            "Vision profiler detail sampling (post-window process totals, includes warm-up): "
            + profiler_sampling_summary_text
            + ".",
            "Detailed event sampling (process totals from final or last resource snapshot, includes warm-up): "
            + detailed_event_sampling_summary_text
            + ".",
            f"Final server/video exporter status by process: {exporter_status_summary}.",
            f"Instrumentation overhead comparison: {report['measurement_overhead_comparison']['status'] if report['measurement_overhead_comparison'] else 'not requested'}.",
            "",
            "Next experiments (not production recommendations): profile allocations and copies inside the highest-ranked measured stages; compare image conversion and detector preprocessing paths at the same accuracy; measure coordinate conversion, JSON serialization, integration lock waits, and executor waits; quantify repeated work on unchanged observations; compare polling with observation-triggered scheduling; compare shared workers, dedicated workers, and process isolation; compare in-process handoff with bounded loopback TCP using identical payloads; then repeat state-only versus trajectory and video workloads. For every candidate, report expected benefit as a measured bound, accuracy effects, implementation complexity, and a validation experiment.",
            "Accuracy gate: no production candidate is recommendable from this run alone. Checked-in still labels cover ball precision/recall and position error; temporal identity continuity, cue quality and labeled freshness need temporal annotations.",
            "",
        ]
    )
    return "\n".join(lines)


def _capture_transport_settings(args: argparse.Namespace) -> dict[str, Any] | None:
    """Return explicit benchmark settings, or None when the hook is disabled."""
    option_names = (
        "capture_transport_width_px",
        "capture_transport_height_px",
        "capture_transport_warmup_frames",
        "capture_transport_measured_frames",
        "capture_transport_repetitions",
        "capture_transport_sampling_limit",
        "capture_transport_seed",
        "capture_transport_max_payload_bytes",
        "capture_transport_max_total_frames",
        "capture_transport_max_total_samples",
        "capture_transport_timeout_seconds",
    )
    executable = getattr(args, "capture_transport_benchmark_executable", None)
    supplied_settings = [
        name for name in option_names if getattr(args, name, None) is not None
    ]
    if executable is None:
        if supplied_settings:
            raise ValueError(
                "capture transport settings require "
                "--capture-transport-benchmark-executable"
            )
        return None

    missing = [name for name in option_names if getattr(args, name, None) is None]
    if missing:
        flags = ", ".join("--" + name.replace("_", "-") for name in missing)
        raise ValueError(
            "the capture transport benchmark requires explicit settings: " + flags
        )

    settings = {
        "executable": Path(executable),
        "width_px": args.capture_transport_width_px,
        "height_px": args.capture_transport_height_px,
        "warmup_frames": args.capture_transport_warmup_frames,
        "measured_frames": args.capture_transport_measured_frames,
        "repetitions": args.capture_transport_repetitions,
        "sampling_limit": args.capture_transport_sampling_limit,
        "seed": args.capture_transport_seed,
        "max_payload_bytes": args.capture_transport_max_payload_bytes,
        "max_total_frames": args.capture_transport_max_total_frames,
        "max_total_samples": args.capture_transport_max_total_samples,
        "timeout_seconds": args.capture_transport_timeout_seconds,
    }
    if settings["width_px"] < 1 or settings["height_px"] < 1:
        raise ValueError("capture transport dimensions must be positive")
    if settings["warmup_frames"] < 0 or settings["measured_frames"] < 1:
        raise ValueError(
            "capture transport warmup must be non-negative and measured frames positive"
        )
    for name in (
        "repetitions",
        "sampling_limit",
        "max_payload_bytes",
        "max_total_frames",
        "max_total_samples",
    ):
        if settings[name] < 1:
            raise ValueError(
                f"capture transport {name.replace('_', ' ')} must be positive"
            )
    if not 0 <= settings["seed"] < 2**64:
        raise ValueError("capture transport seed must be an unsigned 64-bit integer")
    if (
        not math.isfinite(settings["timeout_seconds"])
        or settings["timeout_seconds"] <= 0
    ):
        raise ValueError("capture transport timeout must be positive")
    return settings


def _validated_capture_transport_summary(
    document: Any, *, expected_settings: dict[str, Any]
) -> dict[str, Any]:
    """Validate benchmark schema and summarize only fields used in the report."""
    if not isinstance(document, dict) or document.get("schema_version") != 3:
        raise ValueError("capture transport benchmark must emit JSON schema_version 3")
    metadata = document.get("metadata")
    if not isinstance(metadata, dict) or metadata.get("build_profile") not in {
        "debug",
        "release",
    }:
        raise ValueError(
            "capture transport benchmark has invalid build profile metadata"
        )
    raw_repetitions = document.get("repetitions")
    if (
        not isinstance(raw_repetitions, list)
        or len(raw_repetitions) != expected_settings["repetitions"]
    ):
        raise ValueError(
            "capture transport benchmark repetition count does not match the requested workload"
        )
    workload = document.get("workload")
    if not isinstance(workload, dict):
        raise ValueError("capture transport benchmark is missing workload metadata")
    producer_consumer_model = workload.get("producer_consumer_model")
    if not (
        isinstance(producer_consumer_model, str)
        and "in_process_producer_thread" in producer_consumer_model
        and "helper_process_writer" in producer_consumer_model
    ):
        raise ValueError(
            "capture transport benchmark does not report the paired producer/consumer model"
        )
    expected_workload = {
        "pixel_format": "RGB8",
        "width_px": expected_settings["width_px"],
        "height_px": expected_settings["height_px"],
        "warmup_frames_per_repetition": expected_settings["warmup_frames"],
        "measured_frames_per_repetition": expected_settings["measured_frames"],
        "repetitions": expected_settings["repetitions"],
        "seed": expected_settings["seed"],
        "max_payload_bytes": expected_settings["max_payload_bytes"],
        "max_total_frames_per_arm": expected_settings["max_total_frames"],
        "max_total_samples": expected_settings["max_total_samples"],
        "sampling_limit_per_arm_per_repetition": expected_settings["sampling_limit"],
        "payload_bytes_per_frame": (
            3 * expected_settings["width_px"] * expected_settings["height_px"]
        ),
        "application_frame_buffer_count_per_arm": 2,
        "application_frame_buffer_bytes_per_arm": (
            6 * expected_settings["width_px"] * expected_settings["height_px"]
        ),
        "frame_records_per_arm_over_all_repetitions": (
            (expected_settings["warmup_frames"] + expected_settings["measured_frames"])
            * expected_settings["repetitions"]
        ),
        "payload_content_model": "one deterministic RGB8 payload template repeated for every frame record in each arm",
        "payload_preparation_and_sha256": "generated and SHA-256 checked during arm setup/readiness; excluded from measured frame loops",
        "timed_operations": "in-process Vec ownership handoff versus helper frame serialization and parent pipe reads into a reusable buffer; payload generation and SHA-256 are excluded",
    }
    if any(workload.get(key) != value for key, value in expected_workload.items()):
        raise ValueError(
            "capture transport benchmark workload metadata does not match the requested settings"
        )
    unavailable_measurements = document.get("measurements_not_taken")
    if not isinstance(unavailable_measurements, dict) or any(
        not isinstance(unavailable_measurements.get(name), dict)
        or unavailable_measurements[name].get("status") != "unavailable"
        for name in (
            "camera_source_to_parent_latency",
            "detector_accuracy_and_latency",
            "process_cpu_time",
            "resident_memory",
            "operating_system_pipe_capacity",
            "capture_mailbox_and_client_delivery",
        )
    ):
        raise ValueError(
            "capture transport benchmark must explicitly label unsupported measurements"
        )

    validated_repetitions: list[dict[str, Any]] = []
    for index, repetition in enumerate(raw_repetitions, start=1):
        if not isinstance(repetition, dict) or repetition.get("repetition") != index:
            raise ValueError(
                "capture transport benchmark repetitions are malformed or unordered"
            )
        arms: dict[str, dict[str, Any]] = {}
        for arm_name in ("in_process", "supervised_stdout_pipe"):
            arm = repetition.get(arm_name)
            if not isinstance(arm, dict):
                raise ValueError(
                    f"capture transport repetition {index} is missing {arm_name} results"
                )
            if arm.get("measured_frames") != expected_settings["measured_frames"]:
                raise ValueError(
                    f"capture transport repetition {index} {arm_name} frame count mismatch"
                )
            payload_sha256 = arm.get("frame_payload_sha256")
            if not (
                isinstance(payload_sha256, str)
                and len(payload_sha256) == 64
                and all(character in "0123456789abcdef" for character in payload_sha256)
            ):
                raise ValueError(
                    f"capture transport repetition {index} {arm_name} has an invalid payload SHA-256"
                )
            throughput = arm.get("frames_per_second")
            logical_payload_rate = arm.get("logical_payload_mib_per_second")
            data_movement_model = arm.get("data_movement_model")
            expected_movement = (
                "sync_channel" if arm_name == "in_process" else "OS stdout pipe"
            )
            completion = arm.get("sampled_frame_completion")
            if (
                isinstance(throughput, bool)
                or not isinstance(throughput, (int, float))
                or not math.isfinite(throughput)
                or throughput < 0
                or isinstance(logical_payload_rate, bool)
                or not isinstance(logical_payload_rate, (int, float))
                or not math.isfinite(logical_payload_rate)
                or logical_payload_rate < 0
                or not isinstance(data_movement_model, str)
                or expected_movement not in data_movement_model
                or not isinstance(completion, dict)
                or isinstance(completion.get("p95_ns"), bool)
                or not isinstance(completion.get("p95_ns"), int)
                or completion["p95_ns"] < 0
            ):
                raise ValueError(
                    f"capture transport repetition {index} {arm_name} timing fields are malformed"
                )
            sampled_frames = completion.get("sampled_frames")
            if (
                isinstance(sampled_frames, bool)
                or not isinstance(sampled_frames, int)
                or sampled_frames != expected_settings["sampling_limit"]
                or sampled_frames > expected_settings["sampling_limit"]
                or sampled_frames > expected_settings["measured_frames"]
            ):
                raise ValueError(
                    f"capture transport repetition {index} {arm_name} sample count is malformed"
                )
            arms[arm_name] = {
                "frames_per_second": float(throughput),
                "logical_payload_mib_per_second": float(logical_payload_rate),
                "data_movement_model": data_movement_model,
                "parent_frame_completion_p95_ns": completion["p95_ns"],
                "measured_frames": expected_settings["measured_frames"],
                "frame_payload_sha256": payload_sha256,
            }
        if (
            arms["in_process"]["frame_payload_sha256"]
            != arms["supervised_stdout_pipe"]["frame_payload_sha256"]
        ):
            raise ValueError(
                f"capture transport repetition {index} has mismatched in-process and pipe payload SHA-256"
            )
        validated_repetitions.append({"repetition": index, **arms})

    return {
        "artifact": "capture-transport.json",
        "schema_version": 3,
        "experiment": document.get("experiment"),
        "scope": document.get("scope"),
        "build_profile": metadata["build_profile"],
        "workload": workload,
        "producer_consumer_model": producer_consumer_model,
        "repetitions": validated_repetitions,
        "limitations": [
            "No camera, detector, capture mailbox, or downstream client is included.",
            "Camera source-to-parent latency, process CPU time, and resident memory are unavailable.",
            "The operating-system stdout pipe capacity is not queried.",
        ],
    }


def _run_capture_transport_benchmark(
    settings: dict[str, Any], output_path: Path
) -> dict[str, Any]:
    """Run the opt-in Rust microbenchmark and persist its validated report."""
    command = [
        str(settings["executable"]),
        "--width-px",
        str(settings["width_px"]),
        "--height-px",
        str(settings["height_px"]),
        "--warmup-frames",
        str(settings["warmup_frames"]),
        "--measured-frames",
        str(settings["measured_frames"]),
        "--repetitions",
        str(settings["repetitions"]),
        "--sampling-limit",
        str(settings["sampling_limit"]),
        "--seed",
        str(settings["seed"]),
        "--max-payload-bytes",
        str(settings["max_payload_bytes"]),
        "--max-total-frames",
        str(settings["max_total_frames"]),
        "--max-total-samples",
        str(settings["max_total_samples"]),
    ]
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=settings["timeout_seconds"],
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(
            "capture transport benchmark exceeded the configured "
            f"{settings['timeout_seconds']} second timeout"
        ) from error
    except OSError as error:
        raise RuntimeError(
            f"could not start capture transport benchmark {settings['executable']}: {error}"
        ) from error
    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        raise RuntimeError(
            "capture transport benchmark exited with status "
            f"{completed.returncode}: {stderr or 'no error output'}"
        )
    try:
        document = json.loads(completed.stdout)
    except (json.JSONDecodeError, TypeError) as error:
        raise ValueError(
            "capture transport benchmark did not emit valid JSON"
        ) from error
    summary = _validated_capture_transport_summary(
        document,
        expected_settings=settings,
    )
    output_path.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


class _QualificationConfiguration:
    """Read-only configuration overlay used to validate one camera profile."""

    def __init__(self, values: dict[str, Any]) -> None:
        self._values = values

    def get_all(self) -> dict[str, Any]:
        return copy.deepcopy(self._values)


def _merge_configuration_value(
    destination: dict[str, Any], path: tuple[str, ...], value: Any
) -> None:
    current = destination
    for component in path[:-1]:
        nested = current.get(component)
        if not isinstance(nested, dict):
            nested = {}
            current[component] = nested
        current = nested
    current[path[-1]] = value


def _rust_live_profile_arguments(args: argparse.Namespace) -> dict[str, Any] | None:
    """Return validated explicit capture-profile fields for the live mode."""
    field_names = (
        "rust_live_settings",
        "rust_live_device_selector",
        "rust_live_camera_backend",
        "rust_live_camera_width_px",
        "rust_live_camera_height_px",
        "rust_live_camera_fps",
        "rust_live_camera_pixel_format",
        "rust_live_image_space",
        "rust_live_output_width_px",
        "rust_live_output_height_px",
    )
    provided = {name: getattr(args, name, None) for name in field_names}
    enabled = bool(getattr(args, "rust_live_qualification", False))
    if not enabled:
        if any(value is not None for value in provided.values()) or bool(
            getattr(args, "rust_live_timing_audit", False)
        ):
            raise ValueError(
                "Rust live profile settings require --rust-live-qualification"
            )
        return None
    missing = [
        name.replace("_", "-") for name, value in provided.items() if value is None
    ]
    if missing:
        raise ValueError(
            "Rust live qualification requires explicit settings: " + ", ".join(missing)
        )
    if args.rust_live_camera_backend != "gstreamer":
        raise ValueError("Rust live qualification currently requires GStreamer capture")
    if args.rust_live_image_space != "raw":
        raise ValueError("Rust live qualification requires a raw camera image profile")
    for name in (
        "rust_live_camera_width_px",
        "rust_live_camera_height_px",
        "rust_live_camera_fps",
        "rust_live_output_width_px",
        "rust_live_output_height_px",
    ):
        value = getattr(args, name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(
                "Rust live camera and output dimensions/rate must be positive"
            )
    pixel_format = args.rust_live_camera_pixel_format
    if not isinstance(pixel_format, str) or not pixel_format.strip():
        raise ValueError("Rust live camera pixel format must be non-empty")
    selector = args.rust_live_device_selector
    prefix = "gstreamer-device-v1:"
    digest = selector[len(prefix) :] if isinstance(selector, str) else ""
    if (
        not isinstance(selector, str)
        or not selector.startswith(prefix)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError(
            "Rust live qualification requires an exact lowercase "
            "gstreamer-device-v1:<sha256> selector"
        )
    return provided


def _read_rust_live_settings(
    settings_path: Path,
    profile_arguments: dict[str, Any],
) -> Any:
    """Build validated supervisor settings from a strict explicit JSON document."""
    from backend.perception.capture_configuration import resolve_capture_settings
    from backend.perception.rust_live_supervisor import (
        RustLiveInferenceSettings,
        RustLiveSettings,
    )

    try:
        raw = json.loads(settings_path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ValueError(f"could not read Rust live settings: {error}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"Rust live settings are not valid JSON: {error}") from error
    if not isinstance(raw, dict):
        raise ValueError("Rust live settings must be a JSON object")
    expected_top_level = {"executable_path", "project_root", "inference", "supervisor"}
    if set(raw) != expected_top_level:
        missing = sorted(expected_top_level - set(raw))
        unexpected = sorted(set(raw) - expected_top_level)
        raise ValueError(
            "Rust live settings keys mismatch; "
            f"missing={missing}, unexpected={unexpected}"
        )

    def path_value(value: Any, field_name: str) -> Path:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Rust live setting {field_name} must be a path string")
        candidate = Path(value).expanduser()
        if not candidate.is_absolute():
            candidate = ROOT / candidate
        return candidate.resolve()

    inference_values = raw["inference"]
    supervisor_values = raw["supervisor"]
    if not isinstance(inference_values, dict) or not isinstance(
        supervisor_values, dict
    ):
        raise ValueError("Rust live inference and supervisor settings must be objects")
    expected_inference = {field.name for field in fields(RustLiveInferenceSettings)}
    expected_supervisor = {
        field.name
        for field in fields(RustLiveSettings)
        if field.name
        not in {"capture_settings", "inference", "executable_path", "project_root"}
    }
    if set(inference_values) != expected_inference:
        raise ValueError(
            "Rust live inference keys mismatch; "
            f"missing={sorted(expected_inference - set(inference_values))}, "
            f"unexpected={sorted(set(inference_values) - expected_inference)}"
        )
    if set(supervisor_values) != expected_supervisor:
        raise ValueError(
            "Rust live supervisor keys mismatch; "
            f"missing={sorted(expected_supervisor - set(supervisor_values))}, "
            f"unexpected={sorted(set(supervisor_values) - expected_supervisor)}"
        )

    inference_values = dict(inference_values)
    for field_name in ("onnx_path", "source_checkpoint_path"):
        inference_values[field_name] = path_value(
            inference_values[field_name], f"inference.{field_name}"
        )
    inference = RustLiveInferenceSettings(**inference_values)
    supervisor_values = dict(supervisor_values)
    supervisor_values["capture_helper_executable_path"] = path_value(
        supervisor_values["capture_helper_executable_path"],
        "supervisor.capture_helper_executable_path",
    )

    capture_configuration = config.get_all()
    if not isinstance(capture_configuration, dict):
        raise ValueError("effective application configuration is unavailable")
    profile_overrides = (
        (
            ("vision", "camera", "backend"),
            profile_arguments["rust_live_camera_backend"],
        ),
        (
            ("vision", "camera", "resolution"),
            [
                profile_arguments["rust_live_camera_width_px"],
                profile_arguments["rust_live_camera_height_px"],
            ],
        ),
        (
            ("vision", "camera", "fps"),
            profile_arguments["rust_live_camera_fps"],
        ),
        (("vision", "camera", "video_file_path"), ""),
        (("vision", "camera", "control_policy"), "source_defaults"),
        (
            ("perception", "capture", "device_selector"),
            profile_arguments["rust_live_device_selector"],
        ),
        (
            ("perception", "capture", "pixel_format"),
            profile_arguments["rust_live_camera_pixel_format"],
        ),
        (
            ("perception", "capture", "output_resolution"),
            [
                profile_arguments["rust_live_output_width_px"],
                profile_arguments["rust_live_output_height_px"],
            ],
        ),
        (
            ("perception", "capture", "image_space"),
            profile_arguments["rust_live_image_space"],
        ),
        (("perception", "capture", "rectification_revision"), 0),
        (("perception", "capture", "timestamp_semantics"), "read_completion_only"),
    )
    for path, value in profile_overrides:
        _merge_configuration_value(capture_configuration, path, value)
    capture_settings = resolve_capture_settings(
        _QualificationConfiguration(capture_configuration), schema_version=3
    )
    settings = RustLiveSettings(
        capture_settings=capture_settings,
        inference=inference,
        executable_path=path_value(raw["executable_path"], "executable_path"),
        project_root=path_value(raw["project_root"], "project_root"),
        **supervisor_values,
    )
    settings.validate()
    try:
        bind_ip = ipaddress.ip_address(settings.bind_address)
    except ValueError as error:
        if settings.bind_address != "localhost":
            raise ValueError(
                "Rust live qualification must bind its unauthenticated ingress to loopback"
            ) from error
    else:
        if not bind_ip.is_loopback:
            raise ValueError(
                "Rust live qualification must bind its unauthenticated ingress to loopback"
            )
    return settings


def _json_compatible_setting_value(value: Any) -> Any:
    """Convert nested native settings dataclasses into report-safe JSON values."""
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _json_compatible_setting_value(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, dict):
        return {
            str(key): _json_compatible_setting_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_json_compatible_setting_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(
        f"Rust live audit setting {type(value).__name__} is not JSON-compatible"
    )


async def _collect_rust_live_qualification_repetition(
    settings: Any,
    *,
    repetition: int,
    warmup_seconds: float,
    measured_seconds: float,
    poll_interval_seconds: float,
    resource_sample_interval_seconds: float,
    history_size: int,
    recorder: AsyncAuditRecorder,
    progress: bool,
    timing_audit_enabled: bool = False,
    audit_queue_capacity: int = 1,
    audit_session_results: list[dict[str, Any]] | None = None,
    timing_recorder: AsyncAuditRecorder | None = None,
    supervisor_factory: Callable[..., Any] | None = None,
    timing_summary_bridge_factory: Callable[..., Any] | None = None,
    timing_summary_executable: str | Path | None = None,
    resource_sampler_executable: str | Path | None = None,
) -> dict[str, Any]:
    """Measure one bounded live-camera run through the established supervisor."""
    from backend.perception.rust_live_supervisor import (
        RustLiveAuditIngress,
        RustLiveSupervisor,
    )
    from backend.perception.scene_state_adapter import SceneSnapshotStatePublisher

    factory = supervisor_factory or RustLiveSupervisor
    timing_bridge = None
    timing_ingress = None
    if timing_audit_enabled:
        bridge_factory = timing_summary_bridge_factory or RustLiveTimingSummaryBridge
        maximum_line_bytes = min(
            max(settings.maximum_message_bytes + 4096, 1_048_576),
            RustLiveTimingSummaryBridge._MAXIMUM_LINE_BYTES,
        )
        timing_bridge = bridge_factory(
            maximum_samples=history_size,
            queue_capacity=audit_queue_capacity,
            maximum_line_bytes=maximum_line_bytes,
            write_timeout_seconds=settings.write_timeout_ms / 1000,
            executable=timing_summary_executable,
        )

        def on_timing_record(record: dict[str, Any]) -> bool:
            bridge_accepted = timing_bridge.try_record(record)
            audit_recorder = timing_recorder or recorder
            event_accepted = audit_recorder.record(
                {
                    "event_type": "rust_live_pipeline_timing_record",
                    "repetition": repetition,
                    "rust_process_id": record.get("process_id"),
                    "audit_receiver_monotonic_ns": record.get(
                        "audit_receiver_monotonic_ns"
                    ),
                    "rust_record": record,
                }
            )
            return bridge_accepted and event_accepted

        timing_ingress = RustLiveAuditIngress(
            queue_capacity=audit_queue_capacity,
            maximum_record_bytes=settings.maximum_message_bytes,
            connect_timeout_ms=settings.connect_timeout_ms,
            write_timeout_ms=settings.write_timeout_ms,
            startup_timeout_seconds=settings.startup_timeout_seconds,
            retained_record_capacity=history_size,
            on_record=on_timing_record,
        )

    broadcaster = RustLiveQualificationBroadcaster(
        repetition=repetition,
        recorder=recorder,
        history_size=history_size,
    )
    publisher = SceneSnapshotStatePublisher(broadcaster)
    supervisor = (
        factory(settings, publisher, audit_ingress=timing_ingress)
        if timing_ingress is not None
        else factory(settings, publisher)
    )
    startup_started_ns = time.perf_counter_ns()
    timing_ingress_summary: dict[str, Any] = {"status": "disabled"}
    scene_transport_summary: dict[str, Any] = {"status": "disabled"}
    scene_transport_audit_enabled = False
    scene_transport_audit_finished = False
    resource_sampler: RustLiveChildResourceSampler | None = None
    resource_summary: dict[str, Any] = {
        "status": "unavailable",
        "unavailable_reason": "supervisor did not expose a child process id",
        "process_role": "rust_live_inference_child",
        "process_id": None,
        "sample_count": 0,
    }
    measurement_failure: dict[str, Any] | None = None
    measurement_started_ns: int | None = None
    measurement_ended_ns: int | None = None
    try:
        if timing_bridge is not None:
            await timing_bridge.start()
        if timing_ingress is not None:
            await timing_ingress.start()
        startup_status = await supervisor.start()
        startup_completed_ns = time.perf_counter_ns()
        if not getattr(startup_status, "ready", False):
            raise RuntimeError("Rust live supervisor returned before ready state")
        if progress:
            print(
                f"Rust live qualification repetition {repetition}: ready; "
                f"warming up for {warmup_seconds:g}s"
            )
        warmup_deadline = time.monotonic() + warmup_seconds
        while time.monotonic() < warmup_deadline:
            status = supervisor.status
            if not getattr(status, "ready", False):
                raise RuntimeError(
                    "Rust live supervisor lost readiness during warm-up: "
                    f"{_format_supervisor_failure(status)}"
                )
            await asyncio.sleep(
                min(poll_interval_seconds, warmup_deadline - time.monotonic())
            )

        process_id = getattr(supervisor, "process_id", None)
        if isinstance(process_id, int) and not isinstance(process_id, bool):
            resource_sampler = RustLiveChildResourceSampler(
                process_id=process_id,
                interval_seconds=resource_sample_interval_seconds,
                maximum_samples=history_size,
                startup_timeout_seconds=settings.startup_timeout_seconds,
                write_timeout_seconds=settings.write_timeout_ms / 1000,
                executable=resource_sampler_executable,
            )
            await resource_sampler.start()
        elif process_id is not None:
            resource_summary["unavailable_reason"] = (
                "supervisor exposed an invalid child process id"
            )

        measurement_started_ns = broadcaster.begin_measurement()
        if timing_bridge is not None:
            await timing_bridge.begin_measurement(measurement_started_ns)
        enable_transport_audit = getattr(supervisor, "enable_transport_audit", None)
        if callable(enable_transport_audit):
            enable_transport_audit()
            scene_transport_audit_enabled = True
            scene_transport_start_ns = time.perf_counter_ns()
        measurement_deadline = time.monotonic() + measured_seconds
        while time.monotonic() < measurement_deadline:
            status = supervisor.status
            if not getattr(status, "ready", False):
                measurement_failure = _supervisor_failure_detail(status)
                break
            await asyncio.sleep(
                min(poll_interval_seconds, measurement_deadline - time.monotonic())
            )
        if measurement_failure is None:
            final_measurement_status = supervisor.status
            if not getattr(final_measurement_status, "ready", False):
                measurement_failure = _supervisor_failure_detail(
                    final_measurement_status
                )
        measurement_ended_ns = broadcaster.end_measurement()
        if timing_bridge is not None:
            await timing_bridge.end_measurement(measurement_ended_ns)
        if scene_transport_audit_enabled:
            scene_transport_summary = _finish_scene_transport_audit(supervisor)
            scene_transport_audit_finished = True
            scene_transport_end_ns = measurement_ended_ns
            scene_transport_duration = max(
                0.0,
                (scene_transport_end_ns - scene_transport_start_ns) / 1_000_000_000,
            )
            scene_transport_summary["measured_seconds"] = scene_transport_duration
            scene_transport_summary["accepted_snapshots_per_second"] = (
                scene_transport_summary.get("accepted_snapshots", 0)
                / scene_transport_duration
                if scene_transport_duration > 0
                else None
            )
            scene_transport_summary["measurement_window"] = {
                "start_monotonic_ns": scene_transport_start_ns,
                "end_monotonic_ns": scene_transport_end_ns,
                "clock": "collector_process_perf_counter",
            }
    finally:
        if broadcaster._measurement_open:
            measurement_ended_ns = broadcaster.end_measurement()
            if timing_bridge is not None:
                await timing_bridge.end_measurement(measurement_ended_ns)
        try:
            if resource_sampler is not None:
                await resource_sampler.close()
                resource_summary = resource_sampler.summary()
        finally:
            try:
                if scene_transport_audit_enabled and not scene_transport_audit_finished:
                    scene_transport_summary = _finish_scene_transport_audit(supervisor)
                await supervisor.close()
            finally:
                if timing_ingress is not None:
                    await timing_ingress.close()
                    timing_ingress_summary = timing_ingress.summary()
                if timing_bridge is not None and sys.exc_info()[0] is not None:
                    await timing_bridge.close()

    final_status = supervisor.status
    worker_stats = getattr(supervisor, "worker_stats", None)
    worker_stats_value = (
        {
            field.name: getattr(worker_stats, field.name)
            for field in fields(worker_stats)
        }
        if worker_stats is not None and hasattr(worker_stats, "__dataclass_fields__")
        else None
    )
    capture_acknowledgement = getattr(
        supervisor, "capture_session_acknowledgement", None
    )
    capture_acknowledgement_value = (
        {
            "configuration_revision": capture_acknowledgement.configuration_revision,
            "request_sha256": capture_acknowledgement.request_sha256,
            "session_id": capture_acknowledgement.session_id,
            "source_report": capture_acknowledgement.source_report,
        }
        if capture_acknowledgement is not None
        else None
    )
    duration = (
        (measurement_ended_ns - measurement_started_ns) / 1_000_000_000
        if measurement_started_ns is not None and measurement_ended_ns is not None
        else 0.0
    )
    state_summary = (
        broadcaster.measurement.summary(duration)
        if broadcaster._measurement is not None
        else {
            "duration_seconds": duration,
            "state_messages_received": 0,
            "state_messages_per_second": 0,
            "unique_source_observations": 0,
            "unique_observations_per_second": 0,
            "retained_state_trace_count": 0,
            "retained_state_trace_limit": history_size,
            "state_trace_truncated": False,
            "accuracy_evidence": "unavailable without temporal ground-truth labels",
        }
    )
    timing_result = None
    try:
        if timing_bridge is not None:
            timing_result = await timing_bridge.finish(
                (
                    broadcaster.measurement.retained_trace
                    if broadcaster._measurement is not None
                    else []
                ),
                state_trace_truncated=state_summary["state_trace_truncated"],
                upstream_dropped_records=(
                    timing_ingress_summary.get("consumer_dropped_records") or 0
                ),
            )
    finally:
        if timing_bridge is not None:
            bridge_close_task = asyncio.create_task(timing_bridge.close())
            try:
                await asyncio.shield(bridge_close_task)
            except asyncio.CancelledError:
                await bridge_close_task
                raise
    timing_summary = (
        timing_result.get("summary", {})
        if timing_result is not None
        else {"status": "disabled"}
    )
    stale_diagnostic = (
        timing_result.get("stale_inference_timing_diagnostic", {})
        if timing_result is not None
        else None
    )
    if timing_ingress is not None:
        bridge_metadata = (
            timing_bridge.failure_metadata()
            if timing_bridge is not None
            else {"status": "unavailable", "failure": "bridge was not created"}
        )
        timing_summary["status"] = (
            "complete"
            if timing_summary.get("status") == "complete"
            and timing_ingress_summary.get("status") == "complete"
            and bridge_metadata.get("status") == "complete"
            else "partial"
        )
        timing_summary["ingress_status"] = timing_ingress_summary
        timing_summary["bridge"] = bridge_metadata
    if audit_session_results is not None:
        audit_session_result = {
            "repetition": repetition,
            **timing_ingress_summary,
        }
        if stale_diagnostic is not None:
            audit_session_result["stale_inference_timing_diagnostic"] = stale_diagnostic
        if timing_bridge is not None:
            audit_session_result["timing_summary_bridge"] = (
                timing_bridge.failure_metadata()
            )
        audit_session_results.append(audit_session_result)
    return {
        "repetition": repetition,
        "measurement_status": "partial" if measurement_failure else "complete",
        "measurement_failure": measurement_failure,
        "warmup_seconds_requested": warmup_seconds,
        "warmup_state_messages": broadcaster.warmup_messages,
        "measured_seconds": duration,
        "startup_to_first_localization_ms": (startup_completed_ns - startup_started_ns)
        / 1_000_000,
        "state": state_summary,
        "rust_scene_transport": scene_transport_summary,
        "rust_pipeline_timing": timing_summary,
        "rust_live_audit_ingress": timing_ingress_summary,
        "supervisor_status_after_stop": {
            "ready": getattr(final_status, "ready", None),
            "process_status": getattr(final_status, "process_status", None),
            "process_return_code": getattr(final_status, "process_return_code", None),
            "failure": getattr(final_status, "failure", None),
            "source_health": getattr(final_status, "source_health", None),
            "stderr_tail": getattr(final_status, "stderr_tail", ""),
            "ingress_accepted_snapshots": getattr(
                final_status, "ingress_accepted_snapshots", None
            ),
            "ingress_queue_replacements": getattr(
                final_status, "ingress_queue_replacements", None
            ),
            "state_snapshots_published": getattr(
                final_status, "state_snapshots_published", None
            ),
        },
        "worker_stats_after_stop": worker_stats_value,
        "process_resources": resource_summary,
        "capture_session_acknowledgement": capture_acknowledgement_value,
        "state_trace": broadcaster.measurement.retained_trace,
    }


def _supervisor_failure_detail(status: Any) -> dict[str, Any]:
    """Keep the bounded process and source diagnostics with a partial run."""
    return {
        "process_status": getattr(status, "process_status", None),
        "process_return_code": getattr(status, "process_return_code", None),
        "failure": getattr(status, "failure", None),
        "source_health": getattr(status, "source_health", None),
        "last_observation_age_seconds": getattr(
            status, "last_observation_age_seconds", None
        ),
        "stderr_tail": getattr(status, "stderr_tail", ""),
    }


def _finish_scene_transport_audit(supervisor: Any) -> dict[str, Any]:
    """Finalize optional receiver counters without failing the qualification."""
    finish = getattr(supervisor, "finish_transport_audit", None)
    if not callable(finish):
        return {"status": "disabled"}
    try:
        summary = finish()
    except Exception as error:
        return {
            "status": "partial",
            "error": f"{type(error).__name__}: {error}",
        }
    if isinstance(summary, dict):
        return dict(summary)
    if hasattr(summary, "__dataclass_fields__"):
        return {field.name: getattr(summary, field.name) for field in fields(summary)}
    return {
        "status": "partial",
        "error": "supervisor returned an unsupported scene transport audit summary",
    }


def _format_supervisor_failure(status: Any) -> str:
    detail = _supervisor_failure_detail(status)
    return json.dumps(detail, sort_keys=True, default=str)


def _rust_live_qualification_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Rust live camera qualification",
        "",
        f"Run: `{report['audit_run_id']}`",
        "",
        "This is an opt-in local qualification/smoke measurement. It is not a detector-accuracy result or production-readiness decision.",
        "Camera controls use the camera driver's source defaults; this runner does not write exposure, gain, focus, brightness, or other camera controls.",
        "No camera image/frame bytes are recorded. Bounded state-only traces contain detected locations and identity/status metadata.",
        "The timestamp marks parent completion of the full frame-record read; it is not camera exposure. Helper timing segments are measured in the helper process and are not compared with parent monotonic timestamps.",
        "Ball precision/recall, position error, temporal continuity, cue quality, and freshness accuracy require synchronized ground-truth annotations and remain unavailable here.",
        "",
        "| Repetition | Status | States/s | Unique observations/s | Repeated | Source gaps | Out of order | State receipt interval p95 (ms) | Observed balls | Identified balls |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for run in report["repetitions"]:
        state = run["state"]
        lines.append(
            "| {repetition} | {status} | {rate:.2f} | {unique_rate:.2f} | {repeated} | {gaps} | {out_of_order} | {p95} | {observed} | {identified} |".format(
                repetition=run["repetition"],
                status=run.get("measurement_status", "unknown"),
                rate=state["state_messages_per_second"],
                unique_rate=state["unique_observations_per_second"],
                repeated=state["repeated_source_observations"],
                gaps=state["source_sequence_gaps"],
                out_of_order=state["source_out_of_order_messages"],
                p95=(
                    f"{state['state_receipt_interval_ms']['p95']:.3f}"
                    if state["state_receipt_interval_ms"]["p95"] is not None
                    else "unavailable"
                ),
                observed=state["observed_ball_measurements"],
                identified=state["identified_ball_measurements"],
            )
        )
        failure = run.get("measurement_failure")
        if isinstance(failure, dict):
            lines.append(
                "\nRepetition {repetition} ended early: `{detail}`.".format(
                    repetition=run["repetition"],
                    detail=json.dumps(failure, sort_keys=True, default=str),
                )
            )
    lines.extend(
        [
            "",
            "## Rust scene transport",
            "",
            "Receiver-side payload and latest-only queue counters are enabled only for the measured window. Queue depth is sampled immediately before enqueue; the queue capacity is fixed by the existing transport contract. These aggregate counters do not split producer serialization, socket write, and receiver parsing.",
            "",
            "| Repetition | Status | Accepted snapshots/s | Payload bytes min / mean / max | Queue capacity / high-water | Queue replacements / fraction | Sequence gaps |",
            "|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for run in report["repetitions"]:
        transport = run.get("rust_scene_transport", {})
        payload = " / ".join(
            (
                (f"{value:.1f}" if isinstance(value, float) else str(value))
                if isinstance(value, (int, float)) and not isinstance(value, bool)
                else "unavailable"
            )
            for value in (
                transport.get("payload_bytes_minimum"),
                transport.get("payload_bytes_mean"),
                transport.get("payload_bytes_maximum"),
            )
        )
        replacement_fraction = transport.get("queue_replacement_fraction")
        replacement_value = (
            f"{replacement_fraction:.3f}"
            if isinstance(replacement_fraction, (int, float))
            else "unavailable"
        )
        lines.append(
            "| {repetition} | {status} | {rate} | {payload} | {capacity} / {high_water} | {replacements} / {fraction} | {gaps} |".format(
                repetition=run["repetition"],
                status=transport.get("status", "disabled"),
                rate=(
                    f"{transport['accepted_snapshots_per_second']:.2f}"
                    if isinstance(
                        transport.get("accepted_snapshots_per_second"), (int, float)
                    )
                    else "unavailable"
                ),
                payload=payload,
                capacity=transport.get("queue_capacity", "unavailable"),
                high_water=transport.get("queue_depth_high_water", "unavailable"),
                replacements=transport.get("queue_replacements", "unavailable"),
                fraction=replacement_value,
                gaps=transport.get("sequence_gaps", "unavailable"),
            )
        )
    lines.extend(
        [
            "",
            "## Rust child resources",
            "",
            "CPU utilization and resident memory are sampled only during each measured window. When the detail limit is reached, summary statistics describe the retained tail; the table shows the exact statistics window. Unavailable measurements remain explicit.",
            "",
            "| Repetition | Status | Retained / total samples | Statistics window | CPU p50 / p95 / p99 (%) | CPU time delta (s) | RSS min / max (MiB) | RSS first→last delta (MiB) |",
            "|---:|---|---:|---|---:|---:|---:|---:|",
        ]
    )
    for run in report["repetitions"]:
        resources = run.get("process_resources", {})
        cpu = resources.get("cpu_percent", {})
        cpu_time = resources.get("cpu_time_seconds", {})
        memory = resources.get("resident_memory_bytes", {})
        statistics_window = resources.get("statistics_window", "unavailable")
        lines.append(
            "| {repetition} | {status} | {retained} / {total} | {window} | {cpu} | {cpu_delta} | {memory_range} | {memory_delta} |".format(
                repetition=run["repetition"],
                status=resources.get("status", "unavailable"),
                retained=resources.get("retained_sample_count", 0),
                total=resources.get("sample_count", 0),
                window={
                    "all_samples": "all samples",
                    "retained_tail": "retained tail",
                }.get(statistics_window, statistics_window),
                cpu=" / ".join(
                    f"{value:.1f}" if isinstance(value, (int, float)) else "unavailable"
                    for value in (cpu.get("p50"), cpu.get("p95"), cpu.get("p99"))
                ),
                cpu_delta=(
                    "{:.3f}".format(cpu_time["delta"])
                    if isinstance(cpu_time.get("delta"), (int, float))
                    else "unavailable"
                ),
                memory_range=" / ".join(
                    (
                        f"{value / (1024 * 1024):.1f}"
                        if isinstance(value, int)
                        else "unavailable"
                    )
                    for value in (memory.get("minimum"), memory.get("maximum"))
                ),
                memory_delta=(
                    "{:.1f}".format(memory["delta"] / (1024 * 1024))
                    if isinstance(memory.get("delta"), int)
                    else "unavailable"
                ),
            )
        )
    if any(
        run.get("rust_pipeline_timing", {}).get("status") != "disabled"
        for run in report["repetitions"]
    ):
        lines.extend(
            [
                "",
                "## Rust pipeline timing (opt-in)",
                "",
                "Durations are measured within the Rust process and correlated to received states by engine, session, frame, and transport sequence. They do not establish camera-exposure-to-client latency.",
                "",
                "| Repetition | Timing status | Matched frames | Helper pull/copy p95 (ms) | Parent framed read p95 (ms) | Parent read→inference p95 (ms) | Preprocess p95 (ms) | Inference p95 (ms) | Decode/NMS p95 (ms) | Scene socket write p95 (ms) | Pressure frames | Evicted tracks |",
                "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for run in report["repetitions"]:
            timing = run.get("rust_pipeline_timing", {})
            correlation = timing.get("state_timing_correlation", {})
            stages = timing.get("stage_durations_ms", {})
            socket_writes = timing.get("matched_scene_socket_write_durations_ms", {})
            written = socket_writes.get("written", {})
            pressure = timing.get("association_capacity_pressure", {})
            lines.append(
                "| {repetition} | {status} | {matched} | {helper_pull_copy} | {parent_read} | {read_to_inference} | {preprocess} | {inference} | {decode} | {write} | {pressure_frames} | {evicted_tracks} |".format(
                    repetition=run["repetition"],
                    status=timing.get("status", "unavailable"),
                    matched=correlation.get("matched_frames", 0),
                    helper_pull_copy="{pull} / {copy}".format(
                        pull=_format_percentile_ms(
                            stages, "capture_helper_appsink_pull_wait", "p95"
                        ),
                        copy=_format_percentile_ms(
                            stages, "capture_helper_rgb_mapping_copy", "p95"
                        ),
                    ),
                    parent_read=_format_percentile_ms(
                        stages, "parent_framed_record_read", "p95"
                    ),
                    read_to_inference=_format_percentile_ms(
                        stages,
                        "parent_record_read_complete_to_inference_start",
                        "p95",
                    ),
                    preprocess=_format_percentile_ms(stages, "preprocess", "p95"),
                    inference=_format_percentile_ms(stages, "inference", "p95"),
                    decode=_format_percentile_ms(stages, "decode_and_nms", "p95"),
                    write=(
                        f"{written['p95']:.3f}"
                        if isinstance(written.get("p95"), (int, float))
                        else "unavailable"
                    ),
                    pressure_frames=pressure.get("measured_window_pressure_frames", 0),
                    evicted_tracks=pressure.get("measured_window_evicted_tracks", 0),
                )
            )
            stale = timing.get("stale_inference_timing_ms", {})
            stale_stages = stale.get("stage_durations_ms", {})
            lines.append(
                "Stale-after-inference outcomes for repetition {repetition}: "
                "{timed}/{total} timed; engine total p95={engine_total} ms; "
                "age at completion p95={age} ms; deadline overrun p95={overrun} ms.".format(
                    repetition=run["repetition"],
                    timed=stale.get("timed_stale_outcomes", 0),
                    total=stale.get("stale_outcomes", 0),
                    engine_total=_format_percentile_ms(
                        stale_stages, "engine_total", "p95"
                    ),
                    age=_format_percentile_ms(
                        stale_stages,
                        "parent_record_read_complete_to_inference_complete",
                        "p95",
                    ),
                    overrun=(
                        "{:.3f}".format(stale["deadline_overrun_ms"]["p95"])
                        if isinstance(
                            stale.get("deadline_overrun_ms", {}).get("p95"),
                            (int, float),
                        )
                        else "unavailable"
                    ),
                )
            )
            pressure_status = pressure.get("measurement_status", "unavailable")
            total_pressure_frames = pressure.get("run_total_pressure_frames")
            total_evictions = pressure.get("run_total_evicted_tracks")
            lines.append(
                "Association-pressure totals for repetition {repetition}: "
                "{status}; full process including warm-up={frames} pressure frames, "
                "{evictions} evicted tracks.".format(
                    repetition=run["repetition"],
                    status=pressure_status,
                    frames=(
                        total_pressure_frames
                        if isinstance(total_pressure_frames, int)
                        else "unavailable"
                    ),
                    evictions=(
                        total_evictions
                        if isinstance(total_evictions, int)
                        else "unavailable"
                    ),
                )
            )
    lines.extend(
        [
            "",
            f"Capture selector: `{report['capture_profile']['device_selector']}`.",
            f"Capture profile: `{report['capture_profile']['input_width_px']}x{report['capture_profile']['input_height_px']}@{report['capture_profile']['fps']} {report['capture_profile']['input_pixel_format']}` to `{report['capture_profile']['output_width_px']}x{report['capture_profile']['output_height_px']} RGB8`.",
            f"Camera-control policy: `{report['capture_profile']['camera_controls']}`.",
            f"State-event exporter: accepted={report['event_export']['accepted_events']}, written={report['event_export']['written_events']}, dropped={report['event_export']['dropped_events']}, sampled={report['event_export']['sampled_out_events']}, export errors={report['event_export']['export_errors']}.",
            "Pipeline-timing event exporter: "
            + (
                "disabled."
                if report.get("rust_pipeline_timing_event_file") is None
                else (
                    "written={written}, dropped={dropped}, sampled={sampled}, "
                    "export errors={errors}; trace: `{path}`."
                ).format(
                    written=report["rust_pipeline_timing_event_export"][
                        "written_events"
                    ],
                    dropped=report["rust_pipeline_timing_event_export"][
                        "dropped_events"
                    ],
                    sampled=report["rust_pipeline_timing_event_export"][
                        "sampled_out_events"
                    ],
                    errors=report["rust_pipeline_timing_event_export"]["export_errors"],
                    path=report["rust_pipeline_timing_event_file"],
                )
            ),
            "",
            f"State-only event trace: `{report['state_event_file']}`. Frame image persistence: `{report['image_frame_persistence']}`.",
            "",
        ]
    )
    return "\n".join(lines)


def _format_percentile_ms(
    stages: dict[str, Any], name: str, percentile_name: str
) -> str:
    summary = stages.get(name, {})
    value = summary.get(percentile_name) if isinstance(summary, dict) else None
    return f"{value:.3f}" if isinstance(value, (int, float)) else "unavailable"


async def _run_rust_live_qualification(
    args: argparse.Namespace,
    *,
    capture_transport_settings: dict[str, Any] | None,
    supervisor_factory: Callable[..., Any] | None = None,
) -> Path:
    """Run opt-in local physical-source qualification using the report pipeline."""
    profile = _rust_live_profile_arguments(args)
    if profile is None:
        raise ValueError("Rust live qualification was not explicitly enabled")
    if (
        not math.isfinite(args.warmup)
        or not math.isfinite(args.duration)
        or not math.isfinite(args.sample_rate)
        or not math.isfinite(args.rust_live_status_poll_interval)
        or not math.isfinite(args.rust_live_resource_sample_interval)
        or args.warmup < 0
        or args.duration <= 0
        or args.repetitions < 1
        or args.history_size < 1
        or args.queue_capacity < 1
        or args.event_export_limit < 1
        or not 0 < args.sample_rate <= 1
        or args.rust_live_status_poll_interval <= 0
        or args.rust_live_resource_sample_interval <= 0
    ):
        raise ValueError("Rust live warm-up, duration and bounds are invalid")
    timing_audit_enabled = bool(getattr(args, "rust_live_timing_audit", False))
    settings = _read_rust_live_settings(args.rust_live_settings, profile)

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
    run_directory: Path = Path(args.output_root) / run_id
    run_directory.mkdir(parents=True, exist_ok=False)
    state_recorder = AsyncAuditRecorder(
        run_directory / "rust-live-state-events",
        queue_capacity=args.queue_capacity,
        sample_rate=args.sample_rate,
        max_exported_events=args.event_export_limit,
    )
    timing_recorder = (
        AsyncAuditRecorder(
            run_directory / "rust-live-pipeline-timing-events",
            queue_capacity=args.queue_capacity,
            sample_rate=args.sample_rate,
            max_exported_events=args.event_export_limit,
        )
        if timing_audit_enabled
        else None
    )
    audit_session_results: list[dict[str, Any]] = []
    try:
        repetitions = []
        for index in range(args.repetitions):
            repetitions.append(
                await _collect_rust_live_qualification_repetition(
                    settings,
                    repetition=index + 1,
                    warmup_seconds=args.warmup,
                    measured_seconds=args.duration,
                    poll_interval_seconds=args.rust_live_status_poll_interval,
                    resource_sample_interval_seconds=(
                        args.rust_live_resource_sample_interval
                    ),
                    history_size=args.history_size,
                    recorder=state_recorder,
                    progress=not args.quiet,
                    timing_audit_enabled=timing_audit_enabled,
                    audit_queue_capacity=args.queue_capacity,
                    audit_session_results=audit_session_results,
                    timing_recorder=timing_recorder,
                    supervisor_factory=supervisor_factory,
                    timing_summary_executable=getattr(
                        args, "rust_live_timing_summary_executable", None
                    ),
                    resource_sampler_executable=getattr(
                        args, "rust_live_resource_sampler_executable", None
                    ),
                )
            )
        capture_transport_summary = None
        if capture_transport_settings is not None:
            capture_transport_summary = _run_capture_transport_benchmark(
                capture_transport_settings,
                run_directory / "capture-transport.json",
            )
        state_recorder.close(
            timeout_seconds=float(
                config.get("performance_audit.export_drain_timeout_seconds", 5.0)
            )
        )
        if timing_recorder is not None:
            timing_recorder.close(
                timeout_seconds=float(
                    config.get("performance_audit.export_drain_timeout_seconds", 5.0)
                )
            )
        timing_event_export_status = (
            timing_recorder.status()
            if timing_recorder is not None
            else {"status": "disabled"}
        )
        if timing_recorder is not None and (
            timing_event_export_status.get("dropped_events", 0) > 0
            or timing_event_export_status.get("export_errors", 0) > 0
        ):
            for repetition_result in repetitions:
                repetition_result["rust_pipeline_timing"]["status"] = "partial"
                repetition_result["rust_pipeline_timing"][
                    "timing_event_export"
                ] = timing_event_export_status
        requested_capture = settings.capture_settings.requested
        inference = settings.inference
        metadata = _project_metadata()
        effective_configuration = metadata.get("effective_configuration")
        if isinstance(effective_configuration, dict):
            effective_configuration["camera"] = {
                "device_selector": requested_capture.get("device_id"),
                "backend": requested_capture.get("backend"),
                "requested_profile": requested_capture.get("requested_profile"),
                "camera_controls": requested_capture.get("camera_controls"),
                "timestamp_semantics": requested_capture.get("timestamp_semantics"),
            }
        metadata["configured_video_input"] = {
            "path": None,
            "exists": False,
            "sha256": None,
        }
        event_export_status = state_recorder.status()
        report = {
            "schema_version": 1,
            "audit_mode": "rust_live_camera_qualification",
            "audit_run_id": run_id,
            "created_at": datetime.now().astimezone().isoformat(),
            "qualification_status": (
                "partial_measurement"
                if any(
                    run.get("measurement_status") != "complete" for run in repetitions
                )
                else "smoke_measurement_only"
            ),
            "metadata": metadata,
            "capture_profile": {
                "device_selector": requested_capture.get("device_id"),
                "backend": requested_capture.get("backend"),
                "requested_profile": requested_capture.get("requested_profile"),
                "input_width_px": profile["rust_live_camera_width_px"],
                "input_height_px": profile["rust_live_camera_height_px"],
                "fps": profile["rust_live_camera_fps"],
                "input_pixel_format": profile["rust_live_camera_pixel_format"],
                "output_width_px": profile["rust_live_output_width_px"],
                "output_height_px": profile["rust_live_output_height_px"],
                "output_pixel_format": "RGB8",
                "image_space": requested_capture.get("image_space"),
                "camera_controls": requested_capture.get("camera_controls"),
                "source_timestamp_semantics": requested_capture.get(
                    "timestamp_semantics"
                ),
            },
            "model": {
                "onnx_path": str(inference.onnx_path),
                "onnx_sha256": _hash_file(inference.onnx_path),
                "source_checkpoint_path": str(inference.source_checkpoint_path),
                "source_checkpoint_sha256": _hash_file(
                    inference.source_checkpoint_path
                ),
                "provider": inference.provider,
                "image_size": inference.image_size,
                "ball_class_id": inference.ball_class_id,
                "cue_class_id": inference.cue_class_id,
            },
            "settings": {
                "executable_path": str(settings.executable_path),
                "project_root": str(settings.project_root),
                "capture_request": settings.capture_settings.to_dict(),
                "inference": {
                    field.name: _json_compatible_setting_value(
                        getattr(inference, field.name)
                    )
                    for field in fields(inference)
                },
                "supervisor": {
                    field.name: _json_compatible_setting_value(
                        getattr(settings, field.name)
                    )
                    for field in fields(settings)
                    if field.name
                    not in {
                        "capture_settings",
                        "inference",
                        "executable_path",
                        "project_root",
                    }
                },
                "qualification": {
                    "warmup_seconds": args.warmup,
                    "measured_seconds_per_repetition": args.duration,
                    "repetitions": args.repetitions,
                    "history_size": args.history_size,
                    "status_poll_interval_seconds": args.rust_live_status_poll_interval,
                    "resource_sample_interval_seconds": (
                        args.rust_live_resource_sample_interval
                    ),
                    "event_queue_capacity": args.queue_capacity,
                    "event_sample_rate": args.sample_rate,
                    "event_export_limit": args.event_export_limit,
                    "rust_live_timing_audit_enabled": timing_audit_enabled,
                    "rust_live_timing_audit_queue_capacity": (
                        args.queue_capacity if timing_audit_enabled else None
                    ),
                    "rust_live_timing_audit_maximum_record_bytes": (
                        settings.maximum_message_bytes if timing_audit_enabled else None
                    ),
                    "rust_live_timing_summary_executable": (
                        str(args.rust_live_timing_summary_executable)
                        if timing_audit_enabled
                        and getattr(args, "rust_live_timing_summary_executable", None)
                        else None
                    ),
                    "rust_live_resource_sampler_executable": (
                        str(args.rust_live_resource_sampler_executable)
                        if getattr(args, "rust_live_resource_sampler_executable", None)
                        else None
                    ),
                },
            },
            "repetitions": repetitions,
            "rust_live_audit_sessions": audit_session_results,
            "capture_session_acknowledgements": [
                run["capture_session_acknowledgement"] for run in repetitions
            ],
            "state_event_file": str(state_recorder.output_path),
            "event_export": event_export_status,
            "rust_pipeline_timing_event_file": (
                str(timing_recorder.output_path)
                if timing_recorder is not None
                else None
            ),
            "rust_pipeline_timing_event_export": (timing_event_export_status),
            "source_timestamp_claim": "read_completion_only; camera exposure time is unavailable",
            "source_to_state_latency": {
                "status": "unavailable_without_verified_shared_monotonic_clock_domain"
            },
            "image_frame_persistence": "none; only bounded detected-state metadata is retained",
            "accuracy_evidence": "unavailable_without_temporal_ground-truth labels",
            "limitations": [
                "This opt-in local qualification bypasses API provider selection and does not enable Rust live in the application runtime.",
                "The bounded broadcaster records adapted state locations in memory and the existing asynchronous exporter persists state-only traces; RGB image/frame bytes are never retained or exported.",
                "The source timestamp uses the helper's read-completion timestamp. It is not camera exposure time, and cross-process monotonic comparability has not been established.",
                "The event exporter is bounded and non-blocking; dropped, sampled, or failed state events are reported and reduce trace completeness.",
                "No accuracy claim is possible without synchronized temporal labels for ball detections, positions, identities, cue quality, and tracking continuity.",
            ],
        }
        report["state_event_loss"] = {
            "dropped_events": event_export_status["dropped_events"],
            "sampled_out_events": event_export_status["sampled_out_events"],
            "export_limit_dropped_events": event_export_status[
                "export_limit_dropped_events"
            ],
            "export_errors": event_export_status["export_errors"],
        }
        if capture_transport_summary is not None:
            report["capture_transport_benchmark"] = capture_transport_summary
        (run_directory / "summary.json").write_text(
            json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8"
        )
        (run_directory / "report.md").write_text(
            _rust_live_qualification_markdown(report), encoding="utf-8"
        )
        if not args.quiet:
            print(json.dumps(report["event_export"], indent=2))
            print(f"Rust live qualification report: {run_directory / 'report.md'}")
    except BaseException as error:
        state_recorder.close(timeout_seconds=1.0)
        if timing_recorder is not None:
            timing_recorder.close(timeout_seconds=1.0)
        failure = {
            "schema_version": 1,
            "audit_mode": "rust_live_camera_qualification",
            "audit_run_id": run_id,
            "outcome": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
            "event_export": state_recorder.status(),
            "rust_live_audit_sessions": audit_session_results,
            "rust_pipeline_timing_event_export": (
                timing_recorder.status()
                if timing_recorder is not None
                else {"status": "disabled"}
            ),
            "image_frame_persistence": "none",
            "limitations": [
                "This failure record contains no detector-accuracy result.",
                "No camera image/frame bytes are recorded by the qualification runner.",
            ],
        }
        (run_directory / "failure.json").write_text(
            json.dumps(failure, indent=2, default=str) + "\n", encoding="utf-8"
        )
        (run_directory / "report.md").write_text(
            "# Rust live camera qualification\n\n"
            f"Run: `{run_id}`\n\n"
            "**Run failed before a complete measurement was produced.**\n\n"
            f"Error type: `{failure['error_type']}`\n\n"
            f"Error: {failure['error']}\n\n"
            f"State-event exporter status: `{failure['event_export']}`\n\n"
            "No camera image/frame bytes are recorded by this runner. "
            "This failure is not an accuracy or performance result.\n",
            encoding="utf-8",
        )
        raise
    return run_directory


async def run_audit(args: argparse.Namespace) -> Path:
    """Run configured repetitions and save versioned reports and traces."""
    capture_transport_settings = _capture_transport_settings(args)
    if _rust_live_profile_arguments(args) is not None:
        return await _run_rust_live_qualification(
            args,
            capture_transport_settings=capture_transport_settings,
        )
    streams = [
        value.strip().lower() for value in args.streams.split(",") if value.strip()
    ]
    if "state" not in streams or set(streams) - {"state", "trajectory"}:
        raise ValueError("--streams must include state and may include trajectory")
    if (
        args.warmup < 0
        or args.duration <= 0
        or args.repetitions < 1
        or args.history_size < 1
        or args.queue_capacity < 1
        or args.event_export_limit < 1
        or args.heartbeat_interval < 1
        or args.subscriber_delay_ms < 0
        or not 0 < args.sample_rate <= 1
    ):
        raise ValueError(
            "warmup must be non-negative; duration and repetitions positive"
        )

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
    run_directory: Path = Path(args.output_root) / run_id
    run_directory.mkdir(parents=True, exist_ok=False)
    event_recorder = AsyncAuditRecorder(
        run_directory / "events",
        queue_capacity=args.queue_capacity,
        sample_rate=args.sample_rate,
        max_exported_events=args.event_export_limit,
    )
    try:
        repetitions = []
        for index in range(args.repetitions):
            repetition_result = await collect_repetition(
                args.url,
                streams=streams,
                warmup_seconds=args.warmup,
                measured_seconds=args.duration,
                repetition=index + 1,
                recorder=event_recorder,
                video_url=args.video_url,
                progress=not args.quiet,
                history_size=args.history_size,
                subscriber_delay_ms=args.subscriber_delay_ms,
                heartbeat_interval_seconds=args.heartbeat_interval,
                trajectory_queue_capacity=args.queue_capacity,
                trajectory_maximum_records=args.event_export_limit + 2,
            )
            repetition_result["collector_settings"] = {
                "history_size": args.history_size,
                "queue_capacity": args.queue_capacity,
                "trajectory_summary_queue_capacity": args.queue_capacity,
                "sample_rate": args.sample_rate,
                "event_export_limit": args.event_export_limit,
                "trajectory_summary_maximum_records_requested": (
                    args.event_export_limit + 2
                ),
                "heartbeat_interval_seconds": args.heartbeat_interval,
            }
            repetitions.append(repetition_result)
            if not args.quiet:
                print()
        capture_transport_summary = None
        if capture_transport_settings is not None:
            capture_transport_summary = _run_capture_transport_benchmark(
                capture_transport_settings,
                run_directory / "capture-transport.json",
            )
        event_recorder.close()
        client_events = _read_jsonl(event_recorder.output_path)
        audit_root = Path(
            str(
                config.get(
                    "performance_audit.output_directory",
                    "thoughts/performance-audits",
                )
            )
        )
        if not audit_root.is_absolute():
            audit_root = ROOT / audit_root
        server_event_path_set = {
            audit_root / "live" / str(run["server_audit_run_id"]) / "events.jsonl"
            for run in repetitions
            if run.get("server_audit_run_id")
        }
        live_window_seconds = (args.warmup + args.duration) * args.repetitions + 60
        recent_video_paths = audit_root / "video-process"
        if recent_video_paths.is_dir():
            for event_path in recent_video_paths.glob("*/events.jsonl"):
                try:
                    if event_path.stat().st_mtime >= time.time() - live_window_seconds:
                        server_event_path_set.add(event_path)
                except OSError:
                    continue
        server_event_paths = list(server_event_path_set)
        server_events = [
            event for path in server_event_paths for event in _read_jsonl(path)
        ]
        measured_server_events: list[dict[str, Any]] = []
        for repetition in repetitions:
            start_ns = repetition.get("measurement_started_monotonic_ns")
            end_ns = repetition.get("measurement_ended_monotonic_ns")
            clocks_comparable = repetition.get("monotonic_clocks_comparable") is True
            run_events = (
                [
                    event
                    for event in server_events
                    if isinstance(event.get("event_monotonic_ns"), int)
                    and isinstance(start_ns, int)
                    and isinstance(end_ns, int)
                    and start_ns <= event["event_monotonic_ns"] <= end_ns
                ]
                if clocks_comparable
                else []
            )
            measured_server_events.extend(run_events)
            repetition["server_measurements"] = _server_measurement_summary(run_events)
            _merge_final_export_status(
                repetition["server_measurements"]["resources"],
                repetition.get("server_event_export"),
            )
            repetition["server_measurement_status"] = (
                "same_clock_domain_window_available"
                if clocks_comparable
                else "unavailable_without_comparable_monotonic_clocks"
            )
        metadata = _project_metadata()
        server_measurements = _server_measurement_summary(measured_server_events)
        for repetition in repetitions:
            _merge_final_export_status(
                server_measurements["resources"],
                repetition.get("server_event_export"),
            )
        latency_budget_ms = float(
            config.get("performance_audit.latency_reference_budget_ms", 50.0)
        )
        report = {
            "schema_version": 6,
            "audit_run_id": run_id,
            "created_at": datetime.now().astimezone().isoformat(),
            "websocket_url": args.url,
            "scenario": args.scenario,
            "scenario_label": args.scenario,
            "video_url": args.video_url,
            "metadata": metadata,
            "repetitions": repetitions,
            "delivery_latency": _delivery_latency_summary(
                client_events,
                measured_server_events,
                deadline_budget_ms=latency_budget_ms,
                server_clock_domain_id=(
                    repetitions[0].get("server_clock_domain_id")
                    if repetitions
                    and len({run.get("server_clock_domain_id") for run in repetitions})
                    == 1
                    else None
                ),
            ),
            "latency_reference_budget_ms": latency_budget_ms,
            "server_event_count": len(measured_server_events),
            "server_event_file_count": len(server_events),
            "resource_summary": server_measurements["resources"],
            "server_event_export_status_by_process": _event_export_status_by_process(
                server_measurements["resources"]
            ),
            "measurement_overhead_comparison": _compare_overhead(
                args.compare_with,
                repetitions,
                metadata,
                float(
                    config.get(
                        "performance_audit.instrumentation_overhead_budget_percent",
                        2.0,
                    )
                ),
            ),
            "event_export": event_recorder.status(),
            "server_measurements": server_measurements,
            "limitations": [
                "The receiver observes delivery only while connected and does not infer unobserved exposures.",
                "Source publication-to-receipt latency uses monotonic counters only when the producer, server, and receiver share a verified host clock. The wall-clock publication-age estimate is secondary and is not used for latency ranking.",
                "This report does not claim projector display time.",
                "Live state accuracy requires separate labeled observations.",
            ],
        }
        if capture_transport_summary is not None:
            report["capture_transport_benchmark"] = capture_transport_summary
        (run_directory / "summary.json").write_text(
            json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8"
        )
        (run_directory / "report.md").write_text(
            _markdown_report(report), encoding="utf-8"
        )
        print(json.dumps(report["event_export"], indent=2))
        print(f"Audit report: {run_directory / 'report.md'}")
    except BaseException:
        event_recorder.close(timeout_seconds=1.0)
        raise
    return run_directory


def build_parser() -> argparse.ArgumentParser:
    audit_settings = config.get("performance_audit", {})
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        help="Configuration file used by the backend; pass the same file as the service",
    )
    websocket_path = str(config.get("api.api_prefix", "/api/v1")).rstrip("/")
    websocket_path += "/websocket/ws"
    parser.add_argument("--url", default=f"ws://127.0.0.1:8000{websocket_path}")
    parser.add_argument("--streams", default="state")
    parser.add_argument("--scenario", default="state-only")
    parser.add_argument(
        "--subscriber-delay-ms",
        type=float,
        default=0.0,
        help="Artificial delay after each received update for slow-consumer testing",
    )
    parser.add_argument(
        "--compare-with",
        type=Path,
        help="Compare with a prior instrumentation-off summary.json",
    )
    parser.add_argument(
        "--video-url", help="Optional MJPEG viewer URL for interference runs"
    )
    parser.add_argument(
        "--warmup",
        type=float,
        default=float(audit_settings.get("warmup_seconds", 30)),
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=float(audit_settings.get("measured_seconds", 120)),
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=int(audit_settings.get("repetitions", 3)),
    )
    parser.add_argument(
        "--soak-duration",
        type=float,
        default=float(audit_settings.get("soak_seconds", 1200)),
        help="Documented long-run duration; supply as --duration to run the soak",
    )
    parser.add_argument(
        "--soak",
        action="store_true",
        help="Run one bounded-memory soak using the configured soak duration",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT
        / str(audit_settings.get("output_directory", "thoughts/performance-audits")),
    )
    parser.add_argument(
        "--queue-capacity",
        type=int,
        default=int(audit_settings.get("event_queue_capacity", 2048)),
    )
    parser.add_argument(
        "--history-size",
        type=int,
        default=int(audit_settings.get("history_size", 10000)),
    )
    parser.add_argument(
        "--event-export-limit",
        type=int,
        default=int(audit_settings.get("event_export_limit", 200000)),
    )
    parser.add_argument(
        "--sample-rate",
        type=float,
        default=float(audit_settings.get("sample_rate", 1.0)),
    )
    parser.add_argument(
        "--heartbeat-interval",
        type=float,
        default=float(audit_settings.get("heartbeat_interval_seconds", 20.0)),
        help="Application heartbeat interval that keeps the state subscription active",
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--rust-live-qualification",
        action="store_true",
        help="Opt in to a local, state-only Rust physical-camera qualification run",
    )
    parser.add_argument(
        "--rust-live-timing-audit",
        action="store_true",
        help=(
            "Enable the bounded Rust pipeline timing side channel; requires "
            "--rust-live-qualification and adds instrumentation overhead"
        ),
    )
    parser.add_argument(
        "--rust-live-timing-summary-executable",
        type=Path,
        help=(
            "Optional path to the Rust timing-summary executable; otherwise the "
            "collector searches PATH and existing Cargo debug/release artifacts"
        ),
    )
    parser.add_argument(
        "--rust-live-resource-sampler-executable",
        type=Path,
        help=(
            "Optional path to the Rust process-resource sampler; otherwise the "
            "collector searches PATH and existing Cargo debug/release artifacts"
        ),
    )
    parser.add_argument(
        "--rust-live-settings",
        type=Path,
        help=(
            "Required JSON settings document with executable_path, project_root, "
            "and complete inference/supervisor settings objects"
        ),
    )
    parser.add_argument(
        "--rust-live-device-selector",
        help="Required exact gstreamer-device-v1:<sha256> webcam selector",
    )
    parser.add_argument(
        "--rust-live-camera-backend",
        choices=("gstreamer",),
        help="Required camera adapter for the stable selector",
    )
    parser.add_argument(
        "--rust-live-camera-width-px",
        type=int,
        help="Required negotiated raw camera input width",
    )
    parser.add_argument(
        "--rust-live-camera-height-px",
        type=int,
        help="Required negotiated raw camera input height",
    )
    parser.add_argument(
        "--rust-live-camera-fps",
        type=int,
        help="Required requested camera input frame rate",
    )
    parser.add_argument(
        "--rust-live-camera-pixel-format",
        help="Required negotiated raw camera input pixel format",
    )
    parser.add_argument(
        "--rust-live-image-space",
        choices=("raw",),
        help="Required raw camera image-space contract",
    )
    parser.add_argument(
        "--rust-live-output-width-px",
        type=int,
        help="Required emitted RGB8 frame width passed to the detector",
    )
    parser.add_argument(
        "--rust-live-output-height-px",
        type=int,
        help="Required emitted RGB8 frame height passed to the detector",
    )
    parser.add_argument(
        "--rust-live-status-poll-interval",
        type=float,
        default=float(audit_settings.get("resource_sample_interval_seconds", 1.0)),
        help="How often the qualification checks supervisor readiness",
    )
    parser.add_argument(
        "--rust-live-resource-sample-interval",
        type=float,
        default=float(audit_settings.get("resource_sample_interval_seconds", 1.0)),
        help="How often to sample the supervised Rust inference process",
    )
    parser.add_argument(
        "--capture-transport-benchmark-executable",
        type=Path,
        help="Opt in to running the isolated Rust capture transport microbenchmark",
    )
    parser.add_argument(
        "--capture-transport-width-px",
        type=int,
        help="Required benchmark RGB8 frame width when the executable is supplied",
    )
    parser.add_argument(
        "--capture-transport-height-px",
        type=int,
        help="Required benchmark RGB8 frame height when the executable is supplied",
    )
    parser.add_argument(
        "--capture-transport-warmup-frames",
        type=int,
        help="Required per-arm, per-repetition warmup frame count",
    )
    parser.add_argument(
        "--capture-transport-measured-frames",
        type=int,
        help="Required measured frame count per arm and repetition",
    )
    parser.add_argument(
        "--capture-transport-repetitions",
        type=int,
        help="Required paired benchmark repetition count",
    )
    parser.add_argument(
        "--capture-transport-sampling-limit",
        type=int,
        help="Required maximum sampled frame completion records per arm and repetition",
    )
    parser.add_argument(
        "--capture-transport-seed",
        type=int,
        help="Required deterministic RGB8 payload seed",
    )
    parser.add_argument(
        "--capture-transport-max-payload-bytes",
        type=int,
        help="Required maximum allocation size for one RGB8 frame",
    )
    parser.add_argument(
        "--capture-transport-max-total-frames",
        type=int,
        help="Required maximum generated frame count per arm over all repetitions",
    )
    parser.add_argument(
        "--capture-transport-max-total-samples",
        type=int,
        help="Required maximum retained timing samples across both arms and repetitions",
    )
    parser.add_argument(
        "--capture-transport-timeout-seconds",
        type=float,
        help="Required wall-clock timeout for the benchmark subprocess",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path)
    config_arguments, _ = config_parser.parse_known_args(arguments)
    if config_arguments.config is not None:
        if not config_arguments.config.is_file():
            raise FileNotFoundError(
                f"collector config does not exist: {config_arguments.config}"
            )
        Config.set_config_file(config_arguments.config)
    parser = build_parser()
    args = parser.parse_args(arguments)
    try:
        _capture_transport_settings(args)
        _rust_live_profile_arguments(args)
    except ValueError as error:
        parser.error(str(error))
    if args.soak:
        args.duration = args.soak_duration
        args.repetitions = 1
    asyncio.run(run_audit(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
