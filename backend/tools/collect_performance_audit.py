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
import socket
import subprocess
import sys
import threading
import time
import uuid
from collections import Counter, OrderedDict, deque
from dataclasses import fields, is_dataclass
from datetime import datetime
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


class RustLiveTimingAccumulator:
    """Summarize bounded Rust stage, outcome, and correlated socket records."""

    _DURATION_FIELDS = (
        "capture_helper_appsink_pull_wait",
        "capture_helper_rgb_buffer_allocation",
        "capture_helper_rgb_mapping_copy",
        "parent_framed_record_read",
        "parent_record_read_complete_to_inference_start",
        "parent_record_read_complete_to_scene_queue_submit",
        # Preserve summaries for retained schema-v1 audit artifacts.
        "host_read_complete_to_inference_start",
        "preprocess",
        "inference",
        "decode_and_nms",
        "engine_total",
        "calibration",
        "tracking_and_state_assembly",
        "scene_json_serialization",
        "scene_queue_submit",
        "host_read_complete_to_scene_queue_submit",
    )
    _INFERENCE_OUTCOME_DURATION_FIELDS = (
        "capture_helper_appsink_pull_wait",
        "capture_helper_rgb_buffer_allocation",
        "capture_helper_rgb_mapping_copy",
        "parent_framed_record_read",
        "parent_record_read_complete_to_inference_start",
        "preprocess",
        "inference",
        "decode_and_nms",
        "engine_total",
        "parent_record_read_complete_to_inference_complete",
        "maximum_frame_age",
    )

    def __init__(self, maximum_samples: int) -> None:
        if maximum_samples < 1:
            raise ValueError("maximum_samples must be positive")
        self._records: deque[dict[str, Any]] = deque(maxlen=maximum_samples)
        self._record_counts: Counter[str] = Counter()
        self._outcome_counts: Counter[str] = Counter()
        self._reason_counts: Counter[str] = Counter()
        self._measurement_started_ns: int | None = None
        self._measurement_ended_ns: int | None = None
        self._footer: dict[str, Any] | None = None
        self._record_count = 0

    @property
    def retained_records(self) -> list[dict[str, Any]]:
        return list(self._records)

    def begin_measurement(self, monotonic_ns: int | None = None) -> int:
        self._measurement_started_ns = time.perf_counter_ns()
        if monotonic_ns is not None:
            self._measurement_started_ns = monotonic_ns
        self._measurement_ended_ns = None
        return self._measurement_started_ns

    def end_measurement(self, monotonic_ns: int | None = None) -> int:
        self._measurement_ended_ns = time.perf_counter_ns()
        if monotonic_ns is not None:
            self._measurement_ended_ns = monotonic_ns
        return self._measurement_ended_ns

    def add(self, record: dict[str, Any]) -> None:
        """Count all records and retain only a configured diagnostic tail."""
        self._record_count += 1
        record_type = record.get("record_type")
        if isinstance(record_type, str):
            self._record_counts[record_type] += 1
        if record_type == "observation_outcome":
            outcome = record.get("outcome")
            reason = record.get("reason")
            if isinstance(outcome, str):
                self._outcome_counts[outcome] += 1
            if isinstance(reason, str):
                self._reason_counts[reason] += 1
        if record_type == "footer":
            self._footer = record
        self._records.append(record)

    @staticmethod
    def _identity_key(event: dict[str, Any], *, timing: bool) -> tuple[Any, ...] | None:
        engine = event.get("engine_instance_id")
        session = event.get("source_session_id")
        frame = event.get("frame_id" if timing else "source_frame_id")
        sequence = event.get(
            "transport_sequence" if timing else "source_transport_sequence"
        )
        if (
            not isinstance(engine, str)
            or not engine
            or not isinstance(session, str)
            or not session
            or not isinstance(frame, str)
            or not frame
            or not isinstance(sequence, int)
            or isinstance(sequence, bool)
        ):
            return None
        return engine, session, frame, sequence

    def summary(
        self,
        measured_state_trace: list[dict[str, Any]],
        *,
        state_trace_truncated: bool,
    ) -> dict[str, Any]:
        """Join only identical frame/sequence identities and report retention."""
        state_keys = {
            key
            for event in measured_state_trace
            if (key := self._identity_key(event, timing=False)) is not None
        }
        frame_records: dict[tuple[Any, ...], dict[str, Any]] = {}
        duplicate_frame_records = 0
        write_records_by_key: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
        write_attempts = 0
        write_outcomes: Counter[str] = Counter()
        measurement_start = self._measurement_started_ns
        measurement_end = self._measurement_ended_ns
        for record in self._records:
            record_type = record.get("record_type")
            received_ns = record.get("audit_receiver_monotonic_ns")
            if (
                measurement_start is None
                or measurement_end is None
                or not isinstance(received_ns, int)
                or not measurement_start <= received_ns <= measurement_end
            ):
                continue
            key = self._identity_key(record, timing=True)
            if key is None:
                continue
            if record_type == "frame":
                if key in frame_records:
                    duplicate_frame_records += 1
                else:
                    frame_records[key] = record
            elif record_type == "scene_socket_write":
                write_attempts += 1
                outcome = record.get("outcome")
                if isinstance(outcome, str):
                    write_outcomes[outcome] += 1
                write_records_by_key.setdefault(key, []).append(record)

        matched_keys = state_keys & frame_records.keys()
        unmatched_state_keys = state_keys - frame_records.keys()
        unmatched_timing_keys = frame_records.keys() - state_keys
        durations: dict[str, list[float]] = {name: [] for name in self._DURATION_FIELDS}
        matched_detections: list[int] = []
        matched_tracks: list[int] = []
        for key in matched_keys:
            record = frame_records[key]
            raw_durations = record.get("durations_ms")
            if isinstance(raw_durations, dict):
                for name in self._DURATION_FIELDS:
                    value = raw_durations.get(name)
                    if (
                        isinstance(value, (int, float))
                        and not isinstance(value, bool)
                        and math.isfinite(value)
                        and value >= 0
                    ):
                        durations[name].append(float(value))
            for field_name, target in (
                ("detection_count", matched_detections),
                ("track_count", matched_tracks),
            ):
                value = record.get(field_name)
                if isinstance(value, int) and not isinstance(value, bool):
                    target.append(value)

        stage_summary = {
            name: {
                "count": len(values),
                "sample_limit": self._records.maxlen,
                "p50": percentile(values, 0.50),
                "p95": percentile(values, 0.95),
                "p99": percentile(values, 0.99),
            }
            for name, values in durations.items()
        }
        matched_writes = [
            event for key in matched_keys for event in write_records_by_key.get(key, [])
        ]
        matched_write_durations = {
            outcome: [
                float(event["duration_ms"])
                for event in matched_writes
                if event.get("outcome") == outcome
                and isinstance(event.get("duration_ms"), (int, float))
                and not isinstance(event.get("duration_ms"), bool)
                and math.isfinite(event["duration_ms"])
                and event["duration_ms"] >= 0
            ]
            for outcome in ("written", "failed")
        }
        write_duration_summary = {
            outcome: {
                "count": len(values),
                "p50": percentile(values, 0.50),
                "p95": percentile(values, 0.95),
                "p99": percentile(values, 0.99),
            }
            for outcome, values in matched_write_durations.items()
        }
        measured_outcomes: Counter[str] = Counter()
        measured_stale_outcomes = 0
        timed_stale_outcomes = 0
        stale_inference_durations: dict[str, list[float]] = {
            name: [] for name in self._INFERENCE_OUTCOME_DURATION_FIELDS
        }
        stale_deadline_overruns: list[float] = []
        out_of_window_outcomes = 0
        association_pressure_frames = 0
        association_capacity_evictions = 0
        malformed_pressure_outcomes = 0
        pressure_reason_prefix = "association_capacity_pressure_evicted_"
        for record in self._records:
            if record.get("record_type") != "observation_outcome":
                continue
            received_ns = record.get("audit_receiver_monotonic_ns")
            if (
                isinstance(received_ns, int)
                and measurement_start is not None
                and measurement_end is not None
                and measurement_start <= received_ns <= measurement_end
            ):
                outcome = record.get("outcome")
                if isinstance(outcome, str):
                    measured_outcomes[outcome] += 1
                if record.get("reason") == "stale_after_inference":
                    measured_stale_outcomes += 1
                    raw_timings = record.get("inference_timings_ms")
                    if isinstance(raw_timings, dict):
                        timed_stale_outcomes += 1
                        for name in self._INFERENCE_OUTCOME_DURATION_FIELDS:
                            duration = raw_timings.get(name)
                            if (
                                isinstance(duration, (int, float))
                                and not isinstance(duration, bool)
                                and math.isfinite(duration)
                                and duration >= 0
                            ):
                                stale_inference_durations[name].append(float(duration))
                        age = raw_timings.get(
                            "parent_record_read_complete_to_inference_complete"
                        )
                        maximum_age = raw_timings.get("maximum_frame_age")
                        if (
                            isinstance(age, (int, float))
                            and not isinstance(age, bool)
                            and math.isfinite(age)
                            and age >= 0
                            and isinstance(maximum_age, (int, float))
                            and not isinstance(maximum_age, bool)
                            and math.isfinite(maximum_age)
                            and maximum_age >= 0
                        ):
                            stale_deadline_overruns.append(
                                max(0.0, float(age) - float(maximum_age))
                            )
                if outcome == "degraded" and record.get("stage") == "track_association":
                    reason = record.get("reason")
                    suffix = (
                        reason[len(pressure_reason_prefix) :]
                        if isinstance(reason, str)
                        and reason.startswith(pressure_reason_prefix)
                        else ""
                    )
                    if suffix.isdecimal():
                        association_pressure_frames += 1
                        association_capacity_evictions += int(suffix)
                    else:
                        malformed_pressure_outcomes += 1
            else:
                out_of_window_outcomes += 1

        stale_inference_summary = {
            "stale_outcomes": measured_stale_outcomes,
            "timed_stale_outcomes": timed_stale_outcomes,
            "stage_durations_ms": {
                name: {
                    "count": len(values),
                    "sample_limit": self._records.maxlen,
                    "p50": percentile(values, 0.50),
                    "p95": percentile(values, 0.95),
                    "p99": percentile(values, 0.99),
                }
                for name, values in stale_inference_durations.items()
            },
            "deadline_overrun_ms": {
                "count": len(stale_deadline_overruns),
                "p50": percentile(stale_deadline_overruns, 0.50),
                "p95": percentile(stale_deadline_overruns, 0.95),
                "p99": percentile(stale_deadline_overruns, 0.99),
            },
        }

        footer = self._footer
        footer_pipeline_counters = (
            footer.get("pipeline_counters") if isinstance(footer, dict) else None
        )
        if not isinstance(footer_pipeline_counters, dict):
            footer_pipeline_counters = {}
        footer_pressure_frames = footer_pipeline_counters.get(
            "association_capacity_pressure_frames"
        )
        footer_capacity_evictions = footer_pipeline_counters.get(
            "association_capacity_evictions"
        )
        footer_export_complete = (
            footer is not None
            and footer.get("export_status") == "complete"
            and footer.get("dropped_records", 0) == 0
        )
        producer_dropped_records = (
            footer.get("dropped_records") if footer is not None else None
        )
        return {
            "status": "complete" if footer_export_complete else "partial",
            "producer_dropped_records": producer_dropped_records,
            "record_counts": dict(self._record_counts),
            "observation_outcomes_received": dict(self._outcome_counts),
            "observation_outcomes_in_measurement_window": dict(measured_outcomes),
            "out_of_window_outcome_records": out_of_window_outcomes,
            "observation_outcome_reasons_received": dict(self._reason_counts),
            "association_capacity_pressure": {
                "measurement_status": (
                    "complete" if footer_export_complete else "partial"
                ),
                "measured_window_pressure_frames": association_pressure_frames,
                "measured_window_evicted_tracks": association_capacity_evictions,
                "malformed_pressure_outcome_records": malformed_pressure_outcomes,
                "run_total_pressure_frames": footer_pressure_frames,
                "run_total_evicted_tracks": footer_capacity_evictions,
                "run_total_includes_warmup": True,
            },
            "state_timing_correlation": {
                "retained_measured_states": len(state_keys),
                "retained_timing_frames": len(frame_records),
                "matched_frames": len(matched_keys),
                "measured_states_without_timing": len(unmatched_state_keys),
                "timing_frames_without_measured_state": len(unmatched_timing_keys),
                "duplicate_timing_frame_records": duplicate_frame_records,
                "state_trace_truncated": state_trace_truncated,
                "timing_trace_truncated": self._record_count > len(self._records),
                "identity_fields": [
                    "engine_instance_id",
                    "source_session_id",
                    "source_frame_id",
                    "transport_sequence",
                ],
            },
            "stage_durations_ms": stage_summary,
            "stale_inference_timing_ms": stale_inference_summary,
            "matched_detection_count": {
                "count": len(matched_detections),
                "mean": (
                    sum(matched_detections) / len(matched_detections)
                    if matched_detections
                    else None
                ),
            },
            "matched_track_count": {
                "count": len(matched_tracks),
                "mean": (
                    sum(matched_tracks) / len(matched_tracks)
                    if matched_tracks
                    else None
                ),
            },
            "scene_socket_write_attempts_received": write_attempts,
            "scene_socket_write_outcomes_received": dict(write_outcomes),
            "matched_scene_socket_write_durations_ms": write_duration_summary,
            "measurement_monotonic_window_ns": {
                "start": measurement_start,
                "end": measurement_end,
                "clock": "collector_process_perf_counter",
                "applies_to": "audit record receive times and outcome window only",
            },
            "footer": footer,
            "records_received": self._record_count,
            "retained_records": len(self._records),
            "retention_limit": self._records.maxlen,
        }

    def stale_inference_diagnostic_summary(self) -> dict[str, Any]:
        """Summarize retained stale outcomes even when readiness never starts.

        This is explicitly an all-retained-record diagnostic, not a measured
        performance window. Startup failures have no valid warm-up/measurement
        boundary, but their timing evidence still explains why no state arrived.
        """
        stale_records = [
            record
            for record in self._records
            if record.get("record_type") == "observation_outcome"
            and record.get("reason") == "stale_after_inference"
        ]
        durations: dict[str, list[float]] = {
            name: [] for name in self._INFERENCE_OUTCOME_DURATION_FIELDS
        }
        deadline_overruns: list[float] = []
        timed_records = 0
        for record in stale_records:
            raw_timings = record.get("inference_timings_ms")
            if not isinstance(raw_timings, dict):
                continue
            timed_records += 1
            for name in self._INFERENCE_OUTCOME_DURATION_FIELDS:
                value = raw_timings.get(name)
                if (
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and math.isfinite(value)
                    and value >= 0
                ):
                    durations[name].append(float(value))
            age = raw_timings.get("parent_record_read_complete_to_inference_complete")
            maximum_age = raw_timings.get("maximum_frame_age")
            if (
                isinstance(age, (int, float))
                and not isinstance(age, bool)
                and math.isfinite(age)
                and age >= 0
                and isinstance(maximum_age, (int, float))
                and not isinstance(maximum_age, bool)
                and math.isfinite(maximum_age)
                and maximum_age >= 0
            ):
                deadline_overruns.append(max(0.0, float(age) - float(maximum_age)))
        return {
            "scope": "all_retained_run_records_not_a_measured_baseline",
            "records_received": self._record_count,
            "retained_records": len(self._records),
            "retention_limit": self._records.maxlen,
            "records_truncated": self._record_count > len(self._records),
            "stale_outcomes": len(stale_records),
            "timed_stale_outcomes": timed_records,
            "stage_durations_ms": {
                name: {
                    "count": len(values),
                    "p50": percentile(values, 0.50),
                    "p95": percentile(values, 0.95),
                    "p99": percentile(values, 0.99),
                }
                for name, values in durations.items()
            },
            "deadline_overrun_ms": {
                "count": len(deadline_overruns),
                "p50": percentile(deadline_overruns, 0.50),
                "p95": percentile(deadline_overruns, 0.95),
                "p99": percentile(deadline_overruns, 0.99),
            },
        }


class RustLiveChildResourceSampler:
    """Sample a supervised Rust child off the event loop with bounded retention."""

    def __init__(
        self,
        recorder: AsyncAuditRecorder,
        *,
        process_id: int,
        interval_seconds: float,
        maximum_samples: int,
        process_factory: Callable[[int], Any] | None = None,
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
        self._recorder = recorder
        self._process_id = process_id
        self._interval_seconds = interval_seconds
        self._samples: deque[dict[str, Any]] = deque(maxlen=maximum_samples)
        self._sample_count = 0
        self._available_sample_count = 0
        self._unavailable_sample_count = 0
        self._process: Any = None
        self._unavailable_reason: str | None = None
        self._stop = threading.Event()
        self._samples_lock = threading.Lock()
        self._thread: threading.Thread | None = None
        try:
            if process_factory is None:
                import psutil

                process_factory = psutil.Process
            self._process = process_factory(process_id)
            self._process.cpu_percent(interval=None)
        except Exception as error:
            self._unavailable_reason = f"{type(error).__name__}: {error}"

    def start(self) -> None:
        """Begin periodic sampling after the measured window has warmed up."""
        if self._thread is not None:
            raise RuntimeError("resource sampler has already started")
        self._thread = threading.Thread(
            target=self._run,
            name=f"rust-live-resources-{self._process_id}",
            daemon=True,
        )
        self._thread.start()

    def close(self) -> None:
        """Stop sampling promptly before the supervised process is terminated."""
        self._stop.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=max(0.1, self._interval_seconds * 2))

    def summary(self) -> dict[str, Any]:
        """Summarize retained CPU and RSS samples, exposing retention loss."""
        with self._samples_lock:
            samples = list(self._samples)
            sample_count = self._sample_count
            available_sample_count = self._available_sample_count
            unavailable_sample_count = self._unavailable_sample_count
        summary_status = (
            "unavailable"
            if available_sample_count == 0
            else ("partial" if unavailable_sample_count > 0 else "available")
        )
        unavailable_reason = self._unavailable_reason
        if not samples and unavailable_reason is None:
            unavailable_reason = (
                "no samples were collected during the measurement window"
            )
        cpu_percent = [
            float(sample["cpu_percent"])
            for sample in samples
            if isinstance(sample.get("cpu_percent"), (int, float))
            and not isinstance(sample.get("cpu_percent"), bool)
            and math.isfinite(sample["cpu_percent"])
        ]
        cpu_times = [
            float(sample["cpu_time_seconds"])
            for sample in samples
            if isinstance(sample.get("cpu_time_seconds"), (int, float))
            and not isinstance(sample.get("cpu_time_seconds"), bool)
            and math.isfinite(sample["cpu_time_seconds"])
        ]
        memory = [
            int(sample["resident_memory_bytes"])
            for sample in samples
            if isinstance(sample.get("resident_memory_bytes"), int)
            and not isinstance(sample.get("resident_memory_bytes"), bool)
        ]
        return {
            "status": summary_status,
            "unavailable_reason": unavailable_reason,
            "process_role": "rust_live_inference_child",
            "process_id": self._process_id,
            "sample_count": sample_count,
            "available_sample_count": available_sample_count,
            "unavailable_sample_count": unavailable_sample_count,
            "retained_sample_count": len(samples),
            "retention_limit": self._samples.maxlen,
            "samples_truncated": sample_count > len(samples),
            "cpu_percent": {
                "count": len(cpu_percent),
                "p50": percentile(cpu_percent, 0.50),
                "p95": percentile(cpu_percent, 0.95),
                "p99": percentile(cpu_percent, 0.99),
                "maximum": max(cpu_percent) if cpu_percent else None,
            },
            "cpu_time_seconds": {
                "first": cpu_times[0] if cpu_times else None,
                "last": cpu_times[-1] if cpu_times else None,
                "delta": cpu_times[-1] - cpu_times[0] if len(cpu_times) > 1 else None,
            },
            "resident_memory_bytes": {
                "count": len(memory),
                "first": memory[0] if memory else None,
                "last": memory[-1] if memory else None,
                "minimum": min(memory) if memory else None,
                "maximum": max(memory) if memory else None,
                "delta": memory[-1] - memory[0] if len(memory) > 1 else None,
            },
        }

    def _run(self) -> None:
        while not self._stop.is_set():
            self._sample_once()
            if self._stop.wait(self._interval_seconds):
                return

    def _sample_once(self) -> None:
        sampled_ns = time.perf_counter_ns()
        event: dict[str, Any] = {
            "event_type": "process_resource_sample",
            "sampled_process_role": "rust_live_inference_child",
            "process_id": self._process_id,
            "host_id": socket.gethostname(),
            "event_monotonic_ns": sampled_ns,
            "sampled_monotonic_ns": sampled_ns,
        }
        if self._process is None:
            event.update(
                {
                    "resource_measurement_status": "unavailable",
                    "resource_measurement_error": self._unavailable_reason,
                    "cpu_percent": None,
                    "cpu_time_seconds": None,
                    "resident_memory_bytes": None,
                }
            )
        else:
            try:
                cpu_percent = self._process.cpu_percent(interval=None)
                memory_bytes = self._process.memory_info().rss
                cpu_times = self._process.cpu_times()
                event.update(
                    {
                        "resource_measurement_status": "available",
                        "cpu_percent": cpu_percent,
                        "cpu_time_seconds": float(cpu_times.user + cpu_times.system),
                        "resident_memory_bytes": memory_bytes,
                    }
                )
            except Exception as error:
                message = f"{type(error).__name__}: {error}"
                event.update(
                    {
                        "resource_measurement_status": "unavailable",
                        "resource_measurement_error": message,
                        "cpu_percent": None,
                        "cpu_time_seconds": None,
                        "resident_memory_bytes": None,
                    }
                )
                if self._unavailable_reason is None:
                    self._unavailable_reason = message
        with self._samples_lock:
            self._sample_count += 1
            if event["resource_measurement_status"] == "available":
                self._available_sample_count += 1
            else:
                self._unavailable_sample_count += 1
            self._samples.append(event)
        self._recorder.record(event)


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
    """Correlate monotonic timestamps only inside one verified OS clock domain."""
    publications = {
        (event.get("domain"), event.get("stream_sequence")): event.get(
            "event_monotonic_ns"
        )
        for event in server_events
        if event.get("event_type") in {"state_published", "trajectory_published"}
        and isinstance(event.get("stream_sequence"), int)
    }
    deliveries = {
        (
            event.get("client_id"),
            event.get("domain"),
            event.get("stream_sequence"),
        ): event.get("event_monotonic_ns")
        for event in server_events
        if event.get("event_type") == "websocket_delivery_complete"
        and event.get("domain") in {"state", "trajectory"}
        and isinstance(event.get("stream_sequence"), int)
    }
    state_publication_to_receipt: list[float] = []
    trajectory_publication_to_receipt: list[float] = []
    socket_to_receipt: list[float] = []
    client_clock_domains = {
        event.get("clock_domain_id")
        for event in client_events
        if isinstance(event.get("clock_domain_id"), str)
        and event.get("clock_domain_id")
    }
    server_clock_domains = {
        event.get("clock_domain_id")
        for event in server_events
        if isinstance(event.get("clock_domain_id"), str)
        and event.get("clock_domain_id")
    }
    if not server_events and server_clock_domain_id:
        # Instrumentation-off baselines intentionally have no server event
        # trace. The read-only status endpoint still reports the verified OS
        # boot clock identity, allowing the client to measure source-to-receipt
        # time without starting any server-side audit workers.
        server_clock_domains.add(server_clock_domain_id)
    clocks_comparable = (
        len(client_clock_domains) == 1
        and len(server_clock_domains) == 1
        and client_clock_domains == server_clock_domains
    )
    source_to_state_publication = [
        (event["event_monotonic_ns"] - event["source_publication_monotonic_ns"]) / 1e6
        for event in server_events
        if event.get("event_type") == "state_published"
        and clocks_comparable
        and isinstance(event.get("event_monotonic_ns"), int)
        and isinstance(event.get("source_publication_monotonic_ns"), int)
        and event["source_publication_monotonic_ns"] > 0
        and event["event_monotonic_ns"] >= event["source_publication_monotonic_ns"]
    ]
    source_to_client_receipt = [
        float(event["source_publication_to_client_receipt_ms"])
        for event in client_events
        if event.get("event_type") == "state_received"
        and clocks_comparable
        and isinstance(
            event.get("source_publication_to_client_receipt_ms"), (int, float)
        )
        and math.isfinite(event["source_publication_to_client_receipt_ms"])
    ]
    component_ages: dict[str, list[float]] = {}
    clock_name = time.get_clock_info("perf_counter").implementation
    delivery_domain_by_event = {
        "state_received": "state",
        "trajectory_received": "trajectory",
    }
    for event in client_events:
        client_event_type = event.get("event_type")
        if not isinstance(client_event_type, str):
            continue
        domain = delivery_domain_by_event.get(client_event_type)
        if domain is None:
            continue
        sequence = event.get("stream_sequence")
        received_ns = event.get("receive_monotonic_ns")
        if not isinstance(sequence, int) or not isinstance(received_ns, int):
            continue
        published_ns = publications.get((domain, sequence))
        if (
            clocks_comparable
            and isinstance(published_ns, int)
            and received_ns >= published_ns
        ):
            latency_ms = (received_ns - published_ns) / 1e6
            if domain == "state":
                state_publication_to_receipt.append(latency_ms)
            else:
                trajectory_publication_to_receipt.append(latency_ms)
        sent_ns = deliveries.get((event.get("client_id"), domain, sequence))
        if clocks_comparable and isinstance(sent_ns, int) and received_ns >= sent_ns:
            socket_to_receipt.append((received_ns - sent_ns) / 1e6)
    received_sequences = {
        event.get("stream_sequence")
        for event in client_events
        if event.get("event_type") == "state_received"
    }
    for event in server_events:
        if (
            event.get("event_type") != "state_published"
            or event.get("stream_sequence") not in received_sequences
        ):
            continue
        ages = event.get("component_age_seconds")
        if not isinstance(ages, dict):
            continue
        for component, age in ages.items():
            if isinstance(age, (int, float)) and math.isfinite(age):
                component_ages.setdefault(str(component), []).append(float(age) * 1000)

    def summarize(values: list[float]) -> dict[str, Any]:
        return {
            "count": len(values),
            "p50": percentile(values, 0.50),
            "p95": percentile(values, 0.95),
            "p99": percentile(values, 0.99),
        }

    state_delivery = summarize(state_publication_to_receipt)
    source_to_state_summary = summarize(source_to_state_publication)
    source_to_client_summary = summarize(source_to_client_receipt)

    def deadline_misses(values: list[float]) -> dict[str, Any]:
        return {
            "sample_count": len(values),
            "missed_deadlines": sum(value > deadline_budget_ms for value in values),
        }

    return {
        "clock": clock_name,
        "clock_domain": (
            f"verified OS monotonic clock domain: {next(iter(client_clock_domains))}"
            if clocks_comparable
            else "not comparable: client and server lack one shared verified OS monotonic clock domain"
        ),
        "clocks_comparable": clocks_comparable,
        "deadline_budget_ms": deadline_budget_ms,
        "source_publication_to_state_ms": source_to_state_summary,
        "source_publication_to_client_receipt_ms": source_to_client_summary,
        "state_publication_to_receipt_ms": state_delivery,
        "deadline_misses": {
            "source_publication_to_state": deadline_misses(source_to_state_publication),
            "source_publication_to_client_receipt": deadline_misses(
                source_to_client_receipt
            ),
            "state_publication_to_receipt": deadline_misses(
                state_publication_to_receipt
            ),
        },
        "socket_send_to_receipt_ms": summarize(socket_to_receipt),
        "trajectory_publication_to_receipt_ms": summarize(
            trajectory_publication_to_receipt
        ),
        "trajectory_socket_send_to_receipt_ms": summarize(
            [
                (
                    event["receive_monotonic_ns"]
                    - deliveries[
                        event.get("client_id"),
                        "trajectory",
                        event["stream_sequence"],
                    ]
                )
                / 1e6
                for event in client_events
                if event.get("event_type") == "trajectory_received"
                and clocks_comparable
                and isinstance(event.get("receive_monotonic_ns"), int)
                and isinstance(
                    deliveries.get(
                        (
                            event.get("client_id"),
                            "trajectory",
                            event.get("stream_sequence"),
                        )
                    ),
                    int,
                )
                and event["receive_monotonic_ns"]
                >= deliveries[
                    event.get("client_id"),
                    "trajectory",
                    event["stream_sequence"],
                ]
            ]
        ),
        "component_observation_age_at_publication_ms": {
            component: summarize(values) for component, values in component_ages.items()
        },
    }


def _server_measurement_summary(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize bounded server trace evidence without inventing missing values."""

    def values(
        event_type: str, field: str, *, domain: str | None = None
    ) -> list[float]:
        output = []
        for event in events:
            value = event.get(field)
            if (
                event.get("event_type") == event_type
                and (domain is None or event.get("domain") == domain)
                and isinstance(value, (int, float))
                and math.isfinite(value)
            ):
                output.append(float(value))
        return output

    def values_for_event_types(
        event_types: set[str], field: str, *, domain: str | None = None
    ) -> list[float]:
        """Collect one timing across every outcome that performed the work."""
        output = []
        for event in events:
            value = event.get(field)
            if (
                event.get("event_type") in event_types
                and (domain is None or event.get("domain") == domain)
                and isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(value)
            ):
                output.append(float(value))
        return output

    def stats(samples: list[float]) -> dict[str, Any]:
        return {
            "count": len(samples),
            "p50": percentile(samples, 0.50),
            "p95": percentile(samples, 0.95),
            "p99": percentile(samples, 0.99),
            "mean": sum(samples) / len(samples) if samples else None,
        }

    def unique_observations(event_type: str) -> tuple[int, int]:
        source_keys = []
        total = 0
        for event in events:
            if event.get("event_type") != event_type:
                continue
            total += 1
            frame_number = event.get("source_frame_number")
            if isinstance(frame_number, int) and not isinstance(frame_number, bool):
                source_keys.append(
                    (
                        event.get("vision_instance_id"),
                        event.get("vision_process_id", event.get("process_id")),
                        event.get("source_process_id"),
                        frame_number,
                    )
                )
        return len(set(source_keys)), total

    processed_unique, processed_total = unique_observations("vision_frame_complete")
    integration_unique, integration_total = unique_observations(
        "integration_observation_processed"
    )
    published_unique, published_total = unique_observations("state_published")
    resource_cpu = values("process_resource_sample", "cpu_percent")
    process_cpu_time = values("process_resource_sample", "cpu_time_seconds")
    resource_memory = values("process_resource_sample", "resident_memory_bytes")
    resource_samples_by_process: dict[str, list[dict[str, Any]]] = {}
    for event in events:
        if event.get("event_type") != "process_resource_sample":
            continue
        process_key = (
            f"{event.get('host_id', 'unknown')}:{event.get('process_id', 'unknown')}"
        )
        resource_samples_by_process.setdefault(process_key, []).append(event)
    resources_by_process = {}
    for process_key, samples in resource_samples_by_process.items():
        cpu = [
            float(event["cpu_percent"])
            for event in samples
            if isinstance(event.get("cpu_percent"), (int, float))
            and math.isfinite(event["cpu_percent"])
        ]
        cpu_time = [
            float(event["cpu_time_seconds"])
            for event in samples
            if isinstance(event.get("cpu_time_seconds"), (int, float))
            and math.isfinite(event["cpu_time_seconds"])
        ]
        memory = [
            int(event["resident_memory_bytes"])
            for event in samples
            if isinstance(event.get("resident_memory_bytes"), int)
        ]
        timed_memory_samples = [
            (int(event["event_monotonic_ns"]), int(event["resident_memory_bytes"]))
            for event in samples
            if isinstance(event.get("event_monotonic_ns"), int)
            and isinstance(event.get("resident_memory_bytes"), int)
        ]
        memory_span_seconds = (
            max(0, timed_memory_samples[-1][0] - timed_memory_samples[0][0]) / 1e9
            if len(timed_memory_samples) > 1
            else None
        )
        memory_change_bytes = (
            timed_memory_samples[-1][1] - timed_memory_samples[0][1]
            if len(timed_memory_samples) > 1
            else None
        )
        exporter_status = next(
            (
                event["audit_export_status"]
                for event in reversed(samples)
                if isinstance(event.get("audit_export_status"), dict)
            ),
            None,
        )
        resources_by_process[process_key] = {
            "sample_count": len(samples),
            "cpu_percent": stats(cpu),
            "cpu_time_seconds": {
                "first": cpu_time[0] if cpu_time else None,
                "last": cpu_time[-1] if cpu_time else None,
                "delta": cpu_time[-1] - cpu_time[0] if len(cpu_time) > 1 else None,
            },
            "resident_memory_bytes": {
                "first": memory[0] if memory else None,
                "last": memory[-1] if memory else None,
                "minimum": min(memory) if memory else None,
                "maximum": max(memory) if memory else None,
                "change_first_to_last": memory_change_bytes,
                "observed_span_seconds": memory_span_seconds,
                "change_bytes_per_second": (
                    memory_change_bytes / memory_span_seconds
                    if memory_change_bytes is not None
                    and memory_span_seconds is not None
                    and memory_span_seconds > 0
                    else None
                ),
            },
            "event_export_status": exporter_status,
        }
    event_loop_lag = values("api_event_loop_sample", "event_loop_lag_seconds")
    outstanding_tasks = values("api_event_loop_sample", "outstanding_tasks")
    component_status_counts: dict[str, dict[str, int]] = {}
    for event in events:
        if event.get("event_type") != "vision_frame_complete":
            continue
        component_status = event.get("component_observation_status")
        if not isinstance(component_status, dict):
            continue
        for component, status in component_status.items():
            statuses = component_status_counts.setdefault(str(component), {})
            status_name = str(status)
            statuses[status_name] = statuses.get(status_name, 0) + 1
    vision_queue_depths = values("vision_queue_frame_dropped", "queue_depth_after_drop")
    processing_queue_depths = values("vision_frame_complete", "processing_queue_depth")
    stage_samples: dict[str, dict[str, list[float]]] = {}
    for event in events:
        if event.get("event_type") != "vision_frame_complete":
            continue
        for metric, field in (
            ("inclusive", "inclusive_stages_seconds"),
            ("exclusive", "exclusive_stages_seconds"),
        ):
            stages = event.get(field)
            if not isinstance(stages, dict):
                continue
            for stage, duration in stages.items():
                if (
                    isinstance(duration, bool)
                    or not isinstance(duration, (int, float))
                    or not math.isfinite(duration)
                    or duration < 0
                ):
                    continue
                stage_samples.setdefault(str(stage), {}).setdefault(metric, []).append(
                    float(duration)
                )
    stage_timings = {
        stage: {
            metric: stats(values_for_stage)
            for metric, values_for_stage in timing.items()
        }
        for stage, timing in stage_samples.items()
    }
    ranked_exclusive_stages: list[dict[str, Any]] = []
    for stage, timing in stage_samples.items():
        exclusive_samples = timing.get("exclusive", [])
        if not exclusive_samples:
            continue
        ranked_exclusive_stages.append(
            {
                "stage": stage,
                "total_exclusive_seconds": sum(exclusive_samples),
                "exclusive": stats(exclusive_samples),
                "inclusive": stats(timing.get("inclusive", [])),
            }
        )
    ranked_exclusive_stages.sort(
        key=lambda item: float(item["total_exclusive_seconds"]), reverse=True
    )
    total_exclusive_seconds = sum(
        float(item["total_exclusive_seconds"]) for item in ranked_exclusive_stages
    )
    for item in ranked_exclusive_stages:
        item["share_of_observed_exclusive_stage_seconds"] = (
            item["total_exclusive_seconds"] / total_exclusive_seconds
            if total_exclusive_seconds
            else None
        )
    trajectory_timing_outcomes = {
        "trajectory_prediction_published",
        "trajectory_prediction_failed",
        "trajectory_prediction_discarded_stale",
        "trajectory_publication_failed",
    }
    prediction_outcome_counts = {
        event_type: sum(event.get("event_type") == event_type for event in events)
        for event_type in sorted(trajectory_timing_outcomes)
    }
    source_publication_to_state_ms = [
        (event["event_monotonic_ns"] - event["source_publication_monotonic_ns"]) / 1e6
        for event in events
        if event.get("event_type") == "state_published"
        and isinstance(event.get("event_monotonic_ns"), int)
        and isinstance(event.get("source_publication_monotonic_ns"), int)
        and event["source_publication_monotonic_ns"] > 0
        and event["event_monotonic_ns"] >= event["source_publication_monotonic_ns"]
    ]
    return {
        "record_count_semantics": (
            "Counts below are derived from events retained by the asynchronous audit "
            "exporter. High-frequency pipeline and publication events are sampled; "
            "export queue drops and export limits can also reduce these counts. "
            "Use the headless subscriber's state accumulator for delivered-state "
            "throughput and repeat counts."
        ),
        "vision": {
            "recorded_frame_detail_events": processed_total,
            "recorded_unique_source_observations_in_frame_details": processed_unique,
            "recorded_duplicate_or_reused_frame_details": max(
                0, processed_total - processed_unique
            ),
            "recorded_component_observation_status_counts": component_status_counts,
            "service_time_seconds": stats(
                values("vision_frame_complete", "service_time_seconds")
            ),
            "queue_wait_seconds": stats(
                values("vision_frame_complete", "queue_delay_seconds")
            ),
            "shared_memory_read_copy_seconds": stats(
                values("vision_frame_complete", "shared_memory_read_copy_seconds")
            ),
            "stage_timings_seconds": stage_timings,
            "exclusive_stage_work_ranked": ranked_exclusive_stages,
            "recorded_source_frame_rejection_events": sum(
                event.get("event_type") == "vision_source_frame_rejected"
                for event in events
            ),
            "recorded_source_restart_events": sum(
                event.get("event_type") == "vision_source_restart_detected"
                for event in events
            ),
            "recorded_vision_queue_drop_events": sum(
                event.get("event_type") == "vision_queue_frame_dropped"
                for event in events
            ),
            "recorded_vision_queue_depth_after_drop_max": (
                max(vision_queue_depths) if vision_queue_depths else None
            ),
            "sampled_processing_queue_depth": stats(processing_queue_depths),
            "sampled_processing_queue_depth_max": (
                max(processing_queue_depths) if processing_queue_depths else None
            ),
        },
        "integration": {
            "recorded_observation_detail_events": integration_total,
            "recorded_unique_source_observations_in_observation_details": integration_unique,
            "recorded_observation_rejection_events": sum(
                event.get("event_type") == "integration_observation_rejected"
                for event in events
            ),
            "coordinate_conversion_seconds": stats(
                values(
                    "integration_observation_processed", "coordinate_conversion_seconds"
                )
            ),
            "core_update_seconds": stats(
                values("integration_observation_processed", "core_update_seconds")
            ),
            "integration_lock_wait_seconds": stats(
                values(
                    "integration_observation_processed",
                    "integration_lock_wait_seconds",
                )
            ),
        },
        "state_delivery": {
            "recorded_state_publication_detail_events": published_total,
            "recorded_unique_source_observations_in_publication_details": published_unique,
            "recorded_repeated_source_observation_publication_details": max(
                0, published_total - published_unique
            ),
            "serialization_seconds": stats(
                values(
                    "websocket_message_serialized",
                    "serialization_seconds",
                    domain="state",
                )
            ),
            "source_publication_to_state_ms": stats(source_publication_to_state_ms),
            "outbound_queue_wait_seconds": stats(
                values(
                    "websocket_delivery_complete", "queue_wait_seconds", domain="state"
                )
            ),
            "outbound_queue_depth_before_enqueue": stats(
                values(
                    "websocket_message_serialized",
                    "queue_depth_before_enqueue",
                    domain="state",
                )
            ),
            "recorded_coalesced_update_events": sum(
                event.get("event_type") == "websocket_update_coalesced"
                and event.get("domain") == "state"
                for event in events
            ),
            "recorded_dropped_update_events": sum(
                event.get("event_type") == "websocket_update_dropped"
                and event.get("domain") == "state"
                for event in events
            ),
            "socket_send_seconds": stats(
                values(
                    "websocket_delivery_complete", "socket_send_seconds", domain="state"
                )
            ),
            "payload_bytes": stats(
                values("websocket_delivery_complete", "payload_bytes", domain="state")
            ),
        },
        "video_capture": {
            "camera_read_seconds": stats(
                values("video_frame_published", "camera_read_seconds")
            ),
            "camera_frame_queue_retrieval_seconds": stats(
                values("video_frame_published", "camera_frame_queue_retrieval_seconds")
            ),
            "undistortion_seconds": stats(
                values("video_frame_published", "undistortion_seconds")
            ),
            "shared_memory_publication_seconds": stats(
                values("video_frame_published", "shared_memory_publication_seconds")
            ),
            "camera_capture_queue_frames_skipped": sum(
                values("video_frame_published", "camera_capture_queue_frames_skipped")
            ),
            "camera_total_dropped_frames_last": max(
                values("video_frame_published", "camera_total_dropped_frames"),
                default=None,
            ),
        },
        "trajectory": {
            "published_messages": sum(
                event.get("event_type") == "trajectory_published" for event in events
            ),
            "requests_coalesced": sum(
                event.get("event_type") == "trajectory_request_coalesced"
                for event in events
            ),
            "predictions_discarded_stale": sum(
                event.get("event_type") == "trajectory_prediction_discarded_stale"
                for event in events
            ),
            "prediction_outcome_counts": prediction_outcome_counts,
            "prediction_outcome_event_count": sum(prediction_outcome_counts.values()),
            "prediction_queue_wait_seconds": stats(
                values_for_event_types(
                    trajectory_timing_outcomes, "prediction_queue_wait_seconds"
                )
            ),
            "calculator_lock_wait_seconds": stats(
                values_for_event_types(
                    trajectory_timing_outcomes, "calculator_lock_wait_seconds"
                )
            ),
            "executor_wait_seconds": stats(
                values_for_event_types(
                    trajectory_timing_outcomes, "executor_wait_seconds"
                )
            ),
            "physics_compute_seconds": stats(
                values_for_event_types(
                    trajectory_timing_outcomes, "physics_compute_seconds"
                )
            ),
            "trajectory_publication_seconds": stats(
                values_for_event_types(
                    trajectory_timing_outcomes, "trajectory_publication_seconds"
                )
            ),
        },
        "resources": {
            "cpu_percent": stats(resource_cpu),
            "process_cpu_time_seconds": {
                "count": len(process_cpu_time),
                "first": process_cpu_time[0] if process_cpu_time else None,
                "last": process_cpu_time[-1] if process_cpu_time else None,
                "delta": (
                    process_cpu_time[-1] - process_cpu_time[0]
                    if len(process_cpu_time) > 1
                    else None
                ),
            },
            "resident_memory_bytes": {
                "count": len(resource_memory),
                "first": resource_memory[0] if resource_memory else None,
                "last": resource_memory[-1] if resource_memory else None,
                "minimum": min(resource_memory) if resource_memory else None,
                "maximum": max(resource_memory) if resource_memory else None,
            },
            "event_loop_lag_seconds": stats(event_loop_lag),
            "outstanding_tasks": stats(outstanding_tasks),
            "by_process": resources_by_process,
            "accelerator_metrics": "unavailable unless separately exposed by detector backend",
            "allocation_counts": "unavailable; RSS is a process memory proxy",
        },
        "viewers": {
            "encoded_frames": sum(
                event.get("event_type") == "viewer_frame_encoded" for event in events
            ),
            "jpeg_encode_seconds": stats(
                values("viewer_frame_encoded", "jpeg_encode_seconds", domain="viewers")
            ),
            "frame_read_seconds": stats(
                values("viewer_frame_encoded", "frame_read_seconds", domain="viewers")
            ),
            "send_backpressure_seconds": stats(
                values(
                    "viewer_send_backpressure",
                    "send_backpressure_seconds",
                    domain="viewers",
                )
            ),
        },
    }


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
) -> dict[str, Any] | None:
    """Compare instrumentation-on runs with an instrumentation-off report."""
    if baseline_path is None:
        return None
    try:
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read baseline summary {baseline_path}: {error}")

    def average(runs: list[dict[str, Any]], field: str) -> float | None:
        values = [
            float(run[field])
            for run in runs
            if isinstance(run.get(field), (int, float)) and math.isfinite(run[field])
        ]
        return sum(values) / len(values) if values else None

    def average_source_delivery_p95(runs: list[dict[str, Any]]) -> float | None:
        values = [
            float(run["source_publication_to_client_receipt_ms"]["p95"])
            for run in runs
            if isinstance(run.get("source_publication_to_client_receipt_ms"), dict)
            and isinstance(
                run["source_publication_to_client_receipt_ms"].get("p95"),
                (int, float),
            )
        ]
        return sum(values) / len(values) if values else None

    baseline_runs = baseline.get("repetitions", [])
    current_throughput = average(repetitions, "unique_observations_per_second")
    baseline_throughput = average(baseline_runs, "unique_observations_per_second")
    current_p95 = average_source_delivery_p95(repetitions)
    baseline_p95 = average_source_delivery_p95(baseline_runs)
    throughput_change = (
        (current_throughput - baseline_throughput) / baseline_throughput * 100
        if current_throughput is not None and baseline_throughput
        else None
    )
    p95_change = (
        (current_p95 - baseline_p95) / baseline_p95 * 100
        if current_p95 is not None and baseline_p95
        else None
    )

    current_configuration = metadata.get("effective_configuration", {})
    baseline_metadata = baseline.get("metadata", {})
    baseline_configuration = baseline_metadata.get("effective_configuration", {})
    comparable_keys = (
        "camera",
        "detection",
        "processing",
        "vision_performance",
        "video",
        "calibration",
        "websocket_delivery",
        "stream_quality",
    )
    same_configuration = all(
        current_configuration.get(key) == baseline_configuration.get(key)
        for key in comparable_keys
    )
    activation_keys = {
        "enabled",
        "state_enabled",
        "trajectory_enabled",
        "viewers_enabled",
    }

    def audit_settings(configuration: dict[str, Any]) -> dict[str, Any]:
        audit = configuration.get("performance_audit", {})
        if not isinstance(audit, dict):
            return {}
        return {
            key: value for key, value in audit.items() if key not in activation_keys
        }

    same_audit_configuration = audit_settings(current_configuration) == audit_settings(
        baseline_configuration
    )

    def workload_signature(run: dict[str, Any]) -> tuple[Any, ...]:
        video_viewer = run.get("video_viewer")
        video_url = video_viewer.get("url") if isinstance(video_viewer, dict) else None
        streams = run.get("streams")
        normalized_streams = tuple(sorted(streams)) if isinstance(streams, list) else ()
        return (
            normalized_streams,
            video_url,
            run.get("subscriber_delay_ms"),
            run.get("warmup_seconds"),
            run.get("requested_measured_seconds"),
        )

    current_workloads = {workload_signature(run) for run in repetitions}
    baseline_workloads = {workload_signature(run) for run in baseline_runs}
    same_workload = (
        len(current_workloads) == 1
        and len(baseline_workloads) == 1
        and current_workloads == baseline_workloads
        and all(bool(workload[0]) for workload in current_workloads)
    )
    reference_workload = same_workload and all(
        workload[0] == ("state",) and workload[1] is None and workload[2] == 0.0
        for workload in current_workloads
    )
    current_collector_settings = {
        json.dumps(run.get("collector_settings"), sort_keys=True)
        for run in repetitions
        if isinstance(run.get("collector_settings"), dict)
    }
    baseline_collector_settings = {
        json.dumps(run.get("collector_settings"), sort_keys=True)
        for run in baseline_runs
        if isinstance(run.get("collector_settings"), dict)
    }
    same_collector_settings = (
        len(current_collector_settings) == 1
        and len(baseline_collector_settings) == 1
        and current_collector_settings == baseline_collector_settings
    )
    model_hash = metadata.get("model", {}).get("sha256")
    baseline_model_hash = baseline_metadata.get("model", {}).get("sha256")
    same_model = bool(
        isinstance(model_hash, str) and model_hash and model_hash == baseline_model_hash
    )
    input_hash = metadata.get("configured_video_input", {}).get("sha256")
    baseline_input_hash = baseline_metadata.get("configured_video_input", {}).get(
        "sha256"
    )
    same_input = bool(
        isinstance(input_hash, str) and input_hash and input_hash == baseline_input_hash
    )
    same_revision = bool(
        isinstance(metadata.get("revision"), str)
        and metadata.get("revision")
        and metadata.get("revision") == baseline_metadata.get("revision")
    )
    same_dirty_worktree = bool(
        isinstance(metadata.get("dirty_worktree_sha256"), str)
        and metadata.get("dirty_worktree_sha256")
        and metadata.get("dirty_worktree_sha256")
        == baseline_metadata.get("dirty_worktree_sha256")
    )
    current_domains = [run.get("server_audit_domains") for run in repetitions]
    baseline_domains = [run.get("server_audit_domains") for run in baseline_runs]
    baseline_instrumentation_off = bool(baseline_domains) and all(
        isinstance(domains, dict) and not any(domains.values())
        for domains in baseline_domains
    )
    current_state_only = bool(current_domains) and all(
        domains == {"state": True, "trajectory": False, "viewers": False}
        for domains in current_domains
    )
    same_machine = (
        metadata.get("host") == baseline_metadata.get("host")
        and metadata.get("platform") == baseline_metadata.get("platform")
        and metadata.get("processor") == baseline_metadata.get("processor")
        and metadata.get("hardware") == baseline_metadata.get("hardware")
    )
    enough_data = (
        len(repetitions) >= 3
        and len(baseline_runs) >= 3
        and min(
            [
                run.get("source_publication_to_client_receipt_ms", {}).get("count", 0)
                for run in [*repetitions, *baseline_runs]
            ],
            default=0,
        )
        >= 100
    )
    passes_budget = (
        throughput_change is not None
        and p95_change is not None
        and throughput_change >= -budget_percent
        and p95_change <= budget_percent
    )
    comparable = (
        same_machine
        and baseline_instrumentation_off
        and current_state_only
        and same_revision
        and same_dirty_worktree
        and same_configuration
        and same_audit_configuration
        and same_workload
        and reference_workload
        and same_collector_settings
        and same_model
        and same_input
    )
    return {
        "baseline_summary": str(baseline_path),
        "same_revision": same_revision,
        "same_dirty_worktree": same_dirty_worktree,
        "same_model_hash": same_model,
        "same_input_hash": same_input,
        "same_relevant_configuration": same_configuration,
        "same_audit_configuration": same_audit_configuration,
        "same_collector_settings": same_collector_settings,
        "same_workload": same_workload,
        "reference_state_only_workload": reference_workload,
        "same_machine": same_machine,
        "baseline_instrumentation_off": baseline_instrumentation_off,
        "current_state_audit_only": current_state_only,
        "unique_observation_throughput_change_percent": throughput_change,
        "source_publication_to_client_receipt_p95_change_percent": p95_change,
        "overhead_budget_percent": budget_percent,
        "enough_repetitions_and_samples": enough_data,
        "status": (
            "pass"
            if enough_data and comparable and passes_budget
            else (
                "inconclusive"
                if not enough_data
                or not comparable
                or current_throughput is None
                or baseline_throughput is None
                or current_p95 is None
                or baseline_p95 is None
                else "over_budget"
            )
        ),
        "interpretation": (
            "Instrumentation overhead is within the configured budget."
            if passes_budget
            else "No performance recommendation until longer or better-matched runs meet the acceptance budget."
        ),
    }


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
) -> dict[str, Any]:
    """Connect, warm up, and collect one state stream measurement window."""
    accumulator = StateAccumulator(maximum_samples=history_size)
    trajectory_accumulator = TrajectoryAccumulator(maximum_samples=history_size)
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
            raise

        video_counts_before_measurement = (
            video_load.snapshot() if video_load is not None else None
        )
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

    elapsed = max(0.0, (measured_finished_ns - started_ns) / 1_000_000_000)
    server_end = await asyncio.to_thread(_read_json_url, performance_url)
    audit_end = await asyncio.to_thread(_wait_for_audit_export_drain, audit_url)
    summary = accumulator.summary(elapsed)
    summary.update(trajectory_accumulator.summary(elapsed))
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
) -> dict[str, Any]:
    """Measure one bounded live-camera run through the established supervisor."""
    from backend.perception.rust_live_supervisor import (
        RustLiveAuditIngress,
        RustLiveSupervisor,
    )
    from backend.perception.scene_state_adapter import SceneSnapshotStatePublisher

    factory = supervisor_factory or RustLiveSupervisor
    timing_accumulator = (
        RustLiveTimingAccumulator(history_size) if timing_audit_enabled else None
    )
    timing_ingress = None
    if timing_accumulator is not None:

        def on_timing_record(record: dict[str, Any]) -> bool:
            timing_accumulator.add(record)
            audit_recorder = timing_recorder or recorder
            return audit_recorder.record(
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
    try:
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
                recorder,
                process_id=process_id,
                interval_seconds=resource_sample_interval_seconds,
                maximum_samples=history_size,
            )
            resource_sampler.start()
        elif process_id is not None:
            resource_summary["unavailable_reason"] = (
                "supervisor exposed an invalid child process id"
            )

        measurement_started_ns = broadcaster.begin_measurement()
        if timing_accumulator is not None:
            timing_accumulator.begin_measurement(measurement_started_ns)
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
        if timing_accumulator is not None:
            timing_accumulator.end_measurement(measurement_ended_ns)
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
            if timing_accumulator is not None:
                timing_accumulator.end_measurement(measurement_ended_ns)
        try:
            if resource_sampler is not None:
                resource_sampler.close()
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
                if audit_session_results is not None:
                    audit_session_result = {
                        "repetition": repetition,
                        **timing_ingress_summary,
                    }
                    if timing_accumulator is not None:
                        audit_session_result["stale_inference_timing_diagnostic"] = (
                            timing_accumulator.stale_inference_diagnostic_summary()
                        )
                    audit_session_results.append(audit_session_result)

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
    duration = (measurement_ended_ns - measurement_started_ns) / 1_000_000_000
    state_summary = broadcaster.measurement.summary(duration)
    timing_summary = (
        timing_accumulator.summary(
            broadcaster.measurement.retained_trace,
            state_trace_truncated=state_summary["state_trace_truncated"],
        )
        if timing_accumulator is not None
        else {"status": "disabled"}
    )
    if timing_ingress is not None:
        timing_summary["status"] = (
            "complete"
            if timing_summary.get("status") == "complete"
            and timing_ingress_summary.get("status") == "complete"
            else "partial"
        )
        timing_summary["ingress_status"] = timing_ingress_summary
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
            "CPU utilization and resident memory are sampled only during each measured window. Sample retention is bounded; unavailable measurements remain explicit.",
            "",
            "| Repetition | Status | Samples | CPU p50 / p95 / p99 (%) | CPU time delta (s) | RSS min / max (MiB) | RSS first→last delta (MiB) |",
            "|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for run in report["repetitions"]:
        resources = run.get("process_resources", {})
        cpu = resources.get("cpu_percent", {})
        cpu_time = resources.get("cpu_time_seconds", {})
        memory = resources.get("resident_memory_bytes", {})
        lines.append(
            "| {repetition} | {status} | {samples} | {cpu} | {cpu_delta} | {memory_range} | {memory_delta} |".format(
                repetition=run["repetition"],
                status=resources.get("status", "unavailable"),
                samples=resources.get("sample_count", 0),
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
            "rust_pipeline_timing_event_export": (
                timing_recorder.status()
                if timing_recorder is not None
                else {"status": "disabled"}
            ),
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
            )
            repetition_result["collector_settings"] = {
                "history_size": args.history_size,
                "queue_capacity": args.queue_capacity,
                "sample_rate": args.sample_rate,
                "event_export_limit": args.event_export_limit,
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
