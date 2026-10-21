"""Bounded asynchronous event export for explicitly enabled audit sessions."""

from __future__ import annotations

import json
import logging
import os
import platform
import queue
import random
import socket
import subprocess
import threading
import time
import uuid
from pathlib import Path
from subprocess import run as _run_subprocess
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

logger = logging.getLogger(__name__)


def monotonic_clock_domain_id() -> str | None:
    """Identify the OS boot clock domain used by ``perf_counter``.

    Hostnames are not sufficient evidence for comparing monotonic counters:
    containers on one kernel can have different names, and cloned hosts can
    reuse one. Linux boot IDs and macOS boot-session UUIDs are stable across
    processes sharing the same operating-system monotonic clock. Unsupported
    or inaccessible identities return ``None`` so reports keep durations
    segmented instead of claiming an unverified cross-process latency.
    """
    system = platform.system()
    clock_implementation = time.get_clock_info("perf_counter").implementation
    boot_identity: str | None = None
    if system == "Linux":
        try:
            boot_identity = (
                Path("/proc/sys/kernel/random/boot_id")
                .read_text(encoding="ascii")
                .strip()
            )
        except OSError:
            return None
    elif system == "Darwin":
        try:
            result = _run_subprocess(
                ["/usr/sbin/sysctl", "-n", "kern.bootsessionuuid"],
                check=True,
                capture_output=True,
                text=True,
                timeout=1.0,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        boot_identity = result.stdout.strip()
    else:
        return None
    if not boot_identity:
        return None
    return f"{system}:{boot_identity}:{clock_implementation}"


def audit_enabled_from_config(
    configured: bool, environment_override: str | None
) -> bool:
    """Resolve explicit config or process-local CLI activation."""
    return bool(configured) or (environment_override or "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def audit_domains_from_config(
    configured: bool,
    environment_override: str | None,
    *,
    state_enabled: bool = True,
    trajectory_enabled: bool = False,
    viewers_enabled: bool = False,
) -> dict[str, bool]:
    """Resolve independently enabled audit domains behind the global opt-in."""
    activated = audit_enabled_from_config(configured, environment_override)
    return {
        "state": activated and state_enabled,
        "trajectory": activated and trajectory_enabled,
        "viewers": activated and viewers_enabled,
    }


class AsyncAuditRecorder:
    """Write sampled audit events off the processing thread using a bounded queue.

    When the queue is full, the newest audit event is discarded and counted.
    Audit collection therefore cannot back up the state pipeline.
    """

    def __init__(
        self,
        output_path: str | Path,
        *,
        queue_capacity: int = 2048,
        sample_rate: float = 1.0,
        max_exported_events: int = 200000,
        writer: Callable[[dict[str, Any]], None] | None = None,
        clock_ns: Callable[[], int] = time.perf_counter_ns,
    ) -> None:
        if queue_capacity < 1:
            raise ValueError("queue_capacity must be positive")
        if not 0 < sample_rate <= 1:
            raise ValueError("sample_rate must be in (0, 1]")
        if max_exported_events < 1:
            raise ValueError("max_exported_events must be positive")
        self.run_id = str(uuid.uuid4())
        self.process_id = os.getpid()
        self.host_id = socket.gethostname()
        self.clock_domain_id = monotonic_clock_domain_id()
        self.clock_implementation = time.get_clock_info("perf_counter").implementation
        configured_path = Path(output_path)
        self.output_path = (
            configured_path / self.run_id / "events.jsonl"
            if configured_path.suffix == ""
            else configured_path
        )
        self.sample_rate = sample_rate
        self.max_exported_events = max_exported_events
        self._clock_ns = clock_ns
        self._writer = writer or self._write_json_line
        self._queue: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=queue_capacity)
        self._random = random.Random()
        self._lock = threading.Lock()
        self._sequence = 0
        self.accepted = 0
        self.written = 0
        self.dropped = 0
        self.sampled_out = 0
        self.export_limit_dropped = 0
        self.export_errors = 0
        self._closed = False
        self._stop = threading.Event()
        self._worker = threading.Thread(
            target=self._run, name="performance-audit-exporter", daemon=True
        )
        self._worker.start()

    def record(self, event: Mapping[str, Any]) -> bool:
        """Enqueue one event without waiting for disk or a slow consumer."""
        with self._lock:
            if self._closed:
                return False
            self._sequence += 1
            sequence = self._sequence
            if self._random.random() > self.sample_rate:
                self.sampled_out += 1
                return True
            envelope = {
                **dict(event),
                "schema_version": 1,
                "run_id": self.run_id,
                "process_id": self.process_id,
                "host_id": self.host_id,
                "clock_domain_id": self.clock_domain_id,
                "clock_implementation": self.clock_implementation,
                "event_sequence": sequence,
                "event_monotonic_ns": self._clock_ns(),
            }
            try:
                self._queue.put_nowait(envelope)
            except queue.Full:
                self.dropped += 1
                return False
            self.accepted += 1
        return True

    def status(self) -> dict[str, Any]:
        """Return counts used to disclose sampling and export loss."""
        with self._lock:
            return {
                "run_id": self.run_id,
                "process_id": self.process_id,
                "host_id": self.host_id,
                "clock_domain_id": self.clock_domain_id,
                "clock_implementation": self.clock_implementation,
                "accepted_events": self.accepted,
                "written_events": self.written,
                "dropped_events": self.dropped,
                "sampled_out_events": self.sampled_out,
                "export_limit_dropped_events": self.export_limit_dropped,
                "export_errors": self.export_errors,
                "queue_depth": self._queue.qsize(),
                "queue_capacity": self._queue.maxsize,
                "sample_rate": self.sample_rate,
                "max_exported_events": self.max_exported_events,
                "worker_alive": self._worker.is_alive(),
                "closed": self._closed,
            }

    def close(self, timeout_seconds: float = 5.0) -> None:
        """Drain queued records for a bounded time and stop the worker."""
        with self._lock:
            if not self._closed:
                self._closed = True
                self._stop.set()
        self._worker.join(timeout=max(0.0, timeout_seconds))

    def _run(self) -> None:
        while True:
            if self._stop.is_set() and self._queue.empty():
                return
            try:
                item = self._queue.get(timeout=0.05)
            except queue.Empty:
                continue
            try:
                with self._lock:
                    export_limit_reached = self.written >= self.max_exported_events
                    if export_limit_reached:
                        self.export_limit_dropped += 1
                if export_limit_reached:
                    continue
                self._writer(item)
                with self._lock:
                    self.written += 1
            except Exception:
                with self._lock:
                    self.export_errors += 1
                logger.debug("Performance audit event export failed", exc_info=True)
            finally:
                self._queue.task_done()

    def _write_json_line(self, event: dict[str, Any]) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(event, separators=(",", ":"), default=str))
            output.write("\n")


class AuditResourceSampler:
    """Sample process CPU and resident memory away from the hot path."""

    def __init__(
        self,
        recorder: AsyncAuditRecorder,
        interval_seconds: float,
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError("resource sample interval must be positive")
        self._recorder = recorder
        self._interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._process: Any = None
        try:
            import psutil

            self._process = psutil.Process()
            self._process.cpu_percent(interval=None)
        except Exception:
            logger.debug("Process resource measurements unavailable", exc_info=True)
        self._thread = threading.Thread(
            target=self._run, name="performance-audit-resources", daemon=True
        )
        self._thread.start()

    def close(self) -> None:
        """Stop polling without blocking application shutdown indefinitely."""
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=max(0.1, self._interval_seconds * 2))

    def _run(self) -> None:
        while not self._stop.wait(self._interval_seconds):
            sampled_ns = time.perf_counter_ns()
            if self._process is None:
                self._recorder.record(
                    {
                        "event_type": "process_resource_sample",
                        "resource_measurement_status": "unavailable",
                        "cpu_percent": None,
                        "resident_memory_bytes": None,
                        "audit_export_status": self._recorder.status(),
                    }
                )
                continue
            try:
                cpu_percent = self._process.cpu_percent(interval=None)
                memory_bytes = self._process.memory_info().rss
                cpu_times = getattr(self._process, "cpu_times", None)
                cpu_time_seconds = None
                if callable(cpu_times):
                    cpu_time = cpu_times()
                    cpu_time_seconds = float(cpu_time.user + cpu_time.system)
                self._recorder.record(
                    {
                        "event_type": "process_resource_sample",
                        "resource_measurement_status": "available",
                        "sampled_monotonic_ns": sampled_ns,
                        "cpu_percent": cpu_percent,
                        "cpu_time_seconds": cpu_time_seconds,
                        "resident_memory_bytes": memory_bytes,
                        "audit_export_status": self._recorder.status(),
                    }
                )
            except Exception:
                logger.debug("Process resource sample failed", exc_info=True)
                self._recorder.record(
                    {
                        "event_type": "process_resource_sample",
                        "resource_measurement_status": "unavailable",
                        "sampled_monotonic_ns": sampled_ns,
                        "cpu_percent": None,
                        "resident_memory_bytes": None,
                        "audit_export_status": self._recorder.status(),
                    }
                )
