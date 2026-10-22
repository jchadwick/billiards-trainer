"""Drain the latest Rust scene and publish it through the retained state stream."""

from __future__ import annotations

import asyncio
import logging
import math
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from backend.perception.scene_state_adapter import (
    SceneSnapshotStatePublisher,
    SceneStateCompatibilityError,
)

if TYPE_CHECKING:
    from backend.perception.scene_ingress import SceneSnapshotIngress

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SceneStateWorkerStats:
    """Bounded counters for scene consumption and compatibility publication."""

    received: int
    published: int
    health_updates_published: int
    ignored_non_observations: int
    incompatible: int
    publication_errors: int
    latest_transport_sequence: int | None
    source_health: str
    last_observation_age_seconds: float | None
    source_silence_timeout_seconds: float


class SceneSnapshotStateWorker:
    """Consume ingress snapshots independently of Core and prediction work.

    The worker must only run when the caller has selected Rust as the sole
    location-state publisher. The loop consumes the ingress' single newest
    pending slot, so slow WebSocket delivery cannot create an unbounded queue
    of stale scenes.
    """

    def __init__(
        self,
        ingress: SceneSnapshotIngress,
        publisher: SceneSnapshotStatePublisher,
        *,
        source_silence_timeout_seconds: float,
    ) -> None:
        if (
            isinstance(source_silence_timeout_seconds, bool)
            or not isinstance(source_silence_timeout_seconds, (int, float))
            or not math.isfinite(source_silence_timeout_seconds)
            or source_silence_timeout_seconds <= 0
        ):
            raise ValueError("source silence timeout must be finite and positive")
        self._ingress = ingress
        self._publisher = publisher
        self._source_silence_timeout_seconds = float(source_silence_timeout_seconds)
        self._task: asyncio.Task[None] | None = None
        self._received = 0
        self._published = 0
        self._health_updates_published = 0
        self._ignored_non_observations = 0
        self._incompatible = 0
        self._publication_errors = 0
        self._latest_transport_sequence: int | None = None
        self._source_silence_notified = False
        self._last_successful_observation_monotonic_ns: int | None = None
        self._first_observation_published = asyncio.Event()

    @property
    def stats(self) -> SceneStateWorkerStats:
        """Return counters without retaining per-snapshot history."""
        last_observation = self._last_successful_observation_monotonic_ns
        age_seconds = (
            max(0.0, (time.monotonic_ns() - last_observation) / 1_000_000_000)
            if last_observation is not None
            else None
        )
        source_health = (
            "awaiting_localization"
            if age_seconds is None
            else (
                "source_silent"
                if self._source_silence_notified
                or age_seconds >= self._source_silence_timeout_seconds
                else "healthy"
            )
        )
        return SceneStateWorkerStats(
            received=self._received,
            published=self._published,
            health_updates_published=self._health_updates_published,
            ignored_non_observations=self._ignored_non_observations,
            incompatible=self._incompatible,
            publication_errors=self._publication_errors,
            latest_transport_sequence=self._latest_transport_sequence,
            source_health=source_health,
            last_observation_age_seconds=age_seconds,
            source_silence_timeout_seconds=self._source_silence_timeout_seconds,
        )

    async def start(self) -> None:
        """Start the ingress before creating the single consumer task."""
        if self._task is not None:
            return
        await self._ingress.start()
        self._task = asyncio.create_task(self._run(), name="rust-scene-state-publisher")

    async def wait_for_first_observation(self) -> None:
        """Wait until a compatible, successful localization reaches the broadcaster."""
        await self._first_observation_published.wait()

    async def close(self) -> None:
        """Cancel consumption and close the loopback scene listener."""
        task = self._task
        self._task = None
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        await self._ingress.close()

    async def _run(self) -> None:
        while True:
            try:
                if (
                    self._last_successful_observation_monotonic_ns is None
                    or self._source_silence_notified
                ):
                    received = await self._ingress.receive_latest()
                else:
                    deadline = (
                        self._last_successful_observation_monotonic_ns / 1_000_000_000
                        + self._source_silence_timeout_seconds
                    )
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        await self._publish_source_silence()
                        continue
                    received = await asyncio.wait_for(
                        self._ingress.receive_latest(), timeout=remaining
                    )
            except asyncio.TimeoutError:
                await self._publish_source_silence()
                continue

            self._received += 1
            snapshot = received.snapshot
            self._latest_transport_sequence = snapshot.get("transport_sequence")
            try:
                public_state_delivery = received.public_state_delivery
                if public_state_delivery is not None:
                    outcome = await self._publisher.publish_native_state(
                        public_state_delivery,
                        received_at=received.received_at,
                        received_monotonic_ns=received.received_monotonic_ns,
                    )
                    if outcome == "localization":
                        self._published += 1
                        self._first_observation_published.set()
                        self._last_successful_observation_monotonic_ns = (
                            received.received_monotonic_ns
                        )
                        self._source_silence_notified = False
                    elif outcome == "health":
                        self._health_updates_published += 1
                    else:
                        self._ignored_non_observations += 1
                    continue

                published = await self._publisher.publish(
                    snapshot,
                    received_at=received.received_at,
                    received_monotonic_ns=received.received_monotonic_ns,
                )
                if not published:
                    health_published = await self._publisher.publish_health(snapshot)
                    if health_published:
                        self._health_updates_published += 1
                        continue
            except SceneStateCompatibilityError as error:
                self._incompatible += 1
                logger.warning(
                    "Rust scene state was not compatible with the retained "
                    "image-coordinate stream: %s",
                    error,
                )
                continue
            except Exception:
                self._publication_errors += 1
                logger.exception("Failed to publish a Rust scene snapshot")
                continue

            if published:
                self._published += 1
                self._first_observation_published.set()
                # The ingress receipt clock is local to this process and uses
                # monotonic time. Non-observation traffic must not refresh this
                # localization freshness deadline.
                self._last_successful_observation_monotonic_ns = (
                    received.received_monotonic_ns
                )
                self._source_silence_notified = False
            else:
                self._ignored_non_observations += 1

    async def _publish_source_silence(self) -> None:
        """Emit one stale update for each interval without a fresh observation."""
        if self._source_silence_notified:
            return
        try:
            published = await self._publisher.publish_source_silence()
            if published:
                self._health_updates_published += 1
        except Exception:
            self._publication_errors += 1
            logger.exception("Failed to publish Rust source-silence state")
        self._source_silence_notified = True
