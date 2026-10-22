"""Map validated Rust scene snapshots to the retained WebSocket state shape.

This is the compatibility boundary described by ADR-0009 and ADR-0013. It
publishes source image measurements in canonical pixel coordinates and carries
calibrated millimeter coordinates as separate, explicitly unit-bearing fields.
It does not wait for Core, prediction, or identity work.

The v1 scene contract does not contain an image-space ball radius. The existing
state contract requires one for rendering and physics compatibility, so this
adapter uses the retained canonical pool-ball radius and marks its provenance
as assumed. It must not be interpreted as a measured radius or used to assess
detector accuracy.
"""

from __future__ import annotations

import math
import re
import time
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal, Protocol

from backend.core.constants_4k import BALL_RADIUS_4K, CANONICAL_HEIGHT, CANONICAL_WIDTH


class SceneStateCompatibilityError(ValueError):
    """A scene cannot be mapped to the legacy image-coordinate contract."""


class GameStateBroadcaster(Protocol):
    """Small surface required from the retained WebSocket broadcaster."""

    async def broadcast_game_state(
        self,
        balls: object,
        cue: dict[str, Any] | None = None,
        table: dict[str, Any] | None = None,
        timestamp: Any = None,
        observation_status: dict[str, Any] | None = None,
        game_type: str | None = None,
    ) -> None: ...


@dataclass(frozen=True)
class SceneStatePayload:
    """Legacy public state fields derived from one complete Rust snapshot."""

    balls: list[dict[str, Any]]
    observation_status: dict[str, Any]
    table: dict[str, Any] | None = None
    game_type: str | None = None


@dataclass(frozen=True)
class SceneStateReadModel:
    """Current public scene state plus gateway-local freshness timestamps.

    ``updated_at`` is when this state or health status was published by the
    gateway. ``localization_received_at`` changes only for a successful
    localization, so a later failure or source-silence notice cannot make old
    ball positions appear fresh.
    """

    payload: SceneStatePayload
    updated_at: datetime
    localization_received_at: datetime | None
    ball_localization_received_at: dict[str, datetime]


class SceneSnapshotStateAdapter:
    """Convert a successful raw-image scene to the existing public state shape.

    The scene's image space must be raw and each selected point must have the
    current source frame's exact dimensions. Rectified points cannot be placed
    on the existing browser preview without a corresponding rectified preview
    or an explicit inverse-distortion map, so those frames fail closed here.
    """

    def adapt(
        self,
        snapshot: Mapping[str, Any],
        *,
        table: dict[str, Any] | None = None,
        game_type: str | None = None,
    ) -> SceneStatePayload | None:
        """Return a public payload for a new successful observation.

        Annotation, health, skipped, degraded, and failed snapshots are not
        location observations and therefore do not publish a fresh state.
        """
        if snapshot.get("schema_version") != 1:
            raise SceneStateCompatibilityError("unsupported scene schema version")
        if (
            snapshot.get("update_kind") != "observation"
            or snapshot.get("outcome") != "success"
        ):
            return None

        source = _mapping(snapshot.get("source"), "source")
        session_id = _identifier(source.get("session_id"), "source.session_id")
        frame_id = _identifier(source.get("frame_id"), "source.frame_id")
        source_width = _positive_dimension(source.get("width_px"), "source.width_px")
        source_height = _positive_dimension(source.get("height_px"), "source.height_px")
        source_timestamp = _mapping(source.get("timestamp"), "source.timestamp")
        if source_timestamp.get("stage") not in {
            "exposure",
            "camera_read_complete",
            "shared_memory_publication",
            "frame_arrival",
        }:
            raise SceneStateCompatibilityError(
                "source timestamp must identify capture or frame publication"
            )
        source_timestamp_ticks = source_timestamp.get("ticks")
        if (
            not isinstance(source_timestamp_ticks, str)
            or re.fullmatch(r"[0-9]+", source_timestamp_ticks) is None
            or source_timestamp.get("unit") != "nanoseconds"
        ):
            raise SceneStateCompatibilityError(
                "source timestamp must contain decimal nanosecond ticks"
            )
        source_timestamp_kind = ":".join(
            str(source_timestamp.get(field))
            for field in ("clock", "domain_id", "stage")
        )
        frame_publication_age, frame_age_status = _frame_age_at_publication(
            source_timestamp,
            snapshot.get("published_at"),
        )

        scale_x = CANONICAL_WIDTH / source_width
        scale_y = CANONICAL_HEIGHT / source_height
        balls_value = snapshot.get("balls")
        if not isinstance(balls_value, list):
            raise SceneStateCompatibilityError("balls must be an array")

        balls: list[dict[str, Any]] = []
        omitted_track_ids: list[str] = []
        for index, raw_ball in enumerate(balls_value):
            ball = _mapping(raw_ball, f"balls[{index}]")
            track_id = _identifier(ball.get("track_id"), f"balls[{index}].track_id")
            status = ball.get("status")
            if status not in {"observed", "tracked", "occluded", "lost"}:
                raise SceneStateCompatibilityError(
                    f"balls[{index}].status is not recognized"
                )

            observation = _mapping(
                ball.get("last_observation"), f"balls[{index}].last_observation"
            )
            measurement_value = ball.get("measurement")
            measurement = (
                _mapping(measurement_value, f"balls[{index}].measurement")
                if measurement_value is not None
                else None
            )
            estimate_value = ball.get("estimate")
            estimate = (
                _mapping(estimate_value, f"balls[{index}].estimate")
                if estimate_value is not None
                else None
            )

            # Measured coordinates always win for an actually observed ball.
            # Otherwise use the explicitly tagged estimate when it has image
            # geometry, falling back to the last measured location if needed.
            if status == "observed":
                point_record = measurement
                position_kind = "measured"
            elif estimate is not None and estimate.get("image_center_px") is not None:
                point_record = estimate
                position_kind = str(estimate.get("kind"))
            else:
                point_record = measurement
                position_kind = "last_observed"

            if point_record is None or point_record.get("image_center_px") is None:
                omitted_track_ids.append(track_id)
                continue
            point = _mapping(
                point_record.get("image_center_px"),
                f"balls[{index}].image_center_px",
            )
            image_space = point_record.get("image_space")
            width = _positive_dimension(
                point_record.get("image_width_px"),
                f"balls[{index}].image_width_px",
            )
            height = _positive_dimension(
                point_record.get("image_height_px"),
                f"balls[{index}].image_height_px",
            )
            if image_space != "raw":
                raise SceneStateCompatibilityError(
                    f"balls[{index}] uses {image_space!r} image coordinates; "
                    "the retained preview requires raw coordinates"
                )
            if (width, height) != (source_width, source_height):
                raise SceneStateCompatibilityError(
                    f"balls[{index}] image dimensions differ from the source frame"
                )
            x = _finite_coordinate(point.get("x"), f"balls[{index}].x")
            y = _finite_coordinate(point.get("y"), f"balls[{index}].y")
            if not (0 <= x <= width and 0 <= y <= height):
                raise SceneStateCompatibilityError(
                    f"balls[{index}] image center is outside its source frame"
                )

            if measurement is not None:
                last_observation_confidence = _probability(
                    measurement.get("detector_confidence"),
                    f"balls[{index}].detector_confidence",
                )
            else:
                last_observation_confidence = None

            # Detector confidence describes a detection, not the current
            # accuracy of an extrapolated or cached point. Keep it separately
            # for auditing and report current position confidence unavailable.
            confidence = last_observation_confidence if status == "observed" else 0.0

            identity_value = ball.get("identity")
            identity = _mapping(identity_value, f"balls[{index}].identity")
            number = identity.get("number")
            if number is not None and (
                isinstance(number, bool)
                or not isinstance(number, int)
                or not 1 <= number <= 15
            ):
                raise SceneStateCompatibilityError(
                    f"balls[{index}].identity.number is invalid"
                )
            if number is None:
                ball_type = "unknown"
            elif number == 8:
                ball_type = "eight"
            elif 9 <= number <= 15:
                ball_type = "stripe"
            else:
                ball_type = "solid"

            table_position = _point_or_none(
                point_record.get("table_center_mm"),
                f"balls[{index}].table_center_mm",
            )
            ball_payload: dict[str, Any] = {
                "id": track_id,
                "position": {
                    "x": x * scale_x,
                    "y": y * scale_y,
                    "scale": [1.0, 1.0],
                },
                # This retains the established renderer/physics contract. It
                # is explicitly marked as assumed, never as a detector result.
                "radius": BALL_RADIUS_4K,
                "radius_scale": [1.0, 1.0],
                "radius_unit": "canonical_pixels",
                "radius_source": "legacy_standard_pool_ball_assumption",
                "confidence": confidence,
                "confidence_kind": (
                    "detector_observation" if status == "observed" else "unavailable"
                ),
                "last_observation_confidence": last_observation_confidence,
                "number": number,
                "type": ball_type,
                "identity_status": "identified" if number is not None else "unknown",
                "identity_confidence": identity.get("confidence"),
                "identity_evidence": identity.get("evidence", []),
                "is_observed": status == "observed",
                "tracking_status": status,
                "position_kind": position_kind,
                "source_observation": {
                    "engine_instance_id": snapshot.get("engine_instance_id"),
                    "session_id": observation.get("session_id"),
                    "frame_id": observation.get("frame_id"),
                    "timestamp": observation.get("timestamp"),
                },
                "image_space": image_space,
                "image_width_px": width,
                "image_height_px": height,
            }
            if table_position is not None:
                ball_payload["table_position"] = {
                    "x": table_position["x"],
                    "y": table_position["y"],
                    "unit": "millimeters",
                    "origin": "playing_surface_center",
                }
            balls.append(ball_payload)

        revisions = _mapping(snapshot.get("revisions"), "revisions")
        calibration_state = revisions.get("calibration_state")
        cue = _mapping(snapshot.get("cue"), "cue")
        if omitted_track_ids and not balls:
            position_status = "unavailable"
        elif omitted_track_ids or not all(ball.get("is_observed") for ball in balls):
            position_status = "cached"
        else:
            # An empty list from a successful observation is positive evidence
            # that no balls were detected, unlike a lost track with no usable
            # image position (handled as unavailable above).
            position_status = "observed"
        observation_status: dict[str, Any] = {
            "phase": "rust_scene_observation",
            "source_session_id": session_id,
            "source_frame_id": frame_id,
            "source_timestamp": source_timestamp_ticks,
            "source_timestamp_unit": "nanoseconds",
            "source_timestamp_kind": source_timestamp_kind,
            "ball_observation_source_timestamp": source_timestamp_ticks,
            "ball_observation_source_timestamp_unit": "nanoseconds",
            "ball_observation_source_timestamp_kind": source_timestamp_kind,
            "source_timestamp_clock": source_timestamp.get("clock"),
            "source_timestamp_domain_id": source_timestamp.get("domain_id"),
            "source_timestamp_is_exposure": source_timestamp.get("stage") == "exposure",
            "published_at": snapshot.get("published_at"),
            "frame_publication_age_status": frame_age_status,
            "frame_age_basis": "source_timestamp_to_snapshot_publication",
            "engine_instance_id": snapshot.get("engine_instance_id"),
            "snapshot_revision": snapshot.get("snapshot_revision"),
            "transport_sequence": snapshot.get("transport_sequence"),
            "configuration_revision": revisions.get("configuration"),
            "rectification_revision": revisions.get("rectification"),
            "calibration_revision": revisions.get("calibration"),
            "calibration_state": calibration_state,
            "source_image_space": "raw",
            "source_width_px": source_width,
            "source_height_px": source_height,
            "ball_positions_status": position_status,
            "ball_detection_status": "observed",
            "ball_radius_source": "legacy_standard_pool_ball_assumption",
            "omitted_track_ids_without_image_position": omitted_track_ids,
            "cue_detection_status": cue.get("status"),
            "observation_identity": {
                "engine_instance_id": snapshot.get("engine_instance_id"),
                "session_id": session_id,
                "frame_id": frame_id,
                "snapshot_revision": snapshot.get("snapshot_revision"),
            },
        }
        if frame_publication_age is not None:
            observation_status["frame_publication_age_seconds"] = frame_publication_age
        return SceneStatePayload(
            balls=balls,
            observation_status=observation_status,
            table=table,
            game_type=game_type,
        )


class SceneSnapshotStatePublisher:
    """Publish Rust observations through the existing state stream directly."""

    def __init__(
        self,
        broadcaster: GameStateBroadcaster,
        adapter: SceneSnapshotStateAdapter | None = None,
    ) -> None:
        self._broadcaster = broadcaster
        self._adapter = adapter or SceneSnapshotStateAdapter()
        self._last_successful_payload: SceneStatePayload | None = None
        self._last_identity: tuple[str, str] | None = None
        self._read_model: SceneStateReadModel | None = None

    @property
    def read_model(self) -> SceneStateReadModel | None:
        """Return a defensive snapshot for HTTP bootstrap and diagnostics."""
        return deepcopy(self._read_model)

    async def publish(
        self,
        snapshot: Mapping[str, Any],
        *,
        table: dict[str, Any] | None = None,
        game_type: str | None = None,
        received_at: datetime | None = None,
        received_monotonic_ns: int | None = None,
    ) -> bool:
        """Publish one successful source observation without a Core round trip."""
        payload = self._adapter.adapt(
            snapshot,
            table=table,
            game_type=game_type,
        )
        if payload is None:
            return False
        _advance_frame_age_after_ingress(
            payload.observation_status, received_monotonic_ns
        )
        localization_received_at = received_at or datetime.now(timezone.utc)
        if (
            localization_received_at.tzinfo is None
            or localization_received_at.utcoffset() is None
        ):
            raise ValueError("scene receipt time must be timezone-aware")
        localization_received_at = localization_received_at.astimezone(timezone.utc)
        identity = _payload_identity(payload)
        previous_identity = self._last_identity
        previous_read_model = self._read_model
        if identity is not None and identity != self._last_identity:
            # A restarted engine/session may reuse track IDs. Never carry
            # locations from the old identity into a new source session.
            self._last_successful_payload = None
        self._last_identity = identity
        self._last_successful_payload = payload
        published_at = datetime.now(timezone.utc)
        previous_balls = (
            {ball["id"]: ball for ball in previous_read_model.payload.balls}
            if previous_read_model is not None and identity == previous_identity
            else {}
        )
        previous_ball_receipts = (
            previous_read_model.ball_localization_received_at
            if previous_read_model is not None and identity == previous_identity
            else {}
        )
        ball_receipts: dict[str, datetime] = {}
        for ball in payload.balls:
            ball_id = str(ball["id"])
            previous_ball = previous_balls.get(ball_id)
            previous_receipt = previous_ball_receipts.get(ball_id)
            if (
                previous_ball is not None
                and previous_receipt is not None
                and previous_ball.get("source_observation")
                == ball.get("source_observation")
            ):
                ball_receipts[ball_id] = previous_receipt
            else:
                ball_receipts[ball_id] = localization_received_at
        self._read_model = SceneStateReadModel(
            payload=payload,
            updated_at=published_at,
            localization_received_at=localization_received_at,
            ball_localization_received_at=ball_receipts,
        )
        await self._broadcaster.broadcast_game_state(
            balls=payload.balls,
            cue=None,
            table=payload.table,
            timestamp=published_at,
            observation_status=payload.observation_status,
            game_type=payload.game_type,
        )
        return True

    async def publish_native_state(
        self,
        delivery: Mapping[str, Any],
        *,
        received_at: datetime,
        received_monotonic_ns: int,
    ) -> Literal["localization", "health", "ignored"]:
        """Cache and relay Rust-owned public state without adapting the scene.

        The service adds only gateway-local receipt times needed by HTTP
        bootstrap and client delivery. Identity, public coordinates, stale and
        expired semantics, and per-ball source provenance come from Rust.
        """
        if delivery.get("delivery_type") != "billiards_public_state_delivery":
            raise SceneStateCompatibilityError(
                "native public-state delivery discriminator is unsupported"
            )
        snapshot = _mapping(delivery.get("source_snapshot"), "source_snapshot")
        current_state_value = delivery.get("current_state")
        if current_state_value is None:
            if snapshot.get("update_kind") == "annotation":
                return "ignored"
            raise SceneStateCompatibilityError(
                "native public-state delivery has no current state"
            )
        current_state = _mapping(current_state_value, "current_state")
        state_payload = _mapping(current_state.get("payload"), "current_state.payload")
        balls_value = state_payload.get("balls")
        if not isinstance(balls_value, list) or not all(
            isinstance(ball, dict) for ball in balls_value
        ):
            raise SceneStateCompatibilityError(
                "native public-state balls must be an array of objects"
            )
        observation_status = _mapping(
            state_payload.get("observation_status"),
            "current_state.payload.observation_status",
        )
        source = _mapping(snapshot.get("source"), "source_snapshot.source")
        engine_instance_id = _identifier(
            snapshot.get("engine_instance_id"),
            "source_snapshot.engine_instance_id",
        )
        session_id = _identifier(source.get("session_id"), "source.session_id")
        if (
            observation_status.get("engine_instance_id") != engine_instance_id
            or observation_status.get("source_session_id") != session_id
        ):
            raise SceneStateCompatibilityError(
                "native state identity differs from the paired source snapshot"
            )
        freshness = current_state.get("freshness")
        if freshness not in {"fresh", "stale", "expired"}:
            raise SceneStateCompatibilityError(
                "native public-state freshness is not recognized"
            )
        identity = (engine_instance_id, session_id)
        successful_observation = (
            snapshot.get("update_kind") == "observation"
            and snapshot.get("outcome") == "success"
        )
        localization = successful_observation and freshness == "fresh"
        health = not localization and snapshot.get("update_kind") != "annotation"
        if not localization and not health:
            return "ignored"

        if (
            received_at.tzinfo is None
            or received_at.utcoffset() is None
            or isinstance(received_monotonic_ns, bool)
            or not isinstance(received_monotonic_ns, int)
            or received_monotonic_ns < 0
        ):
            raise ValueError("native state gateway receipt timestamps are invalid")
        received_at = received_at.astimezone(timezone.utc)
        payload = SceneStatePayload(
            balls=deepcopy(balls_value),
            observation_status=deepcopy(dict(observation_status)),
            table=deepcopy(state_payload.get("table")),
            game_type=state_payload.get("game_type"),
        )
        _advance_frame_age_after_ingress(
            payload.observation_status, received_monotonic_ns
        )

        previous_identity = self._last_identity
        previous_read_model = self._read_model
        if identity != previous_identity:
            self._last_successful_payload = None
        self._last_identity = identity
        if localization:
            self._last_successful_payload = payload

        previous_balls = (
            {ball["id"]: ball for ball in previous_read_model.payload.balls}
            if previous_read_model is not None and identity == previous_identity
            else {}
        )
        previous_ball_receipts = (
            previous_read_model.ball_localization_received_at
            if previous_read_model is not None and identity == previous_identity
            else {}
        )
        ball_receipts: dict[str, datetime] = {}
        for ball in payload.balls:
            ball_id = str(ball["id"])
            previous_ball = previous_balls.get(ball_id)
            previous_receipt = previous_ball_receipts.get(ball_id)
            if (
                previous_ball is not None
                and previous_receipt is not None
                and previous_ball.get("source_observation")
                == ball.get("source_observation")
            ):
                ball_receipts[ball_id] = previous_receipt
            else:
                ball_receipts[ball_id] = received_at
        localization_received_at = (
            received_at
            if localization
            else (
                previous_read_model.localization_received_at
                if previous_read_model is not None and identity == previous_identity
                else None
            )
        )
        published_at = datetime.now(timezone.utc)
        self._read_model = SceneStateReadModel(
            payload=payload,
            updated_at=published_at,
            localization_received_at=localization_received_at,
            ball_localization_received_at=ball_receipts,
        )
        await self._broadcaster.broadcast_game_state(
            balls=payload.balls,
            cue=None,
            table=payload.table,
            timestamp=published_at,
            observation_status=payload.observation_status,
            game_type=payload.game_type,
        )
        return "localization" if localization else "health"

    async def publish_health(self, snapshot: Mapping[str, Any]) -> bool:
        """Publish a health failure as stale/unavailable cached state.

        This communicates an explicit failed or skipped source update without
        changing any ball's source observation time. An annotation update is
        deliberately ignored because it does not say localization failed.
        """
        if (
            snapshot.get("schema_version") != 1
            or snapshot.get("update_kind") != "health"
            or snapshot.get("outcome") not in {"degraded", "failed", "skipped"}
        ):
            return False

        source = _mapping(snapshot.get("source"), "source")
        session_id = _identifier(source.get("session_id"), "source.session_id")
        frame_id = _identifier(source.get("frame_id"), "source.frame_id")
        source_timestamp = _mapping(source.get("timestamp"), "source.timestamp")
        ticks = source_timestamp.get("ticks")
        if (
            not isinstance(ticks, str)
            or re.fullmatch(r"[0-9]+", ticks) is None
            or source_timestamp.get("unit") != "nanoseconds"
        ):
            raise SceneStateCompatibilityError(
                "health timestamp must contain decimal nanosecond ticks"
            )

        engine_instance_id = _identifier(
            snapshot.get("engine_instance_id"), "engine_instance_id"
        )
        identity = (engine_instance_id, session_id)
        previous_identity = self._last_identity
        previous_read_model = self._read_model
        cached_payload = self._last_successful_payload
        if identity != self._last_identity:
            # A failure from a new run is not evidence about the previous run's
            # positions. Clear that state rather than showing it as current.
            cached_payload = None
            self._last_successful_payload = None
        self._last_identity = identity

        balls = (
            [_mark_ball_stale(ball) for ball in cached_payload.balls]
            if cached_payload is not None
            else []
        )
        revisions = _mapping(snapshot.get("revisions"), "revisions")
        outcome = str(snapshot.get("outcome"))
        status: dict[str, Any] = {
            "phase": "rust_scene_health",
            "source_session_id": session_id,
            "source_frame_id": frame_id,
            "source_timestamp": ticks,
            "source_timestamp_unit": "nanoseconds",
            "source_timestamp_kind": ":".join(
                str(source_timestamp.get(field))
                for field in ("clock", "domain_id", "stage")
            ),
            "engine_instance_id": engine_instance_id,
            "snapshot_revision": snapshot.get("snapshot_revision"),
            "transport_sequence": snapshot.get("transport_sequence"),
            "configuration_revision": revisions.get("configuration"),
            "rectification_revision": revisions.get("rectification"),
            "calibration_revision": revisions.get("calibration"),
            "calibration_state": revisions.get("calibration_state"),
            "ball_positions_status": "cached" if balls else "unavailable",
            "ball_detection_status": outcome,
            "health_reason": snapshot.get("reason"),
            "observation_identity": {
                "engine_instance_id": engine_instance_id,
                "session_id": session_id,
                "frame_id": frame_id,
                "snapshot_revision": snapshot.get("snapshot_revision"),
            },
        }
        stale_payload = SceneStatePayload(
            balls=balls,
            observation_status=status,
            table=cached_payload.table if cached_payload is not None else None,
            game_type=cached_payload.game_type if cached_payload is not None else None,
        )
        previous_localization_received_at = (
            previous_read_model.localization_received_at
            if identity == previous_identity and previous_read_model is not None
            else None
        )
        published_at = datetime.now(timezone.utc)
        self._read_model = SceneStateReadModel(
            payload=stale_payload,
            updated_at=published_at,
            localization_received_at=previous_localization_received_at,
            ball_localization_received_at=(
                previous_read_model.ball_localization_received_at.copy()
                if identity == previous_identity and previous_read_model is not None
                else {}
            ),
        )
        await self._broadcaster.broadcast_game_state(
            balls=balls,
            cue=None,
            table=cached_payload.table if cached_payload is not None else None,
            timestamp=published_at,
            observation_status=status,
            game_type=cached_payload.game_type if cached_payload is not None else None,
        )
        return True

    async def publish_source_silence(self) -> bool:
        """Mark the last state stale after the configured source silence limit.

        This event intentionally carries no new source timestamp: the source
        produced no observation or health record from which one could be taken.
        """
        if self._last_identity is None:
            return False
        cached_payload = self._last_successful_payload
        balls = (
            [_mark_ball_stale(ball) for ball in cached_payload.balls]
            if cached_payload is not None
            else []
        )
        engine_instance_id, session_id = self._last_identity
        previous_status = (
            self._read_model.payload.observation_status
            if self._read_model is not None
            else {}
        )
        status = previous_status.copy()
        # A silence transition is a gateway-generated health event. Keep the
        # last observation identity nested for provenance, but don't leave its
        # source timestamp at the top level where it could look newly received.
        for field in (
            "source_frame_id",
            "source_timestamp",
            "source_timestamp_unit",
            "source_timestamp_kind",
            "source_timestamp_clock",
            "source_timestamp_domain_id",
            "source_timestamp_is_exposure",
            "published_at",
            "frame_publication_age_seconds",
            "frame_publication_age_status",
            "frame_age_basis",
            "snapshot_revision",
            "transport_sequence",
        ):
            status.pop(field, None)
        status.update(
            {
                "phase": "rust_scene_health",
                "source_session_id": session_id,
                "engine_instance_id": engine_instance_id,
                "observation_identity": previous_status.get(
                    "observation_identity",
                    {
                        "engine_instance_id": engine_instance_id,
                        "session_id": session_id,
                    },
                ),
                "ball_positions_status": "cached" if balls else "unavailable",
                "ball_detection_status": "failed",
                "health_reason": "source_silence_timeout",
                "freshness_status": "source_silent",
            }
        )
        stale_payload = SceneStatePayload(
            balls=balls,
            observation_status=status,
            table=cached_payload.table if cached_payload is not None else None,
            game_type=cached_payload.game_type if cached_payload is not None else None,
        )
        localization_received_at = (
            self._read_model.localization_received_at
            if self._read_model is not None
            else None
        )
        published_at = datetime.now(timezone.utc)
        self._read_model = SceneStateReadModel(
            payload=stale_payload,
            updated_at=published_at,
            localization_received_at=localization_received_at,
            ball_localization_received_at=(
                self._read_model.ball_localization_received_at.copy()
                if self._read_model is not None
                else {}
            ),
        )
        await self._broadcaster.broadcast_game_state(
            balls=balls,
            cue=None,
            table=cached_payload.table if cached_payload is not None else None,
            game_type=cached_payload.game_type if cached_payload is not None else None,
            timestamp=published_at,
            observation_status=status,
        )
        return True


def _mark_ball_stale(ball: Mapping[str, Any]) -> dict[str, Any]:
    """Keep detector confidence historical when a cached point becomes stale."""
    last_observation_confidence = ball.get("last_observation_confidence")
    if (
        last_observation_confidence is None
        and ball.get("confidence_kind") == "detector_observation"
    ):
        last_observation_confidence = ball.get("confidence")
    return {
        **ball,
        "is_observed": False,
        "confidence": 0.0,
        "confidence_kind": "unavailable",
        "last_observation_confidence": last_observation_confidence,
    }


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SceneStateCompatibilityError(f"{path} must be an object")
    return value


def _frame_age_at_publication(
    source_timestamp: Mapping[str, Any], published_at_value: Any
) -> tuple[float | None, str]:
    """Calculate age only when source and snapshot ticks share a clock domain.

    The Rust capture path labels its process-local monotonic timestamps with
    the same producer domain when it synthesizes a host read-completion time.
    Real camera timestamps may use another clock; those remain explicitly
    unavailable instead of comparing unrelated counters.
    """
    published_at = (
        published_at_value if isinstance(published_at_value, Mapping) else None
    )
    if published_at is None:
        return None, "publication_timestamp_unavailable"
    if (
        source_timestamp.get("clock") != "monotonic"
        or published_at.get("clock") != "monotonic"
        or source_timestamp.get("unit") != "nanoseconds"
        or published_at.get("unit") != "nanoseconds"
        or published_at.get("stage") != "snapshot_publication"
    ):
        return None, "comparable_monotonic_clock_unavailable"
    source_domain = source_timestamp.get("domain_id")
    if (
        not isinstance(source_domain, str)
        or not source_domain
        or published_at.get("domain_id") != source_domain
    ):
        return None, "clock_domain_mismatch"

    published_ticks = published_at.get("ticks")
    if (
        not isinstance(published_ticks, str)
        or re.fullmatch(r"[0-9]+", published_ticks) is None
    ):
        return None, "publication_timestamp_invalid"
    source_ticks = source_timestamp.get("ticks")
    if (
        not isinstance(source_ticks, str)
        or re.fullmatch(r"[0-9]+", source_ticks) is None
    ):
        return None, "source_timestamp_invalid"

    elapsed_ns = int(published_ticks) - int(source_ticks)
    if elapsed_ns < 0:
        return None, "publication_precedes_source"
    return elapsed_ns / 1_000_000_000, "available"


def _advance_frame_age_after_ingress(
    observation_status: dict[str, Any], received_monotonic_ns: int | None
) -> None:
    """Carry producer-measured age across Python's local ingress boundary.

    Producer monotonic ticks cannot be compared with this Python process. The
    age at scene publication is comparable as a duration, however, so add only
    Python's monotonic elapsed time since local receipt. This keeps TCP receive
    and worker scheduling delay from resetting the apparent observation age.
    """
    age_at_publication = observation_status.get("frame_publication_age_seconds")
    if (
        age_at_publication is None
        or isinstance(age_at_publication, bool)
        or not isinstance(age_at_publication, (int, float))
        or not math.isfinite(age_at_publication)
        or age_at_publication < 0
    ):
        return
    if (
        isinstance(received_monotonic_ns, bool)
        or not isinstance(received_monotonic_ns, int)
        or received_monotonic_ns <= 0
    ):
        observation_status.pop("frame_publication_age_seconds", None)
        observation_status["frame_publication_age_status"] = (
            "local_ingress_clock_unavailable"
        )
        return
    try:
        elapsed_ns = time.monotonic_ns() - received_monotonic_ns
    except (OSError, RuntimeError):
        elapsed_ns = -1
    if elapsed_ns < 0:
        observation_status.pop("frame_publication_age_seconds", None)
        observation_status["frame_publication_age_status"] = (
            "local_ingress_clock_invalid"
        )
        return
    advanced_age = float(age_at_publication) + elapsed_ns / 1_000_000_000
    if not math.isfinite(advanced_age):
        observation_status.pop("frame_publication_age_seconds", None)
        observation_status["frame_publication_age_status"] = "local_ingress_age_invalid"
        return
    observation_status["frame_publication_age_seconds"] = advanced_age


def _payload_identity(payload: SceneStatePayload) -> tuple[str, str] | None:
    identity = payload.observation_status.get("observation_identity")
    if not isinstance(identity, Mapping):
        return None
    engine_instance_id = identity.get("engine_instance_id")
    session_id = identity.get("session_id")
    if isinstance(engine_instance_id, str) and isinstance(session_id, str):
        return engine_instance_id, session_id
    return None


def _identifier(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SceneStateCompatibilityError(f"{path} must be a non-empty string")
    return value


def _positive_dimension(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise SceneStateCompatibilityError(f"{path} must be a positive integer")
    return int(value)


def _finite_coordinate(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SceneStateCompatibilityError(f"{path} must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise SceneStateCompatibilityError(f"{path} must be finite")
    return numeric


def _probability(value: Any, path: str) -> float:
    numeric = _finite_coordinate(value, path)
    if not 0 <= numeric <= 1:
        raise SceneStateCompatibilityError(f"{path} must be between zero and one")
    return numeric


def _point_or_none(value: Any, path: str) -> dict[str, float] | None:
    if value is None:
        return None
    point = _mapping(value, path)
    return {
        "x": _finite_coordinate(point.get("x"), f"{path}.x"),
        "y": _finite_coordinate(point.get("y"), f"{path}.y"),
    }
