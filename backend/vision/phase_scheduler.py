"""Opt-in phase-aware scheduling for billiards vision work.

The scheduler consumes timestamped observations, but never performs image
processing itself.  It deliberately defaults to a legacy pass-through mode:
until representative camera recordings validate these profiles, an existing
caller can continue running every vision operation as before.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence

if TYPE_CHECKING:
    from backend.vision.motion_monitor import MotionAssessment


class VisionPhase(str, Enum):
    """Current operating phase of the billiards vision workflow."""

    RECOVERING = "RECOVERING"
    IDLE = "IDLE"
    AIMING = "AIMING"
    LIMITED_MOTION = "LIMITED_MOTION"
    MANY_MOVING = "MANY_MOVING"


@dataclass(frozen=True)
class BallPosition:
    """One ball center and detector confidence in source-frame pixels."""

    x: float
    y: float
    confidence: float = 1.0


@dataclass(frozen=True)
class BallMapObservation:
    """Result of an attempted ball-map observation.

    ``None`` at the scheduler input means the job was skipped or not due and
    has no effect.  An explicit unsuccessful or uncertain observation means
    the attempted result cannot support map trust.
    """

    timestamp: float | None
    balls: tuple[BallPosition, ...]
    full: bool = True
    successful: bool = True
    uncertain: bool = False


@dataclass(frozen=True)
class CueObservation:
    """Result of an attempted cue observation."""

    timestamp: float | None
    detected: bool
    reliable: bool = True


@dataclass(frozen=True)
class PhaseDecision:
    """Immutable work and eligibility decision for one scheduler update."""

    enabled: bool
    phase: VisionPhase
    processing_max_dimension: int | None
    map_trusted: bool
    map_revision: int
    moving_ball_count: int | None
    run_motion_monitor: bool
    verify_ball_map: bool
    detect_cue: bool
    track_moving_balls: bool
    can_calculate_aim: bool
    reason: str

    @property
    def motion_monitor_due(self) -> bool:
        """Alias that makes the scheduling meaning explicit at call sites."""
        return self.run_motion_monitor

    @property
    def ball_map_verification_due(self) -> bool:
        return self.verify_ball_map

    @property
    def cue_detection_due(self) -> bool:
        return self.detect_cue

    @property
    def detailed_tracking_due(self) -> bool:
        return self.track_moving_balls


_DEFAULT_SETTINGS: dict[str, Any] = {
    "enabled": False,
    "quiet_dwell_seconds": 0.8,
    "limited_motion_quiet_dwell_seconds": 0.35,
    "many_moving_exit_quiet_dwell_seconds": 0.8,
    "many_moving_enter_count": 4,
    "many_moving_exit_count": 2,
    "many_moving_enter_dwell_seconds": 0.0,
    "consistent_full_observations_required": 3,
    "ball_position_tolerance_px": 6.0,
    "minimum_ball_confidence": 0.5,
    "minimum_detected_balls": 1,
    # The current application computes aim assist automatically. Until a UI
    # intent signal is wired, fresh cue presence on a trusted quiet map is a
    # useful setup/aiming signal. Callers can override this with False.
    "cue_implies_aiming": True,
    "maximum_motion_observation_gap_seconds": 0.35,
    "maximum_ball_observation_gap_seconds": 1.0,
    # Bounds the UI's observed-only shot paths while keeping enough samples to
    # show gaps without allowing a long session to grow the browser store.
    "actual_path_history_max_points": 120,
    "ball_crop_margin_fraction": 0.08,
    "ball_detection_crop": {phase.value: False for phase in VisionPhase},
    "maximum_observation_age_seconds": 0.5,
    "maximum_cue_age_seconds": 0.25,
    "motion_enter_fraction_threshold": 0.003,
    "motion_exit_fraction_threshold": 0.001,
    "processing_max_dimension": {phase.value: None for phase in VisionPhase},
    "intervals_seconds": {
        "RECOVERING": {
            "motion_monitor": 0.05,
            "ball_map_verification": 0.1,
            "cue_detection": 0.25,
            "detailed_tracking": 0.0,
        },
        "IDLE": {
            "motion_monitor": 0.1,
            "ball_map_verification": 0.75,
            "cue_detection": 1.0,
            "detailed_tracking": 0.0,
        },
        "AIMING": {
            "motion_monitor": 0.05,
            "ball_map_verification": 0.5,
            "cue_detection": 0.05,
            "detailed_tracking": 0.0,
        },
        "LIMITED_MOTION": {
            "motion_monitor": 0.03,
            "ball_map_verification": 0.1,
            "cue_detection": 0.1,
            "detailed_tracking": 0.033,
        },
        "MANY_MOVING": {
            "motion_monitor": 0.05,
            "ball_map_verification": 0.0,
            "cue_detection": 0.0,
            "detailed_tracking": 0.0,
        },
    },
}

_SCHEDULED_WORK = (
    "motion_monitor",
    "ball_map_verification",
    "cue_detection",
    "detailed_tracking",
)


class VisionPhaseScheduler:
    """Choose phase, map trust, and independent work cadences.

    ``update`` requires the caller's monotonic time.  Observation timestamps
    use the same clock domain and may be omitted to use that update time.
    Missing observations mean no new evidence; they never count as a failed
    detection or as quiet settlement evidence.
    """

    def __init__(self, settings: Mapping[str, Any] | None = None) -> None:
        if settings is None:
            try:
                from backend.config import config as config_manager

                settings = config_manager.get("vision.processing.phase_scheduler", {})
            except (ImportError, AttributeError, TypeError):
                settings = {}

        self.settings = self._validated_settings(settings)
        self.enabled = bool(self.settings["enabled"])
        self.settings["cue_implies_aiming"] = bool(self.settings["cue_implies_aiming"])
        self.phase = VisionPhase.RECOVERING
        self.map_trusted = False
        self.map_revision = 0
        self._map_invalidated = True
        self._trusted_snapshot: tuple[BallPosition, ...] | None = None
        self._candidate_snapshot: tuple[BallPosition, ...] | None = None
        self._consistent_full_observations = 0
        self._last_full_ball_timestamp: float | None = None
        self._last_motion_timestamp: float | None = None
        self._motion_latched = False
        self._persistent_change_latched = False
        self._last_motion_quiet = False
        self._quiet_dwell = 0.0
        self._last_quiet_timestamp: float | None = None
        self._last_cue_observation: CueObservation | None = None
        self._last_cue_timestamp: float | None = None
        self._moving_ball_count: int | None = None
        self._moving_count_timestamp: float | None = None
        self._many_enter_since: float | None = None
        self._last_update_timestamp: float | None = None
        self._last_observation_timestamps: dict[str, float] = {}
        self._calibration_revision: Any = _UNSET
        self._next_due: dict[str, float | None] = dict.fromkeys(_SCHEDULED_WORK, None)
        self._last_reason = "startup"

    @staticmethod
    def _validated_settings(
        provided: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        settings = dict(_DEFAULT_SETTINGS)
        if isinstance(provided, Mapping):
            settings.update(provided)

        def finite_number(key: str, minimum: float) -> float:
            try:
                value = float(settings[key])
            except (TypeError, ValueError):
                value = float(_DEFAULT_SETTINGS[key])
            if not math.isfinite(value):
                value = float(_DEFAULT_SETTINGS[key])
            settings[key] = max(minimum, value)
            return float(settings[key])

        for key in (
            "quiet_dwell_seconds",
            "limited_motion_quiet_dwell_seconds",
            "many_moving_exit_quiet_dwell_seconds",
            "many_moving_enter_dwell_seconds",
            "ball_position_tolerance_px",
            "maximum_motion_observation_gap_seconds",
            "maximum_ball_observation_gap_seconds",
            "maximum_observation_age_seconds",
            "maximum_cue_age_seconds",
            "motion_enter_fraction_threshold",
            "motion_exit_fraction_threshold",
        ):
            finite_number(key, 0.0)
        settings["ball_crop_margin_fraction"] = min(
            0.5, finite_number("ball_crop_margin_fraction", 0.0)
        )
        settings["motion_enter_fraction_threshold"] = min(
            1.0, settings["motion_enter_fraction_threshold"]
        )
        settings["motion_exit_fraction_threshold"] = min(
            settings["motion_enter_fraction_threshold"],
            min(1.0, settings["motion_exit_fraction_threshold"]),
        )
        settings["minimum_ball_confidence"] = min(
            1.0, finite_number("minimum_ball_confidence", 0.0)
        )

        def positive_int(key: str, minimum: int) -> int:
            try:
                value = int(settings[key])
            except (TypeError, ValueError, OverflowError):
                value = int(_DEFAULT_SETTINGS[key])
            settings[key] = max(minimum, value)
            return int(settings[key])

        positive_int("many_moving_enter_count", 1)
        positive_int("many_moving_exit_count", 0)
        positive_int("consistent_full_observations_required", 1)
        positive_int("minimum_detected_balls", 1)
        positive_int("actual_path_history_max_points", 2)
        if settings["many_moving_exit_count"] >= settings["many_moving_enter_count"]:
            settings["many_moving_exit_count"] = max(
                0, settings["many_moving_enter_count"] - 1
            )
        settings["enabled"] = bool(settings.get("enabled", False))
        crop_by_phase = settings.get("ball_detection_crop", {})
        settings["ball_detection_crop"] = {
            phase.value: (
                bool(
                    crop_by_phase.get(phase.value, crop_by_phase.get(phase.name, False))
                )
                if isinstance(crop_by_phase, Mapping)
                else False
            )
            for phase in VisionPhase
        }

        provided_intervals = settings.get("intervals_seconds")
        defaults = _DEFAULT_SETTINGS["intervals_seconds"]
        intervals: dict[str, dict[str, float]] = {}
        for phase in VisionPhase:
            phase_defaults = defaults[phase.value]
            phase_values = (
                provided_intervals.get(phase.value, {})
                if isinstance(provided_intervals, Mapping)
                else {}
            )
            intervals[phase.value] = {}
            for task in _SCHEDULED_WORK:
                raw = phase_values.get(task, phase_defaults[task])
                try:
                    value = float(raw)
                except (TypeError, ValueError):
                    value = phase_defaults[task]
                if not math.isfinite(value):
                    value = phase_defaults[task]
                intervals[phase.value][task] = max(0.0, value)
        settings["intervals_seconds"] = intervals

        provided_dimensions = settings.get("processing_max_dimension")
        dimensions: dict[str, int | None] = {}
        for phase in VisionPhase:
            raw_dimension = (
                provided_dimensions.get(phase.value)
                if isinstance(provided_dimensions, Mapping)
                else None
            )
            if raw_dimension is None:
                dimensions[phase.value] = None
                continue
            try:
                dimension = int(raw_dimension)
            except (TypeError, ValueError, OverflowError):
                dimension = 0
            dimensions[phase.value] = dimension if dimension > 0 else None
        settings["processing_max_dimension"] = dimensions
        return settings

    def update(
        self,
        *,
        now: float,
        motion_assessment: MotionAssessment | None = None,
        ball_map_observation: BallMapObservation | None = None,
        cue_observation: CueObservation | None = None,
        moving_ball_count: int | None = None,
        aiming_requested: bool | None = None,
        calibration_revision: Any = None,
    ) -> PhaseDecision:
        """Consume new evidence and return an immutable scheduling decision.

        ``motion_assessment=None``, ``ball_map_observation=None``, and
        ``cue_observation=None`` represent skipped or unavailable work.  Only
        explicit invalid observations can invalidate trust; skipped work does
        not count as a failure or as settlement.
        """
        now = self._finite_timestamp(now, "now")
        if (
            self._last_update_timestamp is not None
            and now < self._last_update_timestamp
        ):
            raise ValueError("now must be monotonic and non-decreasing")
        self._last_update_timestamp = now

        if not self.enabled:
            return self._legacy_passthrough_decision()

        phase_before = self.phase
        if self._update_moving_count(moving_ball_count, now):
            self._update_many_entry_dwell(now)

        if self._calibration_revision is _UNSET:
            self._calibration_revision = calibration_revision
        elif calibration_revision != self._calibration_revision:
            self._calibration_revision = calibration_revision
            self.map_revision += 1
            self._invalidate_map(clear_candidate=True)
            self._reset_quiet_dwell()
            self._enter_recovering("calibration_changed")

        motion_state = self._consume_motion_assessment(motion_assessment, now)

        if motion_state == "stale":
            self._invalidate_map(clear_candidate=True)
            self._reset_quiet_dwell()
            self._enter_recovering("stale_motion_observation")
        elif motion_state == "uncertain":
            self._invalidate_map(clear_candidate=True)
            self._reset_quiet_dwell()
            self._enter_recovering("motion_uncertain")
        elif motion_state == "moving":
            self._handle_motion(now)
        elif motion_state == "quiet":
            self._handle_quiet_motion(now)
        else:
            self._handle_motion_gap(now)

        self._consume_ball_map_observation(ball_map_observation, now)
        self._consume_cue_observation(cue_observation, now)
        aiming_intended = self._aiming_intended(aiming_requested, now)
        self._refresh_map_trust(now, aiming_intended)
        self._transition_aiming(aiming_intended, now)

        phase_changed = phase_before != self.phase
        due = self._scheduled_work(now, phase_changed)
        return PhaseDecision(
            enabled=True,
            phase=self.phase,
            processing_max_dimension=self.settings["processing_max_dimension"][
                self.phase.value
            ],
            map_trusted=self.map_trusted,
            map_revision=self.map_revision,
            moving_ball_count=self._fresh_moving_ball_count(now),
            run_motion_monitor=due["motion_monitor"],
            verify_ball_map=due["ball_map_verification"],
            detect_cue=due["cue_detection"],
            track_moving_balls=due["detailed_tracking"],
            can_calculate_aim=self.map_trusted and self.phase is VisionPhase.AIMING,
            reason=self._last_reason,
        )

    def _consume_motion_assessment(
        self, assessment: MotionAssessment | None, now: float
    ) -> str:
        if assessment is not None:
            timestamp = (
                now
                if assessment.timestamp is None
                else self._finite_timestamp(
                    assessment.timestamp, "motion assessment timestamp"
                )
            )
            if (
                self._last_motion_timestamp is not None
                and timestamp <= self._last_motion_timestamp
            ):
                return "skipped"
            if now - timestamp > self.settings["maximum_observation_age_seconds"]:
                self._last_motion_timestamp = timestamp
                return "stale"
            if not assessment.valid:
                self._last_motion_timestamp = timestamp
                return "uncertain"

            self._last_motion_timestamp = timestamp
            fatal_reasons = set(assessment.uncertainty_reasons) - {
                "active_motion",
                "stable_scene_changed",
                # A previous-frame or stable-scene baseline is intentionally
                # unavailable during monitor startup and immediately after a
                # calibrated geometry change. The first assessment still
                # cannot count as quiet when active motion has no baseline,
                # but subsequent valid frame differences can build quiet
                # dwell while the stationary ball map is being reconciled.
                "previous_frame_unavailable",
                "stable_reference_unavailable",
            }
            if (
                assessment.likely_occlusion
                or assessment.illumination_change
                or fatal_reasons
            ):
                self._last_motion_quiet = False
                return "uncertain"
            motion_score = assessment.motion_score
            if assessment.motion_detected:
                motion_score = max(
                    motion_score, self.settings["motion_enter_fraction_threshold"]
                )
            if self._motion_latched:
                self._motion_latched = (
                    motion_score > self.settings["motion_exit_fraction_threshold"]
                )
            else:
                self._motion_latched = (
                    motion_score >= self.settings["motion_enter_fraction_threshold"]
                )
            if self._motion_latched:
                self._last_motion_quiet = False
                return "moving"

            # A persistent change can be a manually moved ball.  It invalidates
            # the map once per change episode. Repeated comparisons to the old
            # stable reference must not erase new observations needed to
            # reconcile the changed ball positions.
            if assessment.persistent_change_detected:
                if not self._persistent_change_latched:
                    self._invalidate_map(clear_candidate=True)
                    self._enter_recovering("persistent_scene_change")
                self._persistent_change_latched = True
            else:
                self._persistent_change_latched = False
            self._advance_quiet_dwell(timestamp)
            self._last_motion_quiet = True
            return "quiet"

        if self._last_motion_timestamp is None:
            self._last_motion_quiet = False
            return "missing"
        if (
            now - self._last_motion_timestamp
            > self.settings["maximum_motion_observation_gap_seconds"]
        ):
            self._last_motion_quiet = False
            return "stale"
        return "missing"

    def _handle_motion(self, now: float) -> None:
        self._invalidate_map(clear_candidate=True)
        self._reset_quiet_dwell()
        if self.phase is VisionPhase.MANY_MOVING:
            self._last_reason = "many_balls_still_moving"
            return
        if self._many_entry_satisfied(now):
            self._enter_phase(VisionPhase.MANY_MOVING, "many_balls_moving")
        else:
            self._enter_phase(VisionPhase.LIMITED_MOTION, "motion_detected")

    def _handle_quiet_motion(self, now: float) -> None:
        count = self._fresh_moving_ball_count(now)
        if self.phase is VisionPhase.MANY_MOVING:
            if (
                count is not None
                and count <= self.settings["many_moving_exit_count"]
                and self._quiet_dwell
                >= self.settings["many_moving_exit_quiet_dwell_seconds"]
            ):
                self._enter_recovering("many_moving_quiet")
                self._invalidate_map(clear_candidate=True)
                self._consistent_full_observations = 0
            return
        if (
            self.phase is VisionPhase.LIMITED_MOTION
            and self._quiet_dwell >= self.settings["limited_motion_quiet_dwell_seconds"]
        ):
            self._enter_recovering("limited_motion_quiet")
            self._invalidate_map(clear_candidate=True)
            self._consistent_full_observations = 0

    def _handle_motion_gap(self, now: float) -> None:
        if self._last_motion_timestamp is None:
            return
        if (
            now - self._last_motion_timestamp
            > self.settings["maximum_motion_observation_gap_seconds"]
            and self.phase is not VisionPhase.RECOVERING
        ):
            self._invalidate_map(clear_candidate=True)
            self._reset_quiet_dwell()
            self._enter_recovering("motion_observation_gap")

    def _consume_ball_map_observation(
        self,
        observation: BallMapObservation | None,
        now: float,
    ) -> None:
        if observation is None or not observation.full:
            return
        timestamp = (
            now
            if observation.timestamp is None
            else self._finite_timestamp(
                observation.timestamp, "ball map observation timestamp"
            )
        )
        last_timestamp = self._last_observation_timestamps.get("ball_map")
        if last_timestamp is not None and timestamp <= last_timestamp:
            return
        self._last_observation_timestamps["ball_map"] = timestamp
        if now - timestamp > self.settings["maximum_observation_age_seconds"]:
            self._invalidate_map(clear_candidate=True)
            self._consistent_full_observations = 0
            self._enter_recovering("stale_ball_map_observation")
            return
        if not observation.successful or observation.uncertain:
            self._invalidate_map(clear_candidate=True)
            self._consistent_full_observations = 0
            self._candidate_snapshot = None
            self._enter_recovering("ball_map_uncertain")
            return
        if not self._has_fresh_quiet_evidence(now):
            # Full detections during motion or without recent motion evidence
            # cannot establish a stationary map.
            return

        balls = tuple(observation.balls)
        if not self._valid_ball_snapshot(balls):
            self._invalidate_map(clear_candidate=True)
            self._consistent_full_observations = 0
            self._candidate_snapshot = None
            self._enter_recovering("ball_map_invalid")
            return

        if self._candidate_snapshot is None:
            self._candidate_snapshot = balls
            self._consistent_full_observations = 1
        elif self._snapshots_consistent(self._candidate_snapshot, balls):
            self._candidate_snapshot = balls
            self._consistent_full_observations += 1
        else:
            self._invalidate_map(clear_candidate=False)
            self._candidate_snapshot = balls
            self._consistent_full_observations = 1
            self._enter_recovering("ball_map_changed")
        self._last_full_ball_timestamp = timestamp

    def _consume_cue_observation(
        self, observation: CueObservation | None, now: float
    ) -> None:
        if observation is None:
            return
        timestamp = (
            now
            if observation.timestamp is None
            else self._finite_timestamp(
                observation.timestamp, "cue observation timestamp"
            )
        )
        if (
            self._last_cue_timestamp is not None
            and timestamp <= self._last_cue_timestamp
        ):
            return
        self._last_cue_timestamp = timestamp
        if now - timestamp > self.settings["maximum_observation_age_seconds"]:
            self._last_cue_observation = None
            return
        self._last_cue_observation = observation

    def _refresh_map_trust(self, now: float, aiming_requested: bool) -> None:
        if self.phase is not VisionPhase.RECOVERING or self.map_trusted:
            return
        if not self._has_fresh_quiet_evidence(now):
            return
        if (
            self._fresh_moving_ball_count(now) is not None
            and self._fresh_moving_ball_count(now)
            >= self.settings["many_moving_enter_count"]
        ):
            return
        if self._quiet_dwell < self.settings["quiet_dwell_seconds"]:
            return
        if (
            self._consistent_full_observations
            < self.settings["consistent_full_observations_required"]
        ):
            return
        if (
            self._candidate_snapshot is None
            or self._last_full_ball_timestamp is None
            or now - self._last_full_ball_timestamp
            > self.settings["maximum_ball_observation_gap_seconds"]
        ):
            return

        self.map_trusted = True
        self._map_invalidated = False
        self._trusted_snapshot = self._candidate_snapshot
        self._last_reason = "ball_map_verified"
        self._enter_phase(
            (
                VisionPhase.AIMING
                if aiming_requested and self._cue_is_fresh(now)
                else VisionPhase.IDLE
            ),
            "ball_map_verified",
        )

    def _transition_aiming(self, aiming_requested: bool, now: float) -> None:
        if not self.map_trusted:
            return
        cue_ready = self._cue_is_fresh(now)
        if self.phase is VisionPhase.AIMING and (not aiming_requested or not cue_ready):
            self._enter_phase(VisionPhase.IDLE, "aiming_evidence_unavailable")
        elif (
            self.phase is VisionPhase.IDLE
            and aiming_requested
            and cue_ready
            and self._has_fresh_quiet_evidence(now)
        ):
            self._enter_phase(VisionPhase.AIMING, "aiming_ready")

    def _aiming_intended(self, aiming_requested: bool | None, now: float) -> bool:
        """Resolve explicit UI intent or the configured cue-presence signal."""
        if aiming_requested is not None:
            return bool(aiming_requested)
        return bool(self.settings["cue_implies_aiming"] and self._cue_is_fresh(now))

    def _update_moving_count(self, count: int | None, now: float) -> bool:
        if count is None:
            return False
        try:
            parsed = int(count)
        except (TypeError, ValueError, OverflowError):
            return False
        if parsed < 0 or parsed != count:
            return False
        self._moving_ball_count = parsed
        self._moving_count_timestamp = now
        return True

    def _update_many_entry_dwell(self, now: float) -> None:
        count = self._fresh_moving_ball_count(now)
        if count is not None and count >= self.settings["many_moving_enter_count"]:
            if self._many_enter_since is None:
                self._many_enter_since = now
        else:
            self._many_enter_since = None

    def _many_entry_satisfied(self, now: float) -> bool:
        count = self._fresh_moving_ball_count(now)
        return bool(
            count is not None
            and count >= self.settings["many_moving_enter_count"]
            and self._many_enter_since is not None
            and now - self._many_enter_since
            >= self.settings["many_moving_enter_dwell_seconds"]
        )

    def _fresh_moving_ball_count(self, now: float) -> int | None:
        if self._moving_ball_count is None or self._moving_count_timestamp is None:
            return None
        if (
            now - self._moving_count_timestamp
            > self.settings["maximum_observation_age_seconds"]
        ):
            return None
        return self._moving_ball_count

    def _cue_is_fresh(self, now: float) -> bool:
        observation = self._last_cue_observation
        if observation is None or self._last_cue_timestamp is None:
            return False
        return bool(
            observation.detected
            and observation.reliable
            and now - self._last_cue_timestamp
            <= self.settings["maximum_cue_age_seconds"]
        )

    def _advance_quiet_dwell(self, timestamp: float) -> None:
        previous = self._last_quiet_timestamp
        if previous is not None:
            elapsed = timestamp - previous
            if (
                elapsed > 0.0
                and elapsed <= self.settings["maximum_motion_observation_gap_seconds"]
            ):
                self._quiet_dwell += elapsed
            elif elapsed > self.settings["maximum_motion_observation_gap_seconds"]:
                self._quiet_dwell = 0.0
        self._last_quiet_timestamp = timestamp

    def _reset_quiet_dwell(self) -> None:
        self._quiet_dwell = 0.0
        self._last_quiet_timestamp = None
        self._last_motion_quiet = False

    def _has_fresh_quiet_evidence(self, now: float) -> bool:
        return bool(
            self._last_motion_quiet
            and self._last_motion_timestamp is not None
            and now - self._last_motion_timestamp
            <= self.settings["maximum_motion_observation_gap_seconds"]
        )

    def _valid_ball_snapshot(self, balls: Sequence[BallPosition]) -> bool:
        if len(balls) < self.settings["minimum_detected_balls"]:
            return False
        for ball in balls:
            if not all(
                math.isfinite(value) for value in (ball.x, ball.y, ball.confidence)
            ):
                return False
            if not 0.0 <= ball.confidence <= 1.0:
                return False
            if ball.confidence < self.settings["minimum_ball_confidence"]:
                return False
        return True

    def _snapshots_consistent(
        self, first: Sequence[BallPosition], second: Sequence[BallPosition]
    ) -> bool:
        if len(first) != len(second):
            return False
        tolerance = self.settings["ball_position_tolerance_px"]
        unmatched = list(second)
        for ball in first:
            if not unmatched:
                return False
            nearest_index, nearest_ball = min(
                enumerate(unmatched),
                key=lambda indexed: math.hypot(
                    ball.x - indexed[1].x, ball.y - indexed[1].y
                ),
            )
            nearest_distance = math.hypot(
                ball.x - nearest_ball.x, ball.y - nearest_ball.y
            )
            if nearest_distance > tolerance:
                return False
            unmatched.pop(nearest_index)
        return not unmatched

    def _invalidate_map(self, *, clear_candidate: bool) -> None:
        if self.map_trusted and not self._map_invalidated:
            self.map_revision += 1
        self.map_trusted = False
        self._map_invalidated = True
        if clear_candidate:
            self._candidate_snapshot = None
            self._consistent_full_observations = 0
            self._last_full_ball_timestamp = None

    def _enter_recovering(self, reason: str) -> None:
        self._enter_phase(VisionPhase.RECOVERING, reason)

    def _enter_phase(self, phase: VisionPhase, reason: str) -> None:
        if self.phase is not phase:
            self.phase = phase
            self._next_due = dict.fromkeys(_SCHEDULED_WORK, None)
        self._last_reason = reason

    def _scheduled_work(self, now: float, phase_changed: bool) -> dict[str, bool]:
        intervals = self.settings["intervals_seconds"][self.phase.value]
        due: dict[str, bool] = {}
        for task in _SCHEDULED_WORK:
            interval = intervals[task]
            if self.phase is VisionPhase.MANY_MOVING and task != "motion_monitor":
                # Detailed perception work is disabled as soon as the phase
                # transition occurs, including on its first decision.
                due[task] = False
                self._next_due[task] = None
                continue
            if interval <= 0.0:
                due[task] = False
                self._next_due[task] = None
                continue
            next_due = self._next_due[task]
            is_due = phase_changed or next_due is None or now >= next_due
            due[task] = is_due
            if is_due:
                # Skip missed slots instead of replaying a backlog of stale work.
                self._next_due[task] = now + interval
        return due

    def _legacy_passthrough_decision(self) -> PhaseDecision:
        """Keep all work eligible until a profile is explicitly enabled."""
        return PhaseDecision(
            enabled=False,
            phase=VisionPhase.RECOVERING,
            processing_max_dimension=self.settings["processing_max_dimension"][
                VisionPhase.RECOVERING.value
            ],
            map_trusted=False,
            map_revision=self.map_revision,
            moving_ball_count=None,
            run_motion_monitor=True,
            verify_ball_map=True,
            detect_cue=True,
            track_moving_balls=True,
            can_calculate_aim=True,
            reason="scheduler_disabled_legacy_passthrough",
        )

    @staticmethod
    def _finite_timestamp(value: Any, name: str) -> float:
        try:
            timestamp = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"{name} must be a finite number") from None
        if not math.isfinite(timestamp):
            raise ValueError(f"{name} must be a finite number")
        return timestamp


_UNSET = object()
