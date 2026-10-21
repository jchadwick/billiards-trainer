"""Calibrated table-scene motion and change monitoring.

The monitor compares each observation with the preceding observation to detect
active motion and with an explicitly committed reference to detect persistent
scene changes. Assessment never updates the stable reference; callers can only
replace it after both settlement and ball-map reconciliation have succeeded.
"""

from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Optional, Sequence

import cv2
import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

Point = tuple[float, float]
Polygon = Sequence[Point]
CandidateBall = tuple[float, float, float]

# These fallbacks keep the monitor safe when an older or external config file
# does not yet contain vision.processing.motion_monitor. The shipped config.json
# contains the same defaults and remains the source for deployment tuning.
_DEFAULT_SETTINGS: dict[str, Any] = {
    "enabled": True,
    "difference_threshold": 20,
    "active_motion_fraction_threshold": 0.003,
    "persistent_change_fraction_threshold": 0.005,
    "minimum_region_area_px": 12,
    "minimum_region_area_fraction": 0.0002,
    "likely_occlusion_min_area_fraction": 0.008,
    "likely_occlusion_max_area_fraction": 0.4,
    "illumination_median_delta_threshold": 12.0,
    "illumination_contrast_change_threshold": 0.2,
    "candidate_neighborhood_radius_scale": 1.8,
    "minimum_candidate_radius_px": 2.0,
    "max_changed_regions": 32,
    "normalize_illumination": True,
}


@dataclass(frozen=True)
class ChangedRegion:
    """A connected changed area in source-frame pixel coordinates."""

    bbox: tuple[int, int, int, int]
    centroid: Point
    area_px: int
    area_fraction: float


@dataclass(frozen=True)
class BallNeighborhoodChange:
    """Frame-difference measurements near one caller-supplied ball candidate."""

    candidate_index: int
    center: Point
    radius: float
    active_change_fraction: float | None
    stable_change_fraction: float | None


@dataclass(frozen=True)
class MotionAssessment:
    """Motion, persistent-change, occlusion, and illumination metrics."""

    valid: bool
    uncertain: bool
    uncertainty_reasons: tuple[str, ...]
    motion_detected: bool
    motion_score: float
    active_motion_fraction: float | None
    persistent_change_detected: bool
    stable_scene_change_fraction: float | None
    changed_regions: tuple[ChangedRegion, ...]
    active_changed_regions: tuple[ChangedRegion, ...]
    likely_occlusion: bool
    occlusion_score: float
    illumination_change: bool
    illumination_change_score: float
    illumination_median_delta: float | None
    illumination_contrast_change: float | None
    candidate_ball_changes: tuple[BallNeighborhoodChange, ...]
    table_mask_pixels: int
    frame_resolution: tuple[int, int] | None
    timestamp: float | None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation of this assessment."""
        return asdict(self)


class MotionMonitor:
    """Compare calibrated table pixels against temporal scene references.

    Polygon points, candidate-ball centers, and radii are expressed in native
    camera-frame pixels. ``assess`` updates only the immediately preceding
    frame used for active-motion detection. It never adapts the committed
    stable-scene reference.
    """

    def __init__(
        self,
        settings: Mapping[str, Any] | None = None,
        *,
        table_polygon: Polygon | None = None,
    ) -> None:
        """Create a monitor, optionally with settings and calibrated corners.

        If settings are omitted, values are read from the shared configuration
        at ``vision.processing.motion_monitor`` and merged with safe defaults.
        """
        if settings is None:
            try:
                from backend.config import config as config_manager

                settings = config_manager.get("vision.processing.motion_monitor", {})
            except (ImportError, AttributeError, TypeError):
                settings = {}

        self.settings = self._validated_settings(settings)
        self._configured_polygon = table_polygon
        self._geometry_signature: tuple[Any, ...] | None = None
        self._previous_gray: NDArray[np.uint8] | None = None
        self._stable_gray: NDArray[np.uint8] | None = None
        self._table_mask: NDArray[np.uint8] | None = None
        self._stable_timestamp: float | None = None

    @staticmethod
    def _validated_settings(
        provided: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        settings = dict(_DEFAULT_SETTINGS)
        if isinstance(provided, Mapping):
            settings.update(provided)

        def finite_number(
            key: str, minimum: float, maximum: float | None = None
        ) -> float:
            try:
                value = float(settings[key])
            except (TypeError, ValueError):
                value = float(_DEFAULT_SETTINGS[key])
            if not math.isfinite(value):
                value = float(_DEFAULT_SETTINGS[key])
            value = max(minimum, value)
            if maximum is not None:
                value = min(maximum, value)
            settings[key] = value
            return value

        finite_number("difference_threshold", 0.0, 255.0)
        finite_number("active_motion_fraction_threshold", 0.0, 1.0)
        finite_number("persistent_change_fraction_threshold", 0.0, 1.0)
        finite_number("minimum_region_area_px", 1.0)
        finite_number("minimum_region_area_fraction", 0.0, 1.0)
        finite_number("likely_occlusion_min_area_fraction", 0.0, 1.0)
        finite_number("likely_occlusion_max_area_fraction", 0.0, 1.0)
        finite_number("illumination_median_delta_threshold", 0.0, 255.0)
        finite_number("illumination_contrast_change_threshold", 0.0, 10.0)
        finite_number("candidate_neighborhood_radius_scale", 1.0, 8.0)
        finite_number("minimum_candidate_radius_px", 0.0)
        finite_number("max_changed_regions", 1.0)
        if (
            settings["likely_occlusion_max_area_fraction"]
            < settings["likely_occlusion_min_area_fraction"]
        ):
            settings["likely_occlusion_max_area_fraction"] = settings[
                "likely_occlusion_min_area_fraction"
            ]
        settings["minimum_region_area_px"] = int(settings["minimum_region_area_px"])
        settings["max_changed_regions"] = int(settings["max_changed_regions"])
        settings["enabled"] = bool(settings["enabled"])
        settings["normalize_illumination"] = bool(settings["normalize_illumination"])
        return settings

    @property
    def has_stable_reference(self) -> bool:
        """Whether an explicitly committed scene reference is available."""
        return self._stable_gray is not None

    @property
    def stable_reference_timestamp(self) -> float | None:
        """Timestamp supplied when the stable reference was committed."""
        return self._stable_timestamp

    def reset(self) -> None:
        """Forget both temporal references, for example after camera movement."""
        self._geometry_signature = None
        self._previous_gray = None
        self._stable_gray = None
        self._table_mask = None
        self._stable_timestamp = None

    def assess(
        self,
        frame: Any,
        table_polygon: Polygon | None = None,
        *,
        candidate_balls: Iterable[Any] | None = None,
        timestamp: float | None = None,
    ) -> MotionAssessment:
        """Assess active motion and stable-scene changes for one camera frame.

        ``candidate_balls`` may contain ``(x, y, radius)`` tuples or objects
        with ``position`` and ``radius`` attributes. A failed input or missing
        calibration returns an invalid, uncertain assessment with no motion
        claim.
        """
        selected_polygon = (
            table_polygon if table_polygon is not None else self._configured_polygon
        )
        converted = self._prepare(frame, selected_polygon)
        if converted is None:
            return self._invalid_assessment(timestamp)
        gray, mask, polygon_signature = converted

        height, width = gray.shape
        geometry_signature = (width, height, polygon_signature)
        geometry_changed = (
            self._geometry_signature is not None
            and self._geometry_signature != geometry_signature
        )
        if geometry_changed:
            # Reference pixels are tied to both resolution and calibration.
            # Never compare images across incompatible camera geometry.
            self._previous_gray = None
            self._stable_gray = None
            self._table_mask = None
            self._stable_timestamp = None
        self._geometry_signature = geometry_signature
        self._table_mask = mask

        reasons: list[str] = []
        if not self.settings["enabled"]:
            reasons.append("monitor_disabled")

        active_fraction: float | None = None
        active_regions: tuple[ChangedRegion, ...] = ()
        active_diff: NDArray[np.uint8] | None = None
        active_illumination: tuple[float | None, float | None, float, bool] = (
            None,
            None,
            0.0,
            False,
        )
        if self._previous_gray is None:
            reasons.append("previous_frame_unavailable")
        else:
            active_diff, active_illumination = self._difference(
                self._previous_gray, gray, mask
            )
            active_fraction = self._changed_fraction(active_diff, mask)
            active_regions = self._regions(active_diff, mask)

        stable_fraction: float | None = None
        stable_regions: tuple[ChangedRegion, ...] = ()
        stable_diff: NDArray[np.uint8] | None = None
        stable_illumination: tuple[float | None, float | None, float, bool] = (
            None,
            None,
            0.0,
            False,
        )
        if self._stable_gray is None:
            reasons.append("stable_reference_unavailable")
        else:
            stable_diff, stable_illumination = self._difference(
                self._stable_gray, gray, mask
            )
            stable_fraction = self._changed_fraction(stable_diff, mask)
            stable_regions = self._regions(stable_diff, mask)

        if geometry_changed:
            reasons.append("calibration_or_resolution_changed")

        illumination_sources = [
            result
            for result in (active_illumination, stable_illumination)
            if result[0] is not None
        ]
        if illumination_sources:
            # Prefer the stable reference for the reported persistent lighting
            # change, while still flagging a rapid exposure shift between frames.
            illumination = max(illumination_sources, key=lambda item: item[2])
            median_delta, contrast_change, illum_score, illum_changed = illumination
        else:
            median_delta, contrast_change, illum_score, illum_changed = (
                None,
                None,
                0.0,
                False,
            )

        motion_detected = bool(
            active_fraction is not None
            and active_fraction >= self.settings["active_motion_fraction_threshold"]
        )
        persistent_change = bool(
            stable_fraction is not None
            and stable_fraction >= self.settings["persistent_change_fraction_threshold"]
        )
        if motion_detected:
            reasons.append("active_motion")
        if persistent_change:
            reasons.append("stable_scene_changed")
        if illum_changed:
            reasons.append("illumination_changed")

        local_changes = self._ball_neighborhoods(
            candidate_balls,
            active_diff,
            stable_diff,
            mask,
        )
        likely_occlusion, occlusion_score = self._occlusion_likelihood(
            stable_regions + active_regions,
            local_changes,
            mask,
        )
        if likely_occlusion:
            reasons.append("likely_occlusion")

        # Keep a previous-frame copy for the next active-motion comparison.
        # The stable reference is never changed here.
        self._previous_gray = gray.copy()

        return MotionAssessment(
            valid=self.settings["enabled"],
            uncertain=bool(reasons),
            uncertainty_reasons=tuple(dict.fromkeys(reasons)),
            motion_detected=motion_detected,
            motion_score=active_fraction or 0.0,
            active_motion_fraction=active_fraction,
            persistent_change_detected=persistent_change,
            stable_scene_change_fraction=stable_fraction,
            changed_regions=stable_regions,
            active_changed_regions=active_regions,
            likely_occlusion=likely_occlusion,
            occlusion_score=occlusion_score,
            illumination_change=illum_changed,
            illumination_change_score=illum_score,
            illumination_median_delta=median_delta,
            illumination_contrast_change=contrast_change,
            candidate_ball_changes=local_changes,
            table_mask_pixels=int(np.count_nonzero(mask)),
            frame_resolution=(width, height),
            timestamp=timestamp,
        )

    def commit_verified_reference(
        self,
        frame: Any,
        table_polygon: Polygon | None = None,
        *,
        settled: bool,
        map_reconciled: bool,
        timestamp: float | None = None,
    ) -> bool:
        """Commit a stable reference only after caller verifies both guards.

        If either guard is false, this method is a strict no-op and returns
        ``False``. Invalid frames or calibration also leave references intact.
        """
        if not settled or not map_reconciled:
            return False
        selected_polygon = (
            table_polygon if table_polygon is not None else self._configured_polygon
        )
        converted = self._prepare(frame, selected_polygon)
        if converted is None:
            return False
        gray, mask, polygon_signature = converted
        height, width = gray.shape
        signature = (width, height, polygon_signature)
        if (
            self._geometry_signature is not None
            and self._geometry_signature != signature
        ):
            self._previous_gray = None
        self._geometry_signature = signature
        self._table_mask = mask
        self._stable_gray = gray.copy()
        self._previous_gray = gray.copy()
        self._stable_timestamp = timestamp
        return True

    def _prepare(
        self, frame: Any, polygon: Polygon | None
    ) -> tuple[NDArray[np.uint8], NDArray[np.uint8], tuple[Point, ...]] | None:
        gray = self._to_gray(frame)
        parsed_polygon = self._parse_polygon(polygon)
        if gray is None or parsed_polygon is None:
            return None
        height, width = gray.shape
        if width < 2 or height < 2:
            return None
        points = np.rint(np.asarray(parsed_polygon, dtype=np.float64)).astype(np.int32)
        points[:, 0] = np.clip(points[:, 0], 0, width - 1)
        points[:, 1] = np.clip(points[:, 1], 0, height - 1)
        if abs(float(cv2.contourArea(points))) < 1.0:
            return None
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [points], 255)
        mask_pixels = int(np.count_nonzero(mask))
        if mask_pixels < max(4, int(width * height * 0.0001)):
            return None
        signature = tuple((float(x), float(y)) for x, y in parsed_polygon)
        return gray, mask, signature

    @staticmethod
    def _parse_polygon(polygon: Polygon | None) -> tuple[Point, ...] | None:
        if polygon is None:
            return None
        try:
            points = tuple((float(point[0]), float(point[1])) for point in polygon)
        except (TypeError, ValueError, IndexError):
            return None
        if len(points) < 3 or any(
            not math.isfinite(coordinate) for point in points for coordinate in point
        ):
            return None
        return points

    @staticmethod
    def _to_gray(frame: Any) -> NDArray[np.uint8] | None:
        if not isinstance(frame, np.ndarray) or frame.size == 0:
            return None
        if frame.ndim == 2:
            gray = frame
        elif frame.ndim == 3 and frame.shape[2] in (1, 3, 4):
            if frame.shape[2] == 1:
                gray = frame[:, :, 0]
            elif frame.shape[2] == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
        else:
            return None
        if gray.size == 0 or not np.issubdtype(gray.dtype, np.number):
            return None
        if not np.isfinite(gray).all():
            return None
        if gray.dtype == np.uint8:
            return gray.copy()
        values = gray.astype(np.float32)
        if (
            values.size
            and float(np.nanmax(values)) <= 1.0
            and float(np.nanmin(values)) >= 0
        ):
            values *= 255.0
        return np.clip(values, 0, 255).astype(np.uint8)

    def _difference(
        self,
        baseline: NDArray[np.uint8],
        current: NDArray[np.uint8],
        mask: NDArray[np.uint8],
    ) -> tuple[NDArray[np.uint8], tuple[float | None, float | None, float, bool]]:
        valid = mask > 0
        base_median, base_iqr = self._robust_intensity_stats(baseline, mask)
        current_median, current_iqr = self._robust_intensity_stats(current, mask)
        median_delta = abs(current_median - base_median)
        contrast_change = abs(current_iqr - base_iqr) / max(base_iqr, 8.0)
        median_threshold = self.settings["illumination_median_delta_threshold"]
        contrast_threshold = self.settings["illumination_contrast_change_threshold"]
        illumination_score = max(
            median_delta / max(median_threshold, 1.0),
            contrast_change / max(contrast_threshold, 0.01),
        )
        illumination_changed = bool(
            median_delta >= median_threshold or contrast_change >= contrast_threshold
        )

        current_float = current.astype(np.float32)
        if self.settings["normalize_illumination"]:
            # Use robust table-wide statistics so exposure and broad lighting
            # shifts do not look like motion everywhere on the cloth.
            scale = float(np.clip(base_iqr / max(current_iqr, 8.0), 0.5, 2.0))
            current_float = (current_float - current_median) * scale + base_median
        normalized_current = np.clip(current_float, 0, 255).astype(np.uint8)
        difference = cv2.absdiff(baseline, normalized_current)
        threshold = int(self.settings["difference_threshold"])
        binary: NDArray[np.uint8] = np.where(difference >= threshold, 255, 0).astype(
            np.uint8
        )
        binary[~valid] = 0
        # Remove isolated sensor noise while keeping ball edges and hand-sized
        # regions intact. A small elliptical opening is resolution-independent.
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel).astype(np.uint8)
        binary[~valid] = 0
        return binary, (
            median_delta,
            contrast_change,
            float(illumination_score),
            illumination_changed,
        )

    @staticmethod
    def _robust_intensity_stats(
        image: NDArray[np.uint8], mask: NDArray[np.uint8]
    ) -> tuple[float, float]:
        """Estimate median and interquartile range with an 8-bit histogram.

        Histogram quantiles avoid sorting millions of table pixels on each
        camera observation while remaining robust to balls and small occlusions.
        """
        histogram = cv2.calcHist([image], [0], mask, [256], [0, 256]).reshape(-1)
        cumulative = np.cumsum(histogram)
        sample_count = float(cumulative[-1])
        if sample_count <= 0:
            return 0.0, 0.0

        def quantile(fraction: float) -> float:
            target = max(1.0, sample_count * fraction)
            return float(np.searchsorted(cumulative, target, side="left"))

        lower_quartile = quantile(0.25)
        median = quantile(0.5)
        upper_quartile = quantile(0.75)
        return median, upper_quartile - lower_quartile

    @staticmethod
    def _changed_fraction(changed: NDArray[np.uint8], mask: NDArray[np.uint8]) -> float:
        valid_pixels = int(np.count_nonzero(mask))
        if not valid_pixels:
            return 0.0
        return float(np.count_nonzero((changed > 0) & (mask > 0)) / valid_pixels)

    def _regions(
        self, changed: NDArray[np.uint8], mask: NDArray[np.uint8]
    ) -> tuple[ChangedRegion, ...]:
        count, labels, stats, centroids = cv2.connectedComponentsWithStats(
            np.where((changed > 0) & (mask > 0), 255, 0).astype(np.uint8),
            connectivity=8,
        )
        total_mask_pixels = max(int(np.count_nonzero(mask)), 1)
        minimum_area = max(
            self.settings["minimum_region_area_px"],
            int(total_mask_pixels * self.settings["minimum_region_area_fraction"]),
        )
        regions: list[ChangedRegion] = []
        for label in range(1, count):
            x, y, width, height, area = stats[label]
            if int(area) < minimum_area:
                continue
            cx, cy = centroids[label]
            regions.append(
                ChangedRegion(
                    bbox=(int(x), int(y), int(width), int(height)),
                    centroid=(float(cx), float(cy)),
                    area_px=int(area),
                    area_fraction=float(area / total_mask_pixels),
                )
            )
        regions.sort(key=lambda region: region.area_px, reverse=True)
        return tuple(regions[: self.settings["max_changed_regions"]])

    def _ball_neighborhoods(
        self,
        candidates: Iterable[Any] | None,
        active_diff: NDArray[np.uint8] | None,
        stable_diff: NDArray[np.uint8] | None,
        mask: NDArray[np.uint8],
    ) -> tuple[BallNeighborhoodChange, ...]:
        if candidates is None:
            return ()
        try:
            iterator = iter(candidates)
        except TypeError:
            return ()
        height, width = mask.shape
        changes: list[BallNeighborhoodChange] = []
        for index, candidate in enumerate(iterator):
            parsed = self._candidate_ball(candidate)
            if parsed is None:
                continue
            x, y, radius = parsed
            radius = max(radius, self.settings["minimum_candidate_radius_px"])
            neighborhood_radius = int(
                math.ceil(radius * self.settings["candidate_neighborhood_radius_scale"])
            )
            neighborhood = np.zeros((height, width), dtype=np.uint8)
            center = (int(round(x)), int(round(y)))
            cv2.circle(neighborhood, center, neighborhood_radius, 255, thickness=-1)
            neighborhood = cv2.bitwise_and(neighborhood, mask).astype(np.uint8)
            if not np.any(neighborhood):
                continue
            changes.append(
                BallNeighborhoodChange(
                    candidate_index=index,
                    center=(float(x), float(y)),
                    radius=float(radius),
                    active_change_fraction=(
                        self._changed_fraction(active_diff, neighborhood)
                        if active_diff is not None
                        else None
                    ),
                    stable_change_fraction=(
                        self._changed_fraction(stable_diff, neighborhood)
                        if stable_diff is not None
                        else None
                    ),
                )
            )
        return tuple(changes)

    @staticmethod
    def _candidate_ball(candidate: Any) -> CandidateBall | None:
        try:
            if isinstance(candidate, Mapping):
                position = candidate.get("position", candidate.get("center"))
                radius = candidate.get("radius")
                if position is None and "x" in candidate and "y" in candidate:
                    position = (candidate["x"], candidate["y"])
            elif hasattr(candidate, "position"):
                position = candidate.position
                radius = candidate.radius
            else:
                position = candidate[:2]
                radius = candidate[2]
            x, y = float(position[0]), float(position[1])
            if radius is None:
                return None
            radius = float(radius)
            if not all(math.isfinite(value) for value in (x, y, radius)) or radius < 0:
                return None
            return x, y, radius
        except (TypeError, ValueError, IndexError, AttributeError, KeyError):
            return None

    def _occlusion_likelihood(
        self,
        regions: tuple[ChangedRegion, ...],
        local_changes: tuple[BallNeighborhoodChange, ...],
        mask: NDArray[np.uint8],
    ) -> tuple[bool, float]:
        minimum = self.settings["likely_occlusion_min_area_fraction"]
        maximum = self.settings["likely_occlusion_max_area_fraction"]
        mask_area = max(int(np.count_nonzero(mask)), 1)
        likely_regions: list[ChangedRegion] = []
        for region in regions:
            if not minimum <= region.area_fraction <= maximum:
                continue
            expected_ball_area = 0.0
            for local in local_changes:
                dx = region.centroid[0] - local.center[0]
                dy = region.centroid[1] - local.center[1]
                if math.hypot(dx, dy) <= local.radius * 2.5:
                    expected_ball_area = max(
                        expected_ball_area,
                        math.pi * (local.radius * 1.8) ** 2,
                    )
            if expected_ball_area and region.area_px <= expected_ball_area * 3.0:
                continue
            likely_regions.append(region)
        if not likely_regions:
            return False, 0.0
        largest_fraction = max(region.area_px / mask_area for region in likely_regions)
        score = float(np.clip(largest_fraction / max(maximum, 0.01), 0.0, 1.0))
        return True, score

    @staticmethod
    def _invalid_assessment(timestamp: float | None) -> MotionAssessment:
        return MotionAssessment(
            valid=False,
            uncertain=True,
            uncertainty_reasons=("invalid_frame_or_calibration",),
            motion_detected=False,
            motion_score=0.0,
            active_motion_fraction=None,
            persistent_change_detected=False,
            stable_scene_change_fraction=None,
            changed_regions=(),
            active_changed_regions=(),
            likely_occlusion=False,
            occlusion_score=0.0,
            illumination_change=False,
            illumination_change_score=0.0,
            illumination_median_delta=None,
            illumination_contrast_change=None,
            candidate_ball_changes=(),
            table_mask_pixels=0,
            frame_resolution=None,
            timestamp=timestamp,
        )
