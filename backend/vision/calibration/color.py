"""Color threshold calibration."""

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
from config import config as config_manager
from numpy.typing import NDArray
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)


@dataclass
class ColorThresholds:
    """HSV color thresholds for object detection."""

    hue_min: int
    hue_max: int
    saturation_min: int
    saturation_max: int
    value_min: int
    value_max: int

    def to_dict(self) -> dict[str, int]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, int]) -> "ColorThresholds":
        """Create from dictionary."""
        return cls(**data)

    def apply_mask(self, hsv_image: NDArray[np.float64]) -> NDArray[np.float64]:
        """Apply color mask to HSV image."""
        # Handle hue wraparound (red color)
        if self.hue_min > self.hue_max:
            mask1 = cv2.inRange(
                hsv_image,
                (self.hue_min, self.saturation_min, self.value_min),
                (179, self.saturation_max, self.value_max),
            )
            mask2 = cv2.inRange(
                hsv_image,
                (0, self.saturation_min, self.value_min),
                (self.hue_max, self.saturation_max, self.value_max),
            )
            return cv2.bitwise_or(mask1, mask2)
        else:
            return cv2.inRange(
                hsv_image,
                (self.hue_min, self.saturation_min, self.value_min),
                (self.hue_max, self.saturation_max, self.value_max),
            )


@dataclass
class ColorProfile:
    """Complete color calibration profile."""

    name: str
    table_color: ColorThresholds
    ball_colors: dict[str, ColorThresholds]
    lighting_condition: str
    creation_date: str
    ambient_light_level: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to serializable dictionary."""
        return {
            "name": self.name,
            "table_color": self.table_color.to_dict(),
            "ball_colors": {k: v.to_dict() for k, v in self.ball_colors.items()},
            "lighting_condition": self.lighting_condition,
            "creation_date": self.creation_date,
            "ambient_light_level": self.ambient_light_level,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ColorProfile":
        """Create from dictionary."""
        return cls(
            name=data["name"],
            table_color=ColorThresholds.from_dict(data["table_color"]),
            ball_colors={
                k: ColorThresholds.from_dict(v) for k, v in data["ball_colors"].items()
            },
            lighting_condition=data["lighting_condition"],
            creation_date=data["creation_date"],
            ambient_light_level=data["ambient_light_level"],
        )


class ColorSample:
    """Helper class for collecting color samples."""

    def __init__(self, roi: tuple[int, int, int, int], label: str) -> None:
        """Initialize color sample.

        Args:
            roi: Region of interest (x, y, width, height)
            label: Label for this sample
        """
        self.roi = roi
        self.label = label
        self.pixels: list[tuple[int, int, int]] = []

    def add_pixels(self, hsv_image: NDArray[np.float64]) -> None:
        """Extract pixels from ROI."""
        x, y, w, h = self.roi
        roi_pixels = hsv_image[y : y + h, x : x + w].reshape(-1, 3)
        self.pixels.extend(roi_pixels.tolist())

    def get_statistics(self) -> dict[str, Any]:
        """Get color statistics for collected pixels."""
        if not self.pixels:
            return {}

        pixels_array = np.array(self.pixels)
        return {
            "mean": np.mean(pixels_array, axis=0).tolist(),
            "std": np.std(pixels_array, axis=0).tolist(),
            "min": np.min(pixels_array, axis=0).tolist(),
            "max": np.max(pixels_array, axis=0).tolist(),
            "count": len(self.pixels),
        }


class ColorCalibrator:
    """Color threshold calibration for objects.

    Implements requirements FR-VIS-044 to FR-VIS-047:
    - Auto-detect optimal color thresholds
    - Adapt to ambient lighting changes
    - Provide color picker interface
    - Save and load calibration profiles
    """

    def __init__(self, cache_dir: Optional[str] = None) -> None:
        """Initialize color calibrator.

        Args:
            cache_dir: Directory to cache calibration profiles
        """
        self.cache_dir = Path(cache_dir) if cache_dir else Path.cwd() / "color_profiles"
        self.cache_dir.mkdir(exist_ok=True)

        # Standard ball colors (approximate HSV ranges) - load from config
        default_ball_colors_config = config_manager.get(
            "vision.calibration.color.default_ball_colors", {}
        )
        self.default_ball_colors = {}
        for ball_type, thresholds in default_ball_colors_config.items():
            self.default_ball_colors[ball_type] = ColorThresholds(
                hue_min=thresholds.get("hue_min", 0),
                hue_max=thresholds.get("hue_max", 180),
                saturation_min=thresholds.get("saturation_min", 0),
                saturation_max=thresholds.get("saturation_max", 255),
                value_min=thresholds.get("value_min", 0),
                value_max=thresholds.get("value_max", 255),
            )

        self.current_profile: Optional[ColorProfile] = None
        self.color_samples: list[ColorSample] = []

    def auto_calibrate_table_color(
        self, frame: NDArray[np.uint8], table_mask: Optional[NDArray[np.float64]] = None
    ) -> ColorThresholds:
        """Auto-detect optimal table color thresholds.

        Args:
            frame: Input frame
            table_mask: Optional mask indicating table region

        Returns:
            Optimal color thresholds for table
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # If no mask provided, use center region as approximate table area
        if table_mask is None:
            h, w = hsv.shape[:2]
            table_mask = np.zeros((h, w), dtype=np.uint8)
            # Use center region with configurable margin
            margin_ratio = config_manager.get(
                "vision.calibration.color.auto_calibration.table_mask_margin_ratio", 0.2
            )
            margin_h, margin_w = int(h * margin_ratio), int(w * margin_ratio)
            table_mask[margin_h : h - margin_h, margin_w : w - margin_w] = 255

        # Extract table pixels
        table_pixels = hsv[table_mask > 0]

        if len(table_pixels) == 0:
            logger.warning(
                "No table pixels found, using default green table thresholds"
            )
            default_table = config_manager.get(
                "vision.calibration.color.default_table_color", {}
            )
            return ColorThresholds(
                hue_min=default_table.get("hue_min", 45),
                hue_max=default_table.get("hue_max", 75),
                saturation_min=default_table.get("saturation_min", 40),
                saturation_max=default_table.get("saturation_max", 255),
                value_min=default_table.get("value_min", 40),
                value_max=default_table.get("value_max", 200),
            )

        # Use K-means to find dominant color
        n_clusters = config_manager.get(
            "vision.calibration.color.auto_calibration.kmeans_clusters", 3
        )
        random_state = config_manager.get(
            "vision.calibration.color.auto_calibration.kmeans_random_state", 42
        )
        n_init = config_manager.get(
            "vision.calibration.color.auto_calibration.kmeans_n_init", 10
        )
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=n_init)
        clusters = kmeans.fit(table_pixels.reshape(-1, 3))

        # Find the largest cluster (most common color)
        labels = clusters.labels_
        unique_labels, counts = np.unique(labels, return_counts=True)
        dominant_cluster = unique_labels[np.argmax(counts)]

        # Get pixels belonging to dominant cluster
        dominant_pixels = table_pixels[labels == dominant_cluster]

        # Calculate statistics
        mean_hsv = np.mean(dominant_pixels, axis=0)
        std_hsv = np.std(dominant_pixels, axis=0)

        # Set thresholds based on statistics (±N standard deviations)
        std_multiplier = config_manager.get(
            "vision.calibration.color.auto_calibration.std_deviation_multiplier", 2
        )
        hue_min = max(0, int(mean_hsv[0] - std_multiplier * std_hsv[0]))
        hue_max = min(179, int(mean_hsv[0] + std_multiplier * std_hsv[0]))
        sat_min = max(0, int(mean_hsv[1] - std_multiplier * std_hsv[1]))
        sat_max = min(255, int(mean_hsv[1] + std_multiplier * std_hsv[1]))
        val_min = max(0, int(mean_hsv[2] - std_multiplier * std_hsv[2]))
        val_max = min(255, int(mean_hsv[2] + std_multiplier * std_hsv[2]))

        logger.info(
            f"Auto-calibrated table color: H({hue_min}-{hue_max}) S({sat_min}-{sat_max}) V({val_min}-{val_max})"
        )

        return ColorThresholds(hue_min, hue_max, sat_min, sat_max, val_min, val_max)

    def calibrate_ball_colors(
        self,
        frame: NDArray[np.uint8],
        ball_samples: dict[str, list[tuple[int, int, int, int]]],
    ) -> dict[str, ColorThresholds]:
        """Calibrate color thresholds for different ball types.

        Args:
            frame: Input frame
            ball_samples: Dictionary mapping ball type to list of ROI rectangles (x, y, w, h)

        Returns:
            Dictionary mapping ball type to color thresholds
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        ball_thresholds = {}

        for ball_type, rois in ball_samples.items():
            all_pixels = []

            # Collect pixels from all ROIs for this ball type
            for x, y, w, h in rois:
                roi_pixels = hsv[y : y + h, x : x + w].reshape(-1, 3)
                # Filter out very dark or very light pixels (shadows/highlights)
                min_value = config_manager.get(
                    "vision.calibration.color.auto_calibration.pixel_filter_min_value",
                    30,
                )
                max_value = config_manager.get(
                    "vision.calibration.color.auto_calibration.pixel_filter_max_value",
                    220,
                )
                valid_pixels = roi_pixels[
                    (roi_pixels[:, 2] > min_value) & (roi_pixels[:, 2] < max_value)
                ]
                all_pixels.extend(valid_pixels)

            min_pixels = config_manager.get(
                "vision.calibration.color.auto_calibration.min_pixels_for_calibration",
                100,
            )
            if len(all_pixels) < min_pixels:
                logger.warning(f"Not enough pixels for {ball_type}, using default")
                ball_thresholds[ball_type] = self.default_ball_colors.get(
                    ball_type, ColorThresholds(0, 180, 0, 255, 0, 255)
                )
                continue

            pixels_array = np.array(all_pixels)

            # Use percentiles for robust threshold estimation
            hue_percentiles = config_manager.get(
                "vision.calibration.color.ball_calibration.percentiles.hue", [5, 95]
            )
            sat_percentiles = config_manager.get(
                "vision.calibration.color.ball_calibration.percentiles.saturation",
                [10, 95],
            )
            val_percentiles = config_manager.get(
                "vision.calibration.color.ball_calibration.percentiles.value", [10, 95]
            )
            h_min, h_max = np.percentile(pixels_array[:, 0], hue_percentiles)
            s_min, s_max = np.percentile(pixels_array[:, 1], sat_percentiles)
            v_min, v_max = np.percentile(pixels_array[:, 2], val_percentiles)

            # Handle hue wraparound for red colors
            wraparound_threshold = config_manager.get(
                "vision.calibration.color.ball_calibration.hue_wraparound_threshold", 90
            )
            hue_shift_threshold = config_manager.get(
                "vision.calibration.color.hue_shift_threshold", 90
            )
            if h_max - h_min > wraparound_threshold:  # Likely spans 0 boundary
                h_values = pixels_array[:, 0]
                h_values_shifted = np.where(
                    h_values < hue_shift_threshold, h_values + 180, h_values
                )
                h_mean_shifted = np.mean(h_values_shifted)
                h_std_shifted = np.std(h_values_shifted)

                if h_mean_shifted > 180:
                    h_mean_shifted -= 180

                h_min = (h_mean_shifted - 2 * h_std_shifted) % 180
                h_max = (h_mean_shifted + 2 * h_std_shifted) % 180

            ball_thresholds[ball_type] = ColorThresholds(
                int(h_min), int(h_max), int(s_min), int(s_max), int(v_min), int(v_max)
            )

            logger.info(
                f"Calibrated {ball_type}: H({h_min:.0f}-{h_max:.0f}) S({s_min:.0f}-{s_max:.0f}) V({v_min:.0f}-{v_max:.0f})"
            )

        return ball_thresholds

    def adapt_to_lighting(
        self, frame: NDArray[np.uint8], reference_profile: ColorProfile
    ) -> ColorProfile:
        """Adapt color thresholds to current lighting conditions.

        Args:
            frame: Current frame
            reference_profile: Reference calibration profile

        Returns:
            Adapted color profile
        """
        cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Estimate current lighting level
        current_light = self._estimate_ambient_light(frame)
        reference_light = reference_profile.ambient_light_level

        # Calculate lighting ratio
        light_ratio = current_light / reference_light if reference_light > 0 else 1.0

        logger.info(
            f"Lighting adaptation: current={current_light:.1f}, reference={reference_light:.1f}, ratio={light_ratio:.2f}"
        )

        # Adapt thresholds based on lighting change
        adapted_table = self._adapt_thresholds(
            reference_profile.table_color, light_ratio
        )
        adapted_balls = {}

        for ball_type, thresholds in reference_profile.ball_colors.items():
            adapted_balls[ball_type] = self._adapt_thresholds(thresholds, light_ratio)

        # Create adapted profile
        from datetime import datetime

        adapted_profile = ColorProfile(
            name=f"{reference_profile.name}_adapted",
            table_color=adapted_table,
            ball_colors=adapted_balls,
            lighting_condition=f"adapted_from_{reference_profile.lighting_condition}",
            creation_date=datetime.now().isoformat(),
            ambient_light_level=current_light,
        )

        return adapted_profile

    def _adapt_thresholds(
        self, original: ColorThresholds, light_ratio: float
    ) -> ColorThresholds:
        """Adapt individual color thresholds for lighting changes."""
        # Adjust value thresholds based on lighting ratio
        bright_min_factor = config_manager.get(
            "vision.calibration.color.lighting_adaptation.bright_value_min_factor", 0.8
        )
        bright_max_factor = config_manager.get(
            "vision.calibration.color.lighting_adaptation.bright_value_max_factor", 0.9
        )
        dark_min_factor = config_manager.get(
            "vision.calibration.color.lighting_adaptation.dark_value_min_factor", 1.2
        )
        dark_max_factor = config_manager.get(
            "vision.calibration.color.lighting_adaptation.dark_value_max_factor", 1.1
        )

        if light_ratio > 1.0:  # Brighter lighting
            v_min = min(255, int(original.value_min * light_ratio * bright_min_factor))
            v_max = min(255, int(original.value_max * light_ratio * bright_max_factor))
        else:  # Darker lighting
            v_min = max(0, int(original.value_min * light_ratio * dark_min_factor))
            v_max = max(0, int(original.value_max * light_ratio * dark_max_factor))

        # Slightly adjust saturation for extreme lighting changes
        very_bright_threshold = config_manager.get(
            "vision.calibration.color.lighting_adaptation.very_bright_threshold", 1.5
        )
        very_dark_threshold = config_manager.get(
            "vision.calibration.color.lighting_adaptation.very_dark_threshold", 0.6
        )
        sat_adj_bright = config_manager.get(
            "vision.calibration.color.lighting_adaptation.saturation_adjustment_bright",
            -20,
        )
        sat_adj_dark = config_manager.get(
            "vision.calibration.color.lighting_adaptation.saturation_adjustment_dark",
            -10,
        )

        if light_ratio > very_bright_threshold:  # Very bright
            s_min = max(0, original.saturation_min + sat_adj_bright)
        elif light_ratio < very_dark_threshold:  # Very dark
            s_min = max(0, original.saturation_min + sat_adj_dark)
        else:
            s_min = original.saturation_min

        return ColorThresholds(
            original.hue_min,
            original.hue_max,
            s_min,
            original.saturation_max,
            v_min,
            v_max,
        )

    def _estimate_ambient_light(self, frame: NDArray[np.uint8]) -> float:
        """Estimate ambient lighting level from frame."""
        # Convert to grayscale and calculate mean brightness
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Use histogram to avoid extreme values
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])

        # Calculate weighted average (emphasize mid-tones)
        weights = np.arange(256)
        total_pixels = np.sum(hist)
        weighted_sum = np.sum(hist.flatten() * weights)

        return weighted_sum / total_pixels if total_pixels > 0 else 128.0

    def create_color_picker_interface(
        self, frame: NDArray[np.uint8], window_name: str = "Color Picker"
    ) -> dict[str, Any]:
        """Create interactive color picker interface.

        Args:
            frame: Input frame
            window_name: OpenCV window name

        Returns:
            Dictionary with selected color information
        """
        selected_colors = {}
        current_selection = None

        def mouse_callback(event, x, y, flags, param):
            nonlocal current_selection

            if event == cv2.EVENT_LBUTTONDOWN:
                # Get HSV value at clicked point
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                h, s, v = hsv[y, x]

                current_selection = {
                    "position": (x, y),
                    "hsv": (int(h), int(s), int(v)),
                    "bgr": tuple(frame[y, x].astype(int)),
                }

                logger.info(f"Selected color at ({x}, {y}): HSV({h}, {s}, {v})")

        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, mouse_callback)

        display_frame = frame.copy()

        while True:
            if current_selection:
                # Draw crosshair at selected point
                x, y = current_selection["position"]
                cv2.drawMarker(
                    display_frame, (x, y), (0, 255, 0), cv2.MARKER_CROSS, 20, 2
                )

                # Display color info
                text = f"HSV: {current_selection['hsv']}"
                cv2.putText(
                    display_frame,
                    text,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )

            cv2.imshow(window_name, display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s") and current_selection:
                # Save current selection
                color_name = input("Enter color name: ")
                selected_colors[color_name] = current_selection
                logger.info(f"Saved color '{color_name}'")
            elif key == ord("r"):
                # Reset frame
                display_frame = frame.copy()
                current_selection = None

        cv2.destroyWindow(window_name)
        return selected_colors

    def optimize_thresholds(
        self, frame: NDArray[np.uint8], target_objects: list[dict]
    ) -> dict[str, ColorThresholds]:
        """Optimize color thresholds for target objects.

        Args:
            frame: Input frame
            target_objects: List of target object descriptions with sample regions

        Returns:
            Optimized color thresholds
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        optimized_thresholds = {}

        for obj in target_objects:
            obj_name = obj["name"]
            positive_samples = obj.get("positive_samples", [])
            negative_samples = obj.get("negative_samples", [])

            if not positive_samples:
                logger.warning(f"No positive samples for {obj_name}")
                continue

            # Collect positive pixels
            positive_pixels = []
            for x, y, w, h in positive_samples:
                roi_pixels = hsv[y : y + h, x : x + w].reshape(-1, 3)
                positive_pixels.extend(roi_pixels)

            positive_pixels = np.array(positive_pixels)

            # Start with initial thresholds based on positive samples
            h_min, h_max = np.percentile(positive_pixels[:, 0], [5, 95])
            s_min, s_max = np.percentile(positive_pixels[:, 1], [10, 90])
            v_min, v_max = np.percentile(positive_pixels[:, 2], [10, 90])

            # Refine using negative samples
            if negative_samples:
                negative_pixels = []
                for x, y, w, h in negative_samples:
                    roi_pixels = hsv[y : y + h, x : x + w].reshape(-1, 3)
                    negative_pixels.extend(roi_pixels)

                negative_pixels = np.array(negative_pixels)

                # Adjust thresholds to exclude negative samples
                thresholds = ColorThresholds(
                    int(h_min),
                    int(h_max),
                    int(s_min),
                    int(s_max),
                    int(v_min),
                    int(v_max),
                )

                # Iteratively tighten thresholds
                max_iterations = config_manager.get(
                    "vision.calibration.color.threshold_optimization.max_iterations", 10
                )
                for _iteration in range(max_iterations):
                    mask = thresholds.apply_mask(hsv)

                    # Check how many negative pixels are included
                    false_positives = 0
                    for x, y, w, h in negative_samples:
                        roi_mask = mask[y : y + h, x : x + w]
                        false_positives += np.sum(roi_mask > 0)

                    if false_positives == 0:
                        break

                    # Tighten thresholds slightly
                    range_h = thresholds.hue_max - thresholds.hue_min
                    range_s = thresholds.saturation_max - thresholds.saturation_min
                    range_v = thresholds.value_max - thresholds.value_min

                    tightening_factor = config_manager.get(
                        "vision.calibration.color.threshold_optimization.range_tightening_factor",
                        0.05,
                    )
                    thresholds = ColorThresholds(
                        thresholds.hue_min + int(range_h * tightening_factor),
                        thresholds.hue_max - int(range_h * tightening_factor),
                        thresholds.saturation_min + int(range_s * tightening_factor),
                        thresholds.saturation_max - int(range_s * tightening_factor),
                        thresholds.value_min + int(range_v * tightening_factor),
                        thresholds.value_max - int(range_v * tightening_factor),
                    )

                optimized_thresholds[obj_name] = thresholds
            else:
                optimized_thresholds[obj_name] = ColorThresholds(
                    int(h_min),
                    int(h_max),
                    int(s_min),
                    int(s_max),
                    int(v_min),
                    int(v_max),
                )

        return optimized_thresholds

    def save_profile(
        self, profile: ColorProfile, filename: Optional[str] = None
    ) -> bool:
        """Save color calibration profile.

        Args:
            profile: Color profile to save
            filename: Optional filename (auto-generated if None)

        Returns:
            True if saved successfully
        """
        try:
            if filename is None:
                filename = f"{profile.name}_{profile.creation_date[:10]}.json"

            filepath = self.cache_dir / filename

            with open(filepath, "w") as f:
                json.dump(profile.to_dict(), f, indent=2)

            logger.info(f"Color profile saved to {filepath}")
            return True

        except Exception as e:
            logger.error(f"Failed to save color profile: {e}")
            return False

    def load_profile(self, filename: str) -> Optional[ColorProfile]:
        """Load color calibration profile.

        Args:
            filename: Profile filename

        Returns:
            Loaded color profile or None if failed
        """
        try:
            filepath = self.cache_dir / filename

            with open(filepath) as f:
                data = json.load(f)

            profile = ColorProfile.from_dict(data)
            self.current_profile = profile

            logger.info(f"Color profile loaded from {filepath}")
            return profile

        except Exception as e:
            logger.error(f"Failed to load color profile: {e}")
            return None

    def list_profiles(self) -> list[str]:
        """List available color profiles.

        Returns:
            List of profile filenames
        """
        return [f.name for f in self.cache_dir.glob("*.json")]

    def create_default_profile(
        self, frame: NDArray[np.uint8], profile_name: str = "default"
    ) -> ColorProfile:
        """Create default color profile from frame.

        Args:
            frame: Input frame
            profile_name: Name for the profile

        Returns:
            Created color profile
        """
        # Auto-calibrate table color
        table_thresholds = self.auto_calibrate_table_color(frame)

        # Use default ball colors (can be refined later)
        ball_thresholds = self.default_ball_colors.copy()

        # Estimate lighting
        ambient_light = self._estimate_ambient_light(frame)

        from datetime import datetime

        profile = ColorProfile(
            name=profile_name,
            table_color=table_thresholds,
            ball_colors=ball_thresholds,
            lighting_condition="auto_detected",
            creation_date=datetime.now().isoformat(),
            ambient_light_level=ambient_light,
        )

        self.current_profile = profile
        return profile
