#!/usr/bin/env python3
"""Dataset Creator - Frame Extraction Tool for YOLO Training.

This tool extracts frames from video files with smart sampling to create
diverse training datasets for YOLO object detection models.

Features:
- Smart sampling strategies (uniform, scene change detection, random)
- Frame quality analysis
- Estimated ball count detection
- Edge case detection (shadows, clusters, occlusions)
- Metadata generation with frame information
- Configurable output format and quality

Usage:
    python dataset_creator.py video.mp4 --output dataset/raw --count 1000
    python dataset_creator.py video.mp4 --strategy scene_change --min-quality 0.7
    python dataset_creator.py --help
"""

import argparse
import json
import logging
import math
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Literal, Optional

import cv2
import numpy as np
from numpy.typing import NDArray

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Data Models
# =============================================================================


@dataclass
class FrameMetadata:
    """Metadata for an extracted frame."""

    frame_number: int
    timestamp: float  # seconds from start
    filename: str
    video_source: str
    resolution: tuple[int, int]  # (width, height)

    # Quality metrics
    quality_score: float  # 0.0-1.0
    sharpness_score: float  # 0.0-1.0
    brightness: float  # 0.0-1.0

    # Detection metadata
    estimated_ball_count: int
    has_shadows: bool
    has_clusters: bool
    has_occlusions: bool
    has_motion_blur: bool

    # Edge cases
    edge_case_flags: list[str]

    # Scene characteristics
    scene_complexity: float  # 0.0-1.0
    color_variety: float  # 0.0-1.0

    def to_dict(self) -> Dict[str, object]:
        """Convert to JSON-serializable dictionary."""
        data = asdict(self)
        # Convert numpy types to Python types
        for key, value in data.items():
            if hasattr(value, "item"):  # numpy scalar
                data[key] = value.item()
        return data


@dataclass
class ExtractionConfig:
    """Configuration for frame extraction."""

    video_path: Path
    output_dir: Path
    target_count: int
    strategy: Literal["uniform", "scene_change", "random", "hybrid"]
    min_quality: float
    jpg_quality: int
    scene_change_threshold: float
    include_metadata: bool
    analyze_edge_cases: bool


@dataclass
class ExtractionStats:
    """Statistics for extraction session."""

    total_video_frames: int
    frames_extracted: int
    frames_rejected: int
    extraction_time: float
    video_duration: float
    average_quality: float
    edge_case_count: int


# =============================================================================
# Frame Quality Analysis
# =============================================================================


class FrameAnalyzer:
    """Analyzes frame quality and characteristics."""

    @staticmethod
    def calculate_sharpness(frame: NDArray[np.uint8]) -> float:
        """Calculate frame sharpness using Laplacian variance.

        Args:
            frame: Input frame in BGR format

        Returns:
            Sharpness score (0.0-1.0, normalized)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)  # type: ignore[attr-defined]
        laplacian_var = float(np.var(laplacian))

        # Normalize to 0-1 range (empirically, sharp pool table frames have variance > 100)
        normalized = min(laplacian_var / 200.0, 1.0)
        return float(normalized)

    @staticmethod
    def calculate_brightness(frame: NDArray[np.uint8]) -> float:
        """Calculate average frame brightness.

        Args:
            frame: Input frame in BGR format

        Returns:
            Brightness score (0.0-1.0)
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        v_channel = hsv[:, :, 2]
        avg_brightness = np.mean(v_channel) / 255.0
        return float(avg_brightness)

    @staticmethod
    def detect_motion_blur(frame: NDArray[np.uint8], threshold: float = 100.0) -> bool:
        """Detect motion blur in frame.

        Args:
            frame: Input frame in BGR format
            threshold: Sharpness threshold below which blur is detected

        Returns:
            True if motion blur detected
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)  # type: ignore[attr-defined]
        laplacian_var = float(np.var(laplacian))
        return bool(laplacian_var < threshold)

    @staticmethod
    def calculate_scene_complexity(frame: NDArray[np.uint8]) -> float:  # type: ignore[return]
        """Calculate scene complexity based on edge density.

        Args:
            frame: Input frame in BGR format

        Returns:
            Complexity score (0.0-1.0)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.count_nonzero(edges) / edges.size
        return float(edge_density)

    @staticmethod
    def calculate_color_variety(frame: NDArray[np.uint8]) -> float:
        """Calculate color variety in frame.

        Args:
            frame: Input frame in BGR format

        Returns:
            Color variety score (0.0-1.0)
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h_channel = hsv[:, :, 0]

        # Calculate histogram and entropy
        hist = cv2.calcHist([h_channel], [0], None, [180], [0, 180])
        hist = hist / hist.sum()  # Normalize

        # Calculate entropy as measure of color variety
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        normalized_entropy = entropy / np.log2(180)  # Normalize to 0-1

        return float(normalized_entropy)


# =============================================================================
# Ball Detection Analysis
# =============================================================================


class BallAnalyzer:
    """Analyzes ball-related characteristics in frames."""

    def __init__(self):
        """Initialize ball analyzer."""
        self.min_radius = 15
        self.max_radius = 26
        self.hough_param1 = 50
        self.hough_param2 = 30

    def estimate_ball_count(self, frame: NDArray[np.uint8]) -> int:
        """Estimate number of balls in frame using Hough circles.

        Args:
            frame: Input frame in BGR format

        Returns:
            Estimated ball count
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)

        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.0,
            minDist=int(self.min_radius * 2 * 0.8),
            param1=self.hough_param1,
            param2=self.hough_param2,
            minRadius=self.min_radius,
            maxRadius=self.max_radius,
        )

        if circles is None:
            return 0

        return len(circles[0])

    def detect_ball_clusters(self, frame: NDArray[np.uint8]) -> bool:
        """Detect if frame contains ball clusters (balls close together).

        Args:
            frame: Input frame in BGR format

        Returns:
            True if clusters detected
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)

        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.0,
            minDist=int(self.min_radius * 2 * 0.8),
            param1=self.hough_param1,
            param2=self.hough_param2,
            minRadius=self.min_radius,
            maxRadius=self.max_radius,
        )

        if circles is None or len(circles[0]) < 2:
            return False

        # Check for balls closer than 2.5 radii
        circles = circles[0]
        for i in range(len(circles)):
            for j in range(i + 1, len(circles)):
                x1, y1, r1 = circles[i]
                x2, y2, r2 = circles[j]
                distance = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
                avg_radius = (r1 + r2) / 2
                if distance < avg_radius * 2.5:
                    return True

        return False

    def detect_shadows(self, frame: NDArray[np.uint8]) -> bool:  # type: ignore[return]
        """Detect if frame contains significant shadows.

        Args:
            frame: Input frame in BGR format

        Returns:
            True if shadows detected
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        v_channel = hsv[:, :, 2]

        # Find very dark regions (potential shadows)
        _, dark_mask = cv2.threshold(v_channel, 50, 255, cv2.THRESH_BINARY_INV)

        # Check if dark regions form circular patterns near balls
        dark_ratio = np.count_nonzero(dark_mask) / dark_mask.size

        # Shadows typically cover 5-20% of frame
        return 0.05 < dark_ratio < 0.20

    def detect_occlusions(self, frame: NDArray[np.uint8]) -> bool:  # type: ignore[return]
        """Detect potential ball occlusions (hands, cue, other balls).

        Args:
            frame: Input frame in BGR format

        Returns:
            True if occlusions detected
        """
        # Detect skin tones (hands)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        skin_lower = np.array([0, 20, 50])
        skin_upper = np.array([20, 150, 255])
        skin_mask = cv2.inRange(hsv, skin_lower, skin_upper)

        skin_ratio = np.count_nonzero(skin_mask) / skin_mask.size

        # If more than 2% of frame is skin tone, likely has hands/occlusions
        return skin_ratio > 0.02


# =============================================================================
# Scene Change Detection
# =============================================================================


class SceneChangeDetector:
    """Detects scene changes in video."""

    def __init__(self, threshold: float = 30.0):
        """Initialize scene change detector.

        Args:
            threshold: Scene change threshold (higher = less sensitive)
        """
        self.threshold = threshold
        self.previous_frame: Optional[NDArray[np.uint8]] = None

    def detect_change(self, frame: NDArray[np.uint8]) -> tuple[bool, float]:
        """Detect if current frame is significantly different from previous.

        Args:
            frame: Current frame in BGR format

        Returns:
            Tuple of (is_scene_change, difference_score)
        """
        if self.previous_frame is None:
            self.previous_frame = frame.copy()
            return True, 100.0  # First frame is always a "change"

        # Convert to grayscale
        gray1 = cv2.cvtColor(self.previous_frame, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculate histogram difference
        hist1 = cv2.calcHist([gray1], [0], None, [256], [0, 256])
        hist2 = cv2.calcHist([gray2], [0], None, [256], [0, 256])

        # Normalize histograms
        hist1 = hist1 / hist1.sum()
        hist2 = hist2 / hist2.sum()

        # Calculate histogram correlation
        correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

        # Convert correlation to difference score (0-100)
        difference_score = (1.0 - correlation) * 100.0

        # Update previous frame
        self.previous_frame = frame.copy()

        is_change = difference_score > self.threshold
        return is_change, float(difference_score)


# =============================================================================
# Frame Extractor
# =============================================================================


class FrameExtractor:
    """Extracts frames from video with smart sampling."""

    def __init__(self, config: ExtractionConfig):
        """Initialize frame extractor.

        Args:
            config: Extraction configuration
        """
        self.config = config
        self.frame_analyzer = FrameAnalyzer()
        self.ball_analyzer = BallAnalyzer()
        self.scene_detector = SceneChangeDetector(config.scene_change_threshold)

        self.extracted_frames: list[FrameMetadata] = []
        self.stats = ExtractionStats(
            total_video_frames=0,
            frames_extracted=0,
            frames_rejected=0,
            extraction_time=0.0,
            video_duration=0.0,
            average_quality=0.0,
            edge_case_count=0,
        )

    def extract_frames(self) -> ExtractionStats:
        """Extract frames from video according to configuration.

        Returns:
            Extraction statistics
        """
        start_time = datetime.now()

        # Open video
        cap = cv2.VideoCapture(str(self.config.video_path))
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {self.config.video_path}")

        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0

        self.stats.total_video_frames = total_frames
        self.stats.video_duration = duration

        logger.info(f"Video: {self.config.video_path}")
        logger.info(f"Total frames: {total_frames}, FPS: {fps:.2f}, Duration: {duration:.2f}s")
        logger.info(f"Target extraction: {self.config.target_count} frames")
        logger.info(f"Strategy: {self.config.strategy}")

        # Create output directory
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        # Select extraction strategy
        if self.config.strategy == "uniform":
            self._extract_uniform(cap, total_frames, fps)
        elif self.config.strategy == "scene_change":
            self._extract_scene_changes(cap, total_frames, fps)
        elif self.config.strategy == "random":
            self._extract_random(cap, total_frames, fps)
        elif self.config.strategy == "hybrid":
            self._extract_hybrid(cap, total_frames, fps)

        cap.release()

        # Calculate statistics
        end_time = datetime.now()
        self.stats.extraction_time = (end_time - start_time).total_seconds()

        if self.stats.frames_extracted > 0:
            self.stats.average_quality = sum(
                f.quality_score for f in self.extracted_frames
            ) / self.stats.frames_extracted
            self.stats.edge_case_count = sum(
                len(f.edge_case_flags) for f in self.extracted_frames
            )

        # Save metadata
        if self.config.include_metadata:
            self._save_metadata()

        # Print summary
        self._print_summary()

        return self.stats

    def _extract_uniform(self, cap: cv2.VideoCapture, total_frames: int, fps: float) -> None:
        """Extract frames at uniform intervals.

        Args:
            cap: Video capture object
            total_frames: Total frames in video
            fps: Video frames per second
        """
        interval = max(1, total_frames // self.config.target_count)
        logger.info(f"Extracting every {interval}th frame...")

        frame_number = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_number % interval == 0:
                timestamp = frame_number / fps
                self._process_and_save_frame(frame, frame_number, timestamp)

            frame_number += 1

            if self.stats.frames_extracted >= self.config.target_count:
                break

            if frame_number % 100 == 0:
                logger.info(f"Processed {frame_number}/{total_frames} frames, extracted {self.stats.frames_extracted}")

    def _extract_scene_changes(self, cap: cv2.VideoCapture, total_frames: int, fps: float) -> None:
        """Extract frames at scene changes.

        Args:
            cap: Video capture object
            total_frames: Total frames in video
            fps: Video frames per second
        """
        logger.info("Detecting scene changes...")
        frame_number = 0
        frames_since_last_extract = 0
        min_interval = 10  # Minimum frames between extractions

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            timestamp = frame_number / fps
            is_change, score = self.scene_detector.detect_change(frame)

            if is_change and frames_since_last_extract >= min_interval:
                self._process_and_save_frame(frame, frame_number, timestamp)
                frames_since_last_extract = 0
            else:
                frames_since_last_extract += 1

            frame_number += 1

            if self.stats.frames_extracted >= self.config.target_count:
                break

            if frame_number % 100 == 0:
                logger.info(f"Processed {frame_number}/{total_frames} frames, extracted {self.stats.frames_extracted}")

    def _extract_random(self, cap: cv2.VideoCapture, total_frames: int, fps: float) -> None:
        """Extract frames at random positions.

        Args:
            cap: Video capture object
            total_frames: Total frames in video
            fps: Video frames per second
        """
        logger.info("Extracting random frames...")

        # Generate random frame numbers
        random_frames = sorted(np.random.choice(total_frames, self.config.target_count, replace=False))

        for frame_number in random_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            if not ret:
                continue

            timestamp = frame_number / fps
            self._process_and_save_frame(frame, frame_number, timestamp)

            if (len(random_frames) > 100 and frame_number % (len(random_frames) // 10) == 0):
                logger.info(f"Extracted {self.stats.frames_extracted}/{self.config.target_count} frames")

    def _extract_hybrid(self, cap: cv2.VideoCapture, total_frames: int, fps: float) -> None:
        """Extract frames using hybrid strategy (uniform + scene changes).

        Args:
            cap: Video capture object
            total_frames: Total frames in video
            fps: Video frames per second
        """
        logger.info("Using hybrid extraction strategy...")

        # First pass: uniform sampling
        uniform_count = self.config.target_count // 2
        interval = max(1, total_frames // uniform_count)

        frame_number = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_number % interval == 0:
                timestamp = frame_number / fps
                self._process_and_save_frame(frame, frame_number, timestamp)

            frame_number += 1

            if self.stats.frames_extracted >= uniform_count:
                break

        # Second pass: scene changes for remaining quota
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.scene_detector.previous_frame = None
        frame_number = 0
        frames_since_last_extract = 0
        min_interval = 10

        while cap.isOpened() and self.stats.frames_extracted < self.config.target_count:
            ret, frame = cap.read()
            if not ret:
                break

            timestamp = frame_number / fps
            is_change, score = self.scene_detector.detect_change(frame)

            if is_change and frames_since_last_extract >= min_interval:
                # Check if this frame is unique (not already extracted)
                if not self._is_duplicate_frame(frame_number):
                    self._process_and_save_frame(frame, frame_number, timestamp)
                    frames_since_last_extract = 0
            else:
                frames_since_last_extract += 1

            frame_number += 1

    def _is_duplicate_frame(self, frame_number: int) -> bool:
        """Check if frame number was already extracted.

        Args:
            frame_number: Frame number to check

        Returns:
            True if already extracted
        """
        return any(f.frame_number == frame_number for f in self.extracted_frames)

    def _process_and_save_frame(self, frame: NDArray[np.uint8], frame_number: int, timestamp: float) -> None:
        """Process and save a frame if it meets quality criteria.

        Args:
            frame: Frame to process
            frame_number: Frame number in video
            timestamp: Timestamp in seconds
        """
        # Analyze frame quality
        sharpness = self.frame_analyzer.calculate_sharpness(frame)
        brightness = self.frame_analyzer.calculate_brightness(frame)
        quality_score = (sharpness + brightness) / 2.0

        # Check quality threshold
        if quality_score < self.config.min_quality:
            self.stats.frames_rejected += 1
            return

        # Analyze edge cases if configured
        edge_case_flags = []
        has_shadows = False
        has_clusters = False
        has_occlusions = False
        has_motion_blur = False
        estimated_balls = 0

        if self.config.analyze_edge_cases:
            estimated_balls = self.ball_analyzer.estimate_ball_count(frame)
            has_shadows = self.ball_analyzer.detect_shadows(frame)
            has_clusters = self.ball_analyzer.detect_ball_clusters(frame)
            has_occlusions = self.ball_analyzer.detect_occlusions(frame)
            has_motion_blur = self.frame_analyzer.detect_motion_blur(frame)

            if has_shadows:
                edge_case_flags.append("shadows")
            if has_clusters:
                edge_case_flags.append("clusters")
            if has_occlusions:
                edge_case_flags.append("occlusions")
            if has_motion_blur:
                edge_case_flags.append("motion_blur")

        # Calculate scene characteristics
        scene_complexity = self.frame_analyzer.calculate_scene_complexity(frame)
        color_variety = self.frame_analyzer.calculate_color_variety(frame)

        # Generate filename
        filename = f"frame_{frame_number:06d}_{timestamp:.3f}.jpg"
        filepath = self.config.output_dir / filename

        # Save frame
        cv2.imwrite(
            str(filepath),
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, self.config.jpg_quality],
        )

        # Create metadata
        height, width = frame.shape[:2]
        metadata = FrameMetadata(
            frame_number=frame_number,
            timestamp=timestamp,
            filename=filename,
            video_source=str(self.config.video_path),
            resolution=(width, height),
            quality_score=quality_score,
            sharpness_score=sharpness,
            brightness=brightness,
            estimated_ball_count=estimated_balls,
            has_shadows=has_shadows,
            has_clusters=has_clusters,
            has_occlusions=has_occlusions,
            has_motion_blur=has_motion_blur,
            edge_case_flags=edge_case_flags,
            scene_complexity=scene_complexity,
            color_variety=color_variety,
        )

        self.extracted_frames.append(metadata)
        self.stats.frames_extracted += 1

    def _save_metadata(self) -> None:
        """Save extraction metadata to JSON file."""
        metadata_file = self.config.output_dir / "metadata.json"

        metadata = {
            "extraction_date": datetime.now().isoformat(),
            "video_source": str(self.config.video_path),
            "configuration": {
                "target_count": self.config.target_count,
                "strategy": self.config.strategy,
                "min_quality": self.config.min_quality,
                "jpg_quality": self.config.jpg_quality,
                "scene_change_threshold": self.config.scene_change_threshold,
            },
            "statistics": asdict(self.stats),
            "frames": [f.to_dict() for f in self.extracted_frames],
        }

        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Metadata saved to: {metadata_file}")

    def _print_summary(self) -> None:
        """Print extraction summary."""
        logger.info("\n" + "=" * 70)
        logger.info("EXTRACTION SUMMARY")
        logger.info("=" * 70)
        logger.info(f"Total video frames:     {self.stats.total_video_frames}")
        logger.info(f"Frames extracted:       {self.stats.frames_extracted}")
        logger.info(f"Frames rejected:        {self.stats.frames_rejected}")
        logger.info(f"Extraction time:        {self.stats.extraction_time:.2f}s")
        logger.info(f"Average quality:        {self.stats.average_quality:.3f}")
        logger.info(f"Edge case frames:       {self.stats.edge_case_count}")
        logger.info(f"Output directory:       {self.config.output_dir}")
        logger.info("=" * 70)

        # Print edge case distribution
        if self.config.analyze_edge_cases and self.extracted_frames:
            logger.info("\nEDGE CASE DISTRIBUTION:")
            edge_case_counts: Dict[str, int] = {}
            for frame in self.extracted_frames:
                for flag in frame.edge_case_flags:
                    edge_case_counts[flag] = edge_case_counts.get(flag, 0) + 1

            for edge_case, count in sorted(edge_case_counts.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / self.stats.frames_extracted) * 100
                logger.info(f"  {edge_case:15s}: {count:4d} ({percentage:.1f}%)")


# =============================================================================
# CLI Interface
# =============================================================================


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Extract frames from video for YOLO training dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract 1000 frames with uniform sampling
  %(prog)s video.mp4 --output dataset/raw --count 1000

  # Extract frames at scene changes with high quality threshold
  %(prog)s video.mp4 --strategy scene_change --min-quality 0.7

  # Extract 1500 frames with hybrid strategy and edge case analysis
  %(prog)s video.mp4 --count 1500 --strategy hybrid --analyze-edge-cases

  # Extract random frames with custom quality settings
  %(prog)s video.mp4 --strategy random --jpg-quality 98 --min-quality 0.8
        """,
    )

    parser.add_argument(
        "video",
        type=Path,
        help="Path to input video file",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("dataset/raw"),
        help="Output directory for extracted frames (default: dataset/raw)",
    )

    parser.add_argument(
        "--count",
        "-c",
        type=int,
        default=1000,
        help="Target number of frames to extract (default: 1000)",
    )

    parser.add_argument(
        "--strategy",
        "-s",
        choices=["uniform", "scene_change", "random", "hybrid"],
        default="hybrid",
        help="Frame sampling strategy (default: hybrid)",
    )

    parser.add_argument(
        "--min-quality",
        "-q",
        type=float,
        default=0.5,
        help="Minimum quality threshold 0.0-1.0 (default: 0.5)",
    )

    parser.add_argument(
        "--jpg-quality",
        type=int,
        default=95,
        help="JPEG quality 0-100 (default: 95)",
    )

    parser.add_argument(
        "--scene-threshold",
        type=float,
        default=30.0,
        help="Scene change detection threshold (default: 30.0)",
    )

    parser.add_argument(
        "--analyze-edge-cases",
        action="store_true",
        help="Enable edge case analysis (shadows, clusters, occlusions)",
    )

    parser.add_argument(
        "--no-metadata",
        action="store_true",
        help="Don't generate metadata.json file",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point.

    Returns:
        Exit code (0 for success, non-zero for error)
    """
    args = parse_arguments()

    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate inputs
    if not args.video.exists():
        logger.error(f"Video file not found: {args.video}")
        return 1

    if args.count <= 0:
        logger.error("Frame count must be positive")
        return 1

    if not (0.0 <= args.min_quality <= 1.0):
        logger.error("Min quality must be between 0.0 and 1.0")
        return 1

    if not (0 <= args.jpg_quality <= 100):
        logger.error("JPEG quality must be between 0 and 100")
        return 1

    # Create configuration
    config = ExtractionConfig(
        video_path=args.video,
        output_dir=args.output,
        target_count=args.count,
        strategy=args.strategy,
        min_quality=args.min_quality,
        jpg_quality=args.jpg_quality,
        scene_change_threshold=args.scene_threshold,
        include_metadata=not args.no_metadata,
        analyze_edge_cases=args.analyze_edge_cases,
    )

    # Extract frames
    try:
        extractor = FrameExtractor(config)
        stats = extractor.extract_frames()

        if stats.frames_extracted == 0:
            logger.warning("No frames extracted. Try lowering --min-quality threshold.")
            return 1

        logger.info(f"\nSuccessfully extracted {stats.frames_extracted} frames!")
        return 0

    except Exception as e:
        logger.error(f"Extraction failed: {e}", exc_info=args.verbose)
        return 1


if __name__ == "__main__":
    sys.exit(main())
