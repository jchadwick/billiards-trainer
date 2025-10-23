#!/usr/bin/env python3
"""Performance diagnostics tool for billiards vision system.

Runs the vision pipeline with comprehensive timing instrumentation to identify bottlenecks.

Usage:
    python backend/tools/performance_diagnostics.py [--frames 100] [--output report.json]
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import config
from vision import VisionModule
from vision.performance_profiler import PerformanceProfiler

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class InstrumentedVisionModule(VisionModule):
    """Vision module with performance instrumentation."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.profiler = PerformanceProfiler(history_size=200, log_interval=10)

        # Patch YOLO detector if available
        if hasattr(self, "detector") and self.detector:
            self._instrument_yolo_detector()

    def _instrument_yolo_detector(self):
        """Add timing to YOLO detector methods."""
        original_inference = self.detector._run_standard_inference

        def timed_inference(frame):
            self.profiler.start_stage("yolo_preprocessing")
            # Preprocessing happens inside _run_standard_inference
            self.profiler.end_stage("yolo_preprocessing")

            self.profiler.start_stage("yolo_inference")
            result = original_inference(frame)
            self.profiler.end_stage("yolo_inference")

            return result

        self.detector._run_standard_inference = timed_inference

    def _process_single_frame(self, frame, frame_number=None, timestamp=None):
        """Instrumented version of frame processing."""
        if frame_number is None:
            frame_number = self._frame_number
            self._frame_number += 1

        if timestamp is None:
            timestamp = time.time()

        # Start profiling this frame
        self.profiler.start_frame(frame_number)

        try:
            # Preprocessing
            self.profiler.start_stage("preprocessing")
            if self.config.preprocessing_enabled:
                processed_frame = self.preprocessor.process(frame)
            else:
                processed_frame = frame
            self.profiler.end_stage("preprocessing")

            # Masking
            self.profiler.start_stage("masking")
            processed_frame = self._apply_all_masks(processed_frame)
            self.profiler.end_stage("masking")

            detected_balls = []

            # Table detection
            if self.table_detector and self.config.enable_table_detection:
                self.profiler.start_stage("table_detection")
                try:
                    table_confidence_threshold = config.get(
                        "vision.detection.table_detection_confidence_threshold", 0.5
                    )
                    table_result = self.table_detector.detect_complete_table(
                        processed_frame
                    )
                    if (
                        table_result
                        and table_result.confidence > table_confidence_threshold
                    ):
                        from vision.models import Table

                        Table(
                            corners=table_result.corners.to_list(),
                            pockets=[
                                pocket.position for pocket in table_result.pockets
                            ],
                            width=table_result.width,
                            height=table_result.height,
                            surface_color=table_result.surface_color,
                        )
                except Exception as e:
                    logger.warning(f"Table detection failed: {e}")
                self.profiler.end_stage("table_detection")

            # Ball detection
            if self.detector and self.config.enable_ball_detection:
                self.profiler.start_stage("ball_detection")
                try:
                    detected_balls = self.detector.detect_balls_with_classification(
                        processed_frame
                    )

                    # Tracking
                    if self.tracker and self.config.enable_tracking:
                        self.profiler.start_stage("tracking")
                        detected_balls = self.tracker.update_tracking(
                            detected_balls, frame_number, timestamp
                        )
                        self.profiler.end_stage("tracking")

                except Exception as e:
                    logger.error(f"Ball detection failed: {e}")
                self.profiler.end_stage("ball_detection")

            # Cue detection
            if self.cue_detector and self.config.enable_cue_detection:
                self.profiler.start_stage("cue_detection")
                try:
                    from vision.models import BallType

                    cue_ball_pos = None
                    for ball in detected_balls:
                        if ball.ball_type == BallType.CUE:
                            cue_ball_pos = ball.position
                            break

                    self.cue_detector.detect_cue(
                        processed_frame, cue_ball_pos
                    )
                except Exception as e:
                    logger.error(f"Cue detection failed: {e}")
                self.profiler.end_stage("cue_detection")

            # Result building
            self.profiler.start_stage("result_building")
            # ... (rest of result building code)
            self.profiler.end_stage("result_building")

            # End profiling
            self.profiler.end_frame()

            # Call parent implementation for actual result
            return super()._process_single_frame(frame, frame_number, timestamp)

        except Exception as e:
            logger.error(f"Frame processing failed: {e}")
            self.profiler.end_frame()
            return None


def verify_device_configuration() -> dict[str, Any]:
    """Verify what device is actually being used."""
    logger.info("=== Device Configuration ===")

    device_info = {}

    # Check PyTorch
    try:
        import torch

        device_info["torch_version"] = torch.__version__
        device_info["cuda_available"] = torch.cuda.is_available()
        device_info["mps_available"] = (
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        )

        if device_info["cuda_available"]:
            device_info["cuda_device"] = torch.cuda.get_device_name(0)
            logger.info(f"✓ CUDA available: {device_info['cuda_device']}")
        elif device_info["mps_available"]:
            logger.info("✓ Apple Silicon MPS available")
        else:
            logger.warning("⚠ No GPU acceleration available - using CPU")

    except ImportError:
        logger.warning("PyTorch not installed")
        device_info["torch_available"] = False

    # Check CoreML
    try:
        import coremltools

        device_info["coreml_version"] = coremltools.__version__
        logger.info(f"✓ CoreML Tools installed: {coremltools.__version__}")
    except ImportError:
        logger.warning("CoreML Tools not installed")
        device_info["coreml_available"] = False

    # Check config
    yolo_device = config.get("vision.detection.yolo_device", "auto")
    yolo_model = config.get("vision.detection.yolo_model_path", "unknown")
    device_info["configured_device"] = yolo_device
    device_info["model_path"] = yolo_model

    logger.info(f"Configured device: {yolo_device}")
    logger.info(f"Model path: {yolo_model}")

    # Check if model file exists
    if yolo_model != "unknown":
        model_path = Path(yolo_model)
        if model_path.exists():
            size_mb = model_path.stat().st_size / (1024 * 1024)
            device_info["model_exists"] = True
            device_info["model_size_mb"] = size_mb
            logger.info(f"✓ Model file exists: {size_mb:.1f} MB")

            # Check model format
            if model_path.suffix == ".mlpackage" or str(model_path).endswith(
                ".mlpackage"
            ):
                device_info["model_format"] = "CoreML"
                logger.info(
                    "✓ Model format: CoreML (.mlpackage) - Apple Silicon optimized"
                )
            elif model_path.suffix == ".pt":
                device_info["model_format"] = "PyTorch"
                logger.info("Model format: PyTorch (.pt)")
            elif model_path.suffix == ".onnx":
                device_info["model_format"] = "ONNX"
                logger.info("Model format: ONNX (.onnx)")
        else:
            device_info["model_exists"] = False
            logger.error(f"✗ Model file not found: {yolo_model}")

    return device_info


def run_diagnostics(num_frames: int = 100, output_file: str = None) -> dict[str, Any]:
    """Run performance diagnostics.

    Args:
        num_frames: Number of frames to process
        output_file: Optional JSON file to save results

    Returns:
        Dictionary with diagnostic results
    """
    logger.info("=== Starting Performance Diagnostics ===")
    logger.info(f"Will process {num_frames} frames")

    # Verify device configuration
    device_info = verify_device_configuration()

    # Initialize vision module
    logger.info("\n=== Initializing Vision Module ===")
    vision = InstrumentedVisionModule(config_dict=config.config_data)

    # Start capture
    logger.info("\n=== Starting Camera Capture ===")
    if not vision.start_capture():
        logger.error("Failed to start camera capture")
        return {"error": "Failed to start camera capture"}

    # Wait for camera to warm up
    logger.info("Warming up camera...")
    time.sleep(2.0)

    # Process frames
    logger.info(f"\n=== Processing {num_frames} Frames ===")
    start_time = time.time()
    frames_processed = 0

    try:
        for i in range(num_frames):
            frame = vision.get_current_frame()
            if frame is not None:
                result = vision._process_single_frame(frame, frame_number=i)
                if result:
                    frames_processed += 1

            # Progress indicator
            if (i + 1) % 10 == 0:
                elapsed = time.time() - start_time
                fps = (i + 1) / elapsed
                logger.info(f"Progress: {i+1}/{num_frames} frames ({fps:.1f} FPS)")

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        vision.stop_capture()

    total_time = time.time() - start_time
    overall_fps = frames_processed / total_time if total_time > 0 else 0

    # Get final statistics
    logger.info("\n=== Final Results ===")
    stats = vision.profiler.get_current_stats()
    summary = stats.get_summary()

    logger.info(f"Processed {frames_processed} frames in {total_time:.1f}s")
    logger.info(f"Overall FPS: {overall_fps:.1f}")
    logger.info(f"Average FPS: {summary['avg_fps']:.1f}")
    logger.info(f"Average frame time: {summary['avg_total_ms']:.1f}ms")
    logger.info(
        f"Min/Max frame time: {summary['min_total_ms']:.1f}ms / {summary['max_total_ms']:.1f}ms"
    )

    # Get bottlenecks
    bottlenecks = vision.profiler.get_bottlenecks(top_n=5)
    logger.info("\n=== Top Bottlenecks ===")
    for i, (stage, time_ms) in enumerate(bottlenecks, 1):
        pct = (
            (time_ms / summary["avg_total_ms"] * 100)
            if summary["avg_total_ms"] > 0
            else 0
        )
        logger.info(f"{i}. {stage:20s}: {time_ms:6.1f}ms ({pct:5.1f}%)")

    # Realtime status
    realtime_status = vision.profiler.get_realtime_status()
    target_fps = 15.0
    logger.info("\n=== Realtime Performance ===")
    logger.info(f"Target FPS: {target_fps}")
    logger.info(f"Actual FPS: {realtime_status['fps']:.1f}")
    if realtime_status["meeting_target"]:
        logger.info("✓ Meeting target FPS")
    else:
        overhead = realtime_status["overhead_ms"]
        logger.warning(
            f"✗ NOT meeting target - need to reduce processing time by {overhead:.1f}ms"
        )

    # Build results
    results = {
        "device_info": device_info,
        "summary": summary,
        "bottlenecks": [
            {
                "stage": stage,
                "time_ms": time_ms,
                "percentage": (
                    (time_ms / summary["avg_total_ms"] * 100)
                    if summary["avg_total_ms"] > 0
                    else 0
                ),
            }
            for stage, time_ms in bottlenecks
        ],
        "realtime_status": realtime_status,
        "overall_fps": overall_fps,
        "frames_processed": frames_processed,
        "total_time": total_time,
    }

    # Save to file if requested
    if output_file:
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"\n✓ Results saved to: {output_file}")

    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Performance diagnostics for billiards vision system"
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=100,
        help="Number of frames to process (default: 100)",
    )
    parser.add_argument("--output", type=str, help="Output JSON file for results")
    parser.add_argument(
        "--device-check-only",
        action="store_true",
        help="Only check device configuration",
    )

    args = parser.parse_args()

    if args.device_check_only:
        verify_device_configuration()
        return

    results = run_diagnostics(num_frames=args.frames, output_file=args.output)

    # Exit with error code if not meeting target
    if not results.get("realtime_status", {}).get("meeting_target", False):
        sys.exit(1)


if __name__ == "__main__":
    main()
