#!/usr/bin/env python3
"""Performance diagnostics tool for billiards vision system.

Runs the vision pipeline with comprehensive timing instrumentation to identify bottlenecks.

Usage:
    python backend/tools/performance_diagnostics.py [--frames 100] [--output report.json]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.config import config
from backend.vision import VisionModule
from backend.vision.performance_profiler import PerformanceProfiler

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class InstrumentedVisionModule(VisionModule):
    """Vision module using the pipeline's supported performance profiler."""

    def __init__(self, config_dict: dict[str, Any] | None = None) -> None:
        super().__init__(config=config_dict)
        # VisionModule already records the actual production pipeline stages.
        # Ensure diagnostics remain usable when profiling is disabled in the
        # application config, without duplicating or monkey-patching detection.
        if self.profiler is None:
            self.profiler = PerformanceProfiler(
                history_size=config.get("vision.performance.profile_history_size", 100),
                log_interval=config.get("vision.performance.profile_log_interval", 30),
            )


def verify_device_configuration() -> dict[str, Any]:
    """Verify what device is actually being used."""
    logger.info("=== Device Configuration ===")

    device_info: dict[str, Any] = {}

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
        import importlib

        coremltools = importlib.import_module("coremltools")

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


def run_diagnostics(
    num_frames: int = 100, output_file: str | None = None
) -> dict[str, Any]:
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
    vision = InstrumentedVisionModule()
    profiler = vision.profiler
    if profiler is None:
        raise RuntimeError("Performance diagnostics requires an active profiler")

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
    stats = profiler.get_current_stats()
    summary = stats.get_summary()

    logger.info(f"Processed {frames_processed} frames in {total_time:.1f}s")
    logger.info(f"Overall FPS: {overall_fps:.1f}")
    logger.info(f"Average FPS: {summary['avg_fps']:.1f}")
    logger.info(f"Average frame time: {summary['avg_total_ms']:.1f}ms")
    logger.info(
        f"Min/Max frame time: {summary['min_total_ms']:.1f}ms / {summary['max_total_ms']:.1f}ms"
    )

    # Get bottlenecks
    bottlenecks = profiler.get_bottlenecks(top_n=5)
    logger.info("\n=== Top Bottlenecks ===")
    for i, (stage, time_ms) in enumerate(bottlenecks, 1):
        pct = (
            (time_ms / summary["avg_total_ms"] * 100)
            if summary["avg_total_ms"] > 0
            else 0
        )
        logger.info(f"{i}. {stage:20s}: {time_ms:6.1f}ms ({pct:5.1f}%)")

    # Realtime status
    realtime_status = profiler.get_realtime_status()
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


def main() -> None:
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
