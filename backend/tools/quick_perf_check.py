#!/usr/bin/env python3
"""Quick performance check - verifies device and runs basic timing test.

Usage:
    python backend/tools/quick_perf_check.py
"""

import sys
import time
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def check_device_config():
    """Check device configuration and acceleration."""
    logger.info("=" * 60)
    logger.info("DEVICE CONFIGURATION CHECK")
    logger.info("=" * 60)

    results = {}

    # Check PyTorch/MPS
    try:
        import torch

        logger.info(f"\n✓ PyTorch: {torch.__version__}")
        results["torch_version"] = torch.__version__

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            logger.info("✓ Apple Silicon MPS: AVAILABLE")
            results["mps_available"] = True
        else:
            logger.warning("✗ Apple Silicon MPS: NOT AVAILABLE")
            results["mps_available"] = False

        if torch.cuda.is_available():
            logger.info(f"✓ CUDA: {torch.cuda.get_device_name(0)}")
            results["cuda_available"] = True
        else:
            results["cuda_available"] = False

    except ImportError:
        logger.error("✗ PyTorch: NOT INSTALLED")
        results["torch_available"] = False

    # Check Ultralytics
    try:
        import ultralytics

        logger.info(f"✓ Ultralytics: {ultralytics.__version__}")
        results["ultralytics_version"] = ultralytics.__version__
    except ImportError:
        logger.error("✗ Ultralytics: NOT INSTALLED")
        results["ultralytics_available"] = False

    # Check config
    try:
        from config import config

        yolo_device = config.get("vision.detection.yolo_device", "auto")
        yolo_model = config.get("vision.detection.yolo_model_path", "unknown")
        logger.info(f"\nConfigured YOLO device: {yolo_device}")
        logger.info(f"Configured YOLO model: {yolo_model}")
        results["configured_device"] = yolo_device
        results["configured_model"] = yolo_model

        # Check if model exists
        model_path = Path(yolo_model)
        if model_path.exists():
            size_mb = model_path.stat().st_size / (1024 * 1024)
            logger.info(f"✓ Model file exists: {size_mb:.1f} MB")

            # Determine format
            if str(model_path).endswith(".mlpackage"):
                logger.info(
                    "✓ Model format: CoreML (.mlpackage) - Apple Silicon optimized"
                )
                results["model_format"] = "CoreML"
            elif model_path.suffix == ".pt":
                logger.info("  Model format: PyTorch (.pt)")
                results["model_format"] = "PyTorch"
            elif model_path.suffix == ".onnx":
                logger.info("  Model format: ONNX (.onnx)")
                results["model_format"] = "ONNX"

            results["model_exists"] = True
            results["model_size_mb"] = size_mb
        else:
            logger.error(f"✗ Model file NOT FOUND: {yolo_model}")
            results["model_exists"] = False

    except Exception as e:
        logger.error(f"✗ Config check failed: {e}")
        results["config_error"] = str(e)

    return results


def test_yolo_inference():
    """Test YOLO inference speed."""
    logger.info("\n" + "=" * 60)
    logger.info("YOLO INFERENCE SPEED TEST")
    logger.info("=" * 60)

    try:
        from config import config
        from vision.detection.yolo_detector import YOLODetector

        model_path = config.get("vision.detection.yolo_model_path")
        device = config.get("vision.detection.yolo_device", "auto")

        logger.info(f"\nLoading model: {model_path}")
        logger.info(f"Device: {device}")

        detector = YOLODetector(
            model_path=model_path,
            device=device,
            confidence=0.4,
            nms_threshold=0.45,
        )

        if not detector.is_available():
            logger.error("✗ Model failed to load")
            return None

        # Get model info
        info = detector.get_model_info()
        logger.info("✓ Model loaded successfully")
        logger.info(f"  Format: {info.get('model_format', 'unknown')}")
        logger.info(f"  Device: {info.get('device', 'unknown')}")

        # Create test frame (1920x1080 like real camera)
        test_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

        # Warmup runs
        logger.info("\nWarming up (5 frames)...")
        for _ in range(5):
            _ = detector.detect_balls(test_frame)

        # Timed runs
        num_runs = 20
        logger.info(f"\nRunning {num_runs} timed inferences...")

        times = []
        for i in range(num_runs):
            start = time.time()
            detector.detect_balls(test_frame)
            elapsed_ms = (time.time() - start) * 1000
            times.append(elapsed_ms)

            if (i + 1) % 5 == 0:
                avg_so_far = sum(times) / len(times)
                logger.info(
                    f"  {i+1}/{num_runs}: {elapsed_ms:.1f}ms (avg: {avg_so_far:.1f}ms)"
                )

        # Statistics
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        fps = 1000.0 / avg_time if avg_time > 0 else 0

        logger.info(f"\n{'=' * 40}")
        logger.info("YOLO Inference Statistics:")
        logger.info(f"  Average: {avg_time:.1f}ms")
        logger.info(f"  Min:     {min_time:.1f}ms")
        logger.info(f"  Max:     {max_time:.1f}ms")
        logger.info(f"  FPS:     {fps:.1f}")
        logger.info(f"{'=' * 40}")

        # Check against targets
        target_ms = 66.7  # 15 FPS target
        if avg_time <= target_ms:
            logger.info(
                f"✓ YOLO inference meets 15 FPS target ({avg_time:.1f}ms <= {target_ms:.1f}ms)"
            )
        else:
            overhead = avg_time - target_ms
            logger.warning("✗ YOLO inference TOO SLOW for 15 FPS")
            logger.warning(f"  Need to reduce by: {overhead:.1f}ms")

        # Get detector stats
        stats = detector.get_statistics()
        logger.info("\nDetector Statistics:")
        logger.info(f"  Total inferences: {stats['total_inferences']}")
        logger.info(f"  Average time: {stats['avg_inference_time']:.1f}ms")

        return {
            "avg_ms": avg_time,
            "min_ms": min_time,
            "max_ms": max_time,
            "fps": fps,
            "meets_target": avg_time <= target_ms,
            "overhead_ms": max(0, avg_time - target_ms),
        }

    except Exception as e:
        logger.error(f"✗ YOLO test failed: {e}", exc_info=True)
        return None


def test_full_pipeline():
    """Test full vision pipeline."""
    logger.info("\n" + "=" * 60)
    logger.info("FULL PIPELINE SPEED TEST")
    logger.info("=" * 60)

    try:
        from config import config
        from vision import VisionModule

        logger.info("\nInitializing vision module...")
        vision = VisionModule(config={})

        logger.info("Starting camera capture...")
        if not vision.start_capture():
            logger.error("✗ Failed to start camera")
            return None

        # Wait for camera to stabilize
        time.sleep(1.0)

        # Get a frame
        logger.info("Capturing test frames...")
        frame = vision.get_current_frame()
        if frame is None:
            logger.error("✗ No frame available")
            vision.stop_capture()
            return None

        logger.info(f"✓ Got frame: {frame.shape}")

        # Warmup
        logger.info("\nWarming up (3 frames)...")
        for _ in range(3):
            _ = vision._process_single_frame(frame)

        # Timed runs
        num_runs = 10
        logger.info(f"\nProcessing {num_runs} frames...")

        times = []
        for i in range(num_runs):
            start = time.time()
            result = vision._process_single_frame(frame)
            elapsed_ms = (time.time() - start) * 1000
            times.append(elapsed_ms)

            if result:
                logger.info(
                    f"  Frame {i+1}: {elapsed_ms:.1f}ms ({len(result.balls)} balls)"
                )

        vision.stop_capture()

        # Statistics
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        fps = 1000.0 / avg_time if avg_time > 0 else 0

        logger.info(f"\n{'=' * 40}")
        logger.info("Full Pipeline Statistics:")
        logger.info(f"  Average: {avg_time:.1f}ms")
        logger.info(f"  Min:     {min_time:.1f}ms")
        logger.info(f"  Max:     {max_time:.1f}ms")
        logger.info(f"  FPS:     {fps:.1f}")
        logger.info(f"{'=' * 40}")

        # Check against target
        target_fps = 15.0
        target_ms = 1000.0 / target_fps

        if avg_time <= target_ms:
            logger.info(f"✓ Pipeline meets {target_fps} FPS target")
        else:
            overhead = avg_time - target_ms
            logger.warning(f"✗ Pipeline TOO SLOW for {target_fps} FPS")
            logger.warning(f"  Need to reduce by: {overhead:.1f}ms")

        return {
            "avg_ms": avg_time,
            "min_ms": min_time,
            "max_ms": max_time,
            "fps": fps,
            "meets_target": fps >= target_fps,
            "overhead_ms": max(0, avg_time - target_ms),
        }

    except Exception as e:
        logger.error(f"✗ Pipeline test failed: {e}", exc_info=True)
        return None


def main():
    """Main entry point."""
    logger.info("\n")
    logger.info("#" * 60)
    logger.info("BILLIARDS VISION PERFORMANCE CHECK")
    logger.info("#" * 60)

    # 1. Check device configuration
    device_results = check_device_config()

    # 2. Test YOLO inference
    yolo_results = test_yolo_inference()

    # 3. Test full pipeline
    pipeline_results = test_full_pipeline()

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)

    if device_results.get("mps_available"):
        logger.info("✓ Apple Silicon MPS available")
    else:
        logger.warning("⚠ Apple Silicon MPS NOT available - using CPU")

    if device_results.get("model_format") == "CoreML":
        logger.info("✓ Using CoreML model (optimized for Apple Silicon)")
    else:
        logger.warning(f"⚠ Using {device_results.get('model_format', 'unknown')} model")

    if yolo_results:
        if yolo_results["meets_target"]:
            logger.info(
                f"✓ YOLO inference fast enough ({yolo_results['avg_ms']:.1f}ms)"
            )
        else:
            logger.warning(
                f"✗ YOLO inference TOO SLOW ({yolo_results['avg_ms']:.1f}ms, need {yolo_results['overhead_ms']:.1f}ms faster)"
            )

    if pipeline_results:
        if pipeline_results["meets_target"]:
            logger.info(
                f"✓ Full pipeline fast enough ({pipeline_results['avg_ms']:.1f}ms, {pipeline_results['fps']:.1f} FPS)"
            )
        else:
            logger.warning(
                f"✗ Full pipeline TOO SLOW ({pipeline_results['avg_ms']:.1f}ms, {pipeline_results['fps']:.1f} FPS)"
            )

    # Recommendations
    logger.info("\n" + "=" * 60)
    logger.info("RECOMMENDATIONS")
    logger.info("=" * 60)

    if not device_results.get("mps_available"):
        logger.info("1. Install PyTorch with MPS support for Apple Silicon:")
        logger.info("   pip install torch torchvision torchaudio")

    if device_results.get("model_format") != "CoreML":
        logger.info("2. Convert model to CoreML for better performance:")
        logger.info("   Use Ultralytics: model.export(format='coreml')")

    if yolo_results and not yolo_results["meets_target"]:
        logger.info("3. YOLO inference is the bottleneck:")
        logger.info("   - Verify CoreML model is being used")
        logger.info("   - Check that Apple Neural Engine is active")
        logger.info("   - Consider using smaller model (yolov8n instead of yolov8s/m)")

    if pipeline_results and yolo_results:
        non_yolo_time = pipeline_results["avg_ms"] - yolo_results["avg_ms"]
        if non_yolo_time > 20:
            logger.info(f"4. Non-YOLO processing is slow ({non_yolo_time:.1f}ms):")
            logger.info("   - Disable preprocessing if not needed")
            logger.info("   - Reduce tracking complexity")
            logger.info("   - Consider disabling table detection")

    logger.info("\n")


if __name__ == "__main__":
    main()
