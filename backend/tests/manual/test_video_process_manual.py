#!/usr/bin/env python3
"""Manual test for Video Module Process.

This script tests the Video Module process by:
1. Starting the process with a video file
2. Monitoring shared memory for frames
3. Testing graceful shutdown

Usage:
    python backend/tests/manual/test_video_process_manual.py

Requirements:
    - Video file at assets/demo3.mp4
    - Video Module not already running

Expected Output:
    - Process starts successfully
    - Shared memory initialized
    - Frames read from shared memory
    - Clean shutdown on SIGTERM
"""

import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from backend.video.ipc.shared_memory import SharedMemoryFrameReader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    """Test the Video Module process."""
    logger.info("=" * 80)
    logger.info("Testing Video Module Process")
    logger.info("=" * 80)

    # Check video file exists
    video_file = project_root / "assets" / "demo3.mp4"
    if not video_file.exists():
        logger.error(f"Video file not found: {video_file}")
        logger.error("Please ensure assets/demo3.mp4 exists")
        return False

    # Start Video Module process
    logger.info("Starting Video Module process...")
    env = os.environ.copy()
    env["VIDEO_FILE"] = str(video_file)
    env["LOG_LEVEL"] = "INFO"

    process = subprocess.Popen(
        [sys.executable, "-m", "backend.video"],
        env=env,
        cwd=str(project_root),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    logger.info(f"Video Module process started (PID: {process.pid})")

    # Give process time to initialize
    logger.info("Waiting 3s for process to initialize...")
    time.sleep(3)

    try:
        # Check if process is still running
        if process.poll() is not None:
            logger.error("Process exited prematurely!")
            output = process.stdout.read()
            logger.error(f"Process output:\n{output}")
            return False

        # Create reader to test shared memory
        logger.info("Creating SharedMemoryFrameReader...")
        reader = SharedMemoryFrameReader(name="billiards_video")

        # Attach to shared memory
        logger.info("Attaching to shared memory...")
        try:
            reader.attach(timeout=10.0)
            logger.info(
                f"✓ Attached to shared memory: {reader.width}x{reader.height} "
                f"{reader.frame_format.name}"
            )
        except TimeoutError as e:
            logger.error(f"✗ Failed to attach to shared memory: {e}")
            return False

        # Read some frames
        logger.info("Reading frames from shared memory...")
        frames_read = 0
        start_time = time.time()

        for i in range(100):  # Try to read 100 frames
            frame, metadata = reader.read_frame()

            if frame is not None:
                frames_read += 1
                if frames_read in [1, 10, 25, 50, 75, 100]:
                    elapsed = time.time() - start_time
                    fps = frames_read / elapsed if elapsed > 0 else 0
                    logger.info(
                        f"  Frame {frames_read}: "
                        f"number={metadata.frame_number}, "
                        f"size={metadata.width}x{metadata.height}, "
                        f"fps={fps:.1f}"
                    )

            time.sleep(0.01)  # 10ms between reads

        elapsed = time.time() - start_time
        fps = frames_read / elapsed if elapsed > 0 else 0

        logger.info("✓ Frame reading test complete:")
        logger.info(f"  Frames read: {frames_read}")
        logger.info(f"  Elapsed time: {elapsed:.2f}s")
        logger.info(f"  Average FPS: {fps:.1f}")

        # Detach reader
        logger.info("Detaching from shared memory...")
        reader.detach()

        # Test graceful shutdown
        logger.info("Testing graceful shutdown (sending SIGTERM)...")
        process.send_signal(signal.SIGTERM)

        # Wait for process to exit
        logger.info("Waiting for process to exit...")
        try:
            exit_code = process.wait(timeout=10.0)
            logger.info(f"✓ Process exited with code: {exit_code}")

            if exit_code != 0:
                logger.warning(f"✗ Non-zero exit code: {exit_code}")
                return False

        except subprocess.TimeoutExpired:
            logger.error("✗ Process did not exit within timeout")
            process.kill()
            process.wait()
            return False

        logger.info("=" * 80)
        logger.info("✓ All tests passed!")
        logger.info("=" * 80)
        return True

    except Exception as e:
        logger.error(f"✗ Test failed with exception: {e}", exc_info=True)
        return False

    finally:
        # Ensure process is terminated
        if process.poll() is None:
            logger.info("Terminating process...")
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.warning("Killing process...")
                process.kill()
                process.wait()


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
