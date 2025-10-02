"""Minimal V4L2 camera module using subprocess for ffmpeg.

This module provides the simplest possible camera streaming by leveraging
the proven working ffmpeg/v4l2 pipeline that works on the target system.

Key design principles:
1. Use what works: subprocess + ffmpeg (proven to work)
2. No OpenCV initialization (avoids all the hanging issues)
3. Proxy the ffmpeg stream directly to clients
4. Minimal state management

Architecture:
    ffmpeg subprocess -> stdout -> client
    (v4l2:///dev/video0)
"""

import logging
import subprocess
import threading
import time
from typing import Optional

logger = logging.getLogger(__name__)


class V4L2CameraModule:
    """Minimal camera module that uses ffmpeg subprocess for streaming.

    This bypasses all OpenCV complexity and uses the proven working
    ffmpeg command directly.
    """

    def __init__(self, config: dict):
        """Initialize camera module with config.

        Args:
            config: Camera configuration dict with keys:
                - camera_device_id: Device ID (0 = /dev/video0)
                - camera_resolution: Tuple (width, height)
                - camera_fps: Target FPS
        """
        self.device_id = config.get("camera_device_id", 0)
        self.resolution = config.get("camera_resolution", (1920, 1080))
        self.target_fps = config.get("camera_fps", 30)

        # Device path
        self.device_path = f"/dev/video{self.device_id}"

        # Process state
        self._process: Optional[subprocess.Popen] = None
        self._running = False
        self._monitor_thread: Optional[threading.Thread] = None

        logger.info(f"V4L2CameraModule initialized for {self.device_path}")

    def start_capture(self, timeout: float = 10.0) -> bool:
        """Start camera capture using ffmpeg subprocess.

        Args:
            timeout: Maximum time to wait for camera to start

        Returns:
            True if started successfully, False otherwise
        """
        if self._running:
            logger.warning("Camera already running")
            return True

        try:
            logger.info(f"Starting ffmpeg capture from {self.device_path}")

            # Build ffmpeg command similar to working cvlc command
            # cvlc v4l2:///dev/video0:chroma=h264 --sout '#transcode{vcodec=h264,acodec=none}:http{mux=ts,dst=:8080/}'
            # We'll use ffmpeg to read v4l2 and output mjpeg to stdout

            width, height = self.resolution
            cmd = [
                "ffmpeg",
                "-f", "v4l2",
                "-input_format", "mjpeg",  # Try MJPEG first (fast)
                "-video_size", f"{width}x{height}",
                "-framerate", str(self.target_fps),
                "-i", self.device_path,
                "-f", "mjpeg",  # Output MJPEG
                "-q:v", "5",  # Quality (2-31, lower is better)
                "-",  # Output to stdout
            ]

            logger.info(f"Starting ffmpeg: {' '.join(cmd)}")

            # Start ffmpeg process
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,  # Unbuffered
            )

            self._running = True

            # Start monitor thread to watch stderr
            self._monitor_thread = threading.Thread(
                target=self._monitor_stderr,
                daemon=True
            )
            self._monitor_thread.start()

            # Wait a bit to see if process starts successfully
            time.sleep(0.5)

            if self._process.poll() is not None:
                # Process already exited
                stderr = self._process.stderr.read().decode('utf-8', errors='ignore')
                logger.error(f"ffmpeg exited immediately: {stderr}")
                self._running = False
                return False

            logger.info("ffmpeg started successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to start ffmpeg: {e}", exc_info=True)
            self._running = False
            if self._process:
                self._process.kill()
                self._process = None
            return False

    def _monitor_stderr(self):
        """Monitor ffmpeg stderr output for errors."""
        if not self._process or not self._process.stderr:
            return

        try:
            for line in self._process.stderr:
                if not self._running:
                    break
                try:
                    msg = line.decode('utf-8', errors='ignore').strip()
                    if msg:
                        # Log ffmpeg output at debug level
                        logger.debug(f"ffmpeg: {msg}")
                except Exception:
                    pass
        except Exception as e:
            logger.error(f"Error monitoring ffmpeg stderr: {e}")

    def stop_capture(self):
        """Stop camera capture and cleanup."""
        logger.info("Stopping camera capture")
        self._running = False

        if self._process:
            try:
                self._process.terminate()
                self._process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                logger.warning("ffmpeg didn't terminate, killing")
                self._process.kill()
            except Exception as e:
                logger.error(f"Error stopping ffmpeg: {e}")
            finally:
                self._process = None

        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)
            self._monitor_thread = None

        logger.info("Camera capture stopped")

    def generate_stream(self):
        """Generate MJPEG stream by reading from ffmpeg stdout.

        Yields:
            MJPEG frame data in multipart format
        """
        if not self._running or not self._process or not self._process.stdout:
            logger.error("Cannot generate stream - camera not running")
            return

        logger.info("Starting stream generation")

        try:
            # Read MJPEG frames from ffmpeg stdout
            # MJPEG format: each frame starts with FF D8 and ends with FF D9
            frame_buffer = bytearray()

            while self._running:
                # Read data from ffmpeg
                chunk = self._process.stdout.read(4096)
                if not chunk:
                    logger.warning("No data from ffmpeg")
                    break

                frame_buffer.extend(chunk)

                # Look for JPEG markers
                # FF D8 = Start of Image (SOI)
                # FF D9 = End of Image (EOI)

                while True:
                    # Find start marker
                    soi = frame_buffer.find(b'\xff\xd8')
                    if soi == -1:
                        # No start marker, keep accumulating
                        break

                    # Find end marker after start
                    eoi = frame_buffer.find(b'\xff\xd9', soi + 2)
                    if eoi == -1:
                        # No end marker yet, keep accumulating
                        break

                    # Extract complete JPEG frame
                    jpeg_data = bytes(frame_buffer[soi:eoi + 2])

                    # Remove extracted frame from buffer
                    frame_buffer = frame_buffer[eoi + 2:]

                    # Yield as MJPEG multipart frame
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n"
                        + f"Content-Length: {len(jpeg_data)}\r\n\r\n".encode()
                        + jpeg_data
                        + b"\r\n"
                    )

        except Exception as e:
            logger.error(f"Error generating stream: {e}", exc_info=True)
        finally:
            logger.info("Stream generation ended")

    def is_running(self) -> bool:
        """Check if camera is running."""
        return self._running and self._process is not None and self._process.poll() is None

    def get_statistics(self) -> dict:
        """Get camera statistics."""
        return {
            "is_running": self.is_running(),
            "device_path": self.device_path,
            "resolution": self.resolution,
            "target_fps": self.target_fps,
        }
