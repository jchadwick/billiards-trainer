"""GStreamer Shared Memory Consumer for Vision Processing.

Consumes frames from GStreamer shmsrc for zero-copy, low-latency access
to preprocessed camera frames (with fisheye correction and adjustments).
"""

import logging
import threading
import time
from typing import Optional

import cv2
import numpy as np
import gi

gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

logger = logging.getLogger(__name__)


class GStreamerFrameConsumer:
    """Consumes frames from GStreamer shared memory for vision processing."""

    def __init__(self, socket_path: str = "/tmp/camera-backend"):
        """Initialize GStreamer consumer.

        Args:
            socket_path: Path to GStreamer shmsink socket
        """
        Gst.init(None)

        self.socket_path = socket_path
        self.pipeline = None
        self.running = False
        self.width = 1920
        self.height = 1080

        # Thread-safe frame buffer
        self._lock = threading.Lock()
        self._current_frame = None
        self._frame_count = 0
        self._last_frame_time = 0.0

        # GStreamer pipeline components
        self.appsink = None
        self.loop = None
        self.thread = None

    def start(self) -> bool:
        """Start consuming frames from GStreamer.

        Returns:
            True if started successfully
        """
        if self.running:
            return True

        try:
            # Build pipeline: shmsrc -> appsink
            pipeline_str = (
                f"shmsrc socket-path={self.socket_path} is-live=true ! "
                f"video/x-raw,format=BGR,width={self.width},height={self.height},framerate=30/1 ! "
                f"appsink name=sink emit-signals=true sync=false max-buffers=1 drop=true"
            )

            logger.info(f"Creating GStreamer pipeline: {pipeline_str}")
            self.pipeline = Gst.parse_launch(pipeline_str)

            # Get appsink element
            self.appsink = self.pipeline.get_by_name("sink")
            if not self.appsink:
                logger.error("Failed to get appsink element")
                return False

            # Connect to new-sample signal
            self.appsink.connect("new-sample", self._on_new_sample)

            # Start pipeline
            ret = self.pipeline.set_state(Gst.State.PLAYING)
            if ret == Gst.StateChangeReturn.FAILURE:
                logger.error("Failed to start GStreamer pipeline")
                return False

            # Start GLib main loop in separate thread
            self.running = True
            self.loop = GLib.MainLoop()
            self.thread = threading.Thread(target=self._run_loop, daemon=True)
            self.thread.start()

            logger.info("GStreamer consumer started successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to start GStreamer consumer: {e}", exc_info=True)
            return False

    def stop(self):
        """Stop consuming frames."""
        if not self.running:
            return

        self.running = False

        # Stop pipeline
        if self.pipeline:
            self.pipeline.set_state(Gst.State.NULL)

        # Stop main loop
        if self.loop:
            self.loop.quit()

        # Wait for thread
        if self.thread:
            self.thread.join(timeout=2.0)

        logger.info("GStreamer consumer stopped")

    def _run_loop(self):
        """Run GLib main loop."""
        try:
            self.loop.run()
        except Exception as e:
            logger.error(f"GLib main loop error: {e}", exc_info=True)

    def _on_new_sample(self, appsink):
        """Callback for new frame from GStreamer.

        Args:
            appsink: GStreamer appsink element
        """
        try:
            # Pull sample from appsink
            sample = appsink.emit("pull-sample")
            if not sample:
                return Gst.FlowReturn.ERROR

            # Get buffer from sample
            buf = sample.get_buffer()
            if not buf:
                return Gst.FlowReturn.ERROR

            # Map buffer to memory
            success, map_info = buf.map(Gst.MapFlags.READ)
            if not success:
                return Gst.FlowReturn.ERROR

            try:
                # Convert to numpy array
                frame = np.ndarray(
                    shape=(self.height, self.width, 3),
                    dtype=np.uint8,
                    buffer=map_info.data
                )

                # Update shared buffer (thread-safe)
                with self._lock:
                    self._current_frame = frame.copy()
                    self._frame_count += 1
                    self._last_frame_time = time.time()

            finally:
                buf.unmap(map_info)

            return Gst.FlowReturn.OK

        except Exception as e:
            logger.error(f"Error processing frame: {e}", exc_info=True)
            return Gst.FlowReturn.ERROR

    def get_frame(self) -> Optional[np.ndarray]:
        """Get latest frame for vision processing.

        Returns:
            Latest frame as numpy array, or None if not available
        """
        with self._lock:
            if self._current_frame is not None:
                return self._current_frame.copy()
        return None

    def is_connected(self) -> bool:
        """Check if consumer is receiving frames.

        Returns:
            True if frames are being received
        """
        with self._lock:
            # Check if we've received a frame in the last 2 seconds
            if self._last_frame_time > 0:
                return (time.time() - self._last_frame_time) < 2.0
        return False

    def get_statistics(self) -> dict:
        """Get consumer statistics.

        Returns:
            Dictionary with statistics
        """
        with self._lock:
            fps = 0.0
            if self._last_frame_time > 0:
                elapsed = time.time() - self._last_frame_time
                if elapsed > 0 and self._frame_count > 0:
                    fps = self._frame_count / elapsed

            return {
                "running": self.running,
                "connected": self.is_connected(),
                "frame_count": self._frame_count,
                "fps": fps,
                "last_frame_time": self._last_frame_time,
                "socket_path": self.socket_path
            }


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    consumer = GStreamerFrameConsumer("/tmp/camera-backend")

    if consumer.start():
        print("Consumer started, waiting for frames...")

        try:
            while True:
                frame = consumer.get_frame()
                if frame is not None:
                    print(f"Received frame: {frame.shape}")

                    # Display frame
                    cv2.imshow("Frame", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                time.sleep(0.033)  # ~30fps

        except KeyboardInterrupt:
            print("\nStopping...")

        finally:
            consumer.stop()
            cv2.destroyAllWindows()

    else:
        print("Failed to start consumer")
