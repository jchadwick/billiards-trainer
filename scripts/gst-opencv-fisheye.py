#!/usr/bin/env python3

"""GStreamer OpenCV Fisheye Correction Plugin.

This creates a GStreamer element that applies OpenCV fisheye correction
to video streams using calibration data.

Usage in pipeline:
    gst-launch-1.0 v4l2src ! opencv-fisheye config=/path/to/calibration.yaml ! ...
"""

import sys
import cv2
import numpy as np
import yaml
import gi

gi.require_version('Gst', '1.0')
gi.require_version('GstBase', '1.0')
gi.require_version('GstVideo', '1.0')

from gi.repository import GLib, Gst, GstBase, GstVideo

Gst.init(None)


class GstOpenCVFisheye(GstBase.BaseTransform):
    """GStreamer element for OpenCV fisheye correction."""

    __gstmetadata__ = (
        'OpenCV Fisheye Corrector',
        'Filter/Video',
        'Applies fisheye lens correction using OpenCV',
        'Billiards Trainer'
    )

    __gsttemplates__ = (
        Gst.PadTemplate.new(
            "src",
            Gst.PadDirection.SRC,
            Gst.PadPresence.ALWAYS,
            Gst.Caps.from_string("video/x-raw,format=RGB")
        ),
        Gst.PadTemplate.new(
            "sink",
            Gst.PadDirection.SINK,
            Gst.PadPresence.ALWAYS,
            Gst.Caps.from_string("video/x-raw,format=RGB")
        )
    )

    __gproperties__ = {
        "config-file": (
            str,
            "Calibration file",
            "Path to OpenCV calibration YAML file",
            "",
            GLib.ParamFlags.READWRITE
        ),
        "brightness": (
            float,
            "Brightness",
            "Brightness adjustment (-100 to 100)",
            -100.0, 100.0, 0.0,
            GLib.ParamFlags.READWRITE
        ),
        "contrast": (
            float,
            "Contrast",
            "Contrast adjustment (0.5 to 3.0)",
            0.5, 3.0, 1.0,
            GLib.ParamFlags.READWRITE
        ),
        "enable-clahe": (
            bool,
            "Enable CLAHE",
            "Enable Contrast Limited Adaptive Histogram Equalization",
            True,
            GLib.ParamFlags.READWRITE
        )
    }

    def __init__(self):
        super().__init__()

        self.config_file = ""
        self.brightness = 0.0
        self.contrast = 1.0
        self.enable_clahe = True

        self.camera_matrix = None
        self.dist_coeffs = None
        self.new_camera_matrix = None
        self.map1 = None
        self.map2 = None
        self.clahe = None

        self.width = 0
        self.height = 0

    def do_get_property(self, prop):
        if prop.name == 'config-file':
            return self.config_file
        elif prop.name == 'brightness':
            return self.brightness
        elif prop.name == 'contrast':
            return self.contrast
        elif prop.name == 'enable-clahe':
            return self.enable_clahe
        else:
            raise AttributeError(f'Unknown property {prop.name}')

    def do_set_property(self, prop, value):
        if prop.name == 'config-file':
            self.config_file = value
            self._load_calibration()
        elif prop.name == 'brightness':
            self.brightness = value
        elif prop.name == 'contrast':
            self.contrast = value
        elif prop.name == 'enable-clahe':
            self.enable_clahe = value
            if value and self.clahe is None:
                self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def _load_calibration(self):
        """Load fisheye calibration from YAML file."""
        if not self.config_file:
            return

        try:
            with open(self.config_file, 'r') as f:
                calib = yaml.safe_load(f)

            self.camera_matrix = np.array(calib['camera_matrix'])
            self.dist_coeffs = np.array(calib['dist_coeffs'])

            Gst.info(f"Loaded calibration from {self.config_file}")

        except Exception as e:
            Gst.error(f"Failed to load calibration: {e}")
            self.camera_matrix = None
            self.dist_coeffs = None

    def do_set_caps(self, incaps, outcaps):
        """Called when caps are negotiated."""
        struct = incaps.get_structure(0)
        self.width = struct.get_int("width")[1]
        self.height = struct.get_int("height")[1]

        # Pre-compute undistortion maps if we have calibration
        if self.camera_matrix is not None and self.dist_coeffs is not None:
            # Estimate new camera matrix for undistortion
            self.new_camera_matrix = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                self.camera_matrix,
                self.dist_coeffs,
                (self.width, self.height),
                np.eye(3)
            )

            # Compute undistortion maps
            self.map1, self.map2 = cv2.fisheye.initUndistortRectifyMap(
                self.camera_matrix,
                self.dist_coeffs,
                np.eye(3),
                self.new_camera_matrix,
                (self.width, self.height),
                cv2.CV_16SC2
            )

            Gst.info(f"Computed undistortion maps for {self.width}x{self.height}")

        # Initialize CLAHE if enabled
        if self.enable_clahe and self.clahe is None:
            self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        return True

    def do_transform_ip(self, buffer):
        """In-place transform of buffer data."""
        try:
            # Map buffer to numpy array
            success, map_info = buffer.map(Gst.MapFlags.READ | Gst.MapFlags.WRITE)
            if not success:
                return Gst.FlowReturn.ERROR

            # Create numpy array from buffer
            image = np.ndarray(
                shape=(self.height, self.width, 3),
                dtype=np.uint8,
                buffer=map_info.data
            )

            # Apply fisheye correction if calibration is loaded
            if self.map1 is not None and self.map2 is not None:
                image = cv2.remap(
                    image, self.map1, self.map2,
                    interpolation=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT
                )

            # Apply brightness/contrast
            if self.brightness != 0 or self.contrast != 1.0:
                image = cv2.convertScaleAbs(
                    image,
                    alpha=self.contrast,
                    beta=self.brightness
                )

            # Apply CLAHE if enabled
            if self.enable_clahe and self.clahe is not None:
                # Convert RGB to LAB
                lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
                l, a, b = cv2.split(lab)

                # Apply CLAHE to L channel
                l = self.clahe.apply(l)

                # Merge and convert back
                lab = cv2.merge([l, a, b])
                image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

            # Ensure the modified data is written back
            np.copyto(
                np.ndarray(
                    shape=(self.height, self.width, 3),
                    dtype=np.uint8,
                    buffer=map_info.data
                ),
                image
            )

        finally:
            buffer.unmap(map_info)

        return Gst.FlowReturn.OK


# Register the element
GObject = gi.repository.GObject
GObject.type_register(GstOpenCVFisheye)
__gstelementfactory__ = (
    "opencv-fisheye",
    Gst.Rank.NONE,
    GstOpenCVFisheye
)


def create_calibration_file(output_path="/tmp/calibration.yaml"):
    """Create a sample calibration file for testing."""

    # Sample calibration data (you'll replace with actual calibration)
    calibration = {
        'camera_matrix': [
            [800.0, 0.0, 960.0],
            [0.0, 800.0, 540.0],
            [0.0, 0.0, 1.0]
        ],
        'dist_coeffs': [
            [-0.3, 0.1, 0.0, 0.0]  # k1, k2, k3, k4 for fisheye model
        ]
    }

    with open(output_path, 'w') as f:
        yaml.dump(calibration, f)

    print(f"Created sample calibration at {output_path}")


if __name__ == "__main__":
    # Register plugin
    import os
    plugin_path = os.path.abspath(__file__)

    print("OpenCV Fisheye GStreamer Plugin")
    print("================================")
    print()
    print("To use this plugin in GStreamer:")
    print()
    print(f"export GST_PLUGIN_PATH={os.path.dirname(plugin_path)}")
    print()
    print("Example pipeline:")
    print("gst-launch-1.0 v4l2src ! \\")
    print("  videoconvert ! \\")
    print("  opencv-fisheye config-file=/path/to/calibration.yaml \\")
    print("    brightness=10 contrast=1.2 enable-clahe=true ! \\")
    print("  videoconvert ! \\")
    print("  autovideosink")
    print()

    # Create sample calibration if requested
    if len(sys.argv) > 1 and sys.argv[1] == "--create-calibration":
        create_calibration_file()
