"""Interpret the one configured detector source value.

An integer is a directly opened local device. A stable GStreamer selector is
also a direct camera identity. An HTTP(S) URI is a network MJPEG source, such
as the local uStreamer stream. Credentials are deliberately rejected because
the application and camera stream use anonymous access.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal
from urllib.parse import urlsplit

_GSTREAMER_SELECTOR = re.compile(r"^gstreamer-device-v1:[0-9a-f]{64}$")
_MAX_SOURCE_URI_LENGTH = 2048


@dataclass(frozen=True)
class CameraSource:
    """Validated source identity inferred from ``vision.camera.device_id``."""

    kind: Literal["device", "ustreamer_mjpeg"]
    value: int | str


def classify_camera_source(value: object) -> CameraSource:
    """Classify a configured device index, stable selector, or MJPEG URI."""
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return CameraSource("device", value)
    if not isinstance(value, str):
        raise ValueError(
            "camera source must be a nonnegative device index or HTTP(S) URI"
        )
    if _GSTREAMER_SELECTOR.fullmatch(value):
        return CameraSource("device", value)
    if len(value) > _MAX_SOURCE_URI_LENGTH or value != value.strip():
        raise ValueError(
            "camera source URI is too long or contains surrounding whitespace"
        )
    try:
        parsed = urlsplit(value)
        # Accessing .port also validates that a supplied port is numeric and in range.
        _ = parsed.port
    except ValueError as error:
        raise ValueError("camera source URI is malformed") from error
    if parsed.scheme.lower() not in {"http", "https"}:
        raise ValueError("camera source URI must use HTTP or HTTPS")
    if (
        not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise ValueError(
            "camera source URI needs a host and must not contain credentials"
        )
    if parsed.fragment:
        raise ValueError("camera source URI must not contain a fragment")
    return CameraSource("ustreamer_mjpeg", value)


def is_http_camera_source(value: object) -> bool:
    """Return whether a string is an HTTP(S) source candidate.

    Invalid HTTP-looking values raise instead of falling through as video-file
    paths, which would obscure a malformed live-source configuration.
    """
    if not isinstance(value, str):
        return False
    if not value.lower().startswith(("http://", "https://")):
        return False
    return classify_camera_source(value).kind == "ustreamer_mjpeg"


def resolve_configured_camera_source(
    device_id: object, video_file_path: object = None
) -> int | str:
    """Resolve the selected input while preserving legacy recorded-video config.

    An explicit network URI in ``vision.camera.device_id`` wins over the older
    ``video_file_path`` override so a uStreamer source can be selected by
    changing only the detector source value. For a local numeric camera, a
    configured video path remains the existing replay fallback; clear that
    legacy path to use a physical device.
    """
    source = classify_camera_source(device_id)
    if source.kind == "ustreamer_mjpeg":
        return source.value
    if isinstance(video_file_path, str) and video_file_path:
        return video_file_path
    return source.value
