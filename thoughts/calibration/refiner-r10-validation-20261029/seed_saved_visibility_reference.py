#!/usr/bin/env python3
"""Create an isolated offline visibility fixture through the normal preview API.

This is only for the saved-MJPEG qualification package. It does not contact or
change the physical camera, and its approximate playing-area corners are not a
production table calibration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import socket
import struct
import time
from typing import Any


def read_exact(connection: socket.socket, count: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < count:
        chunk = connection.recv(count - len(chunks))
        if not chunk:
            raise ConnectionError("preview/control connection closed early")
        chunks.extend(chunk)
    return bytes(chunks)


def exchange(port: int, magic: bytes, body: bytes | None = None) -> tuple[dict[str, Any], bytes]:
    with socket.create_connection(("127.0.0.1", port), timeout=3) as connection:
        connection.settimeout(5)
        connection.sendall(magic)
        if body is not None:
            connection.sendall(struct.pack(">I", len(body)) + body)
        response_length = struct.unpack(">I", read_exact(connection, 4))[0]
        if response_length == 0 or response_length > 1_048_576:
            raise ValueError(f"invalid response length {response_length}")
        header = json.loads(read_exact(connection, response_length))
        remaining = int(header.get("jpeg_bytes", 0))
        jpeg = bytearray()
        while remaining:
            chunk = connection.recv(min(remaining, 64 * 1024))
            if not chunk:
                raise ConnectionError("preview JPEG ended early")
            jpeg.extend(chunk)
            remaining -= len(chunk)
        return header, bytes(jpeg)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=18322)
    parser.add_argument("--attempts", type=int, default=120)
    args = parser.parse_args()

    last_error: str | None = None
    for _ in range(args.attempts):
        try:
            preview, jpeg = exchange(args.port, b"BTPV0001")
            if preview.get("status") != "ok" or not preview.get("descriptor"):
                raise RuntimeError(f"preview not ready: {preview}")
            break
        except Exception as error:  # startup and source readiness are asynchronous
            last_error = str(error)
            time.sleep(0.5)
    else:
        raise RuntimeError(f"saved-frame preview did not become ready: {last_error}")

    descriptor = preview["descriptor"]
    width = int(descriptor["width_px"])
    height = int(descriptor["height_px"])
    if (width, height) != (3840, 2160):
        raise ValueError(f"expected saved 3840x2160 frame, got {width}x{height}")

    snapshot_response, _ = exchange(args.port, b"BTCR0001")
    if snapshot_response.get("status") != "ok":
        raise RuntimeError(f"calibration store read failed: {snapshot_response}")
    snapshot = snapshot_response.get("snapshot") or {}

    # These points follow the cloth boundary in the frozen 1280x720 table image
    # scaled by 3x. They establish only a test visibility region; the resulting
    # homography and visibility reference must not be promoted to production.
    corners = [
        {"x_px": 246.0, "y_px": 165.0},
        {"x_px": 3645.0, "y_px": 165.0},
        {"x_px": 3645.0, "y_px": 1998.0},
        {"x_px": 246.0, "y_px": 1998.0},
    ]
    request = {
        "schema_version": 1,
        "descriptor": descriptor,
        "table_width_mm": 2336.8,
        "table_height_mm": 1155.7,
        "corners_px": corners,
        "expected_store_revision": snapshot.get("revision"),
    }
    body = json.dumps(request, separators=(",", ":")).encode()
    applied, _ = exchange(args.port, b"BTCA0001", body)
    if applied.get("status") != "ok":
        raise RuntimeError(f"saved-frame calibration API rejected fixture: {applied}")
    saved = applied.get("snapshot") or {}
    binding = saved.get("binding_context") or {}
    if binding.get("camera_profile_id") != descriptor.get("camera_profile_id"):
        raise RuntimeError("saved visibility fixture is not bound to the exact preview profile")
    if saved.get("visibility_reference") is None:
        raise RuntimeError("calibration API did not persist the visibility reference")

    print(
        json.dumps(
            {
                "scope": "offline saved-MJPEG qualification fixture only",
                "production_eligible": False,
                "frame_id": descriptor["source_frame_id"],
                "camera_profile_id": descriptor["camera_profile_id"],
                "dimensions": [width, height],
                "preview_jpeg_sha256": hashlib.sha256(jpeg).hexdigest(),
                "corners_px": corners,
                "store_revision": saved.get("revision"),
                "visibility_reference_present": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
