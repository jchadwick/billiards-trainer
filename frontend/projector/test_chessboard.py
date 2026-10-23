#!/usr/bin/env python3
"""Send show/hide chessboard calibration commands to the projector."""

import json
import socket
import sys
import time
import unittest

# UDP configuration
UDP_IP = "127.0.0.1"
UDP_PORT = 9999


def send_show_chessboard(rows=9, cols=6, square_size=80):
    """Send command to show calibration chessboard."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    msg = create_show_chessboard_message(rows, cols, square_size)

    sock.sendto(json.dumps(msg).encode(), (UDP_IP, UDP_PORT))
    print(f"Sent show_calibration_chessboard to {UDP_IP}:{UDP_PORT}")
    print(json.dumps(msg, indent=2))
    sock.close()


def create_show_chessboard_message(rows=9, cols=6, square_size=80):
    """Return a show command with explicitly projector-pixel dimensions."""
    return {
        "type": "show_calibration_chessboard",
        "timestamp": time.time(),
        "data": {
            "rows": rows,
            "cols": cols,
            "squareSize": square_size,
            "coordinate_space": "projector_pixels",
            "coordinate_scale": [1.0, 1.0],
            "centered": True,
        },
    }


def send_hide_chessboard():
    """Send command to hide calibration chessboard."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    msg = {"type": "hide_calibration_chessboard", "timestamp": time.time(), "data": {}}

    sock.sendto(json.dumps(msg).encode(), (UDP_IP, UDP_PORT))
    print(f"Sent hide_calibration_chessboard to {UDP_IP}:{UDP_PORT}")
    print(json.dumps(msg, indent=2))
    sock.close()


class TestChessboardSenderCoordinateContract(unittest.TestCase):
    def test_show_command_declares_projector_pixel_extent_space(self):
        data = create_show_chessboard_message()["data"]
        self.assertEqual(data["coordinate_space"], "projector_pixels")
        self.assertEqual(data["coordinate_scale"], [1.0, 1.0])
        self.assertGreater(data["squareSize"], 0)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "hide":
        send_hide_chessboard()
    else:
        # Parse optional parameters
        rows = int(sys.argv[1]) if len(sys.argv) > 1 else 9
        cols = int(sys.argv[2]) if len(sys.argv) > 2 else 6
        square_size = int(sys.argv[3]) if len(sys.argv) > 3 else 80

        send_show_chessboard(rows, cols, square_size)
