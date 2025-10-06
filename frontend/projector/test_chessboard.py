#!/usr/bin/env python3
"""
Test script for chessboard calibration display
Sends show/hide commands via UDP to the projector
"""

import socket
import json
import time
import sys

# UDP configuration
UDP_IP = "127.0.0.1"
UDP_PORT = 9999

def send_show_chessboard(rows=9, cols=6, square_size=80):
    """Send command to show calibration chessboard"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    msg = {
        "type": "show_calibration_chessboard",
        "timestamp": time.time(),
        "data": {
            "rows": rows,
            "cols": cols,
            "squareSize": square_size,
            "centered": True
        }
    }

    sock.sendto(json.dumps(msg).encode(), (UDP_IP, UDP_PORT))
    print(f"Sent show_calibration_chessboard to {UDP_IP}:{UDP_PORT}")
    print(json.dumps(msg, indent=2))
    sock.close()

def send_hide_chessboard():
    """Send command to hide calibration chessboard"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    msg = {
        "type": "hide_calibration_chessboard",
        "timestamp": time.time(),
        "data": {}
    }

    sock.sendto(json.dumps(msg).encode(), (UDP_IP, UDP_PORT))
    print(f"Sent hide_calibration_chessboard to {UDP_IP}:{UDP_PORT}")
    print(json.dumps(msg, indent=2))
    sock.close()

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "hide":
        send_hide_chessboard()
    else:
        # Parse optional parameters
        rows = int(sys.argv[1]) if len(sys.argv) > 1 else 9
        cols = int(sys.argv[2]) if len(sys.argv) > 2 else 6
        square_size = int(sys.argv[3]) if len(sys.argv) > 3 else 80

        send_show_chessboard(rows, cols, square_size)
