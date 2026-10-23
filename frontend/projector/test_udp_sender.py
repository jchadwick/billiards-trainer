#!/usr/bin/env python3
"""Send simulated trajectory data to test the projector visualization."""

import json
import math
import socket
import sys
import time
import unittest

# UDP configuration
UDP_IP = "127.0.0.1"
UDP_PORT = 9999
TABLE_COORDINATES = {
    "coordinate_space": "normalized_table",
    "coordinate_scale": [1.0, 1.0],
}
PROJECTOR_PIXEL_RADIUS = {
    "radius_space": "projector_pixels",
    "radius_scale": [1.0, 1.0],
}


def create_trajectory_message(t):
    """Create a simulated trajectory message."""
    # Simulate a ball moving in a curved path
    points = []
    for i in range(20):
        progress = i / 20.0
        x = 0.2 + progress * 0.6
        y = 0.3 + 0.2 * math.sin(progress * math.pi * 2)
        points.append({"x": x, "y": y})

    return {
        "type": "trajectory",
        "timestamp": time.time(),
        "data": {
            **TABLE_COORDINATES,
            # Explicit table boundary and extent exercise the receiver contract.
            "table_boundary": [
                {"x": 0.0, "y": 0.0},
                {"x": 1.0, "y": 0.0},
                {"x": 1.0, "y": 1.0},
                {"x": 0.0, "y": 1.0},
            ],
            "bounds": {"x": 0.0, "y": 0.0, "width": 1.0, "height": 1.0},
            "paths": [{"points": points, "ballType": "cue", "confidence": 0.95}],
            "collisions": [{"x": 0.5, "y": 0.5, "type": "cushion"}],
            "ghostBalls": [
                {
                    "x": 0.7,
                    "y": 0.4,
                    "radius": 15,
                    **PROJECTOR_PIXEL_RADIUS,
                    "number": 8,
                }
            ],
        },
    }


def create_collision_message():
    """Create a collision marker message."""
    return {
        "type": "collision",
        "timestamp": time.time(),
        "data": {**TABLE_COORDINATES, "x": 0.6, "y": 0.4, "type": "ball"},
    }


def create_aim_line_message(t):
    """Create an aim line message."""
    angle = t * 0.5
    return {
        "type": "aim",
        "timestamp": time.time(),
        "data": {
            **TABLE_COORDINATES,
            "x1": 0.5,
            "y1": 0.5,
            "x2": 0.5 + 0.4 * math.cos(angle),
            "y2": 0.5 + 0.4 * math.sin(angle),
        },
    }


def send_test_messages(duration=30, interval=0.5):
    """Send test messages for a specified duration."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    print(f"Sending UDP test messages to {UDP_IP}:{UDP_PORT}")
    print(f"Duration: {duration}s, Interval: {interval}s")
    print("Press Ctrl+C to stop\n")

    start_time = time.time()
    message_count = 0

    try:
        while time.time() - start_time < duration:
            t = time.time() - start_time

            # Send trajectory message
            msg = create_trajectory_message(t)
            sock.sendto(json.dumps(msg).encode(), (UDP_IP, UDP_PORT))
            message_count += 1
            print(f"Sent trajectory message #{message_count}")

            time.sleep(interval / 2)

            # Send aim line
            msg = create_aim_line_message(t)
            sock.sendto(json.dumps(msg).encode(), (UDP_IP, UDP_PORT))
            message_count += 1
            print(f"Sent aim line message #{message_count}")

            # Every 3 seconds, send a collision
            if int(t) % 3 == 0 and int(t) > 0:
                msg = create_collision_message()
                sock.sendto(json.dumps(msg).encode(), (UDP_IP, UDP_PORT))
                message_count += 1
                print(f"Sent collision message #{message_count}")

            time.sleep(interval / 2)

    except KeyboardInterrupt:
        print("\nStopped by user")

    finally:
        sock.close()
        print(f"\nTotal messages sent: {message_count}")


def send_single_test():
    """Send a single test message."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    msg = create_trajectory_message(0)
    sock.sendto(json.dumps(msg).encode(), (UDP_IP, UDP_PORT))

    print(f"Sent single test message to {UDP_IP}:{UDP_PORT}")
    print(json.dumps(msg, indent=2))

    sock.close()


class TestSenderCoordinateContract(unittest.TestCase):
    def test_each_coordinate_message_declares_normalized_space(self):
        messages = [
            create_trajectory_message(0.5),
            create_collision_message(),
            create_aim_line_message(0.5),
        ]
        for message in messages:
            self.assertEqual(message["data"]["coordinate_space"], "normalized_table")
            self.assertEqual(message["data"]["coordinate_scale"], [1.0, 1.0])

    def test_trajectory_boundaries_bounds_points_and_radii_are_explicit(self):
        data = create_trajectory_message(0.5)["data"]
        self.assertEqual(len(data["table_boundary"]), 4)
        for point in data["table_boundary"]:
            self.assertGreaterEqual(point["x"], 0)
            self.assertLessEqual(point["x"], 1)
            self.assertGreaterEqual(point["y"], 0)
            self.assertLessEqual(point["y"], 1)
        bounds = data["bounds"]
        self.assertLessEqual(bounds["x"] + bounds["width"], 1)
        self.assertLessEqual(bounds["y"] + bounds["height"], 1)
        for path in data["paths"]:
            for point in path["points"]:
                self.assertTrue(0 <= point["x"] <= 1)
                self.assertTrue(0 <= point["y"] <= 1)
        ghost = data["ghostBalls"][0]
        self.assertGreater(ghost["radius"], 0)
        self.assertEqual(ghost["radius_space"], "projector_pixels")
        self.assertEqual(ghost["radius_scale"], [1.0, 1.0])

    def test_aim_endpoints_and_collision_centers_are_normalized(self):
        collision = create_collision_message()["data"]
        for time_value in range(0, 61):
            aim = create_aim_line_message(time_value)["data"]
            for point in (
                collision,
                {"x": aim["x1"], "y": aim["y1"]},
                {"x": aim["x2"], "y": aim["y2"]},
            ):
                self.assertTrue(0 <= point["x"] <= 1)
                self.assertTrue(0 <= point["y"] <= 1)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "continuous"

    if mode == "single":
        send_single_test()
    else:
        duration = int(sys.argv[2]) if len(sys.argv) > 2 else 30
        interval = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
        send_test_messages(duration, interval)
