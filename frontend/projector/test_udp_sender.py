#!/usr/bin/env python3
"""
UDP Test Sender for Billiards Projector
Sends simulated trajectory data to test the projector visualization
"""

import socket
import json
import time
import math
import sys

# UDP configuration
UDP_IP = "127.0.0.1"
UDP_PORT = 9999

def create_trajectory_message(t):
    """Create a simulated trajectory message"""
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
            "paths": [
                {
                    "points": points,
                    "ballType": "cue",
                    "confidence": 0.95
                }
            ],
            "collisions": [
                {"x": 0.5, "y": 0.5, "type": "cushion"}
            ],
            "ghostBalls": [
                {"x": 0.7, "y": 0.4, "radius": 15, "number": 8}
            ]
        }
    }

def create_collision_message():
    """Create a collision marker message"""
    return {
        "type": "collision",
        "timestamp": time.time(),
        "data": {
            "x": 0.6,
            "y": 0.4,
            "type": "ball"
        }
    }

def create_aim_line_message(t):
    """Create an aim line message"""
    angle = t * 0.5
    return {
        "type": "aim",
        "timestamp": time.time(),
        "data": {
            "x1": 0.3,
            "y1": 0.3,
            "x2": 0.3 + 0.4 * math.cos(angle),
            "y2": 0.3 + 0.4 * math.sin(angle)
        }
    }

def send_test_messages(duration=30, interval=0.5):
    """Send test messages for a specified duration"""
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
    """Send a single test message"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    msg = create_trajectory_message(0)
    sock.sendto(json.dumps(msg).encode(), (UDP_IP, UDP_PORT))

    print(f"Sent single test message to {UDP_IP}:{UDP_PORT}")
    print(json.dumps(msg, indent=2))

    sock.close()

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "continuous"

    if mode == "single":
        send_single_test()
    else:
        duration = int(sys.argv[2]) if len(sys.argv) > 2 else 30
        interval = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
        send_test_messages(duration, interval)
