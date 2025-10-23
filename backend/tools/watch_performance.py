#!/usr/bin/env python3
"""Real-time performance monitor - watches vision pipeline performance.

Connects to the running backend via API and displays live performance metrics.

Usage:
    python backend/tools/watch_performance.py [--url http://localhost:8000]
"""

import argparse
import sys
import time
from datetime import datetime

import requests


def clear_screen():
    """Clear the terminal screen."""
    print("\033[2J\033[H", end="")


def format_bar(value: float, max_value: float, width: int = 30) -> str:
    """Create a text-based progress bar.

    Args:
        value: Current value
        max_value: Maximum value for the bar
        width: Width of the bar in characters

    Returns:
        String representation of the bar
    """
    if max_value == 0:
        filled = 0
    else:
        filled = int((value / max_value) * width)

    bar = "█" * filled + "░" * (width - filled)
    return bar


def display_performance(stats: dict):
    """Display performance statistics in a nice format.

    Args:
        stats: Performance statistics dictionary
    """
    clear_screen()

    print("=" * 70)
    print("BILLIARDS VISION - REAL-TIME PERFORMANCE MONITOR")
    print("=" * 70)
    print(f"Updated: {datetime.now().strftime('%H:%M:%S')}")
    print()

    # Overall FPS
    fps = stats.get("fps", 0)
    target_fps = stats.get("target_fps", 15.0)
    meeting_target = stats.get("meeting_target", False)

    status = "✓ ON TARGET" if meeting_target else "✗ SLOW"
    status_color = "\033[92m" if meeting_target else "\033[91m"
    reset_color = "\033[0m"

    print(f"FPS: {fps:.1f} / {target_fps:.1f} {status_color}{status}{reset_color}")
    print(f"     {format_bar(fps, target_fps * 1.5, 40)}")
    print()

    # Frame time
    frame_time = stats.get("frame_time_ms", 0)
    target_time = stats.get("target_frame_time_ms", 66.7)
    overhead = stats.get("overhead_ms", 0)

    print(f"Frame Time: {frame_time:.1f}ms / {target_time:.1f}ms")
    if overhead > 0:
        print(f"  Overhead: {overhead:.1f}ms (need to reduce processing time)")
    print(f"     {format_bar(frame_time, target_time * 2, 40)}")
    print()

    # Bottlenecks
    bottlenecks = stats.get("bottlenecks", [])
    if bottlenecks:
        print("Top Bottlenecks:")
        print("-" * 70)
        max_time = max([b["time_ms"] for b in bottlenecks]) if bottlenecks else 1

        for i, bottleneck in enumerate(bottlenecks[:5], 1):
            stage = bottleneck["stage"]
            time_ms = bottleneck["time_ms"]
            pct = (time_ms / frame_time * 100) if frame_time > 0 else 0

            print(f"{i}. {stage:20s}: {time_ms:6.1f}ms ({pct:5.1f}%)")
            print(f"   {format_bar(time_ms, max_time, 50)}")

    print()

    # Stats
    frame_count = stats.get("frame_count", 0)
    total_frames = stats.get("total_frames", 0)
    uptime = stats.get("uptime_seconds", 0)

    print(f"Frames Processed: {total_frames} (recent window: {frame_count})")
    print(f"Uptime: {uptime:.1f}s")

    print()
    print("=" * 70)
    print("Press Ctrl+C to exit")


def watch_performance(base_url: str, interval: float = 1.0):
    """Watch performance stats in real-time.

    Args:
        base_url: Base URL of the API
        interval: Update interval in seconds
    """
    api_url = f"{base_url}/api/v1/vision/performance"

    print(f"Connecting to {api_url}...")
    print("Waiting for data...")
    print()

    consecutive_errors = 0
    max_errors = 5

    try:
        while True:
            try:
                response = requests.get(api_url, timeout=5)
                response.raise_for_status()
                stats = response.json()

                # Reset error counter on success
                consecutive_errors = 0

                # Display the stats
                display_performance(stats)

            except requests.exceptions.RequestException as e:
                consecutive_errors += 1

                if consecutive_errors >= max_errors:
                    print(f"\n✗ Failed to connect after {max_errors} attempts")
                    print(f"Error: {e}")
                    print("\nMake sure the backend is running:")
                    print("  1. Start backend: python backend/main.py")
                    print(f"  2. Verify API is accessible: {api_url}")
                    sys.exit(1)

                print(f"Waiting for backend... ({consecutive_errors}/{max_errors})")
                print(f"Error: {e}")

            time.sleep(interval)

    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")
        sys.exit(0)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Real-time performance monitor for billiards vision"
    )
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:8000",
        help="Base URL of the API (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Update interval in seconds (default: 1.0)",
    )

    args = parser.parse_args()

    watch_performance(args.url, args.interval)


if __name__ == "__main__":
    main()
