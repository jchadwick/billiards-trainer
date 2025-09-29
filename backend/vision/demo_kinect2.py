#!/usr/bin/env python3
"""Kinect v2 demonstration for billiards trainer vision system.

This demo shows how to use the Kinect v2 enhanced vision capabilities,
including depth-based table detection and 3D ball tracking.

Usage:
    python demo_kinect2.py

Requirements:
    - Kinect v2 sensor connected via USB 3.0
    - libfreenect2 installed
    - pylibfreenect2 Python package installed
"""

import argparse
import logging
import time
from pathlib import Path

import cv2
import numpy as np
from capture import CameraCapture
from kinect2_capture import KINECT2_AVAILABLE


def setup_logging():
    """Setup logging for the demo."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def create_depth_colormap(depth_frame: np.ndarray) -> np.ndarray:
    """Create a colored visualization of the depth frame."""
    if depth_frame is None:
        return None

    # Normalize depth values for visualization
    depth_norm = cv2.normalize(depth_frame, None, 0, 255, cv2.NORM_MINMAX)
    depth_norm = depth_norm.astype(np.uint8)

    # Apply colormap for better visualization
    depth_colored = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)

    # Mask out invalid depth values (set to black)
    mask = depth_frame == 0
    depth_colored[mask] = [0, 0, 0]

    return depth_colored


def detect_table_with_depth(color_frame: np.ndarray, depth_frame: np.ndarray) -> dict:
    """Enhanced table detection using depth information."""
    if color_frame is None or depth_frame is None:
        return {}

    # Find the table surface using depth clustering
    # Table should be at a consistent depth level
    valid_depth = depth_frame[depth_frame > 0]
    if len(valid_depth) == 0:
        return {}

    # Find the most common depth (likely the table surface)
    hist, bins = np.histogram(valid_depth, bins=50)
    table_depth = bins[np.argmax(hist)]

    # Create mask for table surface (within ±50mm of table depth)
    table_mask = np.abs(depth_frame - table_depth) < 50
    table_mask = table_mask & (depth_frame > 0)

    # Find contours of the table surface
    table_mask_uint8 = table_mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(
        table_mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return {}

    # Find the largest contour (should be the table)
    largest_contour = max(contours, key=cv2.contourArea)

    # Approximate to quadrilateral
    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    table_corners = cv2.approxPolyDP(largest_contour, epsilon, True)

    return {
        "table_depth": float(table_depth),
        "table_mask": table_mask,
        "table_contour": largest_contour,
        "table_corners": table_corners.reshape(-1, 2)
        if len(table_corners) == 4
        else None,
        "table_area": cv2.contourArea(largest_contour),
    }


def detect_balls_with_depth(
    color_frame: np.ndarray, depth_frame: np.ndarray, table_info: dict
) -> list:
    """Enhanced ball detection using depth information."""
    if not table_info or color_frame is None or depth_frame is None:
        return []

    table_depth = table_info.get("table_depth")
    if table_depth is None:
        return []

    # Look for objects above the table surface (balls should be ~25mm above table)
    ball_height_min = table_depth - 50  # 50mm above table surface
    ball_height_max = table_depth - 10  # at least 10mm above table

    # Create mask for potential ball regions
    ball_mask = (depth_frame > ball_height_min) & (depth_frame < ball_height_max)
    ball_mask = ball_mask & (depth_frame > 0)

    # Find connected components (potential balls)
    ball_mask_uint8 = ball_mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(
        ball_mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    balls = []
    for contour in contours:
        area = cv2.contourArea(contour)

        # Filter by size (balls should have reasonable area)
        if area < 50 or area > 2000:
            continue

        # Get bounding circle
        (x, y), radius = cv2.minEnclosingCircle(contour)

        # Validate circularity
        perimeter = cv2.arcLength(contour, True)
        circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0

        if circularity < 0.7:  # Should be reasonably circular
            continue

        # Get average depth for this ball
        mask = np.zeros(depth_frame.shape, dtype=np.uint8)
        cv2.fillContour(mask, [contour], 255)
        ball_depths = depth_frame[mask > 0]
        avg_depth = np.mean(ball_depths[ball_depths > 0]) if len(ball_depths) > 0 else 0

        balls.append(
            {
                "center": (int(x), int(y)),
                "radius": int(radius),
                "depth": float(avg_depth),
                "area": area,
                "circularity": circularity,
                "contour": contour,
            }
        )

    return balls


def main():
    """Main demo function."""
    parser = argparse.ArgumentParser(description="Kinect v2 Vision Demo")
    parser.add_argument(
        "--save-frames", action="store_true", help="Save frames to disk"
    )
    parser.add_argument(
        "--output-dir", default="/tmp/kinect2_demo", help="Output directory"
    )
    args = parser.parse_args()

    setup_logging()

    if not KINECT2_AVAILABLE:
        print("ERROR: Kinect v2 support not available!")
        print("Please install libfreenect2 and pylibfreenect2:")
        print("  pip install pylibfreenect2")
        return 1

    # Create output directory if saving frames
    if args.save_frames:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving frames to: {output_dir}")

    # Configure Kinect v2 camera
    config = {
        "backend": "kinect2",
        "enable_color": True,
        "enable_depth": True,
        "min_depth": 500,
        "max_depth": 4000,
        "auto_reconnect": True,
    }

    print("Initializing Kinect v2...")
    camera = CameraCapture(config)

    try:
        if not camera.start_capture():
            print("ERROR: Failed to start Kinect v2 capture")
            return 1

        print("Kinect v2 capture started successfully")
        print("Camera info:", camera.get_camera_info())
        print("\nPress 'q' to quit, 's' to save frame, 'i' to show info")

        frame_count = 0
        start_time = time.time()

        while True:
            # Get latest frame
            frame_data = camera.get_latest_frame()
            if frame_data is None:
                time.sleep(0.01)
                continue

            color_frame, frame_info = frame_data
            depth_frame = camera.get_depth_frame()

            frame_count += 1
            current_time = time.time()

            # Create visualization
            display_frame = color_frame.copy()

            if depth_frame is not None:
                # Enhanced detection using depth
                table_info = detect_table_with_depth(color_frame, depth_frame)
                balls = detect_balls_with_depth(color_frame, depth_frame, table_info)

                # Draw table detection
                if table_info.get("table_corners") is not None:
                    corners = table_info["table_corners"]
                    cv2.polylines(display_frame, [corners], True, (0, 255, 0), 2)
                    cv2.putText(
                        display_frame,
                        f"Table Depth: {table_info['table_depth']:.0f}mm",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                    )

                # Draw ball detections
                for i, ball in enumerate(balls):
                    center = ball["center"]
                    radius = ball["radius"]
                    depth = ball["depth"]

                    # Draw circle
                    cv2.circle(display_frame, center, radius, (255, 0, 0), 2)
                    cv2.circle(display_frame, center, 2, (255, 0, 0), -1)

                    # Add depth label
                    cv2.putText(
                        display_frame,
                        f"{depth:.0f}mm",
                        (center[0] - 20, center[1] - radius - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 0, 0),
                        1,
                    )

                # Create depth visualization
                depth_colored = create_depth_colormap(depth_frame)

                if depth_colored is not None:
                    # Resize depth for side-by-side display
                    h, w = color_frame.shape[:2]
                    depth_resized = cv2.resize(depth_colored, (w // 2, h // 2))

                    # Add depth visualization to corner
                    display_frame[0 : h // 2, -w // 2 :] = depth_resized
                    cv2.rectangle(
                        display_frame, (-w // 2, 0), (w, h // 2), (255, 255, 255), 2
                    )
                    cv2.putText(
                        display_frame,
                        "Depth",
                        (-w // 2 + 10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                    )

            # Add frame info
            fps = (
                frame_count / (current_time - start_time)
                if current_time > start_time
                else 0
            )
            cv2.putText(
                display_frame,
                f"FPS: {fps:.1f} | Frame: {frame_count}",
                (10, display_frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            # Display frame
            cv2.imshow("Kinect v2 Billiards Trainer Demo", display_frame)

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s") and args.save_frames:
                # Save current frame
                timestamp = int(time.time() * 1000)
                color_path = output_dir / f"color_{timestamp}.jpg"
                cv2.imwrite(str(color_path), color_frame)

                if depth_frame is not None:
                    depth_path = output_dir / f"depth_{timestamp}.png"
                    cv2.imwrite(str(depth_path), depth_frame.astype(np.uint16))

                print(f"Saved frame {timestamp}")
            elif key == ord("i"):
                # Show camera info
                health = camera.get_health()
                calib = camera.get_kinect_calibration()
                print("\n=== Camera Health ===")
                print(f"Status: {health.status}")
                print(f"Frames captured: {health.frames_captured}")
                print(f"Frames dropped: {health.frames_dropped}")
                print(f"Current FPS: {health.fps:.2f}")
                print(f"Error count: {health.error_count}")
                if calib:
                    print("\n=== Calibration Available ===")
                    print("Kinect v2 calibration parameters loaded")
                print()

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        logging.exception("Demo error")
    finally:
        print("Stopping camera...")
        camera.stop_capture()
        cv2.destroyAllWindows()
        print("Demo finished")

    return 0


if __name__ == "__main__":
    exit(main())
