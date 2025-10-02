#!/usr/bin/env python3
"""Debug script to test table detection on a captured frame."""

import sys
import cv2
import numpy as np

# Add backend to path
sys.path.insert(0, 'backend')

from vision.detection.table import TableDetector

# Load the frame
frame = cv2.imread('/tmp/table_frame.jpg')
if frame is None:
    print("ERROR: Could not load frame from /tmp/table_frame.jpg")
    sys.exit(1)

print(f"Frame shape: {frame.shape}")
print(f"Frame dtype: {frame.dtype}")

# Configure table detector with debug mode
table_config = {
    "expected_aspect_ratio": 2.0,  # 9-foot table is 2:1
    "aspect_ratio_tolerance": 0.3,
    "debug_mode": True,
}

detector = TableDetector(table_config)

print("\n=== Testing table detection ===")

# Step-by-step debugging
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
mask = detector._create_table_color_mask(hsv)
print(f"1. Color mask created: {cv2.countNonZero(mask)} pixels")
cv2.imwrite('/tmp/debug_color_mask.jpg', mask)

contour = detector._find_table_contour(mask)
if contour is None:
    print("2. ERROR: No table contour found!")
else:
    print(f"2. Table contour found: {len(contour)} points, area={cv2.contourArea(contour):.0f}")

    # Visualize contour
    viz_contour = frame.copy()
    cv2.drawContours(viz_contour, [contour], -1, (0, 255, 0), 3)
    cv2.imwrite('/tmp/debug_contour.jpg', viz_contour)

    # Try corner refinement
    epsilon = detector.contour_epsilon_factor * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    print(f"3. Polygon approximation: {len(approx)} points (need 4)")

    if len(approx) != 4:
        # Try different epsilon values
        for epsilon_mult in [0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.1]:
            epsilon = epsilon_mult * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            print(f"   Trying epsilon={epsilon_mult}: {len(approx)} points")
            if len(approx) == 4:
                break

    if len(approx) == 4:
        # We have 4 points, check geometry
        corners_raw = approx.reshape(-1, 2).astype(np.float32)
        from backend.vision.detection.table import TableCorners

        # Sort corners
        center = np.mean(corners_raw, axis=0)
        import math
        def angle_from_center(point):
            return math.atan2(point[1] - center[1], point[0] - center[0])

        corners_with_angles = [(corner, angle_from_center(corner)) for corner in corners_raw]
        corners_with_angles.sort(key=lambda x: x[1])
        sorted_corners = [corner for corner, _ in corners_with_angles]

        # Create TableCorners (assuming order is TL, TR, BL, BR - we'll verify)
        test_corners = TableCorners(
            top_left=tuple(sorted_corners[0]),
            top_right=tuple(sorted_corners[1]),
            bottom_left=tuple(sorted_corners[2]),
            bottom_right=tuple(sorted_corners[3]),
        )

        corner_list = test_corners.to_list()

        # Calculate side lengths
        def dist(p1, p2):
            return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

        width1 = dist(corner_list[0], corner_list[1])
        width2 = dist(corner_list[2], corner_list[3])
        height1 = dist(corner_list[0], corner_list[2])
        height2 = dist(corner_list[1], corner_list[3])

        width_diff = abs(width1 - width2) / max(width1, width2)
        height_diff = abs(height1 - height2) / max(height1, height2)

        avg_width = (width1 + width2) / 2
        avg_height = (height1 + height2) / 2
        aspect_ratio = avg_width / avg_height

        print(f"\n   Geometry validation:")
        print(f"     Width top={width1:.1f}, bottom={width2:.1f}, diff={width_diff*100:.1f}% (need <10%)")
        print(f"     Height left={height1:.1f}, right={height2:.1f}, diff={height_diff*100:.1f}% (need <10%)")
        print(f"     Aspect ratio: {aspect_ratio:.2f} (expected: 2.0 ± 0.3)")

        if width_diff > 0.1:
            print(f"     FAIL: Width difference too large!")
        if height_diff > 0.1:
            print(f"     FAIL: Height difference too large!")
        if abs(aspect_ratio - 2.0) > 0.3:
            print(f"     FAIL: Aspect ratio out of range!")

corners = detector.detect_table_boundaries(frame)

if corners is None:
    print("\n4. ERROR: Table detection failed!")
    print("\n=== Debugging info ===")

    # Try color detection
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Test each color range
    for color_name, color_range in detector.table_color_ranges.items():
        mask = cv2.inRange(hsv, color_range["lower"], color_range["upper"])
        area = cv2.countNonZero(mask)
        percent = (area / (frame.shape[0] * frame.shape[1])) * 100
        print(f"{color_name}: {area} pixels ({percent:.2f}%)")

    # Save debug images if available
    if hasattr(detector, 'debug_images') and detector.debug_images:
        print(f"\nDebug images: {len(detector.debug_images)}")
        for name, img in detector.debug_images:
            debug_path = f'/tmp/debug_{name}.jpg'
            cv2.imwrite(debug_path, img)
            print(f"  Saved: {debug_path}")
else:
    print("SUCCESS: Table detected!")
    print(f"  Top-left: {corners.top_left}")
    print(f"  Top-right: {corners.top_right}")
    print(f"  Bottom-left: {corners.bottom_left}")
    print(f"  Bottom-right: {corners.bottom_right}")

    # Draw corners on frame
    viz = frame.copy()
    corners_list = corners.to_list()
    for i, corner in enumerate(corners_list):
        cv2.circle(viz, (int(corner[0]), int(corner[1])), 10, (0, 255, 0), -1)
        cv2.putText(viz, str(i), (int(corner[0])+15, int(corner[1])),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Draw rectangle
    pts = np.array(corners_list, np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.polylines(viz, [pts], True, (0, 255, 0), 3)

    cv2.imwrite('/tmp/table_detected.jpg', viz)
    print("\nVisualization saved to: /tmp/table_detected.jpg")
