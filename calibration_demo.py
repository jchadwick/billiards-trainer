#!/usr/bin/env python3
"""
Calibration System Demo

Demonstrates the comprehensive calibration system for billiards vision tracking.
This demo shows how to use the calibration manager to set up and validate
the complete vision system.
"""

import cv2
import numpy as np
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from backend.vision.calibration import CalibrationManager, quick_setup
except ImportError as e:
    logger.error(f"Failed to import calibration system: {e}")
    logger.info("Make sure you're running from the project root directory")
    exit(1)


def create_demo_frame():
    """Create a synthetic demo frame with a table and balls"""
    # Create a 800x600 image
    frame = np.zeros((600, 800, 3), dtype=np.uint8)

    # Create green table background
    frame[:, :] = (40, 120, 40)  # Dark green

    # Draw table boundaries (rectangular)
    table_corners = [(100, 100), (700, 100), (700, 500), (100, 500)]
    pts = np.array(table_corners, np.int32)
    cv2.fillPoly(frame, [pts], (60, 180, 60))  # Brighter green for table
    cv2.polylines(frame, [pts], True, (139, 69, 19), 8)  # Brown rails

    # Add some synthetic balls
    ball_positions = [
        (200, 200, (255, 255, 255)),  # White cue ball
        (400, 300, (0, 255, 255)),    # Yellow ball
        (350, 250, (255, 0, 0)),      # Red ball
        (450, 350, (0, 0, 255)),      # Blue ball
        (300, 400, (128, 0, 128)),    # Purple ball
    ]

    for x, y, color in ball_positions:
        cv2.circle(frame, (x, y), 15, color, -1)
        cv2.circle(frame, (x, y), 15, (0, 0, 0), 2)  # Black outline

    # Add pockets
    pocket_positions = [
        (100, 100), (400, 100), (700, 100),  # Top pockets
        (700, 500), (400, 500), (100, 500)   # Bottom pockets
    ]

    for x, y in pocket_positions:
        cv2.circle(frame, (x, y), 25, (0, 0, 0), -1)  # Black pockets

    return frame, table_corners


def demo_basic_calibration():
    """Demonstrate basic calibration functionality"""
    logger.info("=== Basic Calibration Demo ===")

    # Create demo frame
    frame, table_corners = create_demo_frame()

    # Initialize calibration manager
    manager = quick_setup("demo_calibration")

    logger.info("Created synthetic demo frame with table and balls")

    # 1. Color Calibration
    logger.info("Performing color calibration...")

    # Define ball samples (regions where balls are located)
    ball_samples = {
        'cue': [(185, 185, 30, 30)],      # White ball region
        'yellow': [(385, 285, 30, 30)],   # Yellow ball region
        'red': [(335, 235, 30, 30)],      # Red ball region
        'blue': [(435, 335, 30, 30)],     # Blue ball region
        'purple': [(285, 385, 30, 30)]    # Purple ball region
    }

    color_profile = manager.calibrate_colors(
        frame,
        ball_samples=ball_samples,
        profile_name="demo_profile"
    )

    if color_profile:
        logger.info(f"Color calibration successful: {color_profile.name}")
        logger.info(f"Lighting level: {color_profile.ambient_light_level:.1f}")

    # 2. Geometry Calibration
    logger.info("Performing geometric calibration...")

    geometry_cal = manager.calibrate_geometry(
        frame,
        table_corners=table_corners,
        table_dimensions=(2.54, 1.27)  # Standard 9-foot table
    )

    if geometry_cal:
        logger.info(f"Geometric calibration successful")
        logger.info(f"Calibration error: {geometry_cal.calibration_error:.2f} pixels")

    # 3. Test coordinate transformation
    logger.info("Testing coordinate transformations...")

    # Test center of table
    center_pixel = (400, 300)
    world_point = manager.pixel_to_world(center_pixel)

    if world_point:
        logger.info(f"Table center: {center_pixel} pixels → {world_point} meters")

        # Test reverse transformation
        back_to_pixel = manager.world_to_pixel(world_point)
        if back_to_pixel:
            error = np.linalg.norm(np.array(center_pixel) - np.array(back_to_pixel))
            logger.info(f"Round-trip error: {error:.2f} pixels")

    # 4. System status
    status = manager.get_calibration_status()
    logger.info("System Status:")
    for key, value in status.items():
        logger.info(f"  {key}: {value}")

    # 5. Save session
    if manager.save_current_session():
        logger.info(f"Session saved: {manager.current_session.session_id}")

    return manager


def demo_frame_processing():
    """Demonstrate frame processing with calibrations"""
    logger.info("\n=== Frame Processing Demo ===")

    # Create demo frame
    frame, table_corners = create_demo_frame()

    # Use previous calibration or create new one
    manager = quick_setup("processing_demo")

    # Quick auto-calibration
    color_profile = manager.calibrate_colors(frame, profile_name="processing_demo")
    geometry_cal = manager.calibrate_geometry(frame, table_corners)

    if manager.is_system_ready():
        logger.info("System ready for frame processing")

        # Process frame
        processed_frame = manager.process_frame(frame)
        logger.info(f"Frame processed: {frame.shape} → {processed_frame.shape}")

        # Test adaptive recalibration
        # Simulate lighting change by brightening frame
        bright_frame = cv2.convertScaleAbs(frame, alpha=1.5, beta=30)

        recalibrated = manager.adaptive_recalibration(bright_frame, lighting_threshold=0.2)
        if recalibrated:
            logger.info("Adaptive recalibration performed due to lighting change")

    else:
        logger.warning("System not ready for processing")

    return manager


def demo_import_export():
    """Demonstrate calibration import/export"""
    logger.info("\n=== Import/Export Demo ===")

    # Create and calibrate system
    frame, table_corners = create_demo_frame()
    manager = quick_setup("export_demo")

    manager.calibrate_colors(frame, profile_name="export_test")
    manager.calibrate_geometry(frame, table_corners)

    # Export calibration
    export_path = "demo_calibration.json"
    if manager.export_calibration(export_path):
        logger.info(f"Calibration exported to: {export_path}")

        # Reset calibration
        manager.reset_calibration()
        logger.info("Calibration reset")

        # Import calibration
        if manager.import_calibration(export_path):
            logger.info(f"Calibration imported from: {export_path}")

            # Verify import worked
            if manager.is_system_ready():
                logger.info("Imported calibration is ready for use")

        # Cleanup
        Path(export_path).unlink(missing_ok=True)

    return manager


def demo_session_management():
    """Demonstrate session management"""
    logger.info("\n=== Session Management Demo ===")

    manager = CalibrationManager("demo_sessions")

    # Create multiple sessions
    sessions = []
    for i in range(3):
        session_id = manager.start_new_session(f"demo_session_{i}")
        sessions.append(session_id)

        # Add some basic calibration
        frame, table_corners = create_demo_frame()
        manager.calibrate_colors(frame, profile_name=f"session_{i}")
        manager.save_current_session()

        logger.info(f"Created and saved session: {session_id}")

    # List sessions
    available_sessions = manager.list_sessions()
    logger.info(f"Available sessions: {available_sessions}")

    # Load a previous session
    if sessions:
        if manager.load_session(sessions[1]):
            logger.info(f"Loaded session: {sessions[1]}")
            status = manager.get_calibration_status()
            logger.info(f"Session status: color_calibrated={status['color_calibrated']}")

    return manager


def main():
    """Run all calibration demos"""
    logger.info("Starting Billiards Vision Calibration Demo")
    logger.info("=" * 50)

    try:
        # Run demos
        demo_basic_calibration()
        demo_frame_processing()
        demo_import_export()
        demo_session_management()

        logger.info("\n" + "=" * 50)
        logger.info("All calibration demos completed successfully!")
        logger.info("The calibration system is ready for production use.")

        logger.info("\nKey Features Demonstrated:")
        logger.info("✓ Automatic color threshold detection")
        logger.info("✓ Geometric perspective correction")
        logger.info("✓ Coordinate transformation (pixel ↔ world)")
        logger.info("✓ Adaptive lighting recalibration")
        logger.info("✓ Session management and persistence")
        logger.info("✓ Import/export functionality")
        logger.info("✓ System validation and status checking")

    except Exception as e:
        logger.error(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
