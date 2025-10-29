#!/usr/bin/env python3
"""Example usage of VisionConfigurationManager video helper methods.

This example demonstrates how to configure and use video file input
with the VisionConfigurationManager.
"""

from backend.vision.config_manager import VisionConfigurationManager


def example_video_file_input():
    """Example: Configure camera for video file input."""
    print("\n=== Video File Input Example ===\n")

    # Create configuration manager
    manager = VisionConfigurationManager()
    manager.initialize()

    # Configure for video file input
    # Simply set video_file_path - no need for video_source_type
    video_config = {
        "camera": {
            "video_file_path": "/path/to/video.mp4",
            "loop_video": True,
            "video_start_frame": 0,
            "video_end_frame": 1000,
            "fps": 30,
            "resolution": [1920, 1080],
        }
    }

    # Update configuration
    manager.update_config(video_config)

    # Check if using video file input
    is_file = manager.is_video_file_input()
    print(f"Is video file input: {is_file}")

    # Get video file path (returns absolute path)
    try:
        file_path = manager.get_video_file_path()
        print(f"Video file path: {file_path}")
    except FileNotFoundError as e:
        print(f"Warning: {e}")

    # Get complete camera configuration for CameraCapture
    camera_config = manager.resolve_camera_config()
    print("\nCamera configuration:")
    print(f"  device_id: {camera_config['device_id']}")
    print(f"  loop_video: {camera_config['loop_video']}")
    print(f"  video_start_frame: {camera_config['video_start_frame']}")
    print(f"  video_end_frame: {camera_config['video_end_frame']}")
    print(f"  fps: {camera_config['fps']}")


def example_backward_compatibility():
    """Example: Backward compatibility with string device_id."""
    print("\n=== Backward Compatibility Example ===\n")

    # Create configuration manager
    manager = VisionConfigurationManager()
    manager.initialize()

    # Old-style configuration (device_id as string)
    legacy_config = {
        "camera": {
            "device_id": "/path/to/video.mp4",
            "fps": 30,
        }
    }

    manager.update_config(legacy_config)

    # The new helper methods work with legacy config
    is_file = manager.is_video_file_input()
    print(f"Is video file input (legacy): {is_file}")

    # resolve_camera_config preserves string device_id
    camera_config = manager.resolve_camera_config()
    print(f"Device ID (legacy): {camera_config['device_id']}")


def example_stream_input():
    """Example: Configure camera for stream input."""
    print("\n=== Stream Input Example ===\n")

    # Create configuration manager
    manager = VisionConfigurationManager()
    manager.initialize()

    # Configure for stream input
    # Use device_id for stream URLs
    stream_config = {
        "camera": {
            "device_id": "rtsp://example.com/stream",
            "fps": 30,
            "resolution": [1280, 720],
        }
    }

    manager.update_config(stream_config)

    # Get camera configuration
    camera_config = manager.resolve_camera_config()
    print(f"Device ID (stream URL): {camera_config['device_id']}")


def example_camera_input():
    """Example: Configure camera for hardware camera input."""
    print("\n=== Camera Input Example ===\n")

    # Create configuration manager
    manager = VisionConfigurationManager()
    manager.initialize()

    # Configure for hardware camera
    # Use device_id with integer for hardware camera (default is 0)
    camera_config = {
        "camera": {
            "device_id": 0,  # Camera index
            "fps": 60,
            "resolution": [1920, 1080],
            "exposure_mode": "auto",
        }
    }

    manager.update_config(camera_config)

    # Check configuration
    is_file = manager.is_video_file_input()
    print(f"Is video file input: {is_file}")

    # Get camera configuration
    config = manager.resolve_camera_config()
    print(f"Device ID: {config['device_id']}")
    print(f"FPS: {config['fps']}")


def example_validation():
    """Example: Validate video file configuration."""
    print("\n=== Configuration Validation Example ===\n")

    manager = VisionConfigurationManager()

    # Valid configuration with video file
    valid_config = {
        "camera": {
            "video_file_path": "/tmp/test.mp4",
            "device_id": 0,
        },
        "detection": {},
        "processing": {},
    }

    is_valid, errors = manager.validate_config(valid_config)
    print(f"Valid config: {is_valid}")
    if errors:
        print(f"Errors: {errors}")

    # Invalid configuration (video file doesn't exist)
    invalid_config = {
        "camera": {
            "video_file_path": "/nonexistent/video.mp4",
        },
        "detection": {},
        "processing": {},
    }

    is_valid, errors = manager.validate_config(invalid_config)
    print(f"\nInvalid config: {is_valid}")
    print(f"Errors: {errors}")


def example_create_camera_capture():
    """Example: Create CameraCapture with video file."""
    print("\n=== Create CameraCapture Example ===\n")

    # Create configuration manager
    manager = VisionConfigurationManager()
    manager.initialize()

    # Configure for video file - just set video_file_path
    video_config = {
        "camera": {
            "video_file_path": "/path/to/video.mp4",
            "loop_video": True,
            "video_start_frame": 100,
            "video_end_frame": 500,
        }
    }

    manager.update_config(video_config)

    # Create CameraCapture instance
    # The manager will automatically resolve the configuration
    # and pass video file settings to CameraCapture
    try:
        camera = manager.create_camera_capture()
        print("CameraCapture created successfully")
        print(f"Camera info: {camera.get_camera_info()}")
    except Exception as e:
        print(f"Failed to create camera: {e}")


def main():
    """Run all examples."""
    print("=" * 60)
    print("VisionConfigurationManager Video Helper Methods Examples")
    print("=" * 60)

    example_video_file_input()
    example_backward_compatibility()
    example_stream_input()
    example_camera_input()
    example_validation()
    example_create_camera_capture()

    print("\n" + "=" * 60)
    print("Examples complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
