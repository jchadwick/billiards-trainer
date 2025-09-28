#!/usr/bin/env python3
"""Example usage of the projector module.

This demonstrates how the projector module would be integrated
with the billiards trainer system.
"""

import os
import sys
import time

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from projector import (
    BasicRenderer,
    Colors,
    DisplayConfig,
    DisplayManager,
    DisplayMode,
    LineStyle,
    Point2D,
)


def example_trajectory_display():
    """Example of how to display trajectories."""
    print("Billiards Trainer - Projector Integration Example")
    print("=" * 50)

    # Create display configuration
    config = DisplayConfig(
        mode=DisplayMode.WINDOW,
        resolution=(800, 600),
        title="Billiards Trainer - Trajectory Display",
    )

    # Initialize display manager
    display_manager = DisplayManager(config)
    print(f"Display Manager: {display_manager}")

    try:
        # Start display (would work with real display)
        print("Starting display...")
        display_manager.start_display()

        # Create renderer
        renderer = BasicRenderer(display_manager.gl_context)
        renderer.set_projection_matrix(display_manager.width, display_manager.height)

        # Example trajectory data (from vision/physics system)
        trajectory_points = [
            Point2D(100, 300),  # Start position
            Point2D(200, 250),  # Along trajectory
            Point2D(350, 200),  # Collision point
            Point2D(500, 150),  # After collision
            Point2D(650, 100),  # Final position
        ]

        collision_point = Point2D(350, 200)
        ghost_ball = Point2D(320, 200)

        print("Rendering trajectory visualization...")

        # Rendering loop
        for _frame in range(60):  # 1 second at 60 FPS
            # Handle events
            if not display_manager.handle_events():
                break

            # Begin frame
            renderer.begin_frame()
            display_manager.clear_display()

            # Draw trajectory line
            for i in range(len(trajectory_points) - 1):
                color = Colors.TRAJECTORY if i < 2 else Colors.GREEN
                renderer.draw_line(
                    trajectory_points[i],
                    trajectory_points[i + 1],
                    color,
                    width=3.0,
                    style=LineStyle.SOLID,
                )

            # Draw collision indicator
            renderer.draw_circle(
                collision_point, radius=15, color=Colors.COLLISION, filled=True
            )

            # Draw ghost ball
            renderer.draw_circle(
                ghost_ball, radius=12, color=Colors.GHOST_BALL, filled=True
            )

            # Present frame
            display_manager.present_frame()
            renderer.end_frame()

            # Small delay for demo
            time.sleep(1 / 60)

        print(f"Rendered {renderer.get_stats().frames_rendered} frames")

    except Exception as e:
        print(f"Demo failed (expected in test environment): {e}")

    finally:
        # Cleanup
        display_manager.stop_display()
        print("Display stopped")


def example_integration_workflow():
    """Example of how the projector integrates with other modules."""
    print("\nIntegration Workflow Example")
    print("-" * 30)

    # This simulates how the projector would be used in the full system:

    # 1. Vision system detects ball positions
    {
        "cue_ball": Point2D(100, 300),
        "target_ball": Point2D(400, 200),
    }
    print("1. Vision system detected ball positions")

    # 2. Physics engine calculates trajectory
    trajectory = [
        Point2D(100, 300),
        Point2D(250, 250),
        Point2D(400, 200),
    ]
    print("2. Physics engine calculated trajectory")

    # 3. Analysis provides shot assistance
    shot_analysis = {
        "difficulty": 0.3,
        "success_probability": 0.8,
        "recommended_force": "medium",
    }
    print("3. Analysis engine provided shot assistance")

    # 4. Projector displays visualization
    DisplayConfig(mode=DisplayMode.FULLSCREEN, resolution=(1920, 1080))

    print("4. Projector would display:")
    print(f"   - Trajectory: {len(trajectory)} points")
    print(f"   - Success probability: {shot_analysis['success_probability']*100}%")
    print(f"   - Difficulty: {shot_analysis['difficulty']*100}%")

    # 5. Real-time updates as player aims
    print("5. Real-time updates during aiming")

    print("✓ Integration workflow complete")


if __name__ == "__main__":
    # Run trajectory display example
    example_trajectory_display()

    # Show integration workflow
    example_integration_workflow()

    print("\n🎯 Projector module is ready for full system integration!")
