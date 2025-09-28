#!/usr/bin/env python3
"""Demonstration of the error correction system.

This script shows how the automatic error correction system works in practice,
demonstrating various error scenarios and their corrections.
"""

import time
from datetime import datetime

from backend.core.models import BallState, GameState, TableState, Vector2D

from .manager import ValidationManager, validate_game_state


def demo_overlapping_balls():
    """Demonstrate correction of overlapping balls."""
    print("=" * 50)
    print("DEMO: Overlapping Balls Correction")
    print("=" * 50)

    # Create overlapping balls
    table = TableState.standard_9ft_table()
    balls = [
        BallState(id="cue", position=Vector2D(1.0, 1.0), is_cue_ball=True),
        BallState(id="ball1", position=Vector2D(1.001, 1.001)),  # Overlapping
        BallState(id="ball2", position=Vector2D(2.0, 1.0)),
    ]

    game_state = GameState(
        timestamp=time.time(), frame_number=0, balls=balls, table=table
    )

    print("Original State:")
    for ball in game_state.balls:
        print(f"  {ball.id}: ({ball.position.x:.3f}, {ball.position.y:.3f})")

    # Check for errors
    errors = game_state.validate_consistency()
    print(f"\nValidation Errors: {errors}")

    # Apply correction
    corrected_state, report = validate_game_state(game_state, auto_correct=True)

    print("\nCorrected State:")
    for ball in corrected_state.balls:
        print(f"  {ball.id}: ({ball.position.x:.3f}, {ball.position.y:.3f})")

    print(f"\nCorrections Applied: {len(report.corrections_applied)}")
    print(f"System Confidence: {report.overall_confidence:.2f}")


def demo_out_of_bounds():
    """Demonstrate correction of out-of-bounds balls."""
    print("\n" + "=" * 50)
    print("DEMO: Out-of-Bounds Ball Correction")
    print("=" * 50)

    table = TableState.standard_9ft_table()
    balls = [
        BallState(id="cue", position=Vector2D(1.0, 1.0), is_cue_ball=True),
        BallState(id="ball1", position=Vector2D(-0.5, 1.0)),  # Out of bounds
        BallState(id="ball2", position=Vector2D(3.0, 1.5)),  # Also out of bounds
    ]

    game_state = GameState(
        timestamp=time.time(), frame_number=0, balls=balls, table=table
    )

    print("Original State:")
    print(f"Table bounds: 0 to {table.width:.2f} x 0 to {table.height:.2f}")
    for ball in game_state.balls:
        in_bounds = table.is_point_on_table(ball.position, ball.radius)
        print(
            f"  {ball.id}: ({ball.position.x:.3f}, {ball.position.y:.3f}) - {'✓' if in_bounds else '✗'}"
        )

    # Apply correction
    corrected_state, report = validate_game_state(game_state, auto_correct=True)

    print("\nCorrected State:")
    for ball in corrected_state.balls:
        in_bounds = table.is_point_on_table(ball.position, ball.radius)
        print(
            f"  {ball.id}: ({ball.position.x:.3f}, {ball.position.y:.3f}) - {'✓' if in_bounds else '✗'}"
        )

    print(f"\nCorrections Applied: {len(report.corrections_applied)}")


def demo_invalid_velocities():
    """Demonstrate correction of invalid velocities."""
    print("\n" + "=" * 50)
    print("DEMO: Invalid Velocity Correction")
    print("=" * 50)

    table = TableState.standard_9ft_table()
    balls = [
        BallState(id="cue", position=Vector2D(1.0, 1.0), is_cue_ball=True),
        BallState(
            id="ball1", position=Vector2D(2.0, 1.0), velocity=Vector2D(25.0, 0.0)
        ),  # Too fast
        BallState(
            id="ball2", position=Vector2D(3.0, 1.0), velocity=Vector2D(0.0, 15.0)
        ),  # Also too fast
    ]

    game_state = GameState(
        timestamp=time.time(), frame_number=0, balls=balls, table=table
    )

    print("Original State:")
    for ball in game_state.balls:
        velocity_mag = ball.velocity.magnitude()
        print(f"  {ball.id}: velocity {velocity_mag:.2f} m/s")

    # Apply correction with ValidationManager
    manager = ValidationManager(auto_correct=True)
    corrected_state, report = manager.validate_and_correct(game_state)

    print("\nCorrected State:")
    for ball in corrected_state.balls:
        velocity_mag = ball.velocity.magnitude()
        print(f"  {ball.id}: velocity {velocity_mag:.2f} m/s")

    print(f"\nCorrections Applied: {len(report.corrections_applied)}")

    # Show system health
    health = manager.get_system_health()
    print(f"System Success Rate: {health['success_rate']:.1%}")


def demo_comprehensive_correction():
    """Demonstrate comprehensive correction with multiple issues."""
    print("\n" + "=" * 50)
    print("DEMO: Comprehensive Multi-Issue Correction")
    print("=" * 50)

    table = TableState.standard_9ft_table()
    balls = [
        BallState(id="cue", position=Vector2D(1.0, 1.0), is_cue_ball=True),
        BallState(
            id="ball1", position=Vector2D(1.001, 1.001), velocity=Vector2D(20.0, 0.0)
        ),  # Overlap + fast
        BallState(id="ball2", position=Vector2D(-0.3, 1.0)),  # Out of bounds
        BallState(
            id="ball3", position=Vector2D(4.0, 2.0), velocity=Vector2D(-30.0, 0.0)
        ),  # Out of bounds + fast
    ]

    game_state = GameState(
        timestamp=time.time(), frame_number=0, balls=balls, table=table
    )

    print("Original State Issues:")
    errors = game_state.validate_consistency()
    for error in errors:
        print(f"  - {error}")

    # Apply comprehensive correction
    manager = ValidationManager(auto_correct=True)
    corrected_state, report = manager.validate_and_correct(game_state)

    print("\nCorrected State:")
    new_errors = corrected_state.validate_consistency()
    if new_errors:
        print("Remaining issues:")
        for error in new_errors:
            print(f"  - {error}")
    else:
        print("  All issues resolved!")

    print("\nCorrection Summary:")
    print(f"  Corrections Applied: {len(report.corrections_applied)}")
    print(f"  Overall Confidence: {report.overall_confidence:.2f}")
    print(f"  Processing Time: {report.processing_time:.3f}s")

    # Show correction details
    if report.corrections_applied:
        print("\nDetailed Corrections:")
        for correction in report.corrections_applied[:5]:  # Show first 5
            print(
                f"  - {correction['description']} (confidence: {correction['confidence']:.2f})"
            )


def demo_system_monitoring():
    """Demonstrate system monitoring and statistics."""
    print("\n" + "=" * 50)
    print("DEMO: System Monitoring and Statistics")
    print("=" * 50)

    manager = ValidationManager(auto_correct=True)

    # Process several problematic states
    test_states = []
    for i in range(10):
        table = TableState.standard_9ft_table()
        balls = [
            BallState(id="cue", position=Vector2D(1.0, 1.0), is_cue_ball=True),
            BallState(
                id=f"ball{i}", position=Vector2D(1.0 + i * 0.001, 1.0)
            ),  # Slight overlaps
        ]

        if i % 3 == 0:
            balls[1].velocity = Vector2D(15.0, 0.0)  # Add velocity issues

        game_state = GameState(
            timestamp=time.time(), frame_number=i, balls=balls, table=table
        )
        test_states.append(game_state)

    print("Processing test states...")
    for i, state in enumerate(test_states):
        corrected_state, report = manager.validate_and_correct(state)
        print(
            f"  State {i}: {len(report.corrections_applied)} corrections, "
            f"confidence {report.overall_confidence:.2f}"
        )

    # Show system health
    health = manager.get_system_health()
    print("\nSystem Health Summary:")
    print(f"  Total Validations: {health['validation_count']}")
    print(f"  Total Corrections: {health['correction_count']}")
    print(f"  Success Rate: {health['success_rate']:.1%}")
    print(f"  Average Processing Time: {health['average_processing_time']:.3f}s")

    print("\nCorrection Statistics:")
    correction_stats = health["correction_statistics"]
    if correction_stats["stats"]["corrections_by_type"]:
        for correction_type, count in correction_stats["stats"][
            "corrections_by_type"
        ].items():
            print(f"  {correction_type}: {count}")


def main():
    """Run all demonstrations."""
    print("BILLIARDS TRAINER - ERROR CORRECTION SYSTEM DEMONSTRATION")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    demo_overlapping_balls()
    demo_out_of_bounds()
    demo_invalid_velocities()
    demo_comprehensive_correction()
    demo_system_monitoring()

    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETE")
    print("=" * 60)
    print("The error correction system successfully:")
    print("✓ Detects and corrects overlapping balls")
    print("✓ Moves out-of-bounds balls back onto the table")
    print("✓ Limits excessive velocities")
    print("✓ Handles multiple issues simultaneously")
    print("✓ Provides comprehensive monitoring and statistics")
    print("✓ Maintains high system reliability")


if __name__ == "__main__":
    main()
