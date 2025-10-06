"""Demonstration script for the hybrid validator.

Shows the complete functionality of the HybridValidator class without
requiring the full vision module import.
"""

import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(backend_path))

import cv2
import numpy as np
from vision.detection.hybrid_validator import HybridValidator, ValidationConfig

# Direct imports to avoid module initialization issues
from vision.models import Ball, BallType


def create_test_ball_image(ball_type: BallType, radius: int = 20) -> np.ndarray:
    """Create a synthetic ball image for testing.

    Args:
        ball_type: Type of ball to create
        radius: Ball radius in pixels

    Returns:
        Image containing the ball
    """
    size = radius * 4
    image = np.zeros((size, size, 3), dtype=np.uint8)
    center = (size // 2, size // 2)

    # Set color based on ball type
    if ball_type == BallType.CUE:
        color = (255, 255, 255)  # White
    elif ball_type == BallType.EIGHT:
        color = (0, 0, 0)  # Black
    elif ball_type == BallType.SOLID:
        color = (0, 0, 255)  # Red (example solid)
    elif ball_type == BallType.STRIPE:
        # White with colored stripe
        cv2.circle(image, center, radius, (255, 255, 255), -1)
        # Add stripe
        cv2.rectangle(
            image,
            (center[0] - 5, center[1] - radius),
            (center[0] + 5, center[1] + radius),
            (0, 0, 255),
            -1,
        )
        return image
    else:
        color = (128, 128, 128)  # Gray for unknown

    # Draw ball
    cv2.circle(image, center, radius, color, -1)

    # Add specular highlight for realism
    highlight_center = (center[0] - radius // 3, center[1] - radius // 3)
    cv2.circle(image, highlight_center, radius // 4, (255, 255, 255), -1)

    return image


def demonstrate_color_validation():
    """Demonstrate color histogram validation."""
    print("\n" + "=" * 60)
    print("COLOR VALIDATION DEMONSTRATION")
    print("=" * 60)

    validator = HybridValidator()

    ball_types = [BallType.CUE, BallType.EIGHT, BallType.SOLID]

    for ball_type in ball_types:
        # Create test ball
        ball = Ball(
            position=(40.0, 40.0),
            radius=20.0,
            ball_type=ball_type,
            confidence=0.9,
        )

        # Create ball image
        frame = create_test_ball_image(ball_type, radius=20)

        # Validate
        multiplier = validator.validate_ball_detection(ball, frame)

        print(f"\n{ball_type.value.upper()} Ball:")
        print(f"  Original confidence: {ball.confidence:.3f}")
        print(f"  Validation multiplier: {multiplier:.3f}")
        print(f"  Adjusted confidence: {ball.confidence * multiplier:.3f}")


def demonstrate_circularity_validation():
    """Demonstrate circularity validation using Hough circles."""
    print("\n" + "=" * 60)
    print("CIRCULARITY VALIDATION DEMONSTRATION")
    print("=" * 60)

    validator = HybridValidator()

    # Test with perfect circle
    print("\n1. Perfect Circle:")
    frame = create_test_ball_image(BallType.CUE, radius=20)
    ball = Ball(
        position=(40.0, 40.0), radius=20.0, ball_type=BallType.CUE, confidence=0.9
    )
    multiplier = validator.validate_ball_detection(ball, frame)
    print(f"   Validation multiplier: {multiplier:.3f}")

    # Test with non-circular shape (rectangle)
    print("\n2. Non-circular Shape (Rectangle):")
    frame = np.zeros((80, 80, 3), dtype=np.uint8)
    cv2.rectangle(frame, (20, 20), (60, 60), (255, 255, 255), -1)
    ball = Ball(
        position=(40.0, 40.0), radius=20.0, ball_type=BallType.CUE, confidence=0.9
    )
    multiplier = validator.validate_ball_detection(ball, frame)
    print(f"   Validation multiplier: {multiplier:.3f}")

    # Test with ellipse
    print("\n3. Ellipse (Slightly Non-circular):")
    frame = np.zeros((80, 80, 3), dtype=np.uint8)
    cv2.ellipse(frame, (40, 40), (20, 15), 0, 0, 360, (255, 255, 255), -1)
    ball = Ball(
        position=(40.0, 40.0), radius=20.0, ball_type=BallType.CUE, confidence=0.9
    )
    multiplier = validator.validate_ball_detection(ball, frame)
    print(f"   Validation multiplier: {multiplier:.3f}")


def demonstrate_size_validation():
    """Demonstrate size consistency validation."""
    print("\n" + "=" * 60)
    print("SIZE VALIDATION DEMONSTRATION")
    print("=" * 60)

    validator = HybridValidator()
    validator.config.expected_radius = 20.0
    validator.config.radius_tolerance = 0.30  # ±30%

    test_radii = [10.0, 14.0, 17.0, 20.0, 23.0, 26.0, 30.0]

    print(f"\nExpected radius: {validator.config.expected_radius:.1f} pixels")
    print(f"Tolerance: ±{validator.config.radius_tolerance * 100:.0f}%")
    print(f"Valid range: {14.0:.1f} - {26.0:.1f} pixels\n")

    print(f"{'Detected Radius':<20} {'Size Score':<15} {'Status'}")
    print("-" * 50)

    for radius in test_radii:
        score = validator._validate_size(radius)
        status = "✓ PASS" if score > 0.5 else "✗ FAIL"
        print(f"{radius:<20.1f} {score:<15.3f} {status}")


def demonstrate_combined_validation():
    """Demonstrate combined multi-criteria validation."""
    print("\n" + "=" * 60)
    print("COMBINED VALIDATION DEMONSTRATION")
    print("=" * 60)

    validator = HybridValidator()

    scenarios = [
        ("Perfect white ball", BallType.CUE, 20.0, True, True),
        ("Good ball, slight size mismatch", BallType.CUE, 23.0, True, True),
        ("Poor circularity", BallType.CUE, 20.0, False, True),
        ("Wrong color", BallType.EIGHT, 20.0, True, True),  # White image for black ball
    ]

    print("\nScenario Analysis:")

    for description, ball_type, radius, good_circle, good_color in scenarios:
        print(f"\n{description}:")

        # Create ball
        ball = Ball(
            position=(40.0, 40.0), radius=radius, ball_type=ball_type, confidence=0.9
        )

        # Create appropriate frame
        if good_circle:
            frame = create_test_ball_image(
                BallType.CUE if good_color else BallType.EIGHT, radius=int(radius)
            )
        else:
            # Non-circular shape
            frame = np.zeros((80, 80, 3), dtype=np.uint8)
            cv2.rectangle(frame, (20, 20), (60, 60), (255, 255, 255), -1)

        # Validate
        multiplier = validator.validate_ball_detection(ball, frame)

        print(f"  Ball type: {ball_type.value}")
        print(f"  Radius: {radius:.1f} pixels")
        print(f"  Original confidence: {ball.confidence:.3f}")
        print(f"  Validation multiplier: {multiplier:.3f}")
        print(f"  Final confidence: {ball.confidence * multiplier:.3f}")


def demonstrate_batch_processing():
    """Demonstrate batch validation of multiple balls."""
    print("\n" + "=" * 60)
    print("BATCH PROCESSING DEMONSTRATION")
    print("=" * 60)

    validator = HybridValidator()

    # Create multiple balls
    balls_with_rois = []

    for i, ball_type in enumerate([BallType.CUE, BallType.SOLID, BallType.EIGHT]):
        ball = Ball(
            position=(40.0, 40.0),
            radius=20.0,
            ball_type=ball_type,
            confidence=0.85 + i * 0.05,
        )
        frame = create_test_ball_image(ball_type, radius=20)
        balls_with_rois.append((ball, frame))

    # Validate batch
    multipliers = validator.validate_batch(balls_with_rois)

    print(f"\nProcessed {len(balls_with_rois)} balls:")
    for i, ((ball, _), multiplier) in enumerate(zip(balls_with_rois, multipliers)):
        print(f"\nBall {i + 1} ({ball.ball_type.value}):")
        print(f"  Original confidence: {ball.confidence:.3f}")
        print(f"  Multiplier: {multiplier:.3f}")
        print(f"  Adjusted confidence: {ball.confidence * multiplier:.3f}")


def demonstrate_statistics():
    """Demonstrate validation statistics tracking."""
    print("\n" + "=" * 60)
    print("STATISTICS TRACKING DEMONSTRATION")
    print("=" * 60)

    validator = HybridValidator()
    validator.reset_statistics()

    # Run several validations
    for i in range(10):
        ball_type = [BallType.CUE, BallType.SOLID, BallType.EIGHT][i % 3]
        ball = Ball(
            position=(40.0, 40.0), radius=20.0, ball_type=ball_type, confidence=0.9
        )
        frame = create_test_ball_image(ball_type, radius=20)
        validator.validate_ball_detection(ball, frame)

    # Get statistics
    stats = validator.get_statistics()

    print("\nValidation Statistics:")
    print(f"  Total validations: {stats['total_validations']}")
    print(f"  Passed validations: {stats['passed_validations']}")
    print(f"  Failed validations: {stats['failed_validations']}")
    print(f"  Pass rate: {stats['pass_rate'] * 100:.1f}%")
    print(f"  Fail rate: {stats['fail_rate'] * 100:.1f}%")


def demonstrate_configuration():
    """Demonstrate configuration customization."""
    print("\n" + "=" * 60)
    print("CONFIGURATION CUSTOMIZATION DEMONSTRATION")
    print("=" * 60)

    # Create validator with custom config
    config = {
        "color_histogram_enabled": True,
        "circularity_enabled": True,
        "size_validation_enabled": True,
        "expected_radius": 25.0,
        "radius_tolerance": 0.25,
        "color_weight": 0.4,
        "circularity_weight": 0.4,
        "size_weight": 0.2,
    }

    validator = HybridValidator(config)

    print("\nCustom Configuration:")
    print(f"  Expected radius: {validator.config.expected_radius} pixels")
    print(f"  Radius tolerance: ±{validator.config.radius_tolerance * 100:.0f}%")
    print("  Validation weights:")
    print(f"    Color: {validator.config.color_weight}")
    print(f"    Circularity: {validator.config.circularity_weight}")
    print(f"    Size: {validator.config.size_weight}")

    # Update configuration at runtime
    print("\nUpdating configuration at runtime...")
    validator.update_config({"expected_radius": 30.0, "radius_tolerance": 0.35})

    print(f"  Updated radius: {validator.config.expected_radius} pixels")
    print(f"  Updated tolerance: ±{validator.config.radius_tolerance * 100:.0f}%")


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 60)
    print("HYBRID VALIDATOR COMPREHENSIVE DEMONSTRATION")
    print("=" * 60)
    print("\nThis demonstration shows the complete functionality of the")
    print("HybridValidator class, which combines YOLO detections with")
    print("OpenCV verification techniques.")

    demonstrate_color_validation()
    demonstrate_circularity_validation()
    demonstrate_size_validation()
    demonstrate_combined_validation()
    demonstrate_batch_processing()
    demonstrate_statistics()
    demonstrate_configuration()

    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETE")
    print("=" * 60)
    print("\nThe HybridValidator successfully combines:")
    print("  ✓ Color histogram validation")
    print("  ✓ Circularity checks (Hough circles)")
    print("  ✓ Size consistency validation")
    print("  ✓ Multi-criteria confidence scoring")
    print("  ✓ Batch processing")
    print("  ✓ Statistics tracking")
    print("  ✓ Runtime configuration")
    print()


if __name__ == "__main__":
    main()
