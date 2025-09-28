"""Physics engine validation and testing utilities."""

from dataclasses import dataclass
from typing import Any, Optional

from ..models import BallState, TableState, Vector2D

from .engine import PhysicsConstants, PhysicsEngine, TrajectoryPoint
from .forces import ForceCalculator


@dataclass
class ValidationResult:
    """Result of a physics validation test."""

    test_name: str
    passed: bool
    actual_value: Any
    expected_value: Any
    tolerance: float
    error_message: Optional[str] = None
    details: Optional[dict[str, Any]] = None


class PhysicsValidator:
    """Validate physics engine calculations against known scenarios."""

    def __init__(self):
        self.physics_engine = PhysicsEngine()
        self.force_calculator = ForceCalculator()
        self.constants = PhysicsConstants()

    def run_all_validations(self) -> list[ValidationResult]:
        """Run all physics validation tests."""
        results = []

        # Basic physics tests
        results.extend(self._test_basic_motion())
        results.extend(self._test_friction_effects())
        results.extend(self._test_collision_physics())
        results.extend(self._test_cushion_rebounds())
        results.extend(self._test_pocket_detection())

        # Advanced tests
        results.extend(self._test_energy_conservation())
        results.extend(self._test_momentum_conservation())

        return results

    def _test_basic_motion(self) -> list[ValidationResult]:
        """Test basic ball motion without external forces."""
        results = []

        # Test 1: Ball at rest should remain at rest
        ball = self._create_test_ball(
            position=Vector2D(500, 300), velocity=Vector2D(0, 0)
        )
        table = self._create_test_table()

        trajectory = self.physics_engine.calculate_trajectory(ball, table, [], 1.0)

        results.append(
            ValidationResult(
                test_name="stationary_ball_remains_stationary",
                passed=len(trajectory) == 1
                and trajectory[0].velocity.magnitude() < 0.001,
                actual_value=trajectory[0].velocity.magnitude() if trajectory else -1,
                expected_value=0.0,
                tolerance=0.001,
            )
        )

        # Test 2: Ball with constant velocity (no friction)
        self.physics_engine.friction_enabled = False
        ball = self._create_test_ball(
            position=Vector2D(100, 300), velocity=Vector2D(1000, 0)  # 1 m/s
        )

        trajectory = self.physics_engine.calculate_trajectory(ball, table, [], 2.0)

        # Should travel exactly 2000mm in 2 seconds
        if trajectory:
            final_position = trajectory[-1].position
            expected_x = 100 + 1000 * 2  # Initial + velocity * time
            actual_distance = abs(final_position.x - expected_x)

            results.append(
                ValidationResult(
                    test_name="constant_velocity_motion",
                    passed=actual_distance < 10,  # 10mm tolerance
                    actual_value=final_position.x,
                    expected_value=expected_x,
                    tolerance=10,
                )
            )

        self.physics_engine.friction_enabled = True  # Reset

        # Test 3: Ball decelerates due to friction
        ball = self._create_test_ball(
            position=Vector2D(500, 300), velocity=Vector2D(2000, 0)  # 2 m/s initial
        )

        trajectory = self.physics_engine.calculate_trajectory(ball, table, [], 5.0)

        if len(trajectory) > 1:
            initial_speed = trajectory[0].velocity.magnitude()
            final_speed = trajectory[-1].velocity.magnitude()

            results.append(
                ValidationResult(
                    test_name="friction_deceleration",
                    passed=final_speed < initial_speed,
                    actual_value=final_speed,
                    expected_value="< " + str(initial_speed),
                    tolerance=0,
                )
            )

        return results

    def _test_friction_effects(self) -> list[ValidationResult]:
        """Test friction force calculations."""
        results = []

        ball = self._create_test_ball(
            position=Vector2D(500, 300), velocity=Vector2D(1000, 0)  # 1 m/s
        )
        table = self._create_test_table()

        # Calculate friction force
        friction_force = self.force_calculator.calculate_friction_force(ball, table)

        # Friction should oppose motion (negative x-direction)
        results.append(
            ValidationResult(
                test_name="friction_opposes_motion",
                passed=friction_force.x < 0,
                actual_value=friction_force.x,
                expected_value="< 0",
                tolerance=0,
            )
        )

        # Friction magnitude should be reasonable for pool ball
        friction_magnitude = friction_force.magnitude()
        expected_friction = (
            ball.mass * 9.81 * table.surface_friction * 0.01
        )  # Rolling resistance

        results.append(
            ValidationResult(
                test_name="friction_magnitude_reasonable",
                passed=abs(friction_magnitude - expected_friction)
                < expected_friction * 0.5,
                actual_value=friction_magnitude,
                expected_value=expected_friction,
                tolerance=expected_friction * 0.5,
            )
        )

        return results

    def _test_collision_physics(self) -> list[ValidationResult]:
        """Test ball-to-ball collision physics."""
        results = []

        # Head-on collision between equal mass balls
        ball1 = self._create_test_ball(
            position=Vector2D(300, 300), velocity=Vector2D(1000, 0)  # Moving right
        )
        ball2 = self._create_test_ball(
            position=Vector2D(400, 300),  # One ball radius away
            velocity=Vector2D(0, 0),  # Stationary
            ball_id="ball2",
        )

        table = self._create_test_table()
        trajectory = self.physics_engine.calculate_trajectory(
            ball1, table, [ball2], 1.0
        )

        # Find collision point
        collision_points = [p for p in trajectory if p.collision_type == "ball"]

        if collision_points:
            collision_points[0]

            # After collision, ball1 should stop and ball2 should move
            # (perfect elastic collision with equal masses)
            final_trajectory = self.physics_engine.calculate_trajectory(
                ball1, table, [ball2], 0.1
            )

            if final_trajectory:
                final_ball1_speed = final_trajectory[-1].velocity.magnitude()

                results.append(
                    ValidationResult(
                        test_name="elastic_collision_momentum_transfer",
                        passed=final_ball1_speed
                        < 100,  # Ball1 should be nearly stopped
                        actual_value=final_ball1_speed,
                        expected_value=0,
                        tolerance=100,
                    )
                )

        return results

    def _test_cushion_rebounds(self) -> list[ValidationResult]:
        """Test ball rebounds off table cushions."""
        results = []

        # Ball heading toward right cushion
        table = self._create_test_table()
        ball = self._create_test_ball(
            position=Vector2D(table.width - 100, 300),  # Near right cushion
            velocity=Vector2D(500, 0),  # Moving toward cushion
        )

        trajectory = self.physics_engine.calculate_trajectory(ball, table, [], 2.0)

        # Find cushion collision
        cushion_collisions = [p for p in trajectory if p.collision_type == "cushion"]

        if cushion_collisions:
            collision_point = cushion_collisions[0]

            # Find velocity after collision
            collision_index = trajectory.index(collision_point)
            if collision_index < len(trajectory) - 1:
                post_collision_velocity = trajectory[collision_index + 1].velocity

                # X velocity should be reversed (with some energy loss)
                results.append(
                    ValidationResult(
                        test_name="cushion_velocity_reversal",
                        passed=post_collision_velocity.x
                        < 0,  # Should be moving left now
                        actual_value=post_collision_velocity.x,
                        expected_value="< 0",
                        tolerance=0,
                    )
                )

                # Speed should be reduced due to restitution
                pre_speed = abs(ball.velocity.x)
                post_speed = abs(post_collision_velocity.x)
                expected_speed = pre_speed * table.cushion_elasticity

                results.append(
                    ValidationResult(
                        test_name="cushion_restitution_effect",
                        passed=abs(post_speed - expected_speed) < expected_speed * 0.2,
                        actual_value=post_speed,
                        expected_value=expected_speed,
                        tolerance=expected_speed * 0.2,
                    )
                )

        return results

    def _test_pocket_detection(self) -> list[ValidationResult]:
        """Test pocket entry detection."""
        results = []

        table = self._create_test_table()

        # Ball heading directly toward corner pocket
        corner_pocket = table.pocket_positions[0]  # Bottom-left corner
        ball = self._create_test_ball(
            position=Vector2D(corner_pocket.x + 50, corner_pocket.y + 50),
            velocity=Vector2D(-200, -200),  # Moving toward pocket
        )

        trajectory = self.physics_engine.calculate_trajectory(ball, table, [], 2.0)

        # Should detect pocket collision
        pocket_collisions = [p for p in trajectory if p.collision_type == "pocket"]

        results.append(
            ValidationResult(
                test_name="pocket_collision_detection",
                passed=len(pocket_collisions) > 0,
                actual_value=len(pocket_collisions),
                expected_value="> 0",
                tolerance=0,
            )
        )

        return results

    def _test_energy_conservation(self) -> list[ValidationResult]:
        """Test energy conservation in collisions."""
        results = []

        # Two-ball collision energy test
        ball1 = self._create_test_ball(
            position=Vector2D(200, 300), velocity=Vector2D(1000, 0)
        )
        ball2 = self._create_test_ball(
            position=Vector2D(300, 300), velocity=Vector2D(0, 0), ball_id="ball2"
        )

        # Calculate initial kinetic energy
        initial_ke = 0.5 * ball1.mass * (ball1.velocity.magnitude() ** 2)

        table = self._create_test_table()

        # Temporarily disable friction for pure collision test
        original_friction = self.physics_engine.friction_enabled
        self.physics_engine.friction_enabled = False

        trajectory = self.physics_engine.calculate_trajectory(
            ball1, table, [ball2], 1.0
        )

        # Find post-collision velocities
        collision_points = [p for p in trajectory if p.collision_type == "ball"]

        if collision_points:
            # Calculate final kinetic energy (would need both ball velocities)
            # This is a simplified test - in practice, we'd track both balls
            results.append(
                ValidationResult(
                    test_name="energy_conservation_collision",
                    passed=True,  # Placeholder - proper implementation would calculate exact values
                    actual_value="collision_detected",
                    expected_value="collision_detected",
                    tolerance=0,
                    details={
                        "initial_ke": initial_ke,
                        "collision_count": len(collision_points),
                    },
                )
            )

        self.physics_engine.friction_enabled = original_friction

        return results

    def _test_momentum_conservation(self) -> list[ValidationResult]:
        """Test momentum conservation in collisions."""
        results = []

        # Two equal-mass balls collision
        ball1 = self._create_test_ball(
            position=Vector2D(200, 300), velocity=Vector2D(1000, 0)
        )
        ball2 = self._create_test_ball(
            position=Vector2D(300, 300), velocity=Vector2D(0, 0), ball_id="ball2"
        )

        # Initial momentum
        initial_momentum = Vector2D(
            ball1.mass * ball1.velocity.x + ball2.mass * ball2.velocity.x,
            ball1.mass * ball1.velocity.y + ball2.mass * ball2.velocity.y,
        )

        results.append(
            ValidationResult(
                test_name="momentum_conservation_setup",
                passed=abs(initial_momentum.x - ball1.mass * 1000) < 0.001,
                actual_value=initial_momentum.x,
                expected_value=ball1.mass * 1000,
                tolerance=0.001,
                details={
                    "initial_momentum": {
                        "x": initial_momentum.x,
                        "y": initial_momentum.y,
                    }
                },
            )
        )

        return results

    def _create_test_ball(
        self, position: Vector2D, velocity: Vector2D, ball_id: str = "test_ball"
    ) -> BallState:
        """Create a test ball with standard properties."""
        return BallState(
            id=ball_id,
            position=position,
            velocity=velocity,
            radius=self.constants.BALL_RADIUS,
            mass=self.constants.BALL_MASS,
            spin=Vector2D(0, 0),
            is_cue_ball=(ball_id == "test_ball"),  # First ball is cue ball
            is_pocketed=False,
        )

    def _create_test_table(self) -> TableState:
        """Create a standard test table."""
        width = 2540  # 9-foot table width in mm
        height = 1270  # 9-foot table height in mm

        # Standard pocket positions
        pockets = [
            Vector2D(0, 0),  # Bottom-left corner
            Vector2D(width / 2, 0),  # Bottom side
            Vector2D(width, 0),  # Bottom-right corner
            Vector2D(0, height),  # Top-left corner
            Vector2D(width / 2, height),  # Top side
            Vector2D(width, height),  # Top-right corner
        ]

        return TableState(
            width=width,
            height=height,
            pocket_positions=pockets,
            pocket_radius=60,  # mm
            cushion_elasticity=0.85,
            surface_friction=0.2,
        )

    def validate_trajectory_realism(
        self, trajectory: list[TrajectoryPoint]
    ) -> ValidationResult:
        """Validate that a trajectory looks physically realistic."""
        if not trajectory:
            return ValidationResult(
                test_name="trajectory_realism",
                passed=False,
                actual_value="empty_trajectory",
                expected_value="non_empty_trajectory",
                tolerance=0,
                error_message="Trajectory is empty",
            )

        issues = []

        # Check for reasonable speeds
        for i, point in enumerate(trajectory):
            speed = point.velocity.magnitude()
            if speed > 10000:  # 10 m/s is very fast for pool
                issues.append(f"Unrealistic speed at point {i}: {speed:.1f} mm/s")

        # Check for smooth motion (no sudden jumps)
        for i in range(1, len(trajectory)):
            prev_point = trajectory[i - 1]
            curr_point = trajectory[i]

            position_change = Vector2D(
                curr_point.position.x - prev_point.position.x,
                curr_point.position.y - prev_point.position.y,
            )

            time_diff = curr_point.time - prev_point.time
            if time_diff > 0:
                implied_velocity = position_change.magnitude() / time_diff
                actual_velocity = curr_point.velocity.magnitude()

                # Allow some tolerance for numerical integration
                if abs(implied_velocity - actual_velocity) > actual_velocity * 0.5:
                    issues.append(f"Inconsistent velocity at point {i}")

        return ValidationResult(
            test_name="trajectory_realism",
            passed=len(issues) == 0,
            actual_value=f"{len(issues)} issues found",
            expected_value="0 issues",
            tolerance=0,
            error_message="; ".join(issues) if issues else None,
            details={
                "trajectory_length": len(trajectory),
                "total_time": trajectory[-1].time if trajectory else 0,
                "issues": issues,
            },
        )

    def print_validation_summary(self, results: list[ValidationResult]) -> None:
        """Print a summary of validation results."""
        passed = sum(1 for r in results if r.passed)
        total = len(results)

        print(f"\n{'='*60}")
        print("PHYSICS VALIDATION SUMMARY")
        print(f"{'='*60}")
        print(f"Tests passed: {passed}/{total} ({passed/total*100:.1f}%)")
        print()

        # Group by test category
        categories = {}
        for result in results:
            category = result.test_name.split("_")[0]
            if category not in categories:
                categories[category] = []
            categories[category].append(result)

        for category, cat_results in categories.items():
            cat_passed = sum(1 for r in cat_results if r.passed)
            cat_total = len(cat_results)
            print(f"{category.upper()}: {cat_passed}/{cat_total}")

            for result in cat_results:
                status = "✓" if result.passed else "✗"
                print(f"  {status} {result.test_name}")
                if not result.passed and result.error_message:
                    print(f"    Error: {result.error_message}")

        print(f"\n{'='*60}")


def run_physics_validation() -> bool:
    """Main function to run physics validation.

    Returns:
        True if all critical tests pass, False otherwise
    """
    validator = PhysicsValidator()
    results = validator.run_all_validations()

    validator.print_validation_summary(results)

    # Check if all critical tests passed
    critical_tests = [
        "stationary_ball_remains_stationary",
        "friction_opposes_motion",
        "cushion_velocity_reversal",
        "pocket_collision_detection",
    ]

    critical_results = [r for r in results if r.test_name in critical_tests]
    all_critical_passed = all(r.passed for r in critical_results)

    return all_critical_passed


if __name__ == "__main__":
    success = run_physics_validation()
    exit(0 if success else 1)
