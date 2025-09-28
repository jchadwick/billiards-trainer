"""Demonstration script for trajectory rendering system.

This script showcases the trajectory rendering capabilities including:
- Basic trajectory visualization
- Visual effects and animations
- Performance optimization
- Configuration management
- Real-time rendering loop

Run this script to see the trajectory rendering system in action.
"""

import logging
import math
import sys
import time

import pygame

# Add the project root to the path
sys.path.append("/Users/jchadwick/code/billiards-trainer")

from backend.core.game_state import BallState, TableState, Vector2D
from backend.core.physics.trajectory import (
    CollisionType,
    PredictedCollision,
    Trajectory,
    TrajectoryPoint,
    TrajectoryQuality,
)

from .main import DisplayMode, ProjectorModule
from .utils.performance import PerformanceMonitor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrajectoryDemo:
    """Demonstration of trajectory rendering system."""

    def __init__(self):
        """Initialize the demonstration."""
        self.running = False
        self.projector: ProjectorModule = None
        self.performance_monitor: PerformanceMonitor = None

        # Demo state
        self.demo_time = 0.0
        self.current_demo = 0
        self.demo_switch_interval = 10.0  # Switch demo every 10 seconds

        # Sample data
        self.table_state = self._create_sample_table()
        self.sample_trajectories = self._create_sample_trajectories()

    def _create_sample_table(self) -> TableState:
        """Create a sample table state."""
        # Standard pool table dimensions (9-foot table in meters)
        table_width = 2.54  # meters
        table_height = 1.27  # meters

        # Pocket positions (6 pockets)
        pocket_positions = [
            Vector2D(0, 0),  # Bottom-left corner
            Vector2D(table_width / 2, 0),  # Bottom center
            Vector2D(table_width, 0),  # Bottom-right corner
            Vector2D(0, table_height),  # Top-left corner
            Vector2D(table_width / 2, table_height),  # Top center
            Vector2D(table_width, table_height),  # Top-right corner
        ]

        return TableState(
            width=table_width,
            height=table_height,
            pocket_positions=pocket_positions,
            pocket_radius=0.06,  # 6cm pocket radius
            cushion_elasticity=0.85,
            surface_friction=0.015,
        )

    def _create_sample_trajectories(self) -> list[Trajectory]:
        """Create sample trajectories for demonstration."""
        trajectories = []

        # Trajectory 1: Simple straight shot
        traj1 = self._create_straight_trajectory()
        trajectories.append(traj1)

        # Trajectory 2: Bank shot with cushion collision
        traj2 = self._create_bank_shot_trajectory()
        trajectories.append(traj2)

        # Trajectory 3: Complex multi-collision shot
        traj3 = self._create_complex_trajectory()
        trajectories.append(traj3)

        # Trajectory 4: Spin effect demonstration
        traj4 = self._create_spin_trajectory()
        trajectories.append(traj4)

        return trajectories

    def _create_straight_trajectory(self) -> Trajectory:
        """Create a simple straight shot trajectory."""
        # Start position
        start_pos = Vector2D(0.3, 0.6)
        start_vel = Vector2D(2.0, 0.0)

        # Create trajectory points
        points = []
        time_step = 0.05
        current_time = 0.0
        current_pos = Vector2D(start_pos.x, start_pos.y)
        current_vel = Vector2D(start_vel.x, start_vel.y)

        for _i in range(50):
            # Simple physics simulation
            friction = 0.98  # Velocity decay per time step

            point = TrajectoryPoint(
                time=current_time,
                position=Vector2D(current_pos.x, current_pos.y),
                velocity=Vector2D(current_vel.x, current_vel.y),
                acceleration=Vector2D(-current_vel.x * 0.1, -current_vel.y * 0.1),
                spin=Vector2D(0, 0),
                energy=0.5 * current_vel.magnitude() ** 2,
            )
            points.append(point)

            # Update position and velocity
            current_pos.x += current_vel.x * time_step
            current_pos.y += current_vel.y * time_step
            current_vel.x *= friction
            current_vel.y *= friction
            current_time += time_step

            # Stop if velocity is very low
            if current_vel.magnitude() < 0.1:
                break

        # Create ball state
        ball_state = BallState(
            id="cue_ball",
            position=start_pos,
            velocity=start_vel,
            radius=0.028,  # Standard pool ball radius
            mass=0.17,  # Standard pool ball mass
            is_cue_ball=True,
        )

        return Trajectory(
            ball_id="cue_ball",
            initial_state=ball_state,
            points=points,
            collisions=[],
            final_position=points[-1].position if points else start_pos,
            final_velocity=points[-1].velocity if points else Vector2D(0, 0),
            time_to_rest=points[-1].time if points else 0.0,
            success_probability=0.8,
            quality=TrajectoryQuality.HIGH,
        )

    def _create_bank_shot_trajectory(self) -> Trajectory:
        """Create a bank shot with cushion collision."""
        # Create trajectory with cushion bounce
        start_pos = Vector2D(0.5, 0.2)
        start_vel = Vector2D(1.5, 1.2)

        points = []
        collisions = []
        time_step = 0.05
        current_time = 0.0
        current_pos = Vector2D(start_pos.x, start_pos.y)
        current_vel = Vector2D(start_vel.x, start_vel.y)

        for _i in range(80):
            # Check for cushion collision
            if current_pos.y >= self.table_state.height - 0.028:  # Top cushion
                if not any(c.type == CollisionType.BALL_CUSHION for c in collisions):
                    # Add collision
                    collision = PredictedCollision(
                        time=current_time,
                        position=Vector2D(
                            current_pos.x, self.table_state.height - 0.028
                        ),
                        type=CollisionType.BALL_CUSHION,
                        ball1_id="cue_ball",
                        ball2_id=None,
                        impact_angle=math.atan2(current_vel.y, current_vel.x),
                        impact_velocity=current_vel.magnitude(),
                        resulting_velocities={
                            "cue_ball": Vector2D(current_vel.x, -current_vel.y * 0.85)
                        },
                        confidence=0.9,
                        cushion_normal=Vector2D(0, -1),
                    )
                    collisions.append(collision)

                # Reflect velocity
                current_vel.y = -current_vel.y * 0.85  # Cushion elasticity
                current_pos.y = self.table_state.height - 0.028

            point = TrajectoryPoint(
                time=current_time,
                position=Vector2D(current_pos.x, current_pos.y),
                velocity=Vector2D(current_vel.x, current_vel.y),
                acceleration=Vector2D(-current_vel.x * 0.1, -current_vel.y * 0.1),
                spin=Vector2D(0, 0),
                energy=0.5 * current_vel.magnitude() ** 2,
            )
            points.append(point)

            # Update physics
            current_pos.x += current_vel.x * time_step
            current_pos.y += current_vel.y * time_step
            current_vel.x *= 0.98
            current_vel.y *= 0.98
            current_time += time_step

            if current_vel.magnitude() < 0.1:
                break

        ball_state = BallState(
            id="cue_ball",
            position=start_pos,
            velocity=start_vel,
            radius=0.028,
            mass=0.17,
            is_cue_ball=True,
        )

        return Trajectory(
            ball_id="cue_ball",
            initial_state=ball_state,
            points=points,
            collisions=collisions,
            final_position=points[-1].position if points else start_pos,
            final_velocity=points[-1].velocity if points else Vector2D(0, 0),
            time_to_rest=points[-1].time if points else 0.0,
            success_probability=0.6,
            quality=TrajectoryQuality.HIGH,
        )

    def _create_complex_trajectory(self) -> Trajectory:
        """Create a complex trajectory with multiple collisions."""
        # Similar to bank shot but with more bounces
        start_pos = Vector2D(0.2, 0.2)
        start_vel = Vector2D(2.5, 1.8)

        points = []
        collisions = []
        time_step = 0.03
        current_time = 0.0
        current_pos = Vector2D(start_pos.x, start_pos.y)
        current_vel = Vector2D(start_vel.x, start_vel.y)

        for _i in range(120):
            # Check for multiple cushion collisions
            bounced = False

            # Check boundaries
            if current_pos.x <= 0.028:  # Left cushion
                current_vel.x = abs(current_vel.x) * 0.85
                current_pos.x = 0.028
                bounced = True
            elif current_pos.x >= self.table_state.width - 0.028:  # Right cushion
                current_vel.x = -abs(current_vel.x) * 0.85
                current_pos.x = self.table_state.width - 0.028
                bounced = True

            if current_pos.y <= 0.028:  # Bottom cushion
                current_vel.y = abs(current_vel.y) * 0.85
                current_pos.y = 0.028
                bounced = True
            elif current_pos.y >= self.table_state.height - 0.028:  # Top cushion
                current_vel.y = -abs(current_vel.y) * 0.85
                current_pos.y = self.table_state.height - 0.028
                bounced = True

            if bounced and len(collisions) < 5:
                collision = PredictedCollision(
                    time=current_time,
                    position=Vector2D(current_pos.x, current_pos.y),
                    type=CollisionType.BALL_CUSHION,
                    ball1_id="cue_ball",
                    ball2_id=None,
                    impact_angle=math.atan2(current_vel.y, current_vel.x),
                    impact_velocity=current_vel.magnitude(),
                    resulting_velocities={"cue_ball": current_vel},
                    confidence=0.85,
                    cushion_normal=Vector2D(1, 0),  # Simplified
                )
                collisions.append(collision)

            point = TrajectoryPoint(
                time=current_time,
                position=Vector2D(current_pos.x, current_pos.y),
                velocity=Vector2D(current_vel.x, current_vel.y),
                acceleration=Vector2D(-current_vel.x * 0.1, -current_vel.y * 0.1),
                spin=Vector2D(0, 0),
                energy=0.5 * current_vel.magnitude() ** 2,
            )
            points.append(point)

            # Update physics
            current_pos.x += current_vel.x * time_step
            current_pos.y += current_vel.y * time_step
            current_vel.x *= 0.975
            current_vel.y *= 0.975
            current_time += time_step

            if current_vel.magnitude() < 0.05:
                break

        ball_state = BallState(
            id="cue_ball",
            position=start_pos,
            velocity=start_vel,
            radius=0.028,
            mass=0.17,
            is_cue_ball=True,
        )

        return Trajectory(
            ball_id="cue_ball",
            initial_state=ball_state,
            points=points,
            collisions=collisions,
            final_position=points[-1].position if points else start_pos,
            final_velocity=points[-1].velocity if points else Vector2D(0, 0),
            time_to_rest=points[-1].time if points else 0.0,
            success_probability=0.4,
            quality=TrajectoryQuality.HIGH,
        )

    def _create_spin_trajectory(self) -> Trajectory:
        """Create a trajectory demonstrating spin effects."""
        start_pos = Vector2D(0.4, 0.3)
        start_vel = Vector2D(1.0, 1.5)

        points = []
        time_step = 0.04
        current_time = 0.0
        current_pos = Vector2D(start_pos.x, start_pos.y)
        current_vel = Vector2D(start_vel.x, start_vel.y)
        spin = Vector2D(3.0, 2.0)  # Initial spin

        for _i in range(70):
            # Apply spin effects (simplified Magnus force)
            magnus_force = Vector2D(-spin.y * 0.01, spin.x * 0.01)
            current_vel.x += magnus_force.x
            current_vel.y += magnus_force.y

            point = TrajectoryPoint(
                time=current_time,
                position=Vector2D(current_pos.x, current_pos.y),
                velocity=Vector2D(current_vel.x, current_vel.y),
                acceleration=Vector2D(-current_vel.x * 0.08, -current_vel.y * 0.08),
                spin=Vector2D(spin.x, spin.y),
                energy=0.5 * current_vel.magnitude() ** 2,
            )
            points.append(point)

            # Update physics
            current_pos.x += current_vel.x * time_step
            current_pos.y += current_vel.y * time_step
            current_vel.x *= 0.97
            current_vel.y *= 0.97
            spin.x *= 0.95  # Spin decay
            spin.y *= 0.95
            current_time += time_step

            if current_vel.magnitude() < 0.1:
                break

        ball_state = BallState(
            id="cue_ball",
            position=start_pos,
            velocity=start_vel,
            radius=0.028,
            mass=0.17,
            spin=Vector2D(3.0, 2.0),
            is_cue_ball=True,
        )

        return Trajectory(
            ball_id="cue_ball",
            initial_state=ball_state,
            points=points,
            collisions=[],
            final_position=points[-1].position if points else start_pos,
            final_velocity=points[-1].velocity if points else Vector2D(0, 0),
            time_to_rest=points[-1].time if points else 0.0,
            success_probability=0.7,
            quality=TrajectoryQuality.HIGH,
        )

    def initialize(self) -> bool:
        """Initialize the demonstration."""
        try:
            # Create projector configuration
            config = {
                "display": {"resolution": [1280, 720], "refresh_rate": 60},
                "rendering": {"max_fps": 60, "quality": "high"},
                "visual": {
                    "trajectory_color": (0, 255, 0),
                    "trajectory_width": 3.0,
                    "trajectory_opacity": 0.8,
                },
                "effects": {"enable_trails": True, "enable_collision_effects": True},
                "calibration": {},
                "assistance": {},
            }

            # Initialize projector
            self.projector = ProjectorModule(config)

            if not self.projector.start_display(DisplayMode.WINDOW):
                logger.error("Failed to start projector display")
                return False

            # Initialize performance monitor
            self.performance_monitor = PerformanceMonitor(target_fps=60.0)

            logger.info("Trajectory rendering demonstration initialized")
            return True

        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            return False

    def run_demo(self) -> None:
        """Run the trajectory rendering demonstration."""
        if not self.initialize():
            return

        logger.info("Starting trajectory rendering demonstration")
        logger.info("Press SPACE to switch demos, ESC to exit")

        self.running = True
        clock = pygame.time.Clock()
        last_demo_switch = time.time()

        demo_names = [
            "Straight Shot",
            "Bank Shot",
            "Complex Multi-Collision",
            "Spin Effects",
            "Performance Test",
        ]

        try:
            while self.running:
                self.performance_monitor.begin_frame()

                # Handle events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            self.running = False
                        elif event.key == pygame.K_SPACE:
                            self._next_demo()
                            last_demo_switch = time.time()

                # Auto-switch demos
                current_time = time.time()
                if current_time - last_demo_switch > self.demo_switch_interval:
                    self._next_demo()
                    last_demo_switch = current_time

                # Update demo
                self.demo_time += 1.0 / 60.0
                self._update_current_demo()

                # Render frame
                self.projector.render_frame()

                # Update performance metrics
                self.performance_monitor.end_frame()

                # Print performance info every 2 seconds
                if int(self.demo_time) % 2 == 0 and self.demo_time % 1.0 < 1.0 / 60.0:
                    fps = self.performance_monitor.metrics.current_fps
                    demo_name = demo_names[self.current_demo % len(demo_names)]
                    print(
                        f"Demo: {demo_name} | FPS: {fps:.1f} | "
                        f"Trajectories: {self.projector.get_trajectory_count()}"
                    )

                # Target frame rate
                clock.tick(60)

        except KeyboardInterrupt:
            logger.info("Demo interrupted by user")
        except Exception as e:
            logger.error(f"Demo error: {e}")
        finally:
            self._cleanup()

    def _next_demo(self) -> None:
        """Switch to the next demonstration."""
        self.current_demo = (self.current_demo + 1) % 5
        self.projector.clear_display()
        logger.info(f"Switching to demo {self.current_demo + 1}")

    def _update_current_demo(self) -> None:
        """Update the current demonstration."""
        if self.current_demo == 0:
            self._demo_straight_shot()
        elif self.current_demo == 1:
            self._demo_bank_shot()
        elif self.current_demo == 2:
            self._demo_complex_shot()
        elif self.current_demo == 3:
            self._demo_spin_effects()
        elif self.current_demo == 4:
            self._demo_performance_test()

    def _demo_straight_shot(self) -> None:
        """Demonstrate straight shot trajectory."""
        if self.projector.get_trajectory_count() == 0:
            trajectory = self.sample_trajectories[0]
            self.projector.render_trajectory(trajectory, fade_in=True)

    def _demo_bank_shot(self) -> None:
        """Demonstrate bank shot with collision effects."""
        if self.projector.get_trajectory_count() == 0:
            trajectory = self.sample_trajectories[1]
            self.projector.render_trajectory(trajectory, fade_in=True)

            # Add collision effects
            for collision in trajectory.collisions:
                self.projector.render_collision_prediction(collision, intensity=1.5)

    def _demo_complex_shot(self) -> None:
        """Demonstrate complex multi-collision shot."""
        if self.projector.get_trajectory_count() == 0:
            trajectory = self.sample_trajectories[2]
            self.projector.render_trajectory(trajectory, fade_in=True)

            # Show success probability
            if trajectory.final_position:
                from .rendering.renderer import Point2D

                final_pos = Point2D(
                    trajectory.final_position.x, trajectory.final_position.y
                )
                self.projector.render_success_indicator(
                    final_pos, trajectory.success_probability
                )

    def _demo_spin_effects(self) -> None:
        """Demonstrate spin effects visualization."""
        if self.projector.get_trajectory_count() == 0:
            trajectory = self.sample_trajectories[3]
            self.projector.render_trajectory(trajectory, fade_in=True)

            # Update ball state for spin visualization
            ball_state = trajectory.initial_state
            self.projector.update_ball_state(ball_state)

    def _demo_performance_test(self) -> None:
        """Demonstrate performance with multiple trajectories."""
        # Render multiple trajectories to test performance
        max_trajectories = 4
        if self.projector.get_trajectory_count() < max_trajectories:
            for i, trajectory in enumerate(self.sample_trajectories):
                if i < max_trajectories:
                    # Offset trajectories slightly for visibility
                    offset_trajectory = self._offset_trajectory(trajectory, i * 0.1)
                    offset_trajectory.ball_id = f"ball_{i}"
                    self.projector.render_trajectory(offset_trajectory, fade_in=True)

    def _offset_trajectory(self, trajectory: Trajectory, offset: float) -> Trajectory:
        """Create an offset copy of a trajectory."""
        import copy

        offset_traj = copy.deepcopy(trajectory)

        # Offset all points
        for point in offset_traj.points:
            point.position.y += offset

        # Offset initial and final positions
        offset_traj.initial_state.position.y += offset
        if offset_traj.final_position:
            offset_traj.final_position.y += offset

        return offset_traj

    def _cleanup(self) -> None:
        """Clean up resources."""
        if self.projector:
            self.projector.stop_display()

        if self.performance_monitor:
            report = self.performance_monitor.get_performance_report()
            logger.info(f"Final performance report: {report}")

        pygame.quit()
        logger.info("Demonstration cleanup complete")


def main():
    """Main entry point for the demonstration."""
    print("Billiards Trajectory Rendering Demonstration")
    print("=" * 50)
    print("Controls:")
    print("  SPACE - Switch demonstration")
    print("  ESC   - Exit")
    print("=" * 50)

    demo = TrajectoryDemo()
    demo.run_demo()


if __name__ == "__main__":
    main()
