"""Integration tests between config and core modules."""

import pytest
from core.game_state import GameStateManager
from core.models import BallState, Table
from core.physics.engine import PhysicsEngine


@pytest.mark.integration()
class TestConfigCoreIntegration:
    """Test integration between configuration and core modules."""

    def test_config_to_physics_engine(self, config_module):
        """Test configuration integration with physics engine."""
        physics_engine = PhysicsEngine()

        # Apply configuration to physics engine
        physics_config = config_module.get("physics")
        physics_engine.configure(
            friction=physics_config["friction"],
            restitution=physics_config["restitution"],
            gravity=physics_config["gravity"],
        )

        assert physics_engine.friction == 0.15
        assert physics_engine.restitution == 0.9
        assert physics_engine.gravity == 9.81

    def test_config_to_game_state(self, config_module):
        """Test configuration integration with game state."""
        game_manager = GameStateManager()

        # Create table from configuration
        table_config = config_module.get("table")
        table = Table(
            width=table_config["width"],
            height=table_config["height"],
            corners=[
                (0, 0),
                (table_config["width"], 0),
                (table_config["width"], table_config["height"]),
                (0, table_config["height"]),
            ],
        )

        # Create balls from configuration
        ball_config = config_module.get("balls")
        balls = []

        # Create cue ball
        cue_ball = BallState(
            id="cue",
            x=table_config["width"] / 2,
            y=table_config["height"] / 2,
            radius=ball_config["radius"],
            color="white",
        )
        balls.append(cue_ball)

        # Create numbered balls
        for i in range(1, 9):
            ball = BallState(
                id=str(i),
                x=table_config["width"] * 0.3 + i * 0.1,
                y=table_config["height"] * 0.5,
                radius=ball_config["radius"],
                color=list(ball_config["colors"].values())[i],
            )
            balls.append(ball)

        # Set up game state
        from core.models import GameState

        game_state = GameState(
            table=table,
            balls=balls,
            current_player=1,
            shot_clock=30.0,
            game_mode="8-ball",
        )

        game_manager.set_state(game_state)

        assert game_manager.current_state is not None
        assert len(game_manager.current_state.balls) == 9
        assert game_manager.current_state.table.width == 2.84

    def test_config_update_propagation(self, config_module):
        """Test that configuration updates propagate to core modules."""
        physics_engine = PhysicsEngine()

        # Initial configuration
        initial_friction = config_module.get("physics.friction")
        physics_engine.configure(friction=initial_friction)
        assert physics_engine.friction == initial_friction

        # Update configuration
        new_friction = 0.2
        config_module.set("physics.friction", new_friction)

        # Reconfigure physics engine
        updated_friction = config_module.get("physics.friction")
        physics_engine.configure(friction=updated_friction)
        assert physics_engine.friction == new_friction

    def test_ball_physics_integration(self, config_module):
        """Test ball physics with configuration values."""
        # Get configuration values
        ball_config = config_module.get("balls")
        physics_config = config_module.get("physics")

        # Create physics engine with config
        physics_engine = PhysicsEngine()
        physics_engine.configure(
            friction=physics_config["friction"],
            restitution=physics_config["restitution"],
        )

        # Create ball with config radius
        ball = BallState(
            id="test",
            x=1.0,
            y=0.5,
            radius=ball_config["radius"],
            color="white",
            velocity_x=2.0,
            velocity_y=1.0,
        )

        initial_speed = ball.speed

        # Apply friction from configuration
        physics_engine.apply_friction(ball, physics_config["friction"], 0.016)

        # Speed should have decreased according to config friction
        assert ball.speed < initial_speed

    def test_table_bounds_validation(self, config_module):
        """Test table bounds validation with configuration."""
        table_config = config_module.get("table")

        table = Table(
            width=table_config["width"],
            height=table_config["height"],
            corners=[
                (0, 0),
                (table_config["width"], 0),
                (table_config["width"], table_config["height"]),
                (0, table_config["height"]),
            ],
        )

        # Test points within bounds
        assert table.contains_point(
            table_config["width"] / 2, table_config["height"] / 2
        )
        assert table.contains_point(0.1, 0.1)

        # Test points outside bounds
        assert not table.contains_point(-0.1, 0.5)
        assert not table.contains_point(table_config["width"] + 0.1, 0.5)

    def test_config_driven_simulation(self, config_module):
        """Test physics simulation driven by configuration."""
        # Set up game state from configuration
        table_config = config_module.get("table")
        ball_config = config_module.get("balls")
        physics_config = config_module.get("physics")

        # Create table and balls
        table = Table(
            width=table_config["width"],
            height=table_config["height"],
            corners=[
                (0, 0),
                (table_config["width"], 0),
                (table_config["width"], table_config["height"]),
                (0, table_config["height"]),
            ],
        )

        cue_ball = BallState(
            id="cue",
            x=table_config["width"] / 4,
            y=table_config["height"] / 2,
            radius=ball_config["radius"],
            color="white",
            velocity_x=1.0,
            velocity_y=0.0,
        )

        target_ball = BallState(
            id="1",
            x=table_config["width"] * 3 / 4,
            y=table_config["height"] / 2,
            radius=ball_config["radius"],
            color="yellow",
        )

        from core.models import GameState

        game_state = GameState(
            table=table,
            balls=[cue_ball, target_ball],
            current_player=1,
            shot_clock=30.0,
            game_mode="8-ball",
        )

        # Run physics simulation
        physics_engine = PhysicsEngine()
        physics_engine.configure(
            friction=physics_config["friction"],
            restitution=physics_config["restitution"],
        )

        # Simulate several steps
        for _step in range(10):
            physics_engine.simulate_step(game_state, 0.016)  # 60 FPS

        # Cue ball should have moved and slowed down
        assert cue_ball.x != table_config["width"] / 4
        assert cue_ball.speed < 1.0  # Should have slowed due to friction

    def test_configuration_validation_with_core(self, config_module):
        """Test that core modules validate configuration values."""
        # Test invalid ball radius
        with pytest.raises(ValueError):
            BallState(
                id="test",
                x=1.0,
                y=0.5,
                radius=-0.1,  # Invalid negative radius
                color="white",
            )

        # Test invalid table dimensions
        with pytest.raises(ValueError):
            Table(
                width=-1.0,  # Invalid negative width
                height=1.42,
                corners=[(0, 0), (1, 0), (1, 1), (0, 1)],
            )

        # Test physics engine validation
        physics_engine = PhysicsEngine()
        with pytest.raises(ValueError):
            physics_engine.configure(
                friction=-0.1, restitution=0.9  # Invalid negative friction
            )

    def test_config_hot_reload_integration(self, config_module, temp_dir):
        """Test hot reloading configuration in running system."""
        # Set up initial system
        physics_engine = PhysicsEngine()
        initial_friction = config_module.get("physics.friction")
        physics_engine.configure(friction=initial_friction)

        # Simulate configuration file change
        new_config = config_module.data.copy()
        new_config["physics"]["friction"] = 0.25

        # Save new configuration
        config_file = temp_dir / "updated_config.yaml"
        import yaml

        with open(config_file, "w") as f:
            yaml.dump(new_config, f)

        # Reload configuration
        config_module.load_config(config_file)

        # Reconfigure physics engine
        updated_friction = config_module.get("physics.friction")
        physics_engine.configure(friction=updated_friction)

        assert physics_engine.friction == 0.25

    def test_multi_module_configuration_consistency(self, config_module):
        """Test configuration consistency across multiple core modules."""
        # Get ball radius from config
        ball_radius = config_module.get("balls.radius")

        # Create components that use ball radius
        ball = BallState(id="test", x=1.0, y=0.5, radius=ball_radius, color="white")

        physics_engine = PhysicsEngine()
        physics_engine.configure(ball_radius=ball_radius)

        # Both should use the same radius value
        assert ball.radius == ball_radius
        assert physics_engine.ball_radius == ball_radius

    def test_config_performance_impact(self, config_module, performance_timer):
        """Test performance impact of configuration access."""
        # Measure configuration access time
        performance_timer.start()

        for _i in range(1000):
            _ = config_module.get("physics.friction")
            _ = config_module.get("balls.radius")
            _ = config_module.get("table.width")

        performance_timer.stop()

        # Configuration access should be very fast
        assert performance_timer.elapsed_ms < 100  # Less than 100ms for 1000 accesses

    def test_configuration_caching(self, config_module):
        """Test configuration value caching."""
        # Access same value multiple times
        value1 = config_module.get("physics.friction")
        value2 = config_module.get("physics.friction")
        value3 = config_module.get("physics.friction")

        # Should be identical objects (cached)
        assert value1 is value2
        assert value2 is value3

        # Update configuration
        config_module.set("physics.friction", 0.3)

        # Should get new value
        value4 = config_module.get("physics.friction")
        assert value4 != value1
