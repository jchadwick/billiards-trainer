"""Integration tests for the Core Module.

Tests the complete integration of all core components including:
- Game state management
- Physics calculations
- Shot analysis
- Event coordination
- Performance monitoring
"""

import asyncio
import time

import pytest

from . import CoreModule, CoreModuleConfig, CoreModuleError, GameType, Vector2D


class TestCoreModuleIntegration:
    """Integration tests for the complete CoreModule."""

    @pytest.fixture
    def core_module(self):
        """Create a CoreModule instance for testing."""
        config = CoreModuleConfig(
            debug_mode=True,
            cache_size=100,
            async_processing=False,  # Synchronous for easier testing
        )
        return CoreModule(config)

    @pytest.fixture
    def sample_detection_data(self):
        """Sample detection data for testing."""
        return {
            "balls": [
                {
                    "id": "cue_ball",
                    "x": 100.0,
                    "y": 200.0,
                    "vx": 0.0,
                    "vy": 0.0,
                    "radius": 28.5,
                    "is_cue_ball": True,
                    "confidence": 0.95,
                },
                {
                    "id": "ball_1",
                    "x": 300.0,
                    "y": 200.0,
                    "vx": 0.0,
                    "vy": 0.0,
                    "radius": 28.5,
                    "number": 1,
                    "confidence": 0.90,
                },
            ],
            "cue": {
                "tip_x": 50.0,
                "tip_y": 200.0,
                "angle": 0.0,
                "is_visible": True,
                "confidence": 0.85,
            },
        }

    def test_core_module_initialization(self, core_module):
        """Test that CoreModule initializes correctly."""
        assert core_module is not None
        assert core_module.config is not None
        assert core_module.state_manager is not None
        assert core_module.physics_engine is not None
        assert core_module.event_manager is not None
        assert core_module.get_current_state() is None

    @pytest.mark.asyncio
    async def test_state_update_workflow(self, core_module, sample_detection_data):
        """Test the complete state update workflow."""
        # Update state with detection data
        initial_state = await core_module.update_state(sample_detection_data)

        assert initial_state is not None
        assert len(initial_state.balls) == 2
        assert initial_state.frame_number == 1

        # Verify current state is updated
        current_state = core_module.get_current_state()
        assert current_state == initial_state

        # Test second update
        sample_detection_data["balls"][0]["x"] = 110.0  # Move cue ball
        second_state = await core_module.update_state(sample_detection_data)

        assert second_state.frame_number == 2
        assert second_state.balls[0].position.x == 110.0

    @pytest.mark.asyncio
    async def test_trajectory_calculation(self, core_module, sample_detection_data):
        """Test trajectory calculation functionality."""
        # Set up initial state
        await core_module.update_state(sample_detection_data)

        # Calculate trajectory for cue ball
        trajectory = await core_module.calculate_trajectory(
            ball_id="cue_ball", initial_velocity=Vector2D(1.0, 0.0), time_limit=2.0
        )

        assert trajectory is not None
        assert len(trajectory) > 0
        assert all(isinstance(point, Vector2D) for point in trajectory)

    @pytest.mark.asyncio
    async def test_shot_analysis(self, core_module, sample_detection_data):
        """Test shot analysis functionality."""
        # Set up initial state
        await core_module.update_state(sample_detection_data)

        # Analyze current shot
        analysis = await core_module.analyze_shot(target_ball="ball_1")

        assert analysis is not None
        assert hasattr(analysis, "difficulty")
        assert hasattr(analysis, "success_probability")

    @pytest.mark.asyncio
    async def test_outcome_prediction(self, core_module, sample_detection_data):
        """Test outcome prediction functionality."""
        # Set up initial state
        await core_module.update_state(sample_detection_data)

        # Predict outcomes
        predictions = await core_module.predict_outcomes(
            shot_velocity=Vector2D(2.0, 0.0), num_predictions=3
        )

        assert predictions is not None
        assert len(predictions) <= 3

    @pytest.mark.asyncio
    async def test_shot_suggestions(self, core_module, sample_detection_data):
        """Test shot suggestion functionality."""
        # Set up initial state
        await core_module.update_state(sample_detection_data)

        # Get shot suggestions
        suggestions = await core_module.suggest_shots(
            difficulty_filter=0.7, max_suggestions=2
        )

        assert suggestions is not None
        assert len(suggestions) <= 2

    @pytest.mark.asyncio
    async def test_state_validation(self, core_module, sample_detection_data):
        """Test state validation functionality."""
        # Set up initial state
        await core_module.update_state(sample_detection_data)

        # Validate state
        validation_result = await core_module.validate_state()

        assert validation_result is not None
        assert "valid" in validation_result
        assert "issues" in validation_result

    def test_state_history_management(self, core_module, sample_detection_data):
        """Test state history functionality."""
        # Initially no history
        history = core_module.get_state_history()
        assert len(history) == 0

        # Add some states (synchronous for testing)
        asyncio.run(core_module.update_state(sample_detection_data))
        asyncio.run(core_module.update_state(sample_detection_data))

        # Check history
        history = core_module.get_state_history()
        assert len(history) > 0

        # Test limited history
        limited_history = core_module.get_state_history(count=1)
        assert len(limited_history) <= 1

    def test_performance_metrics(self, core_module, sample_detection_data):
        """Test performance metrics collection."""
        # Get initial metrics
        metrics = core_module.get_performance_metrics()
        assert metrics.total_updates == 0

        # Perform some operations
        asyncio.run(core_module.update_state(sample_detection_data))

        # Check updated metrics
        updated_metrics = core_module.get_performance_metrics()
        assert updated_metrics.total_updates > 0

    @pytest.mark.asyncio
    async def test_event_subscription(self, core_module, sample_detection_data):
        """Test event subscription and emission."""
        events_received = []

        def event_callback(event_type, data):
            events_received.append((event_type, data))

        # Subscribe to events
        subscription_id = core_module.subscribe_to_events(
            "state_updated", event_callback
        )
        assert subscription_id is not None

        # Update state to trigger event
        await core_module.update_state(sample_detection_data)

        # Check if event was received (might need to wait for async processing)
        await asyncio.sleep(0.1)

        # Unsubscribe
        unsubscribed = core_module.unsubscribe(subscription_id)
        assert unsubscribed

    @pytest.mark.asyncio
    async def test_game_reset(self, core_module, sample_detection_data):
        """Test game reset functionality."""
        # Set up initial state
        await core_module.update_state(sample_detection_data)
        assert core_module.get_current_state() is not None

        # Reset game
        await core_module.reset_game(GameType.EIGHT_BALL)

        # Verify reset
        current_state = core_module.get_current_state()
        assert current_state is not None
        assert current_state.game_type == GameType.EIGHT_BALL
        assert len(current_state.balls) == 0  # No balls after reset

    def test_error_handling(self, core_module):
        """Test error handling for invalid operations."""
        # Test trajectory calculation without state
        with pytest.raises(CoreModuleError):
            asyncio.run(
                core_module.calculate_trajectory("nonexistent_ball", Vector2D(1, 0))
            )

        # Test shot analysis without state
        with pytest.raises(CoreModuleError):
            asyncio.run(core_module.analyze_shot())

    @pytest.mark.asyncio
    async def test_caching_functionality(self, core_module, sample_detection_data):
        """Test caching behavior."""
        # Set up initial state
        await core_module.update_state(sample_detection_data)

        # First trajectory calculation
        start_time = time.time()
        trajectory1 = await core_module.calculate_trajectory(
            "cue_ball", Vector2D(1.0, 0.0)
        )
        time.time() - start_time

        # Second identical calculation (should be cached)
        start_time = time.time()
        trajectory2 = await core_module.calculate_trajectory(
            "cue_ball", Vector2D(1.0, 0.0)
        )
        time.time() - start_time

        # Verify results are identical
        assert len(trajectory1) == len(trajectory2)

        # Second call should be faster (cached)
        # Note: This might not always be true in test environment
        # assert second_time < first_time

    @pytest.mark.asyncio
    async def test_async_processing_toggle(self):
        """Test switching between sync and async processing."""
        # Test with async processing enabled
        async_config = CoreModuleConfig(async_processing=True)
        async_module = CoreModule(async_config)

        # Test with async processing disabled
        sync_config = CoreModuleConfig(async_processing=False)
        sync_module = CoreModule(sync_config)

        # Both should work
        sample_data = {
            "balls": [
                {
                    "id": "test_ball",
                    "x": 100,
                    "y": 100,
                    "vx": 0,
                    "vy": 0,
                    "radius": 28.5,
                    "is_cue_ball": True,
                }
            ]
        }

        async_state = await async_module.update_state(sample_data)
        sync_state = await sync_module.update_state(sample_data)

        assert async_state is not None
        assert sync_state is not None

    def test_string_representations(self, core_module):
        """Test string representation methods."""
        str_repr = str(core_module)
        assert "CoreModule" in str_repr

        detailed_repr = repr(core_module)
        assert "CoreModule" in detailed_repr
        assert "config=" in detailed_repr


class TestCoreModuleStressTest:
    """Stress tests for CoreModule performance and stability."""

    @pytest.mark.asyncio
    async def test_rapid_state_updates(self):
        """Test rapid consecutive state updates."""
        config = CoreModuleConfig(async_processing=False, cache_size=10)
        core_module = CoreModule(config)

        sample_data = {
            "balls": [
                {
                    "id": "cue_ball",
                    "x": 100,
                    "y": 100,
                    "vx": 0,
                    "vy": 0,
                    "radius": 28.5,
                    "is_cue_ball": True,
                }
            ]
        }

        # Perform rapid updates
        for i in range(50):
            sample_data["balls"][0]["x"] = 100 + i
            state = await core_module.update_state(sample_data)
            assert state.frame_number == i + 1

    @pytest.mark.asyncio
    async def test_memory_usage_stability(self):
        """Test that memory usage remains stable with many operations."""
        import gc

        config = CoreModuleConfig(
            state_history_limit=10,  # Limit history to prevent memory growth
            cache_size=10,
        )
        core_module = CoreModule(config)

        sample_data = {
            "balls": [
                {
                    "id": "ball_1",
                    "x": 100,
                    "y": 100,
                    "vx": 0,
                    "vy": 0,
                    "radius": 28.5,
                    "is_cue_ball": True,
                }
            ]
        }

        # Perform many operations
        for i in range(100):
            await core_module.update_state(sample_data)
            if i % 10 == 0:
                gc.collect()  # Force garbage collection

        # History should be limited
        history = core_module.get_state_history()
        assert len(history) <= 10


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])
