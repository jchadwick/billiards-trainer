"""Comprehensive unit tests for GameStateManager.

Tests all requirements FR-CORE-001 through FR-CORE-015.
"""

import json
import tempfile
import threading
import time
from pathlib import Path

import pytest

from backend.core.game_state import GameStateManager, StateValidationError
from backend.core.models import BallState, GameState, GameType, TableState, Vector2D


class TestGameStateManager:
    """Test suite for GameStateManager."""

    @pytest.fixture()
    def manager(self):
        """Create a fresh GameStateManager for each test."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield GameStateManager(
                max_history_frames=100, persistence_path=Path(temp_dir)
            )

    @pytest.fixture()
    def sample_detection_data(self):
        """Sample detection data from vision module."""
        return {
            "balls": [
                {
                    "id": "cue_ball",
                    "x": 1.27,  # Center of table in meters
                    "y": 0.635,
                    "radius": 0.028575,
                    "is_cue_ball": True,
                    "confidence": 0.95,
                    "timestamp": time.time(),
                },
                {
                    "id": "ball_1",
                    "x": 1.9,
                    "y": 0.6,
                    "radius": 0.028575,
                    "number": 1,
                    "confidence": 0.90,
                    "timestamp": time.time(),
                },
                {
                    "id": "ball_8",
                    "x": 2.0,
                    "y": 0.7,
                    "radius": 0.028575,
                    "number": 8,
                    "confidence": 0.88,
                    "timestamp": time.time(),
                },
            ],
            "cue": {
                "tip_x": 0.8,
                "tip_y": 0.6,
                "angle": 45.0,
                "is_visible": True,
                "confidence": 0.85,
            },
        }

    def test_initialization(self, manager):
        """Test GameStateManager initialization (FR-CORE-001)."""
        assert manager.get_current_state() is None
        assert len(manager.get_state_history()) == 0
        assert manager.get_statistics()["current_frame"] == 0
        assert manager.get_statistics()["validation_enabled"] is True

    def test_update_state_basic(self, manager, sample_detection_data):
        """Test basic state update from detection data (FR-CORE-001)."""
        state = manager.update_state(sample_detection_data)

        assert state is not None
        assert state.frame_number == 1
        assert len(state.balls) == 3
        assert state.cue is not None
        assert state.game_type == GameType.PRACTICE

        # Verify balls were extracted correctly
        cue_ball = next((b for b in state.balls if b.is_cue_ball), None)
        assert cue_ball is not None
        assert cue_ball.position.x == 1.27
        assert cue_ball.position.y == 0.635

    def test_get_current_state(self, manager, sample_detection_data):
        """Test getting current state (FR-CORE-002)."""
        # No state initially
        assert manager.get_current_state() is None

        # Update state
        manager.update_state(sample_detection_data)
        current = manager.get_current_state()

        assert current is not None
        assert current.frame_number == 1
        assert len(current.balls) == 3

    def test_reset_game(self, manager, sample_detection_data):
        """Test game reset functionality (FR-CORE-003)."""
        # Set up some state first
        manager.update_state(sample_detection_data)
        assert manager.get_current_state().frame_number == 1

        # Reset game
        manager.reset_game(GameType.EIGHT_BALL)

        current = manager.get_current_state()
        assert current.frame_number == 0
        assert current.game_type == GameType.EIGHT_BALL
        assert current.is_break is True
        assert len(current.balls) == 0
        assert len(manager.get_state_history()) == 0

    def test_state_history(self, manager, sample_detection_data):
        """Test state history management (FR-CORE-004)."""
        # Update state multiple times
        for i in range(5):
            detection_data = sample_detection_data.copy()
            # Slightly move balls each frame
            for ball in detection_data["balls"]:
                ball["x"] += i * 0.01  # 1cm per frame
            manager.update_state(detection_data)

        history = manager.get_state_history()
        assert len(history) == 4  # Previous states (current not in history)

        # Test limited history
        history_limited = manager.get_state_history(frames=2)
        assert len(history_limited) == 2

        # Verify frame order
        assert history[0].frame_number < history[1].frame_number

    def test_event_detection(self, manager, sample_detection_data):
        """Test game event detection (FR-CORE-005)."""
        # First update
        manager.update_state(sample_detection_data)

        # Second update with ball movement
        detection_data = sample_detection_data.copy()
        detection_data["balls"][0]["x"] = 1.4  # Move cue ball significantly
        state = manager.update_state(detection_data)

        # Should detect ball motion event
        motion_events = [e for e in state.events if e.event_type == "ball_motion"]
        assert len(motion_events) > 0
        assert motion_events[0].data["ball_id"] == "cue_ball"

    def test_ball_pocketed_event(self, manager, sample_detection_data):
        """Test pocketed ball event detection."""
        # First update
        manager.update_state(sample_detection_data)

        # Second update with pocketed ball
        detection_data = sample_detection_data.copy()
        detection_data["balls"][1]["is_pocketed"] = True  # Pocket ball_1
        state = manager.update_state(detection_data)

        # Should detect pocket event
        pocket_events = [e for e in state.events if e.event_type == "ball_pocketed"]
        assert len(pocket_events) == 1
        assert pocket_events[0].data["ball_id"] == "ball_1"

    def test_state_validation_bounds(self, manager):
        """Test state validation for table bounds (FR-CORE-006)."""
        # Create detection data with ball outside bounds
        invalid_data = {
            "balls": [
                {
                    "id": "out_of_bounds",
                    "x": -0.1,  # Outside table
                    "y": 0.6,
                    "radius": 0.028575,
                    "confidence": 1.0,
                }
            ]
        }

        with manager._lock:
            manager._auto_correct_enabled = False

        with pytest.raises(StateValidationError):
            manager.update_state(invalid_data)

    def test_state_validation_overlapping_balls(self, manager):
        """Test state validation for overlapping balls (FR-CORE-007)."""
        invalid_data = {
            "balls": [
                {
                    "id": "ball_1",
                    "x": 1.0,
                    "y": 0.6,
                    "radius": 0.028575,
                    "confidence": 1.0,
                },
                {
                    "id": "ball_2",
                    "x": 1.01,  # Too close to ball_1
                    "y": 0.6,
                    "radius": 0.028575,
                    "confidence": 1.0,
                },
            ]
        }

        with manager._lock:
            manager._auto_correct_enabled = False

        with pytest.raises(StateValidationError):
            manager.update_state(invalid_data)

    def test_state_validation_cue_ball(self, manager):
        """Test state validation for cue ball requirements (FR-CORE-008)."""
        # No cue ball
        invalid_data = {
            "balls": [
                {
                    "id": "ball_1",
                    "x": 1.0,
                    "y": 0.6,
                    "radius": 0.028575,
                    "confidence": 1.0,
                }
            ]
        }

        with manager._lock:
            manager._auto_correct_enabled = False

        with pytest.raises(StateValidationError):
            manager.update_state(invalid_data)

    def test_state_validation_auto_correct(self, manager):
        """Test auto-correction mode (FR-CORE-009)."""
        invalid_data = {
            "balls": [
                {
                    "id": "out_of_bounds",
                    "x": -0.1,
                    "y": 0.6,
                    "radius": 0.028575,
                    "confidence": 1.0,
                }
            ]
        }

        # Should not raise exception in auto-correct mode
        state = manager.update_state(invalid_data)
        assert state.is_valid is False
        assert len(state.validation_errors) > 0

    def test_validation_configuration(self, manager):
        """Test validation configuration (FR-CORE-010)."""
        # Test disabling validation
        manager.set_validation_config(enabled=False)
        assert manager._validation_enabled is False

        # Invalid data should pass when validation disabled
        invalid_data = {
            "balls": [
                {
                    "id": "out_of_bounds",
                    "x": -0.1,
                    "y": 0.6,
                    "radius": 0.028575,
                    "confidence": 1.0,
                }
            ]
        }

        state = manager.update_state(invalid_data)
        # Should not validate when disabled
        assert len(state.validation_errors) == 0

    def test_event_subscription(self, manager, sample_detection_data):
        """Test event subscription and notification (FR-CORE-011)."""
        events_received = []

        def callback(event_type, data):
            events_received.append((event_type, data))

        # Subscribe to events
        sub_id = manager.subscribe_to_events("state_updated", callback)
        assert sub_id is not None

        # Trigger state update
        manager.update_state(sample_detection_data)

        # Should receive event
        assert len(events_received) == 1
        assert events_received[0][0] == "state_updated"

    def test_event_unsubscription(self, manager, sample_detection_data):
        """Test event unsubscription (FR-CORE-012)."""
        events_received = []

        def callback(event_type, data):
            events_received.append((event_type, data))

        # Subscribe and then unsubscribe
        sub_id = manager.subscribe_to_events("state_updated", callback)
        success = manager.unsubscribe(sub_id)
        assert success is True

        # Trigger state update
        manager.update_state(sample_detection_data)

        # Should not receive event
        assert len(events_received) == 0

    def test_state_persistence_save(self, manager, sample_detection_data):
        """Test state persistence - saving (FR-CORE-013)."""
        # Create some state
        manager.update_state(sample_detection_data)

        # Save state
        saved_path = manager.save_state()
        assert saved_path.exists()

        # Verify file content
        with open(saved_path, "rb") as f:
            import pickle

            data = pickle.load(f)
            assert "current_state" in data
            assert data["current_state"].frame_number == 1

    def test_state_persistence_load(self, manager, sample_detection_data):
        """Test state persistence - loading (FR-CORE-014)."""
        # Create and save state
        manager.update_state(sample_detection_data)
        saved_path = manager.save_state()
        original_frame = manager.get_current_state().frame_number

        # Reset and load
        manager.reset_game()
        assert manager.get_current_state().frame_number == 0

        manager.load_state(saved_path)
        loaded_state = manager.get_current_state()
        assert loaded_state.frame_number == original_frame

    def test_state_persistence_errors(self, manager):
        """Test state persistence error handling."""
        # Test loading non-existent file
        fake_path = Path("non_existent_file.pkl")
        with pytest.raises(FileNotFoundError):
            manager.load_state(fake_path)

        # Test loading corrupted file
        corrupt_path = manager._persistence_path / "corrupt.pkl"
        with open(corrupt_path, "w") as f:
            f.write("not pickle data")

        with pytest.raises(ValueError):
            manager.load_state(corrupt_path)

    def test_statistics(self, manager, sample_detection_data):
        """Test statistics reporting (FR-CORE-015)."""
        initial_stats = manager.get_statistics()
        assert initial_stats["current_frame"] == 0
        assert initial_stats["current_balls_count"] == 0

        # Update state and check stats
        manager.update_state(sample_detection_data)
        stats = manager.get_statistics()

        assert stats["current_frame"] == 1
        assert stats["current_balls_count"] == 3
        assert stats["active_balls_count"] == 3
        assert stats["validation_enabled"] is True
        assert "uptime_seconds" in stats

    def test_json_export(self, manager, sample_detection_data):
        """Test JSON export functionality."""
        manager.update_state(sample_detection_data)

        export_path = manager._persistence_path / "test_export.json"
        manager.export_state_json(export_path)

        assert export_path.exists()

        # Verify JSON content
        with open(export_path) as f:
            data = json.load(f)
            assert "frame_number" in data
            assert data["frame_number"] == 1
            assert len(data["balls"]) == 3

    def test_ball_accessors(self, manager, sample_detection_data):
        """Test ball accessor methods."""
        manager.update_state(sample_detection_data)

        # Test get_ball_by_id
        ball = manager.get_ball_by_id("ball_1")
        assert ball is not None
        assert ball.id == "ball_1"

        # Test get_cue_ball
        cue_ball = manager.get_cue_ball()
        assert cue_ball is not None
        assert cue_ball.is_cue_ball is True

        # Test non-existent ball
        missing = manager.get_ball_by_id("non_existent")
        assert missing is None

    def test_thread_safety(self, manager, sample_detection_data):
        """Test thread safety of concurrent operations."""
        results = []
        errors = []

        def update_worker():
            try:
                for i in range(10):
                    data = sample_detection_data.copy()
                    # Modify data slightly for each update
                    for ball in data["balls"]:
                        ball["x"] += i
                    state = manager.update_state(data)
                    results.append(state.frame_number)
            except Exception as e:
                errors.append(e)

        # Start multiple threads
        threads = [threading.Thread(target=update_worker) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should have 30 results (3 threads * 10 updates)
        assert len(results) == 30
        assert len(errors) == 0

        # Frame numbers should be unique and ordered
        assert len(set(results)) == 30
        assert max(results) == 30

        # Verify final state is valid
        final_state = manager.get_current_state()
        assert final_state is not None
        assert final_state.frame_number == 30

    def test_force_validation(self, manager, sample_detection_data):
        """Test force validation method."""
        # No current state
        is_valid, errors = manager.force_validation()
        assert is_valid is False
        assert "No current state" in errors

        # Valid state
        manager.update_state(sample_detection_data)
        is_valid, errors = manager.force_validation()
        assert is_valid is True
        assert len(errors) == 0

    def test_custom_table_configuration(self, manager):
        """Test custom table configuration."""
        # Standard 6 pocket positions for custom table
        custom_pockets = [
            Vector2D(0, 0),  # Bottom left corner
            Vector2D(1.5, 0),  # Bottom middle
            Vector2D(3.0, 0),  # Bottom right corner
            Vector2D(0, 1.5),  # Top left corner
            Vector2D(1.5, 1.5),  # Top middle
            Vector2D(3.0, 1.5),  # Top right corner
        ]

        custom_table = TableState(
            width=3.0,  # Custom dimensions in meters
            height=1.5,
            pocket_positions=custom_pockets,
            pocket_radius=0.07,
        )

        manager.reset_game(GameType.NINE_BALL, custom_table)
        state = manager.get_current_state()

        assert state.table.width == 3.0
        assert state.table.height == 1.5
        assert state.game_type == GameType.NINE_BALL

    def test_cue_state_extraction(self, manager):
        """Test cue state extraction from detection data."""
        cue_data = {
            "balls": [],
            "cue": {
                "tip_x": 0.8,
                "tip_y": 0.4,
                "angle": 30.0,
                "elevation": 5.0,
                "force": 10.5,
                "impact_x": 0.9,
                "impact_y": 0.5,
                "is_visible": True,
                "confidence": 0.92,
            },
        }

        state = manager.update_state(cue_data)
        cue = state.cue

        assert cue is not None
        assert cue.tip_position.x == 0.8
        assert cue.tip_position.y == 0.4
        assert cue.angle == 30.0
        assert cue.elevation == 5.0
        assert cue.estimated_force == 10.5
        assert cue.impact_point.x == 0.9
        assert cue.is_visible is True
        assert cue.confidence == 0.92

    def test_event_manager_integration(self, manager):
        """Test EventManager integration."""
        # Test subscriber count (may have some initial subscribers for coordination)
        initial_count = manager._event_manager.get_subscriber_count()

        def dummy_callback(event_type, data):
            pass

        # Subscribe to multiple events
        sub1 = manager.subscribe_to_events("state_updated", dummy_callback)
        manager.subscribe_to_events("game_reset", dummy_callback)

        assert manager._event_manager.get_subscriber_count() == initial_count + 2
        assert manager._event_manager.get_subscriber_count("state_updated") >= 1

        # Unsubscribe
        manager.unsubscribe(sub1)
        assert manager._event_manager.get_subscriber_count() == initial_count + 1

    @pytest.mark.parametrize(
        "game_type",
        [
            GameType.PRACTICE,
            GameType.EIGHT_BALL,
            GameType.NINE_BALL,
            GameType.STRAIGHT_POOL,
        ],
    )
    def test_different_game_types(self, manager, game_type):
        """Test different game types."""
        manager.reset_game(game_type)
        state = manager.get_current_state()
        assert state.game_type == game_type

    def test_ball_motion_threshold(self, manager, sample_detection_data):
        """Test motion detection threshold."""
        # First update
        manager.update_state(sample_detection_data)

        # Small movement (below threshold)
        small_move_data = sample_detection_data.copy()
        small_move_data["balls"][0]["x"] += 0.002  # Small movement (2mm)
        state = manager.update_state(small_move_data)

        motion_events = [e for e in state.events if e.event_type == "ball_motion"]
        assert len(motion_events) == 0  # Should not trigger

        # Large movement (above threshold)
        large_move_data = sample_detection_data.copy()
        large_move_data["balls"][0]["x"] += 0.02  # Large movement (20mm)
        state = manager.update_state(large_move_data)

        motion_events = [e for e in state.events if e.event_type == "ball_motion"]
        assert len(motion_events) > 0  # Should trigger

    def test_error_handling_malformed_data(self, manager):
        """Test error handling with malformed detection data."""
        # Missing required fields
        malformed_data = {
            "balls": [
                {
                    "id": "test_ball"
                    # Missing x, y coordinates
                }
            ]
        }

        with pytest.raises(KeyError):
            manager.update_state(malformed_data)

        # Empty data
        empty_data = {}
        state = manager.update_state(empty_data)
        assert len(state.balls) == 0

    def test_state_consistency_across_updates(self, manager, sample_detection_data):
        """Test state consistency across multiple updates."""
        # Multiple updates
        for i in range(5):
            data = sample_detection_data.copy()
            # Keep ball IDs consistent but change positions
            for _j, ball in enumerate(data["balls"]):
                ball["x"] += i * 0.01  # 1cm per frame
                ball["y"] += i * 0.005  # 0.5cm per frame

            state = manager.update_state(data)

            # Verify consistent ball count and IDs
            assert len(state.balls) == 3
            ball_ids = {ball.id for ball in state.balls}
            expected_ids = {"cue_ball", "ball_1", "ball_8"}
            assert ball_ids == expected_ids

            # Verify frame number increases
            assert state.frame_number == i + 1


class TestVector2D:
    """Test Vector2D utility class."""

    def test_vector_magnitude(self):
        """Test vector magnitude calculation."""
        v = Vector2D(3.0, 4.0)
        assert v.magnitude() == 5.0

        v_zero = Vector2D(0.0, 0.0)
        assert v_zero.magnitude() == 0.0

    def test_vector_normalize(self):
        """Test vector normalization."""
        v = Vector2D(3.0, 4.0)
        normalized = v.normalize()
        assert abs(normalized.magnitude() - 1.0) < 1e-10

        # Zero vector normalization
        v_zero = Vector2D(0.0, 0.0)
        normalized_zero = v_zero.normalize()
        assert normalized_zero.x == 0.0
        assert normalized_zero.y == 0.0


class TestDataClassIntegrity:
    """Test data class integrity and serialization."""

    def test_ball_state_creation(self):
        """Test BallState creation and properties."""
        ball = BallState(
            id="test_ball",
            position=Vector2D(100, 200),
            velocity=Vector2D(5, 10),
            radius=28.5,
        )

        assert ball.id == "test_ball"
        assert ball.position.x == 100
        assert ball.mass == 0.17  # Default value
        assert ball.is_cue_ball is False  # Default value

    def test_game_state_serialization(self):
        """Test GameState serialization for persistence."""
        from dataclasses import asdict

        # Standard 6 pocket positions
        pockets = [
            Vector2D(0, 0),
            Vector2D(1.27, 0),
            Vector2D(2.54, 0),
            Vector2D(0, 1.27),
            Vector2D(1.27, 1.27),
            Vector2D(2.54, 1.27),
        ]

        table = TableState(
            width=2.54, height=1.27, pocket_positions=pockets, pocket_radius=0.06
        )

        state = GameState(
            timestamp=time.time(),
            frame_number=1,
            balls=[],
            table=table,
            cue=None,
            game_type=GameType.PRACTICE,
            current_player=1,
            scores={1: 0, 2: 0},
            is_break=True,
            last_shot=None,
            events=[],
        )

        # Should be serializable
        state_dict = asdict(state)
        assert "frame_number" in state_dict
        assert state_dict["game_type"] == GameType.PRACTICE


if __name__ == "__main__":
    # Run with pytest
    pytest.main([__file__, "-v"])
