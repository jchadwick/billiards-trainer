"""Unit tests for the core module."""

import math

import pytest
from core.analysis.assistance import ShotAssistant
from core.analysis.prediction import ShotPredictor
from core.events.manager import EventManager
from core.game_state import GameStateManager
from core.models import Ball, Table
from core.physics.engine import PhysicsEngine
from core.utils.geometry import angle_between_points, distance, normalize_vector
from core.utils.math import clamp, lerp


@pytest.mark.unit
class TestBall:
    """Test the Ball model."""

    def test_ball_creation(self):
        """Test creating a ball."""
        ball = Ball(id="cue", x=1.0, y=0.5, radius=0.028575, color="white")

        assert ball.id == "cue"
        assert ball.x == 1.0
        assert ball.y == 0.5
        assert ball.radius == 0.028575
        assert ball.color == "white"

    def test_ball_position_property(self):
        """Test ball position property."""
        ball = Ball(id="1", x=1.0, y=0.5, radius=0.028575, color="yellow")
        assert ball.position == (1.0, 0.5)

    def test_ball_distance_to(self):
        """Test distance calculation between balls."""
        ball1 = Ball(id="1", x=0.0, y=0.0, radius=0.028575, color="yellow")
        ball2 = Ball(id="2", x=3.0, y=4.0, radius=0.028575, color="blue")

        distance = ball1.distance_to(ball2)
        assert distance == 5.0  # 3-4-5 triangle

    def test_ball_overlaps(self):
        """Test ball overlap detection."""
        ball1 = Ball(id="1", x=0.0, y=0.0, radius=0.028575, color="yellow")
        ball2 = Ball(id="2", x=0.05, y=0.0, radius=0.028575, color="blue")
        ball3 = Ball(id="3", x=1.0, y=0.0, radius=0.028575, color="red")

        assert ball1.overlaps(ball2)  # Close balls overlap
        assert not ball1.overlaps(ball3)  # Distant balls don't overlap

    def test_ball_velocity(self):
        """Test ball velocity properties."""
        ball = Ball(
            id="cue",
            x=1.0,
            y=0.5,
            radius=0.028575,
            color="white",
            velocity_x=2.0,
            velocity_y=1.0,
        )

        assert ball.velocity == (2.0, 1.0)
        assert abs(ball.speed - math.sqrt(5)) < 1e-6


@pytest.mark.unit
class TestTable:
    """Test the Table model."""

    def test_table_creation(self):
        """Test creating a table."""
        table = Table(
            width=2.84,
            height=1.42,
            corners=[(0, 0), (2.84, 0), (2.84, 1.42), (0, 1.42)],
        )

        assert table.width == 2.84
        assert table.height == 1.42
        assert len(table.corners) == 4

    def test_table_contains_point(self):
        """Test point containment in table."""
        table = Table(
            width=2.84,
            height=1.42,
            corners=[(0, 0), (2.84, 0), (2.84, 1.42), (0, 1.42)],
        )

        assert table.contains_point(1.42, 0.71)  # Center
        assert table.contains_point(0.1, 0.1)  # Inside
        assert not table.contains_point(-0.1, 0.5)  # Outside left
        assert not table.contains_point(3.0, 0.5)  # Outside right

    def test_table_nearest_rail(self):
        """Test finding nearest rail."""
        table = Table(
            width=2.84,
            height=1.42,
            corners=[(0, 0), (2.84, 0), (2.84, 1.42), (0, 1.42)],
        )

        # Point near left rail
        rail, distance = table.nearest_rail(0.1, 0.71)
        assert rail == "left"
        assert abs(distance - 0.1) < 1e-6

        # Point near bottom rail
        rail, distance = table.nearest_rail(1.42, 0.1)
        assert rail == "bottom"
        assert abs(distance - 0.1) < 1e-6


@pytest.mark.unit
class TestGameState:
    """Test the GameState model."""

    def test_game_state_creation(self, mock_game_state):
        """Test creating a game state."""
        assert len(mock_game_state.balls) == 3
        assert mock_game_state.current_player == 1
        assert mock_game_state.shot_clock == 30.0
        assert mock_game_state.game_mode == "8-ball"

    def test_get_ball_by_id(self, mock_game_state):
        """Test getting ball by ID."""
        cue_ball = mock_game_state.get_ball("cue")
        assert cue_ball is not None
        assert cue_ball.id == "cue"

        nonexistent_ball = mock_game_state.get_ball("99")
        assert nonexistent_ball is None

    def test_get_cue_ball(self, mock_game_state):
        """Test getting cue ball."""
        cue_ball = mock_game_state.get_cue_ball()
        assert cue_ball is not None
        assert cue_ball.id == "cue"

    def test_get_numbered_balls(self, mock_game_state):
        """Test getting numbered balls."""
        numbered_balls = mock_game_state.get_numbered_balls()
        numbered_ids = [ball.id for ball in numbered_balls]
        assert "1" in numbered_ids
        assert "8" in numbered_ids
        assert "cue" not in numbered_ids

    def test_balls_in_motion(self, mock_game_state):
        """Test detecting balls in motion."""
        # Initially no balls moving
        assert not mock_game_state.balls_in_motion()

        # Set cue ball velocity
        cue_ball = mock_game_state.get_cue_ball()
        cue_ball.velocity_x = 1.0
        cue_ball.velocity_y = 0.5

        assert mock_game_state.balls_in_motion()


@pytest.mark.unit
class TestShot:
    """Test the Shot model."""

    def test_shot_creation(self, mock_shot):
        """Test creating a shot."""
        assert mock_shot.cue_ball_start == (1.42, 0.71)
        assert mock_shot.cue_ball_end == (2.0, 0.9)
        assert mock_shot.target_ball == "8"
        assert mock_shot.force == 0.8
        assert mock_shot.angle == 45.0

    def test_shot_distance(self, mock_shot):
        """Test shot distance calculation."""
        distance = mock_shot.distance
        expected = math.sqrt((2.0 - 1.42) ** 2 + (0.9 - 0.71) ** 2)
        assert abs(distance - expected) < 1e-6

    def test_shot_duration(self, mock_shot):
        """Test shot duration estimation."""
        duration = mock_shot.estimated_duration
        assert duration > 0
        assert isinstance(duration, float)


@pytest.mark.unit
class TestGameStateManager:
    """Test the game state manager."""

    def test_manager_creation(self):
        """Test creating a game state manager."""
        manager = GameStateManager()
        assert manager is not None
        assert manager.current_state is None

    def test_set_game_state(self, mock_game_state):
        """Test setting game state."""
        manager = GameStateManager()
        manager.set_state(mock_game_state)

        assert manager.current_state == mock_game_state

    def test_update_ball_position(self, mock_game_state):
        """Test updating ball position."""
        manager = GameStateManager()
        manager.set_state(mock_game_state)

        manager.update_ball_position("cue", 2.0, 1.0)

        cue_ball = manager.current_state.get_ball("cue")
        assert cue_ball.x == 2.0
        assert cue_ball.y == 1.0

    def test_update_ball_velocity(self, mock_game_state):
        """Test updating ball velocity."""
        manager = GameStateManager()
        manager.set_state(mock_game_state)

        manager.update_ball_velocity("cue", 1.5, 0.5)

        cue_ball = manager.current_state.get_ball("cue")
        assert cue_ball.velocity_x == 1.5
        assert cue_ball.velocity_y == 0.5

    def test_remove_ball(self, mock_game_state):
        """Test removing a ball (pocketed)."""
        manager = GameStateManager()
        manager.set_state(mock_game_state)

        initial_count = len(manager.current_state.balls)
        manager.remove_ball("1")

        assert len(manager.current_state.balls) == initial_count - 1
        assert manager.current_state.get_ball("1") is None

    def test_switch_player(self, mock_game_state):
        """Test switching players."""
        manager = GameStateManager()
        manager.set_state(mock_game_state)

        initial_player = manager.current_state.current_player
        manager.switch_player()

        assert manager.current_state.current_player != initial_player


@pytest.mark.unit
class TestPhysicsEngine:
    """Test the physics engine."""

    def test_engine_creation(self):
        """Test creating physics engine."""
        engine = PhysicsEngine()
        assert engine is not None

    def test_ball_collision_detection(self):
        """Test ball-to-ball collision detection."""
        engine = PhysicsEngine()

        ball1 = Ball(id="1", x=0.0, y=0.0, radius=0.028575, color="yellow")
        ball2 = Ball(id="2", x=0.05, y=0.0, radius=0.028575, color="blue")

        collision = engine.detect_ball_collision(ball1, ball2)
        assert collision is not None
        assert collision.ball1 == ball1
        assert collision.ball2 == ball2

    def test_ball_collision_response(self):
        """Test ball-to-ball collision response."""
        engine = PhysicsEngine()

        # Two balls approaching each other
        ball1 = Ball(
            id="1",
            x=0.0,
            y=0.0,
            radius=0.028575,
            color="yellow",
            velocity_x=1.0,
            velocity_y=0.0,
        )
        ball2 = Ball(
            id="2",
            x=0.1,
            y=0.0,
            radius=0.028575,
            color="blue",
            velocity_x=-1.0,
            velocity_y=0.0,
        )

        # Apply collision
        engine.resolve_ball_collision(ball1, ball2)

        # Velocities should have changed (conservation of momentum)
        assert ball1.velocity_x != 1.0 or ball1.velocity_y != 0.0
        assert ball2.velocity_x != -1.0 or ball2.velocity_y != 0.0

    def test_rail_collision(self):
        """Test rail collision detection and response."""
        engine = PhysicsEngine()

        # Ball moving towards left rail
        ball = Ball(
            id="cue",
            x=0.01,
            y=0.5,
            radius=0.028575,
            color="white",
            velocity_x=-1.0,
            velocity_y=0.0,
        )

        table = Table(
            width=2.84,
            height=1.42,
            corners=[(0, 0), (2.84, 0), (2.84, 1.42), (0, 1.42)],
        )

        # Check collision
        collision = engine.detect_rail_collision(ball, table)
        assert collision is not None

        # Apply collision response
        engine.resolve_rail_collision(ball, collision)

        # Velocity X should be reversed (bounced)
        assert ball.velocity_x > 0

    def test_apply_friction(self):
        """Test applying friction to ball."""
        engine = PhysicsEngine()

        ball = Ball(
            id="cue",
            x=1.0,
            y=0.5,
            radius=0.028575,
            color="white",
            velocity_x=2.0,
            velocity_y=1.0,
        )

        initial_speed = ball.speed
        engine.apply_friction(ball, 0.15, 0.016)  # 60 FPS

        # Speed should decrease
        assert ball.speed < initial_speed

    def test_simulate_step(self, mock_game_state):
        """Test physics simulation step."""
        engine = PhysicsEngine()

        # Add velocity to cue ball
        cue_ball = mock_game_state.get_cue_ball()
        cue_ball.velocity_x = 1.0
        cue_ball.velocity_y = 0.5

        initial_x = cue_ball.x
        initial_y = cue_ball.y

        # Simulate one step
        engine.simulate_step(mock_game_state, 0.016)  # 60 FPS

        # Position should have changed
        assert cue_ball.x != initial_x or cue_ball.y != initial_y


@pytest.mark.unit
class TestShotPredictor:
    """Test the shot predictor."""

    def test_predictor_creation(self):
        """Test creating shot predictor."""
        predictor = ShotPredictor()
        assert predictor is not None

    def test_predict_shot_path(self, mock_game_state):
        """Test predicting shot path."""
        predictor = ShotPredictor()

        cue_ball = mock_game_state.get_cue_ball()
        prediction = predictor.predict_path(
            cue_ball, angle=45.0, force=0.8, game_state=mock_game_state
        )

        assert prediction is not None
        assert len(prediction.path) > 0
        assert prediction.duration > 0

    def test_predict_ball_collision(self, mock_game_state):
        """Test predicting ball-to-ball collision."""
        predictor = ShotPredictor()

        cue_ball = mock_game_state.get_cue_ball()
        target_ball = mock_game_state.get_ball("8")

        collision_prediction = predictor.predict_collision(
            cue_ball, target_ball, angle=0.0, force=0.8
        )

        if collision_prediction:
            assert collision_prediction.contact_point is not None
            assert collision_prediction.contact_time > 0

    def test_calculate_required_angle(self, mock_game_state):
        """Test calculating required shot angle."""
        predictor = ShotPredictor()

        cue_ball = mock_game_state.get_cue_ball()
        target_ball = mock_game_state.get_ball("8")

        angle = predictor.calculate_shot_angle(cue_ball, target_ball)
        assert isinstance(angle, float)
        assert -180 <= angle <= 180


@pytest.mark.unit
class TestShotAssistant:
    """Test the shot assistant."""

    def test_assistant_creation(self):
        """Test creating shot assistant."""
        assistant = ShotAssistant()
        assert assistant is not None

    def test_suggest_shots(self, mock_game_state):
        """Test suggesting possible shots."""
        assistant = ShotAssistant()

        suggestions = assistant.suggest_shots(mock_game_state)
        assert isinstance(suggestions, list)

        if suggestions:
            for suggestion in suggestions:
                assert hasattr(suggestion, "target_ball")
                assert hasattr(suggestion, "difficulty")
                assert hasattr(suggestion, "success_probability")

    def test_analyze_shot_difficulty(self, mock_game_state):
        """Test analyzing shot difficulty."""
        assistant = ShotAssistant()

        cue_ball = mock_game_state.get_cue_ball()
        target_ball = mock_game_state.get_ball("8")

        difficulty = assistant.analyze_difficulty(
            cue_ball, target_ball, mock_game_state
        )

        assert 0 <= difficulty <= 1

    def test_find_best_shot(self, mock_game_state):
        """Test finding the best shot."""
        assistant = ShotAssistant()

        best_shot = assistant.find_best_shot(mock_game_state)

        if best_shot:
            assert hasattr(best_shot, "target_ball")
            assert hasattr(best_shot, "angle")
            assert hasattr(best_shot, "force")


@pytest.mark.unit
class TestGeometryUtils:
    """Test geometry utility functions."""

    def test_distance_calculation(self):
        """Test distance calculation."""
        dist = distance(0, 0, 3, 4)
        assert dist == 5.0

        dist = distance(1, 1, 1, 1)
        assert dist == 0.0

    def test_angle_calculation(self):
        """Test angle calculation between points."""
        angle = angle_between_points(0, 0, 1, 0)
        assert abs(angle - 0) < 1e-6

        angle = angle_between_points(0, 0, 0, 1)
        assert abs(angle - 90) < 1e-6

        angle = angle_between_points(0, 0, -1, 0)
        assert abs(abs(angle) - 180) < 1e-6

    def test_vector_normalization(self):
        """Test vector normalization."""
        normalized = normalize_vector(3, 4)
        assert abs(normalized[0] - 0.6) < 1e-6
        assert abs(normalized[1] - 0.8) < 1e-6

        # Zero vector
        normalized = normalize_vector(0, 0)
        assert normalized == (0, 0)

    def test_point_in_polygon(self):
        """Test point in polygon detection."""
        from core.utils.geometry import point_in_polygon

        # Square polygon
        polygon = [(0, 0), (2, 0), (2, 2), (0, 2)]

        assert point_in_polygon(1, 1, polygon)  # Inside
        assert not point_in_polygon(3, 1, polygon)  # Outside
        assert not point_in_polygon(1, 3, polygon)  # Outside


@pytest.mark.unit
class TestMathUtils:
    """Test math utility functions."""

    def test_clamp_function(self):
        """Test value clamping."""
        assert clamp(5, 0, 10) == 5
        assert clamp(-5, 0, 10) == 0
        assert clamp(15, 0, 10) == 10

    def test_lerp_function(self):
        """Test linear interpolation."""
        assert lerp(0, 10, 0.0) == 0
        assert lerp(0, 10, 1.0) == 10
        assert lerp(0, 10, 0.5) == 5

    def test_angle_normalization(self):
        """Test angle normalization."""
        from core.utils.math import normalize_angle

        assert abs(normalize_angle(370) - 10) < 1e-6
        assert abs(normalize_angle(-10) - 350) < 1e-6
        assert abs(normalize_angle(180) - 180) < 1e-6


@pytest.mark.unit
class TestEventManager:
    """Test the event manager."""

    def test_manager_creation(self):
        """Test creating event manager."""
        manager = EventManager()
        assert manager is not None

    def test_subscribe_and_emit(self):
        """Test event subscription and emission."""
        manager = EventManager()
        callback_called = False
        event_data = None

        def test_callback(data):
            nonlocal callback_called, event_data
            callback_called = True
            event_data = data

        # Subscribe to event
        manager.subscribe("test_event", test_callback)

        # Emit event
        manager.emit("test_event", {"test": "data"})

        assert callback_called
        assert event_data == {"test": "data"}

    def test_unsubscribe(self):
        """Test event unsubscription."""
        manager = EventManager()
        callback_called = False

        def test_callback(data):
            nonlocal callback_called
            callback_called = True

        # Subscribe and then unsubscribe
        manager.subscribe("test_event", test_callback)
        manager.unsubscribe("test_event", test_callback)

        # Emit event
        manager.emit("test_event", {"test": "data"})

        assert not callback_called

    def test_multiple_subscribers(self):
        """Test multiple subscribers to same event."""
        manager = EventManager()
        call_count = 0

        def callback1(data):
            nonlocal call_count
            call_count += 1

        def callback2(data):
            nonlocal call_count
            call_count += 1

        # Subscribe multiple callbacks
        manager.subscribe("test_event", callback1)
        manager.subscribe("test_event", callback2)

        # Emit event
        manager.emit("test_event", {})

        assert call_count == 2
