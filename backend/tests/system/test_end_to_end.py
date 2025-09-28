"""System tests for end-to-end validation."""

import time
from unittest.mock import AsyncMock, MagicMock, patch

import cv2
import numpy as np
import pytest


@pytest.mark.system
class TestCompleteWorkflow:
    """Test complete system workflow from detection to projection."""

    @pytest.mark.asyncio
    async def test_full_detection_to_projection_pipeline(
        self, config_module, mock_cv2_camera
    ):
        """Test complete pipeline from camera to projector."""
        # Mock all hardware dependencies
        with (
            patch("cv2.VideoCapture", return_value=mock_cv2_camera),
            patch("moderngl.create_context") as mock_gl_context,
        ):
            # Set up mocked OpenGL context
            mock_context = MagicMock()
            mock_gl_context.return_value = mock_context

            # Import modules after mocking
            from core.game_state import GameStateManager
            from core.physics.engine import PhysicsEngine
            from projector.rendering.opengl.renderer import OpenGLRenderer
            from vision.detection.balls import BallDetector
            from vision.tracking.tracker import BallTracker

            # Initialize components
            detector = BallDetector()
            tracker = BallTracker()
            game_manager = GameStateManager()
            physics_engine = PhysicsEngine()
            renderer = OpenGLRenderer()

            # Set up initial game state
            from core.models import Ball, GameState, Table

            table = Table(
                width=config_module.get("table.width"),
                height=config_module.get("table.height"),
                corners=[(0, 0), (2.84, 0), (2.84, 1.42), (0, 1.42)],
            )

            cue_ball = Ball(
                id="cue",
                x=1.42,
                y=0.71,
                radius=config_module.get("balls.radius"),
                color="white",
            )

            game_state = GameState(
                table=table,
                balls=[cue_ball],
                current_player=1,
                shot_clock=30.0,
                game_mode="8-ball",
            )

            game_manager.set_state(game_state)

            # Simulate complete workflow
            for frame_id in range(10):
                # 1. Camera capture (mocked)
                ret, frame = mock_cv2_camera.read()
                assert ret

                # 2. Ball detection
                detections = detector.detect(frame)

                # 3. Ball tracking
                timestamp = time.time() + frame_id * 0.033
                for detection in detections:
                    detection_data = {
                        "id": detection.get("id", "cue"),
                        "x": detection.get("x", 960),
                        "y": detection.get("y", 540),
                        "confidence": 0.9,
                        "timestamp": timestamp,
                    }

                    if frame_id == 0:
                        tracker.add_detection(detection_data)
                    else:
                        tracker.update(detection_data)

                # 4. Update game state
                for track_id, track_data in tracker.tracks.items():
                    real_x = track_data["x"] / 1920 * 2.84
                    real_y = track_data["y"] / 1080 * 1.42
                    game_manager.update_ball_position(track_id, real_x, real_y)

                # 5. Physics simulation (if balls moving)
                if any(ball.speed > 0.01 for ball in game_manager.current_state.balls):
                    physics_engine.simulate_step(game_manager.current_state, 0.033)

                # 6. Projection rendering
                renderer.clear()

                # Create render objects for balls
                for ball in game_manager.current_state.balls:
                    # Convert to screen coordinates
                    screen_x = ball.x / 2.84 * 1920
                    screen_y = ball.y / 1.42 * 1080

                    from projector.models import RenderObject

                    ball_obj = RenderObject(
                        object_type="circle",
                        position=(screen_x, screen_y),
                        size=(30, 30),
                        color=(255, 255, 255) if ball.id == "cue" else (255, 255, 0),
                    )

                    renderer.render_object(ball_obj)

                renderer.present()

            # Verify pipeline completed successfully
            assert game_manager.current_state is not None
            assert len(tracker.tracks) > 0

    def test_api_integration_workflow(self, test_client, config_module):
        """Test API integration workflow."""
        # 1. Check system health
        response = test_client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

        # 2. Get configuration
        response = test_client.get("/api/v1/config")
        assert response.status_code == 200
        config_data = response.json()
        assert "camera" in config_data
        assert "table" in config_data

        # 3. Update configuration
        camera_update = {"camera": {"device_id": 1, "fps": 60}}
        response = test_client.put("/api/v1/config", json=camera_update)
        assert response.status_code == 200

        # 4. Initialize game
        game_init_data = {"game_mode": "8-ball", "player_count": 2}
        response = test_client.post("/api/v1/game/init", json=game_init_data)
        assert response.status_code in [200, 201]

        # 5. Get initial game state
        response = test_client.get("/api/v1/game/state")
        assert response.status_code == 200
        response.json()

        # 6. Get shot suggestions
        response = test_client.get("/api/v1/shots/suggestions")
        assert response.status_code == 200
        suggestions = response.json()
        assert isinstance(suggestions, list)

        # 7. Predict shot
        shot_data = {"angle": 45.0, "force": 0.8, "english": [0, 0]}
        response = test_client.post("/api/v1/shots/predict", json=shot_data)
        assert response.status_code == 200
        prediction = response.json()
        assert "path" in prediction

        # 8. Execute shot
        response = test_client.post("/api/v1/shots/execute", json=shot_data)
        assert response.status_code == 200

        # 9. Get updated game state
        response = test_client.get("/api/v1/game/state")
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_websocket_integration_workflow(self, test_client):
        """Test WebSocket integration workflow."""
        from api.websocket.handler import WebSocketHandler

        handler = WebSocketHandler()

        # Mock WebSocket connection
        mock_websocket = AsyncMock()
        mock_websocket.accept = AsyncMock()
        mock_websocket.send_json = AsyncMock()
        mock_websocket.receive_json = AsyncMock()
        mock_websocket.close = AsyncMock()

        # Test connection workflow
        await handler.connect(mock_websocket)
        mock_websocket.accept.assert_called_once()

        # Test subscription workflow
        from api.websocket.subscriptions import SubscriptionManager

        subscription_manager = SubscriptionManager()

        await subscription_manager.subscribe(mock_websocket, "game_state_updates")
        await subscription_manager.subscribe(mock_websocket, "ball_tracking_updates")

        # Test real-time updates
        game_state_update = {
            "type": "game_state_update",
            "data": {
                "balls": [
                    {
                        "id": "cue",
                        "x": 1.42,
                        "y": 0.71,
                        "velocity_x": 0,
                        "velocity_y": 0,
                    }
                ],
                "current_player": 1,
                "shot_clock": 30.0,
            },
            "timestamp": time.time(),
        }

        await subscription_manager.publish("game_state_updates", game_state_update)
        mock_websocket.send_json.assert_called_with(game_state_update)

        # Test ball tracking updates
        ball_update = {
            "type": "ball_tracking_update",
            "data": {"ball_id": "cue", "x": 1.45, "y": 0.72, "confidence": 0.95},
            "timestamp": time.time(),
        }

        await subscription_manager.publish("ball_tracking_updates", ball_update)

        # Test disconnection
        await handler.disconnect(mock_websocket)

    def test_configuration_workflow(self, test_client, temp_dir):
        """Test configuration management workflow."""
        # 1. Get default configuration
        response = test_client.get("/api/v1/config")
        assert response.status_code == 200
        response.json()

        # 2. Update camera settings
        camera_update = {
            "camera": {"device_id": 1, "width": 1280, "height": 720, "fps": 60}
        }
        response = test_client.put(
            "/api/v1/config/camera", json=camera_update["camera"]
        )
        assert response.status_code == 200

        # 3. Verify camera update
        response = test_client.get("/api/v1/config/camera")
        assert response.status_code == 200
        camera_config = response.json()
        assert camera_config["device_id"] == 1
        assert camera_config["fps"] == 60

        # 4. Update physics settings
        physics_update = {
            "physics": {"friction": 0.2, "restitution": 0.95, "gravity": 9.81}
        }
        response = test_client.put(
            "/api/v1/config/physics", json=physics_update["physics"]
        )
        assert response.status_code == 200

        # 5. Save configuration profile
        profile_data = {
            "name": "test_profile",
            "description": "Test configuration profile",
        }
        response = test_client.post("/api/v1/config/profiles", json=profile_data)
        assert response.status_code in [200, 201]

        # 6. List configuration profiles
        response = test_client.get("/api/v1/config/profiles")
        assert response.status_code == 200
        profiles = response.json()
        assert isinstance(profiles, list)

        # 7. Load configuration profile
        if profiles:
            profile_id = profiles[0]["id"]
            response = test_client.post(f"/api/v1/config/profiles/{profile_id}/load")
            assert response.status_code == 200

        # 8. Reset to defaults
        response = test_client.post("/api/v1/config/reset")
        assert response.status_code == 200

    def test_error_recovery_workflow(self, test_client):
        """Test error recovery and handling workflow."""
        # 1. Test invalid configuration
        invalid_config = {
            "camera": {
                "device_id": -1,  # Invalid device ID
                "width": -100,  # Invalid width
                "fps": 0,  # Invalid FPS
            }
        }
        response = test_client.put(
            "/api/v1/config/camera", json=invalid_config["camera"]
        )
        assert response.status_code == 422  # Validation error

        # 2. Test nonexistent endpoints
        response = test_client.get("/api/v1/nonexistent")
        assert response.status_code == 404

        # 3. Test invalid shot parameters
        invalid_shot = {
            "angle": 500,  # Invalid angle
            "force": -1,  # Invalid force
            "english": [10, 10],  # Invalid english
        }
        response = test_client.post("/api/v1/shots/predict", json=invalid_shot)
        assert response.status_code == 422

        # 4. Test system recovery after errors
        response = test_client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

        # 5. Test valid request after errors
        valid_config = {
            "camera": {"device_id": 0, "width": 1920, "height": 1080, "fps": 30}
        }
        response = test_client.put("/api/v1/config/camera", json=valid_config["camera"])
        assert response.status_code == 200

    def test_performance_under_load(self, test_client, performance_timer):
        """Test system performance under load."""
        # Test concurrent API requests
        import concurrent.futures

        def make_request():
            response = test_client.get("/health")
            return response.status_code == 200

        # Simulate 50 concurrent requests
        num_requests = 50

        performance_timer.start()

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request) for _ in range(num_requests)]
            results = [
                future.result() for future in concurrent.futures.as_completed(futures)
            ]

        performance_timer.stop()

        # All requests should succeed
        assert all(results)

        # Should handle load efficiently
        avg_response_time = performance_timer.elapsed_ms / num_requests
        assert (
            avg_response_time < 100
        ), f"Average response time {avg_response_time:.2f}ms too slow"

    def test_data_persistence_workflow(self, test_client, temp_dir):
        """Test data persistence workflow."""
        # 1. Create game session
        session_data = {
            "session_name": "test_session",
            "game_mode": "8-ball",
            "players": ["Player 1", "Player 2"],
        }
        response = test_client.post("/api/v1/sessions", json=session_data)
        assert response.status_code in [200, 201]
        session = response.json()
        session_id = session.get("id")

        # 2. Record game events
        shot_event = {
            "type": "shot_taken",
            "player": "Player 1",
            "shot_data": {"angle": 45.0, "force": 0.8, "target_ball": "1"},
            "timestamp": time.time(),
        }
        response = test_client.post(
            f"/api/v1/sessions/{session_id}/events", json=shot_event
        )
        assert response.status_code in [200, 201]

        # 3. Get session history
        response = test_client.get(f"/api/v1/sessions/{session_id}")
        assert response.status_code == 200
        session_data = response.json()
        assert "events" in session_data

        # 4. Export session data
        response = test_client.get(f"/api/v1/sessions/{session_id}/export")
        assert response.status_code == 200

        # 5. List all sessions
        response = test_client.get("/api/v1/sessions")
        assert response.status_code == 200
        sessions = response.json()
        assert isinstance(sessions, list)

    @pytest.mark.slow
    def test_long_running_stability(self, config_module):
        """Test system stability during long-running operation."""
        from core.game_state import GameStateManager
        from vision.detection.balls import BallDetector
        from vision.tracking.tracker import BallTracker

        detector = BallDetector()
        tracker = BallTracker()
        game_manager = GameStateManager()

        # Set up game state
        from core.models import Ball, GameState, Table

        table = Table(
            width=2.84,
            height=1.42,
            corners=[(0, 0), (2.84, 0), (2.84, 1.42), (0, 1.42)],
        )
        cue_ball = Ball(id="cue", x=1.42, y=0.71, radius=0.028575, color="white")
        game_state = GameState(
            table=table,
            balls=[cue_ball],
            current_player=1,
            shot_clock=30.0,
            game_mode="8-ball",
        )
        game_manager.set_state(game_state)

        # Run for extended period (simulate 5 minutes)
        num_frames = 30 * 60 * 5  # 30 FPS * 60 sec * 5 min

        start_time = time.time()
        error_count = 0

        for frame_id in range(0, num_frames, 100):  # Sample every 100 frames
            try:
                # Create synthetic frame
                import cv2
                import numpy as np

                frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
                frame[:, :] = [34, 139, 34]
                cv2.circle(frame, (960, 540), 25, (255, 255, 255), -1)

                # Process frame
                detections = detector.detect(frame)

                # Update tracking
                timestamp = time.time()
                for _detection in detections:
                    detection_data = {
                        "id": "cue",
                        "x": 960,
                        "y": 540,
                        "confidence": 0.9,
                        "timestamp": timestamp,
                    }

                    if frame_id == 0:
                        tracker.add_detection(detection_data)
                    else:
                        tracker.update(detection_data)

                # Periodic cleanup
                if frame_id % 1000 == 0:
                    tracker.cleanup_stale_tracks()

            except Exception as e:
                error_count += 1
                if error_count > 10:  # Too many errors
                    pytest.fail(f"Too many errors during long-running test: {e}")

        end_time = time.time()
        total_runtime = end_time - start_time

        # System should remain stable
        assert error_count < 5, f"Too many errors: {error_count}"
        assert game_manager.current_state is not None
        assert len(tracker.tracks) > 0

        # Should maintain reasonable performance
        processed_frames = num_frames // 100
        avg_fps = processed_frames / total_runtime
        assert avg_fps > 10, f"Performance degraded: {avg_fps:.2f} FPS"


@pytest.mark.system
class TestHardwareIntegration:
    """Test hardware integration scenarios."""

    @patch("cv2.VideoCapture")
    def test_camera_initialization_workflow(self, mock_video_capture, config_module):
        """Test camera initialization and configuration workflow."""
        # Mock camera
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 30.0  # FPS
        mock_cap.read.return_value = (True, np.zeros((1080, 1920, 3), dtype=np.uint8))
        mock_video_capture.return_value = mock_cap

        from vision.calibration.camera import CameraCalibrator

        calibrator = CameraCalibrator()

        # 1. Initialize camera
        camera_config = config_module.get("camera")
        device_id = camera_config["device_id"]

        cap = cv2.VideoCapture(device_id)
        assert cap.isOpened()

        # 2. Configure camera settings
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_config["width"])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_config["height"])
        cap.set(cv2.CAP_PROP_FPS, camera_config["fps"])

        # 3. Test frame capture
        ret, frame = cap.read()
        assert ret
        assert frame.shape == (camera_config["height"], camera_config["width"], 3)

        # 4. Camera calibration
        # (Simplified - would normally use chessboard pattern)
        calibration_data = {
            "camera_matrix": [[1000, 0, 960], [0, 1000, 540], [0, 0, 1]],
            "distortion_coefficients": [0.1, -0.2, 0, 0, 0],
        }

        calibrator.set_calibration_data(calibration_data)
        assert calibrator.is_calibrated()

    @patch("moderngl.create_context")
    def test_projector_initialization_workflow(
        self, mock_create_context, config_module
    ):
        """Test projector initialization and configuration workflow."""
        # Mock OpenGL context
        mock_context = MagicMock()
        mock_create_context.return_value = mock_context

        from projector.calibration.geometry import GeometryCalibrator
        from projector.rendering.opengl.renderer import OpenGLRenderer

        # 1. Initialize projector
        projector_config = config_module.get("projector")
        renderer = OpenGLRenderer(
            width=projector_config["width"], height=projector_config["height"]
        )

        renderer.initialize()
        assert renderer.context is not None

        # 2. Set up projection
        renderer.set_viewport(
            0, 0, projector_config["width"], projector_config["height"]
        )

        # 3. Geometry calibration
        calibrator = GeometryCalibrator()

        # Add calibration points (table corners to projector coordinates)
        calibrator.add_point((0, 0), (100, 100))  # Table origin
        calibrator.add_point((2.84, 0), (1820, 100))  # Table width
        calibrator.add_point((2.84, 1.42), (1820, 980))  # Table corner
        calibrator.add_point((0, 1.42), (100, 980))  # Table height

        matrix = calibrator.calculate_transform_matrix()
        assert matrix is not None

        # 4. Test rendering
        from projector.models import RenderObject

        test_object = RenderObject(
            object_type="circle", position=(960, 540), size=(50, 50), color=(255, 0, 0)
        )

        renderer.clear()
        renderer.render_object(test_object)
        renderer.present()

        # Verify rendering completed without errors
        mock_context.clear.assert_called()

    def test_hardware_error_handling(self, config_module):
        """Test handling of hardware errors and failures."""
        # Test camera failure handling
        with patch("cv2.VideoCapture") as mock_video_capture:
            # Mock failed camera initialization
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = False
            mock_video_capture.return_value = mock_cap

            from vision.detection.balls import BallDetector

            BallDetector()

            # Should handle camera failure gracefully
            try:
                cap = cv2.VideoCapture(0)
                if not cap.isOpened():
                    raise RuntimeError("Camera initialization failed")
            except RuntimeError as e:
                assert "Camera initialization failed" in str(e)

        # Test projector failure handling
        with patch("moderngl.create_context") as mock_create_context:
            # Mock OpenGL context creation failure
            mock_create_context.side_effect = Exception(
                "OpenGL context creation failed"
            )

            from projector.rendering.opengl.renderer import OpenGLRenderer

            renderer = OpenGLRenderer()

            # Should handle OpenGL failure gracefully
            with pytest.raises(Exception) as exc_info:
                renderer.initialize()

            assert "OpenGL context creation failed" in str(exc_info.value)

    def test_hardware_performance_validation(self, config_module, performance_timer):
        """Test hardware performance validation."""
        # Mock high-performance camera
        with patch("cv2.VideoCapture") as mock_video_capture:
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            mock_cap.get.return_value = 60.0  # 60 FPS
            mock_cap.read.return_value = (
                True,
                np.zeros((1080, 1920, 3), dtype=np.uint8),
            )
            mock_video_capture.return_value = mock_cap

            from vision.detection.balls import BallDetector

            detector = BallDetector()
            cap = cv2.VideoCapture(0)

            # Test frame capture performance
            num_frames = 60  # 1 second at 60 FPS

            performance_timer.start()

            for _frame_id in range(num_frames):
                ret, frame = cap.read()
                assert ret

                # Quick detection
                detector.detect(frame)

            performance_timer.stop()

            # Should maintain 60 FPS
            avg_frame_time = performance_timer.elapsed_ms / num_frames
            assert (
                avg_frame_time < 16.67
            ), f"Frame processing too slow: {avg_frame_time:.2f}ms"

        # Mock high-performance projector
        with patch("moderngl.create_context") as mock_create_context:
            mock_context = MagicMock()
            mock_create_context.return_value = mock_context

            from projector.models import RenderObject
            from projector.rendering.opengl.renderer import OpenGLRenderer

            renderer = OpenGLRenderer()
            renderer.initialize()

            # Test rendering performance
            objects = []
            for i in range(100):  # Many objects
                obj = RenderObject(
                    object_type="circle",
                    position=(i * 10, i * 10),
                    size=(20, 20),
                    color=(255, 0, 0),
                )
                objects.append(obj)

            performance_timer.start()

            for frame in range(60):  # 1 second at 60 FPS
                renderer.clear()
                for obj in objects:
                    renderer.render_object(obj)
                renderer.present()

            performance_timer.stop()

            # Should maintain 60 FPS rendering
            avg_frame_time = performance_timer.elapsed_ms / 60
            assert avg_frame_time < 16.67, f"Rendering too slow: {avg_frame_time:.2f}ms"
