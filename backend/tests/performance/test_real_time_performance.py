"""Performance tests for real-time requirements."""

import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import AsyncMock, MagicMock

import cv2
import numpy as np
import pytest
from api.websocket.handler import WebSocketHandler
from core.physics.engine import PhysicsEngine

from backend.vision.detection.balls import BallDetector
from backend.vision.tracking.tracker import ObjectTracker


@pytest.mark.performance()
class TestCameraProcessingPerformance:
    """Test camera processing performance requirements."""

    @pytest.mark.opencv_available()
    def test_30_fps_camera_processing(self, performance_timer, memory_monitor):
        """Test maintaining 30+ FPS camera processing."""
        detector = BallDetector(config={})
        tracker = ObjectTracker()

        # Create test frames
        frames = []
        for i in range(90):  # 3 seconds worth at 30 FPS
            frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
            frame[:, :] = [34, 139, 34]  # Green background

            # Add moving ball
            x = 400 + i * 10
            y = 400 + i * 5
            cv2.circle(frame, (x, y), 25, (255, 255, 255), -1)

            frames.append(frame)

        memory_monitor.start()
        performance_timer.start()

        processed_frames = 0
        for frame_id, frame in enumerate(frames):
            frame_start = time.perf_counter()

            # Detection
            detections = detector.detect_balls(frame)

            # Tracking
            timestamp = time.time() + frame_id * 0.033
            for detection in detections:
                detection_data = {
                    "id": detection.ball_type.value if detection.ball_type else "cue",
                    "x": detection.position[0],
                    "y": detection.position[1],
                    "confidence": detection.confidence,
                    "timestamp": timestamp,
                }

                if frame_id == 0:
                    tracker.add_detection(detection_data)
                else:
                    tracker.update(detection_data)

            frame_end = time.perf_counter()
            frame_time = (frame_end - frame_start) * 1000  # ms

            # Each frame must process in < 33.33ms for 30 FPS
            assert (
                frame_time < 33.33
            ), f"Frame {frame_id} took {frame_time:.2f}ms (> 33.33ms)"

            processed_frames += 1
            memory_monitor.update()

        performance_timer.stop()

        # Overall performance check
        total_time = performance_timer.elapsed
        actual_fps = processed_frames / total_time

        assert actual_fps >= 30.0, f"Achieved {actual_fps:.2f} FPS, required >= 30 FPS"

        # Memory usage should be reasonable
        assert (
            memory_monitor.memory_increase_mb < 100
        ), f"Memory increased by {memory_monitor.memory_increase_mb:.2f} MB"

    def test_60_fps_camera_processing(self, performance_timer):
        """Test maintaining 60 FPS camera processing for high-performance mode."""
        detector = BallDetector(config={})

        # Create lightweight test frames
        frames = []
        for i in range(180):  # 3 seconds worth at 60 FPS
            frame = np.zeros((720, 1280, 3), dtype=np.uint8)  # Lower resolution
            frame[:, :] = [34, 139, 34]

            # Simple ball
            x = 300 + i * 5
            y = 300 + i * 3
            cv2.circle(frame, (x, y), 20, (255, 255, 255), -1)

            frames.append(frame)

        performance_timer.start()

        for frame_id, frame in enumerate(frames):
            frame_start = time.perf_counter()

            # Lightweight detection
            detector.detect_balls(frame)

            frame_end = time.perf_counter()
            frame_time = (frame_end - frame_start) * 1000

            # Each frame must process in < 16.67ms for 60 FPS
            assert (
                frame_time < 16.67
            ), f"Frame {frame_id} took {frame_time:.2f}ms (> 16.67ms)"

        performance_timer.stop()

        total_time = performance_timer.elapsed
        actual_fps = len(frames) / total_time

        assert actual_fps >= 60.0, f"Achieved {actual_fps:.2f} FPS, required >= 60 FPS"

    def test_batch_processing_performance(self, performance_timer):
        """Test batch processing performance for analysis."""
        detector = BallDetector(config={})

        # Create batch of frames
        batch_size = 100
        frames = []
        for _i in range(batch_size):
            frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
            frame[:, :] = [34, 139, 34]
            cv2.circle(frame, (960, 540), 25, (255, 255, 255), -1)
            frames.append(frame)

        performance_timer.start()

        # Process batch
        all_detections = []
        for frame in frames:
            detections = detector.detect_balls(frame)
            all_detections.append(detections)

        performance_timer.stop()

        # Should process batch efficiently
        avg_frame_time = performance_timer.elapsed_ms / batch_size
        assert (
            avg_frame_time < 50
        ), f"Average frame time {avg_frame_time:.2f}ms too slow"

        assert len(all_detections) == batch_size

    def test_concurrent_processing_performance(self, performance_timer):
        """Test concurrent frame processing performance."""
        detector = BallDetector(config={})

        def process_frame(frame_data):
            frame_id, frame = frame_data
            return detector.detect_balls(frame)

        # Create frames
        frames = []
        for i in range(60):
            frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
            frame[:, :] = [34, 139, 34]
            cv2.circle(frame, (400 + i * 10, 400), 25, (255, 255, 255), -1)
            frames.append((i, frame))

        performance_timer.start()

        # Process concurrently
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(process_frame, frames))

        performance_timer.stop()

        # Should complete faster than sequential processing
        assert performance_timer.elapsed_ms < 2000  # Should be fast with concurrency
        assert len(results) == 60


@pytest.mark.performance()
class TestWebSocketLatencyPerformance:
    """Test WebSocket latency performance requirements."""

    @pytest.mark.asyncio()
    async def test_websocket_message_latency(self, performance_timer):
        """Test WebSocket message latency < 50ms."""
        handler = WebSocketHandler()

        # Mock WebSocket
        mock_websocket = AsyncMock()
        mock_websocket.send_json = AsyncMock()

        await handler.connect(mock_websocket)

        # Test message sending latency
        message = {
            "type": "game_state_update",
            "data": {
                "balls": [{"id": "cue", "x": 1.42, "y": 0.71}],
                "timestamp": time.time(),
            },
        }

        latencies = []
        for _i in range(100):
            performance_timer.start()
            await handler.send_personal_message(mock_websocket, message)
            performance_timer.stop()

            latencies.append(performance_timer.elapsed_ms)
            performance_timer = type(performance_timer)()  # Reset timer

        avg_latency = sum(latencies) / len(latencies)
        max_latency = max(latencies)

        assert avg_latency < 50, f"Average latency {avg_latency:.2f}ms > 50ms"
        assert max_latency < 100, f"Max latency {max_latency:.2f}ms > 100ms"

    @pytest.mark.asyncio()
    async def test_websocket_broadcast_performance(self, performance_timer):
        """Test WebSocket broadcast performance."""
        handler = WebSocketHandler()

        # Create multiple mock clients
        clients = []
        for _i in range(10):
            mock_ws = AsyncMock()
            mock_ws.send_json = AsyncMock()
            await handler.connect(mock_ws)
            clients.append(mock_ws)

        message = {
            "type": "ball_update",
            "data": {"ball_id": "cue", "x": 100, "y": 200},
        }

        performance_timer.start()

        # Broadcast to all clients
        await handler.broadcast(message)

        performance_timer.stop()

        # Should broadcast quickly even with multiple clients
        assert performance_timer.elapsed_ms < 100

        # Verify all clients received message
        for client in clients:
            client.send_json.assert_called_with(message)

    @pytest.mark.asyncio()
    async def test_websocket_concurrent_connections(self, performance_timer):
        """Test performance with many concurrent WebSocket connections."""
        handler = WebSocketHandler()

        # Create many mock connections
        num_connections = 50
        clients = []

        performance_timer.start()

        for _i in range(num_connections):
            mock_ws = AsyncMock()
            mock_ws.send_json = AsyncMock()
            await handler.connect(mock_ws)
            clients.append(mock_ws)

        performance_timer.stop()

        # Connection setup should be fast
        assert performance_timer.elapsed_ms < 1000  # 1 second for 50 connections

        # Test broadcast performance with many clients
        message = {"type": "test", "data": {}}

        performance_timer.start()
        await handler.broadcast(message)
        performance_timer.stop()

        # Broadcast should still be fast
        assert performance_timer.elapsed_ms < 500

    @pytest.mark.asyncio()
    async def test_websocket_message_throughput(self, performance_timer):
        """Test WebSocket message throughput."""
        handler = WebSocketHandler()

        mock_websocket = AsyncMock()
        mock_websocket.send_json = AsyncMock()
        await handler.connect(mock_websocket)

        # Send many messages rapidly
        num_messages = 1000
        messages = []
        for i in range(num_messages):
            messages.append(
                {
                    "type": "position_update",
                    "data": {"x": i, "y": i * 2, "timestamp": time.time()},
                }
            )

        performance_timer.start()

        # Send all messages
        for message in messages:
            await handler.send_personal_message(mock_websocket, message)

        performance_timer.stop()

        # Calculate throughput
        throughput = num_messages / performance_timer.elapsed
        assert throughput > 500, f"Throughput {throughput:.2f} messages/sec too low"


@pytest.mark.performance()
class TestPhysicsPerformance:
    """Test physics engine performance requirements."""

    def test_physics_simulation_60fps(self, performance_timer, mock_game_state):
        """Test physics simulation at 60 FPS."""
        physics_engine = PhysicsEngine()

        # Add velocity to balls for active simulation
        for ball in mock_game_state.balls:
            ball.velocity_x = np.random.uniform(-2, 2)
            ball.velocity_y = np.random.uniform(-2, 2)

        # Simulate 3 seconds at 60 FPS
        num_steps = 180  # 3 seconds * 60 FPS
        dt = 1.0 / 60.0  # 60 FPS timestep

        performance_timer.start()

        for step in range(num_steps):
            step_start = time.perf_counter()

            physics_engine.simulate_step(mock_game_state, dt)

            step_end = time.perf_counter()
            step_time = (step_end - step_start) * 1000

            # Each step must complete in < 16.67ms for 60 FPS
            assert step_time < 16.67, f"Step {step} took {step_time:.2f}ms (> 16.67ms)"

        performance_timer.stop()

        total_time = performance_timer.elapsed
        actual_fps = num_steps / total_time

        assert (
            actual_fps >= 60.0
        ), f"Physics achieved {actual_fps:.2f} FPS, required >= 60 FPS"

    def test_physics_collision_detection_performance(self, performance_timer):
        """Test collision detection performance with many balls."""
        physics_engine = PhysicsEngine()

        # Create game state with many balls
        from core.models import Ball, GameState, Table

        table = Table(
            width=2.84,
            height=1.42,
            corners=[(0, 0), (2.84, 0), (2.84, 1.42), (0, 1.42)],
        )

        # Create 16 balls (full rack)
        balls = []
        for i in range(16):
            ball = Ball(
                id=str(i),
                x=0.5 + (i % 4) * 0.5,
                y=0.3 + (i // 4) * 0.3,
                radius=0.028575,
                color="test",
                velocity_x=np.random.uniform(-1, 1),
                velocity_y=np.random.uniform(-1, 1),
            )
            balls.append(ball)

        GameState(
            table=table,
            balls=balls,
            current_player=1,
            shot_clock=30.0,
            game_mode="8-ball",
        )

        # Test collision detection performance
        num_iterations = 100

        performance_timer.start()

        for _iteration in range(num_iterations):
            # Check all ball-to-ball collisions
            for i, ball1 in enumerate(balls):
                for _j, ball2 in enumerate(balls[i + 1 :], i + 1):
                    collision = physics_engine.detect_ball_collision(ball1, ball2)
                    if collision:
                        physics_engine.resolve_ball_collision(ball1, ball2)

        performance_timer.stop()

        avg_iteration_time = performance_timer.elapsed_ms / num_iterations
        assert (
            avg_iteration_time < 10
        ), f"Collision detection took {avg_iteration_time:.2f}ms per iteration"

    def test_physics_memory_efficiency(self, memory_monitor):
        """Test physics engine memory efficiency."""
        physics_engine = PhysicsEngine()

        # Create large game state
        from core.models import Ball, GameState, Table

        table = Table(
            width=2.84,
            height=1.42,
            corners=[(0, 0), (2.84, 0), (2.84, 1.42), (0, 1.42)],
        )

        balls = []
        for i in range(50):  # Many balls
            ball = Ball(
                id=str(i),
                x=np.random.uniform(0.1, 2.74),
                y=np.random.uniform(0.1, 1.32),
                radius=0.028575,
                color="test",
                velocity_x=np.random.uniform(-2, 2),
                velocity_y=np.random.uniform(-2, 2),
            )
            balls.append(ball)

        game_state = GameState(
            table=table,
            balls=balls,
            current_player=1,
            shot_clock=30.0,
            game_mode="8-ball",
        )

        memory_monitor.start()

        # Run simulation for extended period
        for _step in range(1000):
            physics_engine.simulate_step(game_state, 1.0 / 60.0)
            memory_monitor.update()

        # Memory usage should remain reasonable
        assert (
            memory_monitor.memory_increase_mb < 50
        ), f"Memory increased by {memory_monitor.memory_increase_mb:.2f} MB"


@pytest.mark.performance()
class TestMemoryUsagePerformance:
    """Test memory usage performance requirements."""

    def test_continuous_operation_memory_stability(self, memory_monitor):
        """Test memory stability during continuous operation."""
        from core.game_state import GameStateManager
        from vision.detection.balls import BallDetector
        from vision.tracking.tracker import ObjectTracker

        detector = BallDetector(config={})
        tracker = ObjectTracker()
        GameStateManager()

        memory_monitor.start()

        # Simulate 10 minutes of operation at 30 FPS
        num_frames = 30 * 60 * 10  # 30 FPS * 60 sec * 10 min = 18,000 frames

        for frame_id in range(0, num_frames, 100):  # Sample every 100 frames
            # Create frame
            frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
            frame[:, :] = [34, 139, 34]
            cv2.circle(frame, (960, 540), 25, (255, 255, 255), -1)

            # Process frame
            detections = detector.detect_balls(frame)

            # Update tracking
            timestamp = time.time() + frame_id * 0.033
            for detection in detections:
                detection_data = {
                    "id": detection.ball_type.value if detection.ball_type else "cue",
                    "x": detection.position[0],
                    "y": detection.position[1],
                    "confidence": detection.confidence,
                    "timestamp": timestamp,
                }

                if frame_id == 0:
                    tracker.add_detection(detection_data)
                else:
                    tracker.update(detection_data)

            # Cleanup old tracks periodically
            if frame_id % 1000 == 0:
                tracker.cleanup_stale_tracks()

            memory_monitor.update()

        # Memory increase should be minimal for continuous operation
        assert (
            memory_monitor.memory_increase_mb < 200
        ), f"Memory increased by {memory_monitor.memory_increase_mb:.2f} MB"

    def test_memory_leak_detection(self, memory_monitor):
        """Test for memory leaks in core operations."""
        detector = BallDetector(config={})

        memory_monitor.start()

        # Perform repetitive operations that could leak memory
        for cycle in range(100):
            # Create and process many frames
            for _i in range(10):
                frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
                detector.detect_balls(frame)

            # Force garbage collection periodically
            if cycle % 20 == 0:
                import gc

                gc.collect()
                memory_monitor.update()

        # Memory should not continuously increase
        assert (
            memory_monitor.memory_increase_mb < 100
        ), f"Potential memory leak: {memory_monitor.memory_increase_mb:.2f} MB increase"


@pytest.mark.performance()
class TestCPUUtilizationPerformance:
    """Test CPU utilization performance requirements."""

    def test_cpu_usage_monitoring(self, performance_timer):
        """Test CPU usage during typical operations."""
        import psutil

        # Get initial CPU usage
        initial_cpu = psutil.cpu_percent(interval=1)

        detector = BallDetector(config={})
        tracker = ObjectTracker()

        performance_timer.start()

        # Perform CPU-intensive operations
        for i in range(30):  # 1 second at 30 FPS
            frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
            frame[:, :] = [34, 139, 34]
            cv2.circle(frame, (400 + i * 10, 400), 25, (255, 255, 255), -1)

            detections = detector.detect_balls(frame)

            timestamp = time.time() + i * 0.033
            for detection in detections:
                detection_data = {
                    "id": detection.ball_type.value if detection.ball_type else "cue",
                    "x": detection.position[0],
                    "y": detection.position[1],
                    "confidence": detection.confidence,
                    "timestamp": timestamp,
                }

                if i == 0:
                    tracker.add_detection(detection_data)
                else:
                    tracker.update(detection_data)

        performance_timer.stop()

        # Get final CPU usage
        final_cpu = psutil.cpu_percent(interval=1)

        # CPU usage should be reasonable
        cpu_increase = final_cpu - initial_cpu
        assert cpu_increase < 80, f"CPU usage increased by {cpu_increase}%"

        # Should maintain real-time performance
        assert (
            performance_timer.elapsed < 1.5
        ), f"30 frames took {performance_timer.elapsed:.2f}s (should be ~1s)"

    def test_multithreaded_performance(self, performance_timer):
        """Test multithreaded processing performance."""
        from concurrent.futures import ThreadPoolExecutor

        def cpu_intensive_task(task_id):
            # Simulate CPU-intensive vision processing
            frame = np.random.randint(0, 256, (720, 1280, 3), dtype=np.uint8)
            # Simple convolution operation
            kernel = np.ones((5, 5), np.float32) / 25
            result = cv2.filter2D(frame, -1, kernel)
            return result.mean()

        num_tasks = 100
        max_workers = 4

        performance_timer.start()

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(cpu_intensive_task, range(num_tasks)))

        performance_timer.stop()

        # Should complete efficiently with threading
        assert len(results) == num_tasks
        assert (
            performance_timer.elapsed < 10
        ), f"Multithreaded tasks took {performance_timer.elapsed:.2f}s"

        # Calculate throughput
        throughput = num_tasks / performance_timer.elapsed
        assert throughput > 10, f"Throughput {throughput:.2f} tasks/sec too low"
