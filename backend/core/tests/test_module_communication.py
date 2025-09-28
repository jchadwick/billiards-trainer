"""Tests for module communication and integration system."""

import threading
import time
from typing import Any
from unittest.mock import Mock, patch

import pytest

from backend.core.events.handlers import CoreEventHandlers
from backend.core.events.manager import Event, EventManager, EventPriority, EventType
from backend.core.game_state import GameStateManager
from backend.core.integration import (
    APIInterfaceImpl,
    ConfigInterfaceImpl,
    CoreModuleIntegrator,
    ProjectorInterfaceImpl,
    VisionInterfaceImpl,
)
from backend.core.models import GameState, GameType, ShotAnalysis, Trajectory


class TestEventManager:
    """Test the EventManager class."""

    def test_initialization(self):
        """Test EventManager initialization."""
        manager = EventManager(max_history_size=100, enable_logging=False)
        assert manager.is_running
        assert len(manager.event_history) == 0
        assert manager.coordinator is not None

    def test_legacy_api_compatibility(self):
        """Test backward compatibility with legacy API."""
        manager = EventManager(enable_logging=False)

        # Test legacy subscription
        callback_called = False

        def test_callback(event_type: str, data: dict[str, Any]):
            nonlocal callback_called
            callback_called = True
            assert event_type == "test_event"
            assert data["message"] == "test"

        sub_id = manager.subscribe_to_events("test_event", test_callback)
        assert isinstance(sub_id, str)

        # Test legacy emit
        manager.emit_event("test_event", {"message": "test"})

        # Give some time for event processing
        time.sleep(0.1)
        assert callback_called

        # Test unsubscribe
        assert manager.unsubscribe(sub_id)
        assert not manager.unsubscribe("invalid_id")

    def test_enhanced_event_emission(self):
        """Test enhanced event emission with targeting."""
        manager = EventManager(enable_logging=False)

        events_received = []

        def test_callback(event_type: str, data: dict[str, Any]):
            events_received.append((event_type, data))

        manager.subscribe_to_events(EventType.STATE_UPDATED.value, test_callback)

        # Emit enhanced event
        event_id = manager.emit_enhanced_event(
            EventType.STATE_UPDATED,
            {"frame": 1, "balls": 5},
            source_module="test",
            priority=EventPriority.HIGH,
        )

        assert isinstance(event_id, str)
        time.sleep(0.1)  # Allow processing
        assert len(events_received) > 0

    def test_event_history(self):
        """Test event history functionality."""
        manager = EventManager(max_history_size=5, enable_logging=False)

        # Emit multiple events
        for i in range(10):
            manager.emit_enhanced_event(
                EventType.BALL_MOVED,
                {"ball_id": f"ball_{i}", "frame": i},
                source_module="test",
            )

        time.sleep(0.2)  # Allow processing

        # Check history size limit
        assert len(manager.event_history) <= 5

        # Test history filtering
        history = manager.get_event_history(event_type=EventType.BALL_MOVED, limit=3)
        assert len(history) <= 3

    def test_event_filters(self):
        """Test event filtering functionality."""
        manager = EventManager(enable_logging=False)

        # Add filter that blocks certain events
        def test_filter(event: Event) -> bool:
            return "block" not in event.data

        manager.add_event_filter("test_filter", test_filter)

        events_received = []

        def test_callback(event_type: str, data: dict[str, Any]):
            events_received.append(data)

        manager.subscribe_to_events("test_event", test_callback)

        # Emit events - one should be blocked
        manager.emit_enhanced_event("test_event", {"message": "allow"}, "test")
        manager.emit_enhanced_event("test_event", {"message": "block"}, "test")

        time.sleep(0.1)

        # Only one event should pass through
        assert len(events_received) == 1
        assert events_received[0]["message"] == "allow"

        # Remove filter
        assert manager.remove_event_filter("test_filter")
        assert not manager.remove_event_filter("nonexistent")

    def test_module_coordination(self):
        """Test module coordination functionality."""
        manager = EventManager(enable_logging=False)
        coordinator = manager.coordinator

        # Register a module
        mock_interface = Mock()
        coordinator.register_module(
            "test_module", mock_interface, ["feature1", "feature2"]
        )

        time.sleep(0.1)

        # Check module registration
        status = coordinator.get_module_status("test_module")
        assert status is not None
        assert status["status"] == "initialized"
        assert "feature1" in status["capabilities"]

        # Test data sending
        coordinator.send_data_to_module("test_module", "state_update", {"frame": 1})

        # Test broadcast
        coordinator.broadcast_state_update({"global_frame": 1})

    def test_statistics(self):
        """Test statistics collection."""
        manager = EventManager(enable_logging=False)

        # Emit some events
        for i in range(5):
            manager.emit_event("test", {"count": i})

        time.sleep(0.1)

        stats = manager.get_statistics()
        assert stats["events_emitted"] >= 5
        assert stats["is_running"]
        assert "subscription_count" in stats

    def test_cleanup(self):
        """Test proper cleanup."""
        manager = EventManager(enable_logging=False)
        assert manager.is_running

        manager.stop_enhanced_processing()
        assert not manager.is_running


class TestCoreEventHandlers:
    """Test the CoreEventHandlers class."""

    def setup_method(self):
        """Setup test environment."""
        self.event_manager = EventManager(enable_logging=False)
        self.game_state_manager = Mock(spec=GameStateManager)
        self.handlers = CoreEventHandlers(self.event_manager, self.game_state_manager)

    def test_initialization(self):
        """Test handler initialization."""
        assert self.handlers.event_manager is self.event_manager
        assert self.handlers.game_state_manager is self.game_state_manager
        assert len(self.handlers.handlers) > 0

    def test_state_change_handling(self):
        """Test state change event handling."""
        # Mock current state
        mock_state = Mock(spec=GameState)
        mock_state.balls = []
        mock_state.cue = None
        self.game_state_manager.get_current_state.return_value = mock_state

        # Trigger state change
        self.handlers.handle_state_change(
            EventType.STATE_UPDATED.value,
            {
                "frame_number": 1,
                "timestamp": time.time(),
                "balls_count": 0,
                "events": [],
            },
        )

        # Verify calls
        self.game_state_manager.get_current_state.assert_called()

    def test_vision_data_handling(self):
        """Test vision data handling."""
        detection_data = {
            "balls": [
                {
                    "id": "ball_1",
                    "x": 0.5,
                    "y": 0.5,
                    "radius": 0.028575,
                    "confidence": 0.9,
                }
            ]
        }

        mock_state = Mock(spec=GameState)
        self.game_state_manager.update_state.return_value = mock_state

        # Handle vision data
        self.handlers.handle_vision_data(
            EventType.VISION_DATA_RECEIVED.value, {"detection_data": detection_data}
        )

        # Verify state update was called
        self.game_state_manager.update_state.assert_called_with(detection_data)

    def test_ball_movement_handling(self):
        """Test ball movement event handling."""
        # Create mock to capture emitted events
        with patch.object(self.event_manager, "emit_event") as mock_emit:
            self.handlers.handle_ball_movement(
                EventType.BALL_MOVED.value,
                {"ball_id": "ball_1", "distance": 100},  # Significant movement
            )

            # Should emit shot detection event
            mock_emit.assert_called()
            call_args = mock_emit.call_args
            assert call_args[0][0] == EventType.SHOT_DETECTED.value

    def test_collision_handling(self):
        """Test collision event handling."""
        self.handlers.collision_history = []  # Initialize collision history

        self.handlers.handle_collision_event(
            EventType.COLLISION_DETECTED.value,
            {
                "ball1_id": "ball_1",
                "ball2_id": "ball_2",
                "type": "ball",
                "position": {"x": 0.5, "y": 0.5},
            },
        )

        # Verify collision was logged
        assert len(self.handlers.collision_history) == 1
        assert self.handlers.collision_history[0]["ball1_id"] == "ball_1"

    def test_custom_handler_management(self):
        """Test custom handler addition and removal."""
        handler_called = False

        def custom_handler(event_type: str, data: dict[str, Any]):
            nonlocal handler_called
            handler_called = True

        # Add custom handler
        self.handlers.add_custom_handler("custom_event", custom_handler)
        assert "custom_event" in self.handlers.handlers

        # Trigger custom event
        self.event_manager.emit_event("custom_event", {"test": True})
        time.sleep(0.1)
        assert handler_called

        # Remove custom handler
        assert self.handlers.remove_custom_handler("custom_event", custom_handler)
        assert not self.handlers.remove_custom_handler("custom_event", custom_handler)

    def test_handler_statistics(self):
        """Test handler statistics."""
        stats = self.handlers.get_handler_statistics()
        assert "events_handled" in stats
        assert "registered_handlers" in stats
        assert stats["registered_handlers"] > 0

    def teardown_method(self):
        """Cleanup test environment."""
        self.event_manager.stop_enhanced_processing()


class TestCoreModuleIntegrator:
    """Test the CoreModuleIntegrator class."""

    def setup_method(self):
        """Setup test environment."""
        self.event_manager = EventManager(enable_logging=False)
        self.game_state_manager = Mock(spec=GameStateManager)
        self.integrator = CoreModuleIntegrator(
            self.event_manager, self.game_state_manager
        )

    def test_initialization(self):
        """Test integrator initialization."""
        assert self.integrator.event_manager is self.event_manager
        assert self.integrator.game_state_manager is self.game_state_manager
        assert len(self.integrator.connected_modules) == 0

    def test_interface_registration(self):
        """Test module interface registration."""
        # Create mock interfaces
        vision_interface = VisionInterfaceImpl(self.event_manager)
        api_interface = APIInterfaceImpl(self.event_manager)
        projector_interface = ProjectorInterfaceImpl(self.event_manager)
        config_interface = ConfigInterfaceImpl(self.event_manager)

        # Register interfaces
        self.integrator.register_vision_interface(vision_interface)
        self.integrator.register_api_interface(api_interface)
        self.integrator.register_projector_interface(projector_interface)
        self.integrator.register_config_interface(config_interface)

        # Verify registration
        assert self.integrator.vision_interface is vision_interface
        assert self.integrator.api_interface is api_interface
        assert self.integrator.projector_interface is projector_interface
        assert self.integrator.config_interface is config_interface

    def test_module_lifecycle_handling(self):
        """Test module lifecycle event handling."""
        # Simulate module initialization
        self.integrator._handle_module_initialization(
            EventType.MODULE_INITIALIZED.value,
            {
                "module_name": "test_module",
                "version": "1.0.0",
                "capabilities": ["feature1", "feature2"],
                "requirements": ["requirement1"],
            },
        )

        # Verify module was registered
        assert "test_module" in self.integrator.connected_modules
        module_cap = self.integrator.connected_modules["test_module"]
        assert module_cap.name == "test_module"
        assert "feature1" in module_cap.features

        # Simulate module shutdown
        self.integrator._handle_module_shutdown(
            EventType.MODULE_SHUTDOWN.value, {"module_name": "test_module"}
        )

        # Verify module was removed
        assert "test_module" not in self.integrator.connected_modules

    def test_state_broadcasting(self):
        """Test state broadcasting to modules."""
        # Create mock state
        mock_state = Mock(spec=GameState)
        mock_state.timestamp = time.time()
        mock_state.frame_number = 1
        mock_state.balls = []
        mock_state.table = Mock()
        mock_state.table.to_dict.return_value = {"width": 2.54, "height": 1.27}
        mock_state.cue = None
        mock_state.game_type = GameType.PRACTICE
        mock_state.is_break = False
        mock_state.scores = {}
        mock_state.events = []

        # Register mock interfaces
        api_interface = Mock()
        self.integrator.api_interface = api_interface

        # Broadcast state
        self.integrator._broadcast_state_update(mock_state)

        # Verify API interface was called (through event system)
        time.sleep(0.1)  # Allow event processing

    def test_trajectory_update(self):
        """Test trajectory update sending."""
        # Create mock trajectory
        mock_trajectory = Mock(spec=Trajectory)
        mock_trajectory.to_dict.return_value = {
            "ball_id": "ball_1",
            "points": [{"x": 0, "y": 0}, {"x": 1, "y": 1}],
            "collisions": [],
            "time_to_rest": 2.0,
        }

        # Send trajectory update
        self.integrator.send_trajectory_update([mock_trajectory])

        # Verify event was emitted
        time.sleep(0.1)

    def test_shot_analysis_sending(self):
        """Test shot analysis sending."""
        # Create mock shot analysis
        mock_analysis = Mock(spec=ShotAnalysis)
        mock_analysis.to_dict.return_value = {
            "shot_type": "direct",
            "difficulty": 0.5,
            "success_probability": 0.8,
            "recommended_force": 10.0,
            "recommended_angle": 45.0,
        }

        # Register mock interfaces
        api_interface = Mock()
        projector_interface = Mock()
        self.integrator.api_interface = api_interface
        self.integrator.projector_interface = projector_interface

        # Send shot analysis
        self.integrator.send_shot_analysis(mock_analysis)

        # Verify interfaces were called
        api_interface.send_event_notification.assert_called()
        projector_interface.send_overlay_data.assert_called()

    def test_vision_calibration_request(self):
        """Test vision calibration request."""
        # Register mock vision interface
        vision_interface = Mock()
        vision_interface.request_calibration.return_value = {"status": "success"}
        self.integrator.vision_interface = vision_interface

        # Request calibration
        result = self.integrator.request_vision_calibration("full")

        # Verify call and result
        vision_interface.request_calibration.assert_called_with("full")
        assert result["status"] == "success"

        # Test with no interface
        self.integrator.vision_interface = None
        result = self.integrator.request_vision_calibration("full")
        assert result is None

    def test_module_status_queries(self):
        """Test module status queries."""
        # Mock coordinator methods
        self.event_manager.coordinator.get_module_status = Mock(
            return_value={"status": "active"}
        )
        self.event_manager.coordinator.get_all_module_statuses = Mock(
            return_value={"module1": {"status": "active"}}
        )

        # Test single module status
        status = self.integrator.get_module_status("module1")
        assert status["status"] == "active"

        # Test all module statuses
        all_statuses = self.integrator.get_all_module_statuses()
        assert "module1" in all_statuses

    def test_integration_statistics(self):
        """Test integration statistics."""
        stats = self.integrator.get_integration_statistics()
        assert "messages_sent" in stats
        assert "messages_received" in stats
        assert "connected_modules" in stats
        assert "event_manager_stats" in stats

    def test_shutdown(self):
        """Test integration shutdown."""
        # Add some connected modules
        self.integrator.connected_modules["module1"] = Mock()
        self.integrator.connected_modules["module2"] = Mock()

        # Register interfaces
        self.integrator.vision_interface = Mock()
        self.integrator.api_interface = Mock()

        # Shutdown
        self.integrator.shutdown_integration()

        # Verify cleanup
        assert self.integrator.vision_interface is None
        assert self.integrator.api_interface is None

    def teardown_method(self):
        """Cleanup test environment."""
        self.integrator.shutdown_integration()
        self.event_manager.stop_enhanced_processing()


class TestIntegrationInterfaces:
    """Test the integration interface implementations."""

    def setup_method(self):
        """Setup test environment."""
        self.event_manager = EventManager(enable_logging=False)

    def test_vision_interface(self):
        """Test VisionInterfaceImpl."""
        interface = VisionInterfaceImpl(self.event_manager)

        # Test detection data reception
        detection_data = {"balls": [{"id": "ball_1", "x": 0.5, "y": 0.5}]}
        interface.receive_detection_data(detection_data)

        # Test calibration request
        interface.calibration_data["full"] = {
            "camera_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        }
        result = interface.request_calibration("full")
        assert "camera_matrix" in result

        # Test parameter setting
        params = {"confidence_threshold": 0.8}
        assert interface.set_detection_parameters(params)
        assert interface.detection_parameters["confidence_threshold"] == 0.8

    def test_api_interface(self):
        """Test APIInterfaceImpl."""
        interface = APIInterfaceImpl(self.event_manager)

        # Test state update (no exception should be raised)
        interface.send_state_update({"frame": 1})

        # Test event notification
        interface.send_event_notification({"type": "test"})

        # Test WebSocket handler registration
        def test_handler(message):
            pass

        handler_id = interface.register_websocket_handler(test_handler)
        assert handler_id in interface.websocket_handlers
        assert interface.websocket_handlers[handler_id] is test_handler

    def test_projector_interface(self):
        """Test ProjectorInterfaceImpl."""
        interface = ProjectorInterfaceImpl(self.event_manager)

        # Test trajectory data sending
        interface.send_trajectory_data({"trajectories": []})

        # Test overlay data sending
        interface.send_overlay_data({"type": "assistance"})

        # Test settings update
        settings = {"brightness": 0.9}
        interface.update_projection_settings(settings)
        assert interface.projection_settings["brightness"] == 0.9

    def test_config_interface(self):
        """Test ConfigInterfaceImpl."""
        interface = ConfigInterfaceImpl(self.event_manager)

        # Test config get/update
        config = {"setting1": "value1"}
        interface.update_module_config("test_module", config)
        assert interface.get_module_config("test_module") == config

        # Test subscription
        def test_callback(module_name, config):
            pass

        sub_id = interface.subscribe_config_changes(test_callback)
        assert sub_id in interface.config_subscribers

    def teardown_method(self):
        """Cleanup test environment."""
        self.event_manager.stop_enhanced_processing()


class TestEndToEndIntegration:
    """End-to-end integration tests."""

    def setup_method(self):
        """Setup complete integration environment."""
        self.event_manager = EventManager(enable_logging=False)
        self.game_state_manager = GameStateManager()
        self.integrator = CoreModuleIntegrator(
            self.event_manager, self.game_state_manager
        )

        # Register all interfaces
        self.vision_interface = VisionInterfaceImpl(self.event_manager)
        self.api_interface = APIInterfaceImpl(self.event_manager)
        self.projector_interface = ProjectorInterfaceImpl(self.event_manager)
        self.config_interface = ConfigInterfaceImpl(self.event_manager)

        self.integrator.register_vision_interface(self.vision_interface)
        self.integrator.register_api_interface(self.api_interface)
        self.integrator.register_projector_interface(self.projector_interface)
        self.integrator.register_config_interface(self.config_interface)

        # Setup event handlers
        self.handlers = CoreEventHandlers(self.event_manager, self.game_state_manager)

    def test_full_vision_to_api_flow(self):
        """Test complete flow from vision data to API updates."""
        # Create detection data
        detection_data = {
            "balls": [
                {
                    "id": "cue",
                    "x": 0.635,  # meters
                    "y": 0.635,
                    "vx": 0.0,
                    "vy": 0.0,
                    "radius": 0.028575,
                    "is_cue_ball": True,
                    "confidence": 0.95,
                },
                {
                    "id": "ball_1",
                    "x": 1.905,
                    "y": 0.635,
                    "vx": 0.0,
                    "vy": 0.0,
                    "radius": 0.028575,
                    "number": 1,
                    "confidence": 0.90,
                },
            ],
            "cue": {
                "tip_x": 0.5,
                "tip_y": 0.635,
                "angle": 0.0,
                "force": 5.0,
                "is_visible": True,
                "confidence": 0.85,
            },
        }

        # Mock API interface to capture calls
        api_calls = []

        def mock_send_state_update(state_data):
            api_calls.append(state_data)

        self.api_interface.send_state_update = mock_send_state_update

        # Process vision data
        self.vision_interface.receive_detection_data(detection_data)

        # Allow event processing
        time.sleep(0.2)

        # Verify API was called with state update
        assert len(api_calls) > 0
        state_data = api_calls[0]
        assert "balls" in state_data
        assert len(state_data["balls"]) == 2

    def test_module_coordination_lifecycle(self):
        """Test complete module lifecycle coordination."""
        # Simulate module initialization
        self.event_manager.coordinate_module_lifecycle(
            "initialize",
            "test_module",
            version="1.0.0",
            capabilities=["detection", "tracking"],
        )

        time.sleep(0.1)

        # Verify module was registered
        status = self.integrator.get_module_status("test_module")
        assert status is not None

        # Simulate configuration change
        self.event_manager.exchange_config_with_module(
            "test_module", {"setting1": "value1", "setting2": 42}
        )

        time.sleep(0.1)

        # Simulate module shutdown
        self.event_manager.coordinate_module_lifecycle("shutdown", "test_module")

        time.sleep(0.1)

    def test_error_handling_and_recovery(self):
        """Test error handling and recovery mechanisms."""
        # Create invalid detection data
        invalid_data = {
            "balls": [
                {
                    "id": "invalid",
                    # Missing required position data
                    "confidence": 0.5,
                }
            ]
        }

        # Process invalid data - should not crash
        try:
            self.vision_interface.receive_detection_data(invalid_data)
            time.sleep(0.1)
        except Exception as e:
            pytest.fail(f"System should handle invalid data gracefully: {e}")

        # Verify error was logged in statistics
        self.integrator.get_integration_statistics()
        # System should continue operating despite errors

    def test_concurrent_operations(self):
        """Test concurrent module operations."""
        results = []

        def worker_thread(thread_id):
            """Worker thread function."""
            for i in range(10):
                detection_data = {
                    "balls": [
                        {
                            "id": f"ball_{thread_id}_{i}",
                            "x": 0.5 + i * 0.01,
                            "y": 0.5,
                            "confidence": 0.9,
                        }
                    ]
                }
                self.vision_interface.receive_detection_data(detection_data)
                time.sleep(0.01)
            results.append(thread_id)

        # Start multiple worker threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=worker_thread, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join(timeout=5.0)

        # Verify all threads completed
        assert len(results) == 3

        # Allow final event processing
        time.sleep(0.2)

        # System should still be operational
        stats = self.integrator.get_integration_statistics()
        assert stats["messages_received"] > 0

    def teardown_method(self):
        """Cleanup complete integration environment."""
        self.integrator.shutdown_integration()
        self.event_manager.stop_enhanced_processing()


# Performance and stress tests


class TestPerformance:
    """Performance and stress tests for module communication."""

    def test_high_frequency_events(self):
        """Test handling of high-frequency events."""
        manager = EventManager(enable_logging=False)

        events_received = 0

        def fast_callback(event_type: str, data: dict[str, Any]):
            nonlocal events_received
            events_received += 1

        manager.subscribe_to_events("fast_event", fast_callback)

        # Emit many events quickly
        start_time = time.time()
        num_events = 1000

        for i in range(num_events):
            manager.emit_event("fast_event", {"count": i})

        # Wait for processing
        time.sleep(1.0)
        end_time = time.time()

        # Verify performance
        processing_time = end_time - start_time
        events_per_second = events_received / processing_time

        assert events_received > 0
        assert events_per_second > 100  # Should handle at least 100 events/sec

        manager.stop_enhanced_processing()

    def test_large_event_data(self):
        """Test handling of large event data."""
        manager = EventManager(enable_logging=False)

        large_data_received = False

        def large_data_callback(event_type: str, data: dict[str, Any]):
            nonlocal large_data_received
            if len(str(data)) > 10000:  # Large data
                large_data_received = True

        manager.subscribe_to_events("large_event", large_data_callback)

        # Create large event data
        large_data = {
            "balls": [
                {
                    "id": f"ball_{i}",
                    "x": i * 0.01,
                    "y": i * 0.01,
                    "trajectory": [{"x": j, "y": j} for j in range(100)],
                }
                for i in range(100)
            ]
        }

        manager.emit_event("large_event", large_data)
        time.sleep(0.5)

        assert large_data_received

        manager.stop_enhanced_processing()

    def test_memory_usage_stability(self):
        """Test memory usage stability over time."""
        manager = EventManager(max_history_size=100, enable_logging=False)

        # Generate continuous events to test memory stability
        for cycle in range(10):
            for i in range(50):
                manager.emit_enhanced_event(
                    EventType.BALL_MOVED,
                    {"ball_id": f"ball_{i}", "cycle": cycle},
                    "test",
                )
            time.sleep(0.1)

        # History should be limited by max_history_size
        assert len(manager.event_history) <= 100

        manager.stop_enhanced_processing()


if __name__ == "__main__":
    # Run specific test classes or methods
    pytest.main([__file__, "-v"])
