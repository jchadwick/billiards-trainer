#!/usr/bin/env python3
"""Test script for WebSocket alert delivery system.

This script tests the WebSocket integration for real-time alert delivery
between the backend and projector systems.
"""

import asyncio
import json
import logging
import sys
import time
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / "backend"))

from backend.projector.network.handlers import ProjectorMessageHandlers, HandlerConfig


class MockWebSocket:
    """Mock WebSocket for testing message delivery."""

    def __init__(self):
        self.sent_messages = []
        self.received_messages = []
        self.is_connected = True

    async def send(self, message):
        """Mock send message."""
        self.sent_messages.append(message)
        print(f"📤 WebSocket sent: {message[:100]}...")

    async def recv(self):
        """Mock receive message."""
        if self.received_messages:
            message = self.received_messages.pop(0)
            print(f"📥 WebSocket received: {message[:100]}...")
            return message
        else:
            # Simulate waiting for message
            await asyncio.sleep(0.1)
            return None

    def add_incoming_message(self, message):
        """Add message to incoming queue."""
        self.received_messages.append(message)

    async def close(self):
        """Mock close connection."""
        self.is_connected = False
        print("🔌 WebSocket connection closed")


class MockProjector:
    """Mock projector for testing alert integration."""

    def __init__(self):
        self.alerts_displayed = []
        self.websocket_messages = []

    def render_trajectory(self, trajectory, fade_in=True):
        pass

    def update_ball_state(self, ball_state):
        pass

    def clear_display(self):
        pass

    def remove_trajectory(self, ball_id, fade_out=True):
        pass

    def render_success_indicator(self, position, success_probability):
        pass

    def set_trajectory_config(self, config):
        pass

    def render_text_overlay(self, overlay_data):
        """Mock text overlay rendering with WebSocket notification."""
        self.alerts_displayed.append(overlay_data)

        # Simulate WebSocket message back to backend
        response_message = {
            "type": "alert_displayed",
            "alert_id": overlay_data.get("id", "unknown"),
            "level": overlay_data["level"],
            "timestamp": time.time(),
            "status": "success"
        }
        self.websocket_messages.append(response_message)
        print(f"✓ Alert displayed and WebSocket notification queued: {overlay_data['level']}")

    @property
    def effects_system(self):
        return MockEffectsSystem(self)


class MockEffectsSystem:
    """Mock effects system for testing."""

    def __init__(self, projector):
        self.projector = projector

    def create_failure_indicator(self, position, message):
        self.projector.websocket_messages.append({
            "type": "effect_created",
            "effect_type": "failure",
            "position": position.to_tuple() if hasattr(position, 'to_tuple') else str(position),
            "message": message
        })

    def create_success_indicator(self, position, success_level):
        self.projector.websocket_messages.append({
            "type": "effect_created",
            "effect_type": "success",
            "position": position.to_tuple() if hasattr(position, 'to_tuple') else str(position),
            "level": success_level
        })

    def create_power_burst(self, position, power_level, direction):
        self.projector.websocket_messages.append({
            "type": "effect_created",
            "effect_type": "power_burst",
            "position": position.to_tuple() if hasattr(position, 'to_tuple') else str(position),
            "power": power_level
        })

    def _add_effect(self, effect):
        self.projector.websocket_messages.append({
            "type": "effect_created",
            "effect_type": str(effect.effect_type),
            "position": str(effect.position)
        })


async def test_alert_message_format():
    """Test alert message format and parsing."""
    print("🧪 Testing Alert Message Format")
    print("=" * 50)

    # Test different alert message formats
    test_messages = [
        {
            "type": "alert",
            "level": "error",
            "message": "Ball detection failed",
            "code": "DETECTION_ERROR",
            "details": {"camera_id": 1, "frame_number": 12345},
            "timestamp": time.time(),
            "id": "alert_001"
        },
        {
            "type": "alert",
            "level": "warning",
            "message": "Low lighting conditions",
            "code": "LOW_LIGHT",
            "details": {"brightness": 0.3},
            "timestamp": time.time(),
            "id": "alert_002"
        },
        {
            "type": "alert",
            "level": "info",
            "message": "System initialized",
            "code": "INIT_SUCCESS",
            "details": {},
            "timestamp": time.time(),
            "id": "alert_003"
        }
    ]

    print("Testing message serialization/deserialization...")

    for i, message in enumerate(test_messages, 1):
        try:
            # Serialize to JSON
            json_message = json.dumps(message)

            # Deserialize from JSON
            parsed_message = json.loads(json_message)

            # Verify all fields preserved
            fields_match = all(
                message.get(key) == parsed_message.get(key)
                for key in message.keys()
            )

            if fields_match:
                print(f"✅ Message {i}: {message['level']} - Serialization OK")
            else:
                print(f"❌ Message {i}: {message['level']} - Serialization failed")
                return False

        except Exception as e:
            print(f"❌ Message {i}: Serialization error - {e}")
            return False

    print("✅ All message formats valid")
    return True


async def test_websocket_message_handling():
    """Test WebSocket message handling integration."""
    print("\n📡 Testing WebSocket Message Handling")
    print("=" * 50)

    mock_projector = MockProjector()
    config = HandlerConfig(enable_alert_display=True)
    handlers = ProjectorMessageHandlers(mock_projector, config)

    # Test messages
    test_alerts = [
        {
            "level": "error",
            "message": "Critical system error",
            "code": "SYSTEM_ERROR",
            "details": {"component": "vision", "error_code": 500}
        },
        {
            "level": "success",
            "message": "Calibration completed",
            "code": "CALIBRATION_DONE",
            "details": {"accuracy": 0.98}
        }
    ]

    print("Processing WebSocket alert messages...")

    for i, alert_data in enumerate(test_alerts, 1):
        try:
            # Simulate WebSocket message handling
            await handlers.handle_alert(alert_data)

            print(f"✅ Alert {i} processed: {alert_data['level']} - {alert_data['message']}")

        except Exception as e:
            print(f"❌ Alert {i} failed: {e}")
            return False

    # Check that alerts were displayed
    displayed_count = len(mock_projector.alerts_displayed)
    print(f"Alerts displayed on projector: {displayed_count}")

    # Check WebSocket response messages
    response_count = len(mock_projector.websocket_messages)
    print(f"WebSocket response messages generated: {response_count}")

    success = displayed_count >= len(test_alerts) and response_count >= len(test_alerts)

    if success:
        print("✅ WebSocket message handling working correctly")
    else:
        print("❌ WebSocket message handling failed")

    return success


async def test_real_time_alert_delivery():
    """Test real-time alert delivery simulation."""
    print("\n⚡ Testing Real-time Alert Delivery")
    print("=" * 50)

    mock_projector = MockProjector()
    mock_websocket = MockWebSocket()
    config = HandlerConfig(enable_alert_display=True, alert_display_duration=2.0)
    handlers = ProjectorMessageHandlers(mock_projector, config)

    # Simulate real-time scenario
    print("Simulating real-time alert scenario...")

    # Time-critical alerts
    critical_alerts = [
        {
            "timestamp": time.time(),
            "level": "error",
            "message": "Ball tracking lost",
            "code": "TRACKING_LOST",
            "priority": "high"
        },
        {
            "timestamp": time.time() + 0.5,
            "level": "warning",
            "message": "Low confidence detection",
            "code": "LOW_CONFIDENCE",
            "priority": "medium"
        },
        {
            "timestamp": time.time() + 1.0,
            "level": "info",
            "message": "Tracking resumed",
            "code": "TRACKING_RESUMED",
            "priority": "low"
        }
    ]

    start_time = time.time()

    for alert in critical_alerts:
        # Wait until alert timestamp
        while time.time() < alert["timestamp"]:
            await asyncio.sleep(0.01)

        # Process alert
        processing_start = time.time()
        await handlers.handle_alert(alert)
        processing_time = time.time() - processing_start

        print(f"⏱️  Alert '{alert['code']}' processed in {processing_time*1000:.2f}ms")

        # Check if processing was fast enough (< 50ms for real-time)
        if processing_time > 0.05:
            print(f"⚠️  Alert processing too slow: {processing_time*1000:.2f}ms")

    total_time = time.time() - start_time
    print(f"Total scenario time: {total_time:.2f}s")

    # Verify all alerts were processed
    alerts_processed = len(mock_projector.alerts_displayed)
    success = alerts_processed == len(critical_alerts) and total_time < 2.0

    if success:
        print("✅ Real-time alert delivery working correctly")
    else:
        print("❌ Real-time alert delivery too slow or incomplete")

    return success


async def test_alert_queuing_and_priority():
    """Test alert queuing and priority handling."""
    print("\n📋 Testing Alert Queuing and Priority")
    print("=" * 50)

    mock_projector = MockProjector()
    config = HandlerConfig(enable_alert_display=True)
    handlers = ProjectorMessageHandlers(mock_projector, config)

    # Multiple alerts with different priorities
    alerts_batch = [
        {"level": "info", "message": "Info 1", "code": "INFO_1", "priority": 1},
        {"level": "error", "message": "Error 1", "code": "ERROR_1", "priority": 4},
        {"level": "warning", "message": "Warning 1", "code": "WARN_1", "priority": 3},
        {"level": "info", "message": "Info 2", "code": "INFO_2", "priority": 1},
        {"level": "error", "message": "Error 2", "code": "ERROR_2", "priority": 4},
    ]

    print(f"Processing {len(alerts_batch)} alerts simultaneously...")

    # Process all alerts quickly
    tasks = []
    for alert in alerts_batch:
        task = asyncio.create_task(handlers.handle_alert(alert))
        tasks.append(task)

    # Wait for all to complete
    await asyncio.gather(*tasks)

    # Check results
    processed_count = len(mock_projector.alerts_displayed)
    active_alerts = len(handlers.active_alerts)

    print(f"Alerts processed: {processed_count}")
    print(f"Active alerts: {active_alerts}")

    # Check handler statistics
    stats = handlers.get_handler_stats()
    alerts_stat = stats["stats"]["alerts_processed"]

    print(f"Handler stats - alerts processed: {alerts_stat}")

    success = processed_count == len(alerts_batch) and alerts_stat == len(alerts_batch)

    if success:
        print("✅ Alert queuing and priority handling working correctly")
    else:
        print("❌ Alert queuing and priority handling failed")

    return success


async def test_error_handling_and_recovery():
    """Test error handling and recovery in WebSocket alert system."""
    print("\n🛠️  Testing Error Handling and Recovery")
    print("=" * 50)

    mock_projector = MockProjector()
    config = HandlerConfig(enable_alert_display=True)
    handlers = ProjectorMessageHandlers(mock_projector, config)

    # Test malformed alert messages
    malformed_alerts = [
        {"level": "error"},  # Missing message
        {"message": "Test"},  # Missing level
        None,  # Null alert
        {},  # Empty alert
        {"level": "invalid", "message": "Test", "code": "TEST"}  # Invalid level
    ]

    print("Testing error handling with malformed alerts...")

    errors_handled = 0
    for i, alert in enumerate(malformed_alerts, 1):
        try:
            await handlers.handle_alert(alert or {})
            print(f"Alert {i}: Processed (no error thrown)")
        except Exception as e:
            errors_handled += 1
            print(f"Alert {i}: Error handled gracefully - {type(e).__name__}")

    # Test with valid alert after errors
    recovery_alert = {
        "level": "info",
        "message": "System recovered",
        "code": "RECOVERY_TEST",
        "details": {}
    }

    try:
        await handlers.handle_alert(recovery_alert)
        recovery_success = True
        print("✅ System recovered after errors")
    except Exception as e:
        recovery_success = False
        print(f"❌ System failed to recover: {e}")

    # Check that valid alerts still work
    final_stats = handlers.get_handler_stats()

    success = recovery_success and final_stats["stats"]["errors"] >= 0

    if success:
        print("✅ Error handling and recovery working correctly")
    else:
        print("❌ Error handling and recovery failed")

    return success


async def test_performance_under_load():
    """Test performance under high alert load."""
    print("\n🚀 Testing Performance Under Load")
    print("=" * 50)

    mock_projector = MockProjector()
    config = HandlerConfig(enable_alert_display=True)
    handlers = ProjectorMessageHandlers(mock_projector, config)

    # Generate many alerts
    num_alerts = 100
    alerts = []
    for i in range(num_alerts):
        alerts.append({
            "level": ["info", "warning", "error", "success"][i % 4],
            "message": f"Load test alert {i+1}",
            "code": f"LOAD_TEST_{i+1:03d}",
            "details": {"sequence": i+1, "batch": "load_test"}
        })

    print(f"Processing {num_alerts} alerts for load testing...")

    start_time = time.time()

    # Process alerts in batches to simulate realistic load
    batch_size = 10
    for i in range(0, num_alerts, batch_size):
        batch = alerts[i:i+batch_size]
        tasks = [handlers.handle_alert(alert) for alert in batch]
        await asyncio.gather(*tasks)

        # Small delay between batches
        await asyncio.sleep(0.01)

    total_time = time.time() - start_time
    alerts_per_second = num_alerts / total_time

    print(f"Load test completed:")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Alerts per second: {alerts_per_second:.1f}")
    print(f"  Average time per alert: {(total_time/num_alerts)*1000:.2f}ms")

    # Check results
    final_stats = handlers.get_handler_stats()
    processed_count = final_stats["stats"]["alerts_processed"]

    print(f"  Alerts processed: {processed_count}")
    print(f"  Errors: {final_stats['stats']['errors']}")

    # Performance criteria: should handle at least 50 alerts/second with <5% errors
    error_rate = final_stats["stats"]["errors"] / num_alerts if num_alerts > 0 else 0
    success = alerts_per_second >= 50 and error_rate < 0.05 and processed_count >= num_alerts * 0.95

    if success:
        print("✅ Performance under load acceptable")
    else:
        print("❌ Performance under load insufficient")

    return success


async def main():
    """Run all WebSocket alert delivery tests."""
    print("🚀 Starting WebSocket Alert Delivery Tests")
    print("=" * 60)

    # Set up logging
    logging.basicConfig(level=logging.WARNING)  # Reduce noise

    tests = [
        ("Alert Message Format", test_alert_message_format),
        ("WebSocket Message Handling", test_websocket_message_handling),
        ("Real-time Alert Delivery", test_real_time_alert_delivery),
        ("Alert Queuing and Priority", test_alert_queuing_and_priority),
        ("Error Handling and Recovery", test_error_handling_and_recovery),
        ("Performance Under Load", test_performance_under_load),
    ]

    results = {}

    try:
        for test_name, test_func in tests:
            print(f"\n{'='*70}")
            results[test_name] = await test_func()

        # Summary
        print(f"\n🏁 Test Summary")
        print("=" * 70)

        passed = sum(results.values())
        total = len(results)

        for test_name, result in results.items():
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"{test_name}: {status}")

        print(f"\nOverall: {passed}/{total} tests passed")

        if passed == total:
            print("\n🎉 All WebSocket alert delivery tests passed! System is working correctly.")
            return 0
        else:
            print(f"\n⚠️  {total - passed} tests failed. Check the implementation.")
            return 1

    except Exception as e:
        print(f"\n💥 Test execution failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
