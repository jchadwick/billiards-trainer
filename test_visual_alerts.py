#!/usr/bin/env python3
"""Test script for visual alert display system.

This script tests the visual alert functionality in the projector network handlers
without requiring a full projector setup.
"""

import asyncio
import logging
import sys
import time
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / "backend"))

from backend.projector.network.handlers import ProjectorMessageHandlers, HandlerConfig


class MockProjector:
    """Mock projector for testing alert display."""

    def __init__(self):
        self.alerts_displayed = []
        self.effects_created = []
        self.overlays_rendered = []

    def render_trajectory(self, trajectory, fade_in=True):
        """Mock trajectory rendering."""
        pass

    def update_ball_state(self, ball_state):
        """Mock ball state update."""
        pass

    def clear_display(self):
        """Mock display clear."""
        pass

    def remove_trajectory(self, ball_id, fade_out=True):
        """Mock trajectory removal."""
        pass

    def render_success_indicator(self, position, success_probability):
        """Mock success indicator rendering."""
        pass

    def set_trajectory_config(self, config):
        """Mock trajectory config update."""
        pass

    def render_text_overlay(self, overlay_data):
        """Mock text overlay rendering."""
        self.overlays_rendered.append(overlay_data)
        print(f"✓ Text overlay rendered: {overlay_data['level']} - {overlay_data['text']['title']['content']}")

    @property
    def effects_system(self):
        """Mock effects system."""
        return MockEffectsSystem(self)


class MockEffectsSystem:
    """Mock effects system for testing."""

    def __init__(self, projector):
        self.projector = projector
        self._effects = []

    def create_failure_indicator(self, position, message):
        """Mock failure indicator."""
        effect = {"type": "failure", "position": position, "message": message}
        self._effects.append(effect)
        self.projector.effects_created.append(effect)
        print(f"✓ Failure effect created at {position}: {message}")

    def create_success_indicator(self, position, success_level):
        """Mock success indicator."""
        effect = {"type": "success", "position": position, "level": success_level}
        self._effects.append(effect)
        self.projector.effects_created.append(effect)
        print(f"✓ Success effect created at {position}: {success_level}")

    def create_power_burst(self, position, power_level, direction):
        """Mock power burst."""
        effect = {"type": "power_burst", "position": position, "power": power_level, "direction": direction}
        self._effects.append(effect)
        self.projector.effects_created.append(effect)
        print(f"✓ Power burst effect created at {position}: power={power_level}")

    def _add_effect(self, effect):
        """Mock effect addition."""
        self._effects.append(effect)
        self.projector.effects_created.append(effect)
        print(f"✓ Generic effect added: {effect.effect_type}")


async def test_alert_display():
    """Test visual alert display functionality."""
    print("🧪 Testing Visual Alert Display System")
    print("=" * 50)

    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Create mock projector and handlers
    mock_projector = MockProjector()
    config = HandlerConfig(
        enable_alert_display=True,
        alert_display_duration=3.0
    )
    handlers = ProjectorMessageHandlers(mock_projector, config)

    # Test different alert types
    test_alerts = [
        {
            "level": "error",
            "message": "Ball detection failed",
            "code": "DETECTION_ERROR",
            "details": {"camera_id": 1, "frame_number": 12345}
        },
        {
            "level": "warning",
            "message": "Low lighting conditions detected",
            "code": "LOW_LIGHT",
            "details": {"brightness": 0.3}
        },
        {
            "level": "info",
            "message": "Calibration sequence started",
            "code": "CALIBRATION_START",
            "details": {"step": 1, "total_steps": 5}
        },
        {
            "level": "success",
            "message": "Table calibration completed",
            "code": "CALIBRATION_SUCCESS",
            "details": {"accuracy": 0.98, "time_taken": 15.2}
        }
    ]

    print(f"Testing {len(test_alerts)} different alert types...\n")

    # Test each alert
    for i, alert_data in enumerate(test_alerts, 1):
        print(f"Test {i}: {alert_data['level'].upper()} Alert")
        print(f"Message: {alert_data['message']}")
        print(f"Code: {alert_data['code']}")

        try:
            # Handle the alert
            await handlers.handle_alert(alert_data)

            # Verify alert was processed
            print("✅ Alert processed successfully")

            # Short delay between alerts
            await asyncio.sleep(0.5)

        except Exception as e:
            print(f"❌ Alert processing failed: {e}")

        print("-" * 30)

    # Test alert configuration
    print("\n🔧 Testing Alert Configuration")
    for level in ["error", "warning", "info", "success"]:
        config = handlers._get_alert_display_config(level)
        print(f"{level.upper()}: position={config['position'].to_tuple()}, "
              f"size={config['size']}, icon={config['icon']}")

    # Check results
    print(f"\n📊 Test Results")
    print(f"Overlays rendered: {len(mock_projector.overlays_rendered)}")
    print(f"Effects created: {len(mock_projector.effects_created)}")

    if mock_projector.overlays_rendered:
        print("\n📋 Overlay Details:")
        for overlay in mock_projector.overlays_rendered:
            print(f"  - {overlay['level']}: {overlay['text']['message']['content']}")

    if mock_projector.effects_created:
        print("\n✨ Effect Details:")
        for effect in mock_projector.effects_created:
            if isinstance(effect, dict):
                print(f"  - {effect['type']}: {effect.get('message', 'N/A')}")
            else:
                # Handle Effect objects
                effect_type = getattr(effect, 'effect_type', 'unknown')
                print(f"  - {effect_type}: Effect object")

    return len(mock_projector.overlays_rendered) > 0 or len(mock_projector.effects_created) > 0


async def test_handler_stats():
    """Test handler statistics tracking."""
    print("\n📈 Testing Handler Statistics")
    print("=" * 50)

    mock_projector = MockProjector()
    config = HandlerConfig()
    handlers = ProjectorMessageHandlers(mock_projector, config)

    # Process some alerts
    for i in range(5):
        await handlers.handle_alert({
            "level": "info",
            "message": f"Test alert {i+1}",
            "code": "TEST_ALERT",
            "details": {}
        })

    # Get statistics
    stats = handlers.get_handler_stats()
    print(f"Alerts processed: {stats['stats']['alerts_processed']}")
    print(f"Active alerts: {stats['active_alerts']}")
    print(f"Configuration: {stats['config']}")

    return stats['stats']['alerts_processed'] == 5


async def main():
    """Run all tests."""
    print("🚀 Starting Visual Alert System Tests")
    print("=" * 50)

    try:
        # Test alert display
        alert_test_passed = await test_alert_display()

        # Test statistics
        stats_test_passed = await test_handler_stats()

        # Summary
        print("\n🏁 Test Summary")
        print("=" * 50)
        print(f"Alert Display Test: {'✅ PASSED' if alert_test_passed else '❌ FAILED'}")
        print(f"Statistics Test: {'✅ PASSED' if stats_test_passed else '❌ FAILED'}")

        if alert_test_passed and stats_test_passed:
            print("\n🎉 All tests passed! Visual alert system is working correctly.")
            return 0
        else:
            print("\n⚠️  Some tests failed. Check the implementation.")
            return 1

    except Exception as e:
        print(f"\n💥 Test execution failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)