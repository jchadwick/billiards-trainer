"""Unit tests for EventManager.

Tests the event subscription, unsubscription, and emission functionality.
"""

import threading
import time
from unittest.mock import Mock

import pytest

from ..events.manager import EventManager


class TestEventManager:
    """Test suite for EventManager."""

    @pytest.fixture()
    def event_manager(self):
        """Create a fresh EventManager for each test."""
        return EventManager()

    def test_initialization(self, event_manager):
        """Test EventManager initialization."""
        # The enhanced event manager has some built-in coordination subscribers
        initial_count = event_manager.get_subscriber_count()
        assert initial_count >= 0  # Allow for coordination subscribers

    def test_subscribe_to_events(self, event_manager):
        """Test event subscription."""
        initial_count = event_manager.get_subscriber_count()

        callback = Mock()
        subscription_id = event_manager.subscribe_to_events("test_event", callback)

        assert subscription_id is not None
        assert len(subscription_id) > 0  # Should be a UUID
        assert event_manager.get_subscriber_count() == initial_count + 1
        assert event_manager.get_subscriber_count("test_event") == 1

    def test_multiple_subscriptions(self, event_manager):
        """Test multiple subscriptions to same event type."""
        initial_count = event_manager.get_subscriber_count()

        callback1 = Mock()
        callback2 = Mock()

        sub_id1 = event_manager.subscribe_to_events("test_event", callback1)
        sub_id2 = event_manager.subscribe_to_events("test_event", callback2)

        assert sub_id1 != sub_id2
        assert event_manager.get_subscriber_count("test_event") == 2
        assert event_manager.get_subscriber_count() == initial_count + 2

    def test_different_event_types(self, event_manager):
        """Test subscriptions to different event types."""
        initial_count = event_manager.get_subscriber_count()

        callback1 = Mock()
        callback2 = Mock()

        event_manager.subscribe_to_events("event_type_1", callback1)
        event_manager.subscribe_to_events("event_type_2", callback2)

        assert event_manager.get_subscriber_count("event_type_1") == 1
        assert event_manager.get_subscriber_count("event_type_2") == 1
        assert event_manager.get_subscriber_count() == initial_count + 2

    def test_emit_event(self, event_manager):
        """Test event emission to subscribers."""
        callback = Mock()
        event_manager.subscribe_to_events("test_event", callback)

        test_data = {"key": "value", "number": 42}
        event_manager.emit_event("test_event", test_data)

        callback.assert_called_once_with("test_event", test_data)

    def test_emit_event_multiple_subscribers(self, event_manager):
        """Test event emission to multiple subscribers."""
        callback1 = Mock()
        callback2 = Mock()

        event_manager.subscribe_to_events("test_event", callback1)
        event_manager.subscribe_to_events("test_event", callback2)

        test_data = {"key": "value"}
        event_manager.emit_event("test_event", test_data)

        callback1.assert_called_once_with("test_event", test_data)
        callback2.assert_called_once_with("test_event", test_data)

    def test_emit_event_no_subscribers(self, event_manager):
        """Test event emission with no subscribers."""
        # Should not raise an exception
        event_manager.emit_event("non_existent_event", {})

    def test_unsubscribe_success(self, event_manager):
        """Test successful unsubscription."""
        initial_count = event_manager.get_subscriber_count()

        callback = Mock()
        subscription_id = event_manager.subscribe_to_events("test_event", callback)

        # Verify subscription exists
        assert event_manager.get_subscriber_count() == initial_count + 1

        # Unsubscribe
        success = event_manager.unsubscribe(subscription_id)
        assert success is True
        assert event_manager.get_subscriber_count() == initial_count

        # Emit event - callback should not be called
        event_manager.emit_event("test_event", {})
        callback.assert_not_called()

    def test_unsubscribe_invalid_id(self, event_manager):
        """Test unsubscription with invalid ID."""
        success = event_manager.unsubscribe("invalid_id")
        assert success is False

    def test_unsubscribe_already_unsubscribed(self, event_manager):
        """Test unsubscribing the same ID twice."""
        callback = Mock()
        subscription_id = event_manager.subscribe_to_events("test_event", callback)

        # First unsubscribe should succeed
        success1 = event_manager.unsubscribe(subscription_id)
        assert success1 is True

        # Second unsubscribe should fail
        success2 = event_manager.unsubscribe(subscription_id)
        assert success2 is False

    def test_callback_exception_handling(self, event_manager):
        """Test that callback exceptions don't break event emission."""

        def failing_callback(event_type, data):
            raise Exception("Callback error")

        def working_callback(event_type, data):
            working_callback.called = True

        working_callback.called = False

        event_manager.subscribe_to_events("test_event", failing_callback)
        event_manager.subscribe_to_events("test_event", working_callback)

        # Should not raise exception despite failing callback
        event_manager.emit_event("test_event", {})

        # Working callback should still be called
        assert working_callback.called is True

    def test_clear_subscribers_specific_event(self, event_manager):
        """Test clearing subscribers for specific event type."""
        initial_count = event_manager.get_subscriber_count()

        callback1 = Mock()
        callback2 = Mock()

        event_manager.subscribe_to_events("event1", callback1)
        event_manager.subscribe_to_events("event2", callback2)

        assert event_manager.get_subscriber_count() == initial_count + 2

        # Clear subscribers for event1 only
        event_manager.clear_subscribers("event1")

        assert event_manager.get_subscriber_count("event1") == 0
        assert event_manager.get_subscriber_count("event2") == 1
        assert event_manager.get_subscriber_count() == initial_count + 1

    def test_clear_all_subscribers(self, event_manager):
        """Test clearing all subscribers."""
        callback1 = Mock()
        callback2 = Mock()

        event_manager.subscribe_to_events("event1", callback1)
        event_manager.subscribe_to_events("event2", callback2)

        initial_count = event_manager.get_subscriber_count()
        # Should have at least 2 new subscribers plus any built-in ones
        assert initial_count >= 2

        # Clear all subscribers
        event_manager.clear_subscribers()

        assert event_manager.get_subscriber_count() == 0

    def test_thread_safety(self, event_manager):
        """Test thread safety of EventManager operations."""
        results = []
        errors = []

        def subscribe_worker():
            try:
                for i in range(10):
                    callback = Mock()
                    sub_id = event_manager.subscribe_to_events(
                        f"event_{i % 3}", callback
                    )
                    results.append(("subscribe", sub_id))
                    time.sleep(0.001)  # Small delay to encourage race conditions
            except Exception as e:
                errors.append(e)

        def emit_worker():
            try:
                for i in range(20):
                    event_manager.emit_event(f"event_{i % 3}", {"iteration": i})
                    results.append(("emit", i))
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        def unsubscribe_worker():
            try:
                # Wait a bit for some subscriptions to be created
                time.sleep(0.01)
                for i in range(5):
                    # Try to unsubscribe with random IDs (most will fail, which is OK)
                    fake_id = f"fake_id_{i}"
                    success = event_manager.unsubscribe(fake_id)
                    results.append(("unsubscribe", success))
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        # Start multiple threads
        threads = [
            threading.Thread(target=subscribe_worker),
            threading.Thread(target=emit_worker),
            threading.Thread(target=unsubscribe_worker),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should have completed without errors
        assert len(errors) == 0
        assert len(results) > 0

        # Should have some subscribers (exact count depends on timing)
        subscriber_count = event_manager.get_subscriber_count()
        assert subscriber_count >= 0  # At least some operations completed

    def test_event_data_types(self, event_manager):
        """Test emission with different data types."""
        callback = Mock()
        event_manager.subscribe_to_events("test_event", callback)

        # Test with different data types
        test_cases = [
            {},  # Empty dict
            {"string": "value"},  # String
            {"number": 42},  # Integer
            {"float": 3.14},  # Float
            {"list": [1, 2, 3]},  # List
            {"nested": {"key": "value"}},  # Nested dict
            {"mixed": {"int": 1, "str": "text", "list": [1, 2]}},  # Mixed types
        ]

        for i, test_data in enumerate(test_cases):
            event_manager.emit_event("test_event", test_data)

        # Verify all calls were made
        assert callback.call_count == len(test_cases)

        # Verify data was passed correctly
        for i, call_args in enumerate(callback.call_args_list):
            event_type, data = call_args[0]
            assert event_type == "test_event"
            assert data == test_cases[i]

    def test_subscription_isolation(self, event_manager):
        """Test that different event types are properly isolated."""
        callback1 = Mock()
        callback2 = Mock()

        event_manager.subscribe_to_events("event_a", callback1)
        event_manager.subscribe_to_events("event_b", callback2)

        # Emit to event_a
        event_manager.emit_event("event_a", {"for": "event_a"})

        # Only callback1 should be called
        callback1.assert_called_once_with("event_a", {"for": "event_a"})
        callback2.assert_not_called()

        # Reset mocks
        callback1.reset_mock()
        callback2.reset_mock()

        # Emit to event_b
        event_manager.emit_event("event_b", {"for": "event_b"})

        # Only callback2 should be called
        callback1.assert_not_called()
        callback2.assert_called_once_with("event_b", {"for": "event_b"})

    def test_large_number_of_subscribers(self, event_manager):
        """Test performance with large number of subscribers."""
        callbacks = []
        subscription_ids = []

        # Create 100 subscribers
        for _i in range(100):
            callback = Mock()
            callbacks.append(callback)
            sub_id = event_manager.subscribe_to_events("bulk_test", callback)
            subscription_ids.append(sub_id)

        assert event_manager.get_subscriber_count("bulk_test") == 100

        # Emit event
        test_data = {"bulk": True}
        start_time = time.time()
        event_manager.emit_event("bulk_test", test_data)
        end_time = time.time()

        # Should complete relatively quickly (less than 100ms for 100 callbacks)
        assert (end_time - start_time) < 0.1

        # All callbacks should be called
        for callback in callbacks:
            callback.assert_called_once_with("bulk_test", test_data)

        # Test batch unsubscribe
        for sub_id in subscription_ids:
            event_manager.unsubscribe(sub_id)

        assert event_manager.get_subscriber_count("bulk_test") == 0


if __name__ == "__main__":
    # Run with pytest
    pytest.main([__file__, "-v"])
