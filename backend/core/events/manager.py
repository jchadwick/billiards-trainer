"""Event management system for module coordination and comprehensive event handling."""

import asyncio
import contextlib
import inspect
import json
import logging
import threading
import time
import uuid
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from queue import Empty, Queue
from typing import Any, Callable, Optional, Union

logger = logging.getLogger(__name__)


class EventType(Enum):
    """System event types for module coordination."""

    # Game state events
    STATE_UPDATED = "state_updated"
    BALL_MOVED = "ball_moved"
    SHOT_DETECTED = "shot_detected"
    COLLISION_DETECTED = "collision_detected"
    BALL_POCKETED = "ball_pocketed"
    GAME_RESET = "game_reset"

    # Module coordination events
    VISION_DATA_RECEIVED = "vision_data_received"
    TRAJECTORY_CALCULATED = "trajectory_calculated"
    PROJECTION_UPDATE = "projection_update"
    CONFIG_CHANGED = "config_changed"

    # System events
    MODULE_INITIALIZED = "module_initialized"
    MODULE_SHUTDOWN = "module_shutdown"
    ERROR_OCCURRED = "error_occurred"
    CALIBRATION_UPDATED = "calibration_updated"

    # Custom events
    CUSTOM = "custom"


class EventPriority(Enum):
    """Event priority levels."""

    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class Event:
    """System event data structure."""

    id: str
    type: Union[EventType, str]
    data: dict[str, Any]
    timestamp: float
    source_module: str
    priority: EventPriority = EventPriority.NORMAL
    target_modules: Optional[list[str]] = None
    correlation_id: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert event to dictionary for serialization."""
        return {
            "id": self.id,
            "type": self.type.value if isinstance(self.type, EventType) else self.type,
            "data": self.data,
            "timestamp": self.timestamp,
            "source_module": self.source_module,
            "priority": self.priority.value,
            "target_modules": self.target_modules,
            "correlation_id": self.correlation_id,
            "metadata": self.metadata or {},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Event":
        """Create event from dictionary."""
        event_type = data["type"]
        try:
            event_type = EventType(event_type)
        except ValueError:
            pass  # Keep as string for custom events

        return cls(
            id=data["id"],
            type=event_type,
            data=data["data"],
            timestamp=data["timestamp"],
            source_module=data["source_module"],
            priority=EventPriority(data["priority"]),
            target_modules=data.get("target_modules"),
            correlation_id=data.get("correlation_id"),
            metadata=data.get("metadata", {}),
        )


class ModuleCoordinator:
    """Handles module coordination and communication (FR-CORE-051 to FR-CORE-055)."""

    def __init__(self, event_manager: "EventManager"):
        self.event_manager = event_manager
        self.module_states: dict[str, dict[str, Any]] = {}
        self.module_configs: dict[str, dict[str, Any]] = {}
        self.module_interfaces: dict[str, Any] = {}
        self.coordination_lock = threading.RLock()

        # Register for coordination events
        self._setup_coordination_handlers()

    def _setup_coordination_handlers(self):
        """Setup handlers for module coordination events."""
        self.event_manager.subscribe_to_events(
            EventType.MODULE_INITIALIZED.value, self._handle_module_initialization
        )

        self.event_manager.subscribe_to_events(
            EventType.CONFIG_CHANGED.value, self._handle_config_change
        )

    def _handle_module_initialization(self, event_type: str, data: dict[str, Any]):
        """Handle module initialization."""
        with self.coordination_lock:
            module_name = data.get("module_name")
            module_interface = data.get("interface")

            if module_name:
                self.module_interfaces[module_name] = module_interface
                self.module_states[module_name] = {
                    "status": "initialized",
                    "last_update": time.time(),
                    "capabilities": data.get("capabilities", []),
                }

                logger.info(f"Module {module_name} initialized")

    def _handle_config_change(self, event_type: str, data: dict[str, Any]):
        """Handle configuration changes."""
        with self.coordination_lock:
            module_name = data.get("module_name")
            config = data.get("config")

            if module_name and config:
                self.module_configs[module_name] = config
                logger.info(f"Configuration updated for {module_name}")

    def register_module(
        self, module_name: str, interface: Any, capabilities: list[str]
    ):
        """Register a module with the coordinator."""
        self.event_manager.emit_event(
            EventType.MODULE_INITIALIZED.value,
            {
                "module_name": module_name,
                "interface": interface,
                "capabilities": capabilities,
            },
        )

    def send_data_to_module(self, target_module: str, data_type: str, data: Any):
        """Send data to a specific module."""
        event_type_map = {
            "state_update": EventType.STATE_UPDATED.value,
            "trajectory_data": EventType.TRAJECTORY_CALCULATED.value,
            "vision_data": EventType.VISION_DATA_RECEIVED.value,
            "projection_update": EventType.PROJECTION_UPDATE.value,
        }

        event_type = event_type_map.get(data_type, EventType.CUSTOM.value)

        # Use enhanced emit method with target filtering
        self.event_manager.emit_targeted_event(
            event_type,
            {"data": data, "data_type": data_type},
            target_modules=[target_module],
        )

    def broadcast_state_update(self, state_data: dict[str, Any]):
        """Broadcast state update to all modules."""
        self.event_manager.emit_event(EventType.STATE_UPDATED.value, state_data)

    def get_module_status(self, module_name: str) -> Optional[dict[str, Any]]:
        """Get status of a specific module."""
        return self.module_states.get(module_name)

    def get_all_module_statuses(self) -> dict[str, dict[str, Any]]:
        """Get status of all modules."""
        return self.module_states.copy()


class EventManager:
    """Comprehensive thread-safe event management system for module coordination.

    Supports:
    - Event subscriptions with filtering and priorities
    - Module coordination (FR-CORE-051 to FR-CORE-055)
    - Event management (FR-CORE-056 to FR-CORE-060)
    - Event history and replay
    - Async callback support
    - Event serialization and logging
    """

    def __init__(self, max_history_size: int = 10000, enable_logging: bool = True):
        """Initialize comprehensive event manager."""
        # Legacy API compatibility
        self._subscribers: dict[str, dict[str, Callable]] = defaultdict(dict)
        self._lock = threading.RLock()

        # Enhanced event management
        self.event_history: deque = deque(maxlen=max_history_size)
        self.event_filters: dict[str, Callable[[Event], bool]] = {}
        self.enable_logging = enable_logging

        # Async support
        self.executor = ThreadPoolExecutor(
            max_workers=4, thread_name_prefix="event_handler"
        )
        self.event_queue = Queue()
        self.processing_thread: Optional[threading.Thread] = None
        self.is_running = False

        # Module coordination
        self.coordinator = ModuleCoordinator(self)

        # Statistics
        self.stats = {
            "events_emitted": 0,
            "events_processed": 0,
            "subscriptions_created": 0,
            "errors": 0,
        }

        # Start enhanced processing
        self.start_enhanced_processing()

    def subscribe_to_events(self, event_type: str, callback: Callable) -> str:
        """Subscribe to state change events.

        Args:
            event_type: Type of event to subscribe to
            callback: Function to call when event occurs

        Returns:
            Subscription ID for unsubscribing
        """
        subscription_id = str(uuid.uuid4())

        with self._lock:
            self._subscribers[event_type][subscription_id] = callback

        logger.debug(f"Subscribed to {event_type} with ID {subscription_id}")
        return subscription_id

    def unsubscribe(self, subscription_id: str) -> bool:
        """Unsubscribe from events.

        Args:
            subscription_id: ID returned from subscribe_to_events

        Returns:
            True if successfully unsubscribed
        """
        with self._lock:
            for event_type, subscribers in self._subscribers.items():
                if subscription_id in subscribers:
                    del subscribers[subscription_id]
                    logger.debug(f"Unsubscribed {subscription_id} from {event_type}")
                    return True

        logger.warning(f"Subscription ID {subscription_id} not found")
        return False

    def emit_event(self, event_type: str, data: dict[str, Any]) -> None:
        """Emit system event to all subscribers.

        Args:
            event_type: Type of event being emitted
            data: Event data to pass to callbacks
        """
        with self._lock:
            subscribers = self._subscribers[event_type].copy()

        if not subscribers:
            logger.debug(f"No subscribers for event type: {event_type}")
            return

        logger.debug(f"Emitting {event_type} to {len(subscribers)} subscribers")

        # Call subscribers outside lock to prevent deadlocks
        for subscription_id, callback in subscribers.items():
            try:
                # Check if callback is async and handle appropriately
                if inspect.iscoroutinefunction(callback):
                    # Try to get running event loop
                    try:
                        loop = asyncio.get_running_loop()
                        # Create task in the running loop
                        loop.create_task(callback(event_type, data))
                    except RuntimeError:
                        # No running event loop - run in thread pool
                        import threading

                        def run_async():
                            asyncio.run(callback(event_type, data))

                        thread = threading.Thread(target=run_async, daemon=True)
                        thread.start()
                else:
                    # Call sync callback normally
                    callback(event_type, data)
            except Exception as e:
                logger.error(f"Error in event callback {subscription_id}: {e}")
                # Consider removing failing callbacks

    def get_subscriber_count(self, event_type: str = None) -> int:
        """Get number of subscribers for event type.

        Args:
            event_type: Specific event type, or None for total count

        Returns:
            Number of subscribers
        """
        with self._lock:
            if event_type:
                return len(self._subscribers[event_type])
            else:
                return sum(len(subs) for subs in self._subscribers.values())

    def clear_subscribers(self, event_type: str = None) -> None:
        """Clear subscribers for event type or all events.

        Args:
            event_type: Specific event type, or None for all events
        """
        with self._lock:
            if event_type:
                self._subscribers[event_type].clear()
                logger.info(f"Cleared subscribers for {event_type}")
            else:
                self._subscribers.clear()
                logger.info("Cleared all event subscribers")

    # Enhanced event management methods (FR-CORE-056 to FR-CORE-060)

    def start_enhanced_processing(self):
        """Start enhanced event processing system."""
        if self.is_running:
            return

        self.is_running = True
        self.processing_thread = threading.Thread(
            target=self._process_enhanced_events,
            name="enhanced_event_processor",
            daemon=True,
        )
        self.processing_thread.start()

        if self.enable_logging:
            logger.info("Enhanced event manager started")

    def stop_enhanced_processing(self):
        """Stop enhanced event processing system."""
        self.is_running = False

        # Signal processing thread to stop
        with contextlib.suppress(Exception):
            self.event_queue.put(None, timeout=1.0)

        if self.processing_thread:
            self.processing_thread.join(timeout=5.0)

        self.executor.shutdown(wait=True)

        if self.enable_logging:
            logger.info("Enhanced event manager stopped")

    def _process_enhanced_events(self):
        """Enhanced event processing loop."""
        while self.is_running:
            try:
                # Get event from queue (blocking)
                event = self.event_queue.get(timeout=1.0)

                # Check for shutdown signal
                if event is None:
                    break

                self._handle_enhanced_event(event)
                self.stats["events_processed"] += 1

            except Empty:
                continue
            except Exception as e:
                self.stats["errors"] += 1
                if self.enable_logging:
                    logger.error(f"Error processing enhanced event: {e}")

    def _handle_enhanced_event(self, event: Event):
        """Handle enhanced event with filtering and targeting."""
        with self._lock:
            # Add to history (FR-CORE-058)
            self.event_history.append(event)

            # Apply global filters
            for filter_name, filter_func in self.event_filters.items():
                try:
                    if not filter_func(event):
                        return  # Event filtered out
                except Exception as e:
                    logger.error(f"Error in event filter {filter_name}: {e}")

            # Get event type as string for legacy compatibility
            event_type_str = (
                event.type.value
                if isinstance(event.type, EventType)
                else str(event.type)
            )

            # Handle legacy subscribers
            subscribers = self._subscribers.get(event_type_str, {}).copy()

            # Call legacy subscribers outside lock to prevent deadlocks
            for subscription_id, callback in subscribers.items():
                try:
                    # Check target modules filter
                    if event.target_modules:
                        # For legacy callbacks, we can't determine module name
                        # so we skip targeting for now
                        continue

                    self.executor.submit(callback, event_type_str, event.data)

                except Exception as e:
                    self.stats["errors"] += 1
                    if self.enable_logging:
                        logger.error(
                            f"Error in legacy event handler {subscription_id}: {e}"
                        )

    def emit_enhanced_event(
        self,
        event_type: Union[EventType, str],
        data: dict[str, Any],
        source_module: str = "unknown",
        target_modules: Optional[list[str]] = None,
        priority: Union[EventPriority, str] = EventPriority.NORMAL,
        correlation_id: Optional[str] = None,
    ) -> str:
        """Emit enhanced system event (FR-CORE-056)."""
        event_id = str(uuid.uuid4())

        # Handle string priority conversion
        if isinstance(priority, str):
            try:
                priority = EventPriority[priority.upper()]
            except KeyError:
                priority = EventPriority.NORMAL

        event = Event(
            id=event_id,
            type=event_type,
            data=data,
            timestamp=time.time(),
            source_module=source_module,
            priority=priority,
            target_modules=target_modules,
            correlation_id=correlation_id,
        )

        # Add to processing queue
        self.event_queue.put(event)
        self.stats["events_emitted"] += 1

        if self.enable_logging:
            event_type_str = (
                event_type.value
                if isinstance(event_type, EventType)
                else str(event_type)
            )
            logger.debug(f"Enhanced event emitted: {event_id} ({event_type_str})")

        return event_id

    def emit_targeted_event(
        self,
        event_type: str,
        data: dict[str, Any],
        target_modules: list[str],
        source_module: str = "unknown",
    ) -> str:
        """Emit event targeted to specific modules."""
        return self.emit_enhanced_event(
            event_type, data, source_module=source_module, target_modules=target_modules
        )

    def get_event_history(
        self,
        event_type: Optional[Union[EventType, str]] = None,
        source_module: Optional[str] = None,
        time_range: Optional[tuple[float, float]] = None,
        limit: Optional[int] = None,
    ) -> list[Event]:
        """Get event history with filtering (FR-CORE-058)."""
        with self._lock:
            events = list(self.event_history)

        # Apply filters
        if event_type:
            if isinstance(event_type, EventType):
                events = [e for e in events if e.type == event_type]
            else:
                events = [
                    e
                    for e in events
                    if (isinstance(e.type, EventType) and e.type.value == event_type)
                    or str(e.type) == event_type
                ]

        if source_module:
            events = [e for e in events if e.source_module == source_module]

        if time_range:
            start_time, end_time = time_range
            events = [e for e in events if start_time <= e.timestamp <= end_time]

        # Sort by timestamp (newest first)
        events.sort(key=lambda e: e.timestamp, reverse=True)

        # Apply limit
        if limit:
            events = events[:limit]

        return events

    def replay_events(
        self, events: list[Event], target_modules: Optional[list[str]] = None
    ) -> int:
        """Replay events (FR-CORE-058)."""
        replayed_count = 0

        for event in events:
            # Create replay event
            self.emit_enhanced_event(
                event.type,
                event.data.copy(),
                source_module=f"replay_{event.source_module}",
                target_modules=target_modules or event.target_modules,
                correlation_id=event.id,  # Link to original event
            )
            replayed_count += 1

        return replayed_count

    def add_event_filter(self, name: str, filter_func: Callable[[Event], bool]):
        """Add global event filter (FR-CORE-059)."""
        self.event_filters[name] = filter_func
        logger.debug(f"Event filter '{name}' added")

    def remove_event_filter(self, name: str) -> bool:
        """Remove global event filter."""
        removed = self.event_filters.pop(name, None) is not None
        if removed:
            logger.debug(f"Event filter '{name}' removed")
        return removed

    def serialize_event(self, event: Event) -> str:
        """Serialize event to JSON string."""
        return json.dumps(event.to_dict(), indent=2)

    def deserialize_event(self, json_str: str) -> Event:
        """Deserialize event from JSON string."""
        data = json.loads(json_str)
        return Event.from_dict(data)

    def get_statistics(self) -> dict[str, Any]:
        """Get event system statistics."""
        with self._lock:
            return {
                **self.stats.copy(),
                "subscription_count": sum(
                    len(subs) for subs in self._subscribers.values()
                ),
                "event_types_subscribed": len(self._subscribers),
                "history_size": len(self.event_history),
                "queue_size": (
                    self.event_queue.qsize()
                    if hasattr(self.event_queue, "qsize")
                    else 0
                ),
                "is_running": self.is_running,
                "filters_active": len(self.event_filters),
            }

    # Module coordination interface methods (FR-CORE-051 to FR-CORE-055)

    def receive_vision_data(self, detection_data: dict[str, Any]):
        """Receive detection data from Vision module (FR-CORE-051)."""
        self.emit_enhanced_event(
            EventType.VISION_DATA_RECEIVED,
            {"detection_data": detection_data},
            source_module="vision",
            priority=EventPriority.HIGH,
        )

    def send_state_update_to_api(self, state_data: dict[str, Any]):
        """Send state updates to API module (FR-CORE-052)."""
        self.emit_enhanced_event(
            EventType.STATE_UPDATED,
            state_data,
            source_module="core",
            target_modules=["api"],
            priority=EventPriority.HIGH,
        )

    def provide_trajectory_to_projector(self, trajectory_data: dict[str, Any]):
        """Provide trajectory data to Projector module (FR-CORE-053)."""
        self.emit_enhanced_event(
            EventType.TRAJECTORY_CALCULATED,
            {"trajectory_data": trajectory_data},
            source_module="core",
            target_modules=["projector"],
            priority=EventPriority.NORMAL,
        )

    def exchange_config_with_module(
        self, module_name: str, config_data: dict[str, Any]
    ):
        """Exchange configuration with modules (FR-CORE-054)."""
        self.emit_enhanced_event(
            EventType.CONFIG_CHANGED,
            {"module_name": module_name, "config": config_data},
            source_module="config",
            target_modules=[module_name],
        )

    def coordinate_module_lifecycle(self, action: str, module_name: str, **kwargs):
        """Coordinate module initialization and shutdown (FR-CORE-055)."""
        if action == "initialize":
            event_type = EventType.MODULE_INITIALIZED
        elif action == "shutdown":
            event_type = EventType.MODULE_SHUTDOWN
        else:
            event_type = EventType.CUSTOM

        self.emit_enhanced_event(
            event_type,
            {"action": action, "module_name": module_name, **kwargs},
            source_module="coordinator",
            priority=EventPriority.HIGH,
        )

    def send_config_update(self, module_name: str, config: dict[str, Any]) -> None:
        """Send configuration update to module."""
        self.emit_enhanced_event(
            EventType.CONFIG_CHANGED,
            {"module": module_name, "config": config, "timestamp": time.time()},
        )

    def send_projector_command(self, command: dict[str, Any]) -> None:
        """Send command to projector module."""
        self.emit_enhanced_event(
            EventType.CUSTOM,
            {"command": command, "timestamp": time.time()},
            source_module="core",
            target_modules=["projector"],
        )

    def send_api_message(self, message: dict[str, Any]) -> None:
        """Send message through API interface."""
        self.emit_enhanced_event(
            EventType.CUSTOM,
            {"message": message, "timestamp": time.time()},
            source_module="core",
            target_modules=["api"],
        )

    # Cleanup
    def __del__(self):
        """Cleanup resources."""
        with contextlib.suppress(Exception):
            self.stop_enhanced_processing()
