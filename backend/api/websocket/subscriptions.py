"""Advanced subscription management system for selective WebSocket data streaming."""

import asyncio
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Optional, Union

from .manager import StreamType
from .schemas import QualityLevel

logger = logging.getLogger(__name__)


class FilterOperator(Enum):
    """Filter operators for subscription conditions."""

    EQUALS = "eq"
    NOT_EQUALS = "ne"
    GREATER_THAN = "gt"
    GREATER_EQUAL = "ge"
    LESS_THAN = "lt"
    LESS_EQUAL = "le"
    IN = "in"
    NOT_IN = "not_in"
    CONTAINS = "contains"
    MATCHES = "matches"  # Regex pattern


@dataclass
class FilterCondition:
    """Individual filter condition."""

    field_path: str  # e.g., "balls.0.position.x" or "cue.angle"
    operator: FilterOperator
    value: Any
    description: Optional[str] = None


@dataclass
class StreamSubscription:
    """Comprehensive stream subscription configuration."""

    client_id: str
    stream_type: StreamType
    active: bool = True

    # Quality and performance settings
    quality_level: QualityLevel = QualityLevel.AUTO
    max_fps: Optional[float] = None
    min_fps: Optional[float] = None
    max_latency_ms: Optional[float] = None

    # Data filtering
    include_fields: Optional[set[str]] = None
    exclude_fields: Optional[set[str]] = None
    filter_conditions: list[FilterCondition] = field(default_factory=list)

    # Aggregation settings
    enable_aggregation: bool = False
    aggregation_window_ms: float = 100.0  # Aggregate messages within this window

    # Sampling settings
    sample_rate: float = 1.0  # 1.0 = all messages, 0.5 = every other message
    sample_offset: float = 0.0  # For load balancing multiple clients

    # Performance tracking
    messages_sent: int = 0
    messages_filtered: int = 0
    bytes_sent: int = 0
    last_message_time: Optional[datetime] = None
    average_latency_ms: float = 0.0

    # Custom processing
    preprocessor: Optional[Callable[[dict[str, Any]], dict[str, Any]]] = None

    def __post_init__(self):
        if self.include_fields is not None and not isinstance(self.include_fields, set):
            self.include_fields = set(self.include_fields)
        if self.exclude_fields is not None and not isinstance(self.exclude_fields, set):
            self.exclude_fields = set(self.exclude_fields)


class SubscriptionManager:
    """Advanced subscription manager with filtering, aggregation, and performance optimization."""

    def __init__(self):
        self.subscriptions: dict[
            str, dict[StreamType, StreamSubscription]
        ] = defaultdict(dict)
        self.stream_subscribers: dict[StreamType, set[str]] = {
            stream_type: set() for stream_type in StreamType
        }
        self.aggregation_buffers: dict[
            str, dict[StreamType, list[dict[str, Any]]]
        ] = defaultdict(lambda: defaultdict(list))
        self.aggregation_tasks: dict[str, asyncio.Task] = {}
        self.sample_counters: dict[str, dict[StreamType, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        self.performance_stats = {
            "total_subscriptions": 0,
            "active_subscriptions": 0,
            "messages_processed": 0,
            "messages_filtered": 0,
            "aggregation_hit_rate": 0.0,
        }

    async def create_subscription(
        self,
        client_id: str,
        stream_type: Union[StreamType, str],
        quality_level: Union[QualityLevel, str] = QualityLevel.AUTO,
        max_fps: Optional[float] = None,
        min_fps: Optional[float] = None,
        include_fields: Optional[list[str]] = None,
        exclude_fields: Optional[list[str]] = None,
        filter_conditions: Optional[list[dict[str, Any]]] = None,
        enable_aggregation: bool = False,
        aggregation_window_ms: float = 100.0,
        sample_rate: float = 1.0,
        sample_offset: float = 0.0,
        max_latency_ms: Optional[float] = None,
    ) -> bool:
        """Create a new stream subscription with advanced configuration."""
        # Convert string types to enums
        if isinstance(stream_type, str):
            try:
                stream_type = StreamType(stream_type.lower())
            except ValueError:
                logger.error(f"Invalid stream type: {stream_type}")
                return False

        if isinstance(quality_level, str):
            try:
                quality_level = QualityLevel(quality_level.lower())
            except ValueError:
                logger.warning(f"Invalid quality level: {quality_level}, using AUTO")
                quality_level = QualityLevel.AUTO

        # Parse filter conditions
        parsed_conditions = []
        if filter_conditions:
            for condition_dict in filter_conditions:
                try:
                    condition = FilterCondition(
                        field_path=condition_dict["field"],
                        operator=FilterOperator(condition_dict["operator"]),
                        value=condition_dict["value"],
                        description=condition_dict.get("description"),
                    )
                    parsed_conditions.append(condition)
                except (KeyError, ValueError) as e:
                    logger.warning(
                        f"Invalid filter condition: {condition_dict}, error: {e}"
                    )

        # Create subscription
        subscription = StreamSubscription(
            client_id=client_id,
            stream_type=stream_type,
            quality_level=quality_level,
            max_fps=max_fps,
            min_fps=min_fps,
            include_fields=set(include_fields) if include_fields else None,
            exclude_fields=set(exclude_fields) if exclude_fields else None,
            filter_conditions=parsed_conditions,
            enable_aggregation=enable_aggregation,
            aggregation_window_ms=aggregation_window_ms,
            sample_rate=max(0.0, min(1.0, sample_rate)),  # Clamp to [0, 1]
            sample_offset=sample_offset,
            max_latency_ms=max_latency_ms,
        )

        # Store subscription
        self.subscriptions[client_id][stream_type] = subscription
        self.stream_subscribers[stream_type].add(client_id)

        # Start aggregation task if needed
        if enable_aggregation:
            await self._start_aggregation_task(client_id, stream_type)

        self.performance_stats["total_subscriptions"] += 1
        self._update_active_subscriptions()

        logger.info(
            f"Created subscription for {client_id} to {stream_type.value} with quality {quality_level.value}"
        )
        return True

    async def remove_subscription(
        self, client_id: str, stream_type: Union[StreamType, str]
    ) -> bool:
        """Remove a stream subscription."""
        if isinstance(stream_type, str):
            try:
                stream_type = StreamType(stream_type.lower())
            except ValueError:
                return False

        if (
            client_id not in self.subscriptions
            or stream_type not in self.subscriptions[client_id]
        ):
            return False

        # Remove subscription
        del self.subscriptions[client_id][stream_type]
        self.stream_subscribers[stream_type].discard(client_id)

        # Stop aggregation task if running
        task_key = f"{client_id}:{stream_type.value}"
        if task_key in self.aggregation_tasks:
            self.aggregation_tasks[task_key].cancel()
            del self.aggregation_tasks[task_key]

        # Clean up empty client subscriptions
        if not self.subscriptions[client_id]:
            del self.subscriptions[client_id]

        # Clean up aggregation buffers
        if (
            client_id in self.aggregation_buffers
            and stream_type in self.aggregation_buffers[client_id]
        ):
            del self.aggregation_buffers[client_id][stream_type]

        # Clean up sample counters
        if (
            client_id in self.sample_counters
            and stream_type in self.sample_counters[client_id]
        ):
            del self.sample_counters[client_id][stream_type]

        self._update_active_subscriptions()

        logger.info(f"Removed subscription for {client_id} from {stream_type.value}")
        return True

    async def remove_all_subscriptions(self, client_id: str):
        """Remove all subscriptions for a client."""
        if client_id not in self.subscriptions:
            return

        stream_types = list(self.subscriptions[client_id].keys())
        for stream_type in stream_types:
            await self.remove_subscription(client_id, stream_type)

    async def process_message(
        self,
        stream_type: StreamType,
        data: dict[str, Any],
        timestamp: Optional[datetime] = None,
    ) -> dict[str, dict[str, Any]]:
        """Process a message for all subscribers with filtering and optimization."""
        if stream_type not in self.stream_subscribers:
            return {}

        subscribers = self.stream_subscribers[stream_type].copy()
        if not subscribers:
            return {}

        message_timestamp = timestamp or datetime.now(timezone.utc)
        processed_messages = {}

        for client_id in subscribers:
            if (
                client_id not in self.subscriptions
                or stream_type not in self.subscriptions[client_id]
            ):
                continue

            subscription = self.subscriptions[client_id][stream_type]

            if not subscription.active:
                continue

            try:
                # Apply sampling
                if not self._should_sample_message(
                    client_id, stream_type, subscription
                ):
                    subscription.messages_filtered += 1
                    continue

                # Apply filtering
                filtered_data = await self._apply_filters(data, subscription)
                if filtered_data is None:
                    subscription.messages_filtered += 1
                    continue

                # Apply quality adjustments
                quality_adjusted_data = await self._apply_quality_adjustments(
                    filtered_data, subscription, stream_type
                )

                # Apply custom preprocessing
                if subscription.preprocessor:
                    try:
                        quality_adjusted_data = subscription.preprocessor(
                            quality_adjusted_data
                        )
                    except Exception as e:
                        logger.warning(f"Preprocessor failed for {client_id}: {e}")

                # Handle aggregation
                if subscription.enable_aggregation:
                    await self._add_to_aggregation_buffer(
                        client_id, stream_type, quality_adjusted_data, message_timestamp
                    )
                else:
                    processed_messages[client_id] = quality_adjusted_data

                # Update subscription stats
                subscription.messages_sent += 1
                subscription.last_message_time = message_timestamp

                self.performance_stats["messages_processed"] += 1

            except Exception as e:
                logger.error(f"Error processing message for {client_id}: {e}")
                subscription.messages_filtered += 1

        return processed_messages

    async def get_subscription_info(self, client_id: str) -> dict[str, Any]:
        """Get detailed subscription information for a client."""
        if client_id not in self.subscriptions:
            return {"client_id": client_id, "subscriptions": []}

        subscription_info = []
        for stream_type, subscription in self.subscriptions[client_id].items():
            info = {
                "stream_type": stream_type.value,
                "active": subscription.active,
                "quality_level": subscription.quality_level.value,
                "max_fps": subscription.max_fps,
                "min_fps": subscription.min_fps,
                "max_latency_ms": subscription.max_latency_ms,
                "include_fields": (
                    list(subscription.include_fields)
                    if subscription.include_fields
                    else None
                ),
                "exclude_fields": (
                    list(subscription.exclude_fields)
                    if subscription.exclude_fields
                    else None
                ),
                "filter_conditions": [
                    {
                        "field": cond.field_path,
                        "operator": cond.operator.value,
                        "value": cond.value,
                        "description": cond.description,
                    }
                    for cond in subscription.filter_conditions
                ],
                "enable_aggregation": subscription.enable_aggregation,
                "aggregation_window_ms": subscription.aggregation_window_ms,
                "sample_rate": subscription.sample_rate,
                "sample_offset": subscription.sample_offset,
                "messages_sent": subscription.messages_sent,
                "messages_filtered": subscription.messages_filtered,
                "bytes_sent": subscription.bytes_sent,
                "last_message_time": (
                    subscription.last_message_time.isoformat()
                    if subscription.last_message_time
                    else None
                ),
                "average_latency_ms": subscription.average_latency_ms,
            }
            subscription_info.append(info)

        return {"client_id": client_id, "subscriptions": subscription_info}

    def get_performance_stats(self) -> dict[str, Any]:
        """Get overall subscription performance statistics."""
        self._update_performance_stats()
        return self.performance_stats.copy()

    async def update_subscription_quality(
        self, client_id: str, stream_type: StreamType, quality_level: QualityLevel
    ):
        """Update the quality level for a subscription."""
        if (
            client_id in self.subscriptions
            and stream_type in self.subscriptions[client_id]
        ):
            self.subscriptions[client_id][stream_type].quality_level = quality_level
            logger.info(
                f"Updated quality for {client_id}/{stream_type.value} to {quality_level.value}"
            )

    async def pause_subscription(self, client_id: str, stream_type: StreamType):
        """Pause a subscription without removing it."""
        if (
            client_id in self.subscriptions
            and stream_type in self.subscriptions[client_id]
        ):
            self.subscriptions[client_id][stream_type].active = False
            logger.info(f"Paused subscription for {client_id}/{stream_type.value}")

    async def resume_subscription(self, client_id: str, stream_type: StreamType):
        """Resume a paused subscription."""
        if (
            client_id in self.subscriptions
            and stream_type in self.subscriptions[client_id]
        ):
            self.subscriptions[client_id][stream_type].active = True
            logger.info(f"Resumed subscription for {client_id}/{stream_type.value}")

    def _should_sample_message(
        self, client_id: str, stream_type: StreamType, subscription: StreamSubscription
    ) -> bool:
        """Determine if a message should be sampled based on subscription settings."""
        if subscription.sample_rate >= 1.0:
            return True

        # Get current sample counter
        counter = self.sample_counters[client_id][stream_type]
        self.sample_counters[client_id][stream_type] = (
            counter + 1
        ) % 1000000  # Prevent overflow

        # Calculate if this message should be sampled
        threshold = subscription.sample_rate * 1000000
        sample_point = (counter + subscription.sample_offset * 1000000) % 1000000

        return sample_point < threshold

    async def _apply_filters(
        self, data: dict[str, Any], subscription: StreamSubscription
    ) -> Optional[dict[str, Any]]:
        """Apply all filters to message data."""
        filtered_data = data.copy()

        # Apply field inclusion/exclusion
        if subscription.include_fields:
            filtered_data = {
                k: v
                for k, v in filtered_data.items()
                if k in subscription.include_fields
            }
        elif subscription.exclude_fields:
            filtered_data = {
                k: v
                for k, v in filtered_data.items()
                if k not in subscription.exclude_fields
            }

        # Apply filter conditions
        for condition in subscription.filter_conditions:
            if not self._evaluate_filter_condition(filtered_data, condition):
                return None  # Message filtered out

        return filtered_data

    def _evaluate_filter_condition(
        self, data: dict[str, Any], condition: FilterCondition
    ) -> bool:
        """Evaluate a single filter condition against data."""
        try:
            # Navigate to the field using dot notation
            value = data
            for field_part in condition.field_path.split("."):
                if field_part.isdigit():
                    # Array index
                    value = value[int(field_part)]
                else:
                    # Object field
                    value = value[field_part]

            # Apply operator
            if condition.operator == FilterOperator.EQUALS:
                return value == condition.value
            elif condition.operator == FilterOperator.NOT_EQUALS:
                return value != condition.value
            elif condition.operator == FilterOperator.GREATER_THAN:
                return value > condition.value
            elif condition.operator == FilterOperator.GREATER_EQUAL:
                return value >= condition.value
            elif condition.operator == FilterOperator.LESS_THAN:
                return value < condition.value
            elif condition.operator == FilterOperator.LESS_EQUAL:
                return value <= condition.value
            elif condition.operator == FilterOperator.IN:
                return value in condition.value
            elif condition.operator == FilterOperator.NOT_IN:
                return value not in condition.value
            elif condition.operator == FilterOperator.CONTAINS:
                return condition.value in str(value)
            elif condition.operator == FilterOperator.MATCHES:
                import re

                return bool(re.search(condition.value, str(value)))

        except (KeyError, IndexError, TypeError, ValueError):
            # Field not found or type mismatch, condition fails
            return False

        return True

    async def _apply_quality_adjustments(
        self,
        data: dict[str, Any],
        subscription: StreamSubscription,
        stream_type: StreamType,
    ) -> dict[str, Any]:
        """Apply quality adjustments based on subscription settings."""
        if (
            stream_type == StreamType.FRAME
            and subscription.quality_level != QualityLevel.AUTO
        ):
            # Adjust frame quality settings
            if subscription.quality_level == QualityLevel.LOW:
                data["quality"] = 60
                data["width"] = min(data.get("width", 1920), 1280)
                data["height"] = min(data.get("height", 1080), 720)
            elif subscription.quality_level == QualityLevel.MEDIUM:
                data["quality"] = 75
                data["width"] = min(data.get("width", 1920), 1600)
                data["height"] = min(data.get("height", 1080), 900)
            elif subscription.quality_level == QualityLevel.HIGH:
                data["quality"] = 90
                # Keep original resolution

        return data

    async def _add_to_aggregation_buffer(
        self,
        client_id: str,
        stream_type: StreamType,
        data: dict[str, Any],
        timestamp: datetime,
    ):
        """Add message to aggregation buffer."""
        buffer_entry = {"data": data, "timestamp": timestamp}
        self.aggregation_buffers[client_id][stream_type].append(buffer_entry)

    async def _start_aggregation_task(self, client_id: str, stream_type: StreamType):
        """Start aggregation task for a client/stream combination."""
        task_key = f"{client_id}:{stream_type.value}"

        if task_key in self.aggregation_tasks:
            return  # Task already running

        subscription = self.subscriptions[client_id][stream_type]

        async def aggregation_task():
            try:
                while subscription.active:
                    await asyncio.sleep(subscription.aggregation_window_ms / 1000.0)

                    buffer = self.aggregation_buffers[client_id][stream_type]
                    if not buffer:
                        continue

                    # Aggregate messages in buffer
                    aggregated_data = await self._aggregate_messages(
                        buffer, stream_type
                    )

                    if aggregated_data:
                        # Send aggregated message (this would be handled by the caller)
                        logger.debug(
                            f"Aggregated {len(buffer)} messages for {client_id}/{stream_type.value}"
                        )

                    # Clear buffer
                    buffer.clear()

            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.error(f"Error in aggregation task for {task_key}: {e}")

        self.aggregation_tasks[task_key] = asyncio.create_task(aggregation_task())

    async def _aggregate_messages(
        self, buffer: list[dict[str, Any]], stream_type: StreamType
    ) -> Optional[dict[str, Any]]:
        """Aggregate multiple messages into a single message."""
        if not buffer:
            return None

        if stream_type == StreamType.STATE:
            # For game state, use the most recent data
            return buffer[-1]["data"]
        elif stream_type == StreamType.FRAME:
            # For frames, use the most recent frame
            return buffer[-1]["data"]
        elif stream_type == StreamType.TRAJECTORY:
            # For trajectories, combine all line segments
            all_lines = []
            all_collisions = []
            for entry in buffer:
                data = entry["data"]
                all_lines.extend(data.get("lines", []))
                all_collisions.extend(data.get("collisions", []))

            return {
                "lines": all_lines,
                "collisions": all_collisions,
                "aggregated": True,
                "message_count": len(buffer),
            }
        else:
            # For other types, use the most recent
            return buffer[-1]["data"]

    def _update_active_subscriptions(self):
        """Update active subscription count."""
        active_count = 0
        for client_subs in self.subscriptions.values():
            active_count += sum(1 for sub in client_subs.values() if sub.active)
        self.performance_stats["active_subscriptions"] = active_count

    def _update_performance_stats(self):
        """Update performance statistics."""
        total_filtered = sum(
            sub.messages_filtered
            for client_subs in self.subscriptions.values()
            for sub in client_subs.values()
        )

        self.performance_stats["messages_filtered"] = total_filtered

        # Calculate aggregation hit rate
        total_aggregation_buffers = sum(
            len(stream_buffers)
            for client_buffers in self.aggregation_buffers.values()
            for stream_buffers in client_buffers.values()
        )

        if self.performance_stats["messages_processed"] > 0:
            self.performance_stats["aggregation_hit_rate"] = (
                total_aggregation_buffers / self.performance_stats["messages_processed"]
            )


# Global subscription manager instance
subscription_manager = SubscriptionManager()
