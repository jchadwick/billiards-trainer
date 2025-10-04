"""Performance monitoring and metrics collection system."""

import logging
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class SystemMetrics:
    """System-wide performance metrics."""

    timestamp: float = field(default_factory=time.time)

    # CPU metrics
    cpu_percent: float = 0.0
    cpu_count: int = 0
    load_average: list[float] = field(default_factory=list)

    # Memory metrics
    memory_total: int = 0
    memory_used: int = 0
    memory_percent: float = 0.0
    memory_available: int = 0

    # Disk metrics
    disk_total: int = 0
    disk_used: int = 0
    disk_percent: float = 0.0
    disk_free: int = 0

    # Network metrics
    network_bytes_sent: int = 0
    network_bytes_recv: int = 0
    network_packets_sent: int = 0
    network_packets_recv: int = 0

    # Process metrics
    process_count: int = 0
    thread_count: int = 0

    # GPU metrics (if available)
    gpu_percent: float = 0.0
    gpu_memory_used: int = 0
    gpu_memory_total: int = 0
