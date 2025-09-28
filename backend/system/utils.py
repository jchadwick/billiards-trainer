"""System utilities for resource monitoring and process management."""

import asyncio
import logging
import os
import platform
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import psutil

logger = logging.getLogger(__name__)


@dataclass
class ResourceUsage:
    """Current system resource usage."""

    timestamp: float = field(default_factory=time.time)

    # CPU
    cpu_percent: float = 0.0
    cpu_count: int = 0

    # Memory (in bytes)
    memory_total: int = 0
    memory_used: int = 0
    memory_available: int = 0
    memory_percent: float = 0.0

    # Disk (in bytes)
    disk_total: int = 0
    disk_used: int = 0
    disk_free: int = 0
    disk_percent: float = 0.0

    # Network (in bytes)
    network_sent: int = 0
    network_received: int = 0

    # Process info
    process_count: int = 0
    open_files: int = 0
    connections: int = 0


@dataclass
class ProcessInfo:
    """Information about a running process."""

    pid: int
    name: str
    status: str
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    memory_mb: float = 0.0
    threads: int = 0
    create_time: float = 0.0
    command: str = ""


class ResourceMonitor:
    """System resource monitoring utilities."""

    def __init__(self):
        """Initialize resource monitor."""
        self.is_running = False
        self.monitoring_interval = 5.0
        self.history: list[ResourceUsage] = []
        self.max_history = 720  # 1 hour at 5-second intervals

        logger.info("Resource Monitor initialized")

    async def start(self) -> None:
        """Start resource monitoring."""
        self.is_running = True
        logger.info("Resource monitoring started")

    async def stop(self) -> None:
        """Stop resource monitoring."""
        self.is_running = False
        logger.info("Resource monitoring stopped")

    async def get_current_usage(self) -> ResourceUsage:
        """Get current system resource usage.

        Returns:
            Current resource usage
        """
        try:
            # CPU
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()

            # Memory
            memory = psutil.virtual_memory()

            # Disk (root filesystem)
            disk = psutil.disk_usage("/")

            # Network
            network = psutil.net_io_counters()

            # Process counts
            processes = list(psutil.process_iter(["pid", "name"]))
            process_count = len(processes)

            # Open files and connections
            try:
                current_process = psutil.Process()
                open_files = len(current_process.open_files())
                connections = len(current_process.connections())
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                open_files = 0
                connections = 0

            usage = ResourceUsage(
                cpu_percent=cpu_percent,
                cpu_count=cpu_count,
                memory_total=memory.total,
                memory_used=memory.used,
                memory_available=memory.available,
                memory_percent=memory.percent,
                disk_total=disk.total,
                disk_used=disk.used,
                disk_free=disk.free,
                disk_percent=(disk.used / disk.total) * 100,
                network_sent=network.bytes_sent,
                network_received=network.bytes_recv,
                process_count=process_count,
                open_files=open_files,
                connections=connections,
            )

            # Add to history
            self._add_to_history(usage)

            return usage

        except Exception as e:
            logger.error(f"Failed to get resource usage: {e}")
            return ResourceUsage()

    def get_usage_history(self, minutes: int = 60) -> list[ResourceUsage]:
        """Get historical resource usage.

        Args:
            minutes: Number of minutes of history to return

        Returns:
            List of historical usage data
        """
        if not self.history:
            return []

        # Calculate how many entries to return
        entries_per_minute = 60 / self.monitoring_interval
        max_entries = int(minutes * entries_per_minute)

        return self.history[-max_entries:]

    def _add_to_history(self, usage: ResourceUsage) -> None:
        """Add usage data to history with size limit."""
        self.history.append(usage)
        if len(self.history) > self.max_history:
            self.history.pop(0)


class ProcessManager:
    """Process management utilities."""

    def __init__(self):
        """Initialize process manager."""
        self.managed_processes: dict[str, subprocess.Popen] = {}
        logger.info("Process Manager initialized")

    async def start_process(
        self,
        name: str,
        command: list[str],
        cwd: Optional[str] = None,
        env: Optional[dict[str, str]] = None,
    ) -> bool:
        """Start a managed process.

        Args:
            name: Process name identifier
            command: Command and arguments to execute
            cwd: Working directory
            env: Environment variables

        Returns:
            True if process started successfully
        """
        try:
            if name in self.managed_processes:
                logger.warning(f"Process {name} is already running")
                return False

            logger.info(f"Starting process: {name}")

            process = subprocess.Popen(
                command,
                cwd=cwd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            self.managed_processes[name] = process

            # Brief check to ensure process started
            await asyncio.sleep(0.1)
            if process.poll() is not None:
                logger.error(
                    f"Process {name} failed to start (exit code: {process.returncode})"
                )
                del self.managed_processes[name]
                return False

            logger.info(f"Process {name} started successfully (PID: {process.pid})")
            return True

        except Exception as e:
            logger.error(f"Failed to start process {name}: {e}")
            return False

    async def stop_process(self, name: str, timeout: float = 10.0) -> bool:
        """Stop a managed process gracefully.

        Args:
            name: Process name identifier
            timeout: Timeout for graceful shutdown

        Returns:
            True if process stopped successfully
        """
        try:
            if name not in self.managed_processes:
                logger.warning(f"Process {name} is not running")
                return True

            process = self.managed_processes[name]

            if process.poll() is not None:
                # Process already terminated
                del self.managed_processes[name]
                return True

            logger.info(f"Stopping process: {name}")

            # Try graceful shutdown first
            process.terminate()

            try:
                # Wait for graceful shutdown
                process.wait(timeout=timeout)
                logger.info(f"Process {name} stopped gracefully")
            except subprocess.TimeoutExpired:
                # Force kill if graceful shutdown failed
                logger.warning(f"Process {name} did not stop gracefully, force killing")
                process.kill()
                process.wait()

            del self.managed_processes[name]
            return True

        except Exception as e:
            logger.error(f"Failed to stop process {name}: {e}")
            return False

    def get_process_status(self, name: str) -> Optional[ProcessInfo]:
        """Get status of a managed process.

        Args:
            name: Process name identifier

        Returns:
            Process information or None if not found
        """
        try:
            if name not in self.managed_processes:
                return None

            process = self.managed_processes[name]

            if process.poll() is not None:
                # Process has terminated
                del self.managed_processes[name]
                return None

            # Get detailed process info using psutil
            psutil_process = psutil.Process(process.pid)

            return ProcessInfo(
                pid=process.pid,
                name=psutil_process.name(),
                status=psutil_process.status(),
                cpu_percent=psutil_process.cpu_percent(),
                memory_percent=psutil_process.memory_percent(),
                memory_mb=psutil_process.memory_info().rss / 1024 / 1024,
                threads=psutil_process.num_threads(),
                create_time=psutil_process.create_time(),
                command=" ".join(psutil_process.cmdline()),
            )

        except (psutil.NoSuchProcess, psutil.AccessDenied, Exception) as e:
            logger.error(f"Failed to get status for process {name}: {e}")
            return None

    def list_managed_processes(self) -> dict[str, ProcessInfo]:
        """List all managed processes.

        Returns:
            Dictionary of process names to their info
        """
        result = {}

        for name in list(self.managed_processes.keys()):
            info = self.get_process_status(name)
            if info:
                result[name] = info

        return result

    async def cleanup_all_processes(self) -> None:
        """Stop all managed processes."""
        try:
            logger.info("Cleaning up all managed processes")

            # Stop all processes
            stop_tasks = []
            for name in list(self.managed_processes.keys()):
                stop_tasks.append(self.stop_process(name))

            if stop_tasks:
                await asyncio.gather(*stop_tasks, return_exceptions=True)

            logger.info("All managed processes cleaned up")

        except Exception as e:
            logger.error(f"Error during process cleanup: {e}")


class SystemUtils:
    """General system utilities."""

    @staticmethod
    def get_system_info() -> dict[str, Any]:
        """Get basic system information.

        Returns:
            System information dictionary
        """
        try:
            return {
                "platform": platform.platform(),
                "system": platform.system(),
                "release": platform.release(),
                "version": platform.version(),
                "machine": platform.machine(),
                "processor": platform.processor(),
                "architecture": platform.architecture(),
                "hostname": platform.node(),
                "python_version": platform.python_version(),
                "cpu_count": psutil.cpu_count(),
                "memory_total_gb": psutil.virtual_memory().total / (1024**3),
                "disk_total_gb": psutil.disk_usage("/").total / (1024**3),
                "boot_time": psutil.boot_time(),
                "uptime_seconds": time.time() - psutil.boot_time(),
            }

        except Exception as e:
            logger.error(f"Failed to get system info: {e}")
            return {}

    @staticmethod
    def check_port_available(port: int, host: str = "localhost") -> bool:
        """Check if a port is available.

        Args:
            port: Port number to check
            host: Host to check on

        Returns:
            True if port is available
        """
        try:
            import socket

            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(1)
                result = sock.connect_ex((host, port))
                return result != 0  # Port is available if connection failed

        except Exception as e:
            logger.error(f"Failed to check port {port}: {e}")
            return False

    @staticmethod
    def find_available_port(
        start_port: int = 8000, max_attempts: int = 100
    ) -> Optional[int]:
        """Find an available port starting from start_port.

        Args:
            start_port: Starting port number
            max_attempts: Maximum number of ports to try

        Returns:
            Available port number or None if none found
        """
        for port in range(start_port, start_port + max_attempts):
            if SystemUtils.check_port_available(port):
                return port

        return None

    @staticmethod
    async def run_command(
        command: list[str], timeout: float = 30.0, cwd: Optional[str] = None
    ) -> dict[str, Any]:
        """Run a command asynchronously.

        Args:
            command: Command and arguments
            timeout: Timeout in seconds
            cwd: Working directory

        Returns:
            Command result with stdout, stderr, and return code
        """
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), timeout=timeout
                )

                return {
                    "returncode": process.returncode,
                    "stdout": stdout.decode("utf-8", errors="replace"),
                    "stderr": stderr.decode("utf-8", errors="replace"),
                    "success": process.returncode == 0,
                }

            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                return {
                    "returncode": -1,
                    "stdout": "",
                    "stderr": "Command timed out",
                    "success": False,
                }

        except Exception as e:
            return {"returncode": -1, "stdout": "", "stderr": str(e), "success": False}

    @staticmethod
    def create_directory(path: str, mode: int = 0o755) -> bool:
        """Create a directory with proper permissions.

        Args:
            path: Directory path to create
            mode: Directory permissions

        Returns:
            True if directory was created or already exists
        """
        try:
            os.makedirs(path, mode=mode, exist_ok=True)
            return True

        except Exception as e:
            logger.error(f"Failed to create directory {path}: {e}")
            return False

    @staticmethod
    def ensure_file_exists(path: str, default_content: str = "") -> bool:
        """Ensure a file exists, creating it if necessary.

        Args:
            path: File path
            default_content: Content to write if file doesn't exist

        Returns:
            True if file exists or was created successfully
        """
        try:
            if not os.path.exists(path):
                # Create directory if needed
                directory = os.path.dirname(path)
                if directory and not os.path.exists(directory):
                    os.makedirs(directory, exist_ok=True)

                # Create file
                with open(path, "w") as f:
                    f.write(default_content)

            return True

        except Exception as e:
            logger.error(f"Failed to ensure file exists {path}: {e}")
            return False

    @staticmethod
    def get_disk_usage(path: str = "/") -> dict[str, int]:
        """Get disk usage for a path.

        Args:
            path: Path to check

        Returns:
            Dictionary with total, used, and free space in bytes
        """
        try:
            usage = psutil.disk_usage(path)
            return {
                "total": usage.total,
                "used": usage.used,
                "free": usage.free,
                "percent": (usage.used / usage.total) * 100,
            }

        except Exception as e:
            logger.error(f"Failed to get disk usage for {path}: {e}")
            return {"total": 0, "used": 0, "free": 0, "percent": 0.0}
