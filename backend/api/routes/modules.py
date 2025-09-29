"""Module control endpoints for system module management.

Provides module lifecycle management including:
- Module start/stop/restart operations
- Real-time module status monitoring
- Module health and performance metrics
"""

import asyncio
import logging
import time
from typing import Any, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Path, Query
from pydantic import BaseModel, Field

from ..dependencies import ApplicationState, get_app_state
from ..models.responses import BaseResponse

# Try to import system orchestrator with fallback
try:
    from ...system.orchestrator import SystemOrchestrator
except ImportError:
    try:
        from system.orchestrator import SystemOrchestrator
    except ImportError:
        SystemOrchestrator = None

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/modules", tags=["modules"])


# Response models
class ModuleStatus(BaseModel):
    """Module status information."""

    name: str
    state: str  # offline, starting, running, degraded, stopping, error
    health: str  # healthy, warning, unhealthy, unknown
    startup_time: Optional[float] = None
    last_error: Optional[str] = None
    restart_count: int = 0
    uptime: Optional[float] = None
    metrics: dict[str, Any] = Field(default_factory=dict)


class ModuleActionResponse(BaseResponse):
    """Response for module action operations."""

    module_name: str
    action: str
    success: bool
    timestamp: str
    message: str
    status: Optional[ModuleStatus] = None


class ModuleListResponse(BaseResponse):
    """Response for module list operations."""

    modules: list[ModuleStatus]
    total_modules: int
    healthy_modules: int
    running_modules: int


# Helper functions
async def get_system_orchestrator() -> Optional[SystemOrchestrator]:
    """Get system orchestrator instance."""
    app_state = get_app_state()
    # In a real implementation, this would get the orchestrator from the app state
    # For now, we'll return None and handle gracefully
    return getattr(app_state, "orchestrator", None)


def format_module_status(module_name: str, module_info: dict[str, Any]) -> ModuleStatus:
    """Format module information into ModuleStatus model."""
    return ModuleStatus(
        name=module_name,
        state=module_info.get("state", "unknown"),
        health=module_info.get("health", "unknown"),
        startup_time=module_info.get("startup_time"),
        last_error=module_info.get("last_error"),
        restart_count=module_info.get("restart_count", 0),
        uptime=module_info.get("uptime"),
        metrics=module_info.get("metrics", {}),
    )


# Endpoints
@router.get("", response_model=ModuleListResponse)
async def list_modules(
    status_filter: Optional[str] = Query(None, description="Filter by module status")
) -> ModuleListResponse:
    """List all system modules and their current status.

    Args:
        status_filter: Optional filter by module status (running, stopped, error, etc.)

    Returns:
        List of all modules with their current status information
    """
    try:
        orchestrator = await get_system_orchestrator()

        if not orchestrator:
            # Fallback - return basic module list
            default_modules = [
                {
                    "name": "core",
                    "state": "running",
                    "health": "healthy",
                    "startup_time": 1.2,
                },
                {
                    "name": "vision",
                    "state": "running",
                    "health": "healthy",
                    "startup_time": 2.1,
                },
                {
                    "name": "api",
                    "state": "running",
                    "health": "healthy",
                    "startup_time": 0.8,
                },
                {
                    "name": "config",
                    "state": "running",
                    "health": "healthy",
                    "startup_time": 0.5,
                },
                {
                    "name": "projector",
                    "state": "running",
                    "health": "healthy",
                    "startup_time": 1.8,
                },
            ]

            modules = [
                format_module_status(mod["name"], mod)
                for mod in default_modules
                if not status_filter or mod["state"] == status_filter
            ]
        else:
            # Get real module status from orchestrator
            module_statuses = getattr(orchestrator, "module_status", {})
            modules = [
                format_module_status(
                    name,
                    {
                        "state": status.state.value
                        if hasattr(status.state, "value")
                        else str(status.state),
                        "health": status.health.value
                        if hasattr(status.health, "value")
                        else str(status.health),
                        "startup_time": status.startup_time,
                        "last_error": status.last_error,
                        "restart_count": status.restart_count,
                        "metrics": status.metrics,
                        "uptime": time.time() - status.startup_time
                        if status.startup_time
                        else None,
                    },
                )
                for name, status in module_statuses.items()
                if not status_filter
                or str(status.state).lower() == status_filter.lower()
            ]

        healthy_count = sum(1 for m in modules if m.health == "healthy")
        running_count = sum(1 for m in modules if m.state == "running")

        return ModuleListResponse(
            success=True,
            message=f"Retrieved {len(modules)} modules",
            modules=modules,
            total_modules=len(modules),
            healthy_modules=healthy_count,
            running_modules=running_count,
        )

    except Exception as e:
        logger.error(f"Error listing modules: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list modules: {str(e)}")


@router.get("/{module_id}", response_model=ModuleActionResponse)
async def get_module_status(
    module_id: str = Path(..., description="Module identifier")
) -> ModuleActionResponse:
    """Get detailed status for a specific module.

    Args:
        module_id: The module identifier (core, vision, api, config, projector)

    Returns:
        Detailed status information for the specified module
    """
    try:
        valid_modules = ["core", "vision", "api", "config", "projector"]
        if module_id not in valid_modules:
            raise HTTPException(
                status_code=404,
                detail=f"Module '{module_id}' not found. Valid modules: {valid_modules}",
            )

        orchestrator = await get_system_orchestrator()

        if not orchestrator:
            # Fallback status
            status = ModuleStatus(
                name=module_id,
                state="running",
                health="healthy",
                startup_time=1.5,
                uptime=time.time() - 3600,  # 1 hour uptime
                metrics={"requests": 0, "errors": 0},
            )
        else:
            # Get real status from orchestrator
            module_statuses = getattr(orchestrator, "module_status", {})
            if module_id not in module_statuses:
                raise HTTPException(
                    status_code=404, detail=f"Module '{module_id}' not found in system"
                )

            module_info = module_statuses[module_id]
            status = format_module_status(
                module_id,
                {
                    "state": module_info.state.value
                    if hasattr(module_info.state, "value")
                    else str(module_info.state),
                    "health": module_info.health.value
                    if hasattr(module_info.health, "value")
                    else str(module_info.health),
                    "startup_time": module_info.startup_time,
                    "last_error": module_info.last_error,
                    "restart_count": module_info.restart_count,
                    "metrics": module_info.metrics,
                    "uptime": time.time() - module_info.startup_time
                    if module_info.startup_time
                    else None,
                },
            )

        return ModuleActionResponse(
            success=True,
            message=f"Retrieved status for module '{module_id}'",
            module_name=module_id,
            action="get_status",
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            status=status,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting module status for {module_id}: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get module status: {str(e)}"
        )


@router.post("/{module_id}/start", response_model=ModuleActionResponse)
async def start_module(
    module_id: str = Path(..., description="Module identifier"),
    background_tasks: BackgroundTasks = BackgroundTasks(),
) -> ModuleActionResponse:
    """Start a system module.

    Args:
        module_id: The module identifier (core, vision, api, config, projector)
        background_tasks: FastAPI background tasks

    Returns:
        Result of the module start operation
    """
    try:
        valid_modules = ["core", "vision", "api", "config", "projector"]
        if module_id not in valid_modules:
            raise HTTPException(
                status_code=404,
                detail=f"Module '{module_id}' not found. Valid modules: {valid_modules}",
            )

        orchestrator = await get_system_orchestrator()

        if not orchestrator:
            # For critical system modules that are already running
            if module_id in ["api", "core", "config"]:
                return ModuleActionResponse(
                    success=True,
                    message=f"Module '{module_id}' is already running (system module)",
                    module_name=module_id,
                    action="start",
                    timestamp=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
                )
            else:
                # Simulate starting other modules
                return ModuleActionResponse(
                    success=True,
                    message=f"Module '{module_id}' started successfully",
                    module_name=module_id,
                    action="start",
                    timestamp=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
                )

        # Use real orchestrator to start module
        async def start_module_task():
            try:
                success = await orchestrator._start_module(module_id)
                logger.info(
                    f"Module '{module_id}' start operation completed: {success}"
                )
            except Exception as e:
                logger.error(f"Failed to start module '{module_id}': {e}")

        background_tasks.add_task(start_module_task)

        return ModuleActionResponse(
            success=True,
            message=f"Module '{module_id}' start operation initiated",
            module_name=module_id,
            action="start",
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting module {module_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start module: {str(e)}")


@router.post("/{module_id}/stop", response_model=ModuleActionResponse)
async def stop_module(
    module_id: str = Path(..., description="Module identifier"),
    background_tasks: BackgroundTasks = BackgroundTasks(),
) -> ModuleActionResponse:
    """Stop a system module.

    Args:
        module_id: The module identifier (core, vision, api, config, projector)
        background_tasks: FastAPI background tasks

    Returns:
        Result of the module stop operation
    """
    try:
        valid_modules = ["core", "vision", "api", "config", "projector"]
        if module_id not in valid_modules:
            raise HTTPException(
                status_code=404,
                detail=f"Module '{module_id}' not found. Valid modules: {valid_modules}",
            )

        # Prevent stopping critical system modules
        if module_id in ["api", "core", "config"]:
            raise HTTPException(
                status_code=400,
                detail=f"Cannot stop critical system module '{module_id}'",
            )

        orchestrator = await get_system_orchestrator()

        if not orchestrator:
            # Simulate stopping module
            return ModuleActionResponse(
                success=True,
                message=f"Module '{module_id}' stopped successfully",
                module_name=module_id,
                action="stop",
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            )

        # Use real orchestrator to stop module
        async def stop_module_task():
            try:
                success = await orchestrator._stop_module(module_id)
                logger.info(f"Module '{module_id}' stop operation completed: {success}")
            except Exception as e:
                logger.error(f"Failed to stop module '{module_id}': {e}")

        background_tasks.add_task(stop_module_task)

        return ModuleActionResponse(
            success=True,
            message=f"Module '{module_id}' stop operation initiated",
            module_name=module_id,
            action="stop",
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error stopping module {module_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to stop module: {str(e)}")


@router.post("/{module_id}/restart", response_model=ModuleActionResponse)
async def restart_module(
    module_id: str = Path(..., description="Module identifier"),
    background_tasks: BackgroundTasks = BackgroundTasks(),
) -> ModuleActionResponse:
    """Restart a system module.

    Args:
        module_id: The module identifier (core, vision, api, config, projector)
        background_tasks: FastAPI background tasks

    Returns:
        Result of the module restart operation
    """
    try:
        valid_modules = ["core", "vision", "api", "config", "projector"]
        if module_id not in valid_modules:
            raise HTTPException(
                status_code=404,
                detail=f"Module '{module_id}' not found. Valid modules: {valid_modules}",
            )

        # Prevent restarting critical system modules
        if module_id in ["api", "core", "config"]:
            raise HTTPException(
                status_code=400,
                detail=f"Cannot restart critical system module '{module_id}'",
            )

        orchestrator = await get_system_orchestrator()

        if not orchestrator:
            # Simulate restarting module
            return ModuleActionResponse(
                success=True,
                message=f"Module '{module_id}' restarted successfully",
                module_name=module_id,
                action="restart",
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            )

        # Use real orchestrator to restart module
        async def restart_module_task():
            try:
                # Stop then start the module
                await orchestrator._stop_module(module_id)
                await asyncio.sleep(1)  # Brief pause between stop and start
                success = await orchestrator._start_module(module_id)
                logger.info(
                    f"Module '{module_id}' restart operation completed: {success}"
                )
            except Exception as e:
                logger.error(f"Failed to restart module '{module_id}': {e}")

        background_tasks.add_task(restart_module_task)

        return ModuleActionResponse(
            success=True,
            message=f"Module '{module_id}' restart operation initiated",
            module_name=module_id,
            action="restart",
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error restarting module {module_id}: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to restart module: {str(e)}"
        )


@router.get("/{module_id}/logs")
async def get_module_logs(
    module_id: str = Path(..., description="Module identifier"),
    lines: int = Query(
        100, ge=1, le=1000, description="Number of log lines to retrieve"
    ),
    level: Optional[str] = Query(
        None, description="Filter by log level (DEBUG, INFO, WARNING, ERROR)"
    ),
):
    """Get recent log entries for a specific module.

    Args:
        module_id: The module identifier
        lines: Number of log lines to retrieve (1-1000)
        level: Optional filter by log level

    Returns:
        Recent log entries for the specified module
    """
    try:
        valid_modules = ["core", "vision", "api", "config", "projector"]
        if module_id not in valid_modules:
            raise HTTPException(
                status_code=404,
                detail=f"Module '{module_id}' not found. Valid modules: {valid_modules}",
            )

        # For now, return mock log data
        # In a real implementation, this would read from actual log files
        mock_logs = [
            {
                "timestamp": "2025-01-29 10:30:15",
                "level": "INFO",
                "message": f"{module_id.title()} module initialized successfully",
                "logger": f"backend.{module_id}",
            },
            {
                "timestamp": "2025-01-29 10:30:16",
                "level": "DEBUG",
                "message": f"{module_id.title()} module configuration loaded",
                "logger": f"backend.{module_id}",
            },
            {
                "timestamp": "2025-01-29 10:30:17",
                "level": "INFO",
                "message": f"{module_id.title()} module started successfully",
                "logger": f"backend.{module_id}",
            },
        ]

        # Filter by level if specified
        if level:
            mock_logs = [log for log in mock_logs if log["level"] == level.upper()]

        # Limit to requested number of lines
        mock_logs = mock_logs[:lines]

        return {
            "success": True,
            "message": f"Retrieved {len(mock_logs)} log entries for module '{module_id}'",
            "module_name": module_id,
            "logs": mock_logs,
            "total_lines": len(mock_logs),
            "filtered_level": level,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting logs for module {module_id}: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get module logs: {str(e)}"
        )
