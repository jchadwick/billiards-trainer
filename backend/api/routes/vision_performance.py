"""Vision performance monitoring API endpoints."""

import socket
import time
from typing import Any, Dict

from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/vision", tags=["vision-performance"])


@router.get("/performance/audit")
async def get_performance_audit_status() -> dict[str, Any]:
    """Return bounded event-export health and active audit domains."""
    from ..dependencies import app_state

    vision = getattr(app_state, "vision_module", None)
    if vision is None:
        raise HTTPException(status_code=503, detail="Vision module not initialized")
    recorder = getattr(vision, "audit_recorder", None)
    if recorder is None:
        # A one-shot identity query lets the explicit audit client verify
        # cross-process monotonic timing during an instrumentation-off
        # baseline. It creates no recorder, sampler, queue, or background task.
        from ...performance_audit import monotonic_clock_domain_id

        return {
            "audit_enabled": False,
            "host_id": socket.gethostname(),
            "clock_domain_id": monotonic_clock_domain_id(),
            "clock_implementation": time.get_clock_info("perf_counter").implementation,
            "audit_domains": getattr(vision, "audit_domains", {}),
            "event_export": None,
        }
    return {
        "audit_enabled": True,
        "host_id": recorder.host_id,
        "clock_domain_id": recorder.clock_domain_id,
        "clock_implementation": recorder.clock_implementation,
        "audit_domains": getattr(vision, "audit_domains", {}),
        "event_export": recorder.status(),
    }


@router.get("/performance")
async def get_performance_stats() -> dict[str, Any]:
    """Get real-time vision performance statistics.

    Returns detailed timing breakdown for the vision pipeline including:
    - Current FPS and frame times
    - Breakdown by processing stage
    - Bottleneck identification
    - Historical statistics

    Returns:
        Dictionary with performance metrics
    """
    try:
        # Get vision module from request app state
        from ..dependencies import app_state

        if not hasattr(app_state, "vision_module") or app_state.vision_module is None:
            raise HTTPException(status_code=503, detail="Vision module not initialized")

        vision = app_state.vision_module

        # Check if profiler is available
        if not hasattr(vision, "profiler") or vision.profiler is None:
            return {
                "error": "Performance profiling is disabled",
                "message": "Enable performance_audit.enabled or launch with --performance-audit; performance_audit.state_enabled must also be true",
                "profiling_enabled": False,
            }

        # Get real-time status from profiler
        status = vision.profiler.get_realtime_status()
        recorder = getattr(vision, "audit_recorder", None)
        return {
            "profiling_enabled": True,
            "audit_domains": getattr(vision, "audit_domains", {"state": True}),
            "event_export": recorder.status() if recorder is not None else None,
            **status,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get performance stats: {str(e)}"
        )


@router.get("/performance/summary")
async def get_performance_summary() -> dict[str, Any]:
    """Get aggregate performance summary.

    Returns:
        Dictionary with aggregate statistics
    """
    try:
        from ..dependencies import app_state

        if not hasattr(app_state, "vision_module") or app_state.vision_module is None:
            raise HTTPException(status_code=503, detail="Vision module not initialized")

        vision = app_state.vision_module

        if not hasattr(vision, "profiler") or vision.profiler is None:
            raise HTTPException(
                status_code=503, detail="Performance profiling is disabled"
            )

        # Get aggregate stats
        stats = vision.profiler.get_current_stats()
        summary = stats.get_summary()

        return {"profiling_enabled": True, **summary}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get performance summary: {str(e)}"
        )


@router.get("/performance/bottlenecks")
async def get_bottlenecks(top_n: int = 5) -> dict[str, Any]:
    """Get top performance bottlenecks.

    Args:
        top_n: Number of top bottlenecks to return

    Returns:
        Dictionary with bottleneck analysis
    """
    try:
        from ..dependencies import app_state

        if not hasattr(app_state, "vision_module") or app_state.vision_module is None:
            raise HTTPException(status_code=503, detail="Vision module not initialized")

        vision = app_state.vision_module

        if not hasattr(vision, "profiler") or vision.profiler is None:
            raise HTTPException(
                status_code=503, detail="Performance profiling is disabled"
            )

        # Get bottlenecks
        bottlenecks = vision.profiler.get_bottlenecks(top_n=top_n)

        return {
            "profiling_enabled": True,
            "bottlenecks": [
                {"rank": i + 1, "stage": stage, "time_ms": time_ms}
                for i, (stage, time_ms) in enumerate(bottlenecks)
            ],
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get bottlenecks: {str(e)}"
        )
