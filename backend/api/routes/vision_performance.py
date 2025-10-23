"""Vision performance monitoring API endpoints."""

from typing import Any, Dict

from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/vision", tags=["vision-performance"])


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
                "message": "Enable profiling in config: vision.performance.enable_profiling = true",
                "profiling_enabled": False,
            }

        # Get real-time status from profiler
        status = vision.profiler.get_realtime_status()

        return {"profiling_enabled": True, **status}

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

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get bottlenecks: {str(e)}"
        )
