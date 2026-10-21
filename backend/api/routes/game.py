"""Game state endpoints for accessing and manipulating game data.

Provides comprehensive game state management including:
- Current game state retrieval (FR-API-013)
- Historical game state access (FR-API-014)
- Game state reset functionality (FR-API-015)
- Session data export (FR-API-016)
"""

import asyncio
import json
import logging
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from fastapi.responses import FileResponse

from backend.api.dependencies import (
    get_app_state,
    get_core_module,
    get_optional_core_module,
)
from backend.api.models.converters import (
    ball_state_to_ball_info,
    cue_state_to_cue_info,
    game_state_to_response,
    scene_state_to_response,
    table_state_to_table_info,
)
from backend.api.models.responses import (
    BallInfo,
    CueInfo,
    GameHistoryResponse,
    GameResetResponse,
    GameStateResponse,
    SceneGameStateResponse,
    SessionExportResponse,
    TableInfo,
)
from backend.core import CoreModule
from backend.core.models import BallState, CueState, GameState, GameType, TableState

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/game", tags=["Game State Management"])


def _live_trajectory_is_eligible(
    observation_status: Optional[dict[str, Any]], scheduler_enabled: bool
) -> bool:
    """Apply the vision scheduler gate to API-requested live predictions.

    When the scheduler is opt-in and disabled, preserve the legacy endpoint
    behavior. Once enabled, unknown or stale phase state fails closed so a
    request cannot trigger new aim calculations during motion or recovery.
    """
    phase = str((observation_status or {}).get("phase", "")).lower().replace("-", "_")
    if phase in {
        "strike",
        "limited_motion",
        "many_balls_moving",
        "many_moving",
        "recovery",
        "recovering",
        "uncertain",
    }:
        return False
    if not scheduler_enabled and not (observation_status or {}).get("phase_aware"):
        status = observation_status or {}
        return not bool(
            status.get("shot_detected", False)
            or status.get("balls_in_motion", 0)
            or str(status.get("game_state", "")).lower()
            in {"moving", "shooting", "strike"}
        )
    if not observation_status:
        return False
    return (
        observation_status.get("aim_calculation_eligible") is True
        and observation_status.get("map_trusted") is True
    )


def convert_ball_state_to_info(ball: BallState) -> BallInfo:
    """Convert a core ball through the shared transport-field mapper."""
    return ball_state_to_ball_info(ball)


def convert_cue_state_to_info(cue: CueState) -> CueInfo:
    """Convert a core cue through the shared transport-field mapper."""
    return cue_state_to_cue_info(cue)


def convert_table_state_to_info(table: TableState) -> TableInfo:
    """Convert a core table through the shared coordinate mapper."""
    return table_state_to_table_info(table)


def convert_game_state_to_response(game_state: GameState) -> GameStateResponse:
    """Convert core GameState to API GameStateResponse."""
    return game_state_to_response(game_state)


def _get_current_core_state(core_module: CoreModule | None) -> GameState | None:
    """Read the retained Core state when Core is available and initialized."""
    if core_module is None:
        return None
    game_state = core_module.get_current_state()
    if game_state is None:
        state_manager = getattr(core_module, "state_manager", None)
        if state_manager is not None:
            game_state = state_manager.get_current_state()
    return game_state


@router.get("/state", response_model=SceneGameStateResponse | GameStateResponse)
async def get_current_game_state(
    include_events: bool = Query(True, description="Include recent game events"),
    include_trajectories: bool = Query(
        False, description="Include trajectory predictions"
    ),
    core_module: Optional[CoreModule] = Depends(get_optional_core_module),
    app_state: Any = Depends(get_app_state),
) -> GameStateResponse | SceneGameStateResponse:
    """Retrieve current game state snapshot (FR-API-013).

    Returns the complete current state of the billiards game including
    ball positions, cue stick state, table configuration, and optional
    trajectory predictions.
    """
    try:
        # Scene localization is the primary read path and remains available
        # while optional game logic is starting, stopped, or uninitialized.
        scene_publisher = getattr(app_state, "scene_state_publisher", None)
        scene_read_model = (
            getattr(scene_publisher, "read_model", None)
            if scene_publisher is not None
            else None
        )
        response: GameStateResponse | SceneGameStateResponse
        if scene_read_model is not None:
            game_state = _get_current_core_state(core_module)
            response = scene_state_to_response(
                scene_read_model,
                game_state,
                include_events=include_events,
            )
            logger.info("Current scene state retrieved")
            return response

        # The Python-only fallback still requires its original initialized
        # Core state. Do not change its established 503 behavior.
        if core_module is None:
            raise HTTPException(status_code=503, detail="Core module not available")
        game_state = _get_current_core_state(core_module)
        if not game_state:
            raise HTTPException(
                status_code=503,
                detail="Game state is unavailable until the table is calibrated",
            )

        # Keep the Python Core response path unchanged when no scene publisher
        # has been attached by application lifecycle management.
        response = convert_game_state_to_response(game_state)

        # Filter events if not requested
        if not include_events:
            response.events = []

        # Add trajectory predictions if requested
        integration = getattr(app_state, "integration_service", None)
        observation_status = getattr(integration, "latest_observation_status", None)
        config_module = getattr(app_state, "config_module", None)
        scheduler_enabled = bool(
            config_module
            and config_module.get("vision.processing.phase_scheduler.enabled", False)
        )
        if (
            include_trajectories
            and _live_trajectory_is_eligible(observation_status, scheduler_enabled)
            and hasattr(core_module, "predict_trajectories")
        ):
            try:
                # This would be implemented in the core module
                core_module.predict_trajectories(game_state)
                # Add trajectories to response (would need to extend response model)
            except Exception as e:
                logger.warning(f"Failed to get trajectory predictions: {e}")

        logger.info("Current game state retrieved")
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to retrieve current game state: {e}")
        raise HTTPException(
            status_code=500,
            detail="Unable to retrieve current game state",
        )


@router.get("/history", response_model=GameHistoryResponse)
async def get_game_history(
    start_time: Optional[datetime] = Query(
        None, description="Start time for history query"
    ),
    end_time: Optional[datetime] = Query(
        None, description="End time for history query"
    ),
    limit: int = Query(
        100, ge=1, le=10000, description="Maximum number of states to return"
    ),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    include_events: bool = Query(False, description="Include game events in results"),
    game_type: Optional[str] = Query(
        None,
        pattern="^(practice|8ball|9ball|straight)$",
        description="Filter by game type",
    ),
    core_module: CoreModule = Depends(get_core_module),
) -> GameHistoryResponse:
    """Access historical game states (FR-API-014).

    Returns paginated historical game state data with optional filtering
    by time range, game type, and event inclusion.
    """
    try:
        # Validate time range
        if start_time and end_time and end_time <= start_time:
            raise HTTPException(
                status_code=400,
                detail="End time must be after start time",
            )

        # Get historical states from core module
        historical_states = []
        total_count = 0
        has_more = False

        if hasattr(core_module, "get_game_history"):
            try:
                history_data = core_module.get_game_history(
                    start_time=start_time,
                    end_time=end_time,
                    limit=limit,
                    offset=offset,
                    game_type=game_type,
                )
                historical_states = history_data.get("states", [])
                total_count = history_data.get("total_count", 0)
                has_more = offset + len(historical_states) < total_count
            except Exception as e:
                logger.warning(f"Core module history retrieval failed: {e}")
                # Return empty history if core module doesn't support it

        # Convert states to response format
        response_states = []
        for state in historical_states:
            if isinstance(state, GameState):
                response_state = convert_game_state_to_response(state)
                if not include_events:
                    response_state.events = []
                response_states.append(response_state)

        # Determine actual time range of results
        actual_time_range = {}
        if response_states:
            timestamps = [state.timestamp for state in response_states]
            actual_time_range = {"start": min(timestamps), "end": max(timestamps)}

        logger.info(f"Game history retrieved: {len(response_states)} states")

        return GameHistoryResponse(
            states=response_states,
            total_count=total_count,
            has_more=has_more,
            time_range=actual_time_range,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to retrieve game history: {e}")
        raise HTTPException(
            status_code=500,
            detail="Unable to retrieve game history",
        )


@router.post("/reset", response_model=GameResetResponse)
async def reset_game_state(
    game_type: str = Query(
        "practice",
        pattern="^(practice|8ball|9ball|straight)$",
        description="Game type to initialize",
    ),
    preserve_table: bool = Query(True, description="Keep existing table configuration"),
    create_backup: bool = Query(True, description="Create backup of current state"),
    core_module: CoreModule = Depends(get_core_module),
) -> GameResetResponse:
    """Reset game state tracking (FR-API-015).

    Resets the game to initial state with specified game type and
    optional table preservation and backup creation.
    """
    try:
        # Get current state for backup if requested
        current_state = None
        backup_created = False

        if create_backup:
            try:
                if hasattr(core_module, "get_current_state"):
                    current_state = core_module.get_current_state()
                    if current_state:
                        # Save backup (would implement proper backup storage)
                        backup_timestamp = datetime.now(timezone.utc).strftime(
                            "%Y%m%d_%H%M%S"
                        )
                        backup_path = f"/tmp/game_state_backup_{backup_timestamp}.json"

                        with open(backup_path, "w") as f:
                            json.dump(current_state.to_dict(), f, indent=2, default=str)

                        backup_created = True
                        logger.info(f"Game state backup created at {backup_path}")
            except Exception as e:
                logger.warning(f"Failed to create backup: {e}")
                # Continue with reset even if backup fails

        # Convert game type string to enum
        try:
            game_type_enum = GameType(game_type)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Game type '{game_type}' is not supported",
            )

        # A reset needs a real calibrated table. Preserve the current table by
        # default; when preservation is disabled, use the state manager's
        # configured calibration table if one exists.
        try:
            current_state = core_module.get_current_state()
            if current_state is None:
                current_state = core_module.state_manager.get_current_state()
            configured_table = getattr(
                core_module.state_manager, "_default_table", None
            )
            table = (
                current_state.table
                if preserve_table and current_state is not None
                else configured_table
            )
            if table is None:
                raise HTTPException(
                    status_code=409,
                    detail="A calibrated table is required before resetting the game",
                )

            await core_module.reset_game(game_type_enum)
            # CoreModule.reset_game clears its live snapshot and resets the
            # manager using its configured table. Supplying the resolved table
            # here preserves the API's explicit preserve-table behavior even
            # when a runtime table came from frame detections rather than the
            # manager's startup configuration.
            core_module.state_manager.reset_game(game_type_enum, table_config=table)
            new_game_state = core_module.state_manager.get_current_state()
            if new_game_state is None:
                raise RuntimeError("Game state manager did not create a reset state")

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail="Failed to reset game state",
            ) from e

        # Convert new state to response format
        new_state_response = convert_game_state_to_response(new_game_state)

        logger.warning(f"Game state reset to {game_type}")

        return GameResetResponse(
            success=True,
            new_state=new_state_response,
            backup_created=backup_created,
            reset_at=datetime.now(timezone.utc),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to reset game state: {e}")
        raise HTTPException(
            status_code=500,
            detail="Unable to reset game state",
        )


@router.post("/export", response_model=SessionExportResponse)
async def export_session_data(
    background_tasks: BackgroundTasks,
    session_id: Optional[str] = Query(
        None, description="Specific session ID to export"
    ),
    include_raw_frames: bool = Query(False, description="Include raw frame data"),
    include_processed_data: bool = Query(
        True, description="Include processed vision data"
    ),
    include_trajectories: bool = Query(
        True, description="Include trajectory calculations"
    ),
    include_events: bool = Query(True, description="Include game events"),
    format: str = Query("zip", pattern="^(zip|json)$", description="Export format"),
    compression_level: int = Query(
        6, ge=0, le=9, description="Compression level (0-9)"
    ),
    core_module: CoreModule = Depends(get_core_module),
) -> SessionExportResponse:
    """Export game session data (FR-API-016).

    Creates downloadable export of game session data including
    historical states, events, and optional raw frame data.
    """
    try:
        # Generate export ID
        export_id = f"export_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

        # Collect data to export
        export_data: dict[str, Any] = {
            "metadata": {
                "export_id": export_id,
                "exported_at": datetime.now(timezone.utc).isoformat(),
                "exported_by": "api",
                "session_id": session_id,
                "includes": {
                    "raw_frames": include_raw_frames,
                    "processed_data": include_processed_data,
                    "trajectories": include_trajectories,
                    "events": include_events,
                },
            },
            "data": {},
        }

        # Get current game state
        if hasattr(core_module, "get_current_state"):
            current_state = core_module.get_current_state()
            if current_state:
                export_data["data"]["current_state"] = current_state.to_dict()

        # Get historical data if available
        if hasattr(core_module, "get_game_history"):
            try:
                history = core_module.get_game_history(session_id=session_id)
                if include_processed_data and history:
                    export_data["data"]["history"] = [
                        state.to_dict() for state in history.get("states", [])
                    ]
            except Exception as e:
                logger.warning(f"Failed to include history in export: {e}")

        # Get events if requested
        if include_events:
            try:
                if hasattr(core_module, "get_game_events"):
                    events = core_module.get_game_events(session_id=session_id)
                    export_data["data"]["events"] = events
            except Exception as e:
                logger.warning(f"Failed to include events in export: {e}")

        # Get trajectory data if requested
        if include_trajectories:
            try:
                if hasattr(core_module, "get_trajectory_history"):
                    trajectories = core_module.get_trajectory_history(
                        session_id=session_id
                    )
                    export_data["data"]["trajectories"] = trajectories
            except Exception as e:
                logger.warning(f"Failed to include trajectories in export: {e}")

        # Create export file
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

        if format == "json":
            filename = f"session_export_{timestamp}.json"
            file_path = f"/tmp/{filename}"

            with open(file_path, "w") as text_file:
                json.dump(export_data, text_file, indent=2, default=str)

            file_size = Path(file_path).stat().st_size

        else:  # zip format
            filename = f"session_export_{timestamp}.zip"
            file_path = f"/tmp/{filename}"

            with zipfile.ZipFile(
                file_path, "w", zipfile.ZIP_DEFLATED, compresslevel=compression_level
            ) as zipf:
                # Add main data file
                data_json = json.dumps(export_data, indent=2, default=str)
                zipf.writestr("session_data.json", data_json)

                # Add raw frames if requested (placeholder)
                if include_raw_frames:
                    zipf.writestr(
                        "frames/README.txt", "Raw frame data would be included here"
                    )

            file_size = Path(file_path).stat().st_size

        # Calculate checksum
        import hashlib

        with open(file_path, "rb") as binary_file:
            checksum = hashlib.sha256(binary_file.read()).hexdigest()

        # Set expiration time (24 hours from now)
        expires_at = datetime.now(timezone.utc) + timedelta(hours=24)

        logger.info(f"Session data exported: {filename}")

        # Schedule cleanup of export file
        async def cleanup_export_file() -> None:
            await asyncio.sleep(24 * 3600)  # 24 hours
            try:
                Path(file_path).unlink(missing_ok=True)
                logger.info(f"Cleaned up expired export file: {filename}")
            except Exception as e:
                logger.warning(f"Failed to cleanup export file {filename}: {e}")

        background_tasks.add_task(cleanup_export_file)

        return SessionExportResponse(
            export_id=export_id,
            format=format,
            size=file_size,
            file_path=file_path,
            checksum=checksum,
            created_at=datetime.now(timezone.utc),
            expires_at=expires_at,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to export session data: {e}")
        raise HTTPException(
            status_code=500,
            detail="Unable to export session data",
        )


@router.get("/export/{export_id}/download")
async def download_session_export(export_id: str) -> FileResponse:
    """Download exported session data file.

    Returns the exported session data as a downloadable file.
    """
    try:
        # Find export file (in production, would use proper storage/database)
        possible_files = [
            f"/tmp/session_export_{export_id}.zip",
            f"/tmp/session_export_{export_id}.json",
        ]

        file_path = None
        for path in possible_files:
            if Path(path).exists():
                file_path = path
                break

        if not file_path:
            # Look for files containing the export_id
            import glob

            pattern = f"/tmp/*{export_id}*"
            matching_files = glob.glob(pattern)
            if matching_files:
                file_path = matching_files[0]

        if not file_path or not Path(file_path).exists():
            raise HTTPException(
                status_code=404,
                detail=f"Export file with ID '{export_id}' not found or has expired",
            )

        filename = Path(file_path).name
        media_type = (
            "application/zip" if filename.endswith(".zip") else "application/json"
        )

        logger.info(f"Session export downloaded: {filename}")

        return FileResponse(path=file_path, filename=filename, media_type=media_type)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to download export: {e}")
        raise HTTPException(
            status_code=500,
            detail="Unable to download export file",
        )


@router.get("/stats")
async def get_game_statistics(
    time_range: str = Query(
        "24h", pattern="^(1h|6h|24h|7d|30d)$", description="Time range for statistics"
    ),
    core_module: CoreModule = Depends(get_core_module),
) -> dict[str, Any]:
    """Get game statistics and performance metrics.

    Returns aggregated statistics about game sessions and performance.
    """
    try:
        # Calculate time range
        now = datetime.now(timezone.utc)
        time_deltas = {
            "1h": timedelta(hours=1),
            "6h": timedelta(hours=6),
            "24h": timedelta(hours=24),
            "7d": timedelta(days=7),
            "30d": timedelta(days=30),
        }

        start_time = now - time_deltas[time_range]

        # Collect statistics
        stats = {
            "time_range": {
                "start": start_time.isoformat(),
                "end": now.isoformat(),
                "duration": time_range,
            },
            "games": {
                "total_sessions": 0,
                "total_shots": 0,
                "average_session_duration": 0,
                "game_types": {},
            },
            "performance": {
                "average_fps": 0,
                "tracking_accuracy": 0,
                "successful_predictions": 0,
            },
        }

        # Get actual statistics from core module if available
        if hasattr(core_module, "get_game_statistics"):
            try:
                core_stats = core_module.get_game_statistics(
                    start_time=start_time, end_time=now
                )
                stats.update(core_stats)
            except Exception as e:
                logger.warning(f"Failed to get core statistics: {e}")

        logger.info(f"Game statistics retrieved for {time_range}")

        return stats

    except Exception as e:
        logger.error(f"Failed to get game statistics: {e}")
        raise HTTPException(
            status_code=500,
            detail="Unable to retrieve game statistics",
        )
