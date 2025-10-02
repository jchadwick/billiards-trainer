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

from ..dependencies import get_core_module
from ..models.common import ErrorCode, create_error_response
from ..models.responses import (
    BallInfo,
    CueInfo,
    GameEvent,
    GameHistoryResponse,
    GameResetResponse,
    GameStateResponse,
    SessionExportResponse,
    TableInfo,
)

try:
    from ...core import CoreModule
    from ...core.models import BallState, CueState, GameState, GameType, TableState
except ImportError:
    from core import CoreModule
    from core.models import BallState, CueState, GameState, GameType, TableState

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/game", tags=["Game State Management"])


def convert_ball_state_to_info(ball: BallState) -> BallInfo:
    """Convert core BallState to API BallInfo."""
    return BallInfo(
        id=ball.id,
        number=ball.number,
        position=[ball.position.x, ball.position.y],
        velocity=[ball.velocity.x, ball.velocity.y],
        is_cue_ball=ball.is_cue_ball,
        is_pocketed=ball.is_pocketed,
        confidence=ball.confidence,
        last_update=datetime.fromtimestamp(ball.last_update, timezone.utc),
    )


def convert_cue_state_to_info(cue: CueState) -> CueInfo:
    """Convert core CueState to API CueInfo."""
    return CueInfo(
        tip_position=[cue.tip_position.x, cue.tip_position.y],
        angle=cue.angle,
        elevation=cue.elevation,
        estimated_force=cue.estimated_force,
        is_visible=cue.is_visible,
        confidence=cue.confidence,
    )


def convert_table_state_to_info(table: TableState) -> TableInfo:
    """Convert core TableState to API TableInfo."""
    return TableInfo(
        width=table.width,
        height=table.height,
        pocket_positions=[[p.x, p.y] for p in table.pocket_positions],
        pocket_radius=table.pocket_radius,
        surface_friction=table.surface_friction,
    )


def convert_game_state_to_response(game_state: GameState) -> GameStateResponse:
    """Convert core GameState to API GameStateResponse."""
    # Convert balls
    balls = [convert_ball_state_to_info(ball) for ball in game_state.balls]

    # Convert cue if present
    cue = convert_cue_state_to_info(game_state.cue) if game_state.cue else None

    # Convert table
    table = convert_table_state_to_info(game_state.table)

    # Convert events
    events = [
        GameEvent(
            timestamp=datetime.fromtimestamp(event.timestamp, timezone.utc),
            event_type=event.event_type,
            description=event.description,
            data=event.data,
        )
        for event in game_state.events
    ]

    return GameStateResponse(
        timestamp=datetime.fromtimestamp(game_state.timestamp, timezone.utc),
        frame_number=game_state.frame_number,
        balls=balls,
        cue=cue,
        table=table,
        game_type=game_state.game_type.value,
        is_valid=game_state.is_valid,
        confidence=game_state.state_confidence,
        events=events,
    )


@router.get("/state", response_model=GameStateResponse)
async def get_current_game_state(
    include_events: bool = Query(True, description="Include recent game events"),
    include_trajectories: bool = Query(
        False, description="Include trajectory predictions"
    ),
    core_module: CoreModule = Depends(get_core_module),
) -> GameStateResponse:
    """Retrieve current game state snapshot (FR-API-013).

    Returns the complete current state of the billiards game including
    ball positions, cue stick state, table configuration, and optional
    trajectory predictions.
    """
    try:
        # Get current game state from core module
        if hasattr(core_module, "get_current_state"):
            game_state = core_module.get_current_state()
        else:
            # Fallback: create a basic game state
            from ...core.models import GameState, GameType, TableState

            game_state = GameState.create_initial_state(GameType.PRACTICE)

        if not game_state:
            # Return empty state if no game is active
            from ...core.models import TableState

            empty_table = TableState.standard_9ft_table()

            return GameStateResponse(
                timestamp=datetime.now(timezone.utc),
                frame_number=0,
                balls=[],
                cue=None,
                table=convert_table_state_to_info(empty_table),
                game_type="practice",
                is_valid=True,
                confidence=1.0,
                events=[],
            )

        # Convert to response format
        response = convert_game_state_to_response(game_state)

        # Filter events if not requested
        if not include_events:
            response.events = []

        # Add trajectory predictions if requested
        if include_trajectories and hasattr(core_module, "predict_trajectories"):
            try:
                # This would be implemented in the core module
                core_module.predict_trajectories(game_state)
                # Add trajectories to response (would need to extend response model)
            except Exception as e:
                logger.warning(f"Failed to get trajectory predictions: {e}")

        logger.info("Current game state retrieved")
        return response

    except Exception as e:
        logger.error(f"Failed to retrieve current game state: {e}")
        raise HTTPException(
            status_code=500,
            detail=create_error_response(
                "Game State Retrieval Failed",
                "Unable to retrieve current game state",
                ErrorCode.SYSTEM_INTERNAL_ERROR,
                {"error": str(e)},
            ),
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
                detail=create_error_response(
                    "Invalid Time Range",
                    "End time must be after start time",
                    ErrorCode.VAL_PARAMETER_OUT_OF_RANGE,
                    {"start_time": start_time, "end_time": end_time},
                ),
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
            detail=create_error_response(
                "Game History Retrieval Failed",
                "Unable to retrieve game history",
                ErrorCode.SYSTEM_INTERNAL_ERROR,
                {"error": str(e)},
            ),
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
                detail=create_error_response(
                    "Invalid Game Type",
                    f"Game type '{game_type}' is not supported",
                    ErrorCode.VAL_INVALID_FORMAT,
                    {"supported_types": ["practice", "8ball", "9ball", "straight"]},
                ),
            )

        # Reset game state
        try:
            if hasattr(core_module, "reset_game_state"):
                new_game_state = core_module.reset_game_state(
                    game_type=game_type_enum, preserve_table=preserve_table
                )
            else:
                # Fallback: create new initial state
                new_game_state = GameState.create_initial_state(game_type_enum)

        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=create_error_response(
                    "Game Reset Failed",
                    f"Failed to reset game state: {str(e)}",
                    ErrorCode.SYSTEM_INTERNAL_ERROR,
                    {"error": str(e)},
                ),
            )

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
            detail=create_error_response(
                "Game Reset Failed",
                "Unable to reset game state",
                ErrorCode.SYSTEM_INTERNAL_ERROR,
                {"error": str(e)},
            ),
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
        export_data = {
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

            with open(file_path, "w") as f:
                json.dump(export_data, f, indent=2, default=str)

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

        with open(file_path, "rb") as f:
            checksum = hashlib.sha256(f.read()).hexdigest()

        # Set expiration time (24 hours from now)
        expires_at = datetime.now(timezone.utc) + timedelta(hours=24)

        logger.info(f"Session data exported: {filename}")

        # Schedule cleanup of export file
        async def cleanup_export_file():
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
            detail=create_error_response(
                "Session Export Failed",
                "Unable to export session data",
                ErrorCode.SYSTEM_INTERNAL_ERROR,
                {"error": str(e)},
            ),
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
                detail=create_error_response(
                    "Export Not Found",
                    f"Export file with ID '{export_id}' not found or has expired",
                    ErrorCode.RES_NOT_FOUND,
                    {"export_id": export_id},
                ),
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
            detail=create_error_response(
                "Export Download Failed",
                "Unable to download export file",
                ErrorCode.SYSTEM_INTERNAL_ERROR,
                {"error": str(e)},
            ),
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
            detail=create_error_response(
                "Statistics Retrieval Failed",
                "Unable to retrieve game statistics",
                ErrorCode.SYSTEM_INTERNAL_ERROR,
                {"error": str(e)},
            ),
        )
