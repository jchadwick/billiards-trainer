"""Logs API endpoints for retrieving application logs."""

import logging
import os
from pathlib import Path
from typing import Any, Literal, Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, PlainTextResponse

try:
    from ...utils.logging import get_log_directory
except ImportError:
    # Fallback for when relative import fails
    def get_log_directory() -> Path:
        """Get the configured log directory path."""
        return Path(os.getenv("LOG_DIR", "logs"))


logger = logging.getLogger(__name__)

router = APIRouter(tags=["logs"])


@router.get("/logs/files")
async def list_log_files() -> dict[str, list[dict[str, Any]]]:
    """List all available log files.

    Returns:
        Dictionary containing list of log files with metadata
    """
    try:
        log_dir = get_log_directory()

        if not log_dir.exists():
            return {"files": []}

        files = []
        for log_file in sorted(
            log_dir.glob("*.log*"), key=lambda p: p.stat().st_mtime, reverse=True
        ):
            stat = log_file.stat()
            files.append(
                {
                    "name": log_file.name,
                    "path": str(log_file.relative_to(log_dir)),
                    "size": stat.st_size,
                    "modified": stat.st_mtime,
                }
            )

        return {"files": files}

    except Exception as e:
        logger.error(f"Error listing log files: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/logs/view/{filename}")
async def view_log_file(
    filename: str,
    lines: Optional[int] = Query(
        None, description="Number of lines to return from end of file"
    ),
    offset: Optional[int] = Query(
        0, description="Line offset from start (if lines not specified)"
    ),
) -> PlainTextResponse:
    """View contents of a log file.

    Args:
        filename: Name of the log file
        lines: Number of lines to return from end of file (tail)
        offset: Line offset from start if lines not specified

    Returns:
        Log file contents as plain text
    """
    try:
        log_dir = get_log_directory()
        log_file = log_dir / filename

        # Security: ensure file is within log directory
        if not log_file.resolve().is_relative_to(log_dir.resolve()):
            raise HTTPException(status_code=403, detail="Access denied")

        if not log_file.exists():
            raise HTTPException(status_code=404, detail="Log file not found")

        # Read file contents
        with open(log_file, encoding="utf-8") as f:
            if lines is not None:
                # Return last N lines (tail)
                all_lines = f.readlines()
                content = "".join(all_lines[-lines:])
            else:
                # Return from offset
                all_lines = f.readlines()
                content = "".join(all_lines[offset:])

        return PlainTextResponse(content=content)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error reading log file {filename}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/logs/download/{filename}")
async def download_log_file(filename: str) -> FileResponse:
    """Download a log file.

    Args:
        filename: Name of the log file

    Returns:
        Log file as downloadable attachment
    """
    try:
        log_dir = get_log_directory()
        log_file = log_dir / filename

        # Security: ensure file is within log directory
        if not log_file.resolve().is_relative_to(log_dir.resolve()):
            raise HTTPException(status_code=403, detail="Access denied")

        if not log_file.exists():
            raise HTTPException(status_code=404, detail="Log file not found")

        return FileResponse(
            path=log_file,
            filename=filename,
            media_type="text/plain",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading log file {filename}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/logs/tail/{log_type}")
async def tail_log(
    log_type: Literal["app", "error"] = "app",
    lines: int = Query(100, description="Number of lines to return", ge=1, le=10000),
) -> PlainTextResponse:
    """Get the tail of a specific log file.

    Args:
        log_type: Type of log file (app or error)
        lines: Number of lines to return from end

    Returns:
        Last N lines from the specified log
    """
    filename = f"{log_type}.log"
    return await view_log_file(filename=filename, lines=lines)


@router.delete("/logs/clear/{log_type}")
async def clear_log(
    log_type: Literal["app", "error"] = "app",
) -> dict[str, str]:
    """Clear a log file (truncate to empty).

    Args:
        log_type: Type of log file to clear (app or error)

    Returns:
        Success message
    """
    try:
        log_dir = get_log_directory()
        log_file = log_dir / f"{log_type}.log"

        if not log_file.exists():
            raise HTTPException(status_code=404, detail="Log file not found")

        # Truncate the file
        with open(log_file, "w", encoding="utf-8") as f:
            f.truncate(0)

        logger.info(f"Cleared {log_type}.log")
        return {"message": f"Log file {log_type}.log cleared successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error clearing log file {log_type}.log: {e}")
        raise HTTPException(status_code=500, detail=str(e))
