"""Debug endpoints for testing ball detection and websocket events."""

import logging
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/debug", tags=["Debug"])

# Path to static HTML files
STATIC_DIR = Path(__file__).parent.parent.parent / "static" / "debug"


@router.get("/ball-detection", response_class=FileResponse)
async def ball_detection_debug_page():
    """Serve the ball detection debug page.

    This page:
    - Shows the video stream using MJPEG
    - Connects to the websocket to receive game state events
    - Renders ball detection rings and boundaries as an overlay
    - Displays event data and performance metrics in real-time
    """
    html_file = STATIC_DIR / "ball-detection.html"
    return FileResponse(html_file)
