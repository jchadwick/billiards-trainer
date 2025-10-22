"""Debug endpoints for testing ball detection and websocket events."""

import logging

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/debug", tags=["Debug"])


@router.get("/ball-detection", response_class=HTMLResponse)
async def ball_detection_debug_page():
    """Serve a simple HTML page to debug ball detection and websocket events.

    This page:
    - Shows the video stream using MJPEG
    - Connects to the websocket to receive game state events
    - Renders ball detection rings and boundaries as an overlay
    - Displays event data in real-time
    """
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Ball Detection Debug</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #1a1a1a;
            color: #ffffff;
            padding: 20px;
        }

        .container {
            max-width: 1800px;
            margin: 0 auto;
        }

        h1 {
            margin-bottom: 20px;
            color: #4CAF50;
        }

        .layout {
            display: grid;
            grid-template-columns: 1fr 400px;
            gap: 20px;
        }

        .video-container {
            position: relative;
            background: #000;
            border-radius: 8px;
            overflow: hidden;
        }

        #videoStream {
            width: 100%;
            height: auto;
            display: block;
        }

        #overlay {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
        }

        .info-panel {
            background: #2a2a2a;
            border-radius: 8px;
            padding: 20px;
            overflow-y: auto;
            max-height: 800px;
        }

        .status {
            margin-bottom: 20px;
        }

        .status-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
        }

        .status-indicator.connected {
            background: #4CAF50;
            box-shadow: 0 0 8px #4CAF50;
        }

        .status-indicator.disconnected {
            background: #f44336;
        }

        .section {
            margin-bottom: 20px;
            padding: 15px;
            background: #333;
            border-radius: 4px;
        }

        .section h3 {
            margin-bottom: 10px;
            color: #4CAF50;
            font-size: 14px;
            text-transform: uppercase;
        }

        .metric {
            display: flex;
            justify-content: space-between;
            margin-bottom: 8px;
            font-size: 13px;
        }

        .metric-label {
            color: #999;
        }

        .metric-value {
            color: #fff;
            font-weight: bold;
        }

        .ball-list {
            max-height: 300px;
            overflow-y: auto;
        }

        .ball-item {
            padding: 8px;
            margin-bottom: 6px;
            background: #444;
            border-radius: 4px;
            font-size: 12px;
        }

        .event-log {
            max-height: 200px;
            overflow-y: auto;
            font-size: 11px;
            font-family: 'Courier New', monospace;
        }

        .event-log-item {
            padding: 4px;
            margin-bottom: 2px;
            border-left: 3px solid #4CAF50;
            padding-left: 8px;
            color: #aaa;
        }

        .controls {
            margin-bottom: 20px;
        }

        button {
            background: #4CAF50;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 4px;
            cursor: pointer;
            margin-right: 10px;
        }

        button:hover {
            background: #45a049;
        }

        button.secondary {
            background: #666;
        }

        button.secondary:hover {
            background: #777;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎱 Ball Detection Debug Interface</h1>

        <div class="layout">
            <div>
                <div class="controls">
                    <button id="connectBtn">Connect WebSocket</button>
                    <button id="disconnectBtn" class="secondary">Disconnect</button>
                    <button id="clearEventsBtn" class="secondary">Clear Events</button>
                </div>

                <div class="video-container">
                    <img id="videoStream" src="/api/v1/stream/video?fps=30&quality=80" alt="Video Stream">
                    <canvas id="overlay"></canvas>
                </div>
            </div>

            <div class="info-panel">
                <div class="status">
                    <h2 style="font-size: 16px; margin-bottom: 10px;">Connection Status</h2>
                    <span class="status-indicator" id="wsStatus"></span>
                    <span id="wsStatusText">Disconnected</span>
                </div>

                <div class="section">
                    <h3>Statistics</h3>
                    <div class="metric">
                        <span class="metric-label">Events Received:</span>
                        <span class="metric-value" id="eventCount">0</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Balls Detected:</span>
                        <span class="metric-value" id="ballCount">0</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Cue Detected:</span>
                        <span class="metric-value" id="cueDetected">No</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Last Update:</span>
                        <span class="metric-value" id="lastUpdate">Never</span>
                    </div>
                </div>

                <div class="section">
                    <h3>Detected Balls</h3>
                    <div class="ball-list" id="ballList">
                        <div style="color: #666;">No balls detected</div>
                    </div>
                </div>

                <div class="section">
                    <h3>Event Log (Last 20)</h3>
                    <div class="event-log" id="eventLog">
                        <div class="event-log-item">Waiting for events...</div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        let ws = null;
        let eventCount = 0;
        let gameState = { balls: [], cue: null, table: null };
        const eventLog = [];
        const MAX_LOG_ENTRIES = 20;

        // Get elements
        const videoEl = document.getElementById('videoStream');
        const canvas = document.getElementById('overlay');
        const ctx = canvas.getContext('2d');
        const connectBtn = document.getElementById('connectBtn');
        const disconnectBtn = document.getElementById('disconnectBtn');
        const clearEventsBtn = document.getElementById('clearEventsBtn');

        // Setup canvas to match video
        // Note: MJPEG streams don't fire 'load' events, so we need to poll for size changes
        window.addEventListener('resize', updateCanvasSize);

        function updateCanvasSize() {
            // Get the actual displayed size of the video element
            const rect = videoEl.getBoundingClientRect();
            if (rect.width > 0 && rect.height > 0) {
                canvas.width = rect.width;
                canvas.height = rect.height;
                canvas.style.width = rect.width + 'px';
                canvas.style.height = rect.height + 'px';
                drawOverlay();
            }
        }

        // Poll for video size changes (MJPEG streams don't fire load events)
        setInterval(updateCanvasSize, 500);

        // WebSocket connection
        function connect() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${protocol}//${window.location.host}/api/v1/ws`;

            addToLog(`Connecting to ${wsUrl}...`);
            ws = new WebSocket(wsUrl);

            ws.onopen = () => {
                updateStatus(true);
                addToLog('WebSocket connected!');

                // Subscribe to game state updates - correct format expected by backend
                ws.send(JSON.stringify({
                    type: 'subscribe',
                    data: {
                        streams: ['state']  // Array of stream types
                    }
                }));
                addToLog('Sent subscription request for state stream');
            };

            ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    eventCount++;
                    document.getElementById('eventCount').textContent = eventCount;

                    addToLog(`Event: ${data.type || 'unknown'}`);

                    // Handle different message types
                    if (data.type === 'state' && data.data) {
                        // Main state stream from WebSocket
                        handleGameStateUpdate(data.data);
                    } else if (data.type === 'game_state' && data.data) {
                        // Legacy/alternative format
                        handleGameStateUpdate(data.data);
                    } else if (data.type === 'state_updated' && data.data) {
                        // Event format
                        handleGameStateUpdate(data.data);
                    } else {
                        addToLog(`Other event: ${JSON.stringify(data).substring(0, 100)}`);
                    }
                } catch (error) {
                    addToLog(`Error parsing message: ${error.message}`);
                }
            };

            ws.onerror = (error) => {
                addToLog(`WebSocket error: ${error}`);
            };

            ws.onclose = () => {
                updateStatus(false);
                addToLog('WebSocket disconnected');
                ws = null;
            };
        }

        function disconnect() {
            if (ws) {
                ws.close();
            }
        }

        function handleGameStateUpdate(data) {
            gameState = data;

            // Debug: Log received data (only log coordinate_metadata if it changed)
            if (!window.lastLoggedCoordMetadata ||
                JSON.stringify(data.coordinate_metadata) !== window.lastLoggedCoordMetadata) {
                console.log('Game state update - coordinate_metadata:', data.coordinate_metadata);
                window.lastLoggedCoordMetadata = JSON.stringify(data.coordinate_metadata);
            }

            // Update ball count
            const ballCount = data.balls ? data.balls.length : 0;
            document.getElementById('ballCount').textContent = ballCount;

            // Update cue detection
            const cueDetected = data.cue ? 'Yes' : 'No';
            document.getElementById('cueDetected').textContent = cueDetected;

            // Update last update time
            const now = new Date().toLocaleTimeString();
            document.getElementById('lastUpdate').textContent = now;

            // Update ball list
            updateBallList();

            // Draw overlay
            drawOverlay();

            addToLog(`State: ${ballCount} balls, cue: ${cueDetected}`);
        }

        function updateBallList() {
            const ballList = document.getElementById('ballList');

            if (!gameState.balls || gameState.balls.length === 0) {
                ballList.innerHTML = '<div style="color: #666;">No balls detected</div>';
                return;
            }

            ballList.innerHTML = gameState.balls.map((ball, idx) => {
                const pos = ball.position || ball.pos || {};
                const radius = ball.radius || 'N/A';
                const scale = pos.scale ? `[${pos.scale[0].toFixed(1)}, ${pos.scale[1].toFixed(1)}]` : 'N/A';
                return `
                    <div class="ball-item">
                        Ball #${idx + 1}: (${pos.x?.toFixed(0) || '?'}, ${pos.y?.toFixed(0) || '?'})
                        scale=${scale}
                        r=${typeof radius === 'number' ? radius.toFixed(1) : radius}
                    </div>
                `;
            }).join('');
        }

        function drawOverlay() {
            if (!canvas.width || !canvas.height) {
                console.log('Canvas not sized yet');
                return;
            }

            ctx.clearRect(0, 0, canvas.width, canvas.height);

            // Get scale metadata from first ball's position
            // All positions now include mandatory scale metadata
            const firstBall = gameState.balls && gameState.balls[0];
            let sourceResolution;

            if (firstBall && firstBall.position && firstBall.position.scale) {
                // Scale factors tell us conversion: source_res = 4k / scale
                // For 1080p: scale = [2.0, 2.0], so source = [3840/2.0, 2160/2.0] = [1920, 1080]
                // For 4K: scale = [1.0, 1.0], so source = [3840/1.0, 2160/1.0] = [3840, 2160]
                const scaleX = firstBall.position.scale[0];
                const scaleY = firstBall.position.scale[1];
                sourceResolution = {
                    width: 3840 / scaleX,   // 4K canonical width / scale
                    height: 2160 / scaleY   // 4K canonical height / scale
                };
                console.log(`Detected source resolution from scale [${scaleX}, ${scaleY}]: ${sourceResolution.width}x${sourceResolution.height}`);
            } else {
                // Fallback: assume 4K canonical (scale = [1.0, 1.0])
                sourceResolution = { width: 3840, height: 2160 };
                console.log(`No scale metadata found, assuming 4K canonical: ${sourceResolution.width}x${sourceResolution.height}`);
            }

            // Calculate scale factors from source resolution to canvas
            const scaleX = canvas.width / sourceResolution.width;
            const scaleY = canvas.height / sourceResolution.height;

            console.log(`Canvas: ${canvas.width}x${canvas.height}, Source: ${sourceResolution.width}x${sourceResolution.height}, Scale: ${scaleX.toFixed(3)}x${scaleY.toFixed(3)}`);

            // Draw table boundary
            if (gameState.table && gameState.table.boundary) {
                const boundary = gameState.table.boundary;
                ctx.strokeStyle = '#00ff00';
                ctx.lineWidth = 2;
                ctx.strokeRect(
                    boundary.x * scaleX,
                    boundary.y * scaleY,
                    boundary.width * scaleX,
                    boundary.height * scaleY
                );
            }

            // Draw balls
            if (gameState.balls && gameState.balls.length > 0) {
                gameState.balls.forEach((ball, idx) => {
                    const pos = ball.position || ball.pos;
                    if (!pos || pos.x === undefined || pos.y === undefined) return;

                    // Position is in source resolution (determined by scale)
                    // Convert to 4K canonical first, then to canvas
                    const pos4k = {
                        x: pos.x * (pos.scale ? pos.scale[0] : 1.0),
                        y: pos.y * (pos.scale ? pos.scale[1] : 1.0)
                    };

                    // Now convert from 4K canonical to canvas
                    const x = pos4k.x * (canvas.width / 3840);
                    const y = pos4k.y * (canvas.height / 2160);

                    // Scale radius from 4K canonical (36px) to canvas
                    const radius4k = ball.radius || 36;  // Default 36px for 4K
                    const radius = radius4k * (canvas.width / 3840);

                    // Draw ball circle
                    ctx.strokeStyle = '#00ffff';
                    ctx.lineWidth = 2;
                    ctx.beginPath();
                    ctx.arc(x, y, radius, 0, 2 * Math.PI);
                    ctx.stroke();

                    // Draw ball number
                    ctx.fillStyle = '#00ffff';
                    ctx.font = '14px Arial';
                    ctx.fillText(`#${idx + 1}`, x + radius + 5, y + 5);
                });
            }

            // Draw cue
            if (gameState.cue && gameState.cue.line) {
                const line = gameState.cue.line;
                if (line.start && line.end) {
                    ctx.strokeStyle = '#ff00ff';
                    ctx.lineWidth = 3;
                    ctx.beginPath();
                    ctx.moveTo(line.start.x * scaleX, line.start.y * scaleY);
                    ctx.lineTo(line.end.x * scaleX, line.end.y * scaleY);
                    ctx.stroke();
                }
            }
        }

        function updateStatus(connected) {
            const indicator = document.getElementById('wsStatus');
            const text = document.getElementById('wsStatusText');

            if (connected) {
                indicator.classList.add('connected');
                indicator.classList.remove('disconnected');
                text.textContent = 'Connected';
            } else {
                indicator.classList.remove('connected');
                indicator.classList.add('disconnected');
                text.textContent = 'Disconnected';
            }
        }

        function addToLog(message) {
            const timestamp = new Date().toLocaleTimeString();
            eventLog.unshift(`[${timestamp}] ${message}`);
            if (eventLog.length > MAX_LOG_ENTRIES) {
                eventLog.pop();
            }

            const logEl = document.getElementById('eventLog');
            logEl.innerHTML = eventLog.map(entry =>
                `<div class="event-log-item">${entry}</div>`
            ).join('');
        }

        function clearEvents() {
            eventLog.length = 0;
            eventCount = 0;
            document.getElementById('eventCount').textContent = '0';
            document.getElementById('eventLog').innerHTML =
                '<div class="event-log-item">Events cleared</div>';
        }

        // Button handlers
        connectBtn.addEventListener('click', connect);
        disconnectBtn.addEventListener('click', disconnect);
        clearEventsBtn.addEventListener('click', clearEvents);

        // Auto-connect on load and initialize canvas
        setTimeout(() => {
            connect();
            updateCanvasSize(); // Initial canvas setup
        }, 500);
    </script>
</body>
</html>
    """
    return HTMLResponse(content=html_content)
