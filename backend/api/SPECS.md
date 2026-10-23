# API Module Specification

> This file preserves API requirements and design history. For the current
> transport contract, use [the API reference](../../docs/API_REFERENCE.md) and
> the running OpenAPI document. The active API has one WebSocket route and no
> production UDP publisher for the projector.

## Module Purpose

The API module serves as the primary interface between the backend processing system and all client applications. It provides RESTful endpoints for configuration and control, WebSocket connections for real-time data streaming, and handles validation and communication protocols. The API module acts as a proxy to the Streaming Service for video delivery, ensuring efficient multi-client support without camera access conflicts.

**Note**: This is a single-user training system with no authentication requirements.

## Functional Requirements

### 1. HTTP/REST API Requirements

#### 1.1 System Management
- **FR-API-001**: Provide health check endpoint returning system status
- **FR-API-002**: Expose system version and capability information
- **FR-API-003**: Enable graceful shutdown and restart operations
- **FR-API-004**: Provide performance metrics and statistics

#### 1.2 Configuration Management
- **FR-API-005**: Retrieve current system configuration
- **FR-API-006**: Update configuration parameters with validation
- **FR-API-007**: Reset configuration to defaults
- **FR-API-008**: Import/export configuration files

#### 1.3 Calibration Control
- **FR-API-009**: Initiate calibration sequence
- **FR-API-010**: Capture calibration reference points
- **FR-API-011**: Apply calibration transformations
- **FR-API-012**: Validate calibration accuracy

#### 1.4 Game State Access
- **FR-API-013**: Retrieve current game state snapshot
- **FR-API-014**: Access historical game states
- **FR-API-015**: Reset game state tracking
- **FR-API-016**: Export game session data

**Note**: Module logging endpoints return summary information. Detailed logs are accessible directly on the filesystem for troubleshooting.

### 2. Real-time Communication Requirements

#### 2.1 Real-time Streaming (WebSocket)
- Publish subscribed game-state, trajectory, alert, and configuration updates through the API WebSocket protocol.
- Image-space points and geometry use the explicit scale metadata defined by the current transport schemas.
- WebSocket video/frame transmission is disabled in the current deployment contract; use the HTTP video and frame routes.

#### 2.2 WebSocket Client Management
- Support connection lifecycle, selective stream subscriptions, and bounded per-client delivery.
- Apply configured send deadlines and disconnect clients that cannot keep up with control messages.
- Expose connection and delivery metrics through the WebSocket management routes.

#### 2.3 Projector Integration Boundary
- The LÖVE projector is a separate UDP receiver on localhost port 9999. Its custom `paths`/`ghostBalls` payload is not published by the backend.
- Backend WebSocket trajectories use their own schema and are not interchangeable with the projector UDP format.
- A future adapter would need to explicitly map and scale geometry before the projector can consume backend updates.

#### 2.4 WebSocket Endpoint
- The only connection endpoint is `/api/v1/websocket/ws`.
- Subscription and message schemas are defined in `backend/api/websocket/schemas.py`.

## Interface Specifications

### REST API Endpoints

```yaml
openapi: 3.0.0
paths:
  /api/v1/health:
    get:
      summary: Health check
      responses:
        200:
          description: System healthy
          content:
            application/json:
              schema:
                type: object
                properties:
                  status: string
                  uptime: number
                  version: string

  /api/v1/config:
    get:
      summary: Retrieve configuration
      responses:
        200:
          description: Current configuration
    put:
      summary: Update configuration
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
      responses:
        200:
          description: Configuration updated

  /api/v1/calibration:
    post:
      summary: Begin calibration
      responses:
        201:
          description: Calibration started
          content:
            application/json:
              schema:
                type: object
                properties:
                  session_id: string
                  expires_at: string

  /api/v1/game/state:
    get:
      summary: Get current game state
      responses:
        200:
          description: Current game state
          content:
            application/json:
              schema:
                type: object
                properties:
                  timestamp: string
                  balls: array
                  cue: object
                  table: object

/api/v1/stream/video:
    get:
      summary: Video stream endpoint
      produces: ['multipart/x-mixed-replace']
      responses:
        200:
          description: MJPEG video stream

                    details: object
                    issues: array
                    recommendations: array
                    duration: number
```

### Real-time Message Protocol

The API WebSocket route is `/api/v1/websocket/ws`. Clients subscribe by sending
a `subscribe` request with the desired streams in `data.streams`:

```json
{
  "type": "subscribe",
  "timestamp": "2026-01-01T00:00:00Z",
  "data": {"streams": ["state", "trajectory", "alert", "config"]}
}
```

The server sends message envelopes with `type`, `timestamp`, and `data`; current
field and coordinate schemas are defined in
[`websocket/schemas.py`](websocket/schemas.py). Image-space values, including
boundaries and extents, must follow the explicit scale contract in the
[API reference](../../docs/API_REFERENCE.md).

WebSocket video transmission is disabled. The API does not publish UDP
projector messages. The projector's local receiver expects a distinct custom
`paths`/`ghostBalls` payload and requires an adapter to consume backend data.

## Data Models

### Configuration Schema

```python
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from enum import Enum

class PerformanceMode(str, Enum):
    LOW = "low"
    BALANCED = "balanced"
    HIGH = "high"

class SystemConfig(BaseModel):
    debug: bool = False
    log_level: str = "INFO"
    performance_mode: PerformanceMode = PerformanceMode.BALANCED

class CameraConfig(BaseModel):
    device_id: int = 0
    resolution: List[int] = [1920, 1080]
    fps: int = 30
    exposure: str = "auto"
    brightness: float = Field(0.5, ge=0, le=1)
    contrast: float = Field(0.5, ge=0, le=1)

class ColorRange(BaseModel):
    hue: List[int] = Field(..., min_items=2, max_items=2)
    saturation: List[int] = Field(..., min_items=2, max_items=2)
    value: List[int] = Field(..., min_items=2, max_items=2)

class VisionConfig(BaseModel):
    table_color: ColorRange
    ball_detection_sensitivity: float = Field(0.8, ge=0, le=1)
    min_ball_radius: int = 15
    max_ball_radius: int = 35
    cue_detection_threshold: float = 0.7

class NetworkConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: List[str] = ["*"]
    max_connections: int = 100
    websocket_ping_interval: int = 30
    state_publish_interval_ms: int = 500  # Interval for static ball position updates
    motion_event_immediate: bool = True  # Publish motion events immediately


class Configuration(BaseModel):
    system: SystemConfig
    camera: CameraConfig
    vision: VisionConfig
    network: NetworkConfig
    custom: Optional[Dict] = {}
```

### Response Models

```python
class HealthResponse(BaseModel):
    status: str  # "healthy", "degraded", "unhealthy"
    uptime: float  # seconds
    version: str
    components: Dict[str, str]  # Component health statuses

class CalibrationSession(BaseModel):
    session_id: str
    created_at: datetime
    expires_at: datetime
    points_captured: int
    points_required: int
    status: str  # "in_progress", "completed", "failed"

class GameState(BaseModel):
    timestamp: datetime
    frame_number: int
    balls: List[Ball]
    cue: Optional[CueStick]
    table: Table
    trajectories: List[Trajectory]

class ErrorResponse(BaseModel):
    error: str
    message: str
    code: str
    details: Optional[Dict]
    timestamp: datetime
```

## Error Handling

### Error Codes and Responses

```python
ERROR_CODES = {
    # Client errors (4xx)
    "VAL_001": {"status": 400, "message": "Invalid request format"},
    "VAL_002": {"status": 400, "message": "Missing required parameter"},
    "VAL_003": {"status": 400, "message": "Parameter out of range"},
    "RES_001": {"status": 404, "message": "Resource not found"},
    "GAME_002": {"status": 409, "message": "Game state conflict"},
    "RATE_001": {"status": 429, "message": "Rate limit exceeded"},
    "NET_001": {"status": 502, "message": "Network connectivity error"},
    "NET_002": {"status": 504, "message": "Request timeout"},

    # Server errors (5xx)
    "SYS_001": {"status": 500, "message": "Internal server error"},
    "SYS_002": {"status": 503, "message": "Service unavailable"},
    "CAM_001": {"status": 503, "message": "Camera unavailable"},
    "PROC_001": {"status": 503, "message": "Vision processing failed"},
    "WS_001": {"status": 503, "message": "WebSocket service unavailable"}
}
```

## Success Criteria

### Functional Success Criteria

1. **API Availability**
   - All specified REST endpoints are implemented and accessible
   - WebSocket connection establishes within 1 second

2. **Data Streaming**
   - Video frames stream at consistent 30+ FPS
   - Game state updates arrive within 50ms of detection
   - No message loss during normal operation

3. **Configuration Management**
   - Configuration changes apply without restart
   - Invalid configurations are rejected with clear errors
   - Configuration persists across restarts

4. **Error Handling**
   - All errors return appropriate HTTP status codes
   - Error messages provide actionable information
   - System recovers gracefully from transient failures

### Performance Success Criteria

1. **Response Times**
   - 95th percentile REST response time < 100ms
   - 99th percentile REST response time < 500ms
   - WebSocket message latency < 50ms average

3. **Resource Usage**
   - Memory usage < 500MB under normal load
   - CPU usage < 50% with 10 active clients
   - Network bandwidth < 100Mbps for 10 clients


## Testing Requirements

### Unit Testing
- Test all endpoint handlers individually
- Mock external dependencies
- Validate input validation logic
- Test error handling paths
- Coverage target: 90%

### Integration Testing
- Test complete request/response flows
- Verify WebSocket message handling
- Validate configuration updates

### Performance Testing
- Load test with 100+ concurrent users
- Stress test WebSocket connections
- Measure response times under load
- Test graceful degradation
- Monitor resource usage

## Implementation Guidelines

### Code Structure
```python
api/
├── __init__.py
├── main.py              # FastAPI application setup
├── routes/
│   ├── __init__.py
│   ├── health.py        # Health check endpoints
│   ├── config.py        # Configuration management
│   ├── calibration.py   # Calibration control
│   ├── game.py          # Game state access
├── websocket/
│   ├── __init__.py
│   ├── handler.py       # WebSocket connection handler
│   ├── manager.py       # Connection management
│   └── broadcaster.py   # Message broadcasting
├── middleware/
│   ├── __init__.py
│   ├── cors.py
│   ├── rate_limit.py
│   └── error_handler.py
├── models/
│   ├── __init__.py
│   ├── requests.py      # Request schemas
│   ├── responses.py     # Response schemas
│   └── websocket.py     # WebSocket message schemas
└── utils/
    ├── __init__.py
    ├── validators.py    # Custom validators
    └── serializers.py   # Data serialization
```

### Key Dependencies
- **FastAPI**: Web framework
- **uvicorn**: ASGI server
- **pydantic**: Data validation
- **python-multipart**: File uploads
- **websockets**: WebSocket support

### Development Priorities
1. Implement core REST endpoints
2. Add WebSocket support
3. Add input validation
4. Implement error handling
5. Performance optimization
