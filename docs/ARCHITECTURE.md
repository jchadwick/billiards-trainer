# Billiards Trainer System Architecture

## System Architecture Overview

The Billiards Trainer system follows a modular, microservices-inspired architecture with clear separation of concerns between computer vision processing, user interface, and augmented reality display components. The architecture prioritizes real-time performance, scalability, and maintainability.

## High-Level Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     Physical Hardware Layer                  │
├──────────────┬────────────────┬──────────────┬─────────────┤
│   Camera     │  Pool Table    │  Projector   │   Users     │
└──────┬───────┴────────────────┴──────┬───────┴──────┬──────┘
       │                                │              │
┌──────▼─────────────────────────────────────────────────────┐
│                    Backend Service Layer                    │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌──────────┐  ┌──────────┐             │
│  │   Vision    │  │   Core    │  │   API    │             │
│  │   Module    │◄─┤  Module   │◄─┤  Module  │             │
│  └──────┬──────┘  └─────┬────┘  └────┬─────┘             │
│         │               │             │                    │
│  ┌──────▼──────────────▼─────────────▼────┐               │
│  │       Configuration Module              │               │
│  └─────────────────────────────────────────┘               │
│                                                             │
│  ┌─────────────────────────────────────────┐               │
│  │       Projector Client Module           │               │
│  └─────────────────────────────────────────┘               │
└───────────────────────┬─────────────────────────┬──────────┘
                        │                         │
                   WebSocket                 HTTP/REST
                        │                         │
┌───────────────────────▼─────────────────────────▼──────────┐
│                    Frontend Layer                           │
├─────────────────────────────────────────────────────────────┤
│     Web Application (React/Vue)                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
│  │   Live   │  │  Config  │  │ Trajectory│  │ Spectator│  │
│  │   View   │  │  Panel   │  │  Display  │  │   Mode   │  │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Component Architecture

### 1. Backend Service Layer

#### Core Components
- **Vision Module**: Handles all computer vision operations
- **Core Module**: Game logic, physics calculations, and state management
- **API Module**: REST endpoints and WebSocket connections
- **Configuration Module**: System settings and calibration data
- **Projector Client**: Standalone module for AR display

#### Technology Stack
- **Language**: Python 3.12+
- **Framework**: FastAPI for async HTTP and WebSocket
- **Computer Vision**: OpenCV 4.8+ with NumPy
- **Async Runtime**: asyncio for concurrent operations
- **Data Validation**: Pydantic for type safety

### 2. Frontend Layer

#### Components
- **Web Application**: Single-page application for user interaction
- **Live View**: Real-time camera feed with overlay
- **Configuration Panel**: System settings and calibration
- **Trajectory Display**: Visualization of predicted ball paths
- **Spectator Mode**: Read-only view for additional viewers

#### Technology Stack
- **Framework**: React or Vue 3
- **State Management**: Redux/Vuex or Zustand/Pinia
- **WebSocket Client**: Socket.io or native WebSocket
- **UI Components**: Material-UI or Ant Design
- **Build Tools**: Vite for fast development

### 3. Communication Layer

#### Protocols
- **WebSocket**: Real-time bidirectional communication
- **REST API**: Configuration and control endpoints
- **WebRTC**: Optional for peer-to-peer video streaming

#### Data Formats
- **JSON**: Primary data exchange format
- **MessagePack**: Optional binary format for performance
- **Base64**: Image encoding for web transmission

## Data Flow Architecture

### Real-Time Processing Pipeline

```
1. Image Capture (30+ FPS)
   ↓
2. Vision Processing
   - Color space conversion (BGR → HSV)
   - Table detection and boundary extraction
   - Ball detection and classification
   - Cue stick detection and angle calculation
   ↓
3. State Management
   - Game state update
   - Change detection and filtering
   - State history maintenance
   ↓
4. Physics Calculation
   - Trajectory prediction
   - Collision detection
   - Force estimation
   ↓
5. Data Distribution
   - WebSocket broadcast to clients
   - Projector client update
   - State persistence
   ↓
6. Display Rendering
   - Web UI update
   - Projector overlay generation
   - Performance metrics update
```

### API Request Flow

```
Client Request → API Gateway → Authentication → Route Handler
                                                      ↓
Response ← Serialization ← Business Logic ← Data Validation
```

## Deployment Architecture

### Container Architecture

```yaml
services:
  billiards-trainer-backend:
    image: billiards-trainer/backend:latest
    ports:
      - "8000:8000"  # HTTP/WebSocket
    volumes:
      - ./config:/app/config
    devices:
      - /dev/video0  # Camera device
    environment:
      - OPENCV_VIDEOIO_PRIORITY_GSTREAMER=0

  billiards-trainer-frontend:
    image: billiards-trainer/frontend:latest
    ports:
      - "3000:80"  # Web interface
    depends_on:
      - billiards-trainer-backend

  billiards-trainer-projector:
    image: billiards-trainer/projector:latest
    environment:
      - DISPLAY=:0  # X11 display
    volumes:
      - /tmp/.X11-unix:/tmp/.X11-unix
    depends_on:
      - billiards-trainer-backend
```

## Development Architecture

### Module Structure

```
billiards-trainer-python/
├── backend/
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes/         # REST endpoints
│   │   ├── websocket/      # WebSocket handlers
│   │   └── middleware/     # Auth, CORS, etc.
│   ├── core/
│   │   ├── __init__.py
│   │   ├── game_state.py   # State management
│   │   ├── physics.py      # Trajectory calculations
│   │   └── rules.py        # Game rules
│   ├── vision/
│   │   ├── __init__.py
│   │   ├── capture.py      # Camera interface
│   │   ├── detection.py    # Object detection
│   │   └── calibration.py  # System calibration
│   ├── config/
│   │   ├── __init__.py
│   │   ├── settings.py     # Configuration management
│   │   └── schema.py       # Config validation
│   └── projector/
│       ├── __init__.py
│       ├── client.py       # Projector client
│       └── renderer.py     # Overlay generation
├── frontend/
│   ├── src/
│   │   ├── components/     # Reusable components
│   │   ├── views/          # Page components
│   │   ├── services/       # API communication
│   │   ├── stores/         # State management
│   │   └── utils/          # Helper functions
│   └── public/             # Static assets
├── tests/
│   ├── unit/              # Unit tests
│   ├── integration/       # Integration tests
│   └── e2e/               # End-to-end tests
└── docker/
    ├── backend.Dockerfile
    ├── frontend.Dockerfile
    └── docker-compose.yml
```

### Testing Architecture

#### Test Pyramid
1. **Unit Tests** (70%): Individual functions and methods
2. **Integration Tests** (20%): Module interactions
3. **End-to-End Tests** (10%): Full system workflows

#### Test Coverage Requirements
- **Minimum Coverage**: 80% for all modules
- **Critical Paths**: 100% for vision and physics
- **Performance Tests**: Automated regression testing

## Integration Architecture

### External System Integration

#### Camera Integration
- **Protocol**: V4L2 (Linux), DirectShow (Windows)
- **Formats**: MJPEG, H.264, raw formats
- **Resolution**: Configurable, minimum 1080p
- **Frame Rate**: Adaptive 15-60 FPS

#### Projector Integration
- **Display Protocol**: Direct framebuffer or window manager
- **Calibration**: 4-point perspective transform
- **Refresh Rate**: Synchronized with processing pipeline
- **Color Correction**: Gamma and brightness adjustment

### API Integration Points

```python
# RESTful API endpoints
api_endpoints = {
    # System Control
    "GET /api/v1/status": "System health check",
    "GET /api/v1/config": "Retrieve configuration",
    "PUT /api/v1/config": "Update configuration",

    # Calibration
    "POST /api/v1/calibrate/start": "Begin calibration",
    "POST /api/v1/calibrate/capture": "Capture calibration point",
    "POST /api/v1/calibrate/complete": "Finish calibration",

    # Game Control
    "GET /api/v1/game/state": "Current game state",
    "POST /api/v1/game/reset": "Reset detection",

    # WebSocket
    "WS /ws": "Real-time data stream"
}
```

## Configuration Architecture

### Configuration Hierarchy

```yaml
# Default configuration structure
config:
  system:
    debug: false
    log_level: INFO
    performance_mode: balanced

  camera:
    device_id: 0
    resolution: [1920, 1080]
    fps: 30
    exposure: auto

  vision:
    table_color:
      hue: [35, 85]
      saturation: [50, 255]
      value: [50, 255]
    ball_detection:
      min_radius: 15
      max_radius: 35
      sensitivity: 0.8

  projector:
    enabled: true
    correction:
      points: [[0,0], [1920,0], [1920,1080], [0,1080]]
    brightness: 1.0
    contrast: 1.0

  network:
    host: 0.0.0.0
    port: 8000
    cors_origins: ["*"]
```

### Configuration Sources (Priority Order)
1. Environment variables (highest)
2. Command-line arguments
3. Configuration file
4. Database settings
5. Default values (lowest)

## Success Metrics Architecture

### Key Performance Indicators (KPIs)

```python
kpis = {
    "technical": {
        "processing_fps": "> 30",
        "detection_accuracy": "> 98%",
        "end_to_end_latency": "< 50ms",
        "system_uptime": "> 99.9%"
    },
    "business": {
        "setup_time": "< 5 minutes",
        "user_satisfaction": "> 4.5/5",
        "concurrent_users": "> 10",
        "error_rate": "< 0.1%"
    }
}
```

### Monitoring Dashboard

The system includes a comprehensive monitoring dashboard displaying:
- Real-time performance metrics
- System health indicators
- Error logs and alerts
- Usage statistics
- Resource utilization graphs

## Future Architecture Considerations

### Planned Enhancements
1. **Machine Learning Integration**: Deep learning for improved detection
2. **Cloud Deployment**: AWS/Azure/GCP support
3. **Mobile Applications**: Native iOS/Android apps
4. **Multi-table Support**: Manage multiple tables from single instance
5. **AI Coaching**: Suggested shots based on skill level

### Architecture Evolution Path
1. **Phase 1**: Monolithic backend with modular structure
2. **Phase 2**: Microservices with message queue
3. **Phase 3**: Serverless functions for scalability
4. **Phase 4**: Edge computing for reduced latency
