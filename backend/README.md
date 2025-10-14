# Billiards Trainer Backend

The backend service provides real-time computer vision processing, physics calculations, and game state management for the billiards training system.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Detection and Tracking](#detection-and-tracking)
- [Trajectory Calculation](#trajectory-calculation)
- [Configuration](#configuration)
- [Modules](#modules)
- [Getting Started](#getting-started)
- [Development](#development)

---

## Overview

The backend is a Python-based service that:

1. **Captures video** from cameras (Kinect, USB webcams, or video files)
2. **Detects objects** using YOLO+OpenCV hybrid detection (balls, cue stick, table)
3. **Tracks objects** across frames with Kalman filtering
4. **Calculates physics** including trajectories, collisions, and ball motion
5. **Broadcasts updates** via WebSocket to web UI and projector clients

### Key Features

- **YOLO+OpenCV Hybrid Detection**: Fast YOLO localization with precise OpenCV classification
- **Real-time Tracking**: Kalman filter-based object tracking for smooth motion
- **Multiball Trajectory Prediction**: Calculate shot outcomes with collisions up to 5 levels deep
- **Event-Driven Architecture**: Asynchronous event system for loose coupling
- **WebSocket Broadcasting**: Real-time updates to all connected clients
- **Configurable Pipeline**: Extensive configuration system with defaults and overrides

### Technology Stack

- **Python 3.12+**
- **FastAPI**: Async HTTP and WebSocket server
- **OpenCV 4.8+**: Computer vision operations
- **NumPy**: Numerical computations
- **Pydantic**: Data validation and serialization
- **asyncio**: Concurrent operations

---

## Architecture

### Module Organization

```
backend/
├── api/              # API endpoints and WebSocket broadcasting
│   ├── routes/       # REST API route handlers
│   └── websocket/    # WebSocket connection and message broadcasting
├── config/           # Configuration management
│   └── models/       # Pydantic configuration schemas
├── core/             # Game logic and physics
│   ├── collision/    # Collision detection and response
│   ├── events/       # Event system for inter-module communication
│   ├── physics/      # Physics engine and trajectory calculation
│   ├── utils/        # Utility functions (geometry, math, caching)
│   └── validation/   # State validation and correction
├── vision/           # Computer vision processing
│   ├── calibration/  # Camera and color calibration
│   ├── capture/      # Video capture from cameras
│   ├── detection/    # Object detection (YOLO+OpenCV)
│   ├── preprocessing/# Image preprocessing
│   └── tracking/     # Object tracking (Kalman filter)
├── integration_service.py  # Coordinates all modules
└── dev_server.py     # Development server entry point
```

### Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                        Video Capture                            │
│                   (Camera or Video File)                        │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                     VisionModule                                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│  │ Preprocess  │→ │   Detect    │→ │    Track    │            │
│  │             │  │ (YOLO+OpenCV)  │  (Kalman)   │            │
│  └─────────────┘  └─────────────┘  └─────────────┘            │
└───────────────────────────┬─────────────────────────────────────┘
                            │ DetectionResult
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                   IntegrationService                            │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │ 1. Convert DetectionResult → CoreModule state             │ │
│  │ 2. Trigger trajectory calculation if cue detected         │ │
│  │ 3. Validate and correct state                             │ │
│  │ 4. Emit events (state_updated, trajectory_calculated)     │ │
│  └───────────────────────────────────────────────────────────┘ │
└───────────────────────────┬─────────────────────────────────────┘
                            │ Events
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                  WebSocket Broadcaster                          │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │ • broadcast_game_state(balls, cue, table)                 │ │
│  │ • broadcast_detection(balls, cue, table)                  │ │
│  │ • broadcast_trajectory(multiball_result)                  │ │
│  │ • Error handling with circuit breaker pattern             │ │
│  └───────────────────────────────────────────────────────────┘ │
└───────────────────────────┬─────────────────────────────────────┘
                            │ WebSocket JSON Messages
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Connected Clients                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │   Web UI     │  │  Projector   │  │  Spectator   │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
└─────────────────────────────────────────────────────────────────┘
```

---

## Detection and Tracking

### YOLO+OpenCV Hybrid Detection

The detection pipeline uses a **hybrid approach** combining the strengths of both YOLO and OpenCV:

#### YOLO Component (Object Localization)
- **Fast object detection** using YOLOv8 neural network
- Trained on custom billiards dataset with ~10,000 annotated images
- Detects: balls, cue stick, table boundaries
- Runs on CPU or GPU (configurable)
- Output: Bounding boxes with class IDs and confidence scores

#### OpenCV Component (Ball Classification)
- **Precise color-based classification** for ball types
- Uses HSV color space analysis within YOLO bounding boxes
- Classifies: Cue ball, 8-ball, solid balls, stripe balls
- Handles lighting variations with adaptive thresholds
- Output: Refined BallType enum (CUE, EIGHT, OTHER, UNKNOWN)

#### Why Hybrid?

| Aspect | Pure YOLO | Pure OpenCV | Hybrid (YOLO+OpenCV) |
|--------|-----------|-------------|----------------------|
| **Localization** | Excellent | Poor (false positives) | Excellent ✓ |
| **Ball Type Classification** | Limited | Excellent | Excellent ✓ |
| **Speed** | Fast | Slow | Fast ✓ |
| **Lighting Robustness** | Good | Poor | Good ✓ |
| **Training Data Required** | High | None | Medium ✓ |

**Result**: YOLO quickly finds all balls, OpenCV accurately identifies ball types.

### Detection Pipeline

```python
# 1. Capture frame from camera
frame = await camera_capture.get_frame()

# 2. Preprocess (optional: crop ROI, enhance contrast)
preprocessed = image_preprocessor.process(frame)

# 3. YOLO detection - get bounding boxes
yolo_detections = yolo_detector.detect(preprocessed)
# Output: [
#   {"bbox": [x, y, w, h], "class_id": 0, "confidence": 0.85},
#   {"bbox": [x, y, w, h], "class_id": 8, "confidence": 0.92},
#   ...
# ]

# 4. OpenCV classification - refine ball types
for detection in yolo_detections:
    bbox_region = preprocessed[y:y+h, x:x+w]
    ball_type = classify_ball_by_color(bbox_region)  # HSV analysis
    detection["ball_type"] = ball_type

# 5. Convert to Vision module dataclasses
balls = [
    Ball(
        position=(cx, cy),
        radius=r,
        ball_type=BallType.CUE,
        confidence=0.85,
        velocity=(0, 0),  # Filled by tracker
        is_moving=False,
    )
    for detection in yolo_detections
]

# 6. Track across frames (Kalman filter)
tracked_balls = object_tracker.update(balls, frame_timestamp)
# Adds: velocity, acceleration, smoothed positions

# 7. Return DetectionResult
return DetectionResult(
    balls=tracked_balls,
    cue=detected_cue,
    table=detected_table,
    frame_info=frame_info,
)
```

### Object Tracking

**Kalman Filter Tracking** provides smooth motion estimation:

```python
class ObjectTracker:
    """Track objects across frames using Kalman filtering."""

    def __init__(self):
        # State vector: [x, y, vx, vy]
        self.kalman_filters = {}  # track_id → KalmanFilter
        self.track_id_counter = 0

    def update(self, detections: list[Ball], timestamp: float) -> list[Ball]:
        """Update tracking with new detections."""
        # 1. Associate detections with existing tracks (Hungarian algorithm)
        matches = self._associate_detections_to_tracks(detections)

        # 2. Update matched tracks
        for detection, track_id in matches:
            kf = self.kalman_filters[track_id]
            kf.predict()
            kf.update([detection.position[0], detection.position[1]])

            # Extract velocity from Kalman state
            state = kf.get_state()
            detection.velocity = (state[2], state[3])
            detection.is_moving = np.linalg.norm(detection.velocity) > 0.5

        # 3. Create new tracks for unmatched detections
        for detection in unmatched_detections:
            new_track_id = self._create_new_track(detection)

        # 4. Remove stale tracks (no detection for N frames)
        self._remove_stale_tracks()

        return detections
```

**Key Features:**
- **Position Smoothing**: Reduces jitter from noisy detections
- **Velocity Estimation**: Calculates ball velocity from position changes
- **Occlusion Handling**: Maintains tracks when balls temporarily disappear
- **Track Association**: Hungarian algorithm for optimal detection-to-track matching

### Configuration

```json
{
  "vision": {
    "detection": {
      "detection_backend": "yolo",
      "yolo_model_path": "models/yolov8n-pool.onnx",
      "yolo_confidence": 0.4,
      "yolo_nms_threshold": 0.45,
      "yolo_device": "cpu",
      "use_opencv_validation": true,
      "min_ball_radius": 20
    },
    "tracking": {
      "max_track_age": 30,
      "min_detection_confidence": 0.3,
      "iou_threshold": 0.3
    }
  }
}
```

---

## Trajectory Calculation

The backend calculates **multiball trajectories** in real-time when a cue stick is detected:

### Features

- **Multiball Collision Prediction**: Calculates paths for all affected balls
- **Up to 5 Collision Levels Deep**: Tracks cascading collisions
- **Rail Bounces**: Accurate cushion reflection with restitution
- **Physics Simulation**: Friction, rolling resistance, ball-to-ball energy transfer
- **Quality Levels**: LOW (fast, real-time), MEDIUM, HIGH (accurate, slower)

### Calculation Flow

```python
async def _check_trajectory_calculation(self, detection: DetectionResult) -> None:
    """Calculate trajectory when cue is detected and aiming at a ball."""

    # 1. Check prerequisites
    if not detection.cue:
        return  # No cue detected
    if not detection.balls:
        return  # No balls to hit

    # 2. Find target ball (ball cue is pointing at)
    target_ball = self._find_ball_cue_is_pointing_at(detection.cue, detection.balls)
    if not target_ball:
        return  # Cue not pointing at any ball

    # 3. Convert vision detections to core state objects
    cue_state = self._create_cue_state(detection.cue)
    ball_state = self._create_ball_state(target_ball, is_target=True)
    other_balls = self._create_ball_states(detection.balls, exclude_ball=target_ball)

    # 4. Get table state from Core
    table_state = self.core._current_state.table

    # 5. Calculate multiball trajectory
    multiball_result = self.trajectory_calculator.predict_multiball_cue_shot(
        cue_state=cue_state,
        ball_state=ball_state,
        table_state=table_state,
        other_balls=other_balls,
        quality=TrajectoryQuality.LOW,  # Real-time performance
        max_collision_depth=5,
    )

    # 6. Broadcast results to clients
    await self._emit_multiball_trajectory(multiball_result)
```

### Trajectory Result Structure

```python
@dataclass
class MultiballTrajectoryResult:
    """Result of multiball trajectory prediction."""

    # Map of ball_id → trajectory
    trajectories: dict[int, BallTrajectory]

    # Metadata
    calculation_time_ms: float
    quality: TrajectoryQuality
    max_collision_depth: int

@dataclass
class BallTrajectory:
    """Trajectory for a single ball."""

    # Path points (ball center positions)
    points: list[Vector2D]

    # Velocity at each point
    velocities: list[Vector2D]

    # Collision events
    collisions: list[Collision]

    # Final state
    final_position: Vector2D
    final_velocity: Vector2D
    comes_to_rest: bool
    time_to_rest_seconds: float

@dataclass
class Collision:
    """Collision event information."""

    type: CollisionType  # BALL_TO_BALL, BALL_TO_RAIL, BALL_TO_POCKET
    position: Vector2D
    timestamp: float
    velocity_before: Vector2D
    velocity_after: Vector2D
    collision_depth: int  # 0 = initial, 1 = first collision, etc.
    other_ball_id: Optional[int]  # If ball-to-ball collision
```

### Finding Target Ball

```python
def _find_ball_cue_is_pointing_at(self, cue: CueStick, balls: list[Ball]) -> Optional[Ball]:
    """Find which ball the cue is pointing at using perpendicular distance."""

    # Use geometry utility function
    from backend.core.utils.geometry import find_ball_cue_is_pointing_at

    ball_positions = [ball.position for ball in balls]

    target_idx = find_ball_cue_is_pointing_at(
        cue_tip=cue.tip_position,
        cue_angle=cue.angle,
        balls=ball_positions,
        max_perpendicular_distance=40.0,  # pixels
    )

    return balls[target_idx] if target_idx is not None else None
```

**Algorithm**: Uses **perpendicular distance** from cue line to ball center:
- Extend cue stick line forward from tip
- Calculate perpendicular distance from line to each ball center
- Select closest ball within threshold (default 40 pixels)
- Only consider balls in front of cue tip (positive projection)

### Trajectory Broadcasting

Trajectory results are broadcast via WebSocket:

```json
{
  "type": "trajectory",
  "data": {
    "trajectories": {
      "0": {  // Ball ID
        "points": [
          {"x": 100.0, "y": 200.0},
          {"x": 105.2, "y": 198.3},
          ...
        ],
        "collisions": [
          {
            "type": "BALL_TO_RAIL",
            "position": {"x": 150.0, "y": 180.0},
            "depth": 1
          }
        ],
        "comes_to_rest": true,
        "time_to_rest_seconds": 2.3
      }
    },
    "calculation_time_ms": 8.5,
    "quality": "LOW"
  }
}
```

### Configuration

```json
{
  "integration": {
    "shot_prediction_enabled": true,
    "shot_speed_estimate_m_per_s": 2.0
  },
  "core": {
    "physics": {
      "ball_mass_kg": 0.17,
      "ball_radius_m": 0.028575,
      "friction_coefficient": 0.2,
      "rolling_friction": 0.01,
      "restitution_ball_ball": 0.95,
      "restitution_ball_rail": 0.85
    },
    "trajectory": {
      "default_quality": "LOW",
      "max_simulation_time_seconds": 10.0,
      "min_velocity_threshold": 0.01
    }
  }
}
```

---

## Configuration

### Configuration System

The backend uses a **hierarchical configuration system**:

1. **Default Configuration** (`config/default.json`): Baseline values
2. **Environment Configuration** (`config/local.json`, `config/production.json`): Environment overrides
3. **Current Configuration** (`config/current.json`): Active runtime configuration
4. **Runtime Overrides**: API-based configuration updates

```python
from backend.config.manager import ConfigurationModule

config = ConfigurationModule(config_dir=Path("config"))

# Get configuration value with default
detection_backend = config.get("vision.detection.detection_backend", default="yolo")

# Set configuration value (persisted to current.json)
config.set("vision.detection.yolo_confidence", 0.5)

# Get nested configuration
camera_config = config.get_section("vision.camera")
# Returns: {"device_id": 0, "backend": "auto", ...}
```

### Key Configuration Sections

#### Vision Configuration

```json
{
  "vision": {
    "camera": {
      "device_id": 0,
      "backend": "auto",
      "fps": 30,
      "width": 1920,
      "height": 1080
    },
    "detection": {
      "detection_backend": "yolo",
      "yolo_model_path": "models/yolov8n-pool.onnx",
      "yolo_confidence": 0.4,
      "yolo_device": "cpu"
    },
    "processing": {
      "enable_ball_detection": true,
      "enable_cue_detection": true,
      "enable_tracking": true,
      "frame_skip": 0
    }
  }
}
```

#### Core Configuration

```json
{
  "core": {
    "physics": {
      "ball_mass_kg": 0.17,
      "ball_radius_m": 0.028575,
      "friction_coefficient": 0.2,
      "table_length_m": 2.54,
      "table_width_m": 1.27
    },
    "validation": {
      "enable_correction": true,
      "position_snap_threshold": 5.0,
      "velocity_outlier_threshold": 10.0
    }
  }
}
```

#### Integration Configuration

```json
{
  "integration": {
    "target_fps": 30,
    "broadcast_max_retries": 3,
    "circuit_breaker_threshold": 10,
    "shot_prediction_enabled": true
  }
}
```

### Configuration Files

- **`config/default.json`**: System defaults (never edited)
- **`config/local.json`**: Local development overrides
- **`config/production.json`**: Production overrides
- **`config/current.json`**: Active runtime configuration
- **`backend/config/default.json`**: Backend module defaults

See `docs/CONFIG.md` for comprehensive configuration reference.

---

## Modules

### VisionModule

**Location**: `backend/vision/`

**Responsibilities**:
- Video capture from cameras or files
- Object detection (YOLO+OpenCV)
- Object tracking (Kalman filter)
- Table and color calibration

**Key Classes**:
- `VisionModule` - Main interface
- `YOLODetector` - YOLO-based detection
- `ObjectTracker` - Kalman filter tracking
- `CameraCapture` - Video source abstraction
- `TableDetector` - Table boundary detection

**Usage**:
```python
from backend.vision import VisionModule

vision = VisionModule({
    "camera_device_id": 0,
    "detection_backend": "yolo",
    "enable_ball_detection": True,
    "enable_cue_detection": True,
    "enable_tracking": True,
})

await vision.start()
detection = await vision.process_frame()
await vision.stop()
```

### CoreModule

**Location**: `backend/core/`

**Responsibilities**:
- Game state management
- Physics simulation
- Trajectory calculation
- State validation and correction

**Key Classes**:
- `CoreModule` - Main interface
- `TrajectoryCalculator` - Shot prediction
- `StateValidator` - State validation
- `EventManager` - Event system

**Usage**:
```python
from backend.core import CoreModule

core = CoreModule()
await core.start()

# Update state from vision
core.update_balls(ball_states)
core.update_cue(cue_state)

# Get current state
state = core.get_current_state()

await core.stop()
```

### IntegrationService

**Location**: `backend/integration_service.py`

**Responsibilities**:
- Coordinate Vision, Core, and Broadcasting
- Convert between module data formats
- Trigger trajectory calculations
- Handle errors and retries

**Key Methods**:
- `start()` - Start integration loop
- `stop()` - Stop integration loop
- `_process_vision_data()` - Convert vision to core
- `_check_trajectory_calculation()` - Trigger predictions
- `_broadcast_with_retry()` - Broadcast with circuit breaker

**Usage**:
```python
from backend.integration_service import IntegrationService

service = IntegrationService(
    vision_module=vision,
    core_module=core,
    message_broadcaster=broadcaster,
)

await service.start()
# Integration loop running...
await service.stop()
```

### API Module

**Location**: `backend/api/`

**Responsibilities**:
- REST API endpoints
- WebSocket connections
- Message broadcasting
- Client management

**Key Components**:
- `MessageBroadcaster` - WebSocket broadcast manager
- `routes/` - REST API route handlers
- `websocket/` - WebSocket connection handlers

**WebSocket Events**:
- `detection` - Real-time object detection updates
- `game_state` - Game state changes (validated)
- `trajectory` - Trajectory predictions
- `calibration_update` - Calibration changes

---

## Getting Started

### Prerequisites

```bash
# Python 3.12+
python --version

# Install dependencies
pip install -r backend/requirements.txt

# Download YOLO model (if not present)
wget https://path.to/yolov8n-pool.onnx -O models/yolov8n-pool.onnx
```

### Running the Backend

```bash
# Start development server
python backend/dev_server.py

# Or with custom config
python backend/dev_server.py --config config/local.json

# With logging
python backend/dev_server.py --log-level DEBUG
```

### Testing with Video File

```bash
# Use video debugger tool
python tools/video_debugger.py test_video.mp4 --detection-backend yolo

# Options
python tools/video_debugger.py video.mp4 --loop --log-level INFO
```

### API Access

```bash
# REST API (default port 8000)
curl http://localhost:8000/api/health

# WebSocket (default port 8000)
ws://localhost:8000/ws
```

---

## Development

### Project Structure

```
backend/
├── __init__.py
├── api/
│   ├── __init__.py
│   ├── routes/
│   │   ├── calibration.py
│   │   ├── configuration.py
│   │   └── health.py
│   └── websocket/
│       ├── __init__.py
│       └── broadcaster.py
├── config/
│   ├── __init__.py
│   ├── manager.py
│   ├── default.json
│   └── models/
│       └── schemas.py
├── core/
│   ├── __init__.py
│   ├── models.py
│   ├── game_state.py
│   ├── collision/
│   ├── events/
│   ├── physics/
│   ├── utils/
│   └── validation/
├── vision/
│   ├── __init__.py
│   ├── models.py
│   ├── calibration/
│   ├── capture/
│   ├── detection/
│   ├── preprocessing/
│   └── tracking/
├── integration_service.py
└── dev_server.py
```

### Running Tests

```bash
# Run all tests
pytest backend/

# Run specific module tests
pytest backend/vision/detection/tests/
pytest backend/core/physics/tests/

# With coverage
pytest --cov=backend --cov-report=html

# Watch mode
pytest-watch backend/
```

### Code Quality

```bash
# Format code
black backend/
isort backend/

# Lint
flake8 backend/
pylint backend/

# Type checking
mypy backend/
```

### Adding New Features

1. **Create feature branch**: `git checkout -b feature/new-detection-method`
2. **Implement in appropriate module** (vision, core, or api)
3. **Add configuration parameters** to `config/default.json`
4. **Write tests**: Unit tests + integration tests
5. **Update documentation**: This README + docstrings
6. **Submit PR** with description and test results

### Debugging

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
python backend/dev_server.py

# Use video debugger for vision issues
python tools/video_debugger.py video.mp4 --log-level DEBUG

# Profile performance
python -m cProfile -o profile.stats backend/dev_server.py
python -m pstats profile.stats
```

### Common Issues

**YOLO model not found**:
```bash
# Download model to correct path
mkdir -p models
wget https://path.to/yolov8n-pool.onnx -O models/yolov8n-pool.onnx
```

**Camera not opening**:
```bash
# List available cameras
python -c "import cv2; print([i for i in range(10) if cv2.VideoCapture(i).isOpened()])"

# Set correct device_id in config
```

**WebSocket connection refused**:
```bash
# Check server is running
curl http://localhost:8000/api/health

# Check firewall rules
sudo ufw status
```

---

## Performance Optimization

### Detection Performance

- **Use GPU**: Set `yolo_device: "cuda"` for NVIDIA GPUs
- **Reduce Resolution**: Lower camera resolution if FPS is too low
- **Adjust Confidence**: Increase `yolo_confidence` to filter more detections
- **Enable Frame Skip**: Set `frame_skip: 1` to process every other frame

### Tracking Performance

- **Reduce Track Age**: Lower `max_track_age` to remove stale tracks faster
- **Increase IoU Threshold**: Higher `iou_threshold` for stricter matching

### Trajectory Performance

- **Use LOW Quality**: Set `quality: "LOW"` for real-time calculations
- **Reduce Depth**: Lower `max_collision_depth` for faster predictions
- **Limit Simulation Time**: Set lower `max_simulation_time_seconds`

### Memory Optimization

- **Disable Caching**: Set `enable_frame_cache: false` if memory constrained
- **Reduce Buffer Size**: Lower `buffer_size` for camera capture
- **Limit Event Queue**: Set `max_queue_size` for event manager

---

## Related Documentation

- **`docs/ARCHITECTURE.md`** - System architecture overview
- **`docs/CONFIG.md`** - Configuration parameter reference
- **`docs/REFACTORING_SUMMARY.md`** - Recent refactoring changes
- **`docs/API_REFERENCE.md`** - REST API and WebSocket documentation
- **`docs/DEPLOYMENT_GUIDE.md`** - Production deployment guide

---

## Support

For issues, questions, or contributions:
- **GitHub Issues**: Report bugs and request features
- **Pull Requests**: Submit code improvements
- **Documentation**: Update this README for clarifications

**Contributors**: See `CONTRIBUTORS.md` for list of contributors
