# Vector2D and Coordinate Systems Analysis

**Date:** 2025-10-21
**Status:** Complete
**Purpose:** Comprehensive analysis of all Vector2D usage and coordinate systems in the backend

---

## Executive Summary

The billiards-trainer backend has undergone a significant coordinate system refactoring. Vector2D has been moved from `core/models.py` to `core/coordinates.py` and enhanced with coordinate space metadata. This analysis documents ALL files using coordinates and identifies what needs to be modified for any future coordinate-related changes.

### Key Findings

1. **Vector2D Migration Completed**: All Python files now import Vector2D from `core.coordinates` (migration completed 2025-10-21)
2. **Four Coordinate Spaces in Use**:
   - WORLD_METERS (canonical storage format)
   - CAMERA_PIXELS (native camera resolution)
   - TABLE_PIXELS (calibrated table area)
   - NORMALIZED ([0,1] relative coordinates)
3. **Dual Vector2D Implementation**: Both enhanced (with metadata) and legacy (without metadata) versions coexist
4. **Conversion Infrastructure**: Full coordinate converter system in place with pixels_per_meter scaling

---

## 1. Complete File Inventory

### 1.1 Core Coordinate System Files (3 files)

These are the foundation files that define coordinate types and conversions:

| File | Purpose | Coordinate Systems |
|------|---------|-------------------|
| `/backend/core/coordinates.py` | Enhanced Vector2D class with coordinate space metadata | WORLD_METERS, CAMERA_PIXELS, TABLE_PIXELS, NORMALIZED |
| `/backend/core/coordinate_converter.py` | Centralized coordinate conversion utilities | All conversions between spaces |
| `/backend/core/resolution_config.py` | Resolution configuration management | Camera/table resolution tracking |

**Coordinate Systems Defined:**

```python
# From coordinates.py
class CoordinateSpace(Enum):
    WORLD_METERS = "world_meters"        # Canonical format: 2.54m × 1.27m for 9ft table
    CAMERA_PIXELS = "camera_pixels"      # Native camera: typically 1920x1080
    TABLE_PIXELS = "table_pixels"        # Calibrated playing surface
    NORMALIZED = "normalized"            # Relative [0,1] coordinates
```

**Key Classes:**
- `Vector2D`: Enhanced 2D vector with optional coordinate_space and resolution metadata
- `Resolution`: Immutable resolution metadata (width, height)
- `CoordinateConverter`: Central conversion hub using pixels_per_meter
- `PerspectiveTransform`: Homography-based transformations

---

### 1.2 Core Data Models (5 files)

These define the primary data structures using Vector2D:

| File | Vector2D Usage | Coordinate Pattern |
|------|---------------|-------------------|
| `/backend/core/models.py` | BallState, TableState, CueState, Collision, Trajectory, ShotAnalysis | Positions stored in WORLD_METERS (canonical) |
| `/backend/core/game_state.py` | GameState management | Uses models with WORLD_METERS internally |
| `/backend/vision/models.py` | Ball, CueStick, Table (vision-specific) | Positions in CAMERA_PIXELS with metadata |
| `/backend/api/models/converters.py` | API request/response conversions | Converts between coordinate spaces |
| `/backend/api/models/vision_models.py` | Vision API models | Camera pixel coordinates |

**BallState Factory Methods** (from models.py):
```python
# Recommended: Create in world meters
BallState.create(id, x, y)  # Auto-tags WORLD_METERS

# From camera pixels (with auto-conversion if converter provided)
BallState.from_camera_pixels(id, x, y, camera_resolution, converter)

# From table pixels
BallState.from_table_pixels(id, x, y, table_resolution, converter)

# From normalized coordinates
BallState.from_normalized(id, x, y, converter)
```

**Coordinate Conversion Helpers**:
```python
ball.get_position_in_space(target_space, converter, resolution)
ball.to_camera_pixels(converter, resolution)
ball.to_table_pixels(converter, resolution)
ball.has_coordinate_metadata()  # Check if using enhanced Vector2D
```

---

### 1.3 Physics & Simulation Files (6 files)

These use Vector2D for physics calculations (all in WORLD_METERS):

| File | Vector2D Usage | Purpose |
|------|---------------|---------|
| `/backend/core/physics/trajectory.py` | TrajectoryPoint positions, velocities, accelerations | Ball path calculation |
| `/backend/core/physics/collision.py` | Collision points, velocities | Ball-ball and ball-cushion collisions |
| `/backend/core/physics/spin.py` | Spin vectors | Top/back/side spin effects |
| `/backend/core/physics/engine.py` | Physics state management | Main physics loop |
| `/backend/core/collision/geometric_collision.py` | Collision detection points | Geometric collision algorithms |
| `/backend/core/analysis/prediction.py` | Predicted positions | Shot outcome prediction |

**Coordinate Space**: All physics calculations use WORLD_METERS for accuracy and consistency.

**Common Patterns**:
```python
# Velocity and acceleration vectors
velocity: Vector2D = Vector2D.zero(CoordinateSpace.WORLD_METERS)
acceleration: Vector2D = field(default_factory=lambda: Vector2D.zero(CoordinateSpace.WORLD_METERS))

# Physics calculations directly on x, y
force_x = ball.velocity.x * friction
force_y = ball.velocity.y * friction

# Vector math preserves metadata
new_velocity = old_velocity + acceleration * dt
```

---

### 1.4 Analysis & Assistance Files (3 files)

Shot analysis and player assistance:

| File | Vector2D Usage | Coordinate Pattern |
|------|---------------|-------------------|
| `/backend/core/analysis/shot.py` | Shot positions, target positions | WORLD_METERS |
| `/backend/core/analysis/assistance.py` | Recommended aim points, cue positions | WORLD_METERS |
| `/backend/core/utils/example_cue_pointing.py` | Example cue calculations | WORLD_METERS |

---

### 1.5 Validation Files (4 files)

State validation and correction:

| File | Vector2D Usage | Purpose |
|------|---------------|---------|
| `/backend/core/validation/state.py` | Ball position validation | Checks positions are on table |
| `/backend/core/validation/physics.py` | Physics state validation | Validates velocities, energy |
| `/backend/core/validation/correction.py` | Position correction | Fixes invalid positions |
| `/backend/core/validation/table_state.py` | Table boundary validation | Pocket and cushion checks |

**Common Patterns**:
```python
# Position validation
if not table.is_point_on_table(ball.position, ball.radius):
    errors.append(f"Ball {ball.id} outside table bounds")

# Distance calculations (works with both Vector2D types)
distance = ball1.distance_to(ball2)  # Uses simple x,y calculation
```

---

### 1.6 Utility Files (3 files)

Math and geometry utilities:

| File | Vector2D Usage | Purpose |
|------|---------------|---------|
| `/backend/core/utils/math.py` | Vector math operations | Dot product, cross product, normalization |
| `/backend/core/utils/geometry.py` | Geometric calculations | Line intersections, angles, distances |
| `/backend/core/__init__.py` | Core module exports | Exports Vector2D and models |

---

### 1.7 Vision Detection Files (7 files)

Computer vision and ball detection:

| File | Vector2D Usage | Coordinate Pattern |
|------|---------------|-------------------|
| `/backend/vision/detection/detector_adapter.py` | Ball, CueStick positions | CAMERA_PIXELS → conversion helpers |
| `/backend/vision/detection/balls.py` | Ball center detection | CAMERA_PIXELS from YOLO bbox |
| `/backend/vision/detection/cue.py` | Cue tip position detection | CAMERA_PIXELS |
| `/backend/vision/detection/yolo_detector.py` | YOLO bbox to center conversion | CAMERA_PIXELS |
| `/backend/vision/detection/table.py` | Table corner detection | CAMERA_PIXELS |
| `/backend/vision/tracking/tracker.py` | Ball tracking positions | CAMERA_PIXELS with history |
| `/backend/vision/tracking/kalman.py` | Kalman filter positions | CAMERA_PIXELS |

**Vision Models** (from vision/models.py):
```python
@dataclass
class Ball:
    position: tuple[float, float]  # (x, y) in pixel coordinates
    velocity: tuple[float, float]  # pixels/second
    acceleration: tuple[float, float]  # pixels/second²
    coordinate_space: str = "pixel"
    source_resolution: Optional[tuple[int, int]] = None
```

**Conversion Pattern**:
```python
# YOLO detection (normalized or pixel bbox)
bbox = [x, y, w, h]  # or [x1, y1, x2, y2]

# Convert to center + radius in CAMERA_PIXELS
center_x = x + w/2
center_y = y + h/2
radius = (w + h) / 4

# Create Ball with metadata
ball = Ball(
    position=(center_x, center_y),
    coordinate_space="pixel",
    source_resolution=(width, height)
)
```

---

### 1.8 Vision Calibration Files (4 files)

Camera and table calibration:

| File | Vector2D Usage | Purpose |
|------|---------------|---------|
| `/backend/vision/calibration/camera.py` | Camera intrinsics | Distortion correction |
| `/backend/vision/calibration/geometry.py` | Table corner positions | Homography calculation |
| `/backend/vision/calibration/validation.py` | Calibration verification | Accuracy checks |
| `/backend/vision/calibration/manager.py` | Calibration orchestration | Manages calibration process |

**Calibration Data**:
```python
@dataclass
class GeometricCalibration:
    table_corners_pixel: list[tuple[float, float]]  # CAMERA_PIXELS
    table_corners_world: list[tuple[float, float]]  # WORLD_METERS
    homography_matrix: NDArray[np.float64]  # 3x3 transform
    pixels_per_meter: float  # Scale factor
```

---

### 1.9 API Integration Files (5 files)

WebSocket and API endpoints:

| File | Vector2D Usage | Coordinate Pattern |
|------|---------------|-------------------|
| `/backend/api/websocket/broadcaster.py` | Broadcast ball positions | WORLD_METERS → CAMERA_PIXELS for clients |
| `/backend/api/websocket/manager.py` | WebSocket state | Converts coordinates for transmission |
| `/backend/api/routes/debug.py` | Debug coordinate info | Shows multiple coordinate spaces |
| `/backend/api/routes/calibration.py` | Calibration API | CAMERA_PIXELS input/output |
| `/backend/api/routes/config.py` | Configuration | Resolution settings |

**Broadcast Pattern**:
```python
# Backend stores in WORLD_METERS
ball = BallState.create("ball_1", x=1.27, y=0.635)

# Convert to CAMERA_PIXELS for client
camera_pos = ball.to_camera_pixels(converter, Resolution(1920, 1080))

# Send to WebSocket clients
await broadcast({
    "ball_id": ball.id,
    "position_world": {"x": ball.position.x, "y": ball.position.y},
    "position_camera": {"x": camera_pos.x, "y": camera_pos.y}
})
```

---

### 1.10 Integration & Service Files (2 files)

Service layer integration:

| File | Vector2D Usage | Purpose |
|------|---------------|---------|
| `/backend/integration_service.py` | Vision → Core integration | CAMERA_PIXELS → WORLD_METERS |
| `/backend/integration_service_conversion_helpers.py` | Coordinate conversion helpers | Manages conversions |

**Integration Pattern**:
```python
# Vision detects in CAMERA_PIXELS
vision_ball = Ball(position=(960, 540), ...)

# Convert to WORLD_METERS for core
converter = get_coordinate_converter()
ball_state = BallState.from_camera_pixels(
    id="ball_1",
    x=vision_ball.position[0],
    y=vision_ball.position[1],
    camera_resolution=Resolution(1920, 1080),
    converter=converter  # Auto-converts to WORLD_METERS
)
```

---

### 1.11 Test Files (10+ files)

Comprehensive test coverage:

| File | Purpose |
|------|---------|
| `/backend/tests/unit/test_coordinate_converter.py` | CoordinateConverter tests |
| `/backend/tests/unit/test_coordinate_conversion.py` | Conversion accuracy tests |
| `/backend/tests/core/test_resolution_config.py` | Resolution config tests |
| `/backend/tests/unit/test_multiball_trajectory.py` | Physics trajectory tests |
| `/backend/tests/integration/test_vision_core_integration.py` | Vision→Core integration |
| `/backend/tests/conftest.py` | Test fixtures with Vector2D |

---

## 2. Coordinate System Patterns

### 2.1 Current Coordinate Spaces

#### WORLD_METERS (Canonical Storage)
- **Usage**: All persistent storage, physics calculations, game state
- **Units**: Meters (standard 9ft table = 2.54m × 1.27m)
- **Advantages**: Resolution-independent, physically accurate, no rounding errors
- **Files**: All core physics, models, game state

```python
# Standard table dimensions
STANDARD_TABLE_WIDTH_METERS = 2.54   # 9 feet
STANDARD_TABLE_HEIGHT_METERS = 1.27  # 4.5 feet

# Ball positions always stored in meters
ball = BallState.create("ball_1", x=1.27, y=0.635)  # Center of table
```

#### CAMERA_PIXELS (Native Camera)
- **Usage**: Vision detection, camera capture, YOLO output
- **Units**: Pixels (typically 1920×1080)
- **Advantages**: Direct from hardware, no conversion needed for display
- **Files**: All vision detection, calibration, API responses

```python
# Camera native resolution
CAMERA_NATIVE_RESOLUTION = Resolution(width=1920, height=1080)

# Ball detected at camera center
ball = Ball(position=(960, 540), source_resolution=(1920, 1080))
```

#### TABLE_PIXELS (Calibrated Playing Area)
- **Usage**: Table-specific rendering, region-of-interest processing
- **Units**: Pixels (variable resolution, e.g., 640×360)
- **Advantages**: Normalized to table area, good for ROI processing
- **Files**: Vision calibration, table detection

```python
# Convert world position to table pixels for rendering
table_res = Resolution(640, 360)
table_pos = converter.world_meters_to_table_pixels(
    ball.position,
    table_res,
    table_corners
)
```

#### NORMALIZED ([0, 1])
- **Usage**: Resolution-independent positioning, interpolation
- **Units**: Normalized [0,1] range
- **Advantages**: Resolution-independent, good for UI scaling
- **Files**: Some API responses, interpolation utilities

```python
# Normalized center
center = Vector2D.normalized(0.5, 0.5)

# Convert to any resolution
pixels = converter.normalized_to_camera_pixels(center, target_resolution)
```

---

### 2.2 Conversion Patterns

#### Pattern 1: Vision Detection → Core Storage
```python
# 1. Vision detects in CAMERA_PIXELS
vision_ball = Ball(position=(960, 540), coordinate_space="pixel")

# 2. Convert to WORLD_METERS using calibrated converter
converter = get_coordinate_converter()  # Has pixels_per_meter from calibration
ball_state = BallState.from_camera_pixels(
    id="ball_1",
    x=vision_ball.position[0],
    y=vision_ball.position[1],
    camera_resolution=Resolution(1920, 1080),
    converter=converter  # Automatic conversion
)

# 3. Ball now in WORLD_METERS (canonical format)
assert ball_state.position.coordinate_space == CoordinateSpace.WORLD_METERS
```

#### Pattern 2: Core Storage → API Response
```python
# 1. Core stores in WORLD_METERS
ball = BallState.create("ball_1", x=1.27, y=0.635)

# 2. Convert to CAMERA_PIXELS for client display
camera_res = Resolution(1920, 1080)
camera_pos = ball.to_camera_pixels(converter, camera_res)

# 3. Send both for flexibility
response = {
    "position_world": {"x": ball.position.x, "y": ball.position.y},
    "position_camera": {"x": camera_pos.x, "y": camera_pos.y}
}
```

#### Pattern 3: Multi-Resolution Support
```python
# Single ball position converted to different resolutions
ball = BallState.create("ball_1", x=1.27, y=0.635)

# Different client resolutions
pos_1080p = ball.to_camera_pixels(converter, Resolution(1920, 1080))
pos_720p = ball.to_camera_pixels(converter, Resolution(1280, 720))
pos_480p = ball.to_camera_pixels(converter, Resolution(854, 480))
```

---

### 2.3 Pixels Per Meter Scaling

The `pixels_per_meter` constant is the key conversion factor:

```python
# From CoordinateConverter
class CoordinateConverter:
    def __init__(
        self,
        table_width_meters: float = 2.54,
        table_height_meters: float = 1.27,
        pixels_per_meter: float = 754.0,  # Default calibration
        camera_resolution: Resolution = Resolution(1920, 1080)
    ):
        self.pixels_per_meter = pixels_per_meter
```

**Calibration Sources**:
1. Manual calibration (measure table, count pixels)
2. Geometric calibration (table corner detection)
3. Default fallback (754.0 for standard setup)

**Usage in Conversion**:
```python
# Camera pixels → World meters
def camera_pixels_to_world_meters(self, vector, camera_resolution):
    # Scale to native resolution if needed
    scaled_vector = self._scale_if_needed(vector, camera_resolution)

    # Convert pixels to meters
    world_x = scaled_vector.x / self.pixels_per_meter
    world_y = scaled_vector.y / self.pixels_per_meter

    return Vector2D.world_meters(world_x, world_y)

# World meters → Camera pixels
def world_meters_to_camera_pixels(self, vector, camera_resolution):
    # Convert meters to pixels
    pixel_x = vector.x * self.pixels_per_meter
    pixel_y = vector.y * self.pixels_per_meter

    # Scale to target resolution if needed
    return self._scale_if_needed(Vector2D(pixel_x, pixel_y), camera_resolution)
```

---

## 3. Existing Conversion Logic

### 3.1 CoordinateConverter Class

**Location**: `/backend/core/coordinate_converter.py`

**Key Methods**:

| Method | From | To | Notes |
|--------|------|-----|-------|
| `camera_pixels_to_world_meters()` | CAMERA_PIXELS | WORLD_METERS | Uses pixels_per_meter |
| `world_meters_to_camera_pixels()` | WORLD_METERS | CAMERA_PIXELS | Inverse of above |
| `table_pixels_to_world_meters()` | TABLE_PIXELS | WORLD_METERS | Uses table corners if provided |
| `world_meters_to_table_pixels()` | WORLD_METERS | TABLE_PIXELS | Uses table corners if provided |
| `normalized_to_camera_pixels()` | NORMALIZED | CAMERA_PIXELS | Simple multiplication |
| `camera_pixels_to_normalized()` | CAMERA_PIXELS | NORMALIZED | Simple division |
| `convert()` | Any | Any | Generic conversion hub |

**Batch Operations** (for performance):
```python
# Convert multiple points at once
positions = [Vector2D(100, 100), Vector2D(200, 200), ...]
world_positions = converter.camera_pixels_to_world_meters_batch(
    positions,
    camera_resolution
)
```

**Perspective Transform Support**:
```python
# For advanced perspective correction
transform = PerspectiveTransform(matrix=homography_matrix)
corrected_pos = transform.apply(camera_pos)
```

---

### 3.2 BallState Factory Methods

**Location**: `/backend/core/models.py`

```python
# Factory methods with automatic coordinate metadata
class BallState:
    @classmethod
    def create(cls, id, x, y, **kwargs):
        """Create in WORLD_METERS (recommended)"""
        position = Vector2D.world_meters(x, y)
        return cls(id=id, position=position, **kwargs)

    @classmethod
    def from_camera_pixels(cls, id, x, y, camera_resolution, converter=None, **kwargs):
        """Create from CAMERA_PIXELS, optionally convert to WORLD_METERS"""
        position = Vector2D.camera_pixels(x, y, camera_resolution)
        if converter:
            position = converter.convert(position, CoordinateSpace.WORLD_METERS)
        return cls(id=id, position=position, **kwargs)

    @classmethod
    def from_table_pixels(cls, id, x, y, table_resolution, converter=None, **kwargs):
        """Create from TABLE_PIXELS, optionally convert to WORLD_METERS"""
        position = Vector2D.table_pixels(x, y, table_resolution)
        if converter:
            position = converter.convert(position, CoordinateSpace.WORLD_METERS)
        return cls(id=id, position=position, **kwargs)

    @classmethod
    def from_normalized(cls, id, x, y, converter=None, **kwargs):
        """Create from NORMALIZED, optionally convert to WORLD_METERS"""
        position = Vector2D.normalized(x, y)
        if converter:
            position = converter.convert(position, CoordinateSpace.WORLD_METERS)
        return cls(id=id, position=position, **kwargs)
```

---

### 3.3 Integration Service Helpers

**Location**: `/backend/integration_service_conversion_helpers.py`

**Purpose**: High-level conversion helpers for vision→core integration

```python
def convert_vision_ball_to_core_ball(
    vision_ball: vision.Ball,
    converter: CoordinateConverter,
    camera_resolution: Resolution
) -> BallState:
    """Convert vision Ball to core BallState"""
    return BallState.from_camera_pixels(
        id=f"ball_{vision_ball.track_id}",
        x=vision_ball.position[0],
        y=vision_ball.position[1],
        camera_resolution=camera_resolution,
        converter=converter,  # Auto-converts to WORLD_METERS
        number=vision_ball.number,
        confidence=vision_ball.confidence
    )
```

---

## 4. Files Requiring Modification

### 4.1 High-Priority Modification Groups

#### Group A: Core Coordinate Infrastructure (3 files)
**Impact**: ALL coordinate operations
**Requires coordinated changes**: Yes

1. `/backend/core/coordinates.py` - Vector2D class and coordinate space enums
2. `/backend/core/coordinate_converter.py` - Conversion algorithms
3. `/backend/core/resolution_config.py` - Resolution management

**Modification Scenarios**:
- Adding new coordinate space (e.g., PROJECTOR_PIXELS)
- Changing conversion algorithms
- Adding new Vector2D operations

#### Group B: Vision Detection Pipeline (7 files)
**Impact**: All vision input
**Can be modified in parallel**: Yes

1. `/backend/vision/detection/detector_adapter.py` - YOLO → Ball conversion
2. `/backend/vision/detection/balls.py` - Ball detection
3. `/backend/vision/detection/cue.py` - Cue detection
4. `/backend/vision/detection/yolo_detector.py` - YOLO inference
5. `/backend/vision/tracking/tracker.py` - Ball tracking
6. `/backend/vision/tracking/kalman.py` - Kalman filtering
7. `/backend/vision/models.py` - Vision data models

**Modification Scenarios**:
- Changing detection resolution
- Adding new detection metadata
- Improving coordinate accuracy

#### Group C: Physics Engine (6 files)
**Impact**: Simulation accuracy
**Can be modified in parallel**: Yes

1. `/backend/core/physics/trajectory.py` - Trajectory calculation
2. `/backend/core/physics/collision.py` - Collision detection
3. `/backend/core/physics/spin.py` - Spin effects
4. `/backend/core/physics/engine.py` - Physics loop
5. `/backend/core/collision/geometric_collision.py` - Geometric collisions
6. `/backend/core/analysis/prediction.py` - Shot prediction

**Modification Scenarios**:
- Improving physics accuracy
- Adding new force types
- Changing simulation parameters

#### Group D: API & WebSocket (5 files)
**Impact**: Client communication
**Can be modified in parallel**: Yes

1. `/backend/api/websocket/broadcaster.py` - Position broadcasting
2. `/backend/api/models/converters.py` - API conversions
3. `/backend/api/routes/debug.py` - Debug endpoints
4. `/backend/api/routes/calibration.py` - Calibration API
5. `/backend/api/models/vision_models.py` - API models

**Modification Scenarios**:
- Adding new API coordinate formats
- Optimizing WebSocket bandwidth
- Adding coordinate space metadata to responses

#### Group E: Integration Services (2 files)
**Impact**: Vision→Core integration
**Requires coordinated changes**: Yes

1. `/backend/integration_service.py` - Main integration
2. `/backend/integration_service_conversion_helpers.py` - Conversion helpers

**Modification Scenarios**:
- Changing integration pipeline
- Adding new conversion strategies
- Optimizing conversion performance

---

### 4.2 Medium-Priority Groups

#### Group F: Validation & Correction (4 files)
1. `/backend/core/validation/state.py`
2. `/backend/core/validation/physics.py`
3. `/backend/core/validation/correction.py`
4. `/backend/core/validation/table_state.py`

#### Group G: Analysis & Assistance (3 files)
1. `/backend/core/analysis/shot.py`
2. `/backend/core/analysis/assistance.py`
3. `/backend/core/utils/example_cue_pointing.py`

#### Group H: Utilities (3 files)
1. `/backend/core/utils/math.py`
2. `/backend/core/utils/geometry.py`
3. `/backend/core/__init__.py`

---

### 4.3 Low-Priority Groups

#### Group I: Calibration (4 files)
1. `/backend/vision/calibration/camera.py`
2. `/backend/vision/calibration/geometry.py`
3. `/backend/vision/calibration/validation.py`
4. `/backend/vision/calibration/manager.py`

#### Group J: Tests (10+ files)
- All test files can be modified independently
- Tests should be updated after corresponding module changes

---

## 5. Edge Cases & Special Considerations

### 5.1 Legacy Vector2D Support

**Dual Implementation**: System supports both enhanced and legacy Vector2D

```python
# Enhanced Vector2D (with metadata)
ball = BallState.create("ball_1", x=1.27, y=0.635)
assert ball.position.coordinate_space == CoordinateSpace.WORLD_METERS

# Legacy Vector2D (without metadata - for backward compatibility)
legacy_ball = BallState(id="ball_2", position=Vector2D(1.0, 0.5))
assert not legacy_ball.has_coordinate_metadata()
```

**Serialization Compatibility**:
```json
// Enhanced format (with metadata)
{
  "id": "ball_1",
  "position": {
    "x": 1.27,
    "y": 0.635,
    "coordinate_space": "world_meters"
  }
}

// Legacy format (still supported)
{
  "id": "ball_2",
  "position": {
    "x": 1.0,
    "y": 0.5
  }
}
```

---

### 5.2 Distance Calculations

**Important**: Distance calculations work with BOTH Vector2D types by using simple x,y math:

```python
# From BallState.distance_to()
def distance_to(self, other: "BallState") -> float:
    # Simple approach - always use raw distance calculation to avoid type issues
    dx = self.position.x - other.position.x
    dy = self.position.y - other.position.y
    return (dx * dx + dy * dy) ** 0.5
```

This pattern avoids coordinate space validation issues and works regardless of metadata.

---

### 5.3 Tuple vs Vector2D

**Vision Models** use tuples for simplicity:
```python
# Vision Ball uses tuples
class Ball:
    position: tuple[float, float]  # Not Vector2D
    velocity: tuple[float, float]
```

**Core Models** use Vector2D:
```python
# Core BallState uses Vector2D
class BallState:
    position: Vector2D
    velocity: Vector2D
```

**Conversion Required** at vision→core boundary.

---

### 5.4 Resolution Scaling Edge Cases

**Different Source Resolution**:
```python
# Detection at 640x640 (YOLO input)
yolo_pos = (320, 320)

# Convert to camera native (1920x1080)
camera_pos = converter.convert(
    Vector2D(yolo_pos[0], yolo_pos[1]),
    from_space=CoordinateSpace.DETECTION_PIXELS,
    to_space=CoordinateSpace.CAMERA_PIXELS,
    from_resolution=Resolution(640, 640),
    to_resolution=Resolution(1920, 1080)
)
```

**Aspect Ratio Mismatch**:
- YOLO input: 640×640 (1:1)
- Camera native: 1920×1080 (16:9)
- Requires careful handling of letterboxing/cropping

---

### 5.5 Perspective Transform

**Optional Perspective Correction**:
```python
# Create converter with perspective transform
transform = PerspectiveTransform(matrix=homography_matrix)
converter = CoordinateConverter(
    perspective_transform=transform
)

# Conversions automatically apply perspective correction
world_pos = converter.camera_pixels_to_world_meters(camera_pos)
```

**Use Cases**:
- Camera mounted at angle
- Table not parallel to image plane
- Wide-angle lens distortion

---

## 6. Recommended Parallel Modification Strategy

### Phase 1: Core Infrastructure (Sequential)
Must be done first, in order:

1. Modify `coordinates.py` (new coordinate spaces, Vector2D methods)
2. Update `coordinate_converter.py` (new conversion algorithms)
3. Update `resolution_config.py` (new resolution metadata)

### Phase 2: Domain Modules (Parallel)
Can be done in parallel by different developers:

**Team A: Vision Pipeline**
- Group B files (vision detection)
- Group I files (calibration)

**Team B: Physics & Analysis**
- Group C files (physics)
- Group G files (analysis)

**Team C: API & Integration**
- Group D files (API/WebSocket)
- Group E files (integration)

**Team D: Support Systems**
- Group F files (validation)
- Group H files (utilities)

### Phase 3: Tests & Documentation (Parallel)
After Phase 2 completion:

- Update all test files (Group J)
- Update documentation markdown files
- Integration testing

---

## 7. Verification Checklist

Before considering coordinate changes complete:

### Functional Verification
- [ ] Vision detection produces valid CAMERA_PIXELS coordinates
- [ ] Coordinates convert to WORLD_METERS correctly
- [ ] Physics simulation uses WORLD_METERS consistently
- [ ] API responses provide both WORLD_METERS and CAMERA_PIXELS
- [ ] Multi-resolution support works (1080p, 720p, etc.)

### Data Integrity
- [ ] Round-trip conversions preserve accuracy (camera→world→camera)
- [ ] Serialization/deserialization preserves metadata
- [ ] Legacy Vector2D compatibility maintained
- [ ] Distance calculations work with mixed Vector2D types

### Performance
- [ ] Batch conversions used where appropriate
- [ ] No unnecessary conversions in hot paths
- [ ] Caching works for repeated conversions

### Code Quality
- [ ] All imports use `from backend.core.coordinates import Vector2D`
- [ ] Factory methods used instead of direct BallState construction
- [ ] Coordinate space metadata included where available
- [ ] Type hints updated for Vector2D

---

## 8. Summary Statistics

### File Counts by Category

| Category | File Count | Coordinate Spaces Used |
|----------|-----------|----------------------|
| Core Coordinate System | 3 | All 4 spaces |
| Core Data Models | 5 | WORLD_METERS primarily |
| Physics & Simulation | 6 | WORLD_METERS only |
| Analysis & Assistance | 3 | WORLD_METERS only |
| Validation | 4 | WORLD_METERS only |
| Utilities | 3 | All types |
| Vision Detection | 7 | CAMERA_PIXELS primarily |
| Vision Calibration | 4 | CAMERA_PIXELS, WORLD_METERS |
| API Integration | 5 | All 4 spaces (conversion layer) |
| Integration Services | 2 | All 4 spaces (conversion layer) |
| Tests | 10+ | All 4 spaces |

**Total**: ~50 Python files using Vector2D or coordinates

### Coordinate Space Usage

| Coordinate Space | Primary Usage | File Count |
|-----------------|---------------|-----------|
| WORLD_METERS | Storage, physics, analysis | ~25 files |
| CAMERA_PIXELS | Detection, calibration, vision | ~15 files |
| TABLE_PIXELS | ROI processing, rendering | ~5 files |
| NORMALIZED | UI scaling, interpolation | ~3 files |

### Conversion Points

| Conversion | Location | Frequency |
|-----------|----------|-----------|
| CAMERA_PIXELS → WORLD_METERS | Integration service | Every frame |
| WORLD_METERS → CAMERA_PIXELS | WebSocket broadcaster | Every broadcast |
| Any → Any | Debug API | On demand |

---

## 9. Migration Status

✅ **Completed (2025-10-21)**:
- Vector2D moved to `core/coordinates.py`
- All Python files updated to import from `core.coordinates`
- Enhanced Vector2D with coordinate space metadata
- Coordinate converter infrastructure
- Factory methods for BallState creation
- Backward compatibility maintained

⚠️ **Documentation Still Using Old Imports**:
- Several .md files still show old import examples
- Not critical for functionality
- Can be updated separately

🎯 **Future Enhancements**:
- Add DETECTION_PIXELS as distinct space
- Implement perspective transform caching
- Add coordinate space validation in physics
- Optimize batch conversion performance

---

## Appendix A: Quick Reference

### Creating Balls with Coordinates

```python
# Recommended: World meters (canonical)
ball = BallState.create("ball_1", x=1.27, y=0.635, number=1)

# From camera pixels (with conversion)
ball = BallState.from_camera_pixels(
    "ball_1", x=960, y=540,
    camera_resolution=Resolution(1920, 1080),
    converter=converter,
    number=1
)

# From vision detection
vision_ball = Ball(position=(960, 540), ...)
core_ball = convert_vision_ball_to_core_ball(vision_ball, converter, camera_res)
```

### Converting Positions

```python
# Get converter (has calibration data)
from backend.core.coordinate_converter import get_coordinate_converter
converter = get_coordinate_converter()

# Camera pixels → World meters
world_pos = converter.camera_pixels_to_world_meters(
    Vector2D(960, 540),
    Resolution(1920, 1080)
)

# World meters → Camera pixels
camera_pos = converter.world_meters_to_camera_pixels(
    Vector2D(1.27, 0.635),
    Resolution(1920, 1080)
)

# Generic conversion
any_pos = converter.convert(
    vector,
    from_space=CoordinateSpace.CAMERA_PIXELS,
    to_space=CoordinateSpace.WORLD_METERS,
    from_resolution=Resolution(1920, 1080)
)
```

### Checking Coordinate Metadata

```python
# Check if ball has coordinate metadata
if ball.has_coordinate_metadata():
    print(f"Ball is in {ball.position.coordinate_space}")
    print(f"Resolution: {ball.position.resolution}")
else:
    print("Ball uses legacy Vector2D without metadata")
```

---

**End of Analysis**
For questions or updates, see `/backend/core/MODELS_COORDINATE_MIGRATION.md`
