# Coordinate Systems Documentation

**Version**: 1.0
**Last Updated**: 2025-10-21
**Audience**: Developers working with billiards-trainer codebase

## Table of Contents

1. [Overview](#overview)
2. [The Four Coordinate Systems](#the-four-coordinate-systems)
3. [When to Use Each System](#when-to-use-each-system)
4. [Conversion Guide](#conversion-guide)
5. [Common Pitfalls](#common-pitfalls)
6. [Code Examples](#code-examples)
7. [Migration Guide](#migration-guide)
8. [Performance Considerations](#performance-considerations)
9. [Troubleshooting](#troubleshooting)

---

## Overview

The billiards-trainer system operates with multiple coordinate systems across different components. Understanding these systems and their proper usage is critical for:

- Accurate ball detection and tracking
- Correct physics simulation
- Proper rendering in the UI
- Successful coordinate transformations

### Quick Reference

| Coordinate System | Units | Origin | Primary Use |
|------------------|-------|--------|-------------|
| **Camera Native** | Pixels | Top-left (0,0) | Raw camera frames, detection input |
| **World Meters** | Meters | Table center | Physics engine, persistent storage |
| **Table Normalized** | Ratio [0,1] | Table top-left | Relative positioning, interpolation |
| **Canvas Pixels** | Pixels | Top-left (0,0) | Frontend rendering |

### Canonical Format

**All persistent data should be stored in World Meters coordinates.** This provides:
- Resolution independence
- Physical meaning
- Direct compatibility with physics engine
- Standard table dimensions (9ft table = 2.54m × 1.27m)

---

## The Four Coordinate Systems

### 1. Camera Native Coordinates

**Definition**: Pixel coordinates in the camera's native resolution (typically 1920×1080)

**Characteristics**:
- Origin: Top-left corner (0, 0)
- X-axis: Increases to the right
- Y-axis: Increases downward
- Units: Pixels
- Range: 0 ≤ x < 1920, 0 ≤ y < 1080 (for standard camera)

**Used by**:
- `backend/video/` - Video capture module
- `backend/vision/capture.py` - Frame acquisition
- `backend/vision/detection/` - YOLO and OpenCV detectors
- `backend/vision/calibration/geometry.py` - Calibration reference

**Data Structures**:
```python
# backend/vision/models.py
@dataclass
class Ball:
    position: tuple[float, float]  # In camera native pixels
    radius: float  # In pixels

# backend/api/models/vision_models.py
class Point2DModel(BaseModel):
    x: float  # Pixel coordinate
    y: float  # Pixel coordinate
```

**Example Values**:
```python
ball_position = (960.0, 540.0)  # Center of 1920×1080 frame
corner = (37, 45)  # Table corner in pixels
```

---

### 2. World Meters Coordinates

**Definition**: Real-world metric coordinates with origin at table center

**Characteristics**:
- Origin: Center of table playing area
- X-axis: Increases to the right (long axis of table)
- Y-axis: Increases downward (short axis of table)
- Units: Meters
- Range: Typically ±1.27m (x), ±0.635m (y) for standard 9ft table

**Used by**:
- `backend/core/models.py` - Core game state (BallState, TableState)
- `backend/core/physics/` - Physics engine
- `backend/core/collision/` - Collision detection
- API responses and WebSocket messages (canonical format)

**Data Structures**:
```python
# backend/core/models.py
@dataclass
class BallState:
    position: Vector2D  # In meters from table center
    velocity: Vector2D  # In meters/second

@dataclass
class TableState:
    width: float  # In meters (2.54m for 9ft table)
    height: float  # In meters (1.27m for 9ft table)
```

**Example Values**:
```python
ball_position = Vector2D(0.5, 0.2)  # 0.5m right, 0.2m down from center
table_center = Vector2D(0.0, 0.0)  # Origin
cue_tip = Vector2D(-0.8, -0.3)  # 0.8m left, 0.3m up from center
```

**Standard Table Dimensions**:
- 9ft table: 2.54m × 1.27m
- 8ft table: 2.44m × 1.22m
- 7ft table: 2.13m × 1.07m

---

### 3. Table Normalized Coordinates

**Definition**: Normalized coordinates in [0,1] range relative to table bounds

**Characteristics**:
- Origin: Top-left corner of playing area
- X-axis: 0 (left edge) to 1 (right edge)
- Y-axis: 0 (top edge) to 1 (bottom edge)
- Units: Ratio (dimensionless)
- Range: 0 ≤ x ≤ 1, 0 ≤ y ≤ 1

**Used by**:
- Internal calculations requiring resolution independence
- Relative positioning (e.g., "ball at 25% from left, 50% from top")
- Interpolation between positions
- Portable configuration that works across different setups

**Data Structures**:
```python
# Not widely used in current codebase, but planned for future
normalized_position = (0.25, 0.5)  # 25% right, 50% down
```

**Example Values**:
```python
top_left = (0.0, 0.0)
center = (0.5, 0.5)
bottom_right = (1.0, 1.0)
quarter_point = (0.25, 0.75)  # 1/4 from left, 3/4 from top
```

**Conversion to/from World Meters**:
```python
# Normalized → World Meters
world_x = (normalized_x - 0.5) * table_width
world_y = (normalized_y - 0.5) * table_height

# World Meters → Normalized
normalized_x = (world_x / table_width) + 0.5
normalized_y = (world_y / table_height) + 0.5
```

---

### 4. Canvas Pixels Coordinates

**Definition**: Pixel coordinates in the frontend canvas (browser viewport)

**Characteristics**:
- Origin: Top-left corner of canvas element
- X-axis: Increases to the right
- Y-axis: Increases downward
- Units: Pixels
- Range: Variable (depends on browser window size)

**Used by**:
- `frontend/web/src/components/video/` - React components
- `frontend/web/src/utils/coordinates.ts` - Coordinate transformations
- Canvas rendering and overlay drawing

**Data Structures**:
```typescript
// frontend/web/src/types/video.ts
interface Point2D {
  x: number;  // Canvas pixel coordinate
  y: number;
}

interface Size2D {
  width: number;   // Canvas width in pixels
  height: number;  // Canvas height in pixels
}
```

**Example Values**:
```typescript
canvasSize = { width: 800, height: 450 }
ballPosition = { x: 400, y: 225 }  // Center of canvas
cursorPosition = { x: 150, y: 80 }  // Mouse click location
```

---

## When to Use Each System

### Decision Tree

```
What are you doing?
│
├─ Capturing from camera?
│  └─ Use: Camera Native (1920×1080 pixels)
│
├─ Running physics simulation?
│  └─ Use: World Meters (canonical)
│
├─ Storing to database or API?
│  └─ Use: World Meters (canonical)
│
├─ Rendering in browser?
│  └─ Use: Canvas Pixels (variable size)
│
├─ Relative positioning?
│  └─ Use: Table Normalized [0,1]
│
└─ Passing between modules?
   └─ Use: World Meters (canonical)
```

### Use Case Matrix

| Task | Coordinate System | Rationale |
|------|------------------|-----------|
| Ball detection | Camera Native | Raw detector output |
| Calibration | Camera Native | Reference for pixel↔meter mapping |
| Physics calculation | World Meters | Native physics units |
| Trajectory storage | World Meters | Resolution independent |
| API responses | World Meters | Canonical, portable |
| Ball overlay rendering | Canvas Pixels | Direct canvas drawing |
| Cue aim line | Canvas Pixels | UI interaction |
| Config files | World Meters | Portable across setups |
| Relative ball position | Table Normalized | Resolution independent |

### Module Boundaries

**Vision → Core Integration**:
- Input: Camera Native pixels
- Output: World Meters
- Converter: `backend/integration_service_conversion_helpers.py`
- Uses calibration: `pixels_per_meter` factor

**Core → API**:
- Input: World Meters
- Output: World Meters (preserved)
- Converter: `backend/api/models/converters.py`
- Maintains coordinate space

**API → Frontend**:
- Input: World Meters
- Output: Canvas Pixels
- Converter: `frontend/web/src/utils/coordinates.ts`
- Requires: `canvasSize`, `videoSize`, calibration

---

## Conversion Guide

### Backend Conversions (Python)

#### Camera Native → World Meters

```python
# Using GeometricCalibrator
from backend.vision.calibration.geometry import GeometricCalibrator

calibrator = GeometricCalibrator()
calibration = calibrator.load_calibration()
mapping = calibration.coordinate_mapping

# Convert pixel position to meters
pixel_pos = (960.0, 540.0)  # Camera native
world_pos = calibrator.pixel_to_world_coordinates(pixel_pos, mapping)
# Returns: (x_meters, y_meters) from table center
```

**Formula**:
```python
# With calibration data
pixels_per_meter = mapping.scale_factor  # e.g., 754.0
translation_x, translation_y = mapping.translation  # Camera center offset

# Convert
world_x = (pixel_x - translation_x) / pixels_per_meter
world_y = (pixel_y - translation_y) / pixels_per_meter
```

#### World Meters → Camera Native

```python
# Using GeometricCalibrator
world_pos = (0.5, 0.2)  # Meters from table center
pixel_pos = calibrator.world_to_pixel_coordinates(world_pos, mapping)
# Returns: (x_pixels, y_pixels) in camera native
```

**Formula**:
```python
# With calibration data
pixel_x = world_x * pixels_per_meter + translation_x
pixel_y = world_y * pixels_per_meter + translation_y
```

#### World Meters → Table Normalized

```python
# Assuming table dimensions
table_width = 2.54  # meters
table_height = 1.27  # meters

# Convert from world (centered) to normalized
world_pos = (0.5, 0.2)
normalized_x = (world_pos[0] / table_width) + 0.5
normalized_y = (world_pos[1] / table_height) + 0.5
# Returns: (0.697, 0.657) in [0,1] range
```

#### Table Normalized → World Meters

```python
# Convert from normalized to world (centered)
normalized_pos = (0.697, 0.657)
world_x = (normalized_pos[0] - 0.5) * table_width
world_y = (normalized_pos[1] - 0.5) * table_height
# Returns: (0.5, 0.2) meters from center
```

#### Vision Ball → Core BallState

```python
# Complete example from vision detection to core state
from backend.vision.models import Ball
from backend.core.models import BallState, Vector2D
from backend.vision.calibration.geometry import GeometricCalibrator

# Vision detection output
vision_ball = Ball(
    position=(960.0, 540.0),  # Camera native pixels
    radius=22.5,
    ball_type=BallType.CUE,
    confidence=0.95
)

# Load calibration
calibrator = GeometricCalibrator()
calibration = calibrator.load_calibration()
mapping = calibration.coordinate_mapping

# Convert position to world meters
world_pos = calibrator.pixel_to_world_coordinates(
    vision_ball.position,
    mapping
)

# Create core ball state
ball_state = BallState(
    id="cue",
    position=Vector2D(world_pos[0], world_pos[1]),
    velocity=Vector2D(0.0, 0.0),
    radius=vision_ball.radius / mapping.scale_factor,  # Convert radius too
    is_cue_ball=True,
    confidence=vision_ball.confidence
)
```

### Frontend Conversions (TypeScript)

#### Creating Coordinate Transformer

```typescript
// frontend/web/src/utils/coordinates.ts
import { createCoordinateTransform } from '@/utils/coordinates';

// Setup
const videoSize = { width: 1920, height: 1080 };  // Camera native
const canvasSize = { width: 800, height: 450 };   // Current canvas

const transform = createCoordinateTransform(
  videoSize,
  canvasSize,
  { x: 0, y: 0, scale: 1, rotation: 0 }  // No viewport transform
);
```

#### Video (Camera Native) → Canvas

```typescript
// Convert ball position from camera native to canvas pixels
const ballInVideo: Point2D = { x: 960, y: 540 };  // Center of 1920×1080
const ballInCanvas = transform.videoToCanvas(ballInVideo);
// Returns: { x: 400, y: 225 } for 800×450 canvas
```

#### Canvas → Video (Camera Native)

```typescript
// Convert mouse click from canvas back to video coordinates
const clickInCanvas: Point2D = { x: 400, y: 225 };
const clickInVideo = transform.canvasToVideo(clickInCanvas);
// Returns: { x: 960, y: 540 } in camera native
```

#### Complete Rendering Example

```typescript
// Render balls received from API (in world meters) on canvas
import type { BallInfo } from '@/api/models';

interface RenderBallsProps {
  balls: BallInfo[];           // Positions in world meters
  canvasSize: Size2D;
  calibration: CalibrationData;
}

function renderBalls({ balls, canvasSize, calibration }: RenderBallsProps) {
  const ctx = canvas.getContext('2d');
  const { pixelsPerMeter } = calibration;

  // Video size is camera native resolution
  const videoSize = { width: 1920, height: 1080 };

  // Create transformer
  const transform = createCoordinateTransform(videoSize, canvasSize);

  balls.forEach(ball => {
    // Ball position is in world meters [x, y]
    const [worldX, worldY] = ball.position;

    // Convert world meters → camera native pixels
    const pixelX = worldX * pixelsPerMeter + (videoSize.width / 2);
    const pixelY = worldY * pixelsPerMeter + (videoSize.height / 2);

    // Convert camera native → canvas pixels
    const canvasPos = transform.videoToCanvas({ x: pixelX, y: pixelY });

    // Draw ball on canvas
    ctx.beginPath();
    ctx.arc(canvasPos.x, canvasPos.y, 5, 0, Math.PI * 2);
    ctx.fillStyle = ball.is_cue_ball ? 'white' : 'red';
    ctx.fill();
  });
}
```

### Conversion Chains

#### Vision → Physics → Frontend

```
Camera Detection → Core Physics → API → Frontend Rendering
    ↓                ↓            ↓          ↓
Camera Native → World Meters → World → Canvas Pixels
 (1920×1080)     (meters)      Meters   (800×450)
```

**Full Example**:
```python
# 1. Vision detection (Camera Native)
detected_ball = Ball(position=(960.0, 540.0))  # pixels

# 2. Convert to World Meters
world_pos = calibrator.pixel_to_world_coordinates(
    detected_ball.position, mapping
)  # → (0.0, 0.0) meters (table center)

# 3. Store in core state
ball_state = BallState(position=Vector2D(world_pos[0], world_pos[1]))

# 4. Return via API (stays in World Meters)
api_response = {
    "position": [ball_state.position.x, ball_state.position.y],
    "coordinate_space": "world_meters"
}

# 5. Frontend receives and converts to Canvas Pixels
# TypeScript
const [worldX, worldY] = apiResponse.position;
const pixelX = worldX * pixelsPerMeter + (1920 / 2);
const pixelY = worldY * pixelsPerMeter + (1080 / 2);
const canvasPos = transform.videoToCanvas({ x: pixelX, y: pixelY });
// Render at canvasPos
```

---

## Common Pitfalls

### 1. Mixing Coordinate Systems Without Conversion

**Problem**: Performing operations on coordinates from different systems

```python
# ❌ WRONG - Mixing pixels and meters
camera_pos = (960, 540)  # Camera native pixels
world_pos = (1.0, 0.5)   # World meters
distance = math.sqrt(
    (camera_pos[0] - world_pos[0])**2 +
    (camera_pos[1] - world_pos[1])**2
)  # NONSENSE! Comparing pixels to meters
```

**Solution**: Always convert to same coordinate system first

```python
# ✅ CORRECT - Convert to same system
camera_pos = (960, 540)
world_pos = (1.0, 0.5)

# Convert camera to world
camera_world = calibrator.pixel_to_world_coordinates(camera_pos, mapping)

# Now calculate distance in world meters
distance = math.sqrt(
    (camera_world[0] - world_pos[0])**2 +
    (camera_world[1] - world_pos[1])**2
)  # Correct! Both in meters
```

### 2. Assuming All Pixels Are Equal

**Problem**: Different detectors/displays use different resolutions

```python
# ❌ WRONG - YOLO uses 640×640, camera is 1920×1080
yolo_ball = Ball(position=(320, 320))    # YOLO output (640×640)
camera_ball = Ball(position=(320, 320))  # Camera frame (1920×1080)

# These are NOT the same location!
# (320,320) in 640×640 is center
# (320,320) in 1920×1080 is upper-left quadrant
```

**Solution**: Always track resolution and convert explicitly

```python
# ✅ CORRECT - Track source resolution
yolo_pos = (320, 320)      # In 640×640 space
yolo_resolution = (640, 640)

# Scale to camera native
scale_x = 1920 / 640
scale_y = 1080 / 640
camera_pos = (yolo_pos[0] * scale_x, yolo_pos[1] * scale_y)
# camera_pos = (960, 960) in 1920×1080 space
```

### 3. Forgetting to Convert Radius/Scale

**Problem**: Converting position but not dimensions

```python
# ❌ WRONG - Position converted but radius not
world_pos = calibrator.pixel_to_world_coordinates(ball.position, mapping)
ball_state = BallState(
    position=Vector2D(world_pos[0], world_pos[1]),
    radius=ball.radius  # WRONG! Still in pixels
)
```

**Solution**: Convert all dimensions consistently

```python
# ✅ CORRECT - Convert both position and radius
world_pos = calibrator.pixel_to_world_coordinates(ball.position, mapping)
ball_state = BallState(
    position=Vector2D(world_pos[0], world_pos[1]),
    radius=ball.radius / mapping.scale_factor  # Convert to meters
)
```

### 4. Using Table Corners Without Resolution Context

**Problem**: Table corner configuration without knowing resolution

```json
{
  "table": {
    "playing_area_corners": [
      {"x": 37, "y": 45},
      {"x": 606, "y": 39}
    ]
  }
}
```

**Question**: Are these in camera native (1920×1080) or some scaled resolution?

**Solution**: Always include resolution metadata

```json
{
  "table": {
    "playing_area_corners": [
      {
        "x": 37,
        "y": 45,
        "coordinate_space": "camera_native",
        "resolution": {"width": 1920, "height": 1080}
      }
    ]
  }
}
```

### 5. Not Handling Aspect Ratio Differences

**Problem**: Non-uniform scaling when aspect ratios don't match

```typescript
// ❌ WRONG - Distorts image
const scaleX = canvasWidth / videoWidth;
const scaleY = canvasHeight / videoHeight;
canvasX = videoX * scaleX;  // Different scale factors
canvasY = videoY * scaleY;  // cause distortion
```

**Solution**: Use uniform scaling with letterboxing

```typescript
// ✅ CORRECT - Maintains aspect ratio
const scaleX = canvasWidth / videoWidth;
const scaleY = canvasHeight / videoHeight;
const uniformScale = Math.min(scaleX, scaleY);

// Center with letterbox
const scaledWidth = videoWidth * uniformScale;
const scaledHeight = videoHeight * uniformScale;
const offsetX = (canvasWidth - scaledWidth) / 2;
const offsetY = (canvasHeight - scaledHeight) / 2;

canvasX = videoX * uniformScale + offsetX;
canvasY = videoY * uniformScale + offsetY;
```

### 6. Ignoring Coordinate System Origin Differences

**Problem**: Camera native (top-left origin) vs world meters (center origin)

```python
# ❌ WRONG - Forgetting origin shift
# Camera is (0,0) at top-left
# World is (0,0) at table center
pixel_x = world_x * pixels_per_meter  # Missing translation!
```

**Solution**: Account for origin offset

```python
# ✅ CORRECT - Include translation
camera_center_x = 1920 / 2  # 960
camera_center_y = 1080 / 2  # 540

pixel_x = world_x * pixels_per_meter + camera_center_x
pixel_y = world_y * pixels_per_meter + camera_center_y
```

---

## Code Examples

### Example 1: Complete Ball Detection Pipeline

```python
"""
Complete example: Detect ball in camera frame → Store in physics state → Return via API
"""
from backend.vision.detection.yolo_detector import YOLODetector
from backend.vision.calibration.geometry import GeometricCalibrator
from backend.core.models import BallState, Vector2D
from backend.api.models.converters import ball_state_to_ball_info

# 1. Detect ball in camera frame
detector = YOLODetector(model_path="models/yolo11n.pt")
frame = cv2.imread("test_frame.jpg")  # 1920×1080 image
detected_balls = detector.detect_balls_with_classification(frame)

# 2. Load calibration for conversion
calibrator = GeometricCalibrator()
calibration = calibrator.load_calibration()
mapping = calibration.coordinate_mapping

# 3. Convert first ball to world meters
vision_ball = detected_balls[0]
print(f"Vision ball position (camera native): {vision_ball.position}")
# Output: (960.0, 540.0) pixels

world_pos = calibrator.pixel_to_world_coordinates(
    vision_ball.position,
    mapping
)
print(f"World position: {world_pos}")
# Output: (0.0, 0.0) meters (table center)

# 4. Create core ball state (in world meters)
ball_state = BallState(
    id="ball_0",
    position=Vector2D(world_pos[0], world_pos[1]),
    velocity=Vector2D(0.0, 0.0),
    radius=vision_ball.radius / mapping.scale_factor,  # Convert radius
    number=vision_ball.number,
    is_cue_ball=(vision_ball.ball_type == BallType.CUE),
    confidence=vision_ball.confidence
)

# 5. Convert to API response format (stays in world meters)
ball_info = ball_state_to_ball_info(ball_state)
print(f"API position: {ball_info.position}")
# Output: [0.0, 0.0] (world meters preserved)

# 6. Serialize for API response
import json
response = {
    "balls": [ball_info.dict()],
    "coordinate_space": "world_meters",
    "table_dimensions": {
        "width": 2.54,
        "height": 1.27,
        "units": "meters"
    }
}
print(json.dumps(response, indent=2))
```

### Example 2: Rendering Trajectory in Frontend

```typescript
/**
 * Complete example: Receive trajectory from API → Render on canvas
 */
import type { TrajectoryInfo } from '@/api/models';
import { createCoordinateTransform } from '@/utils/coordinates';

interface RenderTrajectoryProps {
  trajectory: TrajectoryInfo;   // Points in world meters
  canvasRef: React.RefObject<HTMLCanvasElement>;
  calibration: {
    pixelsPerMeter: number;
    cameraResolution: { width: number; height: number };
  };
}

function renderTrajectory({
  trajectory,
  canvasRef,
  calibration
}: RenderTrajectoryProps) {
  const canvas = canvasRef.current;
  if (!canvas) return;

  const ctx = canvas.getContext('2d');
  if (!ctx) return;

  // Get canvas size
  const canvasSize = {
    width: canvas.width,
    height: canvas.height
  };

  // Create coordinate transformer
  const videoSize = calibration.cameraResolution;
  const transform = createCoordinateTransform(videoSize, canvasSize);

  // Convert trajectory points from world meters to canvas pixels
  const canvasPoints = trajectory.points.map(([worldX, worldY]) => {
    // Step 1: World meters → Camera native pixels
    const cameraX = worldX * calibration.pixelsPerMeter + (videoSize.width / 2);
    const cameraY = worldY * calibration.pixelsPerMeter + (videoSize.height / 2);

    // Step 2: Camera native → Canvas pixels
    return transform.videoToCanvas({ x: cameraX, y: cameraY });
  });

  // Draw trajectory path
  ctx.beginPath();
  ctx.strokeStyle = trajectory.will_be_pocketed ? '#00ff00' : '#ffaa00';
  ctx.lineWidth = 2;

  canvasPoints.forEach((point, i) => {
    if (i === 0) {
      ctx.moveTo(point.x, point.y);
    } else {
      ctx.lineTo(point.x, point.y);
    }
  });

  ctx.stroke();

  // Draw prediction markers
  canvasPoints.forEach((point, i) => {
    if (i % 5 === 0) {  // Every 5th point
      ctx.beginPath();
      ctx.arc(point.x, point.y, 3, 0, Math.PI * 2);
      ctx.fillStyle = '#ffffff';
      ctx.fill();
    }
  });

  // Draw end point (pocket or rest position)
  if (canvasPoints.length > 0) {
    const endPoint = canvasPoints[canvasPoints.length - 1];
    ctx.beginPath();
    ctx.arc(endPoint.x, endPoint.y, 6, 0, Math.PI * 2);
    ctx.fillStyle = trajectory.will_be_pocketed ? '#00ff00' : '#ff0000';
    ctx.fill();
  }
}
```

### Example 3: Interactive Cue Positioning

```typescript
/**
 * Example: Handle mouse interaction for cue positioning
 */
interface CueInteractionProps {
  canvasRef: React.RefObject<HTMLCanvasElement>;
  onCueUpdate: (cuePosition: { tip: [number, number], angle: number }) => void;
  calibration: CalibrationData;
}

function handleCueInteraction({
  canvasRef,
  onCueUpdate,
  calibration
}: CueInteractionProps) {
  const canvas = canvasRef.current;
  if (!canvas) return;

  const videoSize = { width: 1920, height: 1080 };
  const canvasSize = {
    width: canvas.width,
    height: canvas.height
  };

  const transform = createCoordinateTransform(videoSize, canvasSize);

  canvas.addEventListener('mousemove', (event) => {
    // Get mouse position in canvas coordinates
    const rect = canvas.getBoundingClientRect();
    const canvasX = event.clientX - rect.left;
    const canvasY = event.clientY - rect.top;

    // Convert: Canvas → Camera Native → World Meters

    // Step 1: Canvas → Camera Native
    const cameraPos = transform.canvasToVideo({
      x: canvasX,
      y: canvasY
    });

    // Step 2: Camera Native → World Meters
    const worldX = (cameraPos.x - (videoSize.width / 2)) / calibration.pixelsPerMeter;
    const worldY = (cameraPos.y - (videoSize.height / 2)) / calibration.pixelsPerMeter;

    // Calculate angle (assuming cue butt at origin for simplicity)
    const angle = Math.atan2(worldY, worldX);

    // Update cue state (in world meters)
    onCueUpdate({
      tip: [worldX, worldY],
      angle: angle * (180 / Math.PI)  // Convert to degrees
    });
  });
}
```

### Example 4: Calibration Workflow

```python
"""
Example: Complete calibration workflow to establish coordinate mappings
"""
import cv2
from backend.vision.calibration.geometry import GeometricCalibrator

# 1. Capture calibration frame
frame = cv2.imread("calibration_frame.jpg")  # 1920×1080
print(f"Frame size: {frame.shape[1]}×{frame.shape[0]}")

# 2. Initialize calibrator
calibrator = GeometricCalibrator()

# 3. Detect table corners (or use manual corners)
corners = calibrator.detect_table_corners(frame)
print(f"Detected corners (camera native pixels):")
for i, corner in enumerate(corners):
    print(f"  Corner {i}: {corner}")
# Output:
#   Corner 0: (37.0, 45.0)
#   Corner 1: (606.0, 39.0)
#   Corner 2: (609.0, 403.0)
#   Corner 3: (34.0, 408.0)

# 4. Define real-world table dimensions
table_dimensions = (2.54, 1.27)  # 9ft table in meters

# 5. Perform geometric calibration
calibration = calibrator.calibrate_table_geometry(
    frame,
    table_corners=corners,
    table_dimensions=table_dimensions
)

# 6. Examine calibration results
mapping = calibration.coordinate_mapping
print(f"\nCalibration Results:")
print(f"  Pixels per meter: {mapping.scale_factor:.2f}")
print(f"  Translation: {mapping.translation}")
print(f"  Pixel bounds: {mapping.pixel_bounds}")
print(f"  World bounds: {mapping.world_bounds}")
print(f"  Calibration error: {calibration.calibration_error:.2f} pixels")

# 7. Test conversion
test_pixel = (320.0, 224.0)  # Some point in frame
test_world = calibrator.pixel_to_world_coordinates(test_pixel, mapping)
print(f"\nTest Conversion:")
print(f"  Pixel: {test_pixel}")
print(f"  World: {test_world}")

# 8. Verify round-trip conversion
back_to_pixel = calibrator.world_to_pixel_coordinates(test_world, mapping)
print(f"  Back to pixel: {back_to_pixel}")
print(f"  Round-trip error: {abs(test_pixel[0] - back_to_pixel[0]):.2f}px")

# 9. Save calibration for future use
calibrator.export_calibration("calibration_data.json")
print("\nCalibration saved!")
```

### Example 5: Multi-Resolution Detection

```python
"""
Example: Handle detections from different model resolutions
"""
from backend.vision.detection.yolo_detector import YOLODetector
import cv2

# Load frame at camera native resolution
frame = cv2.imread("frame.jpg")  # 1920×1080

# YOLO model uses 640×640 input
detector = YOLODetector(model_path="yolo11n.pt")

# Detector internally resizes to 640×640, detects, then scales back
detections = detector.detect_balls_with_classification(frame)

# Examine detection coordinates
for i, ball in enumerate(detections):
    print(f"Ball {i}:")
    print(f"  Position: {ball.position}")  # In camera native (1920×1080)
    print(f"  Radius: {ball.radius:.1f}px")  # In camera native scale

# The detector handles the following internally:
#
# 1. Resize: 1920×1080 → 640×640
# 2. Detect: YOLO outputs coordinates in 640×640 space
# 3. Scale back: 640×640 → 1920×1080
#
# So returned coordinates are always in camera native!

# When converting to world meters:
calibrator = GeometricCalibrator()
calibration = calibrator.load_calibration()
mapping = calibration.coordinate_mapping

for ball in detections:
    # No need to worry about YOLO's internal resolution
    # Just convert camera native → world meters
    world_pos = calibrator.pixel_to_world_coordinates(
        ball.position,
        mapping
    )
    print(f"World position: ({world_pos[0]:.3f}m, {world_pos[1]:.3f}m)")
```

---

## Migration Guide

### For Existing Code

If you have code that uses coordinates without explicit metadata, here's how to migrate:

#### Step 1: Identify Coordinate System

Determine which coordinate system your values are in:

```python
# Old code (ambiguous)
position = (100.0, 50.0)

# Questions to ask:
# - Where did this value come from?
# - What units are these?
# - What resolution?

# Answer determines coordinate system:
# - From camera/detector → Camera Native
# - From physics/state → World Meters
# - From normalized config → Table Normalized
# - From canvas → Canvas Pixels
```

#### Step 2: Add Metadata

Once you know the system, add appropriate documentation:

```python
# Old code
position = (100.0, 50.0)

# New code with metadata (comment form)
position = (100.0, 50.0)  # Camera Native (1920×1080)

# Or better: use tuples with context
camera_native_position = (100.0, 50.0)  # Pixels in 1920×1080

# Best: use explicit conversions
from backend.vision.calibration.geometry import GeometricCalibrator
calibrator = GeometricCalibrator()
calibration = calibrator.load_calibration()
mapping = calibration.coordinate_mapping

# Convert to canonical world meters
world_position = calibrator.pixel_to_world_coordinates(
    camera_native_position,
    mapping
)
```

#### Step 3: Update Function Signatures

Make coordinate systems explicit in function signatures:

```python
# Old (ambiguous)
def calculate_distance(pos1, pos2):
    return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

# New (explicit)
def calculate_distance_world_meters(
    pos1: tuple[float, float],  # World meters
    pos2: tuple[float, float]   # World meters
) -> float:
    """
    Calculate distance between two positions in world meters.

    Args:
        pos1: Position in world meters (x, y)
        pos2: Position in world meters (x, y)

    Returns:
        Distance in meters
    """
    return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
```

#### Step 4: Add Validation

Add assertions to catch coordinate system mismatches:

```python
def process_ball_position(
    position: tuple[float, float],
    coordinate_space: str
):
    """Process ball position in specified coordinate space."""
    assert coordinate_space in ["camera_native", "world_meters", "table_normalized"]

    if coordinate_space == "world_meters":
        # Validate range for world meters
        assert -2.0 <= position[0] <= 2.0, f"X out of range for world meters: {position[0]}"
        assert -2.0 <= position[1] <= 2.0, f"Y out of range for world meters: {position[1]}"
    elif coordinate_space == "camera_native":
        # Validate range for camera native
        assert 0 <= position[0] <= 1920, f"X out of range for camera native: {position[0]}"
        assert 0 <= position[1] <= 1080, f"Y out of range for camera native: {position[1]}"

    # Process...
```

#### Step 5: Update Tests

Add tests that verify coordinate conversions:

```python
def test_camera_to_world_conversion():
    """Test conversion from camera native to world meters."""
    calibrator = GeometricCalibrator()

    # Mock calibration
    from backend.vision.calibration.geometry import CoordinateMapping
    mapping = CoordinateMapping(
        pixel_bounds=((0, 0), (1920, 1080)),
        world_bounds=((-1.27, -0.635), (1.27, 0.635)),
        scale_factor=754.0,  # pixels per meter
        rotation_angle=0.0,
        translation=(960.0, 540.0)  # Camera center
    )

    # Test center point
    camera_center = (960.0, 540.0)
    world_center = calibrator.pixel_to_world_coordinates(camera_center, mapping)

    assert abs(world_center[0]) < 0.01  # Close to 0
    assert abs(world_center[1]) < 0.01  # Close to 0

    # Test round-trip conversion
    test_world = (0.5, 0.25)  # 0.5m right, 0.25m down
    test_camera = calibrator.world_to_pixel_coordinates(test_world, mapping)
    back_to_world = calibrator.pixel_to_world_coordinates(test_camera, mapping)

    assert abs(back_to_world[0] - test_world[0]) < 0.001  # 1mm tolerance
    assert abs(back_to_world[1] - test_world[1]) < 0.001
```

### Migration Checklist

- [ ] Identify all coordinate values in your code
- [ ] Determine coordinate system for each value
- [ ] Add comments/documentation for coordinate systems
- [ ] Update function signatures with explicit coordinate types
- [ ] Add validation/assertions for coordinate ranges
- [ ] Convert to World Meters at module boundaries
- [ ] Write tests for coordinate conversions
- [ ] Update API documentation with coordinate metadata
- [ ] Add logging for coordinate transformations (debugging)

---

## Performance Considerations

### Conversion Overhead

Each coordinate conversion has computational cost. Here's how to minimize it:

#### 1. Batch Conversions

```python
# ❌ SLOW - Convert each point individually
for point in points:
    world_point = calibrator.pixel_to_world_coordinates(point, mapping)
    process(world_point)

# ✅ FAST - Vectorized conversion
import numpy as np

points_array = np.array(points)
scale = 1.0 / mapping.scale_factor
translation = np.array(mapping.translation)

# Vectorized conversion
world_points = (points_array - translation) * scale

for world_point in world_points:
    process(world_point)
```

**Speedup**: ~10x for large batches

#### 2. Cache Conversions

```python
# ❌ SLOW - Recompute same conversion
for frame in frames:
    for ball in balls:
        canvas_pos = convert_world_to_canvas(ball.position)
        render(canvas_pos)

# ✅ FAST - Cache conversion parameters
class CachedConverter:
    def __init__(self, mapping, canvas_size):
        self.pixels_per_meter = mapping.scale_factor
        self.camera_center = mapping.translation
        self.canvas_scale = canvas_size[0] / 1920.0

    def world_to_canvas(self, world_pos):
        # Precomputed scale and offset
        camera_x = world_pos[0] * self.pixels_per_meter + self.camera_center[0]
        camera_y = world_pos[1] * self.pixels_per_meter + self.camera_center[1]
        return (camera_x * self.canvas_scale, camera_y * self.canvas_scale)

converter = CachedConverter(mapping, canvas_size)
for frame in frames:
    for ball in balls:
        canvas_pos = converter.world_to_canvas(ball.position)
        render(canvas_pos)
```

**Speedup**: ~5x by avoiding repeated parameter lookups

#### 3. Only Convert When Necessary

```python
# ❌ INEFFICIENT - Convert even when not needed
def calculate_distance(ball1, ball2, mapping):
    # Unnecessary conversion if both already in same space
    world1 = pixel_to_world(ball1.position, mapping)
    world2 = pixel_to_world(ball2.position, mapping)
    return distance(world1, world2)

# ✅ EFFICIENT - Check first
def calculate_distance(ball1, ball2, mapping):
    if ball1.coordinate_space == ball2.coordinate_space == "world_meters":
        # Already in same space, no conversion needed
        return distance(ball1.position, ball2.position)

    # Different spaces, convert to world meters
    world1 = ensure_world_meters(ball1.position, ball1.coordinate_space, mapping)
    world2 = ensure_world_meters(ball2.position, ball2.coordinate_space, mapping)
    return distance(world1, world2)
```

#### 4. Use Integer Pixels When Possible

```python
# For rendering, pixel-perfect accuracy not needed
# Use integers to avoid floating-point overhead

# ❌ SLOW - Float operations
canvas_x = int(float_x)
canvas_y = int(float_y)
ctx.draw_circle(canvas_x, canvas_y, radius)

# ✅ FAST - Integer operations throughout
canvas_x = round(float_x)  # Round once
canvas_y = round(float_y)
ctx.draw_circle(canvas_x, canvas_y, radius)
```

### Benchmarks

Expected performance for coordinate conversions:

| Operation | Time (per conversion) | Notes |
|-----------|---------------------|-------|
| Camera → World | ~200 ns | Simple arithmetic |
| World → Camera | ~200 ns | Simple arithmetic |
| Camera → Canvas | ~500 ns | Includes aspect ratio calc |
| Batch (100 points) | ~5 μs | Vectorized numpy |
| Full frame (16 balls) | ~10 μs | Negligible overhead |

**Conclusion**: Coordinate conversion overhead is negligible (<0.1% of frame processing time).

### Memory Considerations

Storing coordinate metadata has minimal impact:

```python
# Without metadata
class SimpleBall:
    position: tuple[float, float]  # 16 bytes

# With metadata
class MetadataBall:
    position: tuple[float, float]  # 16 bytes
    coordinate_space: str          # 8 bytes (pointer)
    resolution: tuple[int, int]    # 16 bytes

# Overhead: 24 bytes per ball
# For 16 balls: 384 bytes extra (~0.4KB)
# Negligible!
```

---

## Troubleshooting

### Symptoms and Solutions

#### Symptom: Balls Appear in Wrong Location

**Possible Causes**:
1. Using wrong coordinate system
2. Missing calibration
3. Incorrect aspect ratio handling

**Diagnosis**:
```python
# Print coordinate values at each stage
print(f"Camera native: {camera_pos}")
print(f"World meters: {world_pos}")
print(f"Canvas pixels: {canvas_pos}")

# Check calibration
print(f"Pixels per meter: {mapping.scale_factor}")
print(f"Camera center: {mapping.translation}")

# Verify aspect ratio
print(f"Canvas size: {canvas_size}")
print(f"Video size: {video_size}")
print(f"Aspect ratio canvas: {canvas_size[0] / canvas_size[1]}")
print(f"Aspect ratio video: {video_size[0] / video_size[1]}")
```

**Solution**:
- Ensure calibration is loaded correctly
- Check coordinate transformations are in right order
- Verify aspect ratio handling (use uniform scale)

#### Symptom: Balls Appear Too Large/Small

**Possible Cause**: Radius not converted to correct coordinate system

**Diagnosis**:
```python
print(f"Ball radius in pixels: {ball.radius}")
print(f"Pixels per meter: {mapping.scale_factor}")
print(f"Ball radius in meters: {ball.radius / mapping.scale_factor}")
```

**Solution**:
```python
# Convert radius when converting position
world_radius = pixel_radius / mapping.scale_factor
```

#### Symptom: Trajectory Doesn't Align with Balls

**Possible Cause**: Trajectory and balls in different coordinate systems

**Diagnosis**:
```python
# Check both are in same system
print(f"Ball coordinate space: {ball.metadata.coordinate_space}")
print(f"Trajectory coordinate space: {trajectory.metadata.coordinate_space}")
```

**Solution**:
```python
# Convert both to world meters before rendering
world_ball = ensure_world_meters(ball.position)
world_trajectory = [ensure_world_meters(p) for p in trajectory.points]

# Then convert both to canvas together
canvas_ball = world_to_canvas(world_ball)
canvas_trajectory = [world_to_canvas(p) for p in world_trajectory]
```

#### Symptom: Calibration Fails or Produces Invalid Results

**Possible Causes**:
1. Table not visible in frame
2. Poor lighting conditions
3. Occlusions

**Diagnosis**:
```python
# Check corner detection
corners = calibrator.detect_table_corners(frame)
print(f"Detected corners: {corners}")

# Visualize corners
debug_frame = frame.copy()
for corner in corners:
    cv2.circle(debug_frame, tuple(map(int, corner)), 10, (0, 255, 0), -1)
cv2.imwrite("debug_corners.jpg", debug_frame)
```

**Solution**:
- Ensure table is fully visible
- Improve lighting
- Use manual corner selection if auto-detection fails:
  ```python
  manual_corners = [
      (37, 45),    # Top-left
      (606, 39),   # Top-right
      (609, 403),  # Bottom-right
      (34, 408)    # Bottom-left
  ]
  calibration = calibrator.calibrate_table_geometry(
      frame,
      table_corners=manual_corners
  )
  ```

### Debug Logging

Enable coordinate transformation logging:

```python
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("coordinates")

def convert_with_logging(pos, source, target, mapping):
    """Convert coordinates with debug logging."""
    logger.debug(f"Converting {pos} from {source} to {target}")

    if source == "camera_native" and target == "world_meters":
        result = pixel_to_world_coordinates(pos, mapping)
        logger.debug(f"  Scale factor: {mapping.scale_factor}")
        logger.debug(f"  Translation: {mapping.translation}")
        logger.debug(f"  Result: {result}")
        return result

    # ... other conversions
```

### Validation Helpers

```python
def validate_coordinate_conversion(
    original, converted, conversion_type, tolerance=0.01
):
    """Validate coordinate conversion accuracy."""
    # Convert back
    if conversion_type == "camera_to_world":
        back = world_to_pixel_coordinates(converted, mapping)
    elif conversion_type == "world_to_camera":
        back = pixel_to_world_coordinates(converted, mapping)

    # Check round-trip error
    error_x = abs(original[0] - back[0])
    error_y = abs(original[1] - back[1])

    if error_x > tolerance or error_y > tolerance:
        raise ValueError(
            f"Round-trip conversion error too large:\n"
            f"  Original: {original}\n"
            f"  Converted: {converted}\n"
            f"  Back: {back}\n"
            f"  Error: ({error_x:.4f}, {error_y:.4f})\n"
            f"  Tolerance: {tolerance}"
        )

    return True
```

---

## Summary

### Key Takeaways

1. **Four Coordinate Systems**: Camera Native, World Meters, Table Normalized, Canvas Pixels
2. **Canonical Format**: Always store persistent data in World Meters
3. **Explicit Conversion**: Never mix coordinate systems without explicit conversion
4. **Include Metadata**: Document coordinate system and resolution for all spatial data
5. **Validate Early**: Add assertions to catch coordinate system mismatches
6. **Performance**: Coordinate conversion overhead is negligible (<0.1% of frame time)

### Quick Reference Card

```
COORDINATE SYSTEM QUICK REFERENCE

Camera Native (1920×1080 px)
├─ Source: Vision detection
├─ Origin: Top-left (0, 0)
└─ Convert to World: pos / pixels_per_meter - camera_center

World Meters (2.54m × 1.27m)
├─ Source: Physics, persistent storage
├─ Origin: Table center (0, 0)
└─ Convert to Camera: pos * pixels_per_meter + camera_center

Table Normalized [0, 1]
├─ Source: Relative positioning
├─ Origin: Top-left (0, 0)
└─ Convert to World: (pos - 0.5) * table_dimensions

Canvas Pixels (variable)
├─ Source: Frontend rendering
├─ Origin: Top-left (0, 0)
└─ Convert from Camera: apply aspect-preserving scale + offset

CONVERSIONS ALWAYS GO THROUGH WORLD METERS:
Source → World Meters → Target
```

### Further Reading

- Vision Calibration: `/backend/vision/calibration/geometry.py`
- Frontend Transforms: `/frontend/web/src/utils/coordinates.ts`
- API Converters: `/backend/api/models/converters.py`
- Resolution Plan: `/thoughts/resolution_standardization_plan.md`

---

**Document Version**: 1.0
**Last Updated**: 2025-10-21
**Maintained by**: Development Team
