# Coordinate Transformation Flow Diagram

## Current (BROKEN) Flow

```
┌─────────────────────────────────────────┐
│  Vision Module Detects Ball             │
│  Position: (2974, 1256) pixels          │
│  In: 1920×1080 camera frame             │
│  (Ball may be in 4K frame actually!)    │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│  integration_service_conversion_helpers │
│  INCORRECT: Simply divides by 756       │
│  position_m = pixels / 756.0             │
│                                          │
│  Result: (3.934m, 1.661m)               │
│  ❌ WRONG! Outside 2.54×1.27 table!     │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│  Core Module GameState                  │
│  Stores as "world_meters"               │
│  BallState.position = (3.934m, 1.661m)  │
│  ❌ Invalid position!                   │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│  WebSocket Broadcaster                  │
│  Broadcasts with metadata:              │
│  coordinate_space="world_meters"        │
│  Sends: {x: 3.934, y: 1.661}           │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│  Frontend Receives                      │
│  Tries to render ball at (3.934, 1.661) │
│  Table is only 2.54 × 1.27              │
│  ❌ Ball appears way off screen!        │
└─────────────────────────────────────────┘
```

## Correct Flow (WHAT IT SHOULD BE)

```
┌─────────────────────────────────────────┐
│  Vision Module Detects Ball             │
│  Position: (2974, 1256) pixels          │
│  In: 1920×1080 camera frame             │
│  Playing area: (37,45) to (606,326)     │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│  CoordinateConverter                    │
│  Step 1: Apply perspective transform    │
│    Using playing area corners           │
│    Camera (2974,1256)                   │
│    → Table-relative (450, 180) pixels   │
│                                          │
│  Step 2: Convert to meters               │
│    Using correct pixels_per_meter=224   │
│    (450/224, 180/224)                   │
│    → World meters (2.01m, 0.80m)        │
│                                          │
│  ✓ Valid! Within 2.54×1.27 table        │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│  Core Module GameState                  │
│  Stores as "world_meters"               │
│  BallState.position = (2.01m, 0.80m)    │
│  ✓ Valid position!                      │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│  WebSocket Broadcaster                  │
│  Broadcasts with metadata:              │
│  coordinate_space="world_meters"        │
│  Sends: {x: 2.01, y: 0.80}             │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│  Frontend Receives                      │
│  Renders ball at (2.01, 0.80)          │
│  Table is 2.54 × 1.27                   │
│  ✓ Ball appears on table correctly!     │
└─────────────────────────────────────────┘
```

## The Two Critical Issues

### Issue 1: Wrong pixels_per_meter

```
Current config: pixels_per_meter = 756.0
  Calculated from: 1920 pixels / 2.54 meters (full camera width)

  Problem: The table doesn't occupy the full camera width!

Correct config: pixels_per_meter = 224.0
  Should be from: 569 pixels / 2.54 meters (table region width)
  Or: 287 pixels / 1.27 meters (table region height) = 226

  Average: ~224 pixels/meter for the TABLE region
```

### Issue 2: Missing Perspective Transform

The playing area corners define a trapezoid due to camera angle:

```
Camera view (1920×1080):
┌──────────────────────────┐
│                          │
│  (37,45)────(606,39)     │  ← Table top edge: 569px
│    │            │         │
│    │   TABLE    │         │
│    │            │         │
│  (40,322)───(604,326)    │  ← Table bottom: 564px
│                          │
└──────────────────────────┘

Without perspective transform:
- Top and bottom have different widths (569 vs 564 pixels)
- Left and right edges are not parallel
- Direct pixel→meter conversion produces distorted positions

With perspective transform:
- Warps trapezoid to rectangle
- Parallel lines remain parallel
- Accurate metric positions
```

## Config Fix Required

Current config (WRONG):
```json
{
  "vision": {
    "calibration": {
      // pixels_per_meter not set, defaults to 756.0
    }
  }
}
```

Fixed config:
```json
{
  "vision": {
    "calibration": {
      "pixels_per_meter": 224.0  // Correct for table region
    }
  }
}
```

But this alone won't fix it - need to also implement perspective transform!

## Code Fix Required

Location: `backend/integration_service_conversion_helpers.py:125-136`

Current code (WRONG):
```python
position_m = Vector2D(
    self._pixels_to_meters(ball.position[0]),
    self._pixels_to_meters(ball.position[1])
)
```

Fixed code:
```python
# Use CoordinateConverter with perspective transform
from backend.core.coordinate_converter import CoordinateConverter, Resolution

position_m = self.coordinate_converter.camera_pixels_to_world_meters(
    Vector2D(ball.position[0], ball.position[1]),
    camera_resolution=Resolution(frame_width, frame_height)
)
```

This will:
1. Apply perspective transform using playing area corners
2. Convert to meters using correct pixels_per_meter ratio
3. Return valid world meter coordinates
