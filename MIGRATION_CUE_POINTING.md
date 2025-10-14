# Migration: `_find_ball_cue_is_pointing_at` to Backend

## Overview
Moved the `_find_ball_cue_is_pointing_at` method from `tools/video_debugger.py` to the backend as a reusable utility function.

## Changes Made

### 1. New Backend Function
**Location:** `/Users/jchadwick/code/billiards-trainer/backend/core/utils/geometry.py`

**Function Name:** `find_ball_cue_is_pointing_at()`

**Key Improvements:**
- More flexible API accepting multiple input types:
  - Tuples `(x, y)` or `Vector2D` objects for positions
  - Either `cue_angle` (degrees) or `cue_direction` vector
- Comprehensive docstring with examples
- Type hints for better IDE support
- Returns ball **index** instead of ball object (more flexible)
- Configurable `max_perpendicular_distance` parameter

### 2. Module Export
**Location:** `/Users/jchadwick/code/billiards-trainer/backend/core/utils/__init__.py`

Added export:
```python
from .geometry import GeometryUtils, find_ball_cue_is_pointing_at
```

## Usage Examples

### Basic Usage (with angle)
```python
from backend.core.utils import find_ball_cue_is_pointing_at

cue_tip = (100, 200)
cue_angle = 45.0  # degrees
balls = [(150, 250), (200, 300), (300, 100)]

target_idx = find_ball_cue_is_pointing_at(
    cue_tip=cue_tip,
    cue_angle=cue_angle,
    balls=balls,
    max_perpendicular_distance=40.0
)

if target_idx is not None:
    print(f"Aiming at ball at {balls[target_idx]}")
```

### With Backend Models (CueState and BallState)
```python
from backend.core.models import Vector2D, CueState, BallState
from backend.core.utils import find_ball_cue_is_pointing_at

# Create cue state
cue = CueState(
    tip_position=Vector2D(100, 200),
    angle=45.0,
    elevation=0.0,
    estimated_force=5.0,
)

# Create ball states
balls = [
    BallState(id="ball_1", position=Vector2D(150, 250)),
    BallState(id="ball_2", position=Vector2D(200, 300)),
]

# Find target
ball_positions = [ball.position for ball in balls]
target_idx = find_ball_cue_is_pointing_at(
    cue_tip=cue.tip_position,
    cue_angle=cue.angle,
    balls=ball_positions
)

if target_idx is not None:
    target_ball = balls[target_idx]
    print(f"Aiming at {target_ball.id}")
```

### With Direction Vector
```python
from backend.core.models import Vector2D
from backend.core.utils import find_ball_cue_is_pointing_at

cue_tip = Vector2D(100, 200)
cue_direction = Vector2D(1, 1).normalize()  # pointing up-right
balls = [Vector2D(150, 250), Vector2D(200, 300)]

target_idx = find_ball_cue_is_pointing_at(
    cue_tip=cue_tip,
    cue_direction=cue_direction,
    balls=balls
)
```

## Function Signature

```python
def find_ball_cue_is_pointing_at(
    cue_tip: tuple[float, float] | Vector2D,
    cue_direction: tuple[float, float] | Vector2D | None = None,
    cue_angle: float | None = None,
    balls: list[tuple[float, float] | Vector2D] = None,
    max_perpendicular_distance: float = 40.0,
) -> int | None
```

## Algorithm Details

The function uses vector mathematics to determine which ball the cue is pointing at:

1. **Direction Calculation:** Converts angle to direction vector or normalizes provided direction
2. **Projection Check:** Projects each ball position onto the cue direction (dot product)
3. **Behind Filter:** Skips balls with negative projection (behind cue tip)
4. **Perpendicular Distance:** Calculates perpendicular distance using cross product
5. **Threshold Check:** Only considers balls within `max_perpendicular_distance`
6. **Closest Selection:** Returns the closest ball along the cue direction

## Migration Notes

### Original Implementation (video_debugger.py)
- Location: `tools/video_debugger.py` lines 600-659
- Method: `VideoDebugger._find_ball_cue_is_pointing_at()`
- Returned: Ball object or None
- Hardcoded: `max_perpendicular_distance = 40`

### New Implementation (backend)
- Location: `backend/core/utils/geometry.py`
- Function: `find_ball_cue_is_pointing_at()`
- Returns: Ball index (int) or None
- Configurable: `max_perpendicular_distance` parameter
- More flexible input types

## Testing

All tests pass successfully:
- ✓ Basic tuple input
- ✓ Vector2D input
- ✓ Direction vector input
- ✓ No ball in range
- ✓ Multiple balls in line (picks closest)
- ✓ Perpendicular distance threshold
- ✓ Ball within tolerance

See `/Users/jchadwick/code/billiards-trainer/backend/core/utils/example_cue_pointing.py` for comprehensive examples.

## Next Steps

The original `_find_ball_cue_is_pointing_at` method in `video_debugger.py` can now be updated to use this backend function:

```python
def _find_ball_cue_is_pointing_at(self, cue, balls):
    """Find which ball the cue is currently pointing at."""
    from backend.core.utils import find_ball_cue_is_pointing_at

    ball_positions = [ball.position for ball in balls]
    target_idx = find_ball_cue_is_pointing_at(
        cue_tip=cue.tip_position,
        cue_angle=cue.angle,
        balls=ball_positions,
        max_perpendicular_distance=40.0
    )
    return balls[target_idx] if target_idx is not None else None
```

## Benefits

1. **Reusability:** Available to all backend modules
2. **Consistency:** Single implementation across the codebase
3. **Type Safety:** Works with both tuples and Vector2D
4. **Flexibility:** Multiple input options (angle or direction)
5. **Documentation:** Comprehensive docstrings and examples
6. **Testability:** Standalone function easier to test
7. **Configuration:** Adjustable perpendicular distance threshold
