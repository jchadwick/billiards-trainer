# Group 4: Physics Engine - Status Report

**Agent**: Agent 4
**Responsibility**: Update Physics Engine to use 4K pixel coordinates
**Status**: ⏸️ **WAITING FOR GROUP 3**

---

## Current Status

### Blockers
- ✋ **WAITING**: Group 3 (Core Models) has not completed yet
- ❌ `backend/core/constants_4k.py` does not exist
- ❌ `backend/core/models.py` still uses old `CoordinateSpace.WORLD_METERS` system
- ❌ `Vector2D` does not have mandatory `scale` metadata yet

### Evidence of Incomplete Dependencies

**models.py (lines 1-100)**:
```python
# Still using old coordinate system:
from backend.core.coordinates import (
    Vector2D,
    CoordinateSpace,  # ← OLD SYSTEM
    Resolution,
)

# Comments still reference old WORLD_METERS system:
# "Ball positions are canonically stored in WORLD_METERS coordinate space"
```

**Missing Files**:
- `backend/core/constants_4k.py` - Required for 4K constants
- No evidence of mandatory `scale` in Vector2D yet

---

## Dependencies (from Migration Plan)

### Group 4 Dependencies:
1. **Group 1** (Foundation) → Must complete first
   - Create `constants_4k.py`
   - Implement `ResolutionConverter`
   - Create `validation_4k.py`

2. **Group 2** (Vector2D) → Must complete second
   - Update `Vector2D` to make `scale` mandatory
   - Add factory methods: `from_4k()`, `from_resolution()`
   - Implement `to_4k_canonical()` method

3. **Group 3** (Core Models) → Must complete third ⚠️ **CURRENTLY BLOCKING**
   - Update `BallState` to use 4K pixels (not meters)
   - Update `TableState` dimensions to 4K pixels
   - Update `CueState` to 4K pixels
   - Update `GameState` to remove CoordinateMetadata

---

## My Responsibilities (Once Unblocked)

### Files to Update:
1. `/Users/jchadwick/code/billiards-trainer/backend/core/physics/trajectory.py`
2. `/Users/jchadwick/code/billiards-trainer/backend/core/physics/collision.py`
3. `/Users/jchadwick/code/billiards-trainer/backend/core/physics/spin.py`
4. `/Users/jchadwick/code/billiards-trainer/backend/core/physics/engine.py`

### Key Changes Required:

#### 1. Update Physics Constants (engine.py)

**Current (Meters)**:
```python
class PhysicsConstants:
    # Ball properties - all in SI units (meters)
    BALL_RADIUS = 0.028575  # m (57.15mm diameter / 2)
    BALL_MASS = 0.17  # kg (standard pool ball)

    # Table properties
    TABLE_FRICTION_COEFFICIENT = 0.2  # Rolling friction
    CUSHION_RESTITUTION = 0.85
    BALL_RESTITUTION = 0.95

    # Physics simulation
    GRAVITY = 9.81  # m/s^2  ← NEEDS CONVERSION
    TIME_STEP = 0.001  # s
    MIN_VELOCITY = 0.001  # m/s  ← NEEDS CONVERSION
```

**New (4K Pixels)**:
```python
from backend.core.constants_4k import (
    BALL_RADIUS_4K,
    TABLE_WIDTH_4K,
    TABLE_HEIGHT_4K,
    PIXELS_PER_METER,  # For internal physics conversion
)

class PhysicsConstants:
    # Ball properties in 4K pixels
    BALL_RADIUS = BALL_RADIUS_4K  # 36.0 pixels
    BALL_MASS = 0.17  # kg (not spatial, stays same)

    # Unit-less coefficients (unchanged)
    TABLE_FRICTION_COEFFICIENT = 0.2
    CUSHION_RESTITUTION = 0.85
    BALL_RESTITUTION = 0.95

    # Convert accelerations to pixel scale
    PIXELS_PER_METER = TABLE_WIDTH_4K / 2.54  # ~1260 pixels/meter
    GRAVITY_PIXELS = 9.81 * PIXELS_PER_METER  # pixels/s²
    MIN_VELOCITY = 0.001 * PIXELS_PER_METER  # pixels/s
```

#### 2. Update Trajectory Calculations (trajectory.py)

**Current TrajectoryPoint**:
```python
@dataclass
class TrajectoryPoint:
    time: float
    position: Vector2D  # Has no scale
    velocity: Vector2D  # Has no scale
    # ...
```

**New TrajectoryPoint**:
```python
@dataclass
class TrajectoryPoint:
    time: float
    position: Vector2D  # MUST have scale=[1.0, 1.0] for 4K
    velocity: Vector2D  # MUST have scale=[1.0, 1.0] for 4K
    acceleration: Vector2D  # MUST have scale
    spin: Vector2D  # MUST have scale
    energy: float

    @classmethod
    def from_4k(cls, x, y, vx, vy, time) -> "TrajectoryPoint":
        return cls(
            position=Vector2D.from_4k(x, y),
            velocity=Vector2D.from_4k(vx, vy),
            # ...
        )
```

#### 3. Update Collision Detection (collision.py)

All collision math must work in 4K pixels:
- Ball-ball collision distances in pixels (not meters)
- Ball-cushion collision using `BALL_RADIUS_4K` (not meter-based)
- Use pixel-based velocities for collision response

**Current**:
```python
combined_radius = ball1.radius + ball2.radius  # meters
```

**New**:
```python
combined_radius = ball1.radius + ball2.radius  # pixels (4K)
```

#### 4. Update Spin Physics (spin.py)

**Current SpinPhysics**:
```python
class SpinPhysics:
    def __init__(self, config: Optional[dict[str, Any]] = None):
        # Physical constants
        self.ball_radius = 0.028575  # Standard pool ball radius (m)
        self.ball_mass = 0.17  # Standard pool ball mass (kg)
```

**New SpinPhysics**:
```python
from backend.core.constants_4k import BALL_RADIUS_4K, PIXELS_PER_METER

class SpinPhysics:
    def __init__(self, config: Optional[dict[str, Any]] = None):
        # Physical constants in pixels
        self.ball_radius = BALL_RADIUS_4K  # 36 pixels
        self.ball_mass = 0.17  # kg (not spatial)

        # Convert for internal physics calculations if needed
        self.pixels_per_meter = PIXELS_PER_METER
```

---

## Critical: Maintain Physics Accuracy

The migration **MUST NOT** change physics behavior:

### Verification Strategy:
1. **Test Setup**: Create test scenarios in both meter and pixel space
2. **Compare Results**: Physics calculations must match (±0.1% tolerance)
3. **Energy Conservation**: Verify total energy unchanged
4. **Collision Response**: Ball velocities after collision must match

### Example Accuracy Test:
```python
def test_physics_accuracy_unchanged():
    # OLD: Ball at (0.5m, 0m) moving at (1.5 m/s, 0)
    ball_meters = BallState_old.create("ball1", x=0.5, y=0.0)
    ball_meters.velocity = Vector2D(1.5, 0.0)

    # NEW: Same ball in 4K pixels
    # 0.5m from table center = 0.5 * 1260px = 630px offset
    # Table center is at (1920, 1080) in 4K
    ball_pixels = BallState.from_4k("ball1", x=1920+630, y=1080)
    ball_pixels.velocity = Vector2D.from_4k(1.5 * PIXELS_PER_METER, 0)

    # Calculate trajectories
    traj_old = old_engine.calculate_trajectory(ball_meters, ...)
    traj_new = new_engine.calculate_trajectory(ball_pixels, ...)

    # Compare final positions (convert to same units)
    final_old_pos = traj_old.final_position  # meters
    final_new_pos = traj_new.final_position.to_meters()  # convert from pixels

    # Must match within 0.1%
    assert abs(final_new_pos.x - final_old_pos.x) / final_old_pos.x < 0.001
    assert abs(final_new_pos.y - final_old_pos.y) / final_old_pos.y < 0.001
```

---

## Success Criteria

Once unblocked, I must achieve:

- ✅ All physics calculations in 4K pixels (not meters)
- ✅ No meter-based constants remain in physics code
- ✅ Accuracy matches old system (±0.1% tolerance)
- ✅ All `TrajectoryPoint` instances have `scale` metadata
- ✅ All physics tests passing
- ✅ Performance unchanged (±5%)

---

## Next Steps

1. ⏳ **Wait for Group 3** to complete Core Models migration
2. ⏳ Watch for creation of:
   - `backend/core/constants_4k.py`
   - Updated `Vector2D` with mandatory `scale`
   - Updated `BallState` using 4K pixels
3. ✅ Once dependencies complete, begin physics migration
4. ✅ Run accuracy tests to verify physics behavior unchanged

---

**Status**: ⏸️ **BLOCKED** - Waiting for Group 3 (Core Models)
**Estimated Start**: After Group 3 completion
**Estimated Duration**: 5 days (per migration plan)
**Complexity**: Very High (physics accuracy critical)

---

**Last Updated**: 2025-10-21
**Agent**: Agent 4
