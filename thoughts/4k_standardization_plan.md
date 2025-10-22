# 4K Resolution Standardization Migration Plan

**Version**: 1.0
**Date**: 2025-10-21
**Status**: Planning
**Complexity**: High - Full System Migration

---

## Executive Summary

This document provides a comprehensive plan to migrate the billiards-trainer backend from the current multi-coordinate-space system (WORLD_METERS, CAMERA_PIXELS, TABLE_PIXELS, NORMALIZED) to a new **4K pixel-based standardized system**.

### Migration Goals

1. **Eliminate all real-world measurements** (meters, centimeters) - use only pixels
2. **Standardize to 4K resolution** (3840×2160) as canonical storage format
3. **Mandatory scale metadata** on ALL Vector2D instances: `{x, y, scale: [scale_x, scale_y]}`
4. **Maintain system functionality** throughout migration with zero downtime

### Critical Success Factors

- ✅ No data loss during migration
- ✅ All physics calculations remain accurate
- ✅ Backward compatibility during transition
- ✅ Full test coverage for coordinate conversions
- ✅ Clear migration path for each module

---

## Table of Contents

1. [Current State Analysis](#1-current-state-analysis)
2. [New Coordinate System Design](#2-new-coordinate-system-design)
3. [Table Dimensions in Pixels](#3-table-dimensions-in-pixels)
4. [Migration Strategy](#4-migration-strategy)
5. [Breaking Changes](#5-breaking-changes)
6. [Implementation Groups](#6-implementation-groups)
7. [Verification Plan](#7-verification-plan)
8. [Rollback Plan](#8-rollback-plan)
9. [Timeline & Dependencies](#9-timeline--dependencies)

---

## 1. Current State Analysis

### 1.1 Existing Coordinate Spaces

The system currently uses **4 distinct coordinate spaces**:

| Space | Units | Origin | Resolution | Storage |
|-------|-------|--------|------------|---------|
| `WORLD_METERS` | Meters | Table center | N/A | Canonical |
| `CAMERA_PIXELS` | Pixels | Top-left | 1920×1080 | Transient |
| `TABLE_PIXELS` | Pixels | Top-left | Variable | Transient |
| `NORMALIZED` | Ratio [0,1] | Top-left | N/A | Config |

### 1.2 Current Architecture

```
Vision Detection → Core Physics → API → Frontend
    ↓                 ↓            ↓        ↓
CAMERA_PIXELS → WORLD_METERS → WORLD → Canvas
  (1920×1080)      (meters)    METERS   (variable)
```

**Key Components**:
- **CoordinateConverter** (`backend/core/coordinate_converter.py`)
  - Uses `pixels_per_meter` = 754.0 (default calibration)
  - Handles conversions between all 4 spaces
  - ~900 lines of code

- **Vector2D** (`backend/core/coordinates.py`)
  - Enhanced with optional coordinate metadata
  - Supports factory methods: `world_meters()`, `camera_pixels()`, etc.
  - ~850 lines of code

- **BallState** (`backend/core/models.py`)
  - Stores positions in WORLD_METERS
  - Position is `Vector2D` with coordinate metadata
  - ~1660 lines total in models.py

### 1.3 Current Usage Statistics

**Files Using Coordinates**: ~50 files across modules
- Core: 12 files (physics, collision, analysis)
- Vision: 8 files (detection, calibration)
- API: 6 files (routes, models, converters)
- Tests: 15 files
- Integration: 9 files

**Key Dependencies**:
```python
# Standard table (9ft)
STANDARD_TABLE_WIDTH_METERS = 2.54   # meters
STANDARD_TABLE_HEIGHT_METERS = 1.27  # meters

# Default calibration
DEFAULT_PIXELS_PER_METER = 754.0

# Standard camera resolution
CAMERA_NATIVE_RESOLUTION = Resolution(width=1920, height=1080)
```

### 1.4 Problems with Current System

1. **Conceptual complexity**: Developers must track 4 coordinate systems
2. **Conversion overhead**: Constant meter ↔ pixel transformations
3. **Calibration dependency**: `pixels_per_meter` must be accurate
4. **Resolution ambiguity**: Multiple pixel-based spaces cause confusion
5. **Physical meaning mismatch**: Physics uses meters, everything else pixels

---

## 2. New Coordinate System Design

### 2.1 Single Canonical Space: 4K Pixels

**All coordinates will be stored and processed in 4K resolution (3840×2160)**

| Property | Value |
|----------|-------|
| **Name** | `CANONICAL_4K` |
| **Resolution** | 3840 × 2160 pixels |
| **Units** | Pixels only |
| **Origin** | Top-left (0, 0) |
| **X-axis** | Increases right: 0 → 3840 |
| **Y-axis** | Increases down: 0 → 2160 |
| **Aspect Ratio** | 16:9 |

### 2.2 Vector2D Format with Mandatory Scale

**New Vector2D Structure**:
```python
@dataclass
class Vector2D:
    """2D vector with mandatory resolution scale metadata."""
    x: float                    # X coordinate in pixels
    y: float                    # Y coordinate in pixels
    scale: tuple[float, float]  # MANDATORY: [scale_x, scale_y]
```

**Scale Metadata Rules**:

1. **For 4K canonical data**: `scale = [1.0, 1.0]`
   ```python
   # Ball at center of 4K frame
   position = Vector2D(x=1920.0, y=1080.0, scale=[1.0, 1.0])
   ```

2. **For lower resolutions**: Scale relative to 4K
   ```python
   # 1920×1080 (1080p) → 4K
   # scale_x = 3840 / 1920 = 2.0
   # scale_y = 2160 / 1080 = 2.0
   position = Vector2D(x=960.0, y=540.0, scale=[2.0, 2.0])

   # To get 4K coordinates:
   # x_4k = x * scale[0] = 960 * 2.0 = 1920
   # y_4k = y * scale[1] = 540 * 2.0 = 1080
   ```

3. **For higher resolutions**: Scale < 1.0
   ```python
   # 7680×4320 (8K) → 4K
   # scale_x = 3840 / 7680 = 0.5
   # scale_y = 2160 / 4320 = 0.5
   position = Vector2D(x=3840.0, y=2160.0, scale=[0.5, 0.5])

   # To get 4K coordinates:
   # x_4k = 3840 * 0.5 = 1920
   # y_4k = 2160 * 0.5 = 1080
   ```

4. **Validation**: Scale must always be present and non-zero
   ```python
   def __post_init__(self):
       if self.scale is None:
           raise ValueError("Scale metadata is MANDATORY")
       if self.scale[0] <= 0 or self.scale[1] <= 0:
           raise ValueError("Scale factors must be positive")
   ```

### 2.3 Coordinate Conversion Operations

**Converting between resolutions**:

```python
def to_4k_canonical(vector: Vector2D) -> Vector2D:
    """Convert any Vector2D to 4K canonical coordinates."""
    x_4k = vector.x * vector.scale[0]
    y_4k = vector.y * vector.scale[1]
    return Vector2D(x=x_4k, y=y_4k, scale=[1.0, 1.0])

def from_4k_canonical(x_4k: float, y_4k: float, target_resolution: tuple[int, int]) -> Vector2D:
    """Convert from 4K canonical to target resolution."""
    scale_x = 3840 / target_resolution[0]
    scale_y = 2160 / target_resolution[1]
    x = x_4k / scale_x
    y = y_4k / scale_y
    return Vector2D(x=x, y=y, scale=[scale_x, scale_y])
```

**Example conversions**:

| Source Resolution | Source Coords | Scale Factor | 4K Canonical |
|------------------|---------------|--------------|--------------|
| 1920×1080 (1080p) | (960, 540) | [2.0, 2.0] | (1920, 1080) |
| 1280×720 (720p) | (640, 360) | [3.0, 3.0] | (1920, 1080) |
| 640×360 (360p) | (320, 180) | [6.0, 6.0] | (1920, 1080) |
| 3840×2160 (4K) | (1920, 1080) | [1.0, 1.0] | (1920, 1080) |

### 2.4 Removed Concepts

The following will be **completely eliminated**:

- ❌ `WORLD_METERS` coordinate space
- ❌ `pixels_per_meter` calibration factor
- ❌ Meter-based table dimensions
- ❌ Meter-based ball positions
- ❌ Meter-based velocities
- ❌ `CoordinateSpace` enum (replaced by resolution metadata)

---

## 3. Table Dimensions in Pixels

### 3.1 Physical to Pixel Mapping

**Standard 9ft Pool Table**:
- Physical: 2.54m × 1.27m (100" × 50")
- Aspect Ratio: 2:1 (exactly)

**4K Canonical Pixel Dimensions**:

We need to choose pixel dimensions that:
1. Maintain 2:1 aspect ratio
2. Fit comfortably within 4K frame (3840×2160)
3. Provide sufficient resolution for ball tracking
4. Leave margin for camera viewport

**Recommended Table Dimensions**:
```python
TABLE_4K_WIDTH = 3200   # pixels
TABLE_4K_HEIGHT = 1600  # pixels

# Aspect ratio check: 3200 / 1600 = 2.0 ✓
# Fits in 4K: 3200 < 3840 ✓, 1600 < 2160 ✓
# Margins: left/right = 320px each, top/bottom = 280px each
```

**Table center position** (in 4K canonical):
```python
TABLE_CENTER_X = 1920  # Center of 3840
TABLE_CENTER_Y = 1080  # Center of 2160
```

**Table bounds** (in 4K canonical):
```python
TABLE_MIN_X = TABLE_CENTER_X - TABLE_4K_WIDTH / 2   # 1920 - 1600 = 320
TABLE_MAX_X = TABLE_CENTER_X + TABLE_4K_WIDTH / 2   # 1920 + 1600 = 3520
TABLE_MIN_Y = TABLE_CENTER_Y - TABLE_4K_HEIGHT / 2  # 1080 - 800 = 280
TABLE_MAX_Y = TABLE_CENTER_Y + TABLE_4K_HEIGHT / 2  # 1080 + 800 = 1880
```

### 3.2 Ball Size in Pixels

**Standard Pool Ball**:
- Diameter: 57.15mm (2.25 inches)
- Radius: 28.575mm

**Pixel Radius Calculation**:
```python
# Table width: 2.54m = 3200 pixels
# Therefore: 1 pixel = 2.54 / 3200 = 0.79375 mm

# Ball radius in pixels:
BALL_RADIUS_4K = 28.575 / 0.79375 = 36.0 pixels

# Or equivalently:
# BALL_RADIUS_4K = (28.575mm / 2540mm) * 3200px = 36.0 pixels
```

**Standard Ball Properties**:
```python
# 4K canonical scale
BALL_RADIUS_4K_PIXELS = 36.0        # Radius in pixels
BALL_DIAMETER_4K_PIXELS = 72.0      # Diameter in pixels
BALL_MASS_KG = 0.17                 # Mass (kept in kg - not spatial)
```

### 3.3 Pixel-to-Physical Conversion Factor

For documentation and physics understanding:

```python
# Conversion factors (for reference/documentation only)
PIXELS_PER_METER_4K = 3200 / 2.54 = 1259.84 pixels/meter
MM_PER_PIXEL_4K = 2540 / 3200 = 0.79375 mm/pixel

# Note: These are NOT stored or used in code!
# They're purely for human understanding and validation
```

### 3.4 Pocket Positions

**Standard 9ft table has 6 pockets**:

```python
# All in 4K canonical coordinates
POCKET_POSITIONS_4K = [
    Vector2D(x=320, y=280, scale=[1.0, 1.0]),    # Top-left corner
    Vector2D(x=1920, y=280, scale=[1.0, 1.0]),   # Top-middle
    Vector2D(x=3520, y=280, scale=[1.0, 1.0]),   # Top-right corner
    Vector2D(x=320, y=1880, scale=[1.0, 1.0]),   # Bottom-left corner
    Vector2D(x=1920, y=1880, scale=[1.0, 1.0]),  # Bottom-middle
    Vector2D(x=3520, y=1880, scale=[1.0, 1.0]),  # Bottom-right corner
]

# Pocket radius (standard ~4.5 inches = 114.3mm)
POCKET_RADIUS_4K = 114.3 / 0.79375 = 144 pixels
```

### 3.5 Configuration Constants

**New constants module** (`backend/core/constants_4k.py`):

```python
"""4K canonical coordinate system constants."""

# === 4K CANONICAL RESOLUTION ===
CANONICAL_WIDTH = 3840
CANONICAL_HEIGHT = 2160
CANONICAL_RESOLUTION = (CANONICAL_WIDTH, CANONICAL_HEIGHT)

# === TABLE DIMENSIONS (4K PIXELS) ===
TABLE_WIDTH_4K = 3200      # pixels (maintains 2:1 aspect ratio)
TABLE_HEIGHT_4K = 1600     # pixels
TABLE_CENTER_X_4K = 1920   # Center of 4K frame
TABLE_CENTER_Y_4K = 1080   # Center of 4K frame

# Bounds
TABLE_MIN_X_4K = 320       # Left edge
TABLE_MAX_X_4K = 3520      # Right edge
TABLE_MIN_Y_4K = 280       # Top edge
TABLE_MAX_Y_4K = 1880      # Bottom edge

# === BALL DIMENSIONS (4K PIXELS) ===
BALL_RADIUS_4K = 36.0      # Standard 57.15mm ball
BALL_DIAMETER_4K = 72.0
BALL_MASS_KG = 0.17        # Mass in kg (not spatial)

# === POCKET DIMENSIONS (4K PIXELS) ===
POCKET_RADIUS_4K = 144.0   # Standard ~4.5" pocket

# === REFERENCE CONVERSIONS (DOCUMENTATION ONLY) ===
# These are NOT used in code - purely for validation/documentation
MM_PER_PIXEL_4K = 0.79375         # 2540mm / 3200px
PIXELS_PER_MM_4K = 1.259842       # 3200px / 2540mm
PHYSICAL_TABLE_WIDTH_MM = 2540.0  # 9ft table = 2.54m
PHYSICAL_TABLE_HEIGHT_MM = 1270.0 # 4.5ft = 1.27m
```

---

## 4. Migration Strategy

### 4.1 Phased Approach

The migration will proceed in **5 sequential phases**:

#### Phase 1: Foundation (Infrastructure)
- Create new constants and utilities
- Implement new Vector2D with mandatory scale
- Create conversion helpers
- Set up validation framework
- **Duration**: 2-3 days
- **Risk**: Low

#### Phase 2: Core Models
- Migrate BallState, TableState, CueState
- Update all model methods to use 4K pixels
- Maintain backward compatibility wrappers
- **Duration**: 3-4 days
- **Risk**: Medium

#### Phase 3: Physics & Collision
- Convert physics calculations to pixel-based
- Update trajectory calculations
- Migrate collision detection
- **Duration**: 4-5 days
- **Risk**: High (physics must remain accurate)

#### Phase 4: Vision & Calibration
- Update vision detection output format
- Simplify calibration (no more pixels_per_meter)
- Update integration helpers
- **Duration**: 2-3 days
- **Risk**: Medium

#### Phase 5: API & Cleanup
- Update API models and converters
- Remove legacy code paths
- Complete test coverage
- Documentation update
- **Duration**: 2-3 days
- **Risk**: Low

### 4.2 Backward Compatibility Strategy

**Dual-mode operation during transition**:

```python
class Vector2D:
    """Enhanced Vector2D with backward compatibility."""

    def __init__(self, x: float, y: float, scale: Optional[tuple[float, float]] = None):
        self.x = x
        self.y = y

        # LEGACY MODE: Allow missing scale temporarily
        if scale is None:
            warnings.warn(
                "Vector2D without scale is deprecated. "
                "Scale will become mandatory in v2.0",
                DeprecationWarning,
                stacklevel=2
            )
            # Assume 4K canonical for legacy code
            scale = (1.0, 1.0)

        self.scale = scale

    @classmethod
    def from_legacy_meters(cls, x_meters: float, y_meters: float) -> "Vector2D":
        """Temporary helper: Convert old meter-based coords to 4K pixels."""
        # Table center is origin in old system
        # Convert to 4K canonical
        x_4k = TABLE_CENTER_X_4K + (x_meters * PIXELS_PER_METER_4K)
        y_4k = TABLE_CENTER_Y_4K + (y_meters * PIXELS_PER_METER_4K)
        return cls(x=x_4k, y=y_4k, scale=[1.0, 1.0])
```

**Migration helpers**:

```python
# backend/core/migration_utils.py

def convert_ballstate_to_4k(legacy_ball: BallState) -> BallState:
    """Convert legacy meter-based BallState to 4K pixels."""
    # Legacy position in meters from table center
    x_meters = legacy_ball.position.x
    y_meters = legacy_ball.position.y

    # Convert to 4K canonical pixels
    x_4k = TABLE_CENTER_X_4K + (x_meters * PIXELS_PER_METER_4K)
    y_4k = TABLE_CENTER_Y_4K + (y_meters * PIXELS_PER_METER_4K)

    # Convert velocity (m/s → pixels/s)
    vx_4k = legacy_ball.velocity.x * PIXELS_PER_METER_4K
    vy_4k = legacy_ball.velocity.y * PIXELS_PER_METER_4K

    # Convert radius (m → pixels)
    radius_4k = legacy_ball.radius * PIXELS_PER_METER_4K

    return BallState(
        id=legacy_ball.id,
        position=Vector2D(x=x_4k, y=y_4k, scale=[1.0, 1.0]),
        velocity=Vector2D(x=vx_4k, y=vy_4k, scale=[1.0, 1.0]),
        radius=radius_4k,
        mass=legacy_ball.mass,  # Mass stays in kg
        # ... other fields
    )

def convert_4k_to_legacy_meters(ball_4k: BallState) -> BallState:
    """Convert 4K pixel-based BallState back to legacy meters (for API compatibility)."""
    # Get 4K canonical coordinates
    pos_4k = ball_4k.position.to_4k_canonical()

    # Convert to meters from table center
    x_meters = (pos_4k.x - TABLE_CENTER_X_4K) / PIXELS_PER_METER_4K
    y_meters = (pos_4k.y - TABLE_CENTER_Y_4K) / PIXELS_PER_METER_4K

    # ... similar for velocity and radius

    return BallState(...)  # Legacy format
```

### 4.3 Database/Storage Migration

**No database migration needed!**

Current storage already uses JSON serialization. We'll version the data:

```python
# Current (meters)
{
    "version": "1.0",
    "coordinate_system": "world_meters",
    "balls": [
        {
            "id": "cue",
            "position": {"x": 0.5, "y": 0.2},  # meters
            "coordinate_space": "world_meters"
        }
    ]
}

# New (4K pixels)
{
    "version": "2.0",
    "coordinate_system": "4k_canonical",
    "balls": [
        {
            "id": "cue",
            "position": {
                "x": 2550.0,  # pixels
                "y": 1332.0,  # pixels
                "scale": [1.0, 1.0]
            },
            "coordinate_space": "4k_canonical"
        }
    ]
}
```

**Loader with automatic conversion**:

```python
def load_game_state(data: dict) -> GameState:
    """Load game state with automatic version detection and migration."""
    version = data.get("version", "1.0")

    if version == "1.0":
        # Legacy meter-based data
        return _load_legacy_meters(data)
    elif version == "2.0":
        # New 4K pixel-based data
        return _load_4k_canonical(data)
    else:
        raise ValueError(f"Unknown data version: {version}")
```

### 4.4 Physics Engine Adaptation

**Challenge**: Physics calculations typically use SI units (meters, seconds)

**Solution**: Keep physics calculations in SI units internally, but:
1. Convert inputs from pixels to meters
2. Perform physics calculations
3. Convert outputs back to pixels

```python
class PhysicsEngine:
    """Physics engine with pixel-to-SI conversion."""

    def __init__(self):
        # Internal SI conversion factor (for physics calculations only)
        self._mm_per_pixel = MM_PER_PIXEL_4K
        self._m_per_pixel = self._mm_per_pixel / 1000.0

    def simulate_collision(
        self,
        ball1: BallState,  # Position in 4K pixels
        ball2: BallState   # Position in 4K pixels
    ) -> tuple[Vector2D, Vector2D]:  # Velocities in 4K pixels/second
        """Simulate ball collision using physics."""

        # 1. Convert positions to meters (for SI physics)
        pos1_m = self._pixels_to_meters(ball1.position)
        pos2_m = self._pixels_to_meters(ball2.position)

        # 2. Convert velocities to m/s
        vel1_m = self._pixels_to_meters(ball1.velocity)
        vel2_m = self._pixels_to_meters(ball2.velocity)

        # 3. Perform physics calculation in SI units
        new_vel1_m, new_vel2_m = self._calculate_collision_si(
            pos1_m, pos2_m, vel1_m, vel2_m, ball1.mass, ball2.mass
        )

        # 4. Convert resulting velocities back to pixels/s
        new_vel1_px = self._meters_to_pixels(new_vel1_m)
        new_vel2_px = self._meters_to_pixels(new_vel2_m)

        return new_vel1_px, new_vel2_px

    def _pixels_to_meters(self, vec_px: Vector2D) -> Vector2D:
        """Convert pixel vector to meters (for internal physics only)."""
        # Get 4K canonical coordinates first
        vec_4k = vec_px.to_4k_canonical()

        # Convert to meters
        x_m = vec_4k.x * self._m_per_pixel
        y_m = vec_4k.y * self._m_per_pixel

        return Vector2D(x=x_m, y=y_m, scale=[1.0, 1.0])  # scale irrelevant in SI

    def _meters_to_pixels(self, vec_m: Vector2D) -> Vector2D:
        """Convert meter vector back to 4K pixels."""
        x_4k = vec_m.x / self._m_per_pixel
        y_4k = vec_m.y / self._m_per_pixel

        return Vector2D(x=x_4k, y=y_4k, scale=[1.0, 1.0])
```

**Key insight**: Physics stays in SI units (meters), but this is now an **internal implementation detail**, not part of the public API or storage format.

---

## 5. Breaking Changes

### 5.1 API Changes

#### Removed Endpoints/Fields

```python
# REMOVED: Calibration endpoints returning pixels_per_meter
DELETE /api/calibration/pixels_per_meter

# REMOVED: Coordinate space enum values
# - CoordinateSpace.WORLD_METERS
# - CoordinateSpace.TABLE_PIXELS
# - CoordinateSpace.NORMALIZED
```

#### Changed Response Formats

**Before (v1.0)**:
```json
{
  "balls": [
    {
      "id": "cue",
      "position": [0.5, 0.2],  // meters from table center
      "velocity": [1.5, 0.0],  // m/s
      "radius": 0.028575,      // meters
      "coordinate_space": "world_meters"
    }
  ],
  "table": {
    "width": 2.54,             // meters
    "height": 1.27             // meters
  }
}
```

**After (v2.0)**:
```json
{
  "balls": [
    {
      "id": "cue",
      "position": {
        "x": 2550.0,           // 4K pixels
        "y": 1332.0,           // 4K pixels
        "scale": [1.0, 1.0]    // MANDATORY
      },
      "velocity": {
        "x": 1889.76,          // pixels/second (4K)
        "y": 0.0,              // pixels/second
        "scale": [1.0, 1.0]
      },
      "radius": 36.0           // pixels (4K)
    }
  ],
  "table": {
    "width": 3200,             // pixels (4K)
    "height": 1600,            // pixels (4K)
    "center_x": 1920,          // pixels (4K)
    "center_y": 1080           // pixels (4K)
  },
  "resolution": {
    "canonical": [3840, 2160],
    "reference": "4K_UHD"
  }
}
```

### 5.2 Configuration File Changes

**Before** (`config.json`):
```json
{
  "table": {
    "width": 2.54,
    "height": 1.27,
    "pixels_per_meter": 754.0
  },
  "camera": {
    "resolution": [1920, 1080]
  }
}
```

**After** (`config.json`):
```json
{
  "table": {
    "width_4k": 3200,
    "height_4k": 1600,
    "center_4k": [1920, 1080]
  },
  "camera": {
    "native_resolution": [1920, 1080],
    "scale_to_4k": [2.0, 2.0]
  },
  "canonical_resolution": [3840, 2160]
}
```

### 5.3 Code API Changes

#### Vector2D Constructor

**Before**:
```python
# Optional metadata
v = Vector2D(x=100, y=50)  # OK
v = Vector2D.world_meters(1.0, 0.5)  # OK
v = Vector2D.camera_pixels(960, 540, Resolution(1920, 1080))  # OK
```

**After**:
```python
# Mandatory scale
v = Vector2D(x=100, y=50, scale=[2.0, 2.0])  # REQUIRED
v = Vector2D.from_4k(1920, 1080)  # Creates with scale=[1.0, 1.0]
v = Vector2D.from_resolution(960, 540, resolution=(1920, 1080))  # Auto-calculates scale
```

#### BallState Factory Methods

**Before**:
```python
BallState.create(id="ball_1", x=1.27, y=0.635)  # Meters
BallState.from_camera_pixels(id="ball_1", x=960, y=540, ...)
```

**After**:
```python
BallState.from_4k(id="ball_1", x=1920, y=1080)  # 4K pixels
BallState.from_resolution(id="ball_1", x=960, y=540, resolution=(1920, 1080))
```

#### CoordinateConverter Removal

**Before**:
```python
from backend.core.coordinate_converter import CoordinateConverter

converter = CoordinateConverter(pixels_per_meter=754.0)
world_pos = converter.camera_pixels_to_world_meters(camera_pos)
```

**After**:
```python
from backend.core.resolution_converter import ResolutionConverter

converter = ResolutionConverter()
pos_4k = converter.to_4k_canonical(pos, source_resolution=(1920, 1080))
```

### 5.4 Database Schema Changes

**No breaking changes** - JSON storage format is versioned and auto-migrated on load.

Legacy data (v1.0 with meters) will be automatically converted to v2.0 (4K pixels) on first load.

---

## 6. Implementation Groups

### 6.1 Grouping Strategy

To enable **parallel implementation by 10 subagents**, files are grouped by:
1. Module boundaries (minimize cross-dependencies)
2. Functional cohesion (related functionality together)
3. Complexity balance (distribute difficult tasks evenly)
4. Dependency order (foundational groups first)

### 6.2 Group Definitions

#### **Group 1: Foundation & Constants** [Priority: P0 - MUST BE FIRST]

**Owner**: Subagent 1
**Duration**: 2 days
**Complexity**: Low
**Dependencies**: None

**Files**:
- `backend/core/constants_4k.py` [NEW]
- `backend/core/resolution_converter.py` [NEW]
- `backend/core/validation_4k.py` [NEW]

**Tasks**:
1. Create `constants_4k.py` with all 4K canonical constants
2. Implement `ResolutionConverter` class for resolution scaling
3. Create validation utilities for 4K coordinates
4. Write comprehensive unit tests

**Deliverables**:
- ✅ All 4K constants defined
- ✅ Conversion utilities tested
- ✅ Validation helpers ready
- ✅ Documentation complete

---

#### **Group 2: Enhanced Vector2D** [Priority: P0 - MUST BE SECOND]

**Owner**: Subagent 2
**Duration**: 3 days
**Complexity**: Medium
**Dependencies**: Group 1

**Files**:
- `backend/core/coordinates.py` [MODIFY]
- `backend/core/models.py` [MODIFY - Vector2D usage only]

**Tasks**:
1. Update `Vector2D` to make `scale` mandatory
2. Remove `coordinate_space` field (replaced by scale)
3. Add factory methods: `from_4k()`, `from_resolution()`
4. Implement `to_4k_canonical()` method
5. Update all Vector2D operations to preserve scale
6. Create backward compatibility helpers
7. Write migration utilities

**Deliverables**:
- ✅ Vector2D with mandatory scale
- ✅ Factory methods implemented
- ✅ Backward compatibility wrappers
- ✅ Comprehensive tests

**Breaking Changes**:
- `Vector2D(x, y)` without scale raises error
- `coordinate_space` property removed
- Factory methods replaced

---

#### **Group 3: Core Models (BallState, TableState, CueState)** [Priority: P1]

**Owner**: Subagent 3
**Duration**: 4 days
**Complexity**: High
**Dependencies**: Group 2

**Files**:
- `backend/core/models.py` [MODIFY - all state classes]
- `backend/core/game_state.py` [MODIFY]

**Tasks**:
1. Update `BallState`:
   - Position in 4K pixels (not meters)
   - Velocity in pixels/second (not m/s)
   - Radius in pixels (not meters)
   - Update factory methods

2. Update `TableState`:
   - Dimensions in 4K pixels
   - Pocket positions in 4K pixels
   - Remove meter-based fields

3. Update `CueState`:
   - Tip position in 4K pixels
   - Length in pixels (not meters)

4. Update `GameState`:
   - Remove `CoordinateMetadata`
   - Add `resolution` field = 4K canonical

5. Create migration helpers for legacy data

**Deliverables**:
- ✅ All models use 4K pixels
- ✅ Factory methods updated
- ✅ Legacy loaders implemented
- ✅ Full test coverage

**Breaking Changes**:
- `BallState.create()` signature changed
- `TableState.standard_9ft_table()` returns pixels
- All serialization formats changed

---

#### **Group 4: Physics Engine** [Priority: P1]

**Owner**: Subagent 4
**Duration**: 5 days
**Complexity**: Very High
**Dependencies**: Group 3

**Files**:
- `backend/core/physics/trajectory.py` [MODIFY]
- `backend/core/physics/collision.py` [MODIFY]
- `backend/core/physics/spin.py` [MODIFY]

**Tasks**:
1. Update trajectory calculations:
   - Input: positions in 4K pixels
   - Internal: convert to meters for SI physics
   - Output: trajectories in 4K pixels

2. Update collision detection:
   - Ball-ball collisions in pixels
   - Ball-cushion collisions in pixels
   - SI unit conversions internal only

3. Update spin calculations:
   - Spin vectors in pixels/radian

4. Validate physics accuracy:
   - Compare old vs new calculations
   - Ensure numerical stability
   - Verify energy conservation

**Deliverables**:
- ✅ Physics calculations use 4K pixels
- ✅ SI conversions internal only
- ✅ Accuracy validated
- ✅ Performance benchmarks

**Critical Requirement**: Physics must remain **byte-for-byte identical** to current implementation (within floating-point tolerance).

---

#### **Group 5: Collision Detection** [Priority: P1]

**Owner**: Subagent 5
**Duration**: 3 days
**Complexity**: Medium
**Dependencies**: Group 4

**Files**:
- `backend/core/collision/geometric_collision.py` [MODIFY]

**Tasks**:
1. Update geometric collision detection to use 4K pixels
2. Update cushion collision calculations
3. Update pocket detection
4. Validate collision accuracy

**Deliverables**:
- ✅ Collisions in 4K pixels
- ✅ Tests passing
- ✅ Accuracy validated

---

#### **Group 6: Analysis & Prediction** [Priority: P2]

**Owner**: Subagent 6
**Duration**: 3 days
**Complexity**: Medium
**Dependencies**: Group 4, Group 5

**Files**:
- `backend/core/analysis/shot.py` [MODIFY]
- `backend/core/analysis/prediction.py` [MODIFY]
- `backend/core/analysis/assistance.py` [MODIFY]

**Tasks**:
1. Update shot analysis to use 4K pixels
2. Update prediction algorithms
3. Update assistance recommendations
4. Validate analysis accuracy

**Deliverables**:
- ✅ Analysis in 4K pixels
- ✅ Predictions accurate
- ✅ Tests passing

---

#### **Group 7: Validation & Correction** [Priority: P2]

**Owner**: Subagent 7
**Duration**: 2 days
**Complexity**: Low
**Dependencies**: Group 3

**Files**:
- `backend/core/validation/state.py` [MODIFY]
- `backend/core/validation/physics.py` [MODIFY]
- `backend/core/validation/correction.py` [MODIFY]

**Tasks**:
1. Update validation rules for 4K pixels
2. Update physics validation
3. Update state correction
4. Update bounds checking

**Deliverables**:
- ✅ Validation uses 4K pixels
- ✅ Tests passing

---

#### **Group 8: Vision Integration** [Priority: P1]

**Owner**: Subagent 8
**Duration**: 4 days
**Complexity**: Medium
**Dependencies**: Group 2, Group 3

**Files**:
- `backend/vision/models.py` [MODIFY]
- `backend/vision/detection/detector_adapter.py` [MODIFY]
- `backend/integration_service_conversion_helpers.py` [MODIFY]

**Tasks**:
1. Update vision `Ball` model:
   - Position includes scale metadata
   - Radius includes scale

2. Update detector adapter:
   - Output includes resolution scale

3. Update integration helpers:
   - Remove meter conversions
   - Convert to 4K canonical

4. Update calibration simplification:
   - No more `pixels_per_meter`
   - Just resolution scaling

**Deliverables**:
- ✅ Vision outputs 4K canonical
- ✅ Integration simplified
- ✅ Tests passing

**Breaking Changes**:
- Vision `Ball` model format
- Calibration data format

---

#### **Group 9: API Models & Converters** [Priority: P2]

**Owner**: Subagent 9
**Duration**: 3 days
**Complexity**: Medium
**Dependencies**: Group 3

**Files**:
- `backend/api/models/converters.py` [MODIFY]
- `backend/api/models/vision_models.py` [MODIFY]

**Tasks**:
1. Update API model converters:
   - BallInfo in 4K pixels
   - TableInfo in 4K pixels
   - CueInfo in 4K pixels

2. Add versioning to API responses
3. Implement backward compatibility mode
4. Update OpenAPI specs

**Deliverables**:
- ✅ API models use 4K pixels
- ✅ Versioned responses
- ✅ Backward compatibility
- ✅ Documentation updated

---

#### **Group 10: Utilities & Examples** [Priority: P3]

**Owner**: Subagent 10
**Duration**: 2 days
**Complexity**: Low
**Dependencies**: All groups

**Files**:
- `backend/core/utils/geometry.py` [MODIFY]
- `backend/core/utils/math.py` [MODIFY]
- `backend/core/utils/example_cue_pointing.py` [MODIFY]

**Tasks**:
1. Update utility functions for 4K pixels
2. Update example code
3. Create migration examples
4. Update documentation

**Deliverables**:
- ✅ Utils use 4K pixels
- ✅ Examples updated
- ✅ Migration guide complete

---

### 6.3 Dependency Graph

```
Group 1 (Foundation)
    ↓
Group 2 (Vector2D)
    ↓
    ├─→ Group 3 (Models) ─→ Group 7 (Validation)
    │       ↓                      ↓
    │       └─→ Group 4 (Physics) → Group 6 (Analysis)
    │               ↓
    │           Group 5 (Collision)
    │
    └─→ Group 8 (Vision)
            ↓
        Group 9 (API)
            ↓
        Group 10 (Utils)

Critical Path: 1 → 2 → 3 → 4 → 5 → 6
Parallel: Groups 7, 8, 9, 10 can run concurrently with critical path
```

### 6.4 Parallel Execution Plan

**Week 1** (Foundation):
- Subagent 1: Group 1 (Days 1-2)
- Subagent 2: Group 2 (Days 3-5)

**Week 2** (Core + Parallel):
- Subagent 3: Group 3 (Days 1-4)
- Subagent 7: Group 7 (Days 1-2) [Parallel]
- Subagent 8: Group 8 (Days 1-4) [Parallel]

**Week 3** (Physics + API):
- Subagent 4: Group 4 (Days 1-5)
- Subagent 5: Group 5 (Days 1-3)
- Subagent 9: Group 9 (Days 3-5) [Parallel]

**Week 4** (Finalization):
- Subagent 6: Group 6 (Days 1-3)
- Subagent 10: Group 10 (Days 1-2)
- All: Integration testing (Days 3-5)

**Total Duration**: ~4 weeks with 10 subagents

---

## 7. Verification Plan

### 7.1 Unit Testing Requirements

**Each group must achieve**:
- ✅ 100% code coverage for new code
- ✅ 95%+ coverage for modified code
- ✅ All existing tests passing
- ✅ New tests for 4K conversions

**Test categories**:

1. **Conversion Accuracy Tests**:
   ```python
   def test_resolution_scaling_accuracy():
       """Test pixel coordinate scaling maintains accuracy."""
       # 1080p → 4K
       pos_1080p = Vector2D(x=960, y=540, scale=[2.0, 2.0])
       pos_4k = pos_1080p.to_4k_canonical()

       assert pos_4k.x == 1920.0
       assert pos_4k.y == 1080.0
       assert pos_4k.scale == [1.0, 1.0]

   def test_round_trip_conversion():
       """Test round-trip conversion maintains precision."""
       original = Vector2D(x=1234.5678, y=2345.6789, scale=[1.0, 1.0])

       # Convert to 1080p
       pos_1080p = original.to_resolution((1920, 1080))

       # Convert back to 4K
       pos_4k = pos_1080p.to_4k_canonical()

       # Should match within floating-point tolerance
       assert abs(pos_4k.x - original.x) < 1e-6
       assert abs(pos_4k.y - original.y) < 1e-6
   ```

2. **Physics Accuracy Tests**:
   ```python
   def test_physics_ball_collision_unchanged():
       """Verify physics calculations identical to old implementation."""
       # Setup identical scenario in both systems
       ball1_old = BallState_old.create("ball1", x=0.5, y=0.0)  # Meters
       ball2_old = BallState_old.create("ball2", x=-0.5, y=0.0)

       ball1_new = BallState.from_4k("ball1", x=2550, y=1080)  # Pixels (equivalent)
       ball2_new = BallState.from_4k("ball2", x=1290, y=1080)

       # Calculate collision in both systems
       v1_old, v2_old = physics_old.calculate_collision(ball1_old, ball2_old)
       v1_new, v2_new = physics_new.calculate_collision(ball1_new, ball2_new)

       # Convert new to meters for comparison
       v1_new_meters = v1_new.to_meters()
       v2_new_meters = v2_new.to_meters()

       # Must match within 0.1% tolerance
       assert abs(v1_new_meters.x - v1_old.x) / v1_old.x < 0.001
       assert abs(v2_new_meters.x - v2_old.x) / v2_old.x < 0.001
   ```

3. **API Compatibility Tests**:
   ```python
   def test_api_response_format_v2():
       """Verify v2.0 API response format."""
       ball = BallState.from_4k("cue", x=1920, y=1080)
       response = ball_state_to_api_response(ball)

       assert "position" in response
       assert "x" in response["position"]
       assert "y" in response["position"]
       assert "scale" in response["position"]
       assert response["position"]["scale"] == [1.0, 1.0]

   def test_legacy_data_loading():
       """Verify legacy v1.0 data can be loaded and auto-migrated."""
       legacy_data = {
           "version": "1.0",
           "coordinate_system": "world_meters",
           "balls": [{
               "id": "cue",
               "position": {"x": 0.5, "y": 0.2},
               "coordinate_space": "world_meters"
           }]
       }

       game_state = GameState.from_dict(legacy_data)

       # Verify auto-migration to 4K
       assert game_state.version == "2.0"
       assert game_state.balls[0].position.scale == [1.0, 1.0]
       # Position should be converted to 4K pixels
       assert 2000 < game_state.balls[0].position.x < 3000  # Roughly right half
   ```

### 7.2 Integration Testing

**End-to-end workflow tests**:

```python
def test_full_detection_pipeline():
    """Test complete workflow: Detection → Physics → API → Frontend."""

    # 1. Vision detection (1920×1080 camera)
    frame = load_test_frame()  # 1920×1080
    detected_balls = detector.detect(frame)

    # Verify scale metadata present
    assert detected_balls[0].position.scale == [2.0, 2.0]

    # 2. Convert to game state (4K canonical)
    ball_states = [
        BallState.from_vision_detection(ball)
        for ball in detected_balls
    ]

    # Verify 4K canonical
    assert ball_states[0].position.scale == [1.0, 1.0]

    # 3. Run physics simulation
    game_state = GameState(balls=ball_states, ...)
    trajectory = physics_engine.predict_trajectory(game_state)

    # Verify trajectory in 4K
    assert trajectory.points[0].scale == [1.0, 1.0]

    # 4. Serialize for API
    api_response = game_state.to_dict()

    # Verify API format
    assert api_response["version"] == "2.0"
    assert api_response["balls"][0]["position"]["scale"] == [1.0, 1.0]

    # 5. Frontend receives and renders (simulate 800×450 canvas)
    canvas_pos = frontend_converter.to_canvas(
        api_response["balls"][0]["position"],
        canvas_size=(800, 450)
    )

    # Verify canvas coordinates
    assert 0 <= canvas_pos.x <= 800
    assert 0 <= canvas_pos.y <= 450
```

### 7.3 Regression Testing

**Automated comparison with legacy system**:

```python
class RegressionTestSuite:
    """Compare old meter-based system with new 4K pixel system."""

    def test_trajectory_predictions(self):
        """Verify trajectory predictions match old system."""
        # Load test scenarios
        scenarios = load_test_scenarios()  # 100+ real game states

        for scenario in scenarios:
            # Run old system
            old_trajectory = old_physics.predict(scenario)

            # Convert to new system
            scenario_4k = migrate_scenario_to_4k(scenario)
            new_trajectory = new_physics.predict(scenario_4k)

            # Compare trajectories point-by-point
            for old_point, new_point in zip(old_trajectory.points, new_trajectory.points):
                # Convert both to same unit for comparison
                new_point_meters = new_point.to_meters()

                # Must match within 1mm (tolerance for floating-point)
                assert abs(new_point_meters.x - old_point.x) < 0.001
                assert abs(new_point_meters.y - old_point.y) < 0.001

    def test_collision_detection(self):
        """Verify collision detection matches old system."""
        # Test 1000+ collision scenarios
        ...

    def test_api_responses(self):
        """Verify API responses can be consumed by existing clients."""
        ...
```

### 7.4 Performance Benchmarks

**Measure performance impact**:

```python
import pytest

@pytest.mark.benchmark
def test_coordinate_conversion_performance(benchmark):
    """Benchmark coordinate conversion overhead."""
    positions = [
        Vector2D(x=random.random() * 3840, y=random.random() * 2160, scale=[1.0, 1.0])
        for _ in range(1000)
    ]

    def convert_batch():
        return [p.to_resolution((1920, 1080)) for p in positions]

    result = benchmark(convert_batch)

    # Should complete in < 1ms for 1000 conversions
    assert benchmark.stats['mean'] < 0.001

@pytest.mark.benchmark
def test_physics_simulation_performance(benchmark):
    """Benchmark physics simulation performance."""
    game_state = create_test_game_state(num_balls=16)

    def simulate():
        return physics_engine.simulate_shot(game_state, duration=5.0)

    result = benchmark(simulate)

    # Should be no slower than old system (< 10ms for 5-second simulation)
    assert benchmark.stats['mean'] < 0.010
```

**Performance targets**:
- Coordinate conversion: < 1μs per vector
- Physics simulation: Same speed as old system (±5%)
- API serialization: < 100μs per game state
- Memory usage: Same as old system (±10%)

### 7.5 Documentation Validation

**Checklist**:
- ✅ All code comments updated
- ✅ API documentation regenerated
- ✅ Migration guide complete
- ✅ Example code tested
- ✅ README updated
- ✅ Breaking changes documented

---

## 8. Rollback Plan

### 8.1 Rollback Triggers

Rollback if:
- ❌ Physics accuracy degraded > 1%
- ❌ Performance regression > 10%
- ❌ Critical bugs in production
- ❌ Data corruption detected
- ❌ Cannot achieve backward compatibility

### 8.2 Rollback Procedure

**Step 1**: Identify rollback scope
```python
# Check which groups are affected
affected_groups = identify_failing_groups()

# Rollback only affected groups or entire migration
rollback_scope = determine_scope(affected_groups)
```

**Step 2**: Restore code
```bash
# Create rollback branch
git checkout -b rollback/4k-migration

# Revert commits for affected groups
git revert <commit-range>

# Deploy rollback
git push origin rollback/4k-migration
```

**Step 3**: Migrate data back to v1.0
```python
def rollback_data_to_v1():
    """Convert v2.0 data back to v1.0 format."""
    # Load all v2.0 game states
    states_v2 = load_all_game_states()

    for state in states_v2:
        # Convert 4K pixels back to meters
        state_v1 = convert_4k_to_meters(state)

        # Save as v1.0
        save_game_state_v1(state_v1)
```

**Step 4**: Verify rollback
```python
# Run all tests with legacy system
pytest tests/ --legacy-mode

# Verify data integrity
verify_data_integrity()

# Check API compatibility
test_api_endpoints()
```

### 8.3 Partial Rollback

If only specific groups fail:

```python
# Example: Rollback only physics engine (Group 4)
ROLLBACK_GROUPS = [4]

for group in ROLLBACK_GROUPS:
    rollback_group(group)
    restore_legacy_version(group)

# Keep successful migrations
KEEP_GROUPS = [1, 2, 3, 8, 9, 10]
```

### 8.4 Data Recovery

**Backup strategy**:
```python
# Before any migration, backup all data
def backup_before_migration():
    timestamp = datetime.now().isoformat()
    backup_path = f"backups/pre-4k-migration-{timestamp}"

    # Backup database
    backup_database(backup_path)

    # Backup config files
    backup_configs(backup_path)

    # Backup calibration data
    backup_calibration(backup_path)

    return backup_path

# After migration failure, restore
def restore_from_backup(backup_path):
    restore_database(backup_path)
    restore_configs(backup_path)
    restore_calibration(backup_path)
```

---

## 9. Timeline & Dependencies

### 9.1 Overall Timeline

**Total Duration**: 4 weeks (20 working days)

```
Week 1: Foundation
├─ Days 1-2: Group 1 (Foundation)
└─ Days 3-5: Group 2 (Vector2D)

Week 2: Core Models + Parallel Work
├─ Days 6-9: Group 3 (Models)
├─ Days 6-7: Group 7 (Validation) [Parallel]
└─ Days 6-9: Group 8 (Vision) [Parallel]

Week 3: Physics + API
├─ Days 10-14: Group 4 (Physics)
├─ Days 10-12: Group 5 (Collision)
└─ Days 13-15: Group 9 (API) [Parallel]

Week 4: Analysis + Finalization
├─ Days 16-18: Group 6 (Analysis)
├─ Days 16-17: Group 10 (Utils)
└─ Days 18-20: Integration Testing & Deployment
```

### 9.2 Critical Path

**Critical Path** (longest dependency chain):
```
Group 1 → Group 2 → Group 3 → Group 4 → Group 5 → Group 6
  2d        3d        4d        5d        3d        3d
```

**Total Critical Path**: 20 days

**Parallelizable Work**: Groups 7, 8, 9, 10 can run concurrently, reducing overall time.

### 9.3 Resource Allocation

| Week | Subagents Active | Groups in Progress |
|------|------------------|--------------------|
| 1 | 2 | Groups 1, 2 |
| 2 | 3 | Groups 3, 7, 8 |
| 3 | 4 | Groups 4, 5, 9 |
| 4 | 3 | Groups 6, 10, Testing |

**Peak resource usage**: Week 3 (4 subagents)

### 9.4 Milestone Checklist

**Milestone 1: Foundation Complete** (End of Week 1)
- ✅ Constants defined
- ✅ ResolutionConverter implemented
- ✅ Vector2D with mandatory scale
- ✅ All tests passing
- ✅ Documentation complete

**Milestone 2: Core Models Complete** (End of Week 2)
- ✅ BallState, TableState, CueState migrated
- ✅ GameState updated
- ✅ Validation migrated
- ✅ Vision integration complete
- ✅ Legacy loaders working

**Milestone 3: Physics Complete** (End of Week 3)
- ✅ Physics engine migrated
- ✅ Collision detection migrated
- ✅ API models updated
- ✅ Accuracy validated
- ✅ Performance benchmarked

**Milestone 4: System Integration** (End of Week 4)
- ✅ Analysis & prediction migrated
- ✅ Utilities updated
- ✅ End-to-end tests passing
- ✅ Regression tests passing
- ✅ Documentation complete
- ✅ Ready for deployment

---

## Appendices

### Appendix A: Quick Reference

**Common Conversions**:

```python
# Any resolution → 4K canonical
pos_4k = pos.to_4k_canonical()

# 4K canonical → specific resolution
pos_1080p = Vector2D.from_4k_to_resolution(pos_4k, (1920, 1080))

# Create from specific resolution (auto-calculates scale)
pos = Vector2D.from_resolution(x=960, y=540, resolution=(1920, 1080))
# → Vector2D(x=960, y=540, scale=[2.0, 2.0])
```

**Standard Values**:

```python
# 4K Canonical
CANONICAL_RESOLUTION = (3840, 2160)

# Table (4K pixels)
TABLE_WIDTH_4K = 3200
TABLE_HEIGHT_4K = 1600
TABLE_CENTER_4K = (1920, 1080)

# Ball (4K pixels)
BALL_RADIUS_4K = 36.0
BALL_DIAMETER_4K = 72.0
```

### Appendix B: Troubleshooting

**Problem**: Scale metadata missing error

**Solution**:
```python
# Old code (fails):
v = Vector2D(x=100, y=50)  # ❌ Error: scale mandatory

# New code:
v = Vector2D.from_resolution(x=100, y=50, resolution=(1920, 1080))  # ✅
```

**Problem**: Physics results don't match old system

**Solution**:
```python
# Enable debug logging
import logging
logging.getLogger('physics').setLevel(logging.DEBUG)

# Compare step-by-step
old_result = old_physics.simulate(scenario)
new_result = new_physics.simulate(scenario_4k)
compare_trajectories(old_result, new_result)
```

**Problem**: API clients receiving unexpected format

**Solution**:
```python
# Check client API version
client_version = request.headers.get('API-Version', '1.0')

if client_version == '1.0':
    # Return legacy format
    response = convert_to_legacy_format(data)
else:
    # Return new v2.0 format
    response = data
```

### Appendix C: Testing Checklist

For each group:

- [ ] Unit tests written
- [ ] Unit tests passing (100% coverage for new code)
- [ ] Integration tests written
- [ ] Integration tests passing
- [ ] Regression tests passing
- [ ] Performance benchmarks run
- [ ] Documentation updated
- [ ] Code review completed
- [ ] Migration guide updated

---

## Conclusion

This migration plan provides a comprehensive roadmap for transitioning the billiards-trainer backend from a meter-based multi-coordinate-space system to a simplified 4K pixel-based system.

**Key Success Factors**:
1. ✅ Phased approach with clear dependencies
2. ✅ Parallel execution by 10 subagents
3. ✅ Comprehensive testing at every stage
4. ✅ Backward compatibility during transition
5. ✅ Clear rollback procedures

**Expected Benefits**:
- 🎯 Simplified architecture (1 coordinate system vs 4)
- 🎯 Reduced cognitive load for developers
- 🎯 Eliminated calibration dependency on `pixels_per_meter`
- 🎯 Consistent resolution handling across system
- 🎯 Clearer data model with mandatory scale metadata

**Timeline**: 4 weeks with 10 parallel subagents

**Next Steps**:
1. Review and approve this plan
2. Create subagent tasks for each group
3. Begin Phase 1 (Foundation) implementation
4. Monitor progress and adjust as needed

---

**Document Status**: ✅ Ready for Review
**Last Updated**: 2025-10-21
**Version**: 1.0
**Author**: System Architect
