# Resolution Standardization Plan for Billiards Trainer

**Document Version**: 1.0
**Date**: 2025-10-21
**Author**: Claude
**Status**: Proposal

## Executive Summary

The billiards-trainer system currently operates with multiple coordinate systems and resolutions across different components, leading to ambiguity and potential errors in coordinate transformations. This document proposes a comprehensive standardization strategy that introduces explicit resolution metadata in all vector/position data structures, establishes a canonical coordinate system, and provides clear conversion utilities.

## Table of Contents

1. [Current State Analysis](#current-state-analysis)
2. [Problem Statement](#problem-statement)
3. [Proposed Solution](#proposed-solution)
4. [Data Structure Design](#data-structure-design)
5. [Coordinate System Standards](#coordinate-system-standards)
6. [Conversion Strategy](#conversion-strategy)
7. [Migration Plan](#migration-plan)
8. [Performance Considerations](#performance-considerations)
9. [Implementation Roadmap](#implementation-roadmap)

---

## Current State Analysis

### Identified Resolution Contexts

The system operates with at least **4 different coordinate spaces**:

1. **Camera Native Resolution** (1920x1080)
   - Source: Camera hardware via OpenCV
   - Usage: Frame capture, initial detection
   - Location: `backend/video/`, `backend/vision/capture.py`

2. **Detection/Processing Resolution** (Variable)
   - YOLO models may use different input resolutions (e.g., 640x640, 1280x1280)
   - OpenCV detection happens at various scales
   - Location: `backend/vision/detection/`
   - Config: `vision.camera.resolution` = [1920, 1080]

3. **Physics World Coordinates** (Meters)
   - Standard 9ft table: 2.54m × 1.27m
   - Core physics engine operates in metric units
   - Location: `backend/core/models.py` (BallState, TableState)
   - Conversion factor: `pixels_per_meter` calculated during calibration

4. **Table Playing Area** (~640x360 pixels based on config)
   - Calibrated playing surface corners
   - Defined in `config.json`: `table.playing_area_corners`
   - Currently at approximately 640x360 pixel space
   - Location: `backend/core/models.py` (TableState.playing_area_corners)

5. **Frontend Canvas Resolution** (Variable)
   - Browser viewport/canvas dimensions
   - Dynamic based on window size
   - Location: `frontend/web/src/components/video/`
   - Transforms: `frontend/web/src/utils/coordinates.ts`

### Current Conversion Mechanisms

#### Backend Conversions

**Geometric Calibration** (`backend/vision/calibration/geometry.py`):
```python
class GeometricCalibrator:
    def pixel_to_world_coordinates(self, pixel_pos, mapping) -> tuple[float, float]:
        # Converts pixel → meters
        world_x = (px - tx) / mapping.scale_factor
        world_y = (py - ty) / mapping.scale_factor

    def world_to_pixel_coordinates(self, world_pos, mapping) -> tuple[float, float]:
        # Converts meters → pixel
        pixel_x = wx * mapping.scale_factor + tx
        pixel_y = wy * mapping.scale_factor + ty
```

**State Conversion** (`backend/integration_service_conversion_helpers.py`):
```python
# Vision Ball → Core BallState
ball_state = BallState(
    position=Vector2D(ball.position[0], ball.position[1]),  # ASSUMPTION: same units!
    velocity=Vector2D(ball.velocity[0], ball.velocity[1])   # ASSUMPTION: m/s!
)
```

**Core Models** (`backend/core/models.py`):
```python
class TableState:
    def scale_playing_area_corners(self, from_width, from_height, to_width, to_height):
        scale_x = to_width / from_width
        scale_y = to_height / from_height
        self.playing_area_corners = [
            Vector2D(corner.x * scale_x, corner.y * scale_y)
            for corner in self.playing_area_corners
        ]
```

#### Frontend Conversions

**Coordinate Transform** (`frontend/web/src/utils/coordinates.ts`):
```typescript
export function createCoordinateTransform(
  videoSize: Size2D,      // e.g., {width: 1920, height: 1080}
  canvasSize: Size2D,     // e.g., {width: 800, height: 450}
  transform: ViewportTransform
): CoordinateTransform {
  const uniformScale = Math.min(scaleX, scaleY);
  const offsetX = (canvasSize.width - videoSize.width * uniformScale) / 2;

  return {
    videoToCanvas: (point: Point2D) => {...},
    canvasToVideo: (point: Point2D) => {...}
  };
}
```

### Issues Identified

1. **Implicit Assumptions**
   - Vision Ball positions assumed to be in camera pixels
   - Core BallState positions assumed to be in meters
   - No explicit metadata indicating coordinate space
   - Conversion happens at integration boundaries without validation

2. **Mixed Units**
   - Vision models use pixels: `Ball.position: tuple[float, float]`
   - Core models use meters: `BallState.position: Vector2D` (in meters)
   - Frontend uses pixels: `Ball.position: Point2D`
   - No type-level distinction

3. **Scale Factor Ambiguity**
   - `pixels_per_meter` stored in calibration but not carried with positions
   - Playing area corners stored in pixels but reference resolution unclear
   - Config shows corners at ~640x360 but camera is 1920x1080

4. **Transformation Loss**
   - Multiple transformations can compound errors
   - No validation that transformations are invertible
   - No tracking of which coordinate space a value is in

5. **Configuration Confusion**
   ```json
   "vision.camera.resolution": [1920, 1080],
   "table.playing_area_corners": [
     {"x": 37, "y": 45},    // What resolution is this in?
     {"x": 606, "y": 39},   // Appears to be ~640 wide
     ...
   ]
   ```

---

## Problem Statement

**Core Issue**: The system lacks explicit metadata about coordinate spaces and resolutions, leading to:

1. **Ambiguous transformations** - Unclear which resolution coordinates are expressed in
2. **Error-prone conversions** - Manual scaling without validation
3. **Debugging difficulty** - Hard to trace coordinate system bugs
4. **Inflexible architecture** - Adding new detectors/displays requires careful analysis
5. **Performance overhead** - Redundant conversions due to unclear canonical form

**Critical Question**: When a `Vector2D(100, 50)` is passed around, what does it represent?
- 100 pixels in 1920x1080 space?
- 100 pixels in 640x360 space?
- 100 millimeters in world space?
- 0.1 meters in world space?

**Current Answer**: It depends on context, which is error-prone.

---

## Proposed Solution

### Guiding Principles

1. **Explicit is Better Than Implicit** - All coordinates carry metadata about their space
2. **Single Canonical Format** - One authoritative coordinate system for storage
3. **Lazy Conversion** - Convert only when necessary, preserve original format
4. **Type Safety** - Use type system to prevent coordinate space mixing
5. **Performance Conscious** - Minimize conversions in hot paths

### Solution Overview

Introduce **Resolution-Aware Coordinate Types** with three-tier strategy:

1. **Type-Level Distinction** - Different types for different spaces
2. **Canonical Storage** - All persistent coordinates in one standard format
3. **Conversion Utilities** - Well-tested, bidirectional conversion functions

---

## Data Structure Design

### Enhanced Vector Types

#### Backend (Python)

```python
from dataclasses import dataclass
from enum import Enum
from typing import Generic, TypeVar, Optional

class CoordinateSpace(Enum):
    """Enumeration of coordinate spaces in the system."""
    CAMERA_NATIVE = "camera_native"      # Native camera resolution (1920x1080)
    WORLD_METERS = "world_meters"        # Real-world metric coordinates
    TABLE_NORMALIZED = "table_normalized"  # Normalized [0,1] table coordinates
    DETECTION_SPACE = "detection_space"  # Model-specific detection resolution

@dataclass
class Resolution:
    """Resolution metadata for pixel-based coordinate spaces."""
    width: int
    height: int

    def __str__(self) -> str:
        return f"{self.width}x{self.height}"

    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height if self.height > 0 else 0.0

    def scale_to(self, target: 'Resolution') -> tuple[float, float]:
        """Calculate scale factors to convert to target resolution."""
        return (target.width / self.width, target.height / self.height)

@dataclass
class CoordinateMetadata:
    """Metadata describing a coordinate space."""
    space: CoordinateSpace
    resolution: Optional[Resolution] = None  # For pixel spaces
    reference_frame: Optional[str] = None    # For tracking source
    timestamp: Optional[float] = None        # For temporal tracking

    def __post_init__(self):
        # Validate that pixel spaces have resolution
        if self.space != CoordinateSpace.WORLD_METERS and self.resolution is None:
            raise ValueError(f"{self.space} requires resolution metadata")

@dataclass
class Vector2D:
    """2D vector with explicit coordinate space metadata."""
    x: float
    y: float
    metadata: CoordinateMetadata

    def magnitude(self) -> float:
        """Calculate vector magnitude."""
        return math.sqrt(self.x**2 + self.y**2)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "x": self.x,
            "y": self.y,
            "space": self.metadata.space.value,
            "resolution": {
                "width": self.metadata.resolution.width,
                "height": self.metadata.resolution.height
            } if self.metadata.resolution else None,
            "reference_frame": self.metadata.reference_frame,
            "timestamp": self.metadata.timestamp
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'Vector2D':
        """Deserialize from dictionary."""
        resolution = None
        if data.get("resolution"):
            resolution = Resolution(**data["resolution"])

        metadata = CoordinateMetadata(
            space=CoordinateSpace(data["space"]),
            resolution=resolution,
            reference_frame=data.get("reference_frame"),
            timestamp=data.get("timestamp")
        )

        return cls(x=data["x"], y=data["y"], metadata=metadata)

    # Factory methods for common coordinate spaces

    @classmethod
    def camera_native(cls, x: float, y: float, resolution: Resolution, **kwargs) -> 'Vector2D':
        """Create vector in camera native coordinates."""
        metadata = CoordinateMetadata(
            space=CoordinateSpace.CAMERA_NATIVE,
            resolution=resolution,
            **kwargs
        )
        return cls(x=x, y=y, metadata=metadata)

    @classmethod
    def world_meters(cls, x: float, y: float, **kwargs) -> 'Vector2D':
        """Create vector in world metric coordinates."""
        metadata = CoordinateMetadata(
            space=CoordinateSpace.WORLD_METERS,
            resolution=None,
            **kwargs
        )
        return cls(x=x, y=y, metadata=metadata)

    @classmethod
    def table_normalized(cls, x: float, y: float, table_resolution: Resolution, **kwargs) -> 'Vector2D':
        """Create vector in normalized table coordinates [0,1]."""
        metadata = CoordinateMetadata(
            space=CoordinateSpace.TABLE_NORMALIZED,
            resolution=table_resolution,
            **kwargs
        )
        return cls(x=x, y=y, metadata=metadata)

# Backward compatibility - legacy Vector2D without metadata
@dataclass
class LegacyVector2D:
    """Legacy vector without coordinate metadata (DEPRECATED)."""
    x: float
    y: float

    def to_vector2d(self, metadata: CoordinateMetadata) -> Vector2D:
        """Upgrade to new Vector2D with explicit metadata."""
        return Vector2D(x=self.x, y=self.y, metadata=metadata)
```

#### Frontend (TypeScript)

```typescript
/**
 * Coordinate space enumeration
 */
export enum CoordinateSpace {
  CameraNative = 'camera_native',
  WorldMeters = 'world_meters',
  TableNormalized = 'table_normalized',
  DetectionSpace = 'detection_space',
  CanvasPixels = 'canvas_pixels',
}

/**
 * Resolution metadata for pixel-based coordinate spaces
 */
export interface Resolution {
  width: number;
  height: number;
}

/**
 * Metadata describing a coordinate space
 */
export interface CoordinateMetadata {
  space: CoordinateSpace;
  resolution?: Resolution;
  referenceFrame?: string;
  timestamp?: number;
}

/**
 * 2D vector with explicit coordinate space metadata
 */
export interface Vector2D {
  x: number;
  y: number;
  metadata: CoordinateMetadata;
}

/**
 * Type-safe vector constructor
 */
export class Vector2DBuilder {
  static cameraNative(x: number, y: number, resolution: Resolution): Vector2D {
    return {
      x,
      y,
      metadata: {
        space: CoordinateSpace.CameraNative,
        resolution,
        timestamp: Date.now(),
      },
    };
  }

  static worldMeters(x: number, y: number): Vector2D {
    return {
      x,
      y,
      metadata: {
        space: CoordinateSpace.WorldMeters,
        timestamp: Date.now(),
      },
    };
  }

  static tableNormalized(x: number, y: number, tableResolution: Resolution): Vector2D {
    return {
      x,
      y,
      metadata: {
        space: CoordinateSpace.TableNormalized,
        resolution: tableResolution,
        timestamp: Date.now(),
      },
    };
  }

  static canvasPixels(x: number, y: number, canvasSize: Resolution): Vector2D {
    return {
      x,
      y,
      metadata: {
        space: CoordinateSpace.CanvasPixels,
        resolution: canvasSize,
        timestamp: Date.now(),
      },
    };
  }
}

/**
 * Legacy point type without metadata (for backward compatibility)
 */
export interface Point2D {
  x: number;
  y: number;
}

/**
 * Convert legacy Point2D to Vector2D by adding metadata
 */
export function upgradePoint2D(
  point: Point2D,
  metadata: CoordinateMetadata
): Vector2D {
  return {
    ...point,
    metadata,
  };
}
```

---

## Coordinate System Standards

### Canonical Coordinate System: World Meters

**Recommendation**: Use **World Meters** as the canonical coordinate system for all persistent storage.

**Rationale**:
1. **Resolution Independent** - Not tied to any camera or display resolution
2. **Physically Meaningful** - Directly represents real-world table dimensions
3. **Physics Engine Native** - Core physics already uses meters
4. **Standard Table Dimensions** - 9ft table = 2.54m × 1.27m (well-defined)
5. **Calibration Natural** - Calibration establishes pixel↔meter mapping

**Storage Standard**:
- All `BallState.position` in meters
- All `TableState` dimensions in meters
- All `CueState.tip_position` in meters
- All trajectory points in meters

### Secondary Standards

#### Camera Native Resolution: 1920x1080

**Usage**: Source coordinates from camera, calibration reference
**Conversion**: Via calibrated `pixels_per_meter` factor

```python
# Standard camera resolution
CAMERA_NATIVE_RESOLUTION = Resolution(width=1920, height=1080)

# Calibration establishes mapping
pixels_per_meter: float  # e.g., 754.0 pixels/meter
```

#### Table Normalized Coordinates: [0, 1]

**Usage**: Internal calculations, relative positioning
**Benefits**:
- Resolution independent
- Easy boundary checking
- Natural for interpolation

```python
def to_normalized(pos: Vector2D, table: TableState) -> Vector2D:
    """Convert world meters to normalized table coordinates [0,1]."""
    return Vector2D.table_normalized(
        x=pos.x / table.width,
        y=pos.y / table.height,
        table_resolution=Resolution(width=1000, height=1000)  # Conceptual
    )
```

### Coordinate System Hierarchy

```
┌─────────────────────────────────────────────────────────────┐
│                    CANONICAL: World Meters                   │
│                     (2.54m × 1.27m)                         │
│                  • Physics Engine                            │
│                  • Storage                                   │
│                  • Inter-module Exchange                     │
└─────────────────────────────────────────────────────────────┘
                              ▲
                              │
                ┌─────────────┴─────────────┐
                │                           │
    ┌───────────▼──────────┐    ┌──────────▼───────────┐
    │  Camera Native       │    │  Table Normalized    │
    │  (1920x1080 px)      │    │     [0, 1]           │
    │  • Calibration       │    │  • Relative Pos      │
    │  • Detection Input   │    │  • Interpolation     │
    └──────────────────────┘    └──────────────────────┘
                │                           │
        ┌───────┴─────┬────────────────────┘
        │             │
┌───────▼──────┐  ┌──▼──────────────┐
│ Detection    │  │  Frontend        │
│ (Variable)   │  │  Canvas          │
│ • YOLO       │  │  (Variable px)   │
│ • OpenCV     │  │  • Display       │
└──────────────┘  └──────────────────┘
```

---

## Conversion Strategy

### Core Conversion Utilities

```python
from typing import Protocol

class CoordinateConverter(Protocol):
    """Protocol for coordinate converters."""

    def convert(self, vector: Vector2D, target_space: CoordinateSpace,
                target_resolution: Optional[Resolution] = None) -> Vector2D:
        """Convert vector to target coordinate space."""
        ...

class StandardCoordinateConverter:
    """Standard implementation of coordinate conversion."""

    def __init__(self, calibration: GeometricCalibration):
        self.calibration = calibration
        self.pixels_per_meter = calibration.coordinate_mapping.scale_factor
        self.camera_resolution = Resolution(width=1920, height=1080)

    def convert(self, vector: Vector2D, target_space: CoordinateSpace,
                target_resolution: Optional[Resolution] = None) -> Vector2D:
        """
        Convert vector to target coordinate space.

        Conversion paths:
        1. Any → World Meters (canonical) → Target
        2. Direct conversion if available (optimization)
        """
        # If already in target space, validate resolution and return
        if vector.metadata.space == target_space:
            if target_resolution and vector.metadata.resolution != target_resolution:
                return self._scale_resolution(vector, target_resolution)
            return vector

        # Convert to canonical (world meters) first
        world_vector = self._to_world_meters(vector)

        # Convert from canonical to target
        return self._from_world_meters(world_vector, target_space, target_resolution)

    def _to_world_meters(self, vector: Vector2D) -> Vector2D:
        """Convert any coordinate to world meters."""
        if vector.metadata.space == CoordinateSpace.WORLD_METERS:
            return vector

        elif vector.metadata.space == CoordinateSpace.CAMERA_NATIVE:
            # Pixel → Meters conversion using calibration
            if not vector.metadata.resolution:
                raise ValueError("Camera native coordinates require resolution")

            # Apply calibration transform
            world_x = vector.x / self.pixels_per_meter
            world_y = vector.y / self.pixels_per_meter

            return Vector2D.world_meters(x=world_x, y=world_y)

        elif vector.metadata.space == CoordinateSpace.TABLE_NORMALIZED:
            # Normalized [0,1] → Meters
            table_width = 2.54  # meters
            table_height = 1.27  # meters

            return Vector2D.world_meters(
                x=vector.x * table_width,
                y=vector.y * table_height
            )

        elif vector.metadata.space == CoordinateSpace.DETECTION_SPACE:
            # First scale to camera native, then to world
            camera_native = self._scale_resolution(
                vector, self.camera_resolution
            )
            return self._to_world_meters(camera_native)

        else:
            raise ValueError(f"Unknown coordinate space: {vector.metadata.space}")

    def _from_world_meters(self, world_vector: Vector2D,
                          target_space: CoordinateSpace,
                          target_resolution: Optional[Resolution]) -> Vector2D:
        """Convert from world meters to target space."""
        if target_space == CoordinateSpace.WORLD_METERS:
            return world_vector

        elif target_space == CoordinateSpace.CAMERA_NATIVE:
            if target_resolution is None:
                target_resolution = self.camera_resolution

            pixel_x = world_vector.x * self.pixels_per_meter
            pixel_y = world_vector.y * self.pixels_per_meter

            return Vector2D.camera_native(
                x=pixel_x, y=pixel_y, resolution=target_resolution
            )

        elif target_space == CoordinateSpace.TABLE_NORMALIZED:
            table_width = 2.54
            table_height = 1.27

            if target_resolution is None:
                target_resolution = Resolution(width=1000, height=1000)

            return Vector2D.table_normalized(
                x=world_vector.x / table_width,
                y=world_vector.y / table_height,
                table_resolution=target_resolution
            )

        elif target_space == CoordinateSpace.DETECTION_SPACE:
            if target_resolution is None:
                raise ValueError("Detection space requires target resolution")

            # World → Camera Native → Detection Resolution
            camera_vector = self._from_world_meters(
                world_vector, CoordinateSpace.CAMERA_NATIVE, self.camera_resolution
            )
            return self._scale_resolution(camera_vector, target_resolution)

        else:
            raise ValueError(f"Unknown target space: {target_space}")

    def _scale_resolution(self, vector: Vector2D,
                         target_resolution: Resolution) -> Vector2D:
        """Scale vector from one pixel resolution to another."""
        if not vector.metadata.resolution:
            raise ValueError("Cannot scale vector without source resolution")

        source_res = vector.metadata.resolution
        scale_x, scale_y = source_res.scale_to(target_resolution)

        return Vector2D(
            x=vector.x * scale_x,
            y=vector.y * scale_y,
            metadata=CoordinateMetadata(
                space=vector.metadata.space,
                resolution=target_resolution,
                reference_frame=vector.metadata.reference_frame,
                timestamp=vector.metadata.timestamp
            )
        )
```

### Frontend Conversion Utilities

```typescript
/**
 * Coordinate converter for frontend
 */
export class CoordinateConverter {
  private cameraResolution: Resolution = { width: 1920, height: 1080 };
  private pixelsPerMeter: number;

  constructor(calibration: CalibrationData) {
    this.pixelsPerMeter = calibration.pixelsPerMeter || 754.0;
  }

  /**
   * Convert vector to target coordinate space
   */
  convert(
    vector: Vector2D,
    targetSpace: CoordinateSpace,
    targetResolution?: Resolution
  ): Vector2D {
    // Convert to world meters (canonical) first
    const worldVector = this.toWorldMeters(vector);

    // Convert from world meters to target
    return this.fromWorldMeters(worldVector, targetSpace, targetResolution);
  }

  private toWorldMeters(vector: Vector2D): Vector2D {
    if (vector.metadata.space === CoordinateSpace.WorldMeters) {
      return vector;
    }

    if (vector.metadata.space === CoordinateSpace.CameraNative) {
      return Vector2DBuilder.worldMeters(
        vector.x / this.pixelsPerMeter,
        vector.y / this.pixelsPerMeter
      );
    }

    if (vector.metadata.space === CoordinateSpace.TableNormalized) {
      const tableWidth = 2.54; // meters
      const tableHeight = 1.27;
      return Vector2DBuilder.worldMeters(
        vector.x * tableWidth,
        vector.y * tableHeight
      );
    }

    if (vector.metadata.space === CoordinateSpace.CanvasPixels) {
      // Canvas → Camera Native → World Meters
      const cameraVector = this.scaleResolution(vector, this.cameraResolution);
      return this.toWorldMeters(cameraVector);
    }

    throw new Error(`Unknown coordinate space: ${vector.metadata.space}`);
  }

  private fromWorldMeters(
    worldVector: Vector2D,
    targetSpace: CoordinateSpace,
    targetResolution?: Resolution
  ): Vector2D {
    if (targetSpace === CoordinateSpace.WorldMeters) {
      return worldVector;
    }

    if (targetSpace === CoordinateSpace.CameraNative) {
      const resolution = targetResolution || this.cameraResolution;
      return Vector2DBuilder.cameraNative(
        worldVector.x * this.pixelsPerMeter,
        worldVector.y * this.pixelsPerMeter,
        resolution
      );
    }

    if (targetSpace === CoordinateSpace.TableNormalized) {
      const tableWidth = 2.54;
      const tableHeight = 1.27;
      const resolution = targetResolution || { width: 1000, height: 1000 };
      return Vector2DBuilder.tableNormalized(
        worldVector.x / tableWidth,
        worldVector.y / tableHeight,
        resolution
      );
    }

    if (targetSpace === CoordinateSpace.CanvasPixels) {
      if (!targetResolution) {
        throw new Error('Canvas pixels requires target resolution');
      }
      // World → Camera Native → Canvas
      const cameraVector = this.fromWorldMeters(
        worldVector,
        CoordinateSpace.CameraNative,
        this.cameraResolution
      );
      return this.scaleResolution(cameraVector, targetResolution);
    }

    throw new Error(`Unknown target space: ${targetSpace}`);
  }

  private scaleResolution(vector: Vector2D, targetResolution: Resolution): Vector2D {
    if (!vector.metadata.resolution) {
      throw new Error('Cannot scale vector without source resolution');
    }

    const scaleX = targetResolution.width / vector.metadata.resolution.width;
    const scaleY = targetResolution.height / vector.metadata.resolution.height;

    return {
      x: vector.x * scaleX,
      y: vector.y * scaleY,
      metadata: {
        ...vector.metadata,
        resolution: targetResolution,
      },
    };
  }
}
```

---

## Migration Plan

### Phase 1: Foundation (Week 1-2)

**Goal**: Introduce new types without breaking existing code

**Tasks**:
1. ✅ Create new `Vector2D` class with metadata in `backend/core/models.py`
2. ✅ Rename existing `Vector2D` → `LegacyVector2D`
3. ✅ Create `CoordinateConverter` utility class
4. ✅ Add type aliases for backward compatibility
5. ✅ Write comprehensive unit tests for conversion logic
6. ✅ Create frontend TypeScript equivalents in `frontend/web/src/types/coordinates.ts`

**Acceptance Criteria**:
- All existing tests pass
- New conversion utilities tested with >95% coverage
- Documentation updated with examples

### Phase 2: Core Module Migration (Week 3-4)

**Goal**: Update core physics and game state to use new Vector2D

**Tasks**:
1. Update `BallState` to use metadata-aware `Vector2D`
2. Update `TableState` with explicit resolution for `playing_area_corners`
3. Update `CueState` positions with metadata
4. Modify all physics calculations to validate coordinate spaces
5. Add conversion helpers at module boundaries

**Migration Strategy**:
```python
# Old code
position = Vector2D(1.0, 0.5)

# New code
position = Vector2D.world_meters(1.0, 0.5, reference_frame="physics_engine")

# Transition helper
def upgrade_legacy_vector(legacy: LegacyVector2D,
                         default_space: CoordinateSpace) -> Vector2D:
    metadata = CoordinateMetadata(
        space=default_space,
        resolution=CAMERA_NATIVE_RESOLUTION if default_space != CoordinateSpace.WORLD_METERS else None
    )
    return legacy.to_vector2d(metadata)
```

**Acceptance Criteria**:
- All core physics tests pass
- State serialization/deserialization works
- Performance benchmarks show <5% overhead

### Phase 3: Vision Module Migration (Week 5-6)

**Goal**: Vision detection outputs include resolution metadata

**Tasks**:
1. Update `Ball` model in `backend/vision/models.py` to include resolution
2. Update ball detection to tag outputs with detection resolution
3. Update YOLO detector to report model input resolution
4. Modify calibration to expose `pixels_per_meter` and camera resolution
5. Update state conversion helpers to use new Vector2D

**Key Changes**:
```python
# backend/vision/models.py
@dataclass
class Ball:
    position: tuple[float, float]  # Still tuple for simplicity
    radius: float
    # NEW: Add detection metadata
    detection_resolution: Resolution = field(default_factory=lambda: Resolution(1920, 1080))
    coordinate_space: CoordinateSpace = CoordinateSpace.CAMERA_NATIVE

# backend/integration_service_conversion_helpers.py
def vision_ball_to_ball_state(ball: Ball, converter: CoordinateConverter) -> BallState:
    # Create vision vector with metadata
    vision_pos = Vector2D.camera_native(
        x=ball.position[0],
        y=ball.position[1],
        resolution=ball.detection_resolution,
        reference_frame="vision_detection"
    )

    # Convert to canonical world meters
    world_pos = converter.convert(vision_pos, CoordinateSpace.WORLD_METERS)

    return BallState(
        position=world_pos,  # Now has metadata!
        ...
    )
```

**Acceptance Criteria**:
- Vision detection tests pass
- Integration tests show correct conversion
- Calibration exports include resolution info

### Phase 4: API & WebSocket Migration (Week 7-8)

**Goal**: API responses include coordinate metadata

**Tasks**:
1. Update API models to include coordinate metadata
2. Modify WebSocket messages to include resolution info
3. Update API converters to preserve metadata
4. Add validation middleware to check coordinate space consistency

**API Changes**:
```python
# backend/api/models/responses.py
from backend.core.models import Vector2D, CoordinateSpace, Resolution

class BallInfo(BaseModel):
    id: str
    number: Optional[int]
    position: list[float]  # [x, y]
    velocity: list[float]  # [vx, vy]
    # NEW: Add coordinate metadata
    coordinate_space: str  # e.g., "world_meters"
    resolution: Optional[dict[str, int]]  # {"width": 1920, "height": 1080}

    @classmethod
    def from_ball_state(cls, ball: BallState) -> 'BallInfo':
        return cls(
            id=ball.id,
            number=ball.number,
            position=[ball.position.x, ball.position.y],
            velocity=[ball.velocity.x, ball.velocity.y],
            coordinate_space=ball.position.metadata.space.value,
            resolution={
                "width": ball.position.metadata.resolution.width,
                "height": ball.position.metadata.resolution.height
            } if ball.position.metadata.resolution else None
        )
```

**Acceptance Criteria**:
- API responses include coordinate metadata
- WebSocket clients can determine coordinate space
- API documentation updated

### Phase 5: Frontend Migration (Week 9-10)

**Goal**: Frontend uses coordinate-aware types

**Tasks**:
1. Update all component props to use `Vector2D` type
2. Modify overlay components to convert coordinates explicitly
3. Update stores to handle metadata
4. Add coordinate space validation in rendering

**Frontend Changes**:
```typescript
// frontend/web/src/components/video/overlays/BallOverlay.tsx
import { CoordinateConverter } from '@/utils/coordinates';

interface BallOverlayProps {
  balls: Vector2D[];  // Now with metadata!
  canvasSize: Resolution;
  converter: CoordinateConverter;
}

export function BallOverlay({ balls, canvasSize, converter }: BallOverlayProps) {
  return balls.map(ball => {
    // Validate coordinate space
    if (ball.metadata.space !== CoordinateSpace.WorldMeters) {
      console.warn('Unexpected coordinate space:', ball.metadata.space);
    }

    // Convert to canvas pixels for rendering
    const canvasPos = converter.convert(
      ball,
      CoordinateSpace.CanvasPixels,
      canvasSize
    );

    return <circle cx={canvasPos.x} cy={canvasPos.y} r={5} />;
  });
}
```

**Acceptance Criteria**:
- All overlays render correctly
- Coordinate transformations validated
- Performance acceptable (60fps)

### Phase 6: Configuration & Documentation (Week 11)

**Goal**: Update configuration and provide migration guides

**Tasks**:
1. Update `config.json` with resolution metadata
2. Add resolution to calibration files
3. Write migration guide for developers
4. Create troubleshooting guide for coordinate issues
5. Add logging/debugging tools for coordinate transformations

**Config Updates**:
```json
{
  "vision": {
    "camera": {
      "resolution": {"width": 1920, "height": 1080},
      "coordinate_space": "camera_native"
    }
  },
  "table": {
    "dimensions": {"width": 2.54, "height": 1.27},
    "coordinate_space": "world_meters",
    "playing_area_corners": [
      {
        "x": 37,
        "y": 45,
        "space": "camera_native",
        "resolution": {"width": 1920, "height": 1080}
      },
      ...
    ]
  },
  "calibration": {
    "pixels_per_meter": 754.0,
    "camera_resolution": {"width": 1920, "height": 1080},
    "reference_space": "camera_native"
  }
}
```

**Acceptance Criteria**:
- All configuration validated
- Documentation complete
- Migration guide reviewed

### Phase 7: Testing & Validation (Week 12)

**Goal**: Comprehensive testing and validation

**Tasks**:
1. End-to-end integration tests
2. Performance benchmarks
3. Visual validation of rendering
4. Stress testing with different resolutions
5. Backward compatibility verification

**Test Scenarios**:
- [ ] Ball detected at 640x640 → Rendered correctly on 1920x1080 canvas
- [ ] Calibration at 1920x1080 → Coordinates convert correctly
- [ ] Physics calculations in meters → Display in pixels works
- [ ] Frontend canvas resize → Coordinates update correctly
- [ ] Multiple coordinate transformations → No loss of precision

**Acceptance Criteria**:
- All tests pass
- Performance within 5% of baseline
- No visual glitches
- Clean deprecation warnings

---

## Performance Considerations

### Overhead Analysis

**Additional Memory per Vector**:
- `CoordinateMetadata`: ~48 bytes
  - `space`: 4 bytes (enum)
  - `resolution`: 16 bytes (2 ints + overhead)
  - `reference_frame`: 16 bytes (str pointer)
  - `timestamp`: 8 bytes (float)
  - Object overhead: ~4 bytes

**Impact**: For 16 balls × 3 vectors each = 48 vectors = ~2.3KB extra per frame

**Mitigation**:
1. Use object pooling for frequently created vectors
2. Lazy metadata - only add when necessary
3. Strip metadata for internal calculations
4. Use type variants for different use cases

### Optimization Strategies

#### 1. Fast Path for Common Cases

```python
class OptimizedConverter:
    # Cache common conversions
    _cache: dict[tuple[CoordinateSpace, CoordinateSpace], Callable] = {}

    def convert_fast(self, vector: Vector2D, target: CoordinateSpace) -> Vector2D:
        """Fast path for common conversions."""
        key = (vector.metadata.space, target)

        if key in self._cache:
            return self._cache[key](vector)

        # Fall back to full conversion
        return self.convert(vector, target)
```

#### 2. Batch Conversions

```python
def convert_batch(vectors: list[Vector2D],
                 target_space: CoordinateSpace) -> list[Vector2D]:
    """Convert multiple vectors efficiently."""
    if not vectors:
        return []

    # Group by source space
    by_space: dict[CoordinateSpace, list[Vector2D]] = {}
    for v in vectors:
        by_space.setdefault(v.metadata.space, []).append(v)

    # Convert each group
    result = []
    for space, group in by_space.items():
        # Single metadata check per group
        result.extend(self._convert_group(group, target_space))

    return result
```

#### 3. Lazy Metadata Attachment

```python
@dataclass
class LazyVector2D:
    """Vector with optional metadata for performance-critical paths."""
    x: float
    y: float
    _metadata: Optional[CoordinateMetadata] = None

    @property
    def metadata(self) -> CoordinateMetadata:
        if self._metadata is None:
            # Default to world meters for internal calculations
            self._metadata = CoordinateMetadata(
                space=CoordinateSpace.WORLD_METERS
            )
        return self._metadata
```

### Benchmarks

Expected performance targets:

| Operation | Current | With Metadata | Overhead |
|-----------|---------|---------------|----------|
| Vector creation | 50 ns | 75 ns | +50% |
| Vector addition | 20 ns | 20 ns | 0% |
| Coordinate conversion | - | 200 ns | N/A |
| Batch conversion (100) | - | 5 μs | N/A |
| Frame processing (16 balls) | 10 ms | 10.1 ms | +1% |

**Conclusion**: Overhead is minimal and acceptable for the gains in correctness.

---

## Implementation Roadmap

### Summary Timeline

| Phase | Duration | Key Deliverable |
|-------|----------|----------------|
| 1. Foundation | 2 weeks | New types & converters |
| 2. Core Migration | 2 weeks | Physics in new format |
| 3. Vision Migration | 2 weeks | Detection metadata |
| 4. API Migration | 2 weeks | API responses updated |
| 5. Frontend Migration | 2 weeks | UI coordinate-aware |
| 6. Config & Docs | 1 week | Documentation complete |
| 7. Testing & Validation | 1 week | Full system validated |
| **Total** | **12 weeks** | **Complete Migration** |

### Risk Mitigation

**Risk**: Breaking changes impact existing clients
**Mitigation**:
- Maintain backward-compatible API layer
- Gradual rollout with feature flags
- Deprecation warnings before removal

**Risk**: Performance degradation
**Mitigation**:
- Benchmark at each phase
- Optimize hot paths
- Use lazy evaluation

**Risk**: Migration complexity
**Mitigation**:
- Comprehensive test coverage
- Automated migration tools
- Pair programming for critical sections

---

## Appendix

### A. Example Usage

#### Backend Example

```python
from backend.core.models import Vector2D, CoordinateSpace, Resolution
from backend.vision.calibration.geometry import GeometricCalibrator

# Initialize calibrator and converter
calibrator = GeometricCalibrator()
calibration = calibrator.load_calibration()
converter = StandardCoordinateConverter(calibration)

# Camera detects ball at pixel position
camera_resolution = Resolution(width=1920, height=1080)
detected_position = Vector2D.camera_native(
    x=960.0,
    y=540.0,
    resolution=camera_resolution,
    reference_frame="yolo_detector"
)

# Convert to world meters for physics
world_position = converter.convert(
    detected_position,
    CoordinateSpace.WORLD_METERS
)

# Create ball state
ball = BallState(
    id="cue",
    position=world_position,  # In meters!
    velocity=Vector2D.world_meters(1.5, 0.0),  # 1.5 m/s
    is_cue_ball=True
)

# Physics engine uses meters natively
trajectory = physics_engine.calculate_trajectory(ball)

# Convert back to pixels for API response
pixel_positions = [
    converter.convert(point, CoordinateSpace.CAMERA_NATIVE, camera_resolution)
    for point in trajectory.points
]
```

#### Frontend Example

```typescript
import { Vector2DBuilder, CoordinateConverter, CoordinateSpace } from '@/utils/coordinates';

// Receive ball from API with metadata
const apiResponse = {
  balls: [{
    id: 'cue',
    position: [1.27, 0.635],  // meters
    coordinateSpace: 'world_meters'
  }]
};

// Convert to typed vector
const ballPosition = Vector2DBuilder.worldMeters(
  apiResponse.balls[0].position[0],
  apiResponse.balls[0].position[1]
);

// Get canvas size
const canvasSize = { width: 800, height: 450 };

// Convert to canvas pixels for rendering
const converter = new CoordinateConverter(calibrationData);
const canvasPosition = converter.convert(
  ballPosition,
  CoordinateSpace.CanvasPixels,
  canvasSize
);

// Render on canvas
ctx.beginPath();
ctx.arc(canvasPosition.x, canvasPosition.y, 5, 0, Math.PI * 2);
ctx.fill();
```

### B. Coordinate Space Decision Tree

```
Need to store position?
├─ Yes: Use World Meters (canonical)
└─ No: Processing only?
   ├─ Rendering on canvas? → Canvas Pixels
   ├─ Sending to YOLO? → Detection Space (model specific)
   ├─ Camera calibration? → Camera Native
   └─ Relative positioning? → Table Normalized
```

### C. Common Pitfalls

1. **Mixing coordinate spaces without conversion**
   ```python
   # ❌ WRONG
   camera_pos = Vector2D(100, 50)  # pixels
   world_pos = Vector2D(1.0, 0.5)  # meters
   distance = camera_pos.distance_to(world_pos)  # NONSENSE!

   # ✅ CORRECT
   camera_pos = Vector2D.camera_native(100, 50, CAMERA_RES)
   world_pos = Vector2D.world_meters(1.0, 0.5)
   world_camera = converter.convert(camera_pos, CoordinateSpace.WORLD_METERS)
   distance = world_camera.distance_to(world_pos)  # Correct!
   ```

2. **Forgetting resolution when scaling**
   ```python
   # ❌ WRONG
   def scale_position(pos, scale):
       return Vector2D(pos.x * scale, pos.y * scale)

   # ✅ CORRECT
   def scale_position(pos: Vector2D, target_res: Resolution,
                     converter: CoordinateConverter) -> Vector2D:
       # Convert to canonical, then to target resolution
       world = converter.convert(pos, CoordinateSpace.WORLD_METERS)
       return converter.convert(world, pos.metadata.space, target_res)
   ```

3. **Assuming all pixels are equal**
   ```python
   # ❌ WRONG: Different resolutions!
   yolo_ball = Ball(position=(320, 320))  # 640x640 YOLO output
   camera_ball = Ball(position=(320, 320))  # 1920x1080 camera
   # These are NOT the same location!

   # ✅ CORRECT
   yolo_pos = Vector2D(320, 320, metadata=...)  # 640x640
   camera_pos = converter.convert(
       yolo_pos,
       CoordinateSpace.CAMERA_NATIVE,
       Resolution(1920, 1080)
   )
   ```

### D. Testing Checklist

- [ ] Unit tests for Vector2D creation
- [ ] Unit tests for each coordinate conversion
- [ ] Unit tests for resolution scaling
- [ ] Integration test: Vision → Core conversion
- [ ] Integration test: Core → API conversion
- [ ] Integration test: API → Frontend conversion
- [ ] Integration test: Round-trip conversion (no loss)
- [ ] Performance test: Conversion overhead < 5%
- [ ] Performance test: Batch conversions optimized
- [ ] Visual test: Ball positions render correctly
- [ ] Visual test: Trajectories align with balls
- [ ] Visual test: Calibration grid accurate
- [ ] Edge case: Zero resolution
- [ ] Edge case: Very large coordinates
- [ ] Edge case: Negative coordinates
- [ ] Edge case: Multiple coordinate systems in one frame

---

## Conclusion

This resolution standardization plan provides:

1. **Explicit coordinate system metadata** - No more guessing what coordinates mean
2. **Type safety** - Prevent mixing coordinate spaces at compile/runtime
3. **Canonical storage format** - World Meters for all persistent data
4. **Clear conversion paths** - Well-tested, bidirectional conversions
5. **Backward compatibility** - Gradual migration without breaking existing code
6. **Performance conscious** - Minimal overhead with optimization strategies
7. **Comprehensive migration plan** - 12-week roadmap with clear phases

**Next Steps**:
1. Review and approve this plan
2. Create implementation tickets for Phase 1
3. Set up benchmarking infrastructure
4. Begin Foundation phase (Week 1-2)

**Success Metrics**:
- Zero coordinate-related bugs in production
- Sub-5% performance overhead
- 100% test coverage for conversions
- Clear, maintainable code with type safety

---

**Document End**
