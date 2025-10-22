# 4K Migration - Executive Summary

**Status**: Planning Complete ✅
**Full Plan**: See `4k_standardization_plan.md`

---

## What We're Doing

Migrating from **4 coordinate systems** (WORLD_METERS, CAMERA_PIXELS, TABLE_PIXELS, NORMALIZED) to **one standardized 4K pixel system**.

## Key Changes

### Before (Current)
- **Storage**: World Meters (2.54m × 1.27m table)
- **Calibration**: pixels_per_meter = 754.0
- **Complexity**: 4 coordinate spaces, constant conversions
- **Vector2D**: `{x, y, coordinate_space?, resolution?}` (optional metadata)

### After (New)
- **Storage**: 4K Pixels (3200×1600 table in 3840×2160 frame)
- **Calibration**: Resolution scaling only
- **Complexity**: 1 canonical space (4K)
- **Vector2D**: `{x, y, scale: [sx, sy]}` (mandatory scale)

## Table Dimensions

| Measurement | Old (Meters) | New (4K Pixels) |
|-------------|--------------|-----------------|
| Table Width | 2.54m | 3200px |
| Table Height | 1.27m | 1600px |
| Table Center | (0, 0)m | (1920, 1080)px |
| Ball Radius | 0.028575m | 36px |
| Aspect Ratio | 2:1 | 2:1 |

## Scale Metadata

All Vector2D instances must have scale:

```python
# 4K canonical
Vector2D(x=1920, y=1080, scale=[1.0, 1.0])

# 1080p source (scales to 4K)
Vector2D(x=960, y=540, scale=[2.0, 2.0])
# To 4K: x_4k = 960 * 2.0 = 1920 ✓

# 720p source (scales to 4K)
Vector2D(x=640, y=360, scale=[3.0, 3.0])
# To 4K: x_4k = 640 * 3.0 = 1920 ✓
```

## Implementation Plan

### 10 Parallel Groups (4 weeks)

| Group | Owner | Duration | Complexity | Dependencies |
|-------|-------|----------|------------|--------------|
| 1. Foundation | Agent 1 | 2d | Low | None |
| 2. Vector2D | Agent 2 | 3d | Medium | Group 1 |
| 3. Core Models | Agent 3 | 4d | High | Group 2 |
| 4. Physics | Agent 4 | 5d | Very High | Group 3 |
| 5. Collision | Agent 5 | 3d | Medium | Group 4 |
| 6. Analysis | Agent 6 | 3d | Medium | Groups 4,5 |
| 7. Validation | Agent 7 | 2d | Low | Group 3 |
| 8. Vision | Agent 8 | 4d | Medium | Groups 2,3 |
| 9. API | Agent 9 | 3d | Medium | Group 3 |
| 10. Utils | Agent 10 | 2d | Low | All |

**Critical Path**: 1 → 2 → 3 → 4 → 5 → 6 (20 days)
**Parallel**: Groups 7, 8, 9, 10 run concurrently

### Weekly Schedule

```
Week 1: Foundation
  Day 1-2:  Group 1 (Constants, ResolutionConverter)
  Day 3-5:  Group 2 (Vector2D with mandatory scale)

Week 2: Core + Parallel
  Day 6-9:  Group 3 (BallState, TableState, GameState)
  Day 6-7:  Group 7 (Validation) [Parallel]
  Day 6-9:  Group 8 (Vision) [Parallel]

Week 3: Physics + API
  Day 10-14: Group 4 (Physics Engine)
  Day 10-12: Group 5 (Collision Detection)
  Day 13-15: Group 9 (API Models) [Parallel]

Week 4: Finalization
  Day 16-18: Group 6 (Analysis & Prediction)
  Day 16-17: Group 10 (Utilities)
  Day 18-20: Integration Testing
```

## Breaking Changes

### Vector2D
```python
# OLD (optional scale)
v = Vector2D(x=100, y=50)  # OK
v = Vector2D.world_meters(1.0, 0.5)  # OK

# NEW (mandatory scale)
v = Vector2D(x=100, y=50, scale=[2.0, 2.0])  # Required
v = Vector2D.from_4k(1920, 1080)  # Factory method
```

### BallState
```python
# OLD (meters)
BallState.create(id="ball", x=1.27, y=0.635)  # meters

# NEW (4K pixels)
BallState.from_4k(id="ball", x=1920, y=1080)  # pixels
```

### API Response
```json
// OLD (v1.0)
{
  "position": [0.5, 0.2],  // meters
  "coordinate_space": "world_meters"
}

// NEW (v2.0)
{
  "position": {
    "x": 2550.0,          // 4K pixels
    "y": 1332.0,
    "scale": [1.0, 1.0]   // MANDATORY
  }
}
```

## Files Modified

**Total**: ~50 files

**By Category**:
- Core: 12 files (models, physics, collision)
- Vision: 8 files (detection, calibration)
- API: 6 files (routes, models, converters)
- Tests: 15 files
- Integration: 9 files

## Removed Concepts

- ❌ `WORLD_METERS` coordinate space
- ❌ `pixels_per_meter` calibration factor
- ❌ Meter-based table dimensions
- ❌ `CoordinateSpace` enum (4 values → 0)
- ❌ `CoordinateConverter.camera_pixels_to_world_meters()`

## New Concepts

- ✅ `CANONICAL_4K` (3840×2160) - single coordinate space
- ✅ Mandatory `scale` metadata on all Vector2D
- ✅ `ResolutionConverter` (simple scaling)
- ✅ `Vector2D.to_4k_canonical()` - normalize to 4K
- ✅ `Vector2D.from_resolution()` - auto-calculate scale

## Verification Requirements

### Critical Tests
1. **Accuracy**: Physics results match old system (±0.1%)
2. **Performance**: No regression > 10%
3. **Conversion**: Round-trip accuracy < 1e-6
4. **Backward Compat**: Legacy data auto-migrates

### Coverage Targets
- New code: 100% coverage
- Modified code: 95%+ coverage
- Integration tests: Full pipeline
- Regression tests: 100+ scenarios

## Success Metrics

**Must Achieve**:
- ✅ Zero data loss
- ✅ Physics accuracy maintained
- ✅ Performance unchanged (±5%)
- ✅ All tests passing
- ✅ Backward compatibility working

## Rollback Plan

**Triggers**:
- Physics accuracy degraded > 1%
- Performance regression > 10%
- Critical production bugs
- Data corruption

**Procedure**:
1. Identify failing groups
2. Revert affected commits
3. Restore data from backup
4. Run legacy system tests
5. Verify integrity

## Quick Reference

### Common Operations

```python
# Create from 4K canonical
v = Vector2D.from_4k(x=1920, y=1080)
# → Vector2D(x=1920, y=1080, scale=[1.0, 1.0])

# Create from any resolution
v = Vector2D.from_resolution(x=960, y=540, resolution=(1920, 1080))
# → Vector2D(x=960, y=540, scale=[2.0, 2.0])

# Convert to 4K canonical
v_4k = v.to_4k_canonical()
# → Vector2D(x=1920, y=1080, scale=[1.0, 1.0])

# Convert to target resolution
v_720p = v_4k.to_resolution((1280, 720))
# → Vector2D(x=640, y=360, scale=[3.0, 3.0])
```

### Standard Constants

```python
from backend.core.constants_4k import (
    CANONICAL_RESOLUTION,     # (3840, 2160)
    TABLE_WIDTH_4K,           # 3200 pixels
    TABLE_HEIGHT_4K,          # 1600 pixels
    TABLE_CENTER_4K,          # (1920, 1080)
    BALL_RADIUS_4K,           # 36 pixels
    POCKET_RADIUS_4K,         # 144 pixels
)
```

## Next Steps

1. ✅ Review this plan
2. ⏳ Create subagent tasks for each group
3. ⏳ Begin Group 1 (Foundation)
4. ⏳ Monitor progress weekly
5. ⏳ Deploy after Week 4

---

**See Full Plan**: `4k_standardization_plan.md` (detailed specification)
**Status**: Ready for Implementation ✅
