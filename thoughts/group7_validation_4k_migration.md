# Group 7: Validation to 4K - Migration Complete

**Status**: ✅ **COMPLETE**
**Date**: 2025-10-21

## Summary

Successfully migrated all validation modules to use 4K pixel coordinates and converted all velocity limits from meters/second to pixels/second.

## Files Updated

### 1. `/backend/core/validation/state.py`
**Changes**:
- ✅ Added imports from `constants_4k` (TABLE_WIDTH_4K, TABLE_HEIGHT_4K, BALL_RADIUS_4K, POCKET_RADIUS_4K)
- ✅ Added `PIXELS_PER_METER` constant (~1259.84 pixels/meter)
- ✅ Converted `max_velocity` default: `10.0 * PIXELS_PER_METER` (~12,600 px/s)
- ✅ Converted `max_acceleration` default: `50.0 * PIXELS_PER_METER` (~63,000 px/s²)
- ✅ Updated overlap_tolerance: `1.0` pixels (was 0.001 meters)
- ✅ Updated position_tolerance: `0.1` pixels (was 0.0001 meters)
- ✅ Added 4K table dimension validation against TABLE_WIDTH_4K and TABLE_HEIGHT_4K
- ✅ Added ball radius validation against BALL_RADIUS_4K
- ✅ Updated error messages to use "px/s" instead of "m/s"
- ✅ Updated docstrings to specify "4K pixel coordinates"

**Key Validations**:
- Table dimensions should match 4K standards (3200×1600 pixels)
- Ball radius should match BALL_RADIUS_4K (36 pixels)
- Velocity limits are now in pixels/s
- All spatial tolerances are in pixels

### 2. `/backend/core/validation/physics.py`
**Changes**:
- ✅ Added imports from `constants_4k` (TABLE_WIDTH_4K, TABLE_HEIGHT_4K, BALL_RADIUS_4K, POCKET_RADIUS_4K)
- ✅ Added `PIXELS_PER_METER` constant (~1259.84 pixels/meter)
- ✅ Converted `max_velocity` default: `20.0 * PIXELS_PER_METER` (~25,200 px/s)
- ✅ Converted `max_acceleration` default: `100.0 * PIXELS_PER_METER` (~126,000 px/s²)
- ✅ Updated velocity_tolerance: `1.0` pixels/s (was 0.001 m/s)
- ✅ Updated position_tolerance: `1.0` pixels (was 0.001 m)
- ✅ Updated all error messages to use "px/s" and "px/s²"
- ✅ Updated ball radius suggested fix to use BALL_RADIUS_4K
- ✅ Updated docstrings to specify "4K pixel coordinates"

**Key Validations**:
- Trajectory velocity limits in pixels/s
- Ball state velocity limits in pixels/s
- Acceleration limits in pixels/s²
- Force validation (stays in SI units - Newtons)
- Spin validation (stays in rad/s - unitless)

### 3. `/backend/core/validation/correction.py`
**Changes**:
- ✅ Added imports from `constants_4k` (TABLE_WIDTH_4K, TABLE_HEIGHT_4K, BALL_RADIUS_4K, POCKET_RADIUS_4K)
- ✅ Added `PIXELS_PER_METER` constant (~1259.84 pixels/meter)
- ✅ Converted `max_velocity` default: `10.0 * PIXELS_PER_METER` (~12,600 px/s)
- ✅ Updated overlap tolerance: `-1.0` pixels (was -0.001 meters)
- ✅ Updated ball separation buffer: `1.0` pixels (was 0.001 meters)
- ✅ Updated table bounds margin: `1.0` pixels (was 0.001 meters)
- ✅ Updated invalid radius correction to use BALL_RADIUS_4K (36 pixels)
- ✅ Updated error messages to use "px/s" instead of "m/s"
- ✅ Updated docstrings to specify "4K pixel coordinates"

**Key Corrections**:
- Ball overlap separation uses pixel distances
- Out-of-bounds correction uses pixel coordinates
- Invalid velocity clamping uses pixels/s
- Invalid radius sets to BALL_RADIUS_4K

## Conversion Constants

All three files now define and use:

```python
# Physics constants for validation
# PIXELS_PER_METER is used to convert physical limits to 4K pixel scale
PIXELS_PER_METER = TABLE_WIDTH_4K / 2.54  # ~1259.84 pixels/meter (2.54m table width)
```

## Velocity Conversions Applied

| Parameter | Old Value (m/s) | New Value (px/s) | Formula |
|-----------|----------------|------------------|---------|
| StateValidator max_velocity | 10.0 m/s | ~12,600 px/s | 10.0 * PIXELS_PER_METER |
| StateValidator max_acceleration | 50.0 m/s² | ~63,000 px/s² | 50.0 * PIXELS_PER_METER |
| PhysicsValidator max_velocity | 20.0 m/s | ~25,200 px/s | 20.0 * PIXELS_PER_METER |
| PhysicsValidator max_acceleration | 100.0 m/s² | ~126,000 px/s² | 100.0 * PIXELS_PER_METER |
| ErrorCorrector max_velocity | 10.0 m/s | ~12,600 px/s | 10.0 * PIXELS_PER_METER |

## Tolerance Conversions Applied

| Parameter | Old Value (meters) | New Value (pixels) |
|-----------|-------------------|-------------------|
| overlap_tolerance | 0.001 m | 1.0 px |
| position_tolerance | 0.0001 m | 0.1 px |
| velocity_tolerance | 0.001 m/s | 1.0 px/s |

## 4K Standards Enforced

The validation modules now enforce:
- **Table Width**: 3200 pixels (TABLE_WIDTH_4K)
- **Table Height**: 1600 pixels (TABLE_HEIGHT_4K)
- **Ball Radius**: 36 pixels (BALL_RADIUS_4K)
- **Pocket Radius**: 72 pixels (POCKET_RADIUS_4K)

## Documentation Updates

All three files now have updated docstrings specifying:
- "All spatial measurements are in 4K pixels (3840×2160)"
- "Velocities are in pixels/second"
- "Accelerations are in pixels/second²" (physics.py)

## Testing Status

✅ **Syntax Validation**: All files have valid Python syntax
✅ **Import Structure**: All constants_4k imports are correct
✅ **Conversion Applied**: All velocity and spatial conversions use PIXELS_PER_METER
✅ **Documentation**: All docstrings updated to reflect 4K pixel coordinates

**Note**: Full integration tests cannot run yet due to Group 3 (Core Models) still being in progress. The error `ImportError: cannot import name 'CoordinateSpace'` is expected and is not related to these changes.

## Migration Verification

```bash
# All files have valid syntax
✓ state.py syntax is valid
✓ physics.py syntax is valid
✓ correction.py syntax is valid

# All files have required changes
✓ Imports constants_4k
✓ Defines PIXELS_PER_METER
✓ Uses pixels/s for velocity
✓ Mentions 4K pixels in docstrings
```

## Dependencies

These changes depend on:
- ✅ Group 1: constants_4k.py exists and defines all required constants
- ⏸️ Group 3: Core Models (not complete, but doesn't block validation changes)

## Next Steps

Once Group 3 (Core Models) is complete:
1. Run full integration tests
2. Verify validation works with migrated BallState/TableState
3. Test error correction with actual 4K coordinates

## Files Modified

1. `/backend/core/validation/state.py` - 7 edits
2. `/backend/core/validation/physics.py` - 6 edits
3. `/backend/core/validation/correction.py` - 7 edits

**Total**: 20 successful edits across 3 files

---

✅ **Group 7 Migration Complete**
