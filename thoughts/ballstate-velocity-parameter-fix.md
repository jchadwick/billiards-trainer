# BallState.from_4k() Velocity Parameter Fix

**Date:** 2025-10-22
**Issue:** TypeError when creating BallState from vision detections
**Status:** ✅ Fixed

## Problem

The application was crashing with the following error:

```
TypeError: BallState.__init__() got an unexpected keyword argument 'vx'
```

**Traceback:**
```
File "backend/integration_service.py", line 681, in _check_trajectory_calculation
    ball_state = self.state_converter.vision_ball_to_ball_state(
File "backend/integration_service_conversion_helpers.py", line 240, in vision_ball_to_ball_state
    ball_state = BallState.from_4k(
File "backend/core/models.py", line 254, in from_4k
    return cls(id=id, position=position, velocity=velocity, number=number, **kwargs)
TypeError: BallState.__init__() got an unexpected keyword argument 'vx'
```

## Root Cause

In `backend/integration_service_conversion_helpers.py:240-253`, the code was calling `BallState.from_4k()` with individual `vx` and `vy` velocity components:

```python
ball_state = BallState.from_4k(
    id=ball_id,
    x=position_4k.x,
    y=position_4k.y,
    vx=velocity_4k.x,      # ❌ Wrong: passing individual components
    vy=velocity_4k.y,      # ❌ Wrong: passing individual components
    ...
)
```

However:
1. The `BallState.from_4k()` method expects a `velocity` parameter as a `Vector2D` object, not individual `vx`/`vy` components
2. The `BallState` dataclass has a field `velocity: Vector2D`, not separate `vx` and `vy` fields
3. The `from_4k()` method extracts velocity using `kwargs.pop("velocity", Vector2D.from_4k(0, 0))`
4. Since `vx` and `vy` were not recognized, they got passed through `**kwargs` to the `BallState` constructor, causing the TypeError

## Solution

Changed the call to pass the velocity as a `Vector2D` object:

```python
ball_state = BallState.from_4k(
    id=ball_id,
    x=position_4k.x,
    y=position_4k.y,
    velocity=velocity_4k,  # ✅ Correct: passing Vector2D object
    ...
)
```

This is correct because:
- The code already creates `velocity_4k` as a `Vector2D` object (lines 169-172)
- The `from_4k()` method expects this format
- The `BallState` constructor requires a `Vector2D` for velocity

## Files Changed

- `backend/integration_service_conversion_helpers.py:240-253` - Fixed the `BallState.from_4k()` call

## Verification

- ✅ Checked all other uses of `BallState.from_4k()` in the codebase - no similar issues found
- ✅ Existing test in `backend/test_4k_functionality.py` uses the method correctly
- ✅ The fix aligns with the API design documented in `backend/core/models.py`

## Related Context

This fix is part of the 4K canonical coordinate system migration. The velocity conversion logic is working correctly - it was just the final API call that had the wrong parameter names.
