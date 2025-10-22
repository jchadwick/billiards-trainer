# LiveView Position Data Type Mismatch Fix

## Summary

Analyzed and fixed the LiveView component which had a type mismatch where the backend sends dict-format positions `{x, y}` but the component was expecting the wrong fields.

## Investigation Findings

### Data Flow

1. **Backend (integration_service.py)** → Broadcasts ball data via WebSocket
2. **Frontend (LiveView.tsx)** → Directly consumes WebSocket messages (NOT through VideoStore)
3. **VideoStore.ts** → Also consumes WebSocket but for its own state management

### Backend Format (integration_service.py lines 481-508)

The backend sends balls in this format:
```python
{
    "id": str,
    "position": {"x": float, "y": float},  # Dict format, NOT array
    "radius": float,  # NOW INCLUDED (was missing before)
    "velocity": {"x": float, "y": float},
    "number": int,
    "type": str,  # "cue", "solid", "stripe", "eight", "unknown"
    "confidence": float,
    "is_moving": bool,
    "is_cue_ball": bool
}
```

**Note**: The backend sends simple `{x, y}` dicts, NOT `{x, y, scale}` with metadata. That format is only used in the API routes (via converters.py), not in the WebSocket broadcasts from integration_service.py.

### Frontend Expected Format (LiveView.tsx lines 7-14)

```typescript
interface Ball {
  id: string;
  position: { x: number; y: number };  // Expects dict format ✓
  radius: number;                      // Expects radius ✓ (NOW SENT)
  number?: number;
  type: 'cue' | 'solid' | 'stripe' | 'eight';
  confidence: number;
}
```

### Frontend Types (api.ts)

The `api.ts` types were recently updated to use `PositionWithScale`:
```typescript
export interface PositionWithScale {
  x: number;
  y: number;
  scale: [number, number];
}

export interface BallData {
  position: PositionWithScale;  // Expects {x, y, scale}
  // ...
}
```

However, this is **WRONG** for WebSocket data from integration_service.py. The `scale` field is only sent by API routes that use `vector2d_to_dict()`, NOT by WebSocket broadcasts.

## Issues Found and Fixed

### 1. ✅ Backend Missing `radius` Field

**Problem**: Backend was not sending `radius` in ball data, but LiveView needs it to draw balls.

**Fix**: Added `"radius": ball.radius` to integration_service.py line 497

**Before**:
```python
"position": {"x": ball.position[0], "y": ball.position[1]},
"velocity": {"x": ball.velocity[0], "y": ball.velocity[1]},
```

**After**:
```python
"position": {"x": ball.position[0], "y": ball.position[1]},
"radius": ball.radius,
"velocity": {"x": ball.velocity[0], "y": ball.velocity[1]},
```

### 2. ⚠️ Type Mismatch in api.ts (NOT FIXED - Needs Discussion)

**Problem**: The `api.ts` types claim positions are `PositionWithScale` (with scale metadata), but integration_service.py sends simple `{x, y}` dicts without scale.

**Impact**:
- VideoStore.ts was already updated to handle `{x, y}` dicts correctly (lines 552, 557, 569-570, 575)
- LiveView.tsx directly uses `{x, y}` format and works correctly
- But TypeScript types don't match runtime data

**Options**:
1. Update backend to send `scale` metadata in WebSocket broadcasts
2. Update api.ts to have separate types for WebSocket vs REST API data
3. Make `scale` optional in `PositionWithScale`

**Recommendation**: Make `scale` optional since it's not needed for display and only used for coordinate system tracking internally.

### 3. ⚠️ Cue Data Mismatch (NOT FIXED - LiveView doesn't use cue)

**Problem**: Frontend `CueData` interface expects `position` and `detected` fields, but backend sends `tip_position` and `state`.

**Backend sends** (integration_service.py lines 513-519):
```python
{
    "tip_position": {"x": ..., "y": ...},
    "angle": float,
    "length": float,
    "state": str,  # "hidden", "visible", "aiming", "striking"
    "confidence": float
}
```

**Frontend expects** (api.ts lines 104-111):
```typescript
{
    position: PositionWithScale,     // NOT SENT
    detected: boolean,               // NOT SENT
    angle: number,
    confidence: number,
    length?: number,
    tip_position?: PositionWithScale  // Sent as required, not optional
}
```

**Impact**: LiveView doesn't currently use cue data, so this doesn't break anything yet. But VideoStore and other components might fail when trying to access `cue.position`.

**Fix needed**: Either:
- Backend: Add `position` (alias for `tip_position`) and `detected` (derived from `state != 'hidden'`)
- Frontend: Update types to match backend reality

## What Was Actually Broken

### Before Fix:
1. **Backend** was sending balls WITHOUT `radius` field
2. **LiveView** was trying to draw balls using `ball.radius` (line 122)
3. **Result**: Ball circles would have `undefined` radius, causing rendering issues

### After Fix:
1. **Backend** now sends `radius` field ✓
2. **LiveView** can correctly access `ball.radius` ✓
3. **Result**: Balls should render with correct size ✓

## What Still Works (Was Never Broken)

1. **Position format**: Backend sends `{x, y}` dicts, LiveView expects `{x, y}` dicts ✓
2. **Position access**: LiveView uses `ball.position.x` and `ball.position.y` (lines 120-121) ✓
3. **Type classification**: Backend sends `type` as string, LiveView expects union type ✓
4. **Confidence**: Both send/receive as number ✓

## Files Changed

### Backend
- **File**: `/Users/jchadwick/code/billiards-trainer/backend/integration_service.py`
- **Line**: 497
- **Change**: Added `"radius": ball.radius,`

### Frontend
- **File**: `/Users/jchadwick/code/billiards-trainer/frontend/web/src/components/video/LiveView.tsx`
- **Status**: No changes needed (was already correct)

## Recommendations

### High Priority
1. ✅ **DONE**: Add `radius` to backend ball data
2. **TODO**: Test that balls render correctly in LiveView with proper sizes

### Medium Priority
3. **TODO**: Fix cue data mismatch between backend and frontend types
4. **TODO**: Update api.ts types to match actual WebSocket data format:
   - Make `scale` optional in `PositionWithScale`, OR
   - Create separate `Position` type without scale for WebSocket data

### Low Priority
5. **TODO**: Document the difference between REST API and WebSocket data formats
6. **TODO**: Consider standardizing on one format for all position data

## Testing Checklist

- [ ] Start backend with `python -m backend.video`
- [ ] Open frontend LiveView component
- [ ] Verify balls are detected and displayed
- [ ] Verify ball circles have correct radius (not too small/large)
- [ ] Verify ball positions track correctly as they move
- [ ] Check console for any TypeScript errors about position types

## Conclusion

**The bug was**: Backend not sending `radius` field, causing LiveView to draw balls with undefined size.

**The fix**: Added `radius` to backend WebSocket ball data.

**What was NOT broken**: Position data format - backend correctly sends `{x, y}` dicts and frontend correctly accesses `.x` and `.y` properties.

**Type system issue**: The api.ts types claim `PositionWithScale` with mandatory `scale` field, but runtime data doesn't include it. This needs to be addressed separately to match TypeScript types with runtime reality.
