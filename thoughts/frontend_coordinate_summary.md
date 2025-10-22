# Frontend Coordinate Migration Summary

## Quick Reference

### Current State
- **Backend sends**: `position: [x, y]` (array)
- **Frontend receives**: Arrays in API types
- **Frontend converts**: Arrays → Objects in VideoStore
- **Frontend uses internally**: `{x, y}` objects

### New Backend Format
```python
position = {
    "x": 123.4,
    "y": 567.8,
    "space": "screen_space",
    "resolution": "4k"
}
```

## Critical Conversion Point

**File**: `/stores/VideoStore.ts`
**Methods**:
- `handleGameStateMessage()` (Lines 542-635)
- `handleTrajectoryMessage()` (Lines 637-676)

This is THE ONLY place where array-to-object conversion happens!

## Files to Update (6)

### 1. Type Definitions
**File**: `/types/api.ts`

Change ALL coordinate types from arrays to objects:

```typescript
// Before
export interface BallData {
  position: [number, number];
  velocity?: [number, number];
}

// After
export interface Position2D {
  x: number;
  y: number;
  space?: string;
  resolution?: string;
}

export interface BallData {
  position: Position2D;
  velocity?: Position2D;
}
```

Apply to:
- `BallData.position`, `BallData.velocity`
- `CueData.position`, `CueData.tip_position`
- `TableData.corners[]`, `TableData.pockets[]`
- `TrajectoryLine.start`, `TrajectoryLine.end`
- `CollisionData.position`, `CollisionData.velocity_before`, `CollisionData.velocity_after`

### 2. VideoStore Conversion Logic
**File**: `/stores/VideoStore.ts`

```typescript
// OLD (Lines 548-550)
position: { x: ballData.position[0], y: ballData.position[1] },

// NEW
position: { x: ballData.position.x, y: ballData.position.y },
```

Update in:
- `handleGameStateMessage()`: balls, cue, table conversion
- `handleTrajectoryMessage()`: lines and collisions conversion

### 3. Data Processing Service
**File**: `/services/data-handlers.ts`

```typescript
// OLD (Line 407)
const deltaX = ball.position[0] - prevBall.position[0];

// NEW
const deltaX = ball.position.x - prevBall.position.x;
```

Update all array indexing in:
- `createProcessedGameState()` (Lines 388-497)
- `calculateCueTrajectory()` (Lines 499-516)
- `calculateAimingAccuracy()` (Lines 518-552)
- `calculateTableCenter()` (Lines 554-562)
- `calculateTableBounds()` (Lines 564-579)

### 4. Integration Tests
**File**: `/tests/integration/DetectionOverlayIntegration.test.ts`

Update test conversion logic to match new format.

### 5. LiveView Component (BUG FIX)
**File**: `/components/video/LiveView.tsx`

Current bug: Receives array format from WebSocket but tries to use object properties.

**Fix**: Use VideoStore instead of direct WebSocket access.

### 6. (Optional) Metadata Handling

Create new types for space metadata:

```typescript
export interface CoordinateMetadata {
  space: 'screen_space' | 'table_space' | 'world_space';
  resolution: '1080p' | '4k' | '8k';
}

export interface Position2DWithMetadata extends Point2D {
  metadata?: CoordinateMetadata;
}
```

## Files That DON'T Need Changes (13+)

All these already use `{x, y}` object format:
- `/types/video.ts`
- `/stores/types/index.ts`
- `/utils/coordinates.ts`
- All overlay components (BallOverlay, TrajectoryOverlay, CueOverlay, TableOverlay, CalibrationOverlay)
- `/components/video/OverlayCanvas.tsx`
- `/stores/GameStore.ts`
- `/stores/VisionStore.ts`
- All rendering and UI components

## Migration Steps

1. **Update API types** (`/types/api.ts`)
   - Change all `[number, number]` to `Position2D`
   - Add space metadata types

2. **Update VideoStore** (`/stores/VideoStore.ts`)
   - Change `position[0]` to `position.x`
   - Change `position[1]` to `position.y`
   - Optionally extract and store metadata

3. **Update DataProcessingService** (`/services/data-handlers.ts`)
   - Change all array indexing to object property access

4. **Update tests** (`/tests/integration/DetectionOverlayIntegration.test.ts`)
   - Match new conversion format

5. **Fix LiveView** (`/components/video/LiveView.tsx`)
   - Remove direct WebSocket access
   - Use VideoStore for ball data

6. **Test thoroughly**
   - Ball detection overlay
   - Trajectory rendering
   - Cue stick overlay
   - Table calibration
   - Collision predictions

## Risk Assessment

**LOW RISK** - Architecture is well-designed for this change:

✅ Single conversion point (VideoStore)
✅ Clean separation between API types and internal types
✅ Well-abstracted coordinate transformations
✅ No hardcoded coordinate values
✅ Good test coverage

⚠️ **Known Issue**: LiveView component has existing type mismatch

## Testing Checklist

After migration, verify:
- [ ] Ball positions render correctly
- [ ] Ball velocities shown correctly
- [ ] Cue stick tip and tail positions correct
- [ ] Table corners and pockets align
- [ ] Trajectory lines render properly
- [ ] Collision points appear in right locations
- [ ] Calibration overlay works
- [ ] Coordinate transformations (videoToCanvas/canvasToVideo) work
- [ ] No console errors about undefined properties
- [ ] No TypeScript compilation errors

## Rollback Plan

If issues occur:
1. Backend can send BOTH formats temporarily
2. Frontend can detect format and handle both
3. Gradual migration possible

```typescript
// Compatibility helper
function normalizePosition(pos: [number, number] | Position2D): Position2D {
  if (Array.isArray(pos)) {
    return { x: pos[0], y: pos[1] };
  }
  return { x: pos.x, y: pos.y };
}
```
