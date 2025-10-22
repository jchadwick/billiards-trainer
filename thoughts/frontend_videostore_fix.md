# Frontend VideoStore Fix - Dict Format Migration

## Summary
Updated `/Users/jchadwick/code/billiards-trainer/frontend/web/src/stores/VideoStore.ts` to handle the new dict format `{x, y, scale}` from the backend API instead of the legacy array format `[x, y]`.

## Date
2025-10-21

## Changes Made

### 1. Calibration Corner Conversions (Lines 336-351)
**Function**: `fetchCalibrationData()`

**Before**:
```typescript
corners: (apiData.corners || []).map((corner: any, index: number) => ({
  id: `corner-${index}`,
  screenPosition: {
    x: Array.isArray(corner) ? corner[0] : corner.x,
    y: Array.isArray(corner) ? corner[1] : corner.y,
  },
  worldPosition: {
    x: Array.isArray(corner) ? corner[0] : corner.x,
    y: Array.isArray(corner) ? corner[1] : corner.y,
  },
  ...
}))
```

**After**:
```typescript
// Note: API now sends positions as {x, y, scale} dicts instead of [x, y] arrays
corners: (apiData.corners || []).map((corner: any, index: number) => ({
  id: `corner-${index}`,
  screenPosition: {
    x: corner.x,
    y: corner.y,
  },
  worldPosition: {
    x: corner.x,
    y: corner.y,
  },
  ...
}))
```

### 2. Ball Position/Velocity Conversions (Lines 548-561)
**Function**: `handleGameStateMessage()`

**Before**:
```typescript
const balls: Ball[] = gameState.balls.map((ballData: BallData) => ({
  id: ballData.id,
  position: { x: ballData.position[0], y: ballData.position[1] },
  radius: ballData.radius,
  type: this.inferBallType(ballData.id, ballData.color),
  number: this.inferBallNumber(ballData.id, ballData.color),
  velocity: ballData.velocity
    ? { x: ballData.velocity[0], y: ballData.velocity[1] }
    : { x: 0, y: 0 },
  ...
}));
```

**After**:
```typescript
// Note: API now sends positions/velocities as {x, y, scale} dicts instead of [x, y] arrays
const balls: Ball[] = gameState.balls.map((ballData: BallData) => ({
  id: ballData.id,
  position: { x: ballData.position.x, y: ballData.position.y },
  radius: ballData.radius,
  type: this.inferBallType(ballData.id, ballData.color),
  number: this.inferBallNumber(ballData.id, ballData.color),
  velocity: ballData.velocity
    ? { x: ballData.velocity.x, y: ballData.velocity.y }
    : { x: 0, y: 0 },
  ...
}));
```

### 3. Cue Position/Tip Position Conversions (Lines 563-591)
**Function**: `handleGameStateMessage()`

**Before**:
```typescript
cue = {
  tipPosition: {
    x: gameState.cue.position[0],
    y: gameState.cue.position[1],
  },
  tailPosition: gameState.cue.tip_position
    ? {
        x: gameState.cue.tip_position[0],
        y: gameState.cue.tip_position[1],
      }
    : {
        x: gameState.cue.position[0] - Math.cos(gameState.cue.angle) * (gameState.cue.length || 100),
        y: gameState.cue.position[1] - Math.sin(gameState.cue.angle) * (gameState.cue.length || 100),
      },
  ...
};
```

**After**:
```typescript
// Note: API now sends positions as {x, y, scale} dicts instead of [x, y] arrays
cue = {
  tipPosition: {
    x: gameState.cue.position.x,
    y: gameState.cue.position.y,
  },
  tailPosition: gameState.cue.tip_position
    ? {
        x: gameState.cue.tip_position.x,
        y: gameState.cue.tip_position.y,
      }
    : {
        x: gameState.cue.position.x - Math.cos(gameState.cue.angle) * (gameState.cue.length || 100),
        y: gameState.cue.position.y - Math.sin(gameState.cue.angle) * (gameState.cue.length || 100),
      },
  ...
};
```

### 4. Table Corners/Pockets Conversions (Lines 593-610)
**Function**: `handleGameStateMessage()`

**Before**:
```typescript
table = {
  corners: gameState.table.corners.map((corner) => ({
    x: corner[0],
    y: corner[1],
  })),
  pockets: gameState.table.pockets.map((pocket) => ({
    x: pocket[0],
    y: pocket[1],
  })),
  ...
};
```

**After**:
```typescript
// Note: API now sends positions as {x, y, scale} dicts instead of [x, y] arrays
table = {
  corners: gameState.table.corners.map((corner) => ({
    x: corner.x,
    y: corner.y,
  })),
  pockets: gameState.table.pockets.map((pocket) => ({
    x: pocket.x,
    y: pocket.y,
  })),
  ...
};
```

### 5. Trajectory Points Conversions (Lines 649-673)
**Function**: `handleTrajectoryMessage()`

**Before**:
```typescript
const trajectories: Trajectory[] = trajectoryData.lines.map(
  (line, index) => ({
    ballId: trajectoryData.collisions[0]?.ball1_id || movingBallId,
    points: [
      { x: line.start[0], y: line.start[1] },
      { x: line.end[0], y: line.end[1] },
    ],
    collisions: trajectoryData.collisions.map((collision) => ({
      position: { x: collision.position[0], y: collision.position[1] },
      ...
    })),
    ...
  })
);
```

**After**:
```typescript
// Note: API now sends positions as {x, y, scale} dicts instead of [x, y] arrays
const trajectories: Trajectory[] = trajectoryData.lines.map(
  (line, index) => ({
    ballId: trajectoryData.collisions[0]?.ball1_id || movingBallId,
    points: [
      { x: line.start.x, y: line.start.y },
      { x: line.end.x, y: line.end.y },
    ],
    collisions: trajectoryData.collisions.map((collision) => ({
      position: { x: collision.position.x, y: collision.position.y },
      ...
    })),
    ...
  })
);
```

## Functions Updated

1. **`fetchCalibrationData()`** - Lines 327-375
   - Calibration corner conversions

2. **`handleGameStateMessage()`** - Lines 542-635
   - Ball positions and velocities
   - Cue positions (tip and tail)
   - Table corners and pockets

3. **`handleTrajectoryMessage()`** - Lines 637-676
   - Trajectory line start/end points
   - Collision positions

## Pattern Summary

All conversions followed this pattern:

- **Array access** `position[0]`, `position[1]` → **Object access** `position.x`, `position.y`
- **Array access** `velocity[0]`, `velocity[1]` → **Object access** `velocity.x`, `velocity.y`
- **Array destructuring** (none found) would have been → **Object destructuring** `const {x, y} = position`

## Scale Metadata

The backend now sends scale metadata in the format `{x, y, scale: [sx, sy]}`. The scale values are currently not used by the frontend but could be extracted for debugging purposes:

```typescript
// Optional: Extract scale metadata for debugging
const sourceResolution = {
  width: 3840 / position.scale[0],
  height: 2160 / position.scale[1]
};
```

## Internal Format

The frontend continues to use the `{x, y}` format internally (no scale needed). This is consistent with the existing type definitions in `/Users/jchadwick/code/billiards-trainer/frontend/web/src/types/`.

## Testing Recommendations

1. Test calibration data fetching to ensure corners are properly converted
2. Test real-time game state updates via WebSocket
3. Test trajectory rendering with the new position format
4. Verify cue stick positioning and angle calculations
5. Test table bounds calculation from corners

## Notes

- All array index accesses have been replaced with object property accesses
- Added explanatory comments at each conversion point
- No changes to internal data structures or type definitions were needed
- The frontend is now compatible with the backend's dict format with scale metadata
