# Frontend Data Handlers Migration to Dict Format

## Summary
Updated `/Users/jchadwick/code/billiards-trainer/frontend/web/src/services/data-handlers.ts` to handle the new dictionary format `{x, y, scale}` for positions and velocities instead of the legacy array format `[x, y]`.

The `scale` property is a `[number, number]` tuple representing the coordinate space resolution (e.g., `[1920, 1080]` for 1080p or `[3840, 2160]` for 4K), not a string as initially thought.

## Changes Made

### 1. Import PositionWithScale (Lines 5-22)
Added import for `PositionWithScale` type from API types:

```typescript
import {
  // ... other imports
  type PositionWithScale,
  // ... other imports
} from "../types/api";
```

### 2. New Position2D Interface (Lines 66-72)
Added a new interface to represent positions in the dict format:

```typescript
// Position type for new dict format with scale metadata
// Scale is a [width, height] tuple indicating the coordinate space (e.g., [1920, 1080] or [3840, 2160])
export interface Position2D {
  x: number;
  y: number;
  scale?: [number, number]; // Coordinate space metadata as [width, height]
}
```

**Note:** The `scale` property is a `[number, number]` tuple representing resolution (width, height), compatible with `PositionWithScale` from the API.

### 3. Helper Functions (Lines 143-162)
Added two helper functions for conversion and compatibility:

```typescript
/**
 * Convert position from array or object format to Position2D object
 * Handles both legacy [x, y] format and new {x, y, scale?} format
 * PositionWithScale from API is compatible with Position2D
 */
function toPosition2D(pos: [number, number] | PositionWithScale | Position2D): Position2D {
  if (Array.isArray(pos)) {
    // Legacy array format [x, y] - no scale information
    return { x: pos[0], y: pos[1] };
  }
  // New object format {x, y, scale?} - PositionWithScale or Position2D
  return pos as Position2D;
}

/**
 * Extract x, y coordinates from Position2D (ignoring scale metadata for now)
 */
function getXY(pos: Position2D): {x: number, y: number} {
  return { x: pos.x, y: pos.y };
}
```

### 4. Interface Updates

#### ProcessedBallData (Lines 74-80)
**BEFORE:**
```typescript
interpolatedPosition?: [number, number];
predictedPosition?: [number, number];
```

**AFTER:**
```typescript
interpolatedPosition?: Position2D;
predictedPosition?: Position2D;
```

#### ProcessedCueData (Lines 82-89)
**BEFORE:**
```typescript
predictedTrajectory?: [number, number][];
```

**AFTER:**
```typescript
predictedTrajectory?: Position2D[];
```

#### ProcessedTableData (Lines 91-100)
**BEFORE:**
```typescript
centerPoint?: [number, number];
```

**AFTER:**
```typescript
centerPoint?: Position2D;
```

#### ProcessedTrajectory (Lines 102-117)
**BEFORE:**
```typescript
smoothedLines: Array<{
  start: [number, number];
  end: [number, number];
  confidence: number;
}>;
collisionPredictions: Array<{
  position: [number, number];
  // ...
}>;
```

**AFTER:**
```typescript
smoothedLines: Array<{
  start: Position2D;
  end: Position2D;
  confidence: number;
}>;
collisionPredictions: Array<{
  position: Position2D;
  // ...
}>;
```

### 5. Method Updates

#### createProcessedGameState (Lines 427-486)
**Key Changes:**
- Lines 428: Convert `ball.position` to Position2D using `toPosition2D()`
- Lines 438: Convert previous ball position to Position2D
- Lines 439-441: Use `.x` and `.y` properties instead of `[0]` and `[1]`
- Lines 455-459: Create interpolated position as Position2D object with scale metadata
- Lines 462-468: Convert velocity and create predicted position as Position2D object

**BEFORE:**
```typescript
const deltaX = ball.position[0] - prevBall.position[0];
const deltaY = ball.position[1] - prevBall.position[1];
interpolatedPosition = [
  prevBall.position[0] + deltaX * smoothing,
  prevBall.position[1] + deltaY * smoothing,
] as [number, number];
```

**AFTER:**
```typescript
const prevPos = toPosition2D(prevBall.position);
const deltaX = currentPos.x - prevPos.x;
const deltaY = currentPos.y - prevPos.y;
interpolatedPosition = {
  x: prevPos.x + deltaX * smoothing,
  y: prevPos.y + deltaY * smoothing,
  scale: currentPos.scale, // Preserve scale metadata
};
```

#### calculateCueTrajectory (Lines 534-552)
**Key Changes:**
- Line 542: Convert cue position to Position2D
- Lines 546-548: Use `.x` and `.y` properties and create Position2D objects

**BEFORE:**
```typescript
const x = cue.position[0] + Math.cos(angleRad) * factor;
const y = cue.position[1] + Math.sin(angleRad) * factor;
points.push([x, y]);
```

**AFTER:**
```typescript
const cuePos = toPosition2D(cue.position);
const x = cuePos.x + Math.cos(angleRad) * factor;
const y = cuePos.y + Math.sin(angleRad) * factor;
points.push({ x, y, scale: cuePos.scale });
```

#### calculateAimingAccuracy (Lines 554-590)
**Key Changes:**
- Lines 565-566: Create aim direction as object with x, y properties
- Lines 566: Convert cue position to Position2D
- Lines 572-578: Convert ball position and use object properties

**BEFORE:**
```typescript
const aimDirection = [Math.cos(angleRad), Math.sin(angleRad)];
const toBall = [
  ball.position[0] - cue.position[0],
  ball.position[1] - cue.position[1],
];
const normalized = [toBall[0] / distance, toBall[1] / distance];
const dotProduct = aimDirection[0] * normalized[0] + aimDirection[1] * normalized[1];
```

**AFTER:**
```typescript
const aimDirection = { x: Math.cos(angleRad), y: Math.sin(angleRad) };
const cuePos = toPosition2D(cue.position);
const ballPos = toPosition2D(ball.position);
const toBall = {
  x: ballPos.x - cuePos.x,
  y: ballPos.y - cuePos.y,
};
const normalized = { x: toBall.x / distance, y: toBall.y / distance };
const dotProduct = aimDirection.x * normalized.x + aimDirection.y * normalized.y;
```

#### calculateTableCenter (Lines 592-601)
**Key Changes:**
- Line 594: Convert all corners to Position2D
- Lines 595-596: Use `.x` and `.y` properties
- Lines 599-600: Return Position2D object with scale metadata

**BEFORE:**
```typescript
const avgX = table.corners.reduce((sum, corner) => sum + corner[0], 0) / table.corners.length;
const avgY = table.corners.reduce((sum, corner) => sum + corner[1], 0) / table.corners.length;
return [avgX, avgY];
```

**AFTER:**
```typescript
const corners = table.corners.map(toPosition2D);
const avgX = corners.reduce((sum, corner) => sum + corner.x, 0) / corners.length;
const avgY = corners.reduce((sum, corner) => sum + corner.y, 0) / corners.length;
const scale = corners[0]?.scale;
return { x: avgX, y: avgY, scale };
```

#### calculateTableBounds (Lines 603-620)
**Key Changes:**
- Line 610: Convert all corners to Position2D
- Lines 611-612: Use `.x` and `.y` properties

**BEFORE:**
```typescript
const xs = table.corners.map((corner) => corner[0]);
const ys = table.corners.map((corner) => corner[1]);
```

**AFTER:**
```typescript
const corners = table.corners.map(toPosition2D);
const xs = corners.map((corner) => corner.x);
const ys = corners.map((corner) => corner.y);
```

#### createProcessedTrajectory (Lines 645-678)
**Key Changes:**
- Lines 651-652: Convert trajectory line start/end to Position2D
- Line 662: Convert collision position to Position2D

**BEFORE:**
```typescript
let smoothedLines = trajectoryData.lines.map((line) => ({
  start: line.start,
  end: line.end,
  confidence: line.confidence,
}));
const collisionPredictions = trajectoryData.collisions.map((collision) => ({
  position: collision.position,
  // ...
}));
```

**AFTER:**
```typescript
let smoothedLines = trajectoryData.lines.map((line) => ({
  start: toPosition2D(line.start),
  end: toPosition2D(line.end),
  confidence: line.confidence,
}));
const collisionPredictions = trajectoryData.collisions.map((collision) => ({
  position: toPosition2D(collision.position),
  // ...
}));
```

#### smoothTrajectoryLines (Lines 680-714)
**Key Changes:**
- Lines 696-700: Create smoothed start position as Position2D object
- Lines 702-706: Create smoothed end position as Position2D object
- Both preserve scale metadata

**BEFORE:**
```typescript
const smoothedStart: [number, number] = [
  (prev.end[0] + line.start[0] + next.start[0]) / 3,
  (prev.end[1] + line.start[1] + next.start[1]) / 3,
];
const smoothedEnd: [number, number] = [
  (line.end[0] + next.start[0] + next.end[0]) / 3,
  (line.end[1] + next.start[1] + next.end[1]) / 3,
];
```

**AFTER:**
```typescript
const smoothedStart: Position2D = {
  x: (prev.end.x + line.start.x + next.start.x) / 3,
  y: (prev.end.y + line.start.y + next.start.y) / 3,
  scale: line.start.scale, // Preserve scale metadata
};
const smoothedEnd: Position2D = {
  x: (line.end.x + next.start.x + next.end.x) / 3,
  y: (line.end.y + next.start.y + next.end.y) / 3,
  scale: line.end.scale, // Preserve scale metadata
};
```

## Scale Metadata Handling

Throughout the changes, the `scale` metadata is:
1. **Preserved**: When creating new Position2D objects, the scale from the source is preserved
2. **Extracted but not actively used**: The scale metadata is available but not currently used in calculations
3. **Optional**: The scale property is optional in the Position2D interface
4. **Format**: The scale is a `[number, number]` tuple representing `[width, height]` of the coordinate space (e.g., `[1920, 1080]` for 1080p, `[3840, 2160]` for 4K)

This allows for future enhancements where scale information might be used for coordinate space conversions or validations.

## Backward Compatibility

The `toPosition2D()` helper function provides backward compatibility by:
- Accepting both array `[x, y]` and object `{x, y, scale?}` formats
- Converting arrays to Position2D objects automatically
- Allowing gradual migration from the API side

## Edge Cases Handled

1. **Missing velocity data**: Checked before converting (line 466)
2. **Empty corners array**: Safe access with optional chaining (line 603)
3. **Missing ball_id in collisions**: Fallback to empty string (line 667)
4. **Missing is_cue_ball property**: Changed from checking `b.is_cue_ball` to checking `b.id === "cue" || b.id === "0"` (line 565)

## Testing Recommendations

1. Test with legacy array format data from API
2. Test with new dict format data from API
3. Verify scale metadata is preserved through all transformations
4. Test edge cases (empty arrays, missing properties)
5. Verify TypeScript compilation passes
6. Test WebSocket message processing with both formats

## Files Modified

- `/Users/jchadwick/code/billiards-trainer/frontend/web/src/services/data-handlers.ts`

## Compilation Status

✅ TypeScript compilation passes without errors:
```bash
cd /Users/jchadwick/code/billiards-trainer/frontend/web && npx tsc --noEmit --skipLibCheck src/services/data-handlers.ts
# No errors or warnings
```

## Next Steps

1. Update API type definitions (`api.ts`) to reflect the new format when backend is fully migrated
2. Update other frontend components that consume this processed data
3. Add unit tests for the Position2D conversion logic
4. Consider adding coordinate space conversion utilities that leverage the scale metadata
