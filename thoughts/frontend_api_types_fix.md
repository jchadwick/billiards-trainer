# Frontend API Types Migration to PositionWithScale

## Summary

Successfully updated `/Users/jchadwick/code/billiards-trainer/frontend/web/src/types/api.ts` to use the new dict format with scale metadata instead of arrays for all position and velocity fields.

## Changes Made

### New Type Definition (Lines 10-24)

Created the `PositionWithScale` interface:

```typescript
/**
 * Position with scale metadata for coordinate system tracking.
 *
 * The scale metadata indicates the coordinate space this position is in:
 * - [1920, 1080] = Original 1080p coordinate space
 * - [3840, 2160] = Native 4K coordinate space
 *
 * This allows the frontend to properly interpret and transform coordinates
 * based on the display resolution and rendering context.
 */
export interface PositionWithScale {
  x: number;
  y: number;
  scale: [number, number];
}
```

### Types Updated

#### WebSocket Message Types

1. **BallData** (Lines 96, 99)
   - `position: [number, number]` → `position: PositionWithScale`
   - `velocity?: [number, number]` → `velocity?: PositionWithScale`

2. **CueData** (Lines 106, 110)
   - `position: [number, number]` → `position: PositionWithScale`
   - `tip_position?: [number, number]` → `tip_position?: PositionWithScale`

3. **TableData** (Lines 114, 115)
   - `corners: [number, number][]` → `corners: PositionWithScale[]`
   - `pockets: [number, number][]` → `pockets: PositionWithScale[]`

4. **TrajectoryLine** (Lines 131, 132)
   - `start: [number, number]` → `start: PositionWithScale`
   - `end: [number, number]` → `end: PositionWithScale`

5. **CollisionData** (Lines 138, 144, 145)
   - `position: [number, number]` → `position: PositionWithScale`
   - `velocity_before?: [number, number]` → `velocity_before?: PositionWithScale`
   - `velocity_after?: [number, number]` → `velocity_after?: PositionWithScale`

#### REST API Response Types

6. **BallInfo** (Lines 298, 299)
   - `position: [number, number]` → `position: PositionWithScale`
   - `velocity: [number, number]` → `velocity: PositionWithScale`

7. **CueInfo** (Line 307)
   - `tip_position: [number, number]` → `tip_position: PositionWithScale`

8. **TableInfo** (Line 318)
   - `pocket_positions: [number, number][]` → `pocket_positions: PositionWithScale[]`

## Total Changes

- **15 fields updated** across 8 interfaces
- **1 new interface** created with comprehensive JSDoc comments
- **All other fields** remain unchanged

## TypeScript Compilation Status

The type changes compile successfully and correctly identify code that needs updating:

### Files Requiring Updates

The following files are now correctly flagged as needing updates to use the new format:

1. **src/services/data-handlers.ts**
   - Multiple instances of array index access (`position[0]`, `position[1]`)
   - Type mismatches where code creates `[number, number]` arrays instead of `PositionWithScale` objects
   - **42 errors** related to the type change (expected and correct)

2. **src/stores/VideoStore.ts**
   - Array index access on position fields
   - **12 errors** related to the type change (expected and correct)

### Error Examples

Typical errors that need to be fixed in consuming code:

```typescript
// OLD CODE (now errors):
const x = ball.position[0];  // ❌ Property '0' does not exist on type 'PositionWithScale'
const y = ball.position[1];  // ❌ Property '1' does not exist on type 'PositionWithScale'

// NEW CODE (required):
const x = ball.position.x;  // ✅ Correct usage
const y = ball.position.y;  // ✅ Correct usage
const scale = ball.position.scale;  // ✅ Access to scale metadata
```

## Next Steps

The following files need to be updated to work with the new type:

1. `/Users/jchadwick/code/billiards-trainer/frontend/web/src/services/data-handlers.ts` (42 errors)
   - Update all array index access to property access
   - Update code that creates position tuples to create PositionWithScale objects
   - Add scale metadata when creating positions

2. `/Users/jchadwick/code/billiards-trainer/frontend/web/src/stores/VideoStore.ts` (12 errors)
   - Update all array index access to property access
   - Ensure scale metadata is properly handled

## Migration Pattern

When updating consuming code, follow this pattern:

```typescript
// BEFORE:
const position: [number, number] = [x, y];
const x = position[0];
const y = position[1];

// AFTER:
const position: PositionWithScale = {
  x,
  y,
  scale: [3840, 2160]  // Use appropriate scale for the context
};
const x = position.x;
const y = position.y;
const scale = position.scale;
```

## Benefits

1. **Explicit coordinate space tracking** - Every position now carries metadata about which coordinate system it's in
2. **Type safety** - TypeScript catches incorrect usage at compile time
3. **Better API alignment** - Frontend types now match backend response format
4. **Future-proof** - Easy to add coordinate transformation utilities that use the scale metadata
5. **Self-documenting** - The scale property makes it clear what resolution the coordinates are in

## Status

- ✅ API type definitions updated
- ✅ PositionWithScale interface created with documentation
- ✅ TypeScript compilation validates the changes
- ⏳ Consumer code updates needed (data-handlers.ts, VideoStore.ts)
- ⏳ Runtime testing needed after consumer updates

## Notes

- The existing TypeScript errors in other files (accessibility components, diagnostics, etc.) are unrelated to this change
- All position/velocity-related type errors are correctly identified and need to be fixed in consumer code
- The type system is now enforcing the new format, preventing accidental usage of the old array format
