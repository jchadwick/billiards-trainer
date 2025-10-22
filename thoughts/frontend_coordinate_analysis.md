# Frontend Coordinate System Analysis

**Date**: 2025-10-21
**Purpose**: Comprehensive documentation of ALL coordinate and position references in the frontend to understand impact of backend coordinate format change from `[x, y]` arrays to `{x, y, space_metadata}` dictionaries.

## Executive Summary

The frontend currently expects coordinates in **TWO FORMATS**:
1. **API/WebSocket Messages**: Expects `[number, number]` array format from backend
2. **Internal Types**: Uses `{x: number, y: number}` object format

**CRITICAL CONVERSION POINT**: The `VideoStore` contains the conversion logic that transforms incoming array-format coordinates into object format for internal use.

## Type Definitions

### 1. API Types (`/types/api.ts`)
**ALL coordinates come from backend as ARRAYS**:

```typescript
// Lines 78-86: Ball position as ARRAY
export interface BallData {
  id: string;
  position: [number, number];  // ← ARRAY FORMAT
  radius: number;
  color: string;
  velocity?: [number, number];  // ← ARRAY FORMAT
  confidence: number;
  visible: boolean;
}

// Lines 88-95: Cue position as ARRAY
export interface CueData {
  angle: number;
  position: [number, number];  // ← ARRAY FORMAT
  detected: boolean;
  confidence: number;
  length?: number;
  tip_position?: [number, number];  // ← ARRAY FORMAT
}

// Lines 97-103: Table corners and pockets as ARRAYS
export interface TableData {
  corners: [number, number][];  // ← ARRAY FORMAT
  pockets: [number, number][];  // ← ARRAY FORMAT
  rails?: Record<string, any>[];
  calibrated: boolean;
  dimensions?: Record<string, number>;
}

// Lines 114-119: Trajectory lines as ARRAYS
export interface TrajectoryLine {
  start: [number, number];  // ← ARRAY FORMAT
  end: [number, number];    // ← ARRAY FORMAT
  type: 'primary' | 'reflection' | 'collision';
  confidence: number;
}

// Lines 121-131: Collision position as ARRAY
export interface CollisionData {
  position: [number, number];  // ← ARRAY FORMAT
  ball_id?: string;
  ball1_id?: string;
  ball2_id?: string;
  type?: string;
  angle: number;
  velocity_before?: [number, number];  // ← ARRAY FORMAT
  velocity_after?: [number, number];   // ← ARRAY FORMAT
  time_to_collision?: number;
}
```

### 2. Video Types (`/types/video.ts`)
**Internal frontend types use OBJECT FORMAT**:

```typescript
// Lines 5-8: Point2D as OBJECT
export interface Point2D {
  x: number;  // ← OBJECT FORMAT
  y: number;  // ← OBJECT FORMAT
}

// Lines 10-13: Vector2D as OBJECT
export interface Vector2D {
  x: number;  // ← OBJECT FORMAT
  y: number;  // ← OBJECT FORMAT
}

// Lines 49-58: Ball with Point2D position
export interface Ball {
  id: string;
  position: Point2D;  // ← OBJECT FORMAT {x, y}
  radius: number;
  type: 'cue' | 'solid' | 'stripe' | 'eight';
  number?: number;
  velocity: Vector2D;  // ← OBJECT FORMAT {x, y}
  confidence: number;
  color?: string;
}

// Lines 60-68: CueStick with Point2D positions
export interface CueStick {
  tipPosition: Point2D;   // ← OBJECT FORMAT {x, y}
  tailPosition: Point2D;  // ← OBJECT FORMAT {x, y}
  angle: number;
  elevation: number;
  detected: boolean;
  confidence: number;
  length: number;
}

// Lines 70-77: Table with Point2D arrays
export interface Table {
  corners: Point2D[];  // ← OBJECT FORMAT {x, y}[]
  pockets: Point2D[];  // ← OBJECT FORMAT {x, y}[]
  bounds: BoundingBox;
  rails: Point2D[][];  // ← OBJECT FORMAT {x, y}[][]
  detected: boolean;
  confidence: number;
}

// Lines 79-86: Trajectory with Point2D
export interface Trajectory {
  ballId: string;
  points: Point2D[];  // ← OBJECT FORMAT {x, y}[]
  collisions: CollisionPoint[];
  type: 'primary' | 'reflection' | 'collision';
  probability: number;
  color: string;
}

// Lines 88-94: CollisionPoint with Point2D
export interface CollisionPoint {
  position: Point2D;  // ← OBJECT FORMAT {x, y}
  type: 'ball' | 'rail' | 'pocket';
  targetId?: string;
  angle: number;
  impulse: number;
}
```

### 3. Store Types (`/stores/types/index.ts`)
**Also uses OBJECT FORMAT internally**:

```typescript
// Lines 3-6
export interface Point2D {
  x: number;  // ← OBJECT FORMAT
  y: number;  // ← OBJECT FORMAT
}

// Lines 20-29: Ball definition
export interface Ball {
  id: number;
  type: 'solid' | 'stripe' | 'cue' | 'eight';
  color: string;
  position: Point2D;  // ← OBJECT FORMAT {x, y}
  velocity: Point2D;  // ← OBJECT FORMAT {x, y}
  isVisible: boolean;
  isPocketed: boolean;
  confidence: number;
}
```

## Critical Conversion Logic

### VideoStore.ts (THE MAIN CONVERSION POINT)

**Lines 542-635**: `handleGameStateMessage()` - **CONVERTS ARRAY TO OBJECT**

```typescript
private handleGameStateMessage(message: WebSocketMessage): void {
  if (!isGameStateMessage(message)) return;

  const gameState = message.data as GameStateData;

  // Convert WebSocket ball data to frontend ball format
  const balls: Ball[] = gameState.balls.map((ballData: BallData) => ({
    id: ballData.id,
    // ← ARRAY to OBJECT CONVERSION HAPPENS HERE
    position: { x: ballData.position[0], y: ballData.position[1] },
    radius: ballData.radius,
    type: this.inferBallType(ballData.id, ballData.color),
    number: this.inferBallNumber(ballData.id, ballData.color),
    velocity: ballData.velocity
      ? { x: ballData.velocity[0], y: ballData.velocity[1] }  // ← ARRAY to OBJECT
      : { x: 0, y: 0 },
    confidence: ballData.confidence,
    color: ballData.color,
  }));

  // Convert cue data
  let cue: CueStick | null = null;
  if (gameState.cue && gameState.cue.detected) {
    cue = {
      tipPosition: {
        x: gameState.cue.position[0],   // ← ARRAY to OBJECT
        y: gameState.cue.position[1],   // ← ARRAY to OBJECT
      },
      tailPosition: gameState.cue.tip_position
        ? {
            x: gameState.cue.tip_position[0],  // ← ARRAY to OBJECT
            y: gameState.cue.tip_position[1],  // ← ARRAY to OBJECT
          }
        : {
            x: gameState.cue.position[0] - Math.cos(gameState.cue.angle) * (gameState.cue.length || 100),
            y: gameState.cue.position[1] - Math.sin(gameState.cue.angle) * (gameState.cue.length || 100),
          },
      angle: gameState.cue.angle,
      elevation: 0,
      detected: gameState.cue.detected,
      confidence: gameState.cue.confidence,
      length: gameState.cue.length || 100,
    };
  }

  // Convert table data
  let table: Table | null = null;
  if (gameState.table && gameState.table.calibrated) {
    table = {
      corners: gameState.table.corners.map((corner) => ({
        x: corner[0],  // ← ARRAY to OBJECT
        y: corner[1],  // ← ARRAY to OBJECT
      })),
      pockets: gameState.table.pockets.map((pocket) => ({
        x: pocket[0],  // ← ARRAY to OBJECT
        y: pocket[1],  // ← ARRAY to OBJECT
      })),
      // ... rest of table data
    };
  }
}
```

**Lines 637-676**: `handleTrajectoryMessage()` - **CONVERTS ARRAY TO OBJECT**

```typescript
private handleTrajectoryMessage(message: WebSocketMessage): void {
  if (!isTrajectoryMessage(message)) return;

  const trajectoryData = message.data as TrajectoryData;

  const trajectories: Trajectory[] = trajectoryData.lines.map((line, index) => ({
    ballId: trajectoryData.collisions[0]?.ball1_id || movingBallId,
    points: [
      { x: line.start[0], y: line.start[1] },  // ← ARRAY to OBJECT
      { x: line.end[0], y: line.end[1] },      // ← ARRAY to OBJECT
    ],
    collisions: trajectoryData.collisions.map((collision) => ({
      position: { x: collision.position[0], y: collision.position[1] },  // ← ARRAY to OBJECT
      type: collision.ball2_id ? "ball" : (collision.type === "ball_pocket" ? "pocket" : "rail"),
      targetId: collision.ball2_id || undefined,
      angle: collision.angle,
      impulse: 0,
    })),
    type: line.type,
    probability: line.confidence,
    color: this.getTrajectoryColor(line.type),
  }));
}
```

## Data Processing Service (`/services/data-handlers.ts`)

**Lines 388-497**: `createProcessedGameState()` - **USES ARRAY INDEXING**

```typescript
private createProcessedGameState(stateData: GameStateData): ProcessedGameState {
  // Process balls with smoothing and prediction
  const processedBalls = stateData.balls.map((ball) => {
    const prevBall = previousState?.balls.find((b) => b.id === ball.id);

    // ← USES ARRAY INDEXING
    const deltaX = ball.position[0] - prevBall.position[0];
    const deltaY = ball.position[1] - prevBall.position[1];

    // Apply smoothing
    interpolatedPosition = [
      prevBall.position[0] + deltaX * smoothing,  // ← ARRAY FORMAT
      prevBall.position[1] + deltaY * smoothing,  // ← ARRAY FORMAT
    ] as [number, number];

    // Predict future position
    if (ball.velocity) {
      predictedPosition = [
        ball.position[0] + ball.velocity[0] * 2,  // ← ARRAY FORMAT
        ball.position[1] + ball.velocity[1] * 2,  // ← ARRAY FORMAT
      ] as [number, number];
    }
  });

  // Calculate cue trajectory
  const x = cue.position[0] + Math.cos(angleRad) * factor;  // ← ARRAY INDEXING
  const y = cue.position[1] + Math.sin(angleRad) * factor;  // ← ARRAY INDEXING

  // Aiming accuracy calculation
  const toBall = [
    ball.position[0] - cue.position[0],  // ← ARRAY INDEXING
    ball.position[1] - cue.position[1],  // ← ARRAY INDEXING
  ];

  // Table center calculation
  const avgX = table.corners.reduce((sum, corner) => sum + corner[0], 0);  // ← ARRAY INDEXING
  const avgY = table.corners.reduce((sum, corner) => sum + corner[1], 0);  // ← ARRAY INDEXING
}
```

## Rendering Components

### Ball Overlay (`/components/video/overlays/BallOverlay.tsx`)
**Uses OBJECT FORMAT (after conversion)**:

```typescript
// Line 22: Access ball position as object
const center = transform.videoToCanvas(ball.position);

// Lines 94-95: Velocity vector using object properties
x: ball.position.x + (ball.velocity.x / velocityMagnitude) * velocityScale,
y: ball.position.y + (ball.velocity.y / velocityMagnitude) * velocityScale,
```

### Trajectory Overlay (`/components/video/overlays/TrajectoryOverlay.tsx`)
**Uses OBJECT FORMAT**:

```typescript
// Line 46: Start point transformation
const startPoint = transform.videoToCanvas(points[0]);

// Line 66: Point transformation in loop
const currentPoint = transform.videoToCanvas(points[i]);

// Line 655: Collision position transformation
position: { x: collision.position[0], y: collision.position[1] },
```

### Cue Overlay (`/components/video/overlays/CueOverlay.tsx`)
**Uses OBJECT FORMAT**:

```typescript
// Line 21-22: Tip and tail positions
const tipPoint = transform.videoToCanvas(cue.tipPosition);
const tailPoint = transform.videoToCanvas(cue.tailPosition);

// Lines 116-117: Extended trajectory calculation
x: cue.tipPosition.x + Math.cos(cue.angle) * extensionLength,
y: cue.tipPosition.y + Math.sin(cue.angle) * extensionLength,
```

### Table Overlay (`/components/video/overlays/TableOverlay.tsx`)
**Uses OBJECT FORMAT**:

```typescript
// Line 28: Corner transformation
const startCorner = transform.videoToCanvas(table.corners[0]);

// Line 155-156: Center calculation
centerX += corner.x;
centerY += corner.y;

// Line 187-189: Bounds transformation
const topLeft = transform.videoToCanvas({ x: bounds.x, y: bounds.y });
const bottomRight = transform.videoToCanvas({
  x: bounds.x + bounds.width,
  y: bounds.y + bounds.height,
});
```

### LiveView Component (`/components/video/LiveView.tsx`)
**DIRECT WEBSOCKET ACCESS - EXPECTS ARRAY FORMAT**:

```typescript
// Line 56-58: Receives balls directly from WebSocket
if (message.type === 'state' && message.data?.balls) {
  setBalls(message.data.balls);  // ← Expects BallData with position: [x, y]
}

// Lines 120-121: Uses object access (assumes conversion happened)
const x = ball.position.x * scaleX;
const y = ball.position.y * scaleY;
```

**NOTE**: This component has a TYPE MISMATCH. It receives array-format balls but tries to access `.x` and `.y` properties!

### OverlayCanvas Component (`/components/video/OverlayCanvas.tsx`)
**Uses OBJECT FORMAT**:

```typescript
// Line 96: Ball position as object
const center = transform.videoToCanvas(ball.position);

// Lines 141-142: Velocity calculation using object properties
x: ball.position.x + ball.velocity.x * 50,
y: ball.position.y + ball.velocity.y * 50,
```

## Coordinate Transformation (`/utils/coordinates.ts`)

**All transformation functions expect and return OBJECT FORMAT**:

```typescript
// Lines 10-93: createCoordinateTransform
export function createCoordinateTransform(
  videoSize: Size2D,
  canvasSize: Size2D,
  transform: ViewportTransform = { x: 0, y: 0, scale: 1, rotation: 0 }
): CoordinateTransform {
  return {
    videoToCanvas: (point: Point2D): Point2D => {
      let x = point.x * uniformScale + offsetX;  // ← Uses .x property
      let y = point.y * uniformScale + offsetY;  // ← Uses .y property
      return { x, y };  // ← Returns object
    },

    canvasToVideo: (point: Point2D): Point2D => {
      // Reverse transformation
      return { x, y };  // ← Returns object
    },
  };
}
```

## Integration Tests (`/tests/integration/DetectionOverlayIntegration.test.ts`)

**SHOWS EXPLICIT ARRAY TO OBJECT CONVERSION**:

```typescript
// Line 129: Ball position conversion
position: { x: ballData.position[0], y: ballData.position[1] },

// Lines 158-163: Cue position conversion
tipPosition: { x: cueData.position[0], y: cueData.position[1] },
tailPosition: cueData.tip_position
  ? { x: cueData.tip_position[0], y: cueData.tip_position[1] }
  : {
      x: cueData.position[0] - Math.cos(cueData.angle) * (cueData.length || 100),
      y: cueData.position[1] - Math.sin(cueData.angle) * (cueData.length || 100)
    },

// Line 217: Collision position conversion
position: { x: collision.position[0], y: collision.position[1] },

// Line 352: Test data using array format
position: [ball.position[0] + i, ball.position[1] + i * 0.5] as [number, number],
```

## Impact Analysis

### Files That WILL BREAK with new dict format:

1. **`/types/api.ts`** - ALL interfaces expecting arrays need updating
   - `BallData.position`, `BallData.velocity`
   - `CueData.position`, `CueData.tip_position`
   - `TableData.corners`, `TableData.pockets`
   - `TrajectoryLine.start`, `TrajectoryLine.end`
   - `CollisionData.position`, `CollisionData.velocity_before`, `CollisionData.velocity_after`

2. **`/stores/VideoStore.ts`** - PRIMARY CONVERSION LOGIC
   - `handleGameStateMessage()` - Lines 548-619
   - `handleTrajectoryMessage()` - Lines 637-676
   - ALL array indexing `[0]` and `[1]` will fail

3. **`/services/data-handlers.ts`** - EXTENSIVE ARRAY USAGE
   - `createProcessedGameState()` - Lines 407-537
   - `calculateCueTrajectory()` - Lines 499-516
   - `calculateAimingAccuracy()` - Lines 518-552
   - `calculateTableCenter()` - Lines 554-562
   - `calculateTableBounds()` - Lines 564-579

4. **`/tests/integration/DetectionOverlayIntegration.test.ts`**
   - All conversion logic - Lines 129, 158-163, 217, 352

5. **`/components/video/LiveView.tsx`** - POSSIBLE TYPE MISMATCH
   - Currently receives balls from WebSocket (array format)
   - But tries to access `.x` and `.y` properties (object format)
   - Lines 120-121

### Files That Will Continue Working:

All rendering components that use the internal `Point2D` / object format will continue to work AFTER the conversion logic is updated:
- `/components/video/overlays/BallOverlay.tsx`
- `/components/video/overlays/TrajectoryOverlay.tsx`
- `/components/video/overlays/CueOverlay.tsx`
- `/components/video/overlays/TableOverlay.tsx`
- `/components/video/OverlayCanvas.tsx`
- `/utils/coordinates.ts`

## Backend Format Change Details

The backend is changing from:
```python
position = [x, y]  # Old array format
```

To:
```python
position = {
    "x": x,
    "y": y,
    "space": "screen_space",  # or "table_space", "world_space"
    "resolution": "4k"         # or "1080p", etc.
}
```

## Required Changes

### Phase 1: Update API Type Definitions

Change `/types/api.ts`:

```typescript
// OLD
export interface BallData {
  position: [number, number];
  velocity?: [number, number];
}

// NEW
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

### Phase 2: Update Conversion Logic

Update `/stores/VideoStore.ts`:

```typescript
// OLD
position: { x: ballData.position[0], y: ballData.position[1] }

// NEW
position: { x: ballData.position.x, y: ballData.position.y }
```

### Phase 3: Update Data Processing

Update `/services/data-handlers.ts`:

```typescript
// OLD
const deltaX = ball.position[0] - prevBall.position[0];

// NEW
const deltaX = ball.position.x - prevBall.position.x;
```

### Phase 4: Handle Metadata

Decide how to handle space metadata:
- Store it alongside coordinates?
- Use it for validation?
- Display it in debug overlays?

## Scale/Resolution Handling

Currently, the frontend handles resolution scaling in:

1. **Coordinate transformation** (`/utils/coordinates.ts`):
   - `createCoordinateTransform()` - Calculates scale factors
   - `videoToCanvas()` / `canvasToVideo()` - Apply transformations

2. **LiveView** (`/components/video/LiveView.tsx`):
   - Lines 114-115: Calculates scaleX/scaleY from video to canvas
   - Line 122: Applies scale to radius

3. **OverlayCanvas** (`/components/video/OverlayCanvas.tsx`):
   - Lines 45-50: Creates coordinate transform with video/canvas sizes
   - Uses transform for all rendering

**NO EXPLICIT RESOLUTION METADATA IS CURRENTLY USED**. The frontend infers scaling from video dimensions.

## Recommendations

1. **Update type definitions first** to match new backend format
2. **Update conversion logic** in VideoStore and data-handlers
3. **Consider storing space metadata** for debugging/validation
4. **Add migration notes** for any external consumers
5. **Update tests** to use new format
6. **Fix LiveView type mismatch** - it should use VideoStore instead of direct WebSocket

## Files Summary

### Files to Modify (6 total):
1. `/types/api.ts` - Type definitions
2. `/stores/VideoStore.ts` - Primary conversion logic
3. `/services/data-handlers.ts` - Data processing
4. `/tests/integration/DetectionOverlayIntegration.test.ts` - Tests
5. `/components/video/LiveView.tsx` - Fix type mismatch
6. (Optional) Add space metadata types

### Files That Work After Updates (13 total):
1. `/types/video.ts` - Already uses Point2D objects
2. `/stores/types/index.ts` - Already uses Point2D objects
3. `/utils/coordinates.ts` - Already uses Point2D objects
4. `/components/video/overlays/BallOverlay.tsx`
5. `/components/video/overlays/TrajectoryOverlay.tsx`
6. `/components/video/overlays/CueOverlay.tsx`
7. `/components/video/overlays/TableOverlay.tsx`
8. `/components/video/overlays/CalibrationOverlay.tsx`
9. `/components/video/OverlayCanvas.tsx`
10. `/stores/GameStore.ts`
11. `/stores/VisionStore.ts`
12. `/stores/ConfigStore.ts`
13. All other rendering and UI components

## Search Patterns Used

```bash
# Position references
grep -r "position" frontend/web/src

# Coordinate references
grep -r "coordinate" frontend/web/src

# Object property access
grep -r "\.x\|\.y" frontend/web/src

# Array indexing
grep -r "\[0\]\|\[1\]" frontend/web/src

# Scale/resolution
grep -r "scale\|resolution\|viewport\|transform" frontend/web/src
```

## Conclusion

The frontend has a **clean architecture** with a **single conversion point** in VideoStore. The impact of the backend change is **limited and manageable**:

- **6 files** need updates to handle the new dictionary format
- **13+ files** will continue working without changes
- **No hardcoded coordinate values** found
- **Good separation** between API types (arrays) and internal types (objects)
- **Coordinate transformations** are well-abstracted in utilities

The main risk is the **LiveView component** which appears to have a type mismatch that may already be broken.
