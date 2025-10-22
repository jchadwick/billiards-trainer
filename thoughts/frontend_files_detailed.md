# Frontend Files - Detailed Coordinate Usage

## Files Requiring Changes

### 1. /types/api.ts
**Impact**: HIGH - Defines all API response types
**Lines affected**: 78-141

#### Current Format:
```typescript
export interface BallData {
  id: string;
  position: [number, number];        // Line 80
  radius: number;
  color: string;
  velocity?: [number, number];       // Line 83
  confidence: number;
  visible: boolean;
}

export interface CueData {
  angle: number;
  position: [number, number];        // Line 90
  detected: boolean;
  confidence: number;
  length?: number;
  tip_position?: [number, number];   // Line 94
}

export interface TableData {
  corners: [number, number][];       // Line 98
  pockets: [number, number][];       // Line 99
  rails?: Record<string, any>[];
  calibrated: boolean;
  dimensions?: Record<string, number>;
}

export interface TrajectoryLine {
  start: [number, number];           // Line 115
  end: [number, number];             // Line 116
  type: 'primary' | 'reflection' | 'collision';
  confidence: number;
}

export interface CollisionData {
  position: [number, number];        // Line 122
  ball_id?: string;
  ball1_id?: string;
  ball2_id?: string;
  type?: string;
  angle: number;
  velocity_before?: [number, number];  // Line 128
  velocity_after?: [number, number];   // Line 129
  time_to_collision?: number;
}
```

#### Required Changes:
```typescript
export interface Position2D {
  x: number;
  y: number;
  space?: 'screen_space' | 'table_space' | 'world_space';
  resolution?: '1080p' | '4k' | '8k';
}

export interface BallData {
  id: string;
  position: Position2D;
  radius: number;
  color: string;
  velocity?: Position2D;
  confidence: number;
  visible: boolean;
}

export interface CueData {
  angle: number;
  position: Position2D;
  detected: boolean;
  confidence: number;
  length?: number;
  tip_position?: Position2D;
}

export interface TableData {
  corners: Position2D[];
  pockets: Position2D[];
  rails?: Record<string, any>[];
  calibrated: boolean;
  dimensions?: Record<string, number>;
}

export interface TrajectoryLine {
  start: Position2D;
  end: Position2D;
  type: 'primary' | 'reflection' | 'collision';
  confidence: number;
}

export interface CollisionData {
  position: Position2D;
  ball_id?: string;
  ball1_id?: string;
  ball2_id?: string;
  type?: string;
  angle: number;
  velocity_before?: Position2D;
  velocity_after?: Position2D;
  time_to_collision?: number;
}
```

---

### 2. /stores/VideoStore.ts
**Impact**: CRITICAL - Primary conversion point
**Lines affected**: 548-676

#### Ball Conversion (Lines 548-559):
```typescript
// OLD
const balls: Ball[] = gameState.balls.map((ballData: BallData) => ({
  id: ballData.id,
  position: { x: ballData.position[0], y: ballData.position[1] },
  radius: ballData.radius,
  type: this.inferBallType(ballData.id, ballData.color),
  number: this.inferBallNumber(ballData.id, ballData.color),
  velocity: ballData.velocity
    ? { x: ballData.velocity[0], y: ballData.velocity[1] }
    : { x: 0, y: 0 },
  confidence: ballData.confidence,
  color: ballData.color,
}));

// NEW
const balls: Ball[] = gameState.balls.map((ballData: BallData) => ({
  id: ballData.id,
  position: { x: ballData.position.x, y: ballData.position.y },
  radius: ballData.radius,
  type: this.inferBallType(ballData.id, ballData.color),
  number: this.inferBallNumber(ballData.id, ballData.color),
  velocity: ballData.velocity
    ? { x: ballData.velocity.x, y: ballData.velocity.y }
    : { x: 0, y: 0 },
  confidence: ballData.confidence,
  color: ballData.color,
}));
```

#### Cue Conversion (Lines 564-587):
```typescript
// OLD
if (gameState.cue && gameState.cue.detected) {
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
    // ...
  };
}

// NEW
if (gameState.cue && gameState.cue.detected) {
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
    // ...
  };
}
```

#### Table Conversion (Lines 594-601):
```typescript
// OLD
table = {
  corners: gameState.table.corners.map((corner) => ({
    x: corner[0],
    y: corner[1],
  })),
  pockets: gameState.table.pockets.map((pocket) => ({
    x: pocket[0],
    y: pocket[1],
  })),
  // ...
};

// NEW
table = {
  corners: gameState.table.corners.map((corner) => ({
    x: corner.x,
    y: corner.y,
  })),
  pockets: gameState.table.pockets.map((pocket) => ({
    x: pocket.x,
    y: pocket.y,
  })),
  // ...
};
```

#### Trajectory Conversion (Lines 650-663):
```typescript
// OLD
const trajectories: Trajectory[] = trajectoryData.lines.map((line, index) => ({
  ballId: trajectoryData.collisions[0]?.ball1_id || movingBallId,
  points: [
    { x: line.start[0], y: line.start[1] },
    { x: line.end[0], y: line.end[1] },
  ],
  collisions: trajectoryData.collisions.map((collision) => ({
    position: { x: collision.position[0], y: collision.position[1] },
    // ...
  })),
  // ...
}));

// NEW
const trajectories: Trajectory[] = trajectoryData.lines.map((line, index) => ({
  ballId: trajectoryData.collisions[0]?.ball1_id || movingBallId,
  points: [
    { x: line.start.x, y: line.start.y },
    { x: line.end.x, y: line.end.y },
  ],
  collisions: trajectoryData.collisions.map((collision) => ({
    position: { x: collision.position.x, y: collision.position.y },
    // ...
  })),
  // ...
}));
```

---

### 3. /services/data-handlers.ts
**Impact**: HIGH - Extensive array indexing
**Lines affected**: 407-579

#### Ball Position Processing (Lines 407-432):
```typescript
// OLD
const deltaX = ball.position[0] - prevBall.position[0];
const deltaY = ball.position[1] - prevBall.position[1];

interpolatedPosition = [
  prevBall.position[0] + deltaX * smoothing,
  prevBall.position[1] + deltaY * smoothing,
] as [number, number];

if (ball.velocity) {
  predictedPosition = [
    ball.position[0] + ball.velocity[0] * 2,
    ball.position[1] + ball.velocity[1] * 2,
  ] as [number, number];
}

// NEW
const deltaX = ball.position.x - prevBall.position.x;
const deltaY = ball.position.y - prevBall.position.y;

interpolatedPosition = {
  x: prevBall.position.x + deltaX * smoothing,
  y: prevBall.position.y + deltaY * smoothing,
};

if (ball.velocity) {
  predictedPosition = {
    x: ball.position.x + ball.velocity.x * 2,
    y: ball.position.y + ball.velocity.y * 2,
  };
}
```

#### Cue Trajectory Calculation (Lines 508-513):
```typescript
// OLD
for (let i = 0; i <= 10; i++) {
  const factor = (i / 10) * distance;
  const x = cue.position[0] + Math.cos(angleRad) * factor;
  const y = cue.position[1] + Math.sin(angleRad) * factor;
  points.push([x, y]);
}

// NEW
for (let i = 0; i <= 10; i++) {
  const factor = (i / 10) * distance;
  const x = cue.position.x + Math.cos(angleRad) * factor;
  const y = cue.position.y + Math.sin(angleRad) * factor;
  points.push({ x, y });
}
```

#### Aiming Accuracy (Lines 535-538):
```typescript
// OLD
const toBall = [
  ball.position[0] - cue.position[0],
  ball.position[1] - cue.position[1],
];

// NEW
const toBall = {
  x: ball.position.x - cue.position.x,
  y: ball.position.y - cue.position.y,
};
```

#### Table Center (Lines 555-561):
```typescript
// OLD
const avgX = table.corners.reduce((sum, corner) => sum + corner[0], 0) / table.corners.length;
const avgY = table.corners.reduce((sum, corner) => sum + corner[1], 0) / table.corners.length;
return [avgX, avgY];

// NEW
const avgX = table.corners.reduce((sum, corner) => sum + corner.x, 0) / table.corners.length;
const avgY = table.corners.reduce((sum, corner) => sum + corner.y, 0) / table.corners.length;
return { x: avgX, y: avgY };
```

#### Table Bounds (Lines 570-578):
```typescript
// OLD
const xs = table.corners.map((corner) => corner[0]);
const ys = table.corners.map((corner) => corner[1]);

// NEW
const xs = table.corners.map((corner) => corner.x);
const ys = table.corners.map((corner) => corner.y);
```

---

### 4. /tests/integration/DetectionOverlayIntegration.test.ts
**Impact**: MEDIUM - Test conversion logic
**Lines affected**: 129, 158-163, 217, 352

#### Ball Conversion (Line 129):
```typescript
// OLD
position: { x: ballData.position[0], y: ballData.position[1] },

// NEW
position: { x: ballData.position.x, y: ballData.position.y },
```

#### Cue Conversion (Lines 158-163):
```typescript
// OLD
tipPosition: { x: cueData.position[0], y: cueData.position[1] },
tailPosition: cueData.tip_position
  ? { x: cueData.tip_position[0], y: cueData.tip_position[1] }
  : {
      x: cueData.position[0] - Math.cos(cueData.angle) * (cueData.length || 100),
      y: cueData.position[1] - Math.sin(cueData.angle) * (cueData.length || 100)
    },

// NEW
tipPosition: { x: cueData.position.x, y: cueData.position.y },
tailPosition: cueData.tip_position
  ? { x: cueData.tip_position.x, y: cueData.tip_position.y }
  : {
      x: cueData.position.x - Math.cos(cueData.angle) * (cueData.length || 100),
      y: cueData.position.y - Math.sin(cueData.angle) * (cueData.length || 100)
    },
```

#### Collision Conversion (Line 217):
```typescript
// OLD
position: { x: collision.position[0], y: collision.position[1] },

// NEW
position: { x: collision.position.x, y: collision.position.y },
```

---

### 5. /components/video/LiveView.tsx
**Impact**: MEDIUM - Has type mismatch bug
**Lines affected**: 56-58, 120-121

#### Current Issue:
```typescript
// Line 56-58: Receives balls from WebSocket (array format)
if (message.type === 'state' && message.data?.balls) {
  setBalls(message.data.balls);  // BallData[] with position: [x, y]
}

// Lines 120-121: Tries to use object properties
const x = ball.position.x * scaleX;  // ERROR: position is array!
const y = ball.position.y * scaleY;
```

#### Recommended Fix:
Remove direct WebSocket access and use VideoStore instead:

```typescript
import { observer } from 'mobx-react-lite';
import { useStores } from '../../stores/context';

export const LiveView: React.FC<LiveViewProps> = observer(({ className = '' }) => {
  const { videoStore } = useStores();
  const balls = videoStore.currentBalls;  // Already converted to object format

  // Remove WebSocket connection code
  // Use balls from videoStore instead
});
```

---

## Files That DON'T Need Changes

### Already Using Object Format

1. **/types/video.ts** - Defines `Point2D`, `Vector2D` as objects
2. **/stores/types/index.ts** - Uses `Point2D` objects
3. **/utils/coordinates.ts** - All functions use `Point2D` objects
4. **/components/video/overlays/BallOverlay.tsx** - Uses `ball.position.x/y`
5. **/components/video/overlays/TrajectoryOverlay.tsx** - Uses `point.x/y`
6. **/components/video/overlays/CueOverlay.tsx** - Uses `cue.tipPosition.x/y`
7. **/components/video/overlays/TableOverlay.tsx** - Uses `corner.x/y`
8. **/components/video/overlays/CalibrationOverlay.tsx** - Uses `corner.screenPosition.x/y`
9. **/components/video/OverlayCanvas.tsx** - Uses `ball.position` as object
10. **/stores/GameStore.ts** - Uses `ball.position: Point2D`
11. **/stores/VisionStore.ts** - Uses `Point2D` throughout

All UI and rendering components are safe!

---

## Migration Checklist

### Phase 1: Type Updates
- [ ] Update `/types/api.ts` - Add `Position2D` interface
- [ ] Update `BallData.position` and `BallData.velocity`
- [ ] Update `CueData.position` and `CueData.tip_position`
- [ ] Update `TableData.corners` and `TableData.pockets`
- [ ] Update `TrajectoryLine.start` and `TrajectoryLine.end`
- [ ] Update `CollisionData.position`, `velocity_before`, `velocity_after`

### Phase 2: Conversion Logic
- [ ] Update `/stores/VideoStore.ts` - `handleGameStateMessage()`
- [ ] Update ball position conversion
- [ ] Update ball velocity conversion
- [ ] Update cue position conversion
- [ ] Update cue tip_position conversion
- [ ] Update table corners conversion
- [ ] Update table pockets conversion
- [ ] Update `/stores/VideoStore.ts` - `handleTrajectoryMessage()`
- [ ] Update trajectory line start/end conversion
- [ ] Update collision position conversion

### Phase 3: Data Processing
- [ ] Update `/services/data-handlers.ts` - ball position processing
- [ ] Update delta calculations
- [ ] Update interpolation
- [ ] Update prediction
- [ ] Update cue trajectory calculation
- [ ] Update aiming accuracy calculation
- [ ] Update table center calculation
- [ ] Update table bounds calculation

### Phase 4: Tests & Bug Fixes
- [ ] Update `/tests/integration/DetectionOverlayIntegration.test.ts`
- [ ] Fix `/components/video/LiveView.tsx` type mismatch

### Phase 5: Verification
- [ ] Run TypeScript compiler - no errors
- [ ] Run tests - all passing
- [ ] Test ball rendering - positions correct
- [ ] Test cue rendering - tip and tail correct
- [ ] Test table rendering - corners and pockets aligned
- [ ] Test trajectories - lines render correctly
- [ ] Test collisions - points appear correctly
- [ ] Check console - no runtime errors

---

## Metadata Handling (Optional)

If you want to preserve space metadata:

```typescript
// In VideoStore conversion
const balls: Ball[] = gameState.balls.map((ballData: BallData) => ({
  id: ballData.id,
  position: {
    x: ballData.position.x,
    y: ballData.position.y,
    // Store metadata if needed
    _space: ballData.position.space,
    _resolution: ballData.position.resolution,
  },
  // ...
}));

// Create debug overlay to show metadata
function DebugMetadataOverlay({ ball }) {
  return (
    <div>
      Position: ({ball.position.x}, {ball.position.y})
      Space: {ball.position._space}
      Resolution: {ball.position._resolution}
    </div>
  );
}
```
