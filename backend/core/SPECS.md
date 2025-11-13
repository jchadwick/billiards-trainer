# Core Module Specification

## Module Purpose

The Core module serves as the central business logic layer, managing game state, calculating physics-based trajectories, coordinating between other modules, and maintaining the overall system state. It acts as the intelligent hub that transforms raw detection data into meaningful game insights and predictions.

## Functional Requirements

### 1. Game State Management

#### 1.1 State Tracking
- **FR-CORE-001**: Maintain current positions of all balls on the table
- **FR-CORE-002**: Track cue stick position and orientation
- **FR-CORE-003**: Store table dimensions and pocket locations
- **FR-CORE-004**: Maintain game history for last N frames
- **FR-CORE-005**: Detect and track game events (shots, collisions, pocketed balls)
- **FR-CORE-006A**: Assign unique persistent IDs to detected balls
- **FR-CORE-006B**: Track ball number/type as optional metadata (separate from ID)

#### 1.2 State Validation
- **FR-CORE-006**: Validate detection data for physical consistency (basic position/velocity checks)
- **FR-CORE-007**: Filter noise and impossible state transitions (optional enhancement)
- **FR-CORE-008**: Interpolate missing data points (optional enhancement)
- **FR-CORE-009**: Detect and correct detection errors (auto-correction for overlaps/OOB implemented)
- **FR-CORE-010**: Maintain confidence scores for state elements

**Implementation Note**: Basic state validation is implemented with auto-correction for common issues (overlapping balls, out-of-bounds positions, excessive velocities). Advanced noise filtering and interpolation are optional enhancements not required for core functionality.

#### 1.3 State Synchronization
- **FR-CORE-011**: Synchronize state across all connected clients
- **FR-CORE-012**: Handle state conflicts and resolution
- **FR-CORE-013**: Provide state snapshots for new connections
- **FR-CORE-014**: Support state rollback and replay
- **FR-CORE-015**: Maintain state consistency during updates

#### 1.4 State Publishing
- **FR-CORE-016A**: Publish ball positions at configurable intervals (default 500ms)
- **FR-CORE-016B**: Publish ball-in-motion events in real-time (immediately)
- **FR-CORE-016C**: Include ball ID in all state events
- **FR-CORE-016D**: Include optional ball number/type when available
- **FR-CORE-016E**: Distinguish between static ball updates and motion events

### 2. Physics Engine

#### 2.1 Trajectory Calculation
- **FR-CORE-016**: Calculate linear ball trajectories based on cue angle
- **FR-CORE-017**: Compute collision points with other balls
- **FR-CORE-018**: Predict ball paths after collisions
- **FR-CORE-019**: Calculate rebounds off table cushions
- **FR-CORE-020**: Determine if balls will be pocketed

#### 2.2 Collision Handling
- **FR-CORE-022**: Calculate friction and deceleration
- **FR-CORE-025**: Predict multi-ball collision chains (max 2 levels: cue ball hits → those balls hit)

### 3. Game Logic

#### 3.1 Shot Analysis
- **FR-CORE-031**: Identify shot type (break, safety, bank, etc.) - basic implementation

#### 3.2 Game Rules (Training Mode - Optional)
- **FR-CORE-036**: Track game type (8-ball, 9-ball, practice)
- **FR-CORE-037**: Monitor turn order and fouls (optional - training system focus)
- **FR-CORE-038**: Validate legal shots per game rules (optional - training system focus)
- **FR-CORE-039**: Track score and game progress (optional - training system focus)
- **FR-CORE-040**: Detect game completion conditions (optional - training system focus)

**Implementation Note**: This is a training system focused on shot visualization and trajectory prediction. Full game rule enforcement (turn order, fouls, scoring) is optional and not required for core training functionality. Basic game type tracking and illegal shot detection are implemented for context.

### 4. Trajectory Prediction

#### 4.1 Basic Trajectory Prediction
- **FR-CORE-041**: Predict ball positions for next 5 seconds
- **FR-CORE-042**: Calculate collision sequences (max 2 levels)

### 5. Module Coordination

#### 5.1 Data Flow Management
- **FR-CORE-051**: Receive detection data from Vision module
- **FR-CORE-052**: Send state updates to API module
- **FR-CORE-053**: Provide trajectory data to Projector module
- **FR-CORE-054**: Exchange configuration with Config module
- **FR-CORE-055**: Coordinate module initialization and shutdown

#### 5.2 Event Management
- **FR-CORE-056**: Generate system events for state changes
- **FR-CORE-057**: Handle event subscriptions from modules
- **FR-CORE-058**: Maintain event history and replay
- **FR-CORE-059**: Filter and route events by type
- **FR-CORE-060**: Support custom event handlers

## Non-Functional Requirements

### Performance Requirements
- **NFR-CORE-001**: Update game state within 10ms of receiving detection data
- **NFR-CORE-002**: Calculate trajectories in < 20ms
- **NFR-CORE-003**: Support 60+ state updates per second
- **NFR-CORE-004**: Handle 100+ concurrent trajectory requests
- **NFR-CORE-005**: Maintain state history for last 1000 frames

### Accuracy Requirements
- **NFR-CORE-006**: Trajectory prediction within 5% of actual path
- **NFR-CORE-007**: Collision detection accurate to 1mm
- **NFR-CORE-008**: Angle calculations within 0.5 degrees
- **NFR-CORE-010**: Physics simulation realistic to casual observation

### Reliability Requirements
- **NFR-CORE-011**: No state corruption during concurrent access
- **NFR-CORE-012**: Graceful handling of invalid input data
- **NFR-CORE-013**: Automatic recovery from calculation errors
- **NFR-CORE-014**: Maintain state consistency across failures
- **NFR-CORE-015**: Support state persistence and recovery

### Scalability Requirements
- **NFR-CORE-016**: Support multiple game tables simultaneously
- **NFR-CORE-017**: Handle varying complexity (6-ball to 15-ball games)
- **NFR-CORE-018**: Scale physics calculations with available CPU
- **NFR-CORE-019**: Efficient memory usage with large state histories
- **NFR-CORE-020**: Support distributed processing if needed

## Interface Specifications

### Core Module Interface

```python
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass
from enum import Enum
import numpy as np

class ShotType(Enum):
    BREAK = "break"
    DIRECT = "direct"
    BANK = "bank"
    COMBINATION = "combination"
    SAFETY = "safety"
    MASSE = "masse"

class GameType(Enum):
    EIGHT_BALL = "8ball"
    NINE_BALL = "9ball"
    STRAIGHT_POOL = "straight"
    PRACTICE = "practice"

@dataclass
class Vector2D:
    """2D vector for positions and velocities"""
    x: float
    y: float

    def magnitude(self) -> float:
        return np.sqrt(self.x**2 + self.y**2)

    def normalize(self) -> 'Vector2D':
        mag = self.magnitude()
        return Vector2D(self.x/mag, self.y/mag) if mag > 0 else Vector2D(0, 0)

@dataclass
class BallState:
    """Complete ball state information"""
    id: str  # Unique persistent identifier for tracking
    position: Vector2D
    velocity: Vector2D
    radius: float
    mass: float = 0.17  # kg, standard pool ball
    is_cue_ball: bool = False
    is_pocketed: bool = False
    number: Optional[int] = None  # Optional: ball number (1-15) or type
    is_moving: bool = False  # Track if ball is currently in motion

@dataclass
class TableState:
    """Pool table state information"""
    width: float  # mm
    height: float  # mm
    pocket_positions: List[Vector2D]
    pocket_radius: float
    cushion_elasticity: float = 0.85
    surface_friction: float = 0.2
    surface_slope: float = 0.0  # degrees

@dataclass
class CueState:
    """Cue stick state information"""
    tip_position: Vector2D
    angle: float  # degrees

@dataclass
class Collision:
    """Collision prediction information"""
    time: float  # seconds from now
    position: Vector2D
    ball1_id: str
    ball2_id: Optional[str]  # None for cushion collision
    type: str  # "ball", "cushion", "pocket"
    resulting_velocities: Optional[Dict[str, Vector2D]] = None

@dataclass
class Trajectory:
    """Ball trajectory information"""
    ball_id: str
    points: List[Vector2D]  # Path points
    collisions: List[Collision]
    final_position: Vector2D
    final_velocity: Vector2D
    time_to_rest: float  # seconds
    will_be_pocketed: bool
    pocket_id: Optional[int] = None

@dataclass
class ShotAnalysis:
    """Basic shot type identification"""
    shot_type: ShotType

@dataclass
class GameState:
    """Complete game state"""
    timestamp: float
    frame_number: int
    balls: List[BallState]
    table: TableState
    cue: Optional[CueState]
    game_type: GameType
    current_player: Optional[int]
    scores: Dict[int, int]
    is_break: bool
    last_shot: Optional[ShotAnalysis]

class CoreModule:
    """Main core logic interface"""

    def __init__(self, config: Dict[str, Any]):
        """Initialize with configuration"""
        pass

    def update_state(self, detection_data: Dict) -> GameState:
        """Update game state from detection data"""
        pass

    def calculate_trajectory(self,
                           ball_id: str,
                           initial_velocity: Vector2D,
                           time_limit: float = 10.0) -> List[Trajectory]:
        """Calculate ball trajectories from initial conditions"""
        pass

    def analyze_shot(self, target_ball: Optional[str] = None) -> ShotAnalysis:
        """Identify basic shot type"""
        pass

    def predict_trajectory(self, time_horizon: float = 5.0) -> List[Trajectory]:
        """Predict ball trajectories with max 2 collision levels"""
        pass

    def validate_state(self, state: GameState) -> Tuple[bool, List[str]]:
        """Validate game state for consistency"""
        pass

    def get_state_history(self, frames: int = 100) -> List[GameState]:
        """Get historical game states"""
        pass

    def reset_game(self, game_type: GameType = GameType.PRACTICE) -> None:
        """Reset game to initial state"""
        pass

    def subscribe_to_events(self, event_type: str, callback: callable) -> None:
        """Subscribe to state change events"""
        pass
```

### Configuration Schema

```python
from pydantic import BaseModel, Field
from typing import Optional, Dict

class PhysicsConfig(BaseModel):
    """Physics simulation parameters"""
    gravity: float = 9.81  # m/s²
    rolling_friction: float = 0.01
    cushion_coefficient: float = 0.85
    max_iterations: int = 1000
    time_step: float = 0.001  # seconds
    medium_shot_speed: float = 2.0  # m/s - fixed speed for all shots

class PredictionConfig(BaseModel):
    """Trajectory prediction parameters"""
    max_prediction_time: float = 10.0  # seconds
    prediction_resolution: int = 100  # points per trajectory
    collision_threshold: float = 0.001  # meters
    max_collision_depth: int = 2  # Maximum collision levels to calculate

class ValidationConfig(BaseModel):
    """State validation parameters"""
    max_ball_velocity: float = 10.0  # m/s
    min_ball_separation: float = 0.001  # meters
    position_tolerance: float = 0.005  # meters
    velocity_tolerance: float = 0.1  # m/s
    enable_physics_validation: bool = True
    enable_continuity_check: bool = True

class StatePublishingConfig(BaseModel):
    """State publishing configuration"""
    static_ball_interval_ms: int = 500  # Interval for publishing static ball positions
    motion_event_immediate: bool = True  # Publish motion events immediately
    include_ball_numbers: bool = True  # Include optional ball numbers when available
    motion_velocity_threshold: float = 0.01  # m/s threshold to consider ball moving

class CoreConfig(BaseModel):
    """Core module configuration"""
    physics: PhysicsConfig
    prediction: PredictionConfig
    validation: ValidationConfig
    publishing: StatePublishingConfig
    state_history_size: int = 1000
    event_buffer_size: int = 100
    update_frequency: float = 60.0  # Hz
    enable_logging: bool = True
    debug_mode: bool = False
```

## Processing Algorithms

### Physics Simulation

```python
class PhysicsEngine:
    """Physics simulation algorithms"""

    def calculate_ball_trajectory(self,
                                 ball: BallState,
                                 table: TableState,
                                 other_balls: List[BallState],
                                 time_limit: float) -> Trajectory:
        """
        Calculate trajectory using numerical integration:

        1. Initialize state vectors
        2. For each time step:
           a. Calculate forces (friction, air resistance)
           b. Update velocity and position
           c. Check for collisions
           d. Apply collision response
           e. Check for pockets
        3. Continue until ball stops or time limit
        4. Generate trajectory points
        """
        pass

    def detect_collision(self,
                        ball1: BallState,
                        ball2: BallState,
                        dt: float) -> Optional[Collision]:
        """
        Detect ball-ball collision:

        1. Calculate relative position and velocity
        2. Solve quadratic for collision time
        3. Verify collision occurs within time step
        4. Calculate collision point
        5. Return collision details
        """
        pass

    def resolve_collision(self,
                         ball1: BallState,
                         ball2: BallState,
                         collision: Collision) -> Tuple[Vector2D, Vector2D]:
        """
        Calculate post-collision velocities:

        1. Calculate collision normal
        2. Decompose velocities (normal/tangent)
        3. Apply conservation laws (momentum and energy)
        4. Return new velocities
        """
        pass
```

### State Validation

```python
class StateValidator:
    """Game state validation algorithms"""

    def validate_positions(self,
                          balls: List[BallState],
                          table: TableState) -> Tuple[bool, List[str]]:
        """
        Validate ball positions:

        1. Check all balls within table bounds
        2. Verify minimum separation between balls
        3. Check for overlapping balls
        4. Validate against pocketed status
        5. Return validation result and errors
        """
        pass

    def validate_physics(self,
                        current: GameState,
                        previous: GameState,
                        dt: float) -> Tuple[bool, List[str]]:
        """
        Validate physical consistency:

        1. Check velocity continuity
        2. Verify energy conservation
        3. Validate momentum conservation
        4. Check for impossible accelerations
        5. Return validation result
        """
        pass

    def correct_errors(self,
                      state: GameState,
                      errors: List[str]) -> GameState:
        """
        Attempt to correct detected errors:

        1. Separate overlapping balls
        2. Clamp excessive velocities
        3. Move balls inside table
        4. Interpolate missing data
        5. Return corrected state
        """
        pass
```

## Success Criteria

### Functional Success Criteria

1. **State Management**
   - Maintain accurate game state for 99%+ of frames
   - Detect 100% of shot events
   - Track all balls without loss
   - Recover from detection errors within 3 frames

2. **Physics Accuracy**
   - Trajectory predictions match reality within 5cm
   - Collision predictions accurate to 95%+
   - Realistic ball behavior to human observation
   - Correct pocket predictions 90%+ of the time

### Performance Success Criteria

1. **Processing Speed**
   - State updates complete in < 10ms
   - Trajectory calculation < 20ms for 5-second prediction
   - Support 60 FPS processing rate
   - No dropped frames during normal operation

2. **Resource Usage**
   - Memory usage < 500MB for state management
   - CPU usage < 30% for physics calculations
   - Efficient caching of calculated trajectories
   - Linear scaling with number of balls

3. **Responsiveness**
   - Immediate response to detection updates
   - Real-time trajectory updates during aiming
   - Smooth animation of predictions

### Reliability Success Criteria

1. **Stability**
   - No crashes during 24-hour operation
   - Graceful handling of all error conditions
   - Automatic recovery from transient failures
   - Consistent behavior across sessions

2. **Accuracy Validation**
   - Physics validation catches 95%+ of errors
   - State correction successful 90%+ of attempts
   - No accumulated drift over time
   - Maintains calibration accuracy

## Testing Requirements

### Unit Testing
- Test physics calculations with known scenarios
- Validate collision detection algorithms
- Test state management operations
- Verify trajectory calculations
- Coverage target: 90%

### Integration Testing
- Test with real detection data
- Verify module communication
- Test state synchronization
- Validate event system
- Test configuration changes

### Physics Testing
- Compare with real-world measurements
- Validate against physics simulations
- Test edge cases (clusters, near-misses)
- Verify conservation laws
- Benchmark calculation times

### Performance Testing
- Stress test with maximum ball count
- Test sustained high-frequency updates
- Measure memory usage over time
- Profile CPU usage patterns
- Test concurrent access

## Implementation Guidelines

### Code Structure
```python
core/
├── __init__.py
├── game_state.py        # State management
├── physics/
│   ├── __init__.py
│   ├── engine.py       # Main physics engine
│   ├── collision.py    # Collision detection/response
│   └── trajectory.py   # Trajectory calculation
├── analysis/
│   ├── __init__.py
│   ├── shot.py         # Basic shot type identification
│   └── prediction.py   # Simple trajectory prediction
├── validation/
│   ├── __init__.py
│   ├── state.py        # State validation
│   ├── physics.py      # Physics validation
│   └── correction.py   # Error correction
├── events/
│   ├── __init__.py
│   ├── manager.py      # Event management
│   └── handlers.py     # Event handlers
├── models.py           # Data models
└── utils/
    ├── __init__.py
    ├── geometry.py     # Geometric utilities
    ├── math.py         # Math helpers
    └── cache.py        # Caching utilities
```

### Key Dependencies
- **numpy**: Numerical computations
- **scipy**: Advanced physics calculations
- **numba**: JIT compilation for performance
- **asyncio**: Asynchronous operations
- **pydantic**: Data validation

### Development Priorities
1. Implement basic state management
2. Add simple physics engine (fixed medium speed, no spin)
3. Implement collision detection (max 2 levels)
4. Add trajectory calculation
5. Implement state validation
6. Optimize performance
