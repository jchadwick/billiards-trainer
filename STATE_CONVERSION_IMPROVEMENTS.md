# State Conversion Improvements in IntegrationService

## Summary

Improved state conversion in `backend/integration_service.py` to create proper BallState/CueState objects with validation, consistency checks, and better error handling.

## Changes Made

### 1. New Helper Module: `integration_service_conversion_helpers.py`

Created a dedicated module with the `StateConversionHelpers` class that provides:

#### Key Features:
- **Validation during conversion**: Validates positions, velocities, confidence thresholds, and angles
- **Physics validation integration**: Uses `PhysicsValidator` to validate ball states
- **Proper error handling**: Returns `None` on critical failures, logs warnings for non-critical issues
- **Conversion statistics**: Tracks conversion counts and validation warnings

#### Helper Methods:

**`vision_ball_to_ball_state(ball, is_target, timestamp, validate)`**
- Converts Vision `Ball` to Core `BallState` with full validation
- Validates ball confidence against minimum threshold
- Validates position is within reasonable bounds (0-3m x, 0-2m y)
- Validates velocity doesn't exceed max (10 m/s default)
- Clamps invalid velocities instead of rejecting
- Uses physics validator to check ball state consistency
- Generates consistent ball IDs using track_id or position hash
- Returns `None` if critical validation fails

**`vision_cue_to_cue_state(detected_cue, timestamp, validate)`**
- Converts Vision `CueStick` to Core `CueState` with full validation
- Validates cue confidence against minimum threshold
- Validates and normalizes cue angle to 0-360 degrees
- Validates tip position is reasonable
- Validates estimated force is within reasonable range (0-50N)
- Returns `None` if critical validation fails

**Validation Helpers:**
- `_validate_position(x, y, obj_type)`: Checks positions are in valid range
- `_validate_velocity(vx, vy, obj_type)`: Checks velocities are reasonable
- `_generate_ball_id(ball, is_target)`: Creates consistent ball IDs
- `get_conversion_stats()`: Returns conversion statistics

### 2. Integration into IntegrationService

#### New Imports:
```python
from backend.core.validation.physics import PhysicsValidator
from backend.core.validation.table_state import TableStateValidator
from backend.integration_service_conversion_helpers import StateConversionHelpers
```

#### New Initialization:
```python
# Initialize validators
self.physics_validator = PhysicsValidator()
self.table_state_validator = TableStateValidator()

# Initialize state conversion helpers
self.state_converter = StateConversionHelpers(
    config=self.config, physics_validator=self.physics_validator
)
```

#### New Validation Thresholds (configurable):
```python
# Validation thresholds
self.max_ball_velocity = self.config.get("integration.max_ball_velocity_m_per_s", 10.0)
self.max_position_x = self.config.get("integration.max_position_x", 3.0)
self.max_position_y = self.config.get("integration.max_position_y", 2.0)
```

#### New Public Methods:
```python
def vision_cue_to_cue_state(detected_cue, timestamp=None) -> Optional[CueState]
def vision_ball_to_ball_state(ball, is_target=False, timestamp=None) -> Optional[BallState]
```

These are the recommended methods for converting Vision detections to Core states.

#### Updated Private Methods:
- `_create_cue_state()` - Now delegates to `vision_cue_to_cue_state()` (marked DEPRECATED)
- `_create_ball_state()` - Now delegates to `vision_ball_to_ball_state()` (marked DEPRECATED)
- `_create_ball_states()` - Now uses `vision_ball_to_ball_state()` and filters out None results

### 3. Updated Trajectory Calculation

The trajectory calculation in `_check_trajectory_calculation()` now:

1. Uses validated state conversion with proper error handling:
```python
# Create CueState from detected cue using state converter (with validation)
cue_state = self.state_converter.vision_cue_to_cue_state(
    detection.cue, timestamp=detection.timestamp
)
if cue_state is None:
    logger.error("Failed to convert cue state - aborting trajectory calculation")
    return
```

2. Validates each ball conversion:
```python
# Create list of BallState for other balls using state converter (with validation)
other_ball_states = []
for ball in detection.balls:
    if ball == target_ball:
        continue  # Skip the target ball
    ball_state_converted = self.state_converter.vision_ball_to_ball_state(
        ball, is_target=False, timestamp=detection.timestamp
    )
    if ball_state_converted is not None:
        other_ball_states.append(ball_state_converted)
```

3. Validates table state before trajectory calculation (this was already in place)

## Validation Checks

### Ball State Validation:
- ✅ Confidence >= min_ball_confidence (default 0.1)
- ✅ Position X in range [0, max_position_x] (default 3.0m)
- ✅ Position Y in range [0, max_position_y] (default 2.0m)
- ✅ Velocity magnitude <= max_ball_velocity (default 10.0 m/s)
- ✅ Radius > 0 (uses standard 0.028575m if invalid)
- ✅ Mass > 0 (uses standard 0.17kg)
- ✅ Ball type is valid
- ✅ No overlapping balls (via physics validator)
- ✅ Position on table (when table provided)

### Cue State Validation:
- ✅ Confidence >= min_cue_confidence (default 0.1)
- ✅ Angle normalized to [0, 360] degrees
- ✅ Tip position in valid range
- ✅ Estimated force in range [0, 50] Newtons
- ✅ Length > 0

## Consistency with Trajectory Calculation

The conversion now ensures consistency by:

1. **Proper BallState objects**: All required fields set correctly
   - Proper Vector2D for position and velocity
   - Correct mass (0.17 kg) and radius (0.028575m or detected)
   - Zero spin (Vision doesn't detect spin yet)
   - Timestamp set from detection

2. **Proper CueState objects**: All required fields for physics
   - tip_position and impact_point as Vector2D
   - Estimated force from config (default 5.0N)
   - Angle in degrees
   - Zero elevation (2D vision only)
   - Timestamp set from detection

3. **Validated states**: Physics validator ensures states are physically valid
   - No impossible velocities
   - No overlapping balls
   - Positions within table bounds

4. **Consistent IDs**: Ball IDs generated consistently
   - Uses track_id when available (preferred)
   - Falls back to position-based hash
   - Special "target_ball" ID for trajectory engine

## Error Handling

### Non-Critical Warnings (logged but conversion continues):
- Low confidence detections
- Positions slightly out of bounds
- High velocities (clamped to max)
- Unusual angles (normalized)

### Critical Errors (conversion fails, returns None):
- Missing required fields
- Invalid data types
- Physics validation failures (currently just warned)

## Statistics and Monitoring

Conversion statistics available via `state_converter.get_conversion_stats()`:
- `ball_conversions`: Total ball conversions
- `cue_conversions`: Total cue conversions
- `validation_warnings`: Total validation warnings
- `ball_warning_rate`: Warning rate per ball conversion

Periodic logging every 100 conversions includes:
- Conversion details (position, velocity, confidence)
- Cumulative statistics
- Warning counts

## Configuration Options

New configuration keys for `config/current.json`:

```json
{
  "integration": {
    "max_ball_velocity_m_per_s": 10.0,
    "max_position_x": 3.0,
    "max_position_y": 2.0,
    "min_ball_confidence": 0.1,
    "min_cue_confidence": 0.1
  }
}
```

## Migration Guide

### For Code Using IntegrationService:

**Old way (still works but deprecated):**
```python
ball_state = integration_service._create_ball_state(vision_ball)
cue_state = integration_service._create_cue_state(vision_cue)
```

**New way (recommended):**
```python
# With automatic timestamp
ball_state = integration_service.vision_ball_to_ball_state(vision_ball)
cue_state = integration_service.vision_cue_to_cue_state(vision_cue)

# With explicit timestamp
ball_state = integration_service.vision_ball_to_ball_state(
    vision_ball,
    timestamp=detection.timestamp
)

# Check for conversion failure
if ball_state is None:
    logger.error("Ball state conversion failed")
```

### For Direct Access to Helpers:

```python
from backend.integration_service_conversion_helpers import StateConversionHelpers

converter = StateConversionHelpers(config=my_config)
ball_state = converter.vision_ball_to_ball_state(vision_ball)
stats = converter.get_conversion_stats()
```

## Testing Recommendations

1. **Unit Tests**: Test conversion helpers with various inputs
   - Valid balls with good confidence
   - Low confidence balls
   - Balls with high velocities
   - Balls with invalid positions
   - Cues with various angles

2. **Integration Tests**: Test trajectory calculation with validated states
   - Normal gameplay scenarios
   - Edge cases (balls near table boundaries)
   - High-speed shots
   - Multiple ball collisions

3. **Validation Tests**: Verify physics validator catches issues
   - Overlapping balls
   - Impossible velocities
   - Out-of-bounds positions

## Files Modified

1. **backend/integration_service.py** (modified)
   - Added imports for validators and helpers
   - Added validator and helper initialization
   - Added validation threshold configuration
   - Added public conversion methods
   - Updated trajectory calculation to use validated conversion
   - Marked old methods as DEPRECATED

2. **backend/integration_service_conversion_helpers.py** (new)
   - Complete StateConversionHelpers class
   - Ball and cue conversion with validation
   - Helper methods for validation
   - Statistics tracking

3. **STATE_CONVERSION_IMPROVEMENTS.md** (new)
   - This documentation file

## Future Enhancements

1. **Spin Detection**: When vision module supports spin detection, add to ball state
2. **Elevation Detection**: When vision supports cue elevation, add to cue state
3. **Force Estimation**: Use cue velocity tracking to estimate strike force
4. **Adaptive Thresholds**: Adjust validation thresholds based on detection quality
5. **Strict Mode**: Add config option to reject conversions that fail validation
6. **Unit Conversion**: Support multiple unit systems (metric/imperial)
7. **Calibration Integration**: Use camera calibration data for position validation
