# Table State Analysis: Trajectory Calculation Integration

## Executive Summary

This analysis verifies that table state, particularly the `playing_area_corners` field, is properly handled throughout the trajectory calculation pipeline. The system correctly passes table state from vision detection through core game state management to the trajectory calculator.

**Status: ✅ VERIFIED - Table state is properly integrated with validation now added**

## Analysis Results

### 1. Integration Service Has Access to Current TableState ✅

**File:** `/Users/jchadwick/code/billiards-trainer/backend/integration_service.py`

**Lines 665-673:** The integration service correctly retrieves the current table state from Core:

```python
# Get table state from Core's current state
if not self.core._current_state:
    logger.warning(
        "Trajectory calculation skipped: No current game state available. "
        "Core module may not be initialized properly."
    )
    return

table_state = self.core._current_state.table
```

**Lines 675-689:** Added validation to ensure table state is valid before trajectory calculation:

```python
# Validate table state before trajectory calculation
is_valid, validation_errors = self.table_state_validator.validate_for_trajectory(
    table_state, require_playing_area=False
)
if not is_valid:
    logger.error(
        f"Trajectory calculation skipped: Invalid table state. "
        f"Errors: {'; '.join(validation_errors)}"
    )
    return

# Log validation summary periodically
if should_log:
    summary = self.table_state_validator.get_validation_summary(table_state)
    logger.debug(f"Table state validation: {summary}")
```

### 2. TableState Includes playing_area_corners ✅

**File:** `/Users/jchadwick/code/billiards-trainer/backend/core/models.py`

**Lines ~45-50:** TableState dataclass definition includes `playing_area_corners`:

```python
@dataclass
class TableState:
    width: float  # meters
    height: float  # meters
    pocket_positions: list[Vector2D]
    pocket_radius: float = 0.0635
    cushion_elasticity: float = 0.85
    surface_friction: float = 0.2
    surface_slope: float = 0.0
    cushion_height: float = 0.064
    playing_area_corners: Optional[list[Vector2D]] = None  # ✅ Present
```

**Purpose:** Calibrated playing area corners define the actual table boundaries as detected by vision, allowing for accurate trajectory calculations even with perspective distortion.

### 3. Trajectory Calculation Receives Proper TableState ✅

**File:** `/Users/jchadwick/code/billiards-trainer/backend/integration_service.py`

**Lines 691-698:** Table state is passed to trajectory calculator:

```python
# Call trajectory_calculator.predict_multiball_cue_shot() directly
multiball_result = self.trajectory_calculator.predict_multiball_cue_shot(
    cue_state=cue_state,
    ball_state=ball_state,
    table_state=table_state,  # ✅ Passed with playing_area_corners
    other_balls=other_ball_states,
    quality=TrajectoryQuality.LOW,
    max_collision_depth=5,
)
```

### 4. Trajectory Calculator Uses playing_area_corners ✅

**File:** `/Users/jchadwick/code/billiards-trainer/backend/core/physics/trajectory.py`

**Lines 397-399:** Trajectory calculator calls cushion intersection detection:

```python
cushion_hit = self._find_cushion_intersection(
    current_pos, direction, table, ball_radius  # ✅ Full table state passed
)
```

**File:** `/Users/jchadwick/code/billiards-trainer/backend/core/collision/geometric_collision.py`

**Lines ~275-295:** GeometricCollisionDetector properly uses playing_area_corners:

```python
def find_cushion_intersection(
    self,
    position: Vector2D,
    direction: Vector2D,
    table: TableState,
    ball_radius: float,
) -> Optional[GeometricCollision]:
    """Find intersection with table cushions.

    Supports both calibrated playing area corners and rectangular bounds.
    """
    # Use playing area corners if available (calibrated system)
    if table.playing_area_corners and len(table.playing_area_corners) == 4:
        return self._find_cushion_intersection_polygon(  # ✅ Uses corners
            position, direction, table, ball_radius
        )

    # Fall back to rectangular bounds
    return self._find_cushion_intersection_rectangular(
        position, direction, table, ball_radius
    )
```

**Lines ~300-350:** The polygon-based collision detection properly processes corners:

```python
def _find_cushion_intersection_polygon(
    self,
    position: Vector2D,
    direction: Vector2D,
    table: TableState,
    ball_radius: float,
) -> Optional[GeometricCollision]:
    """Find cushion intersection for calibrated playing area polygon."""
    corners = table.playing_area_corners  # ✅ Extract corners
    if not corners or len(corners) != 4:
        return None

    # Define 4 edges: top, right, bottom, left
    edges = [
        (corners[0], corners[1], "top"),
        (corners[1], corners[2], "right"),
        (corners[2], corners[3], "bottom"),
        (corners[3], corners[0], "left"),
    ]
    # ... collision detection using edges ...
```

### 5. Table State Flows From Vision to Core ✅

**File:** `/Users/jchadwick/code/billiards-trainer/backend/integration_service.py`

**Lines 496-517:** Detection data conversion includes table with corners:

```python
# Convert table
if detection.table:
    table = detection.table
    detection_data["table"] = {
        "corners": [
            {"x": float(corner[0]), "y": float(corner[1])}
            for corner in table.corners  # ✅ Corners from vision
        ],
        "width": table.width,
        "height": table.height,
        "pockets": [...],
    }
```

**File:** `/Users/jchadwick/code/billiards-trainer/backend/core/game_state.py`

**Lines 290-321:** GameStateManager extracts corners from detection:

```python
def _extract_table_state(self, detection_data: dict[str, Any]) -> TableState:
    """Extract table state from detection data."""
    table_data = detection_data.get("table")

    # Extract corners for playing area
    playing_area_corners = None
    if "corners" in table_data:
        corners = []
        for corner in table_data["corners"]:
            if isinstance(corner, dict):
                corners.append(Vector2D(corner.get("x", 0.0), corner.get("y", 0.0)))
        if corners:
            playing_area_corners = corners  # ✅ Stored in TableState

    # Create TableState with extracted data
    return TableState(
        width=table_width,
        height=table_height,
        pocket_positions=pocket_positions,
        playing_area_corners=playing_area_corners,  # ✅ Included
        ...
    )
```

### 6. Video Debugger Demonstrates Proper Loading ✅

**File:** `/Users/jchadwick/code/billiards-trainer/tools/video_debugger.py`

**Lines 143-191:** Video debugger loads corners from config and scales them:

```python
def _load_table_state_from_config(self) -> TableState:
    """Load table state from saved configuration file."""
    try:
        config_path = Path(__file__).parent.parent / "config" / "current.json"

        # Load config
        with open(config_path, 'r') as f:
            config = json.load(f)

        # Extract table config
        table_config = config.get("table", {})
        playing_area_corners_data = table_config.get("playing_area_corners")
        calibration_res = table_config.get("calibration_resolution", {})

        # Convert corner dicts to Vector2D objects
        playing_area_corners = [
            Vector2D(corner["x"], corner["y"])
            for corner in playing_area_corners_data  # ✅ Load from config
        ]

        # Create table state with corners
        table_state = TableState.standard_9ft_table()
        table_state.playing_area_corners = playing_area_corners  # ✅ Set corners

        return table_state
```

**Lines 254-274:** Video debugger scales corners to match video resolution:

```python
if self.table_state.width != frame_width or self.table_state.height != frame_height:
    # Scale playing area corners from calibration resolution to current frame resolution
    if hasattr(self, 'calibration_resolution') and self.table_state.playing_area_corners:
        calib_width, calib_height = self.calibration_resolution

        # Use the TableState method to scale the corners
        self.table_state.scale_playing_area_corners(
            from_width=calib_width,
            from_height=calib_height,
            to_width=frame_width,
            to_height=frame_height
        )  # ✅ Properly scaled
```

## New Validation System

### TableStateValidator

**File:** `/Users/jchadwick/code/billiards-trainer/backend/core/validation/table_state.py`

A new comprehensive validator has been created to ensure table state is valid before trajectory calculations:

**Key Features:**
1. **Dimension Validation:** Ensures width, height, pocket positions, and physics parameters are valid
2. **Corner Validation:** Checks that playing_area_corners form a valid convex quadrilateral
3. **Geometry Validation:** Validates corner coordinates are within table bounds
4. **Area Validation:** Ensures minimum reasonable polygon area
5. **Flexible Requirements:** Can optionally require playing_area_corners or allow fallback to rectangular bounds

**Validation Methods:**
- `validate_for_trajectory()`: Main validation method that checks all aspects
- `get_validation_summary()`: Human-readable summary for logging
- `_validate_corner_geometry()`: Ensures corners form valid polygon
- `_calculate_polygon_area()`: Uses shoelace formula to compute area

**Integration:**
- Integrated into `IntegrationService.__init__()` (line 161)
- Called before each trajectory calculation (lines 675-689)
- Provides informative error messages when validation fails
- Logs validation summary periodically for debugging

## Data Flow Diagram

```
┌──────────────────┐
│  Vision Module   │
│  (Detection)     │
└────────┬─────────┘
         │ DetectionResult with table.corners
         ▼
┌──────────────────┐
│ Integration      │
│ Service          │
│ _convert_        │
│ detection_to_    │
│ core_format()    │
└────────┬─────────┘
         │ detection_data dict with corners
         ▼
┌──────────────────┐
│ Core Module      │
│ GameStateManager │
│ _extract_table_  │
│ state()          │
└────────┬─────────┘
         │ TableState with playing_area_corners
         ▼
┌──────────────────┐
│ Core Module      │
│ _current_state.  │
│ table            │
└────────┬─────────┘
         │ TableState reference
         ▼
┌──────────────────┐
│ Integration      │
│ Service          │
│ _check_trajectory│
│ _calculation()   │
└────────┬─────────┘
         │ table_state with corners
         ▼
┌──────────────────┐
│ TableState       │
│ Validator        │
│ validate_for_    │
│ trajectory()     │
└────────┬─────────┘
         │ validated table_state
         ▼
┌──────────────────┐
│ Trajectory       │
│ Calculator       │
│ predict_multiball│
│ _cue_shot()      │
└────────┬─────────┘
         │ table_state with corners
         ▼
┌──────────────────┐
│ Geometric        │
│ Collision        │
│ Detector         │
│ find_cushion_    │
│ intersection()   │
└────────┬─────────┘
         │ Uses playing_area_corners if present
         ▼
┌──────────────────┐
│ _find_cushion_   │
│ intersection_    │
│ polygon()        │
└──────────────────┘
```

## Validation Points

### Current Validation ✅
1. **Integration Service** checks for existence of Core state
2. **Integration Service** validates table state before trajectory calculation
3. **Geometric Collision Detector** checks for presence of playing_area_corners
4. **Geometric Collision Detector** validates corner array length (must be 4)
5. **TableState.__post_init__()** validates basic dimensions and pocket count

### Recommended Additional Validation

While the current system is robust, consider these enhancements:

1. **Corner Coordinate Bounds** - Already implemented in TableStateValidator
2. **Corner Order Validation** - Ensure corners are in consistent order (already validated via area calculation)
3. **Resolution Scaling Logging** - Already present in video_debugger.py
4. **Corner Calibration Confidence** - Could track quality of calibration

## Configuration Example

**File:** `/Users/jchadwick/code/billiards-trainer/config/current.json`

Example of how playing_area_corners should be stored:

```json
{
  "table": {
    "width": 1920,
    "height": 1080,
    "playing_area_corners": [
      {"x": 245.3, "y": 123.7},
      {"x": 1674.2, "y": 89.1},
      {"x": 1745.8, "y": 989.4},
      {"x": 174.5, "y": 1023.9}
    ],
    "calibration_resolution": {
      "width": 1920,
      "height": 1080
    },
    "pocket_positions": [...]
  }
}
```

## Testing Recommendations

1. **Unit Test TableStateValidator:**
   - Test validation with valid table state
   - Test validation with missing playing_area_corners
   - Test validation with invalid corners (wrong count, negative coordinates, collinear)
   - Test validation with invalid physics parameters

2. **Integration Test:**
   - Load config with playing_area_corners
   - Process frames through vision → core → integration
   - Verify trajectory calculations use calibrated corners
   - Verify validation catches invalid states

3. **Visual Test (video_debugger.py):**
   - Run with calibrated table config
   - Verify playing area polygon is drawn correctly
   - Verify trajectories respect playing area boundaries
   - Test corner scaling with different video resolutions

## Conclusion

The table state handling in trajectory calculations is **properly implemented and now validated**. The system:

1. ✅ Correctly loads playing_area_corners from configuration
2. ✅ Properly passes table state through vision → core → integration pipeline
3. ✅ Successfully uses playing_area_corners in geometric collision detection
4. ✅ Falls back to rectangular bounds when corners are not available
5. ✅ Scales corners appropriately for different video resolutions
6. ✅ **NEW:** Validates table state before trajectory calculation
7. ✅ **NEW:** Provides informative error messages when validation fails
8. ✅ **NEW:** Logs validation summary for debugging

The validation system ensures that:
- Invalid table states are caught before trajectory calculation
- Errors are clearly reported with actionable messages
- System degrades gracefully when calibration is incomplete
- Debugging is easier with validation summaries

## Files Modified

1. **NEW:** `/Users/jchadwick/code/billiards-trainer/backend/core/validation/table_state.py`
   - Complete table state validation implementation

2. **UPDATED:** `/Users/jchadwick/code/billiards-trainer/backend/integration_service.py`
   - Added TableStateValidator import and initialization
   - Added validation check before trajectory calculation
   - Added periodic validation summary logging

3. **UPDATED:** `/Users/jchadwick/code/billiards-trainer/backend/core/validation/__init__.py`
   - Exported TableStateValidator and TableStateValidationError

## Next Steps

1. Add unit tests for TableStateValidator
2. Add integration tests verifying validation catches errors
3. Consider adding calibration quality metrics
4. Monitor validation logs in production to identify common issues
