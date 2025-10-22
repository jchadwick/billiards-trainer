# Group 6: Analysis & Prediction Systems Migration Plan

**Agent**: Agent 6
**Status**: Ready for Implementation (when dependencies complete)
**Date**: 2025-10-21

---

## Overview

This document provides a detailed migration plan for updating the Analysis & Prediction systems to use 4K coordinates as specified in the 4K standardization plan.

---

## Files in Scope

### 1. shot.py (885 lines)
**Purpose**: Shot analysis algorithms
**Key Changes**:
- Convert aim point calculations from meters to 4K pixels
- Update distance calculations to use pixel measurements
- Ensure all returned Vector2D instances have scale metadata
- Update shot difficulty calculations for pixel-based system

**Critical Methods**:
- `_calculate_optimal_aim_point()` - Returns aim point Vector2D
- `analyze_shot()` - Returns ShotAnalysis with recommended_aim_point
- `calculate_difficulty()` - Uses distance calculations
- `_calculate_angle_difficulty()` - Geometric calculations
- `_calculate_obstacle_difficulty()` - Distance-based calculations
- `_calculate_precision_difficulty()` - Pocket distance calculations

### 2. assistance.py (1062 lines)
**Purpose**: Player assistance features
**Key Changes**:
- Update aiming guide generation to use 4K coordinates
- Convert safe zone positions to 4K pixels
- Ensure ghost ball positions have scale metadata
- Update power recommendations for pixel-based system

**Critical Methods**:
- `suggest_optimal_aim_point()` - Returns Vector2D aim point
- `_generate_aiming_guide()` - Returns AimingGuide with Vector2D positions
- `_calculate_safe_zones()` - Returns SafeZone with Vector2D centers
- `_find_defensive_zones()` - Returns zones with Vector2D positions
- `_find_scratch_safe_zones()` - Safe zone calculations
- `_find_position_play_zones()` - Position zone calculations
- `_calculate_ghost_ball_position()` - Returns Vector2D ghost ball position

### 3. prediction.py (666 lines)
**Purpose**: Outcome prediction algorithms
**Key Changes**:
- Update trajectory simulations to use 4K coordinates
- Ensure all trajectory points have scale metadata
- Convert final ball positions to 4K pixels
- Update collision position tracking

**Critical Methods**:
- `predict_shot_outcome()` - Returns ShotPrediction with trajectories
- `predict_outcomes()` - Returns list of PredictedOutcome
- `_simulate_shot_physics()` - Simulates ball movement, returns positions
- `_analyze_primary_outcome()` - Returns PredictedOutcome with Vector2D positions
- `_update_ball_positions()` - Updates ball positions during simulation
- `_calculate_shot_velocity()` - Returns Vector2D velocity

### 4. example_cue_pointing.py (198 lines)
**Purpose**: Example code demonstrating cue pointing
**Key Changes**:
- Update all Vector2D creation to use 4K coordinates
- Add scale metadata to all positions
- Update documentation to reflect 4K system

---

## Migration Strategy

### Step 1: Identify All Vector2D Creation Points

**In shot.py**:
```python
# Line 405: _calculate_optimal_aim_point return
return Vector2D(aim_x, aim_y)
# BECOMES:
return Vector2D.from_4k(aim_x, aim_y)

# Line 408: Fallback return
return Vector2D(target_ball.position.x, target_ball.position.y)
# BECOMES:
return Vector2D.from_4k(target_ball.position.x, target_ball.position.y)
```

**In assistance.py**:
```python
# Line 236-239: normalize() creates new Vector2D
direction = Vector2D(
    ghost_ball_pos.x - target_ball.position.x,
    ghost_ball_pos.y - target_ball.position.y,
).normalize()
# BECOMES: Ensure normalize() preserves scale metadata

# Line 241-244: aim point calculation
aim_point = Vector2D(
    target_ball.position.x - direction.x * target_ball.radius,
    target_ball.position.y - direction.y * target_ball.radius,
)
# BECOMES:
aim_point = Vector2D.from_4k(
    target_ball.position.x - direction.x * target_ball.radius,
    target_ball.position.y - direction.y * target_ball.radius,
)

# Line 821-824, 966, 995: Safe zone centers
center=Vector2D(ball.position.x, ball.position.y)
# BECOMES:
center=Vector2D.from_4k(ball.position.x, ball.position.y)

# Line 864-867, 877-879: Spin recommendations
return Vector2D(0.3, 0.0)
# BECOMES:
return Vector2D.from_4k(0.3, 0.0)  # Or keep as relative spin?
```

**In prediction.py**:
```python
# Line 198: Creating Vector2D from tuple
final_ball_positions={ball.id: Vector2D(*trajectory.final_position)}
ball_trajectories={ball.id: [Vector2D(*pt) for pt in trajectory.points]}
# BECOMES:
final_ball_positions={ball.id: Vector2D.from_4k(*trajectory.final_position)}
ball_trajectories={ball.id: [Vector2D.from_4k(*pt) for pt in trajectory.points]}

# Line 287: Collision position
"position": collision.get("position", Vector2D(0, 0)).to_dict()
# BECOMES:
"position": collision.get("position", Vector2D.from_4k(0, 0)).to_dict()

# Line 536-539: Shot velocity calculation
return Vector2D(
    velocity_magnitude * math.cos(angle_rad),
    velocity_magnitude * math.sin(angle_rad),
)
# BECOMES:
# NOTE: Velocity is RELATIVE, not a position - may not need scale?
# OR: Use scale to indicate it's in 4K pixel/second units
return Vector2D.from_4k(
    velocity_magnitude * math.cos(angle_rad),
    velocity_magnitude * math.sin(angle_rad),
)

# Line 615-618: Ball center for collision
position=Vector2D(
    (ball1.position.x + ball2.position.x) / 2,
    (ball1.position.y + ball2.position.y) / 2,
)
# BECOMES:
position=Vector2D.from_4k(
    (ball1.position.x + ball2.position.x) / 2,
    (ball1.position.y + ball2.position.y) / 2,
)
```

### Step 2: Update Distance Calculations

All distance calculations should continue to work in pixels. Ensure that when calculating distances, both vectors are in the same scale (preferably 4K canonical).

**Example from shot.py line 217-220**:
```python
distance = self.geometry_utils.distance_between_points(
    (cue_ball.position.x, cue_ball.position.y),
    (target_ball.position.x, target_ball.position.y),
)
```

**After migration**: If ball positions are in 4K, this continues to work. If they might be in different scales, convert to 4K first:
```python
cue_pos_4k = cue_ball.position.to_4k_canonical()
target_pos_4k = target_ball.position.to_4k_canonical()
distance = self.geometry_utils.distance_between_points(
    (cue_pos_4k.x, cue_pos_4k.y),
    (target_pos_4k.x, target_pos_4k.y),
)
```

### Step 3: Handle Max Distance Normalization

**Example from shot.py line 221**:
```python
max_distance = math.sqrt(game_state.table.width**2 + game_state.table.height**2)
factors["distance"] = min(distance / max_distance, 1.0)
```

After migration, `table.width` and `table.height` will be in pixels (not meters), so this calculation remains the same structurally.

### Step 4: Special Cases - Relative vs Absolute Vectors

**Positions**: Absolute coordinates - MUST have scale metadata pointing to 4K
```python
ball_position = Vector2D.from_4k(1920, 1080)  # scale=[1.0, 1.0]
```

**Velocities**: Relative vectors - Could use scale to indicate units (pixels/second in 4K space)
```python
velocity = Vector2D.from_4k(100, 50)  # 100 px/s, 50 px/s in 4K space
```

**Directions**: Unit vectors - Scale might be [1.0, 1.0] as canonical or could be omitted
```python
direction = Vector2D(1, 0).normalize()  # Unit vector
# After normalize, should have scale from original or be set to canonical
```

**Accelerations**: Relative vectors - Similar to velocity
```python
acceleration = Vector2D.from_4k(10, 0)  # 10 px/s² in 4K space
```

### Step 5: Verification Points

For each file, verify:

1. ✅ All `Vector2D(x, y)` calls replaced with `Vector2D.from_4k(x, y)` or appropriate factory
2. ✅ All returned Vector2D instances have scale metadata
3. ✅ Distance calculations use consistent scale (all in 4K)
4. ✅ Angle calculations remain valid (angles are scale-independent)
5. ✅ Trajectory points all have scale metadata
6. ✅ Ghost ball positions have scale metadata
7. ✅ Safe zone centers have scale metadata
8. ✅ Collision positions have scale metadata

---

## Testing Strategy

### Unit Tests

1. **Aim Point Accuracy**:
   ```python
   def test_aim_point_has_scale():
       """Verify aim point has proper scale metadata."""
       aim_point = analyzer.suggest_optimal_aim_point(game_state, "ball_1")
       assert hasattr(aim_point, 'scale')
       assert aim_point.scale == [1.0, 1.0]  # 4K canonical
   ```

2. **Trajectory Prediction Accuracy**:
   ```python
   def test_trajectory_predictions_match_old_system():
       """Verify predictions match old meter-based system."""
       # Setup identical scenario in both systems
       old_result = old_predictor.predict(old_game_state)
       new_result = new_predictor.predict(new_game_state_4k)

       # Convert to same units for comparison
       old_final_meters = old_result.final_positions["ball_1"]
       new_final_pixels = new_result.final_positions["ball_1"].to_4k_canonical()
       new_final_meters = pixels_to_meters(new_final_pixels)

       # Must match within 0.1%
       assert abs(new_final_meters.x - old_final_meters.x) / old_final_meters.x < 0.001
   ```

3. **Safe Zone Positions**:
   ```python
   def test_safe_zones_have_scale():
       """Verify all safe zones have scale metadata."""
       zones = assistance.calculate_safe_zones(game_state)
       for zone in zones:
           assert hasattr(zone.center, 'scale')
           assert zone.center.scale == [1.0, 1.0]
   ```

### Integration Tests

1. **Full Analysis Pipeline**:
   ```python
   def test_full_analysis_pipeline_4k():
       """Test complete analysis flow in 4K."""
       # 1. Create game state in 4K
       game_state = create_4k_game_state()

       # 2. Analyze shot
       shot_analysis = analyzer.analyze_shot(game_state, "ball_1")

       # 3. Verify aim point
       assert shot_analysis.recommended_aim_point.scale == [1.0, 1.0]

       # 4. Get assistance
       assistance_pkg = engine.provide_assistance(game_state)
       assert assistance_pkg.aiming_guide.target_point.scale == [1.0, 1.0]

       # 5. Predict outcome
       prediction = predictor.predict_shot_outcome(game_state, shot_analysis)
       assert prediction.primary_outcome.final_ball_positions["ball_1"].scale == [1.0, 1.0]
   ```

### Accuracy Tests

1. **Compare with Reference Data**:
   - Load 100 historical shot scenarios
   - Run analysis in both old and new systems
   - Verify results match within tolerance
   - Check that success probabilities are similar
   - Validate trajectory predictions align

---

## Implementation Checklist

### shot.py
- [ ] Update `_calculate_optimal_aim_point()` to use `Vector2D.from_4k()`
- [ ] Update `analyze_shot()` to ensure aim point has scale
- [ ] Update `_calculate_shot_angle()` (should be scale-independent)
- [ ] Verify `calculate_difficulty()` distance calculations work in pixels
- [ ] Update `_calculate_angle_difficulty()` if needed
- [ ] Update `_calculate_obstacle_difficulty()` distance checks
- [ ] Update `_calculate_precision_difficulty()` pocket distance calculations
- [ ] Update `_ball_intersects_line()` geometric calculations
- [ ] Verify all helper methods return scaled vectors

### assistance.py
- [ ] Update `suggest_optimal_aim_point()` return value
- [ ] Update `_generate_aiming_guide()` to use 4K coordinates
- [ ] Update `_calculate_ghost_ball_position()` return value
- [ ] Update `_find_defensive_zones()` zone centers
- [ ] Update `_find_scratch_safe_zones()` zone centers
- [ ] Update `_find_position_play_zones()` zone centers
- [ ] Update `_calculate_optimal_spin()` (check if spin needs scale)
- [ ] Verify all Vector2D creations use `from_4k()`

### prediction.py
- [ ] Update `predict_outcomes()` trajectory points
- [ ] Update `_simulate_shot_physics()` position tracking
- [ ] Update `_analyze_primary_outcome()` final positions
- [ ] Update `_calculate_shot_velocity()` (check velocity scale handling)
- [ ] Update `_update_ball_positions()` position updates
- [ ] Update `_detect_collisions()` collision positions
- [ ] Verify all trajectory points have scale metadata

### example_cue_pointing.py
- [ ] Update all example Vector2D creations
- [ ] Add scale metadata to all positions
- [ ] Update documentation with 4K examples
- [ ] Add examples showing scale usage

### Tests
- [ ] Add unit tests for scale metadata verification
- [ ] Add integration tests for full pipeline
- [ ] Add accuracy comparison tests vs old system
- [ ] Add regression tests with reference scenarios
- [ ] Verify performance (must not degrade > 10%)

---

## Success Criteria

✅ All Vector2D instances in analysis modules have scale metadata
✅ Aim points calculated in 4K canonical coordinates
✅ Trajectory predictions use 4K coordinates
✅ Safe zones positioned in 4K coordinates
✅ Analysis accuracy matches old system (±0.1%)
✅ All tests passing
✅ Performance within 10% of baseline
✅ No hardcoded values (use constants_4k)

---

## Rollback Plan

If issues are encountered:

1. **Identify Issue Scope**:
   - Accuracy degradation?
   - Performance regression?
   - Scale metadata errors?

2. **Isolate Failing Component**:
   - shot.py only?
   - assistance.py only?
   - prediction.py only?
   - All of Group 6?

3. **Rollback Procedure**:
   ```bash
   # Revert specific file
   git checkout HEAD~1 backend/core/analysis/shot.py

   # Or revert entire group
   git revert <commit-range-for-group6>
   ```

4. **Analyze Root Cause**:
   - Review implementation vs plan
   - Check for missed Vector2D conversions
   - Verify dependency implementations (Groups 4 & 5)

5. **Fix and Retry**:
   - Apply fix
   - Re-run tests
   - Validate accuracy

---

## Dependencies Summary

| Group | Status | Required For Group 6 |
|-------|--------|---------------------|
| Group 1 | ⚠️ Partial | Constants available, but Vector2D not migrated |
| Group 2 | ❌ Pending | **CRITICAL** - Need Vector2D with mandatory scale |
| Group 3 | ❌ Pending | **CRITICAL** - Need BallState/TableState in 4K |
| Group 4 | ❌ Pending | **CRITICAL** - Need physics with 4K support |
| Group 5 | ❌ Pending | **CRITICAL** - Need collision with 4K support |

**Ready to Start**: ❌ **NO** - Waiting for Groups 2, 3, 4, 5

---

## Next Actions

1. ✅ Document migration plan (this file)
2. ⏳ Monitor Groups 2-5 progress
3. ⏳ Review actual implementations when complete
4. ⏳ Refine migration strategy based on real implementations
5. ⏳ Begin implementation when dependencies ready

---

**Document Status**: ✅ Complete and Ready
**Implementation Status**: ⏸️ Blocked - Awaiting Dependencies
**Estimated Duration**: 3 days (once unblocked)
