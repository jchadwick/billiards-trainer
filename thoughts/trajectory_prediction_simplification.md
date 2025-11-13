# Trajectory Prediction Simplification

## Summary

Simplified the billiards trainer shot prediction system to focus on core trajectory prediction only, removing advanced features for a Phase 1 implementation.

## Changes Made

### 1. Specifications (`backend/core/SPECS.md`)
**Removed:**
- FR-CORE-043, FR-CORE-044, FR-CORE-045 (final positions, success probability, problem detection)
- Advanced physics specs (spin, masse, jump shots, variable power)
- Force estimation specs
- Shot difficulty calculation specs
- Success probability specs
- Alternative shot suggestions specs
- Risk factor analysis specs

**Kept:**
- Basic trajectory calculation (FR-CORE-016 to FR-CORE-020)
- 2-level collision detection (FR-CORE-025, clarified to max 2 levels)
- Friction and deceleration (FR-CORE-022)
- Basic shot type identification (FR-CORE-031)

### 2. Prediction Module (`backend/core/analysis/prediction.py`)
**Before:** 680 lines
**After:** 122 lines (82% reduction)

**Removed:**
- `OutcomePredictor` class (full outcome prediction with probabilities)
- `OutcomeType`, `PredictedOutcome`, `ShotPrediction` dataclasses
- Success probability calculations
- Alternative outcome generation
- Prediction confidence calculations
- Variable power predictions
- Spin/English calculations

**Added:**
- `TrajectoryPredictor` class - simple trajectory prediction only
- Fixed medium power (2.0 m/s)
- Max 2 collision levels (hardcoded)
- Returns `dict[ball_id, list[Vector2D]]` - just trajectory points

### 3. Shot Analysis (`backend/core/analysis/shot.py`)
**Before:** 903 lines
**After:** 190 lines (79% reduction)

**Removed:**
- `calculate_difficulty()` and all difficulty sub-calculations
- `_calculate_success_probability()`
- `get_alternative_shots()`
- `_identify_potential_problems()`
- `_calculate_risk_factors()`
- `_predict_shot_outcome()`
- All difficulty calculation methods
- All risk calculation methods

**Kept:**
- Basic shot type identification (DIRECT, BANK, COMBINATION, BREAK)
- Basic legality checking (placeholder)
- Path blocking detection

### 4. Trajectory Calculator (`backend/core/physics/trajectory.py`)
**Changed:**
- `max_collision_depth` default: 5 → 2
- Removed spin calculation in `predict_cue_shot()`
- Simplified cue shot to center-strike only

### 5. Integration Service (`backend/integration_service.py`)
**Changed:**
- `max_collision_depth` parameter: 5 → 2
- Removed `confidence` field from trajectory line broadcast

### 6. Module Exports
Updated `__init__.py` files to export only existing classes:
- `TrajectoryPredictor` (new)
- `ShotPredictor` (backward compatibility alias)
- `ShotAnalyzer`
- `ShotAnalysis`
- `IllegalShotReason`

### 7. Dependencies Updated
- `backend/core/__init__.py`: `OutcomePredictor` → `TrajectoryPredictor`
- `backend/core/analysis/assistance.py`: `OutcomePredictor` → `TrajectoryPredictor`

## Simplified System Behavior

### What It Does
1. Detects cue stick angle
2. Calculates fixed medium-power shot velocity (2.0 m/s)
3. Predicts trajectory paths for:
   - Cue ball
   - Balls the cue ball hits (level 1 collisions)
   - Balls those balls hit (level 2 collisions)
4. Handles cushion bounces
5. Broadcasts trajectory points for visualization

### What It Doesn't Do
- Variable power predictions
- Spin/English effects
- Success probability calculations
- Shot difficulty scoring
- Alternative shot suggestions
- Risk factor analysis
- Problem/scratch detection
- 3+ level collision chains

## Testing Status
- ✅ Core analysis imports working
- ⚠️ Tests need updating to match new simplified API

## Next Steps (Future Enhancements)
If needed later, can add back:
- Variable power (Phase 2)
- Spin effects (Phase 2)
- Success probability (Phase 2)
- Difficulty scoring (Phase 2)
- Alternative shots (Phase 2)
- 3+ collision levels (Phase 2)

## Configuration
New configuration values:
- `core.physics.medium_shot_speed`: 2.0 m/s (fixed)
- `core.prediction.max_collision_depth`: 2 (hardcoded)

## Message Format
Trajectory broadcast contains:
```json
{
  "type": "trajectory",
  "lines": [
    {
      "start": [x, y],
      "end": [x, y],
      "type": "predicted|collision|cushion",
      "ball_id": "cue|ball_1|..."
    }
  ],
  "collisions": [
    {
      "time": 0.5,
      "position": [x, y],
      "type": "ball|cushion|pocket",
      "ball1_id": "cue",
      "ball2_id": "ball_1"
    }
  ]
}
```

## File Sizes
| File | Before | After | Reduction |
|------|--------|-------|-----------|
| prediction.py | 680 lines | 122 lines | 82% |
| shot.py | 903 lines | 190 lines | 79% |
| SPECS.md | ~640 lines | ~560 lines | 13% |

**Total reduction:** ~1,500 lines of code removed
