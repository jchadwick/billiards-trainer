# Group 9: API Models and Responses Migration Plan

**Status**: WAITING FOR GROUP 3 (Core Models)
**Date**: 2025-10-21
**Agent**: Agent 9

---

## Current Status

### Dependencies
- **BLOCKED**: Waiting for Group 3 (Core Models) to complete
- Group 3 must migrate Vector2D from `coordinate_space + resolution` to mandatory `scale` tuple
- Group 3 must migrate BallState, TableState, CueState to use 4K pixels

### Current State Analysis

#### Vector2D Current Format
```python
@dataclass
class Vector2D:
    x: float
    y: float
    coordinate_space: Optional[CoordinateSpace] = None  # To be REMOVED
    resolution: Optional[Resolution] = None              # To be REMOVED
```

#### Vector2D Target Format (after Group 3)
```python
@dataclass
class Vector2D:
    x: float                    # X coordinate in pixels
    y: float                    # Y coordinate in pixels
    scale: tuple[float, float]  # MANDATORY: [scale_x, scale_y]
```

---

## Files to Update

### 1. `/Users/jchadwick/code/billiards-trainer/backend/api/models/converters.py`

**Current Issues**:
- Uses `vector2d_to_list()` which returns `[x, y]` only
- No scale metadata in API responses
- Still references `coordinate_space`

**Required Changes**:

```python
# OLD (current)
def vector2d_to_list(vector: Vector2D) -> list[float]:
    """Convert core Vector2D to list format [x, y]."""
    return [vector.x, vector.y]

# NEW (after migration)
def vector2d_to_dict(vector: Vector2D) -> dict[str, Any]:
    """Convert core Vector2D to dict with mandatory scale.

    Returns:
        Dictionary with x, y, and scale fields:
        {
            "x": float,
            "y": float,
            "scale": [scale_x, scale_y]
        }
    """
    return {
        "x": vector.x,
        "y": vector.y,
        "scale": list(vector.scale)  # Convert tuple to list for JSON
    }
```

**Functions to Update**:
- `vector2d_to_list()` → Replace with `vector2d_to_dict()`
- `ball_state_to_ball_info()` → Use new dict format
- `ball_state_to_websocket_data()` → Use new dict format
- `cue_state_to_cue_info()` → Use new dict format
- `table_state_to_table_info()` → Use new dict format
- `trajectory_to_websocket_data()` → Use new dict format

**New Validation**:
```python
def validate_vector2d_has_scale(vector: Vector2D) -> None:
    """Validate that Vector2D has mandatory scale metadata."""
    if vector.scale is None:
        raise ValueError("Vector2D must have scale metadata")
    if len(vector.scale) != 2:
        raise ValueError(f"Scale must be a 2-element tuple, got {len(vector.scale)}")
    if vector.scale[0] <= 0 or vector.scale[1] <= 0:
        raise ValueError(f"Scale factors must be positive, got {vector.scale}")
```

### 2. `/Users/jchadwick/code/billiards-trainer/backend/api/websocket/broadcaster.py`

**Current Issues**:
- `_enrich_coordinate_metadata()` adds `coordinate_space` field (line 768)
- No scale metadata in broadcasted messages
- Uses legacy coordinate metadata format

**Required Changes**:

#### Remove coordinate_space from metadata
```python
# OLD (line 768)
if "coordinate_space" not in enriched:
    enriched["coordinate_space"] = "world_meters"

# NEW
# Remove this entirely - coordinate_space is deprecated
```

#### Update broadcast_game_state signature
```python
# OLD
async def broadcast_game_state(
    self,
    balls: list[dict[str, Any]],
    cue: Optional[dict[str, Any]] = None,
    table: Optional[dict[str, Any]] = None,
    timestamp: Optional[datetime] = None,
    coordinate_metadata: Optional[dict[str, Any]] = None,  # Remove this
):

# NEW
async def broadcast_game_state(
    self,
    balls: list[dict[str, Any]],
    cue: Optional[dict[str, Any]] = None,
    table: Optional[dict[str, Any]] = None,
    timestamp: Optional[datetime] = None,
    # coordinate_metadata removed - scale is now in each Vector2D
):
```

#### Update validation to check for scale
```python
# Add to ball validation (after line 328)
# Validate position has scale metadata
position = ball["position"]
if isinstance(position, dict):
    if "scale" not in position:
        logger.warning(
            f"broadcast_game_state: ball at index {i} position missing 'scale' metadata"
        )
        self.broadcast_stats.validation_failures += 1
        return

    if not isinstance(position["scale"], (list, tuple)) or len(position["scale"]) != 2:
        logger.warning(
            f"broadcast_game_state: ball at index {i} has invalid scale format: {position['scale']}"
        )
        self.broadcast_stats.validation_failures += 1
        return
```

#### Remove coordinate metadata enrichment
```python
# OLD (lines 356-369)
enriched_metadata = self._enrich_coordinate_metadata(coordinate_metadata, table)
state_data = {
    "balls": balls,
    "cue": cue,
    "table": table,
    "timestamp": (timestamp or datetime.now(timezone.utc)).isoformat(),
    "sequence": self._get_next_sequence("state"),
    "ball_count": len(balls),
}
if enriched_metadata:
    state_data["coordinate_metadata"] = enriched_metadata

# NEW
state_data = {
    "balls": balls,  # Now includes scale in each position/velocity
    "cue": cue,
    "table": table,
    "timestamp": (timestamp or datetime.now(timezone.utc)).isoformat(),
    "sequence": self._get_next_sequence("state"),
    "ball_count": len(balls),
    # No coordinate_metadata - scale is embedded in each vector
}
```

#### Remove `_enrich_coordinate_metadata` method
```python
# DELETE lines 730-772
# This method is no longer needed - scale is in each Vector2D
```

### 3. `/Users/jchadwick/code/billiards-trainer/backend/api/websocket/manager.py`

**Current Issues**:
- No specific issues, but may need updates for filtering logic

**Required Changes**:
- Review filter logic to ensure it handles new dict format with scale
- No major changes expected since this file mainly manages connections

### 4. `/Users/jchadwick/code/billiards-trainer/backend/api/routes/debug.py`

**Current Issues**:
- JavaScript expects `coordinate_metadata` (lines 361-436)
- Needs to handle new scale-based format

**Required Changes**:

#### Update JavaScript to handle scale metadata
```javascript
// OLD (lines 413-436)
const coordMetadata = gameState.coordinate_metadata;
let coordinateSpace;
if (coordMetadata && coordMetadata.width && coordMetadata.height) {
    coordinateSpace = {
        width: coordMetadata.width,
        height: coordMetadata.height
    };
} else if (gameState.video_resolution) {
    coordinateSpace = gameState.video_resolution;
} else {
    coordinateSpace = { width: 1920, height: 1080 };
}

// NEW
// Get resolution from first ball's scale metadata
// All positions in 4K canonical, scale tells us source resolution
const firstBall = gameState.balls && gameState.balls[0];
let sourceResolution;
if (firstBall && firstBall.position && firstBall.position.scale) {
    // Scale factors tell us conversion: source_res = 4k / scale
    // For 1080p: scale = [2.0, 2.0], so source = [3840/2.0, 2160/2.0] = [1920, 1080]
    const scaleX = firstBall.position.scale[0];
    const scaleY = firstBall.position.scale[1];
    sourceResolution = {
        width: 3840 / scaleX,   // 4K canonical width / scale
        height: 2160 / scaleY   // 4K canonical height / scale
    };
    console.log(`Detected source resolution from scale: ${sourceResolution.width}x${sourceResolution.height}`);
} else {
    // Fallback: assume 4K canonical (scale = [1.0, 1.0])
    sourceResolution = { width: 3840, height: 2160 };
    console.log(`No scale metadata, assuming 4K canonical`);
}

// Calculate scale from source resolution to canvas
const scaleX = canvas.width / sourceResolution.width;
const scaleY = canvas.height / sourceResolution.height;
```

#### Update ball position access
```javascript
// OLD (line 463)
const pos = ball.position || ball.pos;
if (!pos || pos.x === undefined || pos.y === undefined) return;

const x = pos.x * scaleX;
const y = pos.y * scaleY;

// NEW
const pos = ball.position || ball.pos;
if (!pos || pos.x === undefined || pos.y === undefined) return;

// Position is in source resolution (determined by scale)
// Convert to 4K canonical first, then to canvas
const pos4k = {
    x: pos.x * (pos.scale ? pos.scale[0] : 1.0),
    y: pos.y * (pos.scale ? pos.scale[1] : 1.0)
};

// Now convert from 4K canonical to canvas
const x = pos4k.x * (canvas.width / 3840);
const y = pos4k.y * (canvas.height / 2160);
```

---

## API Response Format Changes

### Old Format (v1.0)
```json
{
  "balls": [
    {
      "id": "ball_1",
      "position": [1920.0, 1080.0],  // Just [x, y]
      "velocity": [0.0, 0.0],
      "coordinate_space": "world_meters"
    }
  ],
  "coordinate_metadata": {
    "coordinate_space": "world_meters",
    "camera_resolution": [1920, 1080],
    "pixels_per_meter": 754.0
  }
}
```

### New Format (v2.0)
```json
{
  "balls": [
    {
      "id": "ball_1",
      "position": {
        "x": 1920.0,
        "y": 1080.0,
        "scale": [1.0, 1.0]  // MANDATORY - indicates 4K canonical
      },
      "velocity": {
        "x": 0.0,
        "y": 0.0,
        "scale": [1.0, 1.0]
      }
    }
  ]
  // No coordinate_metadata - scale is embedded in each vector
}
```

### Example: Different Resolutions

#### 4K Canonical (scale = [1.0, 1.0])
```json
{
  "position": {
    "x": 1920.0,
    "y": 1080.0,
    "scale": [1.0, 1.0]
  }
}
```

#### 1080p Source (scale = [2.0, 2.0])
```json
{
  "position": {
    "x": 960.0,      // Position in 1080p coordinates
    "y": 540.0,
    "scale": [2.0, 2.0]  // Multiply by 2 to get 4K: 960*2=1920, 540*2=1080
  }
}
```

#### 720p Source (scale = [3.0, 3.0])
```json
{
  "position": {
    "x": 640.0,      // Position in 720p coordinates
    "y": 360.0,
    "scale": [3.0, 3.0]  // Multiply by 3 to get 4K: 640*3=1920, 360*3=1080
  }
}
```

---

## Migration Checklist

### Phase 1: Preparation (Before Group 3 completes)
- [x] Analyze current API format
- [x] Identify all conversion functions
- [x] Document required changes
- [x] Create migration plan
- [ ] Write unit tests for new format

### Phase 2: Implementation (After Group 3 completes)
- [ ] Update `converters.py`:
  - [ ] Replace `vector2d_to_list()` with `vector2d_to_dict()`
  - [ ] Update all ball/cue/table converters
  - [ ] Add scale validation
  - [ ] Update batch conversion functions
- [ ] Update `broadcaster.py`:
  - [ ] Remove `coordinate_metadata` parameter
  - [ ] Remove `_enrich_coordinate_metadata()` method
  - [ ] Add scale validation to `broadcast_game_state()`
  - [ ] Update state_data structure
- [ ] Update `debug.py`:
  - [ ] Update JavaScript to read scale from positions
  - [ ] Fix coordinate conversion logic
  - [ ] Update display logic
- [ ] Update tests:
  - [ ] Update API response tests
  - [ ] Update WebSocket broadcast tests
  - [ ] Add scale validation tests

### Phase 3: Verification
- [ ] All unit tests passing
- [ ] Integration tests passing
- [ ] WebSocket broadcasts working
- [ ] Debug page displays correctly
- [ ] No coordinate_space references remain

---

## Breaking Changes Summary

### REMOVED
- `coordinate_metadata` field from API responses
- `coordinate_space` field from Vector2D
- `_enrich_coordinate_metadata()` method
- Array format `[x, y]` for positions/velocities

### ADDED
- Mandatory `scale` field in all Vector2D instances
- Dictionary format `{x, y, scale}` for positions/velocities
- Scale validation in broadcast functions

### CHANGED
- API response format from v1.0 to v2.0
- WebSocket message format
- Debug page coordinate conversion logic

---

## Testing Strategy

### Unit Tests
```python
def test_vector2d_to_dict_includes_scale():
    """Test that vector2d_to_dict includes scale metadata."""
    vec = Vector2D(x=1920, y=1080, scale=(1.0, 1.0))
    result = vector2d_to_dict(vec)

    assert "x" in result
    assert "y" in result
    assert "scale" in result
    assert result["scale"] == [1.0, 1.0]

def test_ball_state_conversion_preserves_scale():
    """Test that BallState conversion preserves scale."""
    ball = BallState.from_4k(id="ball_1", x=1920, y=1080)
    ball_info = ball_state_to_ball_info(ball)

    assert "position" in ball_info.__dict__
    pos = ball_info.position
    assert isinstance(pos, dict)
    assert pos["scale"] == [1.0, 1.0]

def test_broadcast_validates_scale():
    """Test that broadcast validates scale metadata."""
    # Missing scale should fail validation
    balls = [{
        "id": "ball_1",
        "position": {"x": 100, "y": 100}  # No scale
    }]

    with pytest.raises(ValueError):
        await broadcaster.broadcast_game_state(balls)
```

### Integration Tests
```python
async def test_websocket_broadcast_includes_scale():
    """Test that WebSocket broadcasts include scale in positions."""
    # Create game state
    ball = BallState.from_4k(id="cue", x=1920, y=1080)

    # Broadcast
    await broadcaster.broadcast_game_state(
        balls=[ball_state_to_websocket_data(ball)]
    )

    # Check received message
    message = await websocket_client.receive_json()
    assert message["type"] == "state"
    assert "balls" in message["data"]
    ball_data = message["data"]["balls"][0]
    assert "position" in ball_data
    assert "scale" in ball_data["position"]
    assert ball_data["position"]["scale"] == [1.0, 1.0]
```

---

## Next Steps

1. **WAIT** for Group 3 (Core Models) to complete:
   - Vector2D must have mandatory `scale` tuple
   - BallState, TableState, CueState must use 4K pixels
   - All factory methods must set correct scale

2. **Verify** Group 3 completion:
   - Check Vector2D has `scale` field
   - Check `coordinate_space` and `resolution` removed
   - Check BallState uses 4K canonical coordinates

3. **Execute** migration:
   - Follow checklist above
   - Run tests after each change
   - Verify no breaking changes to external APIs

4. **Document** breaking changes:
   - Update API documentation
   - Create migration guide for API consumers
   - Add examples of new format

---

## Estimated Effort

- **Preparation**: 2 hours (DONE)
- **Implementation**: 4-6 hours
- **Testing**: 2-3 hours
- **Documentation**: 1-2 hours
- **Total**: 9-13 hours (approximately 1-2 days)

---

## Risk Assessment

### Low Risk
- Converting response format (well-defined change)
- Adding scale validation (straightforward)
- Removing coordinate_metadata (clean removal)

### Medium Risk
- Debug page JavaScript changes (requires careful testing)
- WebSocket clients may need updates (breaking change)

### Mitigation
- Comprehensive testing at each step
- Maintain backward compatibility layer if needed
- Clear documentation of breaking changes
- Phased rollout with monitoring

---

**Status**: Ready to execute once Group 3 completes
**Last Updated**: 2025-10-21
**Owner**: Agent 9
