# Coordinate Space Metadata Update - Detector Adapter

## Summary

Updated the YOLO detector adapter to include coordinate space metadata in Ball and CueStick objects. This ensures that all detections include information about their coordinate system and source resolution, enabling proper coordinate transformations downstream.

## Changes Made

### File: `/backend/vision/detection/detector_adapter.py`

#### 1. Module Docstring Update
Added coordinate space documentation to the module docstring explaining:
- All YOLO detections are converted to pixel coordinates in the camera frame
- Output objects include `coordinate_space="pixel"` metadata
- Output objects include `source_resolution=(width, height)` metadata
- This enables downstream coordinate transformations (e.g., to world/table coordinates)

#### 2. Updated `yolo_to_ball()` Function
**Lines 357-361:** Added coordinate space metadata when creating Ball objects:
```python
# Create Ball object with coordinate space metadata
ball = Ball(
    position=position,
    radius=radius,
    ball_type=ball_type,
    number=ball_number,
    confidence=float(confidence),
    velocity=(0.0, 0.0),
    acceleration=(0.0, 0.0),
    is_moving=False,
    coordinate_space="pixel",  # YOLO outputs are in camera pixel coordinates
    source_resolution=(
        image_shape[1],
        image_shape[0],
    ),  # (width, height) from (height, width)
)
```

**Key Details:**
- `coordinate_space` is set to `"pixel"` to indicate camera pixel coordinates
- `source_resolution` is extracted from `image_shape` parameter
- Resolution is converted from (height, width) to (width, height) format
- Updated function docstring to document the metadata fields

#### 3. Updated `yolo_to_cue()` Function
**Lines 550-565:** Added coordinate space metadata when creating CueStick objects:
```python
# Create CueStick object with coordinate space metadata
cue = CueStick(
    tip_position=tip_position,
    angle=angle,
    length=length,
    confidence=float(confidence),
    state=CueState.AIMING,
    is_aiming=True,
    tip_velocity=(0.0, 0.0),
    angular_velocity=0.0,
    coordinate_space="pixel",  # YOLO outputs are in camera pixel coordinates
    source_resolution=(
        image_shape[1],
        image_shape[0],
    ),  # (width, height) from (height, width)
)
```

**Key Details:**
- Same metadata pattern as Ball objects
- Updated function docstring to document the metadata fields

#### 4. Updated `yolo_cue_to_cue_stick()` Function
**Lines 623-638:** Added coordinate space metadata for Detection objects:
```python
# Create CueStick object with coordinate space metadata
cue = CueStick(
    tip_position=tip_position,
    angle=angle,
    length=length,
    confidence=float(detection.confidence),
    state=CueState.AIMING,
    is_aiming=True,
    tip_velocity=(0.0, 0.0),
    angular_velocity=0.0,
    coordinate_space="pixel",  # YOLO outputs are in camera pixel coordinates
    source_resolution=(
        image_shape[1],
        image_shape[0],
    ),  # (width, height) from (height, width)
)
```

**Key Details:**
- Handles Detection objects from YOLODetector class
- Same metadata pattern for consistency
- Updated function docstring

## Coordinate Space Values

### Current Implementation
- `coordinate_space="pixel"` - All YOLO detections are in camera pixel coordinates
- `source_resolution=(width, height)` - Resolution of the source image

### Future Extensibility
The existing Ball and CueStick dataclasses (in `/backend/vision/models.py`) already support:
- `coordinate_space: str` field with values like `"pixel"` or `"world"`
- `source_resolution: Optional[tuple[int, int]]` field
- `to_world_coordinates()` method for transforming to world coordinates

## Benefits

1. **Explicit Coordinate System Tracking**: Every detection now explicitly states it's in pixel coordinates
2. **Resolution Preservation**: Source resolution is preserved for coordinate transformations
3. **Downstream Transformations**: Enables proper conversion to table/world coordinates
4. **Debugging**: Makes it easier to track coordinate system issues
5. **Consistency**: All conversion functions now follow the same pattern

## Testing

The changes have been verified for:
- Syntax correctness (Python compilation check passed)
- Consistency across all three conversion functions
- Proper resolution format conversion (height,width) → (width,height)

## Impact

- **Backward Compatible**: The Ball and CueStick dataclasses already had these fields with defaults
- **No Breaking Changes**: Existing code will continue to work
- **Enhanced Functionality**: Downstream code can now reliably transform coordinates
- **Documentation**: Clear inline comments explain the metadata

## Related Files

- `/backend/vision/models.py` - Defines Ball and CueStick dataclasses with coordinate metadata
- `/backend/vision/detection/yolo_detector.py` - Uses detector_adapter for conversion
- Coordinate transformation utilities can now leverage this metadata

## Next Steps

Consider:
1. Update other detection adapters to include similar metadata
2. Implement coordinate transformation utilities that leverage this metadata
3. Add coordinate space validation in downstream processing
4. Document coordinate space conventions across the codebase
