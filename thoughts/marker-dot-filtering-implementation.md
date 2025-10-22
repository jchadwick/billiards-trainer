# Marker Dot Filtering Implementation Plan

## Summary

Implement marker dot filtering in ball detection pipeline to prevent table markers (spots/stickers) from being detected as balls while ensuring no false negatives on actual balls (especially the 8-ball).

## Problem Statement

Table marker dots (e.g., positioning stickers, brand logos) can be detected as balls by YOLO+OpenCV hybrid detector. We need to filter these out while being **super ultra critical** to avoid missing actual balls.

## Marker Dot Characteristics

Based on typical billiard table markers:
- **Size**: Smaller radius than standard billiard balls (typically 5-15px vs 15-26px for balls)
- **Color**: Mostly black or very dark (low brightness/value in HSV)
- **Shape**: Circular but often with patterns/text inside
- **Texture**: Less uniform than billiard balls (may have printing, edges)

## Safety Requirements

1. **NO FALSE NEGATIVES**: Must never filter out actual balls
2. **Conservative filtering**: Only reject detections that clearly match ALL marker criteria
3. **Special care for 8-ball**: Black 8-ball must pass through filters even in marker locations
4. **Configurable**: All thresholds must be configurable for different table setups

## Implementation Strategy

### Phase 1: Configuration

Add marker dot configuration to `config.json` under `vision.detection.marker_filtering`:

```json
{
  "vision": {
    "detection": {
      "marker_filtering": {
        "enabled": true,
        "proximity_threshold_px": 30,  // Distance to consider "near" a marker
        "size_ratio_threshold": 0.6,  // Max size relative to expected ball
        "brightness_threshold": 70,  // Max HSV V-channel value (0-255)
        "circularity_min": 0.85,  // Min circularity for markers (balls are rounder)
        "texture_variance_max": 500  // Max texture variance for markers
      }
    }
  }
}
```

### Phase 2: Marker Dot Loading

Modify `YOLODetector` to:
1. Load marker dots from config on initialization
2. Convert marker dots from screen coordinates to current frame coordinates if resolution differs
3. Cache marker dot positions for efficient proximity checks

### Phase 3: Detection Filtering

Add method `_is_marker_dot()` to `YOLODetector` that checks if a detection is a marker:

```python
def _is_marker_dot(
    self,
    detection: Detection,
    frame: NDArray[np.uint8],
    expected_ball_radius: float = 20.0
) -> tuple[bool, str]:
    """
    Determine if detection is a marker dot.

    Returns:
        (is_marker, reason) - True if marker, with explanation
    """
```

Filtering logic (ALL criteria must match to reject):
1. **Proximity**: Must be within `proximity_threshold_px` of a marker dot location
2. **Size**: Must be smaller than `size_ratio_threshold * expected_ball_radius`
3. **Brightness**: Must have low average brightness (dark/black)
4. **Texture**: Must have low texture variance (flat surface vs ball sheen)

### Phase 4: Integration

Modify `detect_balls()` in `YOLODetector` (line 847-914):
1. After initial YOLO detections and size filtering
2. Before returning results, apply marker dot filtering
3. Log all filtered detections for debugging

### Phase 5: Safety Checks

Implement safety mechanisms:
1. **8-ball exception**: Always allow black balls with proper size
2. **Confidence override**: High-confidence detections bypass marker filtering
3. **Motion detection**: Moving detections bypass marker filtering (markers don't move)
4. **Logging**: Log every filtered detection with detailed metrics

## Code Changes

### File: `backend/vision/detection/yolo_detector.py`

1. Add marker dot storage and loading in `__init__()`:
```python
# Load marker dots from config for filtering
self.marker_dots: list[tuple[float, float]] = []
self._load_marker_dots()
```

2. Add marker dot loading method:
```python
def _load_marker_dots(self) -> None:
    """Load marker dots from configuration."""
    try:
        from ...config import config
        marker_dots_config = config.get("table.marker_dots", [])
        self.marker_dots = [
            (dot["x"], dot["y"]) for dot in marker_dots_config
        ]
        logger.info(f"Loaded {len(self.marker_dots)} marker dot positions for filtering")
    except Exception as e:
        logger.warning(f"Failed to load marker dots: {e}")
        self.marker_dots = []
```

3. Add marker detection method:
```python
def _is_marker_dot(
    self,
    detection: Detection,
    frame: NDArray[np.uint8],
    marker_config: dict,
) -> tuple[bool, str]:
    """Check if detection matches marker dot characteristics.

    Args:
        detection: YOLO detection to check
        frame: Source frame for analysis
        marker_config: Marker filtering configuration

    Returns:
        (is_marker, reason) tuple
    """
    # Only check if marker filtering is enabled
    if not marker_config.get("enabled", True):
        return False, "filtering_disabled"

    # Check proximity to known marker locations
    near_marker = False
    for marker_x, marker_y in self.marker_dots:
        dist = np.sqrt(
            (detection.center[0] - marker_x)**2 +
            (detection.center[1] - marker_y)**2
        )
        if dist < marker_config.get("proximity_threshold_px", 30):
            near_marker = True
            break

    if not near_marker:
        return False, "not_near_marker"

    # Must be near a marker to continue checking
    # Now apply strict criteria

    # Extract detection region
    x1, y1, x2, y2 = [int(v) for v in detection.bbox]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
    region = frame[y1:y2, x1:x2]

    if region.size == 0:
        return False, "empty_region"

    # Criteria 1: Size check
    expected_radius = marker_config.get("expected_ball_radius_px", 20.0)
    size_ratio = max(detection.width, detection.height) / (2 * expected_radius)
    if size_ratio >= marker_config.get("size_ratio_threshold", 0.6):
        return False, f"size_ok_ratio={size_ratio:.2f}"

    # Criteria 2: Brightness check (markers are dark)
    hsv_region = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    avg_brightness = np.mean(hsv_region[:, :, 2])
    if avg_brightness >= marker_config.get("brightness_threshold", 70):
        return False, f"brightness_ok={avg_brightness:.1f}"

    # Criteria 3: Texture variance (markers are flat, balls have highlights)
    gray_region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    texture_var = np.var(gray_region)
    if texture_var >= marker_config.get("texture_variance_max", 500):
        return False, f"texture_ok_var={texture_var:.1f}"

    # All marker criteria matched - likely a marker
    return True, f"marker_detected_size={size_ratio:.2f}_brightness={avg_brightness:.1f}_var={texture_var:.1f}"
```

4. Modify `detect_balls()` to apply filtering:
```python
# After size filtering (around line 910), add marker filtering:
if self.marker_dots:
    from ...config import config
    marker_config = config.get("vision.detection.marker_filtering", {})

    if marker_config.get("enabled", True):
        # Apply marker dot filtering
        final_detections = []
        for det in filtered_detections:
            is_marker, reason = self._is_marker_dot(det, frame, marker_config)

            if is_marker:
                logger.debug(
                    f"Filtered marker dot: {det.class_name} at "
                    f"({det.center[0]:.1f}, {det.center[1]:.1f}), "
                    f"reason: {reason}"
                )
            else:
                final_detections.append(det)

        logger.debug(
            f"Marker filtering: {len(filtered_detections)} detections -> "
            f"{len(final_detections)} after filtering "
            f"({len(filtered_detections) - len(final_detections)} markers removed)"
        )
        filtered_detections = final_detections
```

### File: `config.json`

Add marker filtering configuration:
```json
{
  "vision": {
    "detection": {
      "marker_filtering": {
        "enabled": true,
        "proximity_threshold_px": 30,
        "size_ratio_threshold": 0.6,
        "expected_ball_radius_px": 20,
        "brightness_threshold": 70,
        "texture_variance_max": 500,
        "confidence_override_threshold": 0.7
      }
    }
  }
}
```

## Testing Plan

1. **Test with marker dots only**: Verify markers are filtered
2. **Test with balls only**: Verify no false negatives
3. **Test with 8-ball near markers**: Verify 8-ball passes
4. **Test with moving balls**: Verify motion bypass works
5. **Test edge cases**:
   - Balls partially overlapping marker locations
   - Very small balls (far from camera)
   - Dirty/worn balls with low brightness

## Rollback Plan

If filtering causes false negatives:
1. Set `enabled: false` in config
2. Review logs to identify problematic threshold
3. Adjust thresholds conservatively
4. Re-enable with updated values

## Success Criteria

- [ ] No false negatives on actual balls
- [ ] ≥90% marker dot filtering accuracy
- [ ] 8-ball always detected near markers
- [ ] Configuration allows per-table tuning
- [ ] Comprehensive logging for debugging
