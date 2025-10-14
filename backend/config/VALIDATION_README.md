# Configuration Validation

This module provides comprehensive configuration validation for the billiards trainer backend at startup time.

## Overview

The configuration validation system ensures that:
1. All required configuration parameters are present
2. Parameter values are within valid ranges
3. Parameters have correct types
4. Cross-field validations pass (e.g., min < max)
5. Default values from proven configurations are applied for missing parameters

## Features

### Fail-Fast Behavior

Critical configuration errors will cause the backend to exit immediately during startup with clear error messages. This prevents the system from starting in an invalid state.

### Proven Defaults

The validator includes proven default values based on the working `video_debugger.py` configuration and current production settings. These defaults are automatically applied for any missing configuration parameters.

### Range Validation

All numeric parameters are validated against sensible min/max ranges:

- **FPS**: 1-120
- **Confidence thresholds**: 0.0-1.0
- **Ball radius**: 10-200 pixels
- **Tracking parameters**: Sensible ranges for Kalman filter parameters

### Suboptimal Value Warnings

The system warns about suboptimal but technically valid configurations:

- `min_hits < 5`: May cause unstable tracking (recommended: 10)
- `yolo_confidence > 0.5`: May miss detections (recommended: 0.15)
- `max_distance > 200.0`: May cause incorrect associations (recommended: 100.0)

## Usage

### Automatic Validation

The validation runs automatically when the backend starts:

```python
# In backend/main.py
from backend.config.validation import validate_configuration

config = ConfigurationModule()
validate_configuration(config)  # Raises ConfigValidationError on failure
```

### Manual Validation

You can also validate configuration manually:

```python
from backend.config.validation import ConfigValidator

validator = ConfigValidator(config_manager)
if validator.validate_all():
    print("Configuration is valid")
else:
    print("Errors:", validator.errors)
    print("Warnings:", validator.warnings)
```

### Testing

Run the validation test suite:

```bash
python backend/config/test_validation.py
```

This runs comprehensive tests including:
- Valid configuration
- Missing defaults
- Invalid range detection
- Suboptimal value warnings
- Cross-field validation
- Kernel size validation (must be odd)
- Fail-fast behavior

## Configuration Parameters

### Vision Module

#### Camera Configuration

| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `device_id` | int/str | - | 0 | Camera device ID or video file path |
| `resolution` | [int, int] | [width, height] | [1920, 1080] | Camera resolution |
| `fps` | int | 1-120 | 30 | Frame rate |
| `buffer_size` | int | 1-10 | 1 | Camera buffer size |
| `gain` | float | 0.0-10.0 | 1.0 | Camera gain |
| `reconnect_delay` | float | 0.1-60.0 | 1.0 | Reconnect delay (seconds) |
| `max_reconnect_attempts` | int | 0-100 | 5 | Maximum reconnect attempts |

#### Detection Configuration

| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `yolo_confidence` | float | 0.0-1.0 | 0.15 | YOLO detection confidence |
| `yolo_nms_threshold` | float | 0.0-1.0 | 0.45 | YOLO NMS threshold |
| `table_edge_threshold` | float | 0.0-1.0 | 0.7 | Table edge threshold |
| `min_table_area` | float | 0.0-1.0 | 0.3 | Minimum table area ratio |
| `min_ball_radius` | int | 1-100 | 10 | Minimum ball radius (px) |
| `max_ball_radius` | int | 1-200 | 40 | Maximum ball radius (px) |
| `ball_sensitivity` | float | 0.0-1.0 | 0.8 | Ball detection sensitivity |
| `min_cue_length` | int | 10-1000 | 100 | Minimum cue length (px) |
| `cue_line_threshold` | float | 0.0-1.0 | 0.6 | Cue line threshold |

#### Tracking Configuration

| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `max_age` | int | 1-1000 | 30 | Maximum track age (frames) |
| `min_hits` | int | 1-100 | 10 | Minimum hits to establish track |
| `max_distance` | float | 1.0-1000.0 | 100.0 | Maximum association distance (px) |
| `process_noise` | float | 0.1-100.0 | 5.0 | Kalman filter process noise |
| `measurement_noise` | float | 0.1-100.0 | 20.0 | Kalman filter measurement noise |
| `collision_threshold` | float | 1.0-200.0 | 60.0 | Collision detection threshold (px) |
| `min_hits_during_collision` | int | 1-100 | 30 | Min hits during collision |
| `motion_speed_threshold` | float | 0.0-1000.0 | 10.0 | Motion speed threshold (px/frame) |

#### Processing Configuration

| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `blur_kernel_size` | int | 1-31 (odd) | 5 | Blur kernel size (must be odd) |
| `morphology_kernel_size` | int | 1-31 (odd) | 3 | Morphology kernel size (must be odd) |
| `tracking_max_distance` | float | 1.0-1000.0 | 50.0 | Max tracking distance (px) |
| `frame_skip` | int | 0-10 | 0 | Number of frames to skip |

## Error Messages

### Critical Errors (Fail Fast)

These errors will cause the backend to exit immediately:

- **Missing vision configuration**: `CRITICAL: Vision configuration is missing`
- **Invalid ranges**: `Value X is below minimum Y` or `Value X is above maximum Y`
- **Invalid types**: `Expected type X, got Y`
- **Cross-field conflicts**: `min_ball_radius must be less than max_ball_radius`
- **Invalid kernel sizes**: `Blur kernel size must be odd`

### Warnings (Non-Critical)

These warnings are logged but don't prevent startup:

- **Suboptimal values**: Parameter suggestions for better performance
- **Missing files**: `YOLO model file not found` (will fall back to OpenCV)

## Example Output

### Successful Validation

```
INFO:backend.config.validation:Validating backend configuration...
INFO:backend.config.validation:Applying default configuration values...
INFO:backend.config.validation:Applied 14 default configuration values
INFO:backend.config.validation:✅ Configuration validation passed
INFO:backend.config.validation:Configuration validation complete
```

### With Warnings

```
INFO:backend.config.validation:Validating backend configuration...
INFO:backend.config.validation:Applied 14 default configuration values
WARNING:backend.config.validation:Configuration has 2 warnings (non-critical):
WARNING:backend.config.validation:  ⚠️  vision.tracking.min_hits = 2: min_hits < 5 may cause unstable tracking. Recommended: 10
WARNING:backend.config.validation:  ⚠️  YOLO model file not found: models/yolov8n-pool.onnx
INFO:backend.config.validation:Configuration validation complete
```

### With Errors (Startup Fails)

```
ERROR:backend.config.validation:Configuration validation failed with 3 errors:
ERROR:backend.config.validation:  ❌ Minimum hits to establish track (vision.tracking.min_hits): Value -5 is below minimum 1
ERROR:backend.config.validation:  ❌ YOLO detection confidence threshold (vision.detection.yolo_confidence): Value 1.5 is above maximum 1.0
ERROR:backend.config.validation:  ❌ Camera frame rate (vision.camera.fps): Value 200 is above maximum 120
CRITICAL:Cannot start backend with invalid configuration
```

## Adding New Validation Rules

To add validation for new parameters:

1. Add default value to `ConfigValidator.PROVEN_DEFAULTS`
2. Add validation rule to `ConfigValidator.VALIDATION_RULES`
3. Optionally add warning to `ConfigValidator.SUBOPTIMAL_WARNINGS`

Example:

```python
# In backend/config/validation.py

# 1. Add default
PROVEN_DEFAULTS = {
    "vision": {
        "new_parameter": 42
    }
}

# 2. Add validation rule
VALIDATION_RULES = {
    "vision.new_parameter": {
        "type": int,
        "min": 1,
        "max": 100,
        "description": "New parameter description",
    }
}

# 3. Add warning (optional)
SUBOPTIMAL_WARNINGS = {
    "vision.new_parameter": {
        "recommended": 42,
        "warning_if": lambda x: x < 10,
        "message": "new_parameter < 10 may cause issues. Recommended: 42"
    }
}
```

## Benefits

1. **Early Error Detection**: Catch configuration issues before they cause runtime errors
2. **Clear Error Messages**: Know exactly what's wrong and how to fix it
3. **Automatic Defaults**: Missing parameters are filled in with proven values
4. **Proactive Warnings**: Get notified about suboptimal but valid configurations
5. **Type Safety**: Ensure all parameters have correct types
6. **Range Safety**: Prevent out-of-range values that could cause crashes
7. **Consistency**: Ensure configuration is consistent across fields
