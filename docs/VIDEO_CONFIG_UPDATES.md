# Video File Configuration Helper Methods

## Summary

Added helper methods to `VisionConfigurationManager` for easier video file configuration and enhanced the Pydantic schema to support video source types.

## Changes Made

### 1. VisionConfigurationManager (`backend/vision/config_manager.py`)

Added four new helper methods:

#### `get_video_source_type() -> str`
- Returns the configured video source type: "camera", "file", or "stream"
- Defaults to "camera" if not specified
- Makes it easy to determine the current input source

#### `is_video_file_input() -> bool`
- Returns `True` if video source is configured as a file
- Supports both new `video_source_type` field and legacy string `device_id` for backward compatibility
- Useful for quick checks without parsing configuration

#### `get_video_file_path() -> Optional[str]`
- Returns the absolute path to the video file if configured for file input
- Returns `None` if not using file input
- Validates file exists and is readable
- Raises `FileNotFoundError` if file doesn't exist
- Raises `ValueError` if path is not a file
- Raises `PermissionError` if file is not readable
- Handles both new `video_file_path` field and legacy string `device_id`

#### `resolve_camera_config() -> dict[str, Any]`
- Returns a complete CameraCapture-compatible configuration dictionary
- Maps `video_source_type` to appropriate `device_id`
- Includes all camera settings (resolution, fps, exposure, etc.)
- Includes video file settings (loop_video, video_start_frame, video_end_frame)
- Handles backward compatibility with legacy configurations

#### Updated `get_camera_config()`
- Now uses `resolve_camera_config()` internally
- Maintains backward compatibility

#### Enhanced `validate_config()`
Added validation for video configuration:
- Ensures `video_file_path` is required when `video_source_type="file"`
- Validates file exists and is readable
- Ensures `stream_url` is required when `video_source_type="stream"`
- Validates `video_start_frame < video_end_frame` when both specified
- Allows `device_id` to be either `int` or `str` for backward compatibility

### 2. Configuration Schema (`backend/config/models/schemas.py`)

#### Added `VideoSourceType` Enum
```python
class VideoSourceType(str, Enum):
    CAMERA = "camera"
    FILE = "file"
    STREAM = "stream"
```

#### Enhanced `CameraSettings` with Video Fields

**Video Source Configuration:**
- `video_source_type: VideoSourceType` - Type of video source (default: "camera")
- `video_file_path: Optional[str]` - Path to video file when using file source
- `stream_url: Optional[str]` - Stream URL when using stream source

**Video Playback Control:**
- `loop_video: bool` - Loop video playback for file sources (default: False)
- `video_start_frame: int` - Starting frame number for video files (default: 0)
- `video_end_frame: Optional[int]` - Ending frame number for video files (default: None)

#### Added `validate_video_config()` Model Validator
- Ensures `video_file_path` is provided when `video_source_type="file"`
- Ensures `stream_url` is provided when `video_source_type="stream"`
- Validates frame range (`video_start_frame < video_end_frame`)

## Configuration Examples

### New Style - Video File Input
```json
{
  "camera": {
    "video_source_type": "file",
    "video_file_path": "/path/to/video.mp4",
    "loop_video": true,
    "video_start_frame": 0,
    "video_end_frame": 1000,
    "fps": 30,
    "resolution": [1920, 1080]
  }
}
```

### New Style - Stream Input
```json
{
  "camera": {
    "video_source_type": "stream",
    "stream_url": "rtsp://example.com/stream",
    "fps": 30,
    "resolution": [1280, 720]
  }
}
```

### New Style - Camera Input
```json
{
  "camera": {
    "video_source_type": "camera",
    "device_id": 0,
    "fps": 60,
    "resolution": [1920, 1080]
  }
}
```

### Legacy Style - Still Supported
```json
{
  "camera": {
    "device_id": "/path/to/video.mp4",
    "fps": 30
  }
}
```

## Usage Examples

### Check Video Source Type
```python
manager = VisionConfigurationManager()
manager.initialize()

source_type = manager.get_video_source_type()
# Returns: "camera", "file", or "stream"
```

### Check if Using Video File
```python
if manager.is_video_file_input():
    print("Using video file input")
else:
    print("Using camera input")
```

### Get Video File Path
```python
try:
    file_path = manager.get_video_file_path()
    if file_path:
        print(f"Video file: {file_path}")
except FileNotFoundError as e:
    print(f"Video file not found: {e}")
```

### Get Complete Camera Configuration
```python
camera_config = manager.resolve_camera_config()
# Returns dict with all settings needed for CameraCapture
# Including: device_id, resolution, fps, loop_video,
#            video_start_frame, video_end_frame, etc.

# Use with CameraCapture
from backend.vision.capture import CameraCapture
camera = CameraCapture(camera_config)
```

### Validate Configuration
```python
config = {
    "camera": {
        "video_source_type": "file",
        "video_file_path": "/path/to/video.mp4"
    },
    "detection": {},
    "processing": {}
}

is_valid, errors = manager.validate_config(config)
if not is_valid:
    for error in errors:
        print(f"Validation error: {error}")
```

## Backward Compatibility

All changes are fully backward compatible:

1. **Legacy string device_id**: Configurations with `device_id` as a string path still work
2. **Existing code**: All existing code using `get_camera_config()` continues to work unchanged
3. **Default behavior**: Without explicit configuration, system defaults to camera input (device_id=0)
4. **Optional fields**: All new fields are optional with sensible defaults

## Testing

A comprehensive test suite was created and verified all functionality:
- ✅ Video source type detection
- ✅ File input detection (new and legacy)
- ✅ Video file path resolution and validation
- ✅ Camera configuration resolution
- ✅ Configuration validation
- ✅ Backward compatibility

## Files Modified

1. `/Users/jchadwick/code/billiards-trainer/backend/vision/config_manager.py`
   - Added 4 new helper methods
   - Enhanced validation logic

2. `/Users/jchadwick/code/billiards-trainer/backend/config/models/schemas.py`
   - Added `VideoSourceType` enum
   - Added 6 new fields to `CameraSettings`
   - Added model validator for video configuration

## Files Created

1. `/Users/jchadwick/code/billiards-trainer/backend/vision/config_manager_video_example.py`
   - Comprehensive usage examples for all new methods

2. `/Users/jchadwick/code/billiards-trainer/VIDEO_CONFIG_UPDATES.md`
   - This documentation file

## Code Quality

- ✅ All code follows existing patterns
- ✅ Full type hints provided
- ✅ Comprehensive docstrings
- ✅ Passes ruff linter
- ✅ Passes Python syntax check
- ✅ All tests pass
- ✅ Backward compatible
