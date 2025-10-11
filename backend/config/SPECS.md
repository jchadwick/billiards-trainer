# Configuration Module Specification

## Module Purpose

The Configuration module provides centralized management of all system settings, including persistence, validation, dynamic updates, and profile management. It ensures consistent configuration across all modules, handles environment-specific settings, and enables runtime configuration changes without system restart.

## Functional Requirements

### 1. Configuration Management

#### 1.1 Configuration Loading
- **FR-CFG-001**: Load configuration from multiple sources (files, environment, CLI)
- **FR-CFG-002**: Support configuration file formats (JSON, YAML, INI)
- **FR-CFG-003**: Merge configurations with proper precedence rules
- **FR-CFG-004**: Provide default values for all settings
- **FR-CFG-005**: Support configuration inheritance and overrides

#### 1.2 Configuration Validation
- **FR-CFG-006**: Validate all configuration values against schemas
- **FR-CFG-007**: Check value ranges and constraints
- **FR-CFG-008**: Verify interdependent settings consistency
- **FR-CFG-009**: Provide detailed validation error messages
- **FR-CFG-010**: Suggest corrections for invalid values

#### 1.3 Configuration Persistence
- **FR-CFG-011**: Save configuration changes to persistent storage
- **FR-CFG-012**: Maintain configuration history/versions
- **FR-CFG-013**: Support atomic configuration updates
- **FR-CFG-014**: Provide configuration backup and restore
- **FR-CFG-015**: Handle concurrent configuration access

### 2. Dynamic Configuration

#### 2.1 Runtime Updates
- **FR-CFG-016**: Apply configuration changes without restart
- **FR-CFG-017**: Notify modules of relevant changes
- **FR-CFG-018**: Support configuration hot-reload
- **FR-CFG-019**: Rollback failed configuration changes
- **FR-CFG-020**: Queue configuration changes for batch application

#### 2.2 Module Registration
- **FR-CFG-021**: Allow modules to register configuration needs
- **FR-CFG-022**: Track which modules use which settings
- **FR-CFG-023**: Validate module-specific configurations
- **FR-CFG-024**: Provide module configuration interfaces
- **FR-CFG-025**: Handle module configuration conflicts

### 3. Profile Management

#### 3.1 Configuration Profiles
- **FR-CFG-026**: Support multiple named configuration profiles
- **FR-CFG-027**: Switch between profiles at runtime
- **FR-CFG-028**: Import and export profiles
- **FR-CFG-029**: Merge profiles with base configuration
- **FR-CFG-030**: Auto-select profiles based on conditions

#### 3.2 User Preferences
- **FR-CFG-031**: Store user-specific preferences
- **FR-CFG-032**: Support per-user configuration overrides
- **FR-CFG-033**: Migrate user settings between versions
- **FR-CFG-034**: Reset preferences to defaults
- **FR-CFG-035**: Track preference usage statistics

### 4. Environment Management

#### 4.1 Environment Detection
- **FR-CFG-036**: Detect runtime environment (dev, test, prod)
- **FR-CFG-037**: Load environment-specific configurations
- **FR-CFG-038**: Override settings with environment variables
- **FR-CFG-039**: Validate environment compatibility
- **FR-CFG-040**: Support Docker/container environments

#### 4.2 Hardware Configuration
- **FR-CFG-041**: Detect available hardware capabilities
- **FR-CFG-042**: Auto-configure based on hardware
- **FR-CFG-043**: Warn about hardware limitations
- **FR-CFG-044**: Optimize settings for performance
- **FR-CFG-045**: Handle hardware changes dynamically

### 5. Configuration API

#### 5.1 Access Interface
- **FR-CFG-046**: Provide typed configuration access methods
- **FR-CFG-047**: Support configuration queries and searches
- **FR-CFG-048**: Enable configuration subscriptions
- **FR-CFG-049**: Provide configuration metadata
- **FR-CFG-050**: Support configuration transactions

#### 5.2 Management Interface
- **FR-CFG-051**: Expose configuration REST API
- **FR-CFG-052**: Provide configuration CLI tools
- **FR-CFG-053**: Support bulk configuration operations
- **FR-CFG-054**: Enable configuration import/export
- **FR-CFG-055**: Provide configuration documentation

## Non-Functional Requirements

### Performance Requirements
- **NFR-CFG-001**: Load configuration in < 100ms
- **NFR-CFG-002**: Apply changes in < 50ms
- **NFR-CFG-003**: Support 1000+ configuration parameters
- **NFR-CFG-004**: Handle 100+ concurrent config reads/sec
- **NFR-CFG-005**: Minimal memory footprint (< 50MB)

### Reliability Requirements
- **NFR-CFG-006**: No data loss during configuration updates
- **NFR-CFG-007**: Atomic configuration transactions
- **NFR-CFG-008**: Graceful handling of corrupted configs
- **NFR-CFG-009**: Automatic backup before changes
- **NFR-CFG-010**: Recovery from configuration errors

### Security Requirements
- **NFR-CFG-011**: Encrypt sensitive configuration values
- **NFR-CFG-012**: Secure storage of credentials
- **NFR-CFG-013**: Audit trail for configuration changes
- **NFR-CFG-014**: Role-based configuration access
- **NFR-CFG-015**: Prevent injection through config values

### Usability Requirements
- **NFR-CFG-016**: Self-documenting configuration format
- **NFR-CFG-017**: Clear error messages with solutions
- **NFR-CFG-018**: Configuration validation before apply
- **NFR-CFG-019**: Undo/redo for configuration changes
- **NFR-CFG-020**: Configuration diff and comparison

## Interface Specifications

### Configuration Module Interface

```python
from typing import Any, Dict, List, Optional, Callable, Type
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import json

class ConfigSource(Enum):
    DEFAULT = "default"
    FILE = "file"
    ENVIRONMENT = "environment"
    CLI = "cli"
    API = "api"
    RUNTIME = "runtime"

class ConfigFormat(Enum):
    JSON = "json"
    YAML = "yaml"
    INI = "ini"
    ENV = "env"

@dataclass
class ConfigValue:
    """Configuration value with metadata"""
    key: str
    value: Any
    source: ConfigSource
    timestamp: float
    validated: bool
    schema: Optional[Dict] = None
    description: Optional[str] = None

@dataclass
class ConfigChange:
    """Configuration change event"""
    key: str
    old_value: Any
    new_value: Any
    source: ConfigSource
    timestamp: float
    applied: bool

@dataclass
class ConfigProfile:
    """Named configuration profile"""
    name: str
    description: str
    settings: Dict[str, Any]
    parent: Optional[str] = None  # Parent profile to inherit from
    conditions: Optional[Dict] = None  # Auto-activation conditions

class ConfigurationModule:
    """Main configuration interface"""

    def __init__(self, config_dir: Path = Path("config")):
        """Initialize configuration module"""
        pass

    # Loading and Saving
    def load_config(self,
                   path: Path,
                   format: ConfigFormat = ConfigFormat.JSON) -> bool:
        """Load configuration from file"""
        pass

    def save_config(self,
                   path: Optional[Path] = None,
                   format: ConfigFormat = ConfigFormat.JSON) -> bool:
        """Save current configuration to file"""
        pass

    def reload_config(self) -> bool:
        """Reload configuration from all sources"""
        pass

    # Getting and Setting
    def get(self,
           key: str,
           default: Any = None,
           type_hint: Optional[Type] = None) -> Any:
        """Get configuration value"""
        pass

    def set(self,
           key: str,
           value: Any,
           source: ConfigSource = ConfigSource.RUNTIME,
           persist: bool = False) -> bool:
        """Set configuration value"""
        pass

    def get_all(self, prefix: Optional[str] = None) -> Dict[str, Any]:
        """Get all configuration values"""
        pass

    def update(self,
              values: Dict[str, Any],
              source: ConfigSource = ConfigSource.RUNTIME) -> List[ConfigChange]:
        """Update multiple configuration values"""
        pass

    # Validation
    def validate(self,
                key: Optional[str] = None,
                schema: Optional[Dict] = None) -> Tuple[bool, List[str]]:
        """Validate configuration values"""
        pass

    def register_schema(self,
                       prefix: str,
                       schema: Dict) -> None:
        """Register validation schema for config prefix"""
        pass

    # Profiles
    def create_profile(self,
                      name: str,
                      settings: Optional[Dict] = None) -> ConfigProfile:
        """Create new configuration profile"""
        pass

    def load_profile(self, name: str) -> bool:
        """Load and apply configuration profile"""
        pass

    def list_profiles(self) -> List[ConfigProfile]:
        """List available configuration profiles"""
        pass

    def export_profile(self,
                      name: str,
                      path: Path) -> bool:
        """Export profile to file"""
        pass

    # Subscriptions
    def subscribe(self,
                 pattern: str,
                 callback: Callable[[ConfigChange], None]) -> str:
        """Subscribe to configuration changes"""
        pass

    def unsubscribe(self, subscription_id: str) -> bool:
        """Unsubscribe from configuration changes"""
        pass

    # Module Registration
    def register_module(self,
                       module_name: str,
                       config_spec: Dict) -> None:
        """Register module configuration requirements"""
        pass

    def get_module_config(self, module_name: str) -> Dict[str, Any]:
        """Get module-specific configuration"""
        pass

    # Utilities
    def reset_to_defaults(self, prefix: Optional[str] = None) -> None:
        """Reset configuration to defaults"""
        pass

    def diff(self,
            other: Optional[Dict] = None,
            profile: Optional[str] = None) -> Dict[str, Tuple[Any, Any]]:
        """Compare configurations"""
        pass

    def get_metadata(self, key: str) -> ConfigValue:
        """Get configuration metadata"""
        pass

    def get_history(self,
                   key: Optional[str] = None,
                   limit: int = 10) -> List[ConfigChange]:
        """Get configuration change history"""
        pass
```

### Configuration Schema

```python
from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Any
from pathlib import Path

class ConfigMetadata(BaseModel):
    """Configuration file metadata"""
    version: str = "1.0.0"
    application: str = "billiards-trainer"
    created: str  # ISO timestamp
    modified: str  # ISO timestamp
    profile: Optional[str] = None
    environment: Optional[str] = None

class ConfigPaths(BaseModel):
    """File and directory paths"""
    config_dir: Path = Path("config")
    data_dir: Path = Path("data")
    log_dir: Path = Path("logs")
    cache_dir: Path = Path(".cache")
    profiles_dir: Path = Path("config/profiles")

class ConfigSources(BaseModel):
    """Configuration source priorities"""
    enable_files: bool = True
    enable_environment: bool = True
    enable_cli: bool = True
    file_paths: List[Path] = [
        Path("config/default.json"),
        Path("config/local.json")
    ]
    precedence: List[str] = [
        "cli", "environment", "file", "default"
    ]

class ValidationRules(BaseModel):
    """Configuration validation settings"""
    strict_mode: bool = False
    allow_unknown: bool = False
    type_checking: bool = True
    range_checking: bool = True
    dependency_checking: bool = True
    auto_correct: bool = False

class PersistenceSettings(BaseModel):
    """Configuration persistence settings"""
    auto_save: bool = True
    save_interval: int = 60  # seconds
    backup_count: int = 5
    compression: bool = True
    encryption: bool = False
    atomic_writes: bool = True

class HotReloadSettings(BaseModel):
    """Hot reload configuration"""
    enabled: bool = True
    watch_files: bool = True
    watch_interval: int = 1  # seconds
    reload_delay: int = 0  # milliseconds
    notify_modules: bool = True
    validation_before_reload: bool = True

class ConfigurationSettings(BaseModel):
    """Main configuration settings"""
    metadata: ConfigMetadata
    paths: ConfigPaths
    sources: ConfigSources
    validation: ValidationRules
    persistence: PersistenceSettings
    hot_reload: HotReloadSettings

    # Module configurations
    modules: Dict[str, Dict[str, Any]] = {}

    # User preferences
    preferences: Dict[str, Any] = {}

    # Feature flags
    features: Dict[str, bool] = {}
```

### Module Configuration Registration

```python
# Example module configuration specification
module_config_spec = {
    "module_name": "vision",
    "version": "1.0.0",
    "configuration": {
        "camera": {
            "type": "object",
            "properties": {
                "device_id": {
                    "type": "integer",
                    "default": 0,
                    "minimum": 0,
                    "description": "Camera device index"
                },
                "resolution": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "default": [1920, 1080],
                    "description": "Camera resolution [width, height]"
                },
                "fps": {
                    "type": "integer",
                    "default": 30,
                    "minimum": 15,
                    "maximum": 60,
                    "description": "Frames per second"
                }
            },
            "required": ["device_id"]
        },
        "detection": {
            "type": "object",
            "properties": {
                "sensitivity": {
                    "type": "number",
                    "default": 0.8,
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "Detection sensitivity"
                }
            }
        }
    },
    "dependencies": ["core"],
    "conflicts": [],
    "callbacks": {
        "on_change": "vision.on_config_change",
        "validator": "vision.validate_config"
    }
}
```

## Configuration Flow

### Configuration Loading Process

```python
def configuration_loading_flow():
    """
    Configuration loading and merging process

    1. Load Defaults
       - Built-in default values
       - Module default configurations
       - System default profile

    2. Load Files
       - Read configuration files in order
       - Parse based on file extension
       - Validate against schemas

    3. Load Environment
       - Scan environment variables
       - Parse and convert types
       - Apply prefix filtering

    4. Apply CLI Arguments
       - Parse command-line arguments
       - Override specific values
       - Handle configuration flags

    5. Merge and Validate
       - Merge in precedence order
       - Validate complete configuration
       - Check dependencies
       - Resolve conflicts

    6. Initialize Modules
       - Distribute module configs
       - Register change callbacks
       - Confirm module readiness
    """
    pass
```

### Dynamic Update Process

```python
def dynamic_update_flow(key: str, value: Any):
    """
    Runtime configuration update process

    1. Receive Update
       - Validate permission
       - Check value type
       - Verify constraints

    2. Pre-Update
       - Create backup
       - Notify subscribers
       - Check dependencies

    3. Apply Update
       - Update in-memory
       - Persist if required
       - Update derived values

    4. Post-Update
       - Notify modules
       - Trigger callbacks
       - Log change

    5. Verification
       - Confirm application
       - Test functionality
       - Rollback if failed
    """
    pass
```

## Success Criteria

### Functional Success Criteria

1. **Configuration Loading**
   - Load from all specified sources successfully
   - Merge configurations correctly by precedence
   - Apply defaults for missing values
   - Handle missing files gracefully

2. **Validation**
   - Catch 100% of schema violations
   - Provide actionable error messages
   - Suggest valid alternatives
   - Prevent invalid configurations

3. **Dynamic Updates**
   - Apply changes without restart
   - Notify all affected modules
   - Maintain consistency during updates
   - Support rollback on failure

4. **Profile Management**
   - Switch profiles seamlessly
   - Inherit settings correctly
   - Export/import without data loss
   - Auto-activate based on conditions

### Performance Success Criteria

1. **Load Time**
   - Initial configuration < 100ms
   - Profile switching < 50ms
   - Single value update < 10ms
   - Validation complete < 20ms

2. **Resource Usage**
   - Memory footprint < 50MB
   - CPU usage < 5% idle
   - File I/O minimized
   - Efficient change notifications

3. **Scalability**
   - Handle 1000+ parameters
   - Support 100+ modules
   - Manage 50+ profiles
   - Process 1000+ updates/second

### Reliability Success Criteria

1. **Data Integrity**
   - No configuration corruption
   - Atomic updates guaranteed
   - Successful recovery from crashes
   - Consistent state across modules

2. **Error Handling**
   - Graceful degradation
   - Clear error reporting
   - Automatic recovery attempts
   - Fallback to defaults

## Testing Requirements

### Unit Testing
- Test configuration loading from each source
- Validate merging logic
- Test schema validation
- Verify update mechanisms
- Coverage target: 95%

### Integration Testing
- Test multi-source configuration
- Verify module notifications
- Test profile switching
- Validate persistence
- Test hot reload

### Performance Testing
- Benchmark load times
- Test with large configurations
- Measure update latency
- Profile memory usage
- Stress test concurrent access

### Validation Testing
- Test all validation rules
- Verify error messages
- Test edge cases
- Validate type conversions
- Test constraint checking

## Implementation Guidelines

### Code Structure
```python
config/
├── __init__.py
├── manager.py          # Main configuration manager
├── loader/
│   ├── __init__.py
│   ├── file.py        # File loader
│   ├── env.py         # Environment loader
│   ├── cli.py         # CLI argument loader
│   └── merger.py      # Configuration merger
├── validator/
│   ├── __init__.py
│   ├── schema.py      # Schema validation
│   ├── rules.py       # Validation rules
│   └── types.py       # Type checking
├── storage/
│   ├── __init__.py
│   ├── persistence.py # Save/load logic
│   ├── backup.py      # Backup management
│   └── encryption.py  # Secure storage
├── profiles/
│   ├── __init__.py
│   ├── manager.py     # Profile management
│   └── conditions.py  # Auto-activation
├── models/
│   ├── __init__.py
│   └── schemas.py     # Configuration schemas
└── utils/
    ├── __init__.py
    ├── watcher.py     # File watching
    ├── differ.py      # Configuration diff
    └── converter.py   # Type conversion
```

### Key Dependencies
- **pydantic**: Schema validation
- **python-dotenv**: Environment loading
- **PyYAML**: YAML support
- **watchdog**: File monitoring
- **jsonschema**: JSON schema validation

### Development Priorities
1. Implement basic configuration loading
2. Add validation framework
3. Implement dynamic updates
4. Add profile support
5. Implement persistence
6. Add hot reload
7. Create management API
8. Add encryption support

## Module Configuration Specifications

### Vision Module Configuration

#### Video Source Configuration (vision.camera)

The camera configuration section supports multiple video input sources including hardware cameras, video files, and network streams.

**Video Source Type Selection**

- **Configuration Key**: `vision.camera.video_source_type`
- **Environment Variable**: `VISION__CAMERA__VIDEO_SOURCE_TYPE`
- **Type**: Enum
- **Values**: `"camera"`, `"file"`, `"stream"`
- **Default**: `"camera"`
- **Description**: Determines the video input source type
- **Validation**: Must be one of the specified enum values

**Video File Path**

- **Configuration Key**: `vision.camera.video_file_path`
- **Environment Variable**: `VISION__CAMERA__VIDEO_FILE_PATH`
- **Type**: Optional string (path)
- **Default**: `null`
- **Description**: Absolute or relative path to video file when source type is "file"
- **Validation**:
  - Must be a valid file path when video_source_type is "file"
  - File must exist and be readable
  - Supported formats: .mp4, .avi, .mov, .mkv, .webm
- **Example**: `"/path/to/video/pool_shot.mp4"` or `"data/videos/test_video.mp4"`

**Loop Video Playback**

- **Configuration Key**: `vision.camera.loop_video`
- **Environment Variable**: `VISION__CAMERA__LOOP_VIDEO`
- **Type**: Boolean
- **Default**: `false`
- **Description**: When true, video file will loop indefinitely; when false, processing stops at end of video
- **Validation**: Must be boolean value

**Video Start Frame**

- **Configuration Key**: `vision.camera.video_start_frame`
- **Environment Variable**: `VISION__CAMERA__VIDEO_START_FRAME`
- **Type**: Integer
- **Default**: `0`
- **Description**: Frame number to start video processing (0-based index)
- **Validation**:
  - Must be >= 0
  - Must be less than total video frames
  - Will be clamped to valid range if out of bounds
- **Example**: `150` (start from frame 150)

**Video End Frame**

- **Configuration Key**: `vision.camera.video_end_frame`
- **Environment Variable**: `VISION__CAMERA__VIDEO_END_FRAME`
- **Type**: Optional integer
- **Default**: `null` (process until end of video)
- **Description**: Frame number to stop video processing (0-based index, exclusive)
- **Validation**:
  - Must be > video_start_frame if specified
  - Must be <= total video frames
  - Will be clamped to valid range if out of bounds
- **Example**: `300` (stop at frame 300)

#### Video Processing Configuration (vision.processing)

Enhanced processing settings for video file input optimization.

**Video File Frame Cache**

- **Configuration Key**: `vision.processing.video_file_cache_frames`
- **Environment Variable**: `VISION__PROCESSING__VIDEO_FILE_CACHE_FRAMES`
- **Type**: Integer
- **Default**: `30`
- **Description**: Number of frames to pre-cache from video file for smoother playback
- **Validation**:
  - Must be >= 1
  - Must be <= 1000
  - Higher values use more memory but provide smoother playback
- **Performance Notes**:
  - Typical value: 30 frames (~1 second at 30fps)
  - High-performance: 60-120 frames
  - Low-memory: 5-10 frames

**Video Playback Speed**

- **Configuration Key**: `vision.processing.video_playback_speed`
- **Environment Variable**: `VISION__PROCESSING__VIDEO_PLAYBACK_SPEED`
- **Type**: Float
- **Default**: `1.0`
- **Description**: Playback speed multiplier (1.0 = normal speed, 0.5 = half speed, 2.0 = double speed)
- **Validation**:
  - Must be > 0.0
  - Must be <= 10.0
  - Values < 1.0 slow down playback
  - Values > 1.0 speed up playback
- **Examples**:
  - `0.25`: Quarter speed (slow motion)
  - `0.5`: Half speed
  - `1.0`: Normal speed
  - `2.0`: Double speed (fast forward)

### Projector Module Configuration

#### Video Feed Configuration (projector.network)

Network configuration for streaming video feed to the projector module.

**Stream Video Feed**

- **Configuration Key**: `projector.network.stream_video_feed`
- **Environment Variable**: `PROJECTOR__NETWORK__STREAM_VIDEO_FEED`
- **Type**: Boolean
- **Default**: `false`
- **Description**: Enable streaming of raw video feed to projector module via WebSocket
- **Validation**: Must be boolean value
- **Performance Impact**: Moderate to high network bandwidth usage

**Video Feed Quality**

- **Configuration Key**: `projector.network.video_feed_quality`
- **Environment Variable**: `PROJECTOR__NETWORK__VIDEO_FEED_QUALITY`
- **Type**: Integer
- **Default**: `85`
- **Description**: JPEG/WebP compression quality for video feed (1-100)
- **Validation**:
  - Must be >= 1
  - Must be <= 100
  - Higher values = better quality, larger data size
  - Lower values = lower quality, smaller data size
- **Recommendations**:
  - High quality: 90-100
  - Balanced: 75-85
  - Low bandwidth: 50-70

**Video Feed FPS**

- **Configuration Key**: `projector.network.video_feed_fps`
- **Environment Variable**: `PROJECTOR__NETWORK__VIDEO_FEED_FPS`
- **Type**: Integer
- **Default**: `15`
- **Description**: Target frames per second for video feed streaming
- **Validation**:
  - Must be >= 1
  - Must be <= 60
  - Will be clamped to source video FPS if higher
- **Recommendations**:
  - Debug/monitoring: 5-10 fps
  - Smooth preview: 15-20 fps
  - High quality: 25-30 fps

**Video Feed Scale**

- **Configuration Key**: `projector.network.video_feed_scale`
- **Environment Variable**: `PROJECTOR__NETWORK__VIDEO_FEED_SCALE`
- **Type**: Float
- **Default**: `0.5`
- **Description**: Resolution scale factor for video feed (0.1-1.0)
- **Validation**:
  - Must be >= 0.1
  - Must be <= 1.0
- **Examples**:
  - `1.0`: Full resolution (1920x1080 → 1920x1080)
  - `0.5`: Half resolution (1920x1080 → 960x540)
  - `0.25`: Quarter resolution (1920x1080 → 480x270)
- **Performance Impact**: Lower values significantly reduce bandwidth

**Video Feed Format**

- **Configuration Key**: `projector.network.video_feed_format`
- **Environment Variable**: `PROJECTOR__NETWORK__VIDEO_FEED_FORMAT`
- **Type**: Enum
- **Values**: `"jpeg"`, `"png"`, `"webp"`
- **Default**: `"jpeg"`
- **Description**: Image encoding format for video feed frames
- **Validation**: Must be one of the specified enum values
- **Format Characteristics**:
  - `jpeg`: Fastest, lossy compression, smallest size, best for streaming
  - `png`: Slower, lossless compression, larger size, best quality
  - `webp`: Modern format, good compression, browser-dependent support

### API Module Configuration

#### Video Feed API Configuration (api.video_feed)

HTTP endpoints for accessing the video feed independently of the WebSocket connection.

**Video Feed Enabled**

- **Configuration Key**: `api.video_feed.enabled`
- **Environment Variable**: `API__VIDEO_FEED__ENABLED`
- **Type**: Boolean
- **Default**: `false`
- **Description**: Enable HTTP video feed endpoints
- **Validation**: Must be boolean value
- **Dependencies**: Requires vision module to be active

**Video Feed Endpoint**

- **Configuration Key**: `api.video_feed.endpoint`
- **Environment Variable**: `API__VIDEO_FEED__ENDPOINT`
- **Type**: String
- **Default**: `"/api/v1/video/feed"`
- **Description**: HTTP endpoint path for single-frame video feed access
- **Validation**:
  - Must start with "/"
  - Must be unique (not conflict with other endpoints)
- **Usage**: GET request returns current frame as image

**MJPEG Stream Enabled**

- **Configuration Key**: `api.video_feed.mjpeg_stream`
- **Environment Variable**: `API__VIDEO_FEED__MJPEG_STREAM`
- **Type**: Boolean
- **Default**: `false`
- **Description**: Enable Motion JPEG (MJPEG) streaming endpoint for continuous video feed
- **Validation**: Must be boolean value
- **Performance Impact**: High - maintains persistent HTTP connections

**MJPEG Endpoint**

- **Configuration Key**: `api.video_feed.mjpeg_endpoint`
- **Environment Variable**: `API__VIDEO_FEED__MJPEG_ENDPOINT`
- **Type**: String
- **Default**: `"/api/v1/video/stream"`
- **Description**: HTTP endpoint path for MJPEG continuous stream
- **Validation**:
  - Must start with "/"
  - Must be unique (not conflict with other endpoints)
  - Only active when mjpeg_stream is true
- **Usage**: GET request returns multipart/x-mixed-replace stream

**Maximum Concurrent Clients**

- **Configuration Key**: `api.video_feed.max_clients`
- **Environment Variable**: `API__VIDEO_FEED__MAX_CLIENTS`
- **Type**: Integer
- **Default**: `5`
- **Description**: Maximum number of concurrent clients for video feed endpoints
- **Validation**:
  - Must be >= 1
  - Must be <= 100
  - Prevents resource exhaustion
- **Recommendations**:
  - Development: 5-10
  - Production: 20-50
  - High-load: 50-100

**Frame Buffer Size**

- **Configuration Key**: `api.video_feed.buffer_size`
- **Environment Variable**: `API__VIDEO_FEED__BUFFER_SIZE`
- **Type**: Integer
- **Default**: `10`
- **Description**: Number of frames to buffer for video feed endpoints
- **Validation**:
  - Must be >= 1
  - Must be <= 100
- **Purpose**:
  - Smooths frame delivery during temporary slowdowns
  - Prevents dropped frames with multiple clients
  - Higher values use more memory

### Configuration Examples

#### Example 1: Video File Input for Testing

```json
{
  "vision": {
    "camera": {
      "video_source_type": "file",
      "video_file_path": "data/videos/test_shot.mp4",
      "loop_video": true,
      "video_start_frame": 0,
      "video_end_frame": null
    },
    "processing": {
      "video_file_cache_frames": 30,
      "video_playback_speed": 1.0
    }
  }
}
```

#### Example 2: Video File Segment Processing

```json
{
  "vision": {
    "camera": {
      "video_source_type": "file",
      "video_file_path": "/opt/billiards-trainer/videos/recorded_game.mp4",
      "loop_video": false,
      "video_start_frame": 150,
      "video_end_frame": 450
    },
    "processing": {
      "video_file_cache_frames": 60,
      "video_playback_speed": 0.5
    }
  }
}
```

#### Example 3: Video Feed Streaming to Projector

```json
{
  "projector": {
    "network": {
      "stream_video_feed": true,
      "video_feed_quality": 80,
      "video_feed_fps": 20,
      "video_feed_scale": 0.5,
      "video_feed_format": "jpeg"
    }
  }
}
```

#### Example 4: HTTP Video Feed API

```json
{
  "api": {
    "video_feed": {
      "enabled": true,
      "endpoint": "/api/v1/video/feed",
      "mjpeg_stream": true,
      "mjpeg_endpoint": "/api/v1/video/stream",
      "max_clients": 10,
      "buffer_size": 15
    }
  }
}
```

#### Example 5: Complete Video Configuration

```json
{
  "vision": {
    "camera": {
      "video_source_type": "file",
      "video_file_path": "data/videos/test_video.mp4",
      "loop_video": true,
      "video_start_frame": 0,
      "video_end_frame": null,
      "fps": 30
    },
    "processing": {
      "video_file_cache_frames": 45,
      "video_playback_speed": 1.0,
      "enable_preprocessing": true,
      "enable_tracking": true
    }
  },
  "projector": {
    "network": {
      "stream_video_feed": true,
      "video_feed_quality": 85,
      "video_feed_fps": 15,
      "video_feed_scale": 0.5,
      "video_feed_format": "jpeg"
    }
  },
  "api": {
    "video_feed": {
      "enabled": true,
      "endpoint": "/api/v1/video/feed",
      "mjpeg_stream": true,
      "mjpeg_endpoint": "/api/v1/video/stream",
      "max_clients": 5,
      "buffer_size": 10
    }
  }
}
```

### Environment Variable Examples

```bash
# Video source configuration
export VISION__CAMERA__VIDEO_SOURCE_TYPE="file"
export VISION__CAMERA__VIDEO_FILE_PATH="/opt/videos/pool_game.mp4"
export VISION__CAMERA__LOOP_VIDEO="true"
export VISION__CAMERA__VIDEO_START_FRAME="0"
export VISION__CAMERA__VIDEO_END_FRAME="500"

# Video processing configuration
export VISION__PROCESSING__VIDEO_FILE_CACHE_FRAMES="30"
export VISION__PROCESSING__VIDEO_PLAYBACK_SPEED="1.0"

# Projector video feed configuration
export PROJECTOR__NETWORK__STREAM_VIDEO_FEED="true"
export PROJECTOR__NETWORK__VIDEO_FEED_QUALITY="85"
export PROJECTOR__NETWORK__VIDEO_FEED_FPS="15"
export PROJECTOR__NETWORK__VIDEO_FEED_SCALE="0.5"
export PROJECTOR__NETWORK__VIDEO_FEED_FORMAT="jpeg"

# API video feed configuration
export API__VIDEO_FEED__ENABLED="true"
export API__VIDEO_FEED__ENDPOINT="/api/v1/video/feed"
export API__VIDEO_FEED__MJPEG_STREAM="true"
export API__VIDEO_FEED__MJPEG_ENDPOINT="/api/v1/video/stream"
export API__VIDEO_FEED__MAX_CLIENTS="5"
export API__VIDEO_FEED__BUFFER_SIZE="10"
```

### Configuration Validation Rules

#### Video Source Configuration Validation

1. **Source Type Validation**
   - When `video_source_type` is `"file"`:
     - `video_file_path` must be provided and valid
     - File must exist and be readable
     - File format must be supported
   - When `video_source_type` is `"camera"`:
     - `device_id` must be valid
     - Camera must be accessible
   - When `video_source_type` is `"stream"`:
     - Stream URL must be provided (future enhancement)

2. **Frame Range Validation**
   - `video_start_frame` must be >= 0
   - `video_end_frame` must be > `video_start_frame` if specified
   - Both values will be clamped to actual video frame count
   - Warning if range exceeds video length

3. **Playback Speed Validation**
   - Must be positive number
   - Should warn if speed > 2.0 (may cause frame drops)
   - Should warn if speed < 0.1 (extremely slow)

#### Video Feed Configuration Validation

1. **Quality Validation**
   - `video_feed_quality` must be 1-100
   - Warn if quality > 95 (diminishing returns)
   - Warn if quality < 50 (poor quality)

2. **Performance Validation**
   - Warn if `video_feed_fps` > source camera/video fps
   - Warn if `video_feed_scale` * resolution > 1920x1080
   - Error if `max_clients` * `buffer_size` > 1000 frames (memory concern)

3. **Endpoint Validation**
   - Ensure endpoints don't conflict with existing API routes
   - Ensure endpoints start with "/"
   - Validate endpoint format (no spaces, special chars)

4. **Dependency Validation**
   - `stream_video_feed` requires vision module enabled
   - `api.video_feed.enabled` requires vision module enabled
   - `mjpeg_stream` requires `api.video_feed.enabled` = true

### Configuration Migration Notes

#### Version 1.0.0 → 1.1.0 (Video File Support)

**New Fields Added**:
- `vision.camera.video_source_type` (default: "camera")
- `vision.camera.video_file_path` (default: null)
- `vision.camera.loop_video` (default: false)
- `vision.camera.video_start_frame` (default: 0)
- `vision.camera.video_end_frame` (default: null)
- `vision.processing.video_file_cache_frames` (default: 30)
- `vision.processing.video_playback_speed` (default: 1.0)
- `projector.network.stream_video_feed` (default: false)
- `projector.network.video_feed_quality` (default: 85)
- `projector.network.video_feed_fps` (default: 15)
- `projector.network.video_feed_scale` (default: 0.5)
- `projector.network.video_feed_format` (default: "jpeg")
- `api.video_feed.enabled` (default: false)
- `api.video_feed.endpoint` (default: "/api/v1/video/feed")
- `api.video_feed.mjpeg_stream` (default: false)
- `api.video_feed.mjpeg_endpoint` (default: "/api/v1/video/stream")
- `api.video_feed.max_clients` (default: 5)
- `api.video_feed.buffer_size` (default: 10)

**Breaking Changes**: None - all new fields have sensible defaults

**Migration Actions**:
1. No action required for existing configurations
2. Existing camera configurations will continue to work with default `video_source_type="camera"`
3. To use video files, explicitly set `video_source_type="file"` and provide `video_file_path`
