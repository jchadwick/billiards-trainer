# Configuration System Comparison: Current vs V2

**Date:** 2025-10-20
**Author:** Analysis Agent
**Purpose:** Comprehensive analysis of configuration systems in billiards-trainer (current) and billiards-trainer-v2 to identify best practices for adoption

---

## Executive Summary

### Current Repository (billiards-trainer)
- **Simple singleton Config class** with dot-notation access (`config.get("vision.camera.device_id", 0)`)
- **Single JSON file** (config.json) with all defaults embedded
- **Lazy loading** on first access
- **No validation** system
- **Defaults in JSON file**, not in code

### V2 Repository (billiards-trainer-v2)
- **ConfigManager singleton** with section-based organization
- **Consumer-provided defaults** (strict policy: no hardcoded values)
- **Module-specific default files** (e.g., `physics/config_defaults.py`, `vision/config_defaults.py`)
- **Background persistence thread** with optimistic updates
- **Comprehensive validation** and error handling
- **Thread-safe operations** with RLock

### Recommendation
**Adopt V2's philosophy and patterns**, particularly:
1. Consumer-provided defaults in code (not JSON)
2. Module-specific default files for organization
3. Section-based API with proper typing
4. Validation layer
5. Thread-safe operations

---

## 1. Configuration Loading & Distribution Patterns

### Current System (billiards-trainer)

**File:** `/Users/jchadwick/code/billiards-trainer/backend/config.py`

**Pattern:**
```python
# Singleton with lazy loading
class Config:
    _instance = None
    _config_data = {}
    _config_file = None

    def get(self, key: str, default: Any = None) -> Any:
        # Lazy load on first access
        if not self._config_data and self._config_file is None:
            self._load_config()

        # Navigate dot-notation path
        keys = key.split('.')
        value = self._config_data
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value
```

**Usage:**
```python
from backend.config import config

# All modules use same singleton instance
device_id = config.get("vision.camera.device_id", 0)
fps = config.get("vision.camera.fps", 30)
```

**Strengths:**
- ✅ Simple, easy to understand
- ✅ Dot-notation is intuitive
- ✅ Single import for all consumers
- ✅ Defaults are provided at call site

**Weaknesses:**
- ❌ All defaults in JSON file (1142 lines!)
- ❌ No module organization
- ❌ No validation
- ❌ Not thread-safe
- ❌ No type safety

---

### V2 System (billiards-trainer-v2)

**File:** `/Users/jchadwick/code/billiards-trainer-v2/backend/config/config_manager.py`

**Pattern:**
```python
class ConfigManager:
    """Generic singleton with section-based organization."""
    _instance = None
    _lock = threading.RLock()
    _config = {}
    _dirty = False
    _persistence_thread = None

    def get(self, section: str, key: str, default: T) -> T:
        """Retrieve value with consumer-provided default."""
        with self._lock:
            if section not in self._config:
                return default
            section_dict = self._config[section]
            if key not in section_dict:
                return default
            return section_dict[key]

    def set(self, section: str, key: str, value: Any) -> None:
        """Set value with background persistence."""
        with self._lock:
            if section not in self._config:
                self._config[section] = {}
            self._config[section][key] = value
            self._mark_dirty()
```

**Module-Specific Defaults:**
```python
# physics/config_defaults.py
PHYSICS_CONFIG_SECTION = "physics"
PHYSICS_DEFAULTS = {
    "ball_radius_inches": 1.125,
    "ball_mass_kg": 0.17,
    "table_width_inches": 100.0,
    # ... comprehensive defaults with documentation
}

def get_physics_default(key: str) -> Any:
    """Get default value for physics parameter."""
    if key not in PHYSICS_DEFAULTS:
        raise KeyError(f"No default value found for: {key}")
    return PHYSICS_DEFAULTS[key]
```

**Usage:**
```python
from backend.config import ConfigManager
from .config_defaults import get_physics_default, PHYSICS_CONFIG_SECTION

config = ConfigManager()  # Singleton

# Consumer provides default explicitly
ball_radius = config.get(
    PHYSICS_CONFIG_SECTION,
    "ball_radius_inches",
    default=get_physics_default("ball_radius_inches")
)
```

**Strengths:**
- ✅ **Defaults in code**, close to usage
- ✅ Module-specific organization
- ✅ Thread-safe operations
- ✅ Background persistence
- ✅ Section-based API prevents collisions
- ✅ **No hardcoded values in logic**
- ✅ Type-safe with generics
- ✅ Comprehensive documentation

**Weaknesses:**
- ⚠️ More verbose at call site
- ⚠️ Requires discipline (must provide defaults)
- ⚠️ More files to maintain

---

## 2. Default Value Handling

### Current System

**Location:** Single `config.json` file (1142 lines)

```json
{
  "vision": {
    "camera": {
      "device_id": 0,
      "backend": "auto",
      "resolution": [1920, 1080],
      "fps": 30,
      "exposure_mode": "auto",
      "gain": 1.0,
      // ... 1100+ more lines
    }
  }
}
```

**Issues:**
1. **Disconnected from code** - defaults far from usage
2. **Hard to document** - JSON doesn't support comments well
3. **No type information** - just values
4. **Monolithic** - one giant file
5. **Merge conflicts** - single file touched by all modules

---

### V2 System

**Location:** Module-specific Python files

**Physics Defaults:** `/Users/jchadwick/code/billiards-trainer-v2/backend/physics/config_defaults.py`
```python
"""Default configuration values for the Physics Module.

Physical constants and default values:
- Standard pool ball radius: 1.125 inches (57.15mm)
- Standard pool ball mass: 0.17 kg (6 oz)
- Standard 9-foot table: 100" x 50" playing surface
- Typical coefficient of restitution: 0.95 (nearly elastic)
- Typical cloth friction: 0.01 (smooth felt surface)
"""

PHYSICS_DEFAULTS: Dict[str, Any] = {
    # Ball physical properties (based on standard billiard balls)
    "ball_radius_inches": 1.125,  # Standard: 2.25" diameter
    "ball_mass_kg": 0.17,  # Standard: ~6 oz
    "ball_coefficient_of_restitution": 0.95,  # Nearly elastic

    # Table physical properties
    "table_width_inches": 100.0,  # 9-foot table
    "table_height_inches": 50.0,
    "table_cloth_friction": 0.01,  # Smooth felt

    # Simulation parameters
    "simulation_time_step_seconds": 0.01,  # 10ms per step
    "velocity_stop_threshold_inches_per_second": 0.1,
}
```

**Vision Defaults:** `/Users/jchadwick/code/billiards-trainer-v2/backend/vision/config_defaults.py`
```python
"""Default configuration values for the Vision module.

These values are designed to work with 4K (3840x2160)
billiards table images but can be scaled for different resolutions.

All values can be overridden through the Configuration Module.
NO values should be hardcoded in production logic.
"""

DEFAULT_VISION_CONFIG: Dict[str, Any] = {
    "vision": {
        "detection": {
            "opencv": {
                "min_ball_radius": 35,  # For 4K resolution
                "max_ball_radius": 65,
                "downscale_factor": 0.5,  # Performance optimization
                # ... comprehensive, documented defaults
            }
        }
    }
}
```

**Advantages:**
1. ✅ **Self-documenting** - Python docstrings + inline comments
2. ✅ **Close to usage** - defaults in same module
3. ✅ **Type-safe** - Python type hints
4. ✅ **Organized** - one file per module
5. ✅ **Physical meaning** - can include units, rationale
6. ✅ **Version controlled** - clear history
7. ✅ **No merge conflicts** - module boundaries

---

## 3. Validation Approaches

### Current System

**Validation:** ❌ None (except basic type checking at runtime)

**Example Issues:**
```python
# No validation - bad values accepted silently
config.set("vision.camera.fps", -1)  # Invalid!
config.set("vision.camera.resolution", [0, 0])  # Invalid!
```

---

### V2 System

**Validation:** Vision module has comprehensive validation in `config_manager.py`

```python
def validate_config(self, config: dict[str, Any]) -> tuple[bool, list[str]]:
    """Validate configuration dictionary.

    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []

    # Check video file path if configured
    video_file_path = camera_config.get("video_file_path")
    if video_file_path:
        try:
            path_obj = Path(video_file_path)
            if not path_obj.exists():
                errors.append(f"Video file not found: {video_file_path}")
            elif not path_obj.is_file():
                errors.append(f"Not a file: {video_file_path}")
        except Exception as e:
            errors.append(f"Validation error: {e}")

    # Validate fps range
    if "fps" in camera_config:
        fps = camera_config["fps"]
        if not isinstance(fps, int) or not (15 <= fps <= 120):
            errors.append("fps must be 15-120")

    # Validate resolution
    if "resolution" in camera_config:
        resolution = camera_config["resolution"]
        if not isinstance(resolution, (list, tuple)) or len(resolution) != 2:
            errors.append("resolution must be [width, height]")

    return len(errors) == 0, errors
```

**Benefits:**
- ✅ Early error detection
- ✅ Helpful error messages
- ✅ Type validation
- ✅ Range validation
- ✅ File existence checks

---

## 4. Cross-Module Configuration Sharing

### Current System

**Pattern:** All modules import same singleton

```python
# backend/core/analysis/assistance.py
from backend.config import Config, config

class AssistanceLevel(Enum):
    BEGINNER = "beginner"
    # ...

# Direct usage
difficulty = config.get("core.assistance.difficulty_levels.beginner", 0.2)
```

**Issue:** **Hardcoded keys scattered across codebase**
- Each module must know exact JSON path
- No centralized definition of keys
- Easy to typo: `"core.assistance.level"` vs `"core.assistance.difficulty_level"`

---

### V2 System

**Pattern:** Module-specific section + defaults

```python
# physics/collision.py
from backend.config import ConfigManager
from .config_defaults import get_physics_default, PHYSICS_CONFIG_SECTION

class CollisionDetector:
    def __init__(self, config: Optional[ConfigManager] = None):
        self._config = config if config is not None else ConfigManager()

        # Load configuration with module defaults
        self._collision_tolerance = self._config.get(
            PHYSICS_CONFIG_SECTION,
            "collision_distance_tolerance_inches",
            default=get_physics_default("collision_distance_tolerance_inches")
        )
```

**Benefits:**
1. ✅ **Section isolation** - `PHYSICS_CONFIG_SECTION` constant
2. ✅ **Centralized keys** - defined in `config_defaults.py`
3. ✅ **Type-safe defaults** - from typed dictionaries
4. ✅ **Dependency injection** - `config` parameter for testing
5. ✅ **No magic strings** - use constants

---

## 5. Best Practices from V2

### A. Strict "No Hardcoded Values" Policy

**V2 Documentation:**
```python
"""
Per FR-STR-003, consumers MUST provide their own defaults when calling
ConfigManager.get(). This module serves as the canonical source of
default values for the Physics Module.

All values can be overridden through the Configuration Module.
NO values should be hardcoded in production logic.
"""
```

**Enforcement:**
```python
# ❌ BAD - hardcoded value
ball_radius = 1.125

# ✅ GOOD - configuration with default
ball_radius = config.get(
    PHYSICS_CONFIG_SECTION,
    "ball_radius_inches",
    default=get_physics_default("ball_radius_inches")
)
```

---

### B. Module-Specific Default Files

**Organization:**
```
backend/
├── config/
│   └── config_manager.py          # Generic manager
├── physics/
│   ├── config_defaults.py         # Physics defaults
│   └── collision.py               # Uses physics defaults
└── vision/
    ├── config_defaults.py         # Vision defaults
    └── detection/
        └── yolo_detector.py       # Uses vision defaults
```

**Benefits:**
- ✅ Clear ownership
- ✅ Easy to find defaults
- ✅ No merge conflicts
- ✅ Module boundaries enforced

---

### C. Background Persistence with Debouncing

**V2 Implementation:**
```python
def _persistence_worker(cls) -> None:
    """Background worker that periodically saves dirty configuration.

    Implements debouncing by waiting 5 seconds after the last
    change before actually persisting to disk.
    """
    DEBOUNCE_SECONDS = 5.0
    CHECK_INTERVAL_SECONDS = 1.0

    while not cls._shutdown_event.is_set():
        # Wait for changes
        if cls._shutdown_event.wait(timeout=CHECK_INTERVAL_SECONDS):
            break

        with cls._lock:
            if not cls._dirty:
                continue

            # Check debounce period
            time_since_change = time.time() - cls._last_change_time
            if time_since_change < DEBOUNCE_SECONDS:
                continue

            # Save to disk
            cls._save_to_file()
            cls._dirty = False
```

**Benefits:**
- ✅ Non-blocking updates
- ✅ Prevents disk thrashing
- ✅ Graceful shutdown (saves pending changes)
- ✅ Atomic file writes (temp file + rename)

---

### D. Thread Safety

**V2 Implementation:**
```python
class ConfigManager:
    _lock = threading.RLock()  # Reentrant lock

    def get(self, section: str, key: str, default: T) -> T:
        with self._lock:
            # ... safe access

    def set(self, section: str, key: str, value: Any) -> None:
        with self._lock:
            # ... safe modification
```

**Current System:** ❌ No thread safety

---

### E. Type Safety with Generics

**V2 Implementation:**
```python
from typing import TypeVar

T = TypeVar('T')

def get(self, section: str, key: str, default: T) -> T:
    """Return type matches default parameter type."""
    # ...
    return value  # Type: T

# Usage - type inference works
fps: int = config.get('video', 'fps', default=30)
resolution: list = config.get('video', 'resolution', default=[1920, 1080])
```

---

### F. Comprehensive Documentation

**V2 Style:**
```python
PHYSICS_DEFAULTS: Dict[str, Any] = {
    # Ball physical properties (based on standard billiard balls)
    "ball_radius_inches": 1.125,  # Standard: 2.25" diameter
    "ball_mass_kg": 0.17,  # Standard: ~6 oz

    # Physics constants (typically don't change these)
    "gravity_acceleration_inches_per_second_squared": 386.089,  # 9.81 m/s² in inches/s²
}
```

**Includes:**
- Module-level docstring explaining purpose
- Physical constants and their sources
- Units for all values
- Rationale for defaults
- Warnings about what not to change

---

## 6. Migration Strategy

### Phase 1: Add Module Default Files (Low Risk)

Create default files without changing existing code:

```python
# backend/vision/config_defaults.py
"""Vision module configuration defaults."""

VISION_CONFIG_SECTION = "vision"

VISION_DEFAULTS = {
    "camera": {
        "device_id": 0,
        "backend": "auto",
        "resolution": [1920, 1080],
        "fps": 30,
        # ... extract from config.json
    }
}

def get_vision_default(key: str) -> Any:
    """Get default value for vision configuration."""
    # Navigate nested dict
    keys = key.split('.')
    value = VISION_DEFAULTS
    for k in keys:
        if isinstance(value, dict) and k in value:
            value = value[k]
        else:
            raise KeyError(f"No default value for: {key}")
    return value
```

**Similar for:**
- `backend/core/config_defaults.py`
- `backend/api/config_defaults.py`

---

### Phase 2: Add Helper Methods (Medium Risk)

Extend current Config class with section-based methods:

```python
# backend/config.py
class Config:
    # ... existing code ...

    def get_section(self, section: str) -> dict[str, Any]:
        """Get entire configuration section."""
        return self.get_all().get(section, {})

    def get_with_section(self, section: str, key: str, default: Any = None) -> Any:
        """Get value using section.key instead of full path."""
        full_key = f"{section}.{key}"
        return self.get(full_key, default)
```

---

### Phase 3: Add Validation Layer (Medium Risk)

```python
# backend/config_validation.py
from typing import Any, Callable

class ConfigValidator:
    """Validation rules for configuration values."""

    def __init__(self):
        self.validators: dict[str, Callable] = {}

    def register(self, key: str, validator: Callable[[Any], tuple[bool, str]]):
        """Register validation rule for a key."""
        self.validators[key] = validator

    def validate(self, key: str, value: Any) -> tuple[bool, str]:
        """Validate value against registered rules."""
        if key not in self.validators:
            return True, ""
        return self.validators[key](value)

# Register validators
validator = ConfigValidator()
validator.register("vision.camera.fps", lambda v: (
    (15 <= v <= 120, "FPS must be 15-120")
    if isinstance(v, int) else (False, "FPS must be integer")
))
```

---

### Phase 4: Add Thread Safety (Low Risk)

```python
import threading

class Config:
    _lock = threading.RLock()

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            # ... existing logic

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            # ... existing logic
```

---

### Phase 5: Gradual Refactoring (Ongoing)

Update modules one at a time:

```python
# OLD STYLE
from backend.config import config
device_id = config.get("vision.camera.device_id", 0)

# NEW STYLE
from backend.config import config
from backend.vision.config_defaults import get_vision_default, VISION_CONFIG_SECTION

device_id = config.get_with_section(
    VISION_CONFIG_SECTION,
    "camera.device_id",
    default=get_vision_default("camera.device_id")
)
```

---

## 7. Key Takeaways

### What to Adopt from V2

1. **✅ Consumer-provided defaults in code**
   - Create module-specific `config_defaults.py` files
   - Move defaults out of JSON into Python
   - Document physical meanings and units

2. **✅ Section-based organization**
   - Define section constants (e.g., `VISION_CONFIG_SECTION`)
   - Use sections to prevent key collisions
   - Clear module boundaries

3. **✅ Validation layer**
   - Validate at set time, not runtime
   - Provide helpful error messages
   - Type checking, range checking, file existence

4. **✅ Thread safety**
   - Add RLock to Config class
   - Protect all reads/writes

5. **✅ Better documentation**
   - Docstrings explaining defaults
   - Units for physical quantities
   - Rationale for values

### What to Keep from Current System

1. **✅ Simple dot-notation API**
   - `config.get("vision.camera.fps", 30)` is intuitive
   - Don't force verbose section-based API everywhere

2. **✅ Single import**
   - Keep `from backend.config import config`
   - Singleton pattern works well

3. **✅ Lazy loading**
   - No need for background persistence (current system is simpler)
   - Explicit save/reload is fine for this use case

### What NOT to Adopt

1. **❌ Background persistence thread**
   - Overkill for current use case
   - Adds complexity
   - Explicit save is clearer

2. **❌ Overly verbose API**
   - V2's `config.get(SECTION, "key", default=get_default("key"))` is verbose
   - Current `config.get("section.key", default)` is better

---

## 8. Recommended Hybrid Approach

### Goals
1. Keep current system's simplicity
2. Add V2's best practices (defaults in code, validation, thread safety)
3. Gradual, non-breaking migration

### Enhanced Config Class

```python
"""Enhanced configuration system combining simplicity with best practices."""

import json
import logging
import threading
from pathlib import Path
from typing import Any, Optional, TypeVar

T = TypeVar('T')

logger = logging.getLogger(__name__)


class Config:
    """Configuration singleton with validation and thread safety.

    Enhanced version that maintains backward compatibility while adding:
    - Thread-safe operations
    - Validation layer
    - Section-based helpers
    - Better error messages
    """

    _instance: Optional['Config'] = None
    _lock = threading.RLock()
    _config_data: dict[str, Any] = {}
    _config_file: Optional[Path] = None
    _validator: Optional['ConfigValidator'] = None

    def get(self, key: str, default: T) -> T:
        """Get configuration value with dot notation (thread-safe).

        Args:
            key: Dot-separated key path (e.g., "vision.camera.fps")
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        with self._lock:
            keys = key.split('.')
            value = self._config_data

            for k in keys:
                if isinstance(value, dict) and k in value:
                    value = value[k]
                else:
                    logger.debug(f"Config key not found: {key}, using default: {default}")
                    return default

            return value

    def set(self, key: str, value: Any) -> None:
        """Set configuration value with validation (thread-safe).

        Args:
            key: Dot-separated key path
            value: Value to set

        Raises:
            ValueError: If validation fails
        """
        with self._lock:
            # Validate if validator registered
            if self._validator:
                is_valid, error = self._validator.validate(key, value)
                if not is_valid:
                    raise ValueError(f"Invalid config value for {key}: {error}")

            # Navigate and set
            keys = key.split('.')
            data = self._config_data

            for k in keys[:-1]:
                if k not in data or not isinstance(data[k], dict):
                    data[k] = {}
                data = data[k]

            data[keys[-1]] = value
            logger.info(f"Config updated: {key} = {value}")

    def get_section(self, section: str) -> dict[str, Any]:
        """Get entire configuration section (thread-safe).

        Args:
            section: Section name (e.g., "vision", "core")

        Returns:
            Section dictionary (copy to prevent external modification)
        """
        with self._lock:
            section_data = self._config_data.get(section, {})
            return section_data.copy() if isinstance(section_data, dict) else {}


class ConfigValidator:
    """Validation rules for configuration values."""

    def __init__(self):
        self.validators: dict[str, Callable[[Any], tuple[bool, str]]] = {}

    def register(self, key: str, validator: Callable[[Any], tuple[bool, str]]):
        """Register validation rule."""
        self.validators[key] = validator

    def validate(self, key: str, value: Any) -> tuple[bool, str]:
        """Validate value."""
        if key in self.validators:
            return self.validators[key](value)
        return True, ""
```

### Module Default Files

```python
# backend/vision/config_defaults.py
"""Vision module configuration defaults.

Provides default values for all vision-related configuration parameters.
These defaults are designed for 4K resolution (3840x2160) but can be scaled.

Per the "no hardcoded values" policy, all production logic should use
these defaults via the configuration system rather than hardcoding values.
"""

VISION_CONFIG_SECTION = "vision"

VISION_DEFAULTS = {
    "camera": {
        "device_id": 0,  # Default camera index
        "backend": "auto",  # Auto-detect backend
        "resolution": [1920, 1080],  # Full HD resolution
        "fps": 30,  # Standard frame rate
        "exposure_mode": "auto",
        "gain": 1.0,
        # ... more defaults with documentation
    },
    "detection": {
        "yolo_model_path": "models/billiards-yolov8n.onnx",
        "yolo_confidence": 0.4,  # 40% confidence threshold
        "min_ball_radius": 15,  # Minimum ball radius (pixels)
        "max_ball_radius": 26,  # Maximum ball radius (pixels)
        # ... more defaults
    }
}

def get_vision_default(key: str) -> Any:
    """Get default value for vision configuration key.

    Args:
        key: Dot-separated key path (e.g., "camera.fps")

    Returns:
        Default value

    Raises:
        KeyError: If key not found in defaults

    Example:
        >>> fps = get_vision_default("camera.fps")
        >>> print(fps)
        30
    """
    keys = key.split('.')
    value = VISION_DEFAULTS

    for k in keys:
        if isinstance(value, dict) and k in value:
            value = value[k]
        else:
            raise KeyError(
                f"No default value for vision.{key}. "
                f"Available keys: {list(VISION_DEFAULTS.keys())}"
            )

    return value
```

### Usage Example

```python
# backend/vision/detection/yolo_detector.py
from backend.config import config
from ..config_defaults import get_vision_default

class YOLODetector:
    def __init__(self):
        # Use defaults from module
        self.model_path = config.get(
            "vision.detection.yolo_model_path",
            default=get_vision_default("detection.yolo_model_path")
        )

        self.confidence = config.get(
            "vision.detection.yolo_confidence",
            default=get_vision_default("detection.yolo_confidence")
        )

        # Alternative: get entire section
        detection_config = config.get_section("detection")
        self.min_ball_radius = detection_config.get(
            "min_ball_radius",
            get_vision_default("detection.min_ball_radius")
        )
```

---

## 9. Implementation Checklist

### Immediate (No Breaking Changes)

- [ ] Add `threading.RLock` to Config class
- [ ] Add `get_section()` helper method
- [ ] Create `backend/vision/config_defaults.py`
- [ ] Create `backend/core/config_defaults.py`
- [ ] Create `backend/api/config_defaults.py`
- [ ] Add basic ConfigValidator class

### Short Term (Gradual Migration)

- [ ] Add validation rules for critical settings
- [ ] Update vision module to use `get_vision_default()`
- [ ] Update core module to use `get_core_default()`
- [ ] Update API module to use `get_api_default()`
- [ ] Add unit tests for validation

### Long Term (Optional Enhancements)

- [ ] Add background persistence (if needed)
- [ ] Add config change callbacks (if needed)
- [ ] Add profile support (if needed)
- [ ] Pydantic models for config sections (if desired)

---

## 10. Conclusion

The V2 configuration system demonstrates excellent software engineering practices:
- **Separation of concerns** (defaults near usage)
- **Self-documenting code** (Python docstrings)
- **Type safety** (generics, type hints)
- **Thread safety** (proper locking)
- **Validation** (early error detection)
- **No hardcoded values** (strict policy)

However, the current system has advantages too:
- **Simplicity** (easy to understand)
- **Concise API** (dot-notation)
- **Single source of truth** (one JSON file)

**Best path forward:** Adopt V2's philosophy and patterns while maintaining the current system's simplicity. Create module-specific default files, add thread safety and validation, but keep the intuitive dot-notation API.

This provides the best of both worlds: professional engineering practices without sacrificing developer experience.
