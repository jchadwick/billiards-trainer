# Configuration Guide

Complete guide to creating and managing configuration files for the Billiards Trainer application.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Configuration System Overview](#configuration-system-overview)
3. [Configuration Files](#configuration-files)
4. [Configuration Structure](#configuration-structure)
5. [Common Configuration Scenarios](#common-configuration-scenarios)
6. [Environment Variables](#environment-variables)
7. [Configuration Profiles](#configuration-profiles)
8. [Hot Reload](#hot-reload)
9. [Validation and Troubleshooting](#validation-and-troubleshooting)

## Quick Start

### Creating a Local Configuration

The easiest way to customize your configuration is to create a `config/local.json` file:

```bash
# Create a local config file
cat > config/local.json << 'EOF'
{
  "comment": "Local development configuration",
  "vision": {
    "camera": {
      "device_id": 0,
      "fps": 30
    }
  },
  "system": {
    "debug": true
  }
}
EOF

# Run the application
make run
```

The application will automatically load `config/local.json` and merge it with `config/default.json`.

## Configuration System Overview

### How Configuration Loading Works

The configuration system loads settings from multiple sources in this order (later sources override earlier ones):

1. **Default configuration** (`config/default.json`) - Base settings
2. **Local configuration** (`config/local.json`) - Your custom overrides
3. **Environment variables** - Runtime overrides (prefixed with `BILLIARDS_`)
4. **Command-line arguments** - Highest priority overrides

```
┌─────────────────┐
│ default.json    │  <- Base configuration
└────────┬────────┘
         │ (merged with)
┌────────▼────────┐
│ local.json      │  <- Local overrides
└────────┬────────┘
         │ (merged with)
┌────────▼────────┐
│ Environment     │  <- ENV vars
└────────┬────────┘
         │ (merged with)
┌────────▼────────┐
│ CLI Arguments   │  <- Highest priority
└─────────────────┘
```

### Configuration Precedence

From lowest to highest priority:
- `default.json` (lowest)
- `local.json`
- Environment variables
- CLI arguments (highest)

## Configuration Files

### File Locations

All configuration files are located in the `config/` directory:

```
config/
├── default.json                      # Base configuration (DO NOT EDIT)
├── local.json                        # Your local overrides (CREATE THIS)
├── vision_background_example.json    # Example: Background subtraction
├── vision_yolo_example.json          # Example: YOLO detection
└── vision_tpu_example.json           # Example: TPU acceleration
```

### File Formats

The system supports multiple formats:
- **JSON** (`.json`) - Default, recommended
- **YAML** (`.yaml`, `.yml`) - Alternative format
- **INI** (`.ini`) - Legacy format

### Which Files to Edit

- **DO NOT EDIT**: `config/default.json` - This is the base configuration
- **DO EDIT**: `config/local.json` - Create this file for your customizations
- **REFERENCE**: `config/*_example.json` - Use these as templates

## Configuration Structure

### Full Configuration Schema

```json
{
  "metadata": {
    "version": "1.0.0",
    "application": "billiards-trainer",
    "profile": "default",
    "description": "Configuration description"
  },
  "system": {
    "debug": false,
    "environment": "production"
  },
  "vision": {
    "camera": { ... },
    "detection": { ... },
    "processing": { ... }
  },
  "api": {
    "network": { ... }
  }
}
```

### Vision Configuration

#### Camera Settings

Configure camera/video capture:

```json
{
  "vision": {
    "camera": {
      "device_id": 0,              // Camera index or video file path
      "backend": "auto",            // "auto", "v4l2", "dshow", "gstreamer", "opencv", "kinect2"
      "resolution": [1920, 1080],   // [width, height]
      "fps": 30,                    // Frame rate
      "exposure_mode": "auto",      // "auto" or "manual"
      "exposure_value": null,       // Manual exposure (0.0-1.0)
      "gain": 1.0,                  // Camera gain
      "buffer_size": 1,             // Frame buffer size
      "auto_reconnect": true,       // Auto-reconnect on failure
      "reconnect_delay": 1.0,       // Seconds between reconnect attempts
      "max_reconnect_attempts": 5   // Maximum reconnection attempts
    }
  }
}
```

#### Video File Settings

Configure video file playback:

```json
{
  "vision": {
    "camera": {
      "device_id": "path/to/video.mp4",  // Path to video file
      "loop_video": true,                 // Loop video when it ends
      "video_start_frame": 0,             // Starting frame number
      "video_end_frame": null,            // Ending frame (null = end of video)
      "fps": 30                           // Playback frame rate
    }
  }
}
```

#### Detection Settings

Configure object detection:

```json
{
  "vision": {
    "detection": {
      "detection_backend": "opencv",      // "opencv", "yolo", "hybrid"
      "yolo_model_path": "models/yolov8n-pool.onnx",
      "yolo_confidence": 0.4,             // YOLO confidence threshold
      "yolo_nms_threshold": 0.45,         // Non-maximum suppression threshold
      "yolo_device": "cpu",               // "cpu", "cuda", "tpu"
      "use_opencv_validation": true,      // Validate YOLO results with OpenCV
      "fallback_to_opencv": true,         // Fallback to OpenCV if YOLO fails

      "table_edge_threshold": 0.7,        // Table edge detection sensitivity
      "min_table_area": 0.3,              // Minimum table area (fraction of frame)

      "min_ball_radius": 10,              // Minimum ball radius (pixels)
      "max_ball_radius": 40,              // Maximum ball radius (pixels)
      "ball_detection_method": "hough",   // "hough", "yolo", "contour"
      "ball_sensitivity": 0.8,            // Ball detection sensitivity

      "cue_detection_enabled": true,      // Enable cue stick detection
      "min_cue_length": 100,              // Minimum cue length (pixels)
      "cue_line_threshold": 0.6           // Cue line detection threshold
    }
  }
}
```

#### Processing Settings

Configure image processing:

```json
{
  "vision": {
    "processing": {
      "use_gpu": false,                   // Enable GPU acceleration
      "enable_preprocessing": true,       // Enable image preprocessing
      "blur_kernel_size": 5,              // Gaussian blur kernel size
      "morphology_kernel_size": 3,        // Morphological operation kernel size
      "enable_tracking": true,            // Enable object tracking
      "tracking_max_distance": 50,        // Maximum tracking distance (pixels)
      "frame_skip": 0                     // Skip N frames (0 = no skipping)
    }
  }
}
```

#### Debug Settings

Configure debugging output:

```json
{
  "vision": {
    "debug": true,                        // Enable debug mode
    "save_debug_images": true,            // Save debug images to disk
    "debug_output_path": "/tmp/vision_debug",  // Debug output directory
    "calibration_auto_save": true         // Auto-save calibration data
  }
}
```

### API Configuration

Configure the API server:

```json
{
  "api": {
    "network": {
      "host": "0.0.0.0",                  // Listen address
      "port": 8000                        // Listen port
    },
    "server": {
      "reload": false,                    // Enable hot reload
      "log_level": "info"                 // "debug", "info", "warning", "error"
    }
  }
}
```

## Common Configuration Scenarios

### 1. Using a Video File

Create `config/local.json`:

```json
{
  "comment": "Local config using demo video",
  "vision": {
    "camera": {
      "device_id": "demo2.mp4",
      "loop_video": true,
      "fps": 30
    }
  },
  "system": {
    "debug": true
  }
}
```

### 2. Using a Specific Camera

For USB camera at `/dev/video1`:

```json
{
  "vision": {
    "camera": {
      "device_id": 1,
      "backend": "v4l2",
      "resolution": [1920, 1080],
      "fps": 30
    }
  }
}
```

### 3. Using Kinect v2

```json
{
  "vision": {
    "camera": {
      "device_id": 0,
      "backend": "kinect2",
      "kinect2": {
        "enable_color": true,
        "enable_depth": true,
        "enable_infrared": false,
        "min_depth": 500,
        "max_depth": 4000,
        "depth_smoothing": true
      }
    }
  }
}
```

### 4. YOLO Detection with GPU

```json
{
  "vision": {
    "detection": {
      "detection_backend": "yolo",
      "yolo_device": "cuda",
      "yolo_model_path": "models/yolov8n-pool.onnx",
      "yolo_confidence": 0.4
    },
    "processing": {
      "use_gpu": true
    }
  }
}
```

### 5. Development Mode with Debug Output

```json
{
  "system": {
    "debug": true,
    "environment": "development"
  },
  "vision": {
    "debug": true,
    "save_debug_images": true,
    "debug_output_path": "./debug_images"
  },
  "api": {
    "server": {
      "reload": true,
      "log_level": "debug"
    }
  }
}
```

### 6. Background Subtraction

```json
{
  "vision": {
    "camera": {
      "device_id": 0
    },
    "detection": {
      "use_background_subtraction": true,
      "background_image_path": "config/table_background.png",
      "background_threshold": 30
    }
  }
}
```

### 7. Processing Video File Segment

Process only frames 100-500 of a video:

```json
{
  "vision": {
    "camera": {
      "device_id": "recording.mp4",
      "video_start_frame": 100,
      "video_end_frame": 500,
      "loop_video": false
    }
  }
}
```

## Environment Variables

Override any configuration value using environment variables with the `BILLIARDS_` prefix:

### Syntax

Convert dot-notation config paths to uppercase with underscores:

```bash
# config: vision.camera.device_id
export BILLIARDS_VISION_CAMERA_DEVICE_ID=1

# config: vision.camera.fps
export BILLIARDS_VISION_CAMERA_FPS=60

# config: system.debug
export BILLIARDS_SYSTEM_DEBUG=true

# config: api.network.port
export BILLIARDS_API_NETWORK_PORT=9000
```

### Common Environment Variables

```bash
# API server port (alternative: API_PORT)
export BILLIARDS_API_NETWORK_PORT=8000
export API_PORT=8000  # Also supported

# Debug mode
export BILLIARDS_SYSTEM_DEBUG=true

# Environment type
export ENVIRONMENT=development  # or production

# Camera device
export BILLIARDS_VISION_CAMERA_DEVICE_ID=0

# Video file
export BILLIARDS_VISION_CAMERA_DEVICE_ID="demo.mp4"
```

### Using .env Files

Create a `.env` file in the project root:

```bash
# .env
ENVIRONMENT=development
API_PORT=8000
BILLIARDS_SYSTEM_DEBUG=true
BILLIARDS_VISION_CAMERA_DEVICE_ID=0
```

The application will automatically load this file.

## Configuration Profiles

### What are Profiles?

Profiles are named configuration sets for different scenarios (e.g., "development", "production", "testing").

### Using Profiles

#### Via Configuration Files

Create profile-specific files:

```json
// config/profile-development.json
{
  "metadata": {
    "profile": "development"
  },
  "system": {
    "debug": true
  }
}

// config/profile-production.json
{
  "metadata": {
    "profile": "production"
  },
  "system": {
    "debug": false
  }
}
```

#### Via API

Switch profiles at runtime:

```bash
# List available profiles
curl http://localhost:8000/api/config/profiles

# Switch to a profile
curl -X POST http://localhost:8000/api/config/profiles/development/activate

# Get current profile
curl http://localhost:8000/api/config/profiles/current
```

### Profile Inheritance

Profiles inherit from base configuration and can override specific settings:

```
default.json  →  profile-dev.json  →  local.json
   (base)         (profile)           (overrides)
```

## Hot Reload

### What is Hot Reload?

Hot reload automatically applies configuration changes without restarting the application.

### How It Works

1. The system watches `config/default.json` and `config/local.json`
2. When a file changes, it automatically reloads
3. Affected modules are notified of changes
4. Changes are applied without downtime

### Enabling Hot Reload

Hot reload is enabled by default. To disable:

```json
{
  "system": {
    "hot_reload": false
  }
}
```

### Triggering Manual Reload

Via API:

```bash
# Reload all configuration
curl -X POST http://localhost:8000/api/config/reload

# Reload specific file
curl -X POST http://localhost:8000/api/config/reload?file=local.json
```

### Limitations

Some settings require a restart:
- Camera backend changes
- API network settings (host/port)
- GPU acceleration settings

## Validation and Troubleshooting

### Validating Configuration

Check if your configuration is valid:

```bash
# Via Make
make config-check

# Via Python
python -c "from backend.config.manager import ConfigurationModule; ConfigurationModule()"
```

### Common Validation Errors

#### Invalid Resolution

```
Error: Minimum resolution is 640x480
```

**Fix**: Ensure `vision.camera.resolution` is at least `[640, 480]`

#### Missing Video File

```
Error: Video file not found: demo.mp4
```

**Fix**: Use absolute path or ensure file exists:
```json
{
  "vision": {
    "camera": {
      "device_id": "/absolute/path/to/demo.mp4"
    }
  }
}
```

#### Invalid Frame Range

```
Error: video_start_frame must be less than video_end_frame
```

**Fix**: Ensure start frame is before end frame:
```json
{
  "vision": {
    "camera": {
      "video_start_frame": 0,
      "video_end_frame": 1000
    }
  }
}
```

### Viewing Current Configuration

Via API:

```bash
# Get all configuration
curl http://localhost:8000/api/config

# Get specific section
curl http://localhost:8000/api/config/vision

# Get specific value
curl http://localhost:8000/api/config/vision.camera.fps
```

Via Python:

```python
from backend.config.manager import ConfigurationModule

config = ConfigurationModule()
print(config.get("vision.camera.fps"))
print(config.get_all())
```

### Debug Configuration Loading

Enable debug logging to see configuration loading:

```bash
export BILLIARDS_SYSTEM_DEBUG=true
make run
```

Watch for log messages like:
```
INFO: Loading configuration from config/default.json
INFO: Loading configuration from config/local.json
INFO: Configuration loaded successfully
INFO: Hot reload enabled for 2 configuration files
```

### Configuration Backup and Restore

The system automatically backs up configurations:

```bash
# Backups are stored in
config/backups/

# List backups
ls -la config/backups/

# Restore from backup
cp config/backups/default.json.backup-TIMESTAMP config/default.json
```

### Testing Configuration Changes

Before committing configuration changes:

1. **Validate syntax**: Ensure JSON is valid
   ```bash
   python -m json.tool config/local.json
   ```

2. **Test loading**: Start the application
   ```bash
   make run
   ```

3. **Check logs**: Verify no errors
   ```bash
   make logs
   ```

4. **Run tests**: Ensure system works
   ```bash
   make test
   ```

## Best Practices

### DO:
- ✅ Use `config/local.json` for your customizations
- ✅ Keep configurations minimal (only override what you need)
- ✅ Use comments to document your changes
- ✅ Version control your `local.json` file
- ✅ Use example files as templates
- ✅ Validate configuration after changes

### DON'T:
- ❌ Edit `config/default.json` directly
- ❌ Commit sensitive data (passwords, API keys)
- ❌ Override all settings (use defaults when possible)
- ❌ Hardcode paths (use relative paths)
- ❌ Skip validation

## Additional Resources

- **Configuration Specification**: `backend/config/SPECS.md` - Detailed technical specs
- **Example Configurations**: `config/*_example.json` - Working examples
- **API Reference**: `docs/API_REFERENCE.md` - Configuration API endpoints
- **Architecture**: `docs/ARCHITECTURE.md` - System architecture overview

## Quick Reference

### Minimal Local Configuration

```json
{
  "vision": {
    "camera": {
      "device_id": 0
    }
  }
}
```

### Video File Configuration

```json
{
  "vision": {
    "camera": {
      "device_id": "demo.mp4",
      "loop_video": true
    }
  }
}
```

### Development Configuration

```json
{
  "system": {
    "debug": true,
    "environment": "development"
  },
  "vision": {
    "debug": true,
    "save_debug_images": true
  },
  "api": {
    "server": {
      "reload": true,
      "log_level": "debug"
    }
  }
}
```

### Production Configuration

```json
{
  "system": {
    "debug": false,
    "environment": "production"
  },
  "vision": {
    "debug": false,
    "save_debug_images": false
  },
  "api": {
    "server": {
      "reload": false,
      "log_level": "info"
    }
  }
}
```
