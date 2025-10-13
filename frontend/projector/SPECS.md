# Projector Application Specification

## Purpose

The Projector application is a **native wrapper for fullscreen projector deployment**. It is a thin LÖVE2D-based layer that launches the core visualizer (located in `frontend/visualizer`) in fullscreen mode on a dedicated display/projector device. The projector wrapper handles display management, system integration, and hardware-specific optimizations for production deployment on dedicated projector hardware.

**For core visualization functionality** (rendering trajectories, AR overlays, calibration, network communication, etc.), see `frontend/visualizer/SPECS.md`.

## Wrapper Architecture

The projector wrapper:
1. Loads and initializes the core visualizer module from `frontend/visualizer`
2. Applies projector-specific configuration (display selection, fullscreen mode)
3. Manages system integration (systemd service, logging to system journal)
4. Handles hardware-specific optimizations for the target platform
5. Provides a production deployment environment for the visualizer
6. **Does NOT display video feed** (video feed is only for web wrapper)

The application operates with no user interaction, expecting all commands to arrive over the network (originating from the API, triggered by the user's interaction with the web interface).

Minimal keyboard controls are provided but kept strictly limited to calibration, debugging and maintenance (e.g. toggling debug HUD, exiting fullscreen, etc.), not required for normal usage.

## Deployment

The projector application always runs as a native LÖVE2D application:
- Full hardware acceleration and GPU access
- Direct display management
- System-level integration (systemd services)
- Optimized for dedicated projector hardware

## Functional Requirements

### 1. Display Management

- **FR-PROJ-001**: Initialize projector display output (fullscreen)
- **FR-PROJ-002**: Detect and configure available display devices
- **FR-PROJ-003**: Support multiple display resolutions (720p, 1080p, 4K)
- **FR-PROJ-004**: Handle display hotplug events
- **FR-PROJ-005**: Support V-sync configuration

### 2. Visualizer Integration

- **FR-PROJ-010**: Load and initialize core visualizer module
- **FR-PROJ-011**: Pass configuration to visualizer (display settings, network config, etc.)
- **FR-PROJ-012**: Handle visualizer lifecycle (init, update, draw, cleanup)
- **FR-PROJ-013**: Forward keyboard/input events to visualizer when appropriate
- **FR-PROJ-014**: Manage visualizer error handling and recovery

### 3. System Integration

- **FR-PROJ-020**: Run as systemd service on Linux
- **FR-PROJ-021**: Log to system journal (journald)
- **FR-PROJ-022**: Handle graceful shutdown signals (SIGTERM, SIGINT)
- **FR-PROJ-023**: Support automatic restart on failure
- **FR-PROJ-024**: Monitor system resources (CPU, GPU, memory)

### 4. Hardware Optimization

- **FR-PROJ-030**: Leverage GPU hardware acceleration
- **FR-PROJ-031**: Optimize for Intel integrated graphics (VAAPI/iHD)
- **FR-PROJ-032**: Support multiple GPU configurations
- **FR-PROJ-033**: Handle projector-specific color calibration
- **FR-PROJ-034**: Optimize thermal management for long-running operation

### 5. Configuration Management

- **FR-PROJ-040**: Load configuration from file system
- **FR-PROJ-041**: Support environment variable overrides
- **FR-PROJ-042**: Validate configuration on startup
- **FR-PROJ-043**: Hot-reload configuration when possible
- **FR-PROJ-044**: Provide sensible defaults for all settings

## Non-Functional Requirements

### Performance Requirements

- **NFR-PROJ-001**: Startup time < 5 seconds
- **NFR-PROJ-002**: Memory footprint < 200MB (excluding visualizer)
- **NFR-PROJ-003**: CPU overhead < 5% (wrapper only, not including visualizer)
- **NFR-PROJ-004**: Stable operation for 24+ hours without restart

### Reliability Requirements

- **NFR-PROJ-010**: No wrapper crashes during operation
- **NFR-PROJ-011**: Automatic recovery from visualizer failures
- **NFR-PROJ-012**: Graceful handling of display disconnection
- **NFR-PROJ-013**: Maintain operation through system updates

### Compatibility Requirements

- **NFR-PROJ-020**: Support Ubuntu Server 20.04+
- **NFR-PROJ-021**: Support LÖVE2D 11.4+
- **NFR-PROJ-022**: Work with standard X11 display server
- **NFR-PROJ-023**: Compatible with systemd-based init systems

## Configuration Schema

The projector wrapper uses a JSON configuration file:

```json
{
  "display": {
    "fullscreen": true,
    "display_index": 0,
    "resolution": "1920x1080",
    "vsync": true
  },
  "system": {
    "log_level": "info",
    "log_to_journal": true,
    "auto_restart": true,
    "resource_monitoring": true
  },
  "visualizer": {
    "module_path": "../visualizer",
    "config_file": "../visualizer/config.json"
  },
  "hardware": {
    "gpu_optimization": "auto",
    "thermal_throttle": false
  }
}
```

## Implementation Structure

```
projector/                      # Native Wrapper Application
├── main.lua                   # Wrapper entry point
├── conf.lua                   # LÖVE configuration (fullscreen)
├── wrapper/
│   ├── init.lua              # Wrapper initialization
│   ├── display.lua           # Display management
│   ├── system.lua            # System integration
│   └── config.lua            # Configuration management
├── config/
│   └── default.json          # Default wrapper configuration
├── systemd/
│   └── projector.service     # systemd service definition
└── deploy.sh                 # Deployment script
```

The wrapper loads the visualizer module from `../visualizer/` and delegates all visualization functionality to it.

## Deployment Process

### Installation
```bash
# Install LÖVE2D
sudo apt-get install love

# Deploy wrapper and visualizer
./deploy.sh install

# Configure systemd service
sudo systemctl enable projector.service
sudo systemctl start projector.service
```

### Management
```bash
# Check status
sudo systemctl status projector.service

# View logs
sudo journalctl -u projector.service -f

# Restart
sudo systemctl restart projector.service
```

## Testing Requirements

### Wrapper Testing
- Test display initialization and fullscreen mode
- Verify systemd service integration
- Test configuration loading and validation
- Verify visualizer module loading
- Test graceful shutdown handling
- Validate resource monitoring

### Integration Testing
- Test wrapper + visualizer integration
- Verify configuration pass-through
- Test error propagation and recovery
- Validate long-running stability

### Hardware Testing
- Test on target projector hardware
- Verify GPU acceleration
- Test thermal stability over extended operation
- Validate multiple display configurations

## Success Criteria

1. **Deployment Success**
   - Clean installation on Ubuntu Server
   - Automatic startup via systemd
   - Stable operation for 24+ hours
   - Clean shutdown and restart

2. **Integration Success**
   - Visualizer loads and runs correctly
   - Configuration properly passed to visualizer
   - Error handling works as expected
   - Resource usage within limits

3. **Performance Success**
   - Fast startup (< 5 seconds)
   - Minimal wrapper overhead
   - Stable long-term operation
   - Proper resource cleanup
