# Billiards Projector Application (LÖVE2D)

A modular, extensible projector application for the Billiards Trainer system built with LÖVE2D.

## Features

- **Modular Architecture**: Drop-in module system for easy extension
- **UDP Networking**: Low-latency communication with backend (port 9999)
- **Hot Reload**: Modify modules without restarting (Ctrl+R)
- **Calibration System**: 4-point perspective transformation
- **Trajectory Visualization**: Real-time ball path rendering
- **60+ FPS Performance**: Optimized for real-time projection

## Installation

### Ubuntu/Debian

```bash
# Install LÖVE2D
sudo apt-get update
sudo apt-get install love

# Verify installation
love --version
```

### Running the Projector

```bash
cd frontend/projector
love .
```

### Creating Standalone Executable

```bash
# Create fused executable
cd frontend/projector
zip -r projector.love . -x "*.git*"
cat /usr/bin/love projector.love > projector
chmod +x projector

# Run standalone
./projector
```

## Usage

### Keyboard Controls

- **Ctrl+R**: Hot reload all modules
- **C**: Toggle calibration mode
- **P**: Pause/unpause
- **E**: Clear error messages
- **ESC**: Quit application

### Calibration Mode

When in calibration mode (press C):

- **Arrow Keys**: Move selected corner
- **Shift+Arrow**: Move corner faster (10px steps)
- **Tab**: Select next corner
- **R**: Reset to default calibration
- **Mouse Click**: Select corner by clicking
- **Mouse Drag**: Drag corner to new position
- **C**: Save and exit calibration mode

## Module System

### Creating a New Module

1. Create a new folder in `modules/`:
```
modules/my_module/
├── init.lua       # Module entry point
└── config.json    # Optional configuration
```

2. Implement the module interface in `init.lua`:
```lua
local MyModule = {
    name = "my_module",
    priority = 100,  -- Lower = drawn first
    enabled = true,

    init = function(self)
        -- Initialize module
    end,

    update = function(self, dt)
        -- Update logic (called every frame)
    end,

    draw = function(self)
        -- Render graphics
    end,

    onMessage = function(self, type, data)
        -- Handle network messages
    end,

    cleanup = function(self)
        -- Cleanup resources
    end
}

return MyModule
```

3. Module loads automatically on next startup or hot reload

### Module Priorities

- **0-99**: Background layers (table surface, grids)
- **100-199**: Main content (trajectories, balls)
- **200-299**: Effects and overlays
- **300+**: UI and debug information

## Network Protocol

### UDP Message Format (Port 9999)

```json
{
    "type": "trajectory|collision|state|config",
    "timestamp": 1234567890.123,
    "data": {
        // Type-specific payload
    }
}
```

### Message Types

#### Trajectory
```json
{
    "type": "trajectory",
    "data": {
        "paths": [
            {
                "points": [{"x": 0.1, "y": 0.2}, ...],
                "ballType": "cue",
                "confidence": 0.95
            }
        ],
        "collisions": [
            {"x": 0.5, "y": 0.5, "type": "ball"}
        ],
        "ghostBalls": [
            {"x": 0.3, "y": 0.4, "radius": 10, "number": 8}
        ]
    }
}
```

#### Collision
```json
{
    "type": "collision",
    "data": {
        "x": 0.5,
        "y": 0.5,
        "type": "ball|cushion|pocket"
    }
}
```

#### Aim Line
```json
{
    "type": "aim",
    "data": {
        "x1": 0.2, "y1": 0.3,
        "x2": 0.7, "y2": 0.6
    }
}
```

## Coordinate System

- All coordinates from backend are normalized (0-1 range)
- Calibration system transforms to screen coordinates
- (0,0) = top-left corner of table
- (1,1) = bottom-right corner of table

## Configuration Files

### Display Configuration (`config/display.json`)
```json
{
    "fullscreen": true,
    "display": 1,
    "vsync": true,
    "msaa": 4
}
```

### Calibration (`config/calibration.json`)
Automatically saved when exiting calibration mode.

## Troubleshooting

### Application won't start
- Check LÖVE2D is installed: `love --version`
- Verify you're in the correct directory
- Check for syntax errors in Lua files

### No trajectory display
- Check UDP port 9999 is not blocked
- Verify backend is sending messages
- Check network status in UI overlay (green = connected)

### Calibration issues
- Press 'R' in calibration mode to reset
- Ensure corners form a proper rectangle
- Save calibration with 'C' after adjusting

### Performance issues
- Disable vsync in conf.lua
- Reduce MSAA samples
- Check module update/draw performance

## Development

### Module Best Practices

1. **Keep modules isolated**: Don't directly access other modules
2. **Use message passing**: Communicate via onMessage()
3. **Handle errors gracefully**: Use pcall for risky operations
4. **Clean up resources**: Implement cleanup() properly
5. **Document configuration**: Provide clear config.json examples

### Testing a Module

```lua
-- In main.lua, temporarily add:
local testModule = require("modules.my_module.init")
testModule:init()
-- Test module methods
```

### Debug Mode

Enable debug output by editing main.lua:
```lua
local CONFIG = {
    debug = true,  -- Enable debug mode
    showFPS = true,
    ...
}
```

## Backend Integration

To send messages from the backend to the projector, add UDP broadcasting:

```python
import socket
import json

udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

message = {
    "type": "trajectory",
    "timestamp": time.time(),
    "data": {...}
}

udp_socket.sendto(
    json.dumps(message).encode(),
    ("projector_ip", 9999)
)
```

## License

Part of the Billiards Trainer system.
