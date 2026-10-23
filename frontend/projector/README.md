# Billiards Projector Application (LÖVE2D)

A modular, extensible projector application for the Billiards Trainer system built with LÖVE2D.

## Features

- **Modular Architecture**: Drop-in module system for easy extension
- **UDP Receiver**: Local custom-message input on 127.0.0.1:9999; production backend publishing is not wired yet
- **Hot Reload**: Modify modules without restarting (Ctrl+R)
- **Calibration System**: 4-point perspective transformation
- **Trajectory Visualization**: Real-time ball path rendering

## Installation

### Ubuntu/Debian

Install LÖVE2D from your distribution's configured package repository using
the local package manager, then verify it:

```bash
love --version
```

### Running the Projector

```bash
cd frontend/projector
love .
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

The LÖVE app binds to `127.0.0.1:9999` and receives this custom JSON protocol.
It currently has no production backend UDP publisher or WebSocket client; the
examples below describe the receiver contract only. See
[Backend Integration](#backend-integration) for the integration gap.

```json
{
    "type": "trajectory|collision|aim|state|motion|event|frame|config",
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
        "coordinate_space": "normalized_table",
        "coordinate_scale": [1, 1],
        "table_boundary": [
            {"x": 0, "y": 0}, {"x": 1, "y": 0},
            {"x": 1, "y": 1}, {"x": 0, "y": 1}
        ],
        "bounds": {"x": 0, "y": 0, "width": 1, "height": 1},
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
            {
                "x": 0.3, "y": 0.4,
                "radius": 10,
                "radius_space": "projector_pixels",
                "radius_scale": [1, 1],
                "number": 8
            }
        ]
    }
}
```

#### Collision
```json
{
    "type": "collision",
    "data": {
        "coordinate_space": "normalized_table",
        "coordinate_scale": [1, 1],
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
        "coordinate_space": "normalized_table",
        "coordinate_scale": [1, 1],
        "x1": 0.2, "y1": 0.3,
        "x2": 0.7, "y2": 0.6
    }
}
```

#### State geometry
State messages that contain balls, cue points, or table geometry use the same
`normalized_table` metadata at the `data` level. Ball centers, cue points,
table corners, pockets, nested positions, and playing-area bounds are validated
before the debug HUD sees them. Every nested x/y point inherits the message's
table metadata; a nested `coordinate_space` override must also say
`normalized_table` and carry identity `coordinate_scale`. Ball radii, when
supplied, use explicit `projector_pixels` radius fields as in the trajectory
example. Other width/height extents must declare `extent_space` and identity
`extent_scale`; bounds use normalized table extents directly.

Physical positions and `real_world_dimensions` remain separate data: a physical
point declares `unit: "meters"` (optionally `coordinate_space: "world_meters"`)
and physical dimensions declare `unit: "meters"`. These
records are validated as finite SI values and are never sent through table or
projector transforms. A physical radius uses `radius_unit: "meters"` and must
not include projector-pixel scale fields. Unsupported or conflicting unit/space
overrides reject the message.

#### Motion
Motion data requires normalized table metadata for its position and a separately
tagged table-relative velocity:

```json
{
  "type": "motion",
  "data": {
    "coordinate_space": "normalized_table",
    "coordinate_scale": [1, 1],
    "position": {"x": 0.4, "y": 0.6},
    "velocity": {
      "x": -0.2,
      "y": 0.1,
      "velocity_space": "normalized_table_per_second",
      "velocity_scale": [1, 1]
    }
  }
}
```

The velocity components are finite vectors and are not constrained to the
position range [0, 1]. Missing velocity units or scale metadata are rejected.
Events with nested table geometry follow the same normalized-table contract;
non-spatial telemetry does not need coordinate metadata.

#### Calibration and frame extents
`show_calibration_chessboard` uses `coordinate_space: "projector_pixels"` and
`coordinate_scale: [1, 1]`; `squareSize` and optional `offsetX`/`offsetY` are
pixel extents/positions. A frame carrying width and height declares
`dimensions_space: "image_pixels"` and `dimensions_scale: [1, 1]`. The hide
command and frames with telemetry only contain no coordinates and need no such
metadata.

## Coordinate System

- Coordinate-bearing `trajectory`, `collision`, `aim`, `state`, `motion`, and `event` payloads must declare `coordinate_space: "normalized_table"` and `coordinate_scale: [1, 1]` at the owning message boundary.
- Every point, including the optional four-point `table_boundary`, uses finite x/y values in the inclusive range 0–1. Optional `bounds` uses normalized x/y plus positive width/height and must fit inside the unit table rectangle. Nested points and bounds cannot use a singular `scale` field; they inherit the message-level `coordinate_scale`.
- (0,0) is the table's top-left and (1,1) its bottom-right. The projector calibration maps this normalized table space to the display.
- Ghost-ball `radius` is a positive extent in projector pixels and must declare `radius_space: "projector_pixels"` and `radius_scale: [1, 1]`. It is drawn directly in display pixels; it is not a table-space distance.
- The UDP receiver rejects missing, non-identity, non-finite, or out-of-range coordinate metadata before forwarding a message to modules. Calibration files also store four `normalized_projector` corners with identity scale.

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
- The standalone application has no production message publisher, so it does
  not display live product trajectories.
- The backend WebSocket and projector UDP payloads use different schemas; an
  explicit adapter is required before they can be connected.

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

The production backend does not publish this protocol. It sends state over the
application WebSocket, with a different trajectory schema. No local fake
sender is retained because that would demonstrate an isolated test path rather
than production data delivery. An adapter needs its own protocol decision and
end-to-end test before this receiver can be considered an application client.

## License

Part of the Billiards Trainer system.
