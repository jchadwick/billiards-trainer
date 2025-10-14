# Configuration System

## Overview

The visualizer uses a JSON-based configuration system with validation and user overrides.

## Files

- **default.json** - Default configuration values (do not modify)
- **user.json** - User overrides (optional, create this to customize settings)

## Usage

### Basic Usage

```lua
local Config = require('core.config')

-- Get the global config instance
local config = Config.get_instance()

-- Get values
local width = config:get('display.width')
local wsUrl = config:get('network.websocket_url')

-- Get nested values
local debugEnabled = config:get('debug_hud.enabled')
local connectionSection = config:get('debug_hud.sections.connection')

-- Set values (with validation)
local ok, err = config:set('display.width', 1920)
if not ok then
    print("Error: " .. err)
end

-- Get entire config
local allConfig = config:get_all()

-- Reset to defaults
config:reset()

-- Save current config as user overrides
config:save_user_config()
```

## Configuration Structure

### display
- `width` (number): Display width in pixels
- `height` (number): Display height in pixels
- `fullscreen` (boolean): Fullscreen mode
- `vsync` (boolean): Vertical sync

### network
- `websocket_url` (string): WebSocket server URL
- `auto_connect` (boolean): Auto-connect on startup
- `reconnect_enabled` (boolean): Enable reconnection
- `reconnect_delay` (number): Initial reconnect delay (ms)
- `max_reconnect_delay` (number): Maximum reconnect delay (ms)
- `heartbeat_interval` (number): Heartbeat interval (ms)

### calibration
- `profile` (string): Calibration profile name
- `auto_load` (boolean): Auto-load calibration on startup

### rendering
- `fps_target` (number): Target FPS
- `line_thickness` (number): Line thickness for overlays
- `anti_aliasing` (boolean): Enable anti-aliasing
- `theme` (string): Theme name

### video_feed
- `enabled` (boolean): Enable video feed display
- `opacity` (number): Video feed opacity (0.0-1.0)
- `layer` (string): Video feed layer (background/foreground)
- `subscribe_on_start` (boolean): Subscribe to video feed on start
- `quality` (number): JPEG quality (0-100)
- `fps` (number): Video feed FPS

### debug_hud
- `enabled` (boolean): Enable debug HUD
- `position` (string): HUD position (top_left, top_right, etc.)
- `opacity` (number): HUD opacity (0.0-1.0)
- `font_size` (number): Font size
- `color` (array): RGB color [r, g, b]
- `background` (array): RGBA background [r, g, b, a]
- `sections` (object): Enable/disable HUD sections
  - `connection` (boolean)
  - `balls` (boolean)
  - `cue` (boolean)
  - `table` (boolean)
  - `performance` (boolean)
- `layout` (string): HUD layout style
- `update_rate` (number): Update rate in Hz

### colors
- `table_felt` (array): RGB color for table felt [r, g, b]
- `auto_adapt` (boolean): Auto-adapt colors from video feed
- `trajectory_primary` (array|null): Primary trajectory color
- `trajectory_secondary` (array|null): Secondary trajectory color

## User Overrides

To customize settings without modifying default.json:

1. Create `config/user.json`
2. Add only the settings you want to override:

```json
{
  "display": {
    "width": 1920,
    "height": 1080,
    "fullscreen": true
  },
  "debug_hud": {
    "enabled": false
  }
}
```

3. User settings will be merged with defaults
4. Invalid settings will be ignored with a warning

## Type Validation

All configuration values are validated against their expected types:
- `number`: Numeric values
- `string`: Text values
- `boolean`: true/false
- `table`: Arrays or objects
- `table_or_nil`: Optional tables (can be null)

Setting a value with the wrong type will fail with an error message.

## Testing

A test script is provided to verify the configuration system:

```bash
cd frontend/visualizer
lua test_config.lua
```

Or test within LOVE2D:

```bash
cd frontend/visualizer
lua test_config_love.lua
```
