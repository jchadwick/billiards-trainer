# Configuration System Integration Example

## Current State vs. New Config System

### Before (Hardcoded in main.lua)
```lua
local CONFIG = {
    debug = true,
    showFPS = true,
    backgroundColor = {0.1, 0.1, 0.1},
    showHUD = true,
    calibrationMode = false
}
```

### After (Using Config System)
```lua
local Config = require('core.config')
local config = Config.get_instance()

-- Get values with defaults from JSON config
local showFPS = config:get('debug_hud.enabled')
local backgroundColor = {0.1, 0.1, 0.1}  -- Can add to config if needed
local showHUD = config:get('debug_hud.enabled')
```

## Integration Steps

### Step 1: Add Config to main.lua

At the top of `main.lua`, add:
```lua
local Config = require("core.config")
```

Then in `love.load()`, initialize:
```lua
-- Initialize configuration
_G.Config = Config.get_instance()
print("✓ Configuration loaded")
```

### Step 2: Use Config Values

Replace hardcoded values:
```lua
-- Display settings
local width = _G.Config:get('display.width')
local height = _G.Config:get('display.height')
love.window.setMode(width, height, {
    fullscreen = _G.Config:get('display.fullscreen'),
    vsync = _G.Config:get('display.vsync')
})

-- Debug HUD settings
local showHUD = _G.Config:get('debug_hud.enabled')
local hudOpacity = _G.Config:get('debug_hud.opacity')
```

### Step 3: WebSocket Integration (Future)

When implementing WebSocket in Task Group 2:
```lua
local wsUrl = _G.Config:get('network.websocket_url')
local autoConnect = _G.Config:get('network.auto_connect')
local reconnectEnabled = _G.Config:get('network.reconnect_enabled')

if autoConnect then
    websocket = WebSocket.new(wsUrl)
end
```

### Step 4: Runtime Configuration Changes

Add key bindings to modify config at runtime:
```lua
function love.keypressed(key)
    if key == "f4" then
        -- Toggle debug HUD
        local current = _G.Config:get('debug_hud.enabled')
        _G.Config:set('debug_hud.enabled', not current)
        print("Debug HUD: " .. tostring(not current))
    end

    if key == "f5" then
        -- Save current config
        if _G.Config:save_user_config() then
            print("Configuration saved to user.json")
        end
    end

    if key == "f6" then
        -- Reset to defaults
        _G.Config:reset()
        print("Configuration reset to defaults")
    end
end
```

## Module Integration Examples

### Renderer Module

```lua
-- In core/renderer.lua
function Renderer:init()
    self.lineThickness = _G.Config:get('rendering.line_thickness')
    self.antiAliasing = _G.Config:get('rendering.anti_aliasing')
    self.theme = _G.Config:get('rendering.theme')

    if self.antiAliasing then
        love.graphics.setLineStyle("smooth")
    end
end
```

### WebSocket Module (Future)

```lua
-- In network/websocket.lua
function WebSocket.new()
    local self = setmetatable({}, WebSocket)

    -- Load config
    self.url = _G.Config:get('network.websocket_url')
    self.reconnectEnabled = _G.Config:get('network.reconnect_enabled')
    self.reconnectDelay = _G.Config:get('network.reconnect_delay')
    self.maxReconnectDelay = _G.Config:get('network.max_reconnect_delay')
    self.heartbeatInterval = _G.Config:get('network.heartbeat_interval')

    return self
end
```

### Calibration Module

```lua
-- In rendering/calibration.lua
function Calibration:load()
    local profile = _G.Config:get('calibration.profile')
    local autoLoad = _G.Config:get('calibration.auto_load')

    if autoLoad then
        self:loadProfile(profile)
    end
end
```

## User Customization Example

Users can create `config/user.json`:

```json
{
  "display": {
    "width": 1920,
    "height": 1080,
    "fullscreen": true
  },
  "network": {
    "websocket_url": "ws://192.168.1.100:8000/api/v1/game/state/ws"
  },
  "debug_hud": {
    "enabled": false,
    "position": "bottom_right",
    "sections": {
      "performance": true,
      "balls": false,
      "cue": false
    }
  },
  "rendering": {
    "fps_target": 144,
    "line_thickness": 5
  }
}
```

These settings will automatically override the defaults.

## Benefits

1. **Centralized Configuration**: All settings in one place
2. **Type Safety**: Automatic validation prevents invalid values
3. **User Overrides**: Easy customization without modifying code
4. **Hot Reload**: Change settings at runtime
5. **Persistence**: Save preferences between sessions
6. **Documentation**: Self-documenting through JSON structure
7. **Defaults**: Always have working fallback values

## Migration Strategy

1. Keep existing CONFIG table for now
2. Add config system alongside
3. Gradually migrate settings one section at a time
4. Remove old CONFIG once migration is complete
5. Update documentation as features are migrated
