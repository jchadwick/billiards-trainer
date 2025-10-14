-- Billiards Visualizer Main Entry Point
-- Data-driven visualizer that receives all game state via WebSocket from backend
-- Displays video feed + AR overlays (no direct camera access)

local StateManager = require("core.state_manager")
local MessageHandler = require("core.message_handler")
-- Load configuration system
local Config = require("core.config")
local Network = require("modules.network.init")
local Colors = require("modules.colors.init")
local DebugHUD = require("modules.debug_hud.init")

-- Global references for easy access (will be set to instances after initialization)
_G.Network = Network

-- Legacy CONFIG for backward compatibility
-- Real configuration is in _G.Config (core/config.lua)
local CONFIG = {
    debug = function() return _G.Config and _G.Config:get("debug_hud.enabled") or true end,
    showFPS = function() return _G.Config and _G.Config:get("debug_hud.sections.performance") or true end,
    backgroundColor = function()
        local bg = _G.Config and _G.Config:get("display.background_color") or {0.1, 0.1, 0.1}
        return bg
    end,
    showHUD = function() return _G.Config and _G.Config:get("debug_hud.enabled") or true end,
    calibrationMode = false  -- Runtime toggle, not from config
}

-- Calculate contrasting text color
local TEXT_COLOR = {1, 1, 1, 0.9}  -- White text on dark background
local TEXT_COLOR_SUCCESS = {0, 1, 0, 0.9}  -- Bright green
local TEXT_COLOR_ERROR = {1, 0, 0, 0.9}  -- Red
local TEXT_COLOR_WARNING = {1, 1, 0, 0.9}  -- Yellow

-- State
local state = {
    stateManager = nil,
    messageHandler = nil,
    debugHUD = nil,
    lastError = nil,
    connectionStatus = {
        websocket = false,
        lastMessage = nil,
        messagesReceived = 0
    }
}

function love.load()
    print("Billiards Visualizer Starting...")
    print("This is a data-driven visualizer - all data comes from backend via WebSocket")

    -- Initialize configuration
    _G.Config = Config.get_instance()
    _G.Config:init()
    print("✓ Configuration loaded")

    -- Set window mode from config
    local width = _G.Config:get("display.width") or 1440
    local height = _G.Config:get("display.height") or 810
    love.window.setMode(width, height, {
        fullscreen = _G.Config:get("display.fullscreen") or false,
        vsync = _G.Config:get("display.vsync") or true
    })

    -- Set background color
    love.graphics.setBackgroundColor(CONFIG.backgroundColor())

    -- Initialize core systems
    local success, err = pcall(function()
        -- Load and initialize Renderer (global for modules to use)
        local RendererModule = require("core.renderer")
        _G.Renderer = RendererModule
        _G.Renderer:init()
        print("✓ Renderer initialized")

        -- Load Calibration module (global for modules to use)
        _G.Calibration = require("rendering.calibration")
        _G.Calibration:load()  -- Load saved calibration
        print("✓ Calibration loaded")

        -- Initialize Colors module
        print("Initializing colors module...")
        _G.Colors = Colors
        Colors:init()
        print("✓ Colors module initialized")

        -- Create state manager
        state.stateManager = StateManager.new()
        _G.StateManager = state.stateManager  -- Set global to instance
        print("✓ State Manager initialized")

        -- Create message handler
        state.messageHandler = MessageHandler.new(state.stateManager)
        _G.MessageHandler = state.messageHandler  -- Set global to instance
        print("✓ Message Handler initialized")

        -- Load trajectory module
        state.trajectoryModule = require("modules.trajectory.init")
        print("✓ Trajectory Module loaded")

        -- Wire trajectory callback to message handler
        state.messageHandler:setTrajectoryCallback(function(data)
            state.trajectoryModule:updateTrajectory(data)
        end)

        -- Initialize network module
        print("Initializing network module...")
        state.network = Network
        Network:init()
        print("✓ Network module initialized")

        -- Initialize debug HUD module
        print("Initializing debug HUD module...")
        state.debugHUD = DebugHUD
        DebugHUD:init()
        print("✓ Debug HUD module initialized")
    end)

    if not success then
        state.lastError = err
        print("Initialization error: " .. tostring(err))
    end

    print("Billiards Visualizer Ready")
    print("Press F1 to toggle HUD")
    print("Press ESC to quit")
end

function love.update(dt)
    -- Update connection status from network module
    if state.network then
        local status = state.network:getStatus()
        state.connectionStatus.websocket = (status.state == "CONNECTED")
        state.connectionStatus.messagesReceived = status.messages_received or 0
        if state.connectionStatus.messagesReceived > 0 then
            state.connectionStatus.lastMessage = love.timer.getTime()
        end
    else
        -- Fall back to old timeout-based detection
        local timeSinceUpdate = love.timer.getTime() - state.stateManager:getLastUpdateTime()
        if timeSinceUpdate > 2.0 then
            state.connectionStatus.websocket = false
        end
    end

    -- Update trajectory module
    if state.trajectoryModule then
        state.trajectoryModule:update(dt)
    end

    -- Update network module
    if state.network then
        state.network:update(dt)
    end

    -- Update debug HUD
    if state.debugHUD then
        state.debugHUD:update(dt)
    end
end

function love.draw()
    -- Wrap everything in pcall to catch errors
    local success, err = pcall(function()
        -- Clear screen
        love.graphics.clear(CONFIG.backgroundColor())

        -- Draw trajectory overlays
        if state.trajectoryModule then
            state.trajectoryModule:draw()
        end

        -- TODO: Draw video feed in Task Group 5 (Phase 4)
        -- TODO: Draw table overlays in Task Group 3 (Phase 2)
    end)

    if not success then
        state.lastError = err
        love.graphics.origin()  -- Reset graphics state after error
    end

    -- Draw calibration overlay (if in calibration mode)
    if CONFIG.calibrationMode and _G.Calibration then
        _G.Calibration:drawOverlay()
    end

    -- Draw debug HUD (uses new module)
    if state.debugHUD and state.debugHUD.config.enabled then
        state.debugHUD:draw()
    end
end

-- Legacy drawHUD function (replaced by DebugHUD module)
-- Kept for reference, but no longer used
function drawHUD()
    -- This function is now handled by the DebugHUD module
    -- See modules/debug_hud/init.lua
end

function love.keypressed(key)
    -- Toggle HUD
    if key == "f1" then
        if _G.Config then
            local current = _G.Config:get("debug_hud.enabled")
            _G.Config:set("debug_hud.enabled", not current)
            print("HUD " .. (not current and "enabled" or "disabled"))
        end
    end

    -- Toggle debug
    if key == "f2" then
        if _G.Config then
            local current = _G.Config:get("debug_hud.enabled")
            _G.Config:set("debug_hud.enabled", not current)
            print("Debug mode " .. (not current and "enabled" or "disabled"))
        end
    end

    -- Toggle calibration mode
    if key == "f3" then
        CONFIG.calibrationMode = not CONFIG.calibrationMode
        print("Calibration mode " .. (CONFIG.calibrationMode and "enabled" or "disabled"))
    end

    -- Calibration controls (only in calibration mode)
    if CONFIG.calibrationMode and _G.Calibration then
        if key == "up" then
            _G.Calibration:adjustCorner(0, -1)
        elseif key == "down" then
            _G.Calibration:adjustCorner(0, 1)
        elseif key == "left" then
            _G.Calibration:adjustCorner(-1, 0)
        elseif key == "right" then
            _G.Calibration:adjustCorner(1, 0)
        elseif key == "tab" then
            _G.Calibration:nextCorner()
            print("Calibration corner: " .. _G.Calibration.selectedCorner)
        elseif key == "s" then
            if _G.Calibration:save() then
                print("Calibration saved successfully")
            else
                print("Calibration save failed")
            end
        elseif key == "l" then
            if _G.Calibration:load() then
                print("Calibration loaded successfully")
            else
                print("Calibration load failed")
            end
        elseif key == "r" then
            _G.Calibration:reset()
            print("Calibration reset to defaults")
        end
    end

    -- Clear error
    if key == "e" and state.lastError then
        state.lastError = nil
        print("Error cleared")
    end

    -- Quit
    if key == "escape" then
        love.event.quit()
    end
end

function love.quit()
    print("Billiards Visualizer Shutting down...")

    -- Cleanup network module
    if state.network then
        state.network:disconnect()
    end
    print("Network connection closed")

    -- Cleanup debug HUD module
    if state.debugHUD then
        state.debugHUD:cleanup()
    end
    print("Debug HUD cleaned up")

    -- TODO: Cleanup modules in Task Group 3

    print("Goodbye!")
    return false -- Allow quit
end

function love.errorhandler(msg)
    print("Fatal error: " .. tostring(msg))
    state.lastError = msg
end
