-- Billiards Visualizer Main Entry Point
-- Data-driven visualizer that receives all game state via WebSocket from backend
-- Displays video feed + AR overlays (no direct camera access)

local StateManager = require("core.state_manager")
local MessageHandler = require("core.message_handler")

-- Global references for easy access
_G.StateManager = StateManager
_G.MessageHandler = MessageHandler

-- Configuration
local CONFIG = {
    debug = true,
    showFPS = true,
    backgroundColor = {0.1, 0.1, 0.1},  -- Dark background for visualizer
    showHUD = true
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

    -- Set background color
    love.graphics.setBackgroundColor(CONFIG.backgroundColor)

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

        -- Create state manager
        state.stateManager = StateManager.new()
        print("✓ State Manager initialized")

        -- Create message handler
        state.messageHandler = MessageHandler.new(state.stateManager)
        print("✓ Message Handler initialized")

        -- Load trajectory module
        state.trajectoryModule = require("modules.trajectory.init")
        print("✓ Trajectory Module loaded")

        -- Wire trajectory callback to message handler
        state.messageHandler:setTrajectoryCallback(function(data)
            state.trajectoryModule:updateTrajectory(data)
        end)

        -- TODO: Initialize WebSocket client in Task Group 2 (Phase 1)
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
    -- Update state manager timing
    if state.stateManager then
        local timeSinceUpdate = love.timer.getTime() - state.stateManager:getLastUpdateTime()
        if timeSinceUpdate > 2.0 then
            -- No updates for 2 seconds - mark as disconnected
            state.connectionStatus.websocket = false
        end
    end

    -- Update trajectory module
    if state.trajectoryModule then
        state.trajectoryModule:update(dt)
    end

    -- TODO: Update WebSocket client in Task Group 2 (Phase 1)
end

function love.draw()
    -- Wrap everything in pcall to catch errors
    local success, err = pcall(function()
        -- Clear screen
        love.graphics.clear(CONFIG.backgroundColor)

        -- Draw trajectory overlays
        if state.trajectoryModule then
            state.trajectoryModule:draw()
        end

        -- TODO: Draw video feed in Task Group 5 (Phase 4)
        -- TODO: Draw table overlays in Task Group 3 (Phase 2)
    end)

    if not success then
        state.lastError = err
        love.graphics.push()
        love.graphics.origin()
    end

    -- Draw HUD overlay
    if CONFIG.showHUD then
        drawHUD()
    end

    love.graphics.pop()
end

function drawHUD()
    love.graphics.push()
    love.graphics.origin()

    local y = 10
    local lineHeight = 20

    -- Show FPS
    if CONFIG.showFPS then
        love.graphics.setColor(TEXT_COLOR)
        love.graphics.print(string.format("FPS: %d", love.timer.getFPS()), 10, y)
        y = y + lineHeight
    end

    -- Show errors
    if state.lastError then
        love.graphics.setColor(TEXT_COLOR_ERROR)
        love.graphics.print("ERROR: " .. tostring(state.lastError), 10, y)
        y = y + lineHeight
    end

    -- Show connection status
    if state.connectionStatus.websocket then
        love.graphics.setColor(TEXT_COLOR_SUCCESS)
        love.graphics.print("WebSocket: Connected", 10, y)
    else
        love.graphics.setColor(TEXT_COLOR_ERROR)
        love.graphics.print("WebSocket: Disconnected", 10, y)
    end
    y = y + lineHeight

    -- Show message stats
    love.graphics.setColor(TEXT_COLOR)
    love.graphics.print(string.format("Messages received: %d", state.connectionStatus.messagesReceived), 10, y)
    y = y + lineHeight

    if state.connectionStatus.lastMessage then
        local timeSince = love.timer.getTime() - state.connectionStatus.lastMessage
        love.graphics.setColor(TEXT_COLOR)
        love.graphics.print(string.format("Last message: %.1fs ago", timeSince), 10, y)
        y = y + lineHeight
    end

    -- Show state manager info
    if state.stateManager then
        love.graphics.setColor(TEXT_COLOR)
        local ballCount = 0
        for _ in pairs(state.stateManager:getBalls()) do
            ballCount = ballCount + 1
        end
        love.graphics.print(string.format("Balls tracked: %d", ballCount), 10, y)
        y = y + lineHeight

        local cueBall = state.stateManager:getCueBall()
        if cueBall then
            love.graphics.setColor(TEXT_COLOR_SUCCESS)
            love.graphics.print(string.format("Cue ball: (%.1f, %.1f)", cueBall.position.x, cueBall.position.y), 10, y)
        else
            love.graphics.setColor(TEXT_COLOR_WARNING)
            love.graphics.print("Cue ball: Not detected", 10, y)
        end
        y = y + lineHeight

        local cue = state.stateManager:getCue()
        if cue then
            love.graphics.setColor(TEXT_COLOR_SUCCESS)
            love.graphics.print("Cue stick: Detected", 10, y)
        else
            love.graphics.setColor(TEXT_COLOR)
            love.graphics.print("Cue stick: Not detected", 10, y)
        end
        y = y + lineHeight

        love.graphics.setColor(TEXT_COLOR)
        love.graphics.print(string.format("Sequence #: %d", state.stateManager:getSequenceNumber()), 10, y)
        y = y + lineHeight
    end

    love.graphics.pop()
end

function love.keypressed(key)
    -- Toggle HUD
    if key == "f1" then
        CONFIG.showHUD = not CONFIG.showHUD
    end

    -- Toggle debug
    if key == "f2" then
        CONFIG.debug = not CONFIG.debug
    end

    -- Clear error
    if key == "e" and state.lastError then
        state.lastError = nil
    end

    -- Quit
    if key == "escape" then
        love.event.quit()
    end
end

function love.quit()
    print("Billiards Visualizer Shutting down...")

    -- TODO: Cleanup modules in Task Group 3
    -- TODO: Close WebSocket connection in Task Group 2

    print("Goodbye!")
    return false -- Allow quit
end

function love.errorhandler(msg)
    print("Fatal error: " .. tostring(msg))
    state.lastError = msg
end
