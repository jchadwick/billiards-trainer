-- Billiards Projector Main Entry Point
-- This is the main LÖVE2D entry point that initializes all systems

local ModuleManager = require("core.module_manager")
local Network = require("core.network")
local Calibration = require("core.calibration")
local Renderer = require("core.renderer")

-- Global references for easy access
_G.ModuleManager = ModuleManager
_G.Network = Network
_G.Calibration = Calibration
_G.Renderer = Renderer

-- Configuration
local CONFIG = {
    debug = false,
    showFPS = true,
    -- Background color from env var (format: "R,G,B" where values are 0-255) or default to white for dev
    backgroundColor = (function()
        local bgEnv = os.getenv("PROJECTOR_BACKGROUND")
        if bgEnv then
            local r, g, b = bgEnv:match("(%d+),(%d+),(%d+)")
            if r and g and b then
                return {tonumber(r)/255, tonumber(g)/255, tonumber(b)/255}
            end
        end
        return {1, 1, 1}  -- White for local dev
    end)(),
    hotReloadKey = "r",
    calibrationKey = "c"
}

-- Calculate contrasting text color based on background
local function getTextColor(bgColor)
    -- Calculate perceived brightness using relative luminance formula
    local brightness = (bgColor[1] * 0.299 + bgColor[2] * 0.587 + bgColor[3] * 0.114)
    -- Use white text on dark backgrounds, black text on light backgrounds
    if brightness > 0.5 then
        return {0, 0, 0, 0.8}  -- Black text
    else
        return {1, 1, 1, 0.8}  -- White text
    end
end

local TEXT_COLOR = getTextColor(CONFIG.backgroundColor)
local TEXT_COLOR_SUCCESS = (function()
    local brightness = (CONFIG.backgroundColor[1] * 0.299 + CONFIG.backgroundColor[2] * 0.587 + CONFIG.backgroundColor[3] * 0.114)
    if brightness > 0.5 then
        return {0, 0.6, 0, 0.8}  -- Dark green on light background
    else
        return {0, 1, 0, 0.8}  -- Bright green on dark background
    end
end)()
local TEXT_COLOR_ERROR = {1, 0, 0, 0.8}  -- Red works on both
local TEXT_COLOR_WARNING = (function()
    local brightness = (CONFIG.backgroundColor[1] * 0.299 + CONFIG.backgroundColor[2] * 0.587 + CONFIG.backgroundColor[3] * 0.114)
    if brightness > 0.5 then
        return {0.7, 0.5, 0, 0.8}  -- Dark yellow/orange on light background
    else
        return {1, 1, 0, 0.8}  -- Bright yellow on dark background
    end
end)()

-- State
local state = {
    isCalibrating = false,
    isPaused = false,
    lastError = nil
}

function love.load()
    print("Billiards Projector Starting...")

    -- Set background color
    love.graphics.setBackgroundColor(CONFIG.backgroundColor)

    -- Initialize core systems
    local success, err = pcall(function()
        -- Initialize renderer
        Renderer:init()

        -- Load calibration
        Calibration:load()

        -- Initialize networking
        Network:init()

        -- Load all modules
        ModuleManager:loadModules()
    end)

    if not success then
        state.lastError = err
        print("Initialization error: " .. tostring(err))
    end

    print("Billiards Projector Ready")
end

function love.update(dt)
    if state.isPaused then return end

    -- Update network (receive UDP messages)
    local success, err = pcall(function()
        Network:update(dt)
    end)

    if not success then
        state.lastError = err
        print("Network error: " .. tostring(err))
    end

    -- Update all modules
    ModuleManager:update(dt)
end

function love.draw()
    -- Wrap everything in pcall to catch errors
    local success, err = pcall(function()
        -- Clear screen
        love.graphics.clear(CONFIG.backgroundColor)

        -- Apply calibration transformation
        love.graphics.push()

        if Calibration.matrix then
            -- Apply perspective transformation matrix
            love.graphics.applyTransform(Calibration:getTransform())
        end

        -- Draw all modules
        ModuleManager:draw()

        love.graphics.pop()

        -- Draw UI overlay (not affected by calibration)
        love.graphics.push()
        love.graphics.origin()
    end)

    if not success then
        state.lastError = err
        love.graphics.push()
        love.graphics.origin()
    end

    -- Show FPS
    if CONFIG.showFPS then
        love.graphics.setColor(TEXT_COLOR)
        love.graphics.print(string.format("FPS: %d", love.timer.getFPS()), 10, 10)
    end

    -- Show calibration mode
    if state.isCalibrating then
        love.graphics.setColor(TEXT_COLOR_WARNING)
        love.graphics.print("CALIBRATION MODE - Press C to exit, Arrow keys to adjust", 10, 30)
    end

    -- Show errors
    if state.lastError then
        love.graphics.setColor(TEXT_COLOR_ERROR)
        love.graphics.print("ERROR: " .. tostring(state.lastError), 10, 50)
    end

    -- Show connection status
    local status = Network:getStatus()
    if status.udp then
        love.graphics.setColor(TEXT_COLOR_SUCCESS)
        love.graphics.print(string.format("UDP: Connected (Port %d)", status.udpPort or 9999), 10, 70)
    else
        love.graphics.setColor(TEXT_COLOR_ERROR)
        love.graphics.print("UDP: Disconnected", 10, 70)
    end

    if status.websocket then
        love.graphics.setColor(TEXT_COLOR_SUCCESS)
        love.graphics.print("WebSocket: Connected", 10, 90)
    else
        love.graphics.setColor(TEXT_COLOR)
        love.graphics.print("WebSocket: Disconnected", 10, 90)
    end

    -- Module status
    love.graphics.setColor(TEXT_COLOR)
    love.graphics.print(string.format("Modules: %d loaded", ModuleManager:getModuleCount()), 10, 110)

    -- Message stats
    love.graphics.setColor(TEXT_COLOR)
    love.graphics.print(string.format("Messages received: %d", status.messagesReceived or 0), 10, 130)

    if status.lastMessage then
        love.graphics.setColor(TEXT_COLOR)
        local timeSince = love.timer.getTime() - status.lastMessage
        love.graphics.print(string.format("Last message: %.1fs ago", timeSince), 10, 150)
    end

    -- Debug: Show trajectory data
    local trajectoryModule = ModuleManager:getModule("trajectory")
    if trajectoryModule then
        love.graphics.setColor(TEXT_COLOR)
        local pathCount = trajectoryModule.paths and #trajectoryModule.paths or 0
        local collisionCount = trajectoryModule.collisions and #trajectoryModule.collisions or 0
        local ghostCount = trajectoryModule.ghostBalls and #trajectoryModule.ghostBalls or 0
        love.graphics.print(string.format("Trajectories: %d paths, aim=%s, %d collisions, %d ghosts",
            pathCount,
            trajectoryModule.aimLine and "yes" or "no",
            collisionCount,
            ghostCount), 10, 170)
    end

    -- Debug: Test calibration transform
    if Calibration and Calibration.transform then
        local success, x1, y1 = pcall(Calibration.transform, Calibration, 0.2, 0.3)
        if success then
            local success2, x2, y2 = pcall(Calibration.transform, Calibration, 0.8, 0.7)
            if success2 then
                love.graphics.setColor(TEXT_COLOR)
                love.graphics.print(string.format("Transform: (0.2,0.3)->(%d,%d) (0.8,0.7)->(%d,%d)",
                    math.floor(x1 or 0), math.floor(y1 or 0), math.floor(x2 or 0), math.floor(y2 or 0)), 10, 190)
            end
        else
            love.graphics.setColor(TEXT_COLOR_ERROR)
            love.graphics.print("Transform error: " .. tostring(x1), 10, 190)
        end
    else
        love.graphics.setColor(TEXT_COLOR_ERROR)
        love.graphics.print("Calibration not available", 10, 190)
    end

    love.graphics.pop()
end

function love.keypressed(key)
    -- Hot reload modules
    if key == CONFIG.hotReloadKey and love.keyboard.isDown("lctrl") then
        print("Hot reloading modules...")
        state.lastError = nil
        local success, err = pcall(function()
            ModuleManager:reloadModules()
        end)
        if not success then
            state.lastError = err
            print("Reload error: " .. tostring(err))
        else
            print("Modules reloaded successfully")
        end
    end

    -- Toggle calibration mode
    if key == CONFIG.calibrationKey then
        state.isCalibrating = not state.isCalibrating
        if state.isCalibrating then
            ModuleManager:broadcast("onCalibrationStart")
        else
            ModuleManager:broadcast("onCalibrationEnd")
            Calibration:save()
        end
    end

    -- Pause
    if key == "p" then
        state.isPaused = not state.isPaused
    end

    -- Quit
    if key == "escape" then
        love.event.quit()
    end

    -- Clear error
    if key == "e" and state.lastError then
        state.lastError = nil
    end

    -- Calibration adjustments
    if state.isCalibrating then
        local shift = love.keyboard.isDown("lshift") or love.keyboard.isDown("rshift")
        local step = shift and 10 or 1

        if key == "left" then
            Calibration:adjustCorner(-step, 0)
        elseif key == "right" then
            Calibration:adjustCorner(step, 0)
        elseif key == "up" then
            Calibration:adjustCorner(0, -step)
        elseif key == "down" then
            Calibration:adjustCorner(0, step)
        elseif key == "tab" then
            Calibration:nextCorner()
        elseif key == "r" then
            Calibration:reset()
        end
    end

    -- Pass to modules
    ModuleManager:broadcast("onKeyPressed", key)
end

function love.keyreleased(key)
    ModuleManager:broadcast("onKeyReleased", key)
end

function love.mousepressed(x, y, button)
    if state.isCalibrating then
        Calibration:selectCorner(x, y)
    end
    ModuleManager:broadcast("onMousePressed", x, y, button)
end

function love.mousereleased(x, y, button)
    ModuleManager:broadcast("onMouseReleased", x, y, button)
end

function love.mousemoved(x, y, dx, dy)
    if state.isCalibrating and love.mouse.isDown(1) then
        Calibration:dragCorner(x, y)
    end
    ModuleManager:broadcast("onMouseMoved", x, y, dx, dy)
end

function love.quit()
    print("Billiards Projector Shutting down...")

    -- Cleanup modules
    ModuleManager:cleanup()

    -- Close network connections
    Network:cleanup()

    -- Save calibration
    Calibration:save()

    print("Goodbye!")
    return false -- Allow quit
end

function love.errorhandler(msg)
    print("Fatal error: " .. tostring(msg))
    state.lastError = msg
end
