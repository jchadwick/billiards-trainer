-- Debug HUD Module
-- Displays real-time debug information including ball positions from the API
-- Configured via Config module (debug_hud section)

local DebugHUD = {
    name = "debug_hud",
    priority = 100,  -- Draw last (on top)
    enabled = true,

    -- Configuration (loaded from Config module)
    config = {
        enabled = true,
        position = "top_left",
        opacity = 0.9,
        font_size = 14,
        color = {255, 255, 255},
        background = {0, 0, 0, 128},
        sections = {
            connection = true,
            balls = true,
            cue = true,
            table = false,
            performance = true,
        },
        layout = "standard",
        update_rate = 10,  -- Hz
    },

    -- Internal state
    updateTimer = 0,
    lastUpdateTime = 0,
    networkStats = {
        connected = false,
        messagesReceived = 0,
        lastMessageTime = nil,
        websocketState = "DISCONNECTED",
    },
    gameState = {
        balls = {},
        cue = nil,
        table = nil,
        frameNumber = 0,
    },

    -- Colors
    colors = {
        text = {1, 1, 1, 0.9},
        success = {0, 1, 0, 0.9},
        error = {1, 0, 0, 0.9},
        warning = {1, 1, 0, 0.9},
        background = {0, 0, 0, 0.5},
    },

    -- Font
    font = nil,
}

--- Initialize the debug HUD module
function DebugHUD:init()
    print("Debug HUD module initializing...")

    -- Load configuration from global Config module
    if _G.Config then
        self.config.enabled = _G.Config:get("debug_hud.enabled") or true
        self.config.position = _G.Config:get("debug_hud.position") or "top_left"
        self.config.opacity = _G.Config:get("debug_hud.opacity") or 0.9
        self.config.font_size = _G.Config:get("debug_hud.font_size") or 14

        local sections = _G.Config:get("debug_hud.sections")
        if sections then
            self.config.sections = sections
        end

        print("Debug HUD configuration loaded from Config module")
    else
        print("Warning: Global Config module not found, using defaults")
    end

    -- Create font
    self.font = love.graphics.newFont(self.config.font_size)

    -- Subscribe to state updates from StateManager
    if _G.StateManager then
        print("Debug HUD connected to StateManager")
    else
        print("Warning: StateManager not available")
    end

    print("Debug HUD module initialized")
end

--- Update the debug HUD (called every frame)
-- @param dt number Delta time in seconds
function DebugHUD:update(dt)
    -- Update timer for refresh rate
    self.updateTimer = self.updateTimer + dt

    -- Only update at configured rate
    if self.updateTimer < (1.0 / self.config.update_rate) then
        return
    end

    self.updateTimer = 0
    self.lastUpdateTime = love.timer.getTime()

    -- Update network stats from Network module
    if _G.Network then
        local status = _G.Network:getStatus()
        self.networkStats.connected = (status.state == "CONNECTED")
        self.networkStats.websocketState = status.state or "UNKNOWN"
        self.networkStats.messagesReceived = status.messages_received or 0
        self.networkStats.lastMessageTime = status.lastMessageTime
    end

    -- Update game state from StateManager
    if _G.StateManager then
        self.gameState.balls = _G.StateManager:getBalls()
        self.gameState.cue = _G.StateManager:getCue()
        self.gameState.table = _G.StateManager:getTable()
        self.gameState.frameNumber = _G.StateManager:getSequenceNumber()
    end
end

--- Draw the debug HUD
function DebugHUD:draw()
    if not self.config.enabled then
        return
    end

    love.graphics.push()
    love.graphics.origin()

    -- Set font
    love.graphics.setFont(self.font)

    local x, y = self:getPosition()
    local lineHeight = self.config.font_size + 4

    -- Draw semi-transparent background
    local maxWidth = 400
    local lines = self:countLines()
    love.graphics.setColor(self.colors.background)
    love.graphics.rectangle("fill", x - 5, y - 5, maxWidth + 10, lines * lineHeight + 10)

    -- Draw sections
    if self.config.sections.performance then
        y = self:drawPerformanceSection(x, y, lineHeight)
    end

    if self.config.sections.connection then
        y = self:drawConnectionSection(x, y, lineHeight)
    end

    if self.config.sections.balls then
        y = self:drawBallsSection(x, y, lineHeight)
    end

    if self.config.sections.cue then
        y = self:drawCueSection(x, y, lineHeight)
    end

    if self.config.sections.table then
        y = self:drawTableSection(x, y, lineHeight)
    end

    -- Draw instructions at bottom
    y = y + lineHeight / 2
    love.graphics.setColor(self.colors.text)
    love.graphics.print("F1: Toggle HUD | F2: Toggle Debug | ESC: Quit", x, y)

    love.graphics.pop()
end

--- Get HUD position based on configuration
-- @return number, number x, y position
function DebugHUD:getPosition()
    local margin = 10

    if self.config.position == "top_left" then
        return margin, margin
    elseif self.config.position == "top_right" then
        return love.graphics.getWidth() - 400 - margin, margin
    elseif self.config.position == "bottom_left" then
        return margin, love.graphics.getHeight() - 400 - margin
    elseif self.config.position == "bottom_right" then
        return love.graphics.getWidth() - 400 - margin, love.graphics.getHeight() - 400 - margin
    else
        return margin, margin
    end
end

--- Count total lines to draw
-- @return number Total lines
function DebugHUD:countLines()
    local lines = 1  -- Instructions

    if self.config.sections.performance then
        lines = lines + 1
    end

    if self.config.sections.connection then
        lines = lines + 4
    end

    if self.config.sections.balls then
        local ballCount = 0
        for _ in pairs(self.gameState.balls) do
            ballCount = ballCount + 1
        end
        lines = lines + 2 + ballCount  -- Header + count + balls
    end

    if self.config.sections.cue then
        lines = lines + 1
    end

    if self.config.sections.table then
        lines = lines + 1
    end

    return lines
end

--- Draw performance section
-- @param x number X position
-- @param y number Y position
-- @param lineHeight number Line height
-- @return number New Y position
function DebugHUD:drawPerformanceSection(x, y, lineHeight)
    love.graphics.setColor(self.colors.text)
    love.graphics.print(string.format("FPS: %d", love.timer.getFPS()), x, y)
    return y + lineHeight
end

--- Draw connection section
-- @param x number X position
-- @param y number Y position
-- @param lineHeight number Line height
-- @return number New Y position
function DebugHUD:drawConnectionSection(x, y, lineHeight)
    -- Connection status
    if self.networkStats.connected then
        love.graphics.setColor(self.colors.success)
        love.graphics.print(string.format("WebSocket: Connected (%s)", self.networkStats.websocketState), x, y)
    else
        love.graphics.setColor(self.colors.error)
        love.graphics.print(string.format("WebSocket: Disconnected (%s)", self.networkStats.websocketState), x, y)
    end
    y = y + lineHeight

    -- Message stats
    love.graphics.setColor(self.colors.text)
    love.graphics.print(string.format("Messages: %d", self.networkStats.messagesReceived), x, y)
    y = y + lineHeight

    -- Last message time
    if self.networkStats.lastMessageTime then
        local timeSince = love.timer.getTime() - self.networkStats.lastMessageTime
        local color = timeSince < 1.0 and self.colors.success or
                     timeSince < 5.0 and self.colors.warning or
                     self.colors.error
        love.graphics.setColor(color)
        love.graphics.print(string.format("Last msg: %.1fs ago", timeSince), x, y)
    else
        love.graphics.setColor(self.colors.warning)
        love.graphics.print("Last msg: Never", x, y)
    end
    y = y + lineHeight

    -- Frame number
    love.graphics.setColor(self.colors.text)
    love.graphics.print(string.format("Frame: %d", self.gameState.frameNumber), x, y)
    y = y + lineHeight

    return y
end

--- Draw balls section
-- @param x number X position
-- @param y number Y position
-- @param lineHeight number Line height
-- @return number New Y position
function DebugHUD:drawBallsSection(x, y, lineHeight)
    -- Count balls
    local ballCount = 0
    local cueBall = nil
    for _, ball in pairs(self.gameState.balls) do
        ballCount = ballCount + 1
        if ball.is_cue_ball then
            cueBall = ball
        end
    end

    -- Header
    love.graphics.setColor(self.colors.text)
    love.graphics.print(string.format("=== Balls (%d detected) ===", ballCount), x, y)
    y = y + lineHeight

    -- Cue ball first (if exists)
    if cueBall then
        love.graphics.setColor(self.colors.success)
        love.graphics.print(
            string.format("  Cue Ball: pos=(%.2f, %.2f) vel=(%.2f, %.2f)",
                cueBall.position.x, cueBall.position.y,
                cueBall.velocity.x, cueBall.velocity.y
            ),
            x, y
        )
        y = y + lineHeight
    else
        love.graphics.setColor(self.colors.warning)
        love.graphics.print("  Cue Ball: Not detected", x, y)
        y = y + lineHeight
    end

    -- Other balls
    local otherBalls = {}
    for id, ball in pairs(self.gameState.balls) do
        if not ball.is_cue_ball then
            table.insert(otherBalls, {id = id, ball = ball})
        end
    end

    -- Sort by ID
    table.sort(otherBalls, function(a, b) return a.id < b.id end)

    -- Display balls (limit to first 10 to avoid clutter)
    local displayCount = math.min(#otherBalls, 10)
    for i = 1, displayCount do
        local entry = otherBalls[i]
        local ball = entry.ball

        -- Color based on movement
        local speed = math.sqrt(ball.velocity.x^2 + ball.velocity.y^2)
        local color = speed > 0.01 and self.colors.warning or self.colors.text
        love.graphics.setColor(color)

        local ballLabel = ball.number and string.format("Ball %d", ball.number) or string.format("Ball %s", entry.id)
        love.graphics.print(
            string.format("  %s: pos=(%.2f, %.2f) vel=(%.2f, %.2f)",
                ballLabel,
                ball.position.x, ball.position.y,
                ball.velocity.x, ball.velocity.y
            ),
            x, y
        )
        y = y + lineHeight
    end

    -- Show count if more balls exist
    if #otherBalls > displayCount then
        love.graphics.setColor(self.colors.text)
        love.graphics.print(string.format("  ... and %d more balls", #otherBalls - displayCount), x, y)
        y = y + lineHeight
    end

    return y
end

--- Draw cue section
-- @param x number X position
-- @param y number Y position
-- @param lineHeight number Line height
-- @return number New Y position
function DebugHUD:drawCueSection(x, y, lineHeight)
    if self.gameState.cue then
        love.graphics.setColor(self.colors.success)
        love.graphics.print(
            string.format("Cue: pos=(%.2f, %.2f) angle=%.1f°",
                self.gameState.cue.tip_position.x,
                self.gameState.cue.tip_position.y,
                self.gameState.cue.angle
            ),
            x, y
        )
    else
        love.graphics.setColor(self.colors.text)
        love.graphics.print("Cue: Not detected", x, y)
    end

    return y + lineHeight
end

--- Draw table section
-- @param x number X position
-- @param y number Y position
-- @param lineHeight number Line height
-- @return number New Y position
function DebugHUD:drawTableSection(x, y, lineHeight)
    if self.gameState.table then
        love.graphics.setColor(self.colors.text)
        love.graphics.print(
            string.format("Table: %.2f x %.2f m",
                self.gameState.table.width,
                self.gameState.table.height
            ),
            x, y
        )
    else
        love.graphics.setColor(self.colors.warning)
        love.graphics.print("Table: Not configured", x, y)
    end

    return y + lineHeight
end

--- Configure the debug HUD module
-- @param config table Configuration options
function DebugHUD:configure(config)
    if config.enabled ~= nil then
        self.config.enabled = config.enabled
    end

    if config.position then
        self.config.position = config.position
    end

    if config.sections then
        for key, value in pairs(config.sections) do
            self.config.sections[key] = value
        end
    end

    print("Debug HUD configuration updated")
end

--- Cleanup the debug HUD module
function DebugHUD:cleanup()
    print("Debug HUD module cleaning up")
    self.font = nil
    print("Debug HUD module cleaned up")
end

return DebugHUD
