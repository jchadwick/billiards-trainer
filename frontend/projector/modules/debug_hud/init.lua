-- Debug HUD Module
-- Displays comprehensive debug information overlay

local DebugHUD = {
    name = "debug_hud",
    priority = 900,  -- Render on top of most things
    enabled = false,  -- Disabled by default, toggle with H key

    -- Configuration
    transparency = 0.7,
    sections = {
        ball_info = true,
        cue_info = true,
        table_info = true,
        performance = true,
        network = true
    },

    -- Data tracking
    gameState = {
        balls = {},
        cue = nil,
        table = nil,
        lastUpdate = 0
    },
    trajectoryData = {
        paths = {},
        collisions = {},
        ghostBalls = {},
        lastUpdate = 0
    },
    frameStats = {
        fps = 0,
        latency = 0,
        lastUpdate = 0
    },
    networkStats = {
        messagesReceived = 0,
        lastMessageTime = 0,
        messagesByType = {},
        sequenceNumbers = {}
    },

    -- Layout configuration
    layout = {
        padding = 10,
        lineHeight = 16,
        sectionSpacing = 20,
        fontSize = 14
    },

    -- Colors (will be calculated based on background)
    colors = {
        background = {0, 0, 0, 0.7},
        text = {1, 1, 1, 1},
        good = {0, 1, 0, 1},
        warning = {1, 1, 0, 1},
        error = {1, 0, 0, 1},
        header = {0.7, 0.7, 1, 1}
    }
}

-- Initialize module
function DebugHUD:init()
    -- Calculate text colors based on background
    self:updateColors()
    print("Debug HUD module initialized (press H to toggle)")
end

-- Update colors based on background brightness
function DebugHUD:updateColors()
    -- Get background color from environment or default
    local bgEnv = os.getenv("PROJECTOR_BACKGROUND")
    local bgColor = {1, 1, 1}  -- Default white

    if bgEnv then
        local r, g, b = bgEnv:match("(%d+),(%d+),(%d+)")
        if r and g and b then
            bgColor = {tonumber(r)/255, tonumber(g)/255, tonumber(b)/255}
        end
    end

    -- Calculate brightness
    local brightness = (bgColor[1] * 0.299 + bgColor[2] * 0.587 + bgColor[3] * 0.114)

    -- Set colors based on brightness
    if brightness > 0.5 then
        -- Light background
        self.colors.text = {0, 0, 0, 0.9}
        self.colors.good = {0, 0.6, 0, 0.9}
        self.colors.warning = {0.7, 0.5, 0, 0.9}
        self.colors.error = {0.8, 0, 0, 0.9}
        self.colors.header = {0.2, 0.2, 0.6, 0.9}
        self.colors.background = {1, 1, 1, 0.8}
    else
        -- Dark background
        self.colors.text = {1, 1, 1, 0.9}
        self.colors.good = {0, 1, 0, 0.9}
        self.colors.warning = {1, 1, 0, 0.9}
        self.colors.error = {1, 0, 0, 0.9}
        self.colors.header = {0.7, 0.7, 1, 0.9}
        self.colors.background = {0, 0, 0, 0.8}
    end
end

-- Handle incoming messages
function DebugHUD:onMessage(messageType, data)
    -- Update network statistics for all messages
    self.networkStats.messagesReceived = self.networkStats.messagesReceived + 1
    self.networkStats.lastMessageTime = love.timer.getTime()

    -- Count messages by type
    if not self.networkStats.messagesByType[messageType] then
        self.networkStats.messagesByType[messageType] = 0
    end
    self.networkStats.messagesByType[messageType] = self.networkStats.messagesByType[messageType] + 1

    -- Handle specific message types
    if messageType == "state" then
        self:updateGameState(data)
    elseif messageType == "trajectory" then
        self:updateTrajectoryData(data)
    elseif messageType == "frame" then
        self:updateFrameStats(data)
    end
end

-- Update game state data
function DebugHUD:updateGameState(data)
    if not data then return end

    if data.balls then
        self.gameState.balls = data.balls
    end

    if data.cue then
        self.gameState.cue = data.cue
    end

    if data.table then
        self.gameState.table = data.table
    end

    self.gameState.lastUpdate = love.timer.getTime()
end

-- Update trajectory data
function DebugHUD:updateTrajectoryData(data)
    if not data then return end

    if data.paths then
        self.trajectoryData.paths = data.paths
    end

    if data.collisions then
        self.trajectoryData.collisions = data.collisions
    end

    if data.ghostBalls then
        self.trajectoryData.ghostBalls = data.ghostBalls
    end

    self.trajectoryData.lastUpdate = love.timer.getTime()
end

-- Update frame statistics
function DebugHUD:updateFrameStats(data)
    if not data then return end

    if data.fps then
        self.frameStats.fps = data.fps
    end

    if data.latency then
        self.frameStats.latency = data.latency
    end

    self.frameStats.lastUpdate = love.timer.getTime()
end

-- Update module (called every frame)
function DebugHUD:update(dt)
    -- Nothing to update per frame
end

-- Draw module
function DebugHUD:draw()
    if not self.enabled then return end

    love.graphics.push()
    love.graphics.origin()

    local screenWidth = love.graphics.getWidth()
    local screenHeight = love.graphics.getHeight()

    -- Draw sections
    local y = self.layout.padding

    -- Top-left: Ball Info
    if self.sections.ball_info then
        y = self:drawBallInfo(self.layout.padding, y)
        y = y + self.layout.sectionSpacing
    end

    -- Top-left (continued): Cue Info
    if self.sections.cue_info then
        y = self:drawCueInfo(self.layout.padding, y)
        y = y + self.layout.sectionSpacing
    end

    -- Top-right: Performance
    if self.sections.performance then
        local x = screenWidth - 300
        self:drawPerformance(x, self.layout.padding)
    end

    -- Bottom-left: Table Info
    if self.sections.table_info then
        local y = screenHeight - 150
        self:drawTableInfo(self.layout.padding, y)
    end

    -- Bottom-right: Network
    if self.sections.network then
        local x = screenWidth - 300
        local y = screenHeight - 200
        self:drawNetwork(x, y)
    end

    love.graphics.pop()
end

-- Draw ball information section
function DebugHUD:drawBallInfo(x, y)
    local lines = {}

    -- Header
    table.insert(lines, {text = "BALL INFO", color = self.colors.header})

    -- Ball count
    local ballCount = #self.gameState.balls
    table.insert(lines, {text = string.format("Count: %d", ballCount), color = self.colors.text})

    -- Ball details (show up to 5 balls)
    for i = 1, math.min(5, ballCount) do
        local ball = self.gameState.balls[i]
        local ballText = string.format("  #%d: pos=(%.2f,%.2f)",
            ball.id or i,
            ball.x or 0,
            ball.y or 0
        )

        if ball.vx or ball.vy then
            local speed = math.sqrt((ball.vx or 0)^2 + (ball.vy or 0)^2)
            ballText = ballText .. string.format(" spd=%.2f", speed)
        end

        if ball.confidence then
            ballText = ballText .. string.format(" conf=%.2f", ball.confidence)
        end

        -- Color code by confidence
        local color = self.colors.text
        if ball.confidence then
            if ball.confidence > 0.8 then
                color = self.colors.good
            elseif ball.confidence > 0.5 then
                color = self.colors.warning
            else
                color = self.colors.error
            end
        end

        table.insert(lines, {text = ballText, color = color})
    end

    if ballCount > 5 then
        table.insert(lines, {text = string.format("  ... and %d more", ballCount - 5), color = self.colors.text})
    end

    -- Data age
    local age = love.timer.getTime() - self.gameState.lastUpdate
    local ageColor = age < 1 and self.colors.good or (age < 5 and self.colors.warning or self.colors.error)
    table.insert(lines, {text = string.format("Updated: %.1fs ago", age), color = ageColor})

    return self:drawSection(x, y, lines, 350)
end

-- Draw cue information section
function DebugHUD:drawCueInfo(x, y)
    local lines = {}

    -- Header
    table.insert(lines, {text = "CUE INFO", color = self.colors.header})

    if self.gameState.cue then
        local cue = self.gameState.cue

        -- Angle
        if cue.angle then
            table.insert(lines, {text = string.format("Angle: %.1f°", math.deg(cue.angle)), color = self.colors.text})
        end

        -- Length
        if cue.length then
            table.insert(lines, {text = string.format("Length: %.2f", cue.length), color = self.colors.text})
        end

        -- Aiming state
        if cue.aiming ~= nil then
            local aimColor = cue.aiming and self.colors.good or self.colors.text
            table.insert(lines, {text = string.format("Aiming: %s", cue.aiming and "YES" or "NO"), color = aimColor})
        end

        -- Tip position
        if cue.tip_x and cue.tip_y then
            table.insert(lines, {text = string.format("Tip: (%.2f, %.2f)", cue.tip_x, cue.tip_y), color = self.colors.text})
        end

        -- Confidence
        if cue.confidence then
            local confColor = cue.confidence > 0.8 and self.colors.good or (cue.confidence > 0.5 and self.colors.warning or self.colors.error)
            table.insert(lines, {text = string.format("Confidence: %.2f", cue.confidence), color = confColor})
        end
    else
        table.insert(lines, {text = "No cue detected", color = self.colors.error})
    end

    return self:drawSection(x, y, lines, 350)
end

-- Draw table information section
function DebugHUD:drawTableInfo(x, y)
    local lines = {}

    -- Header
    table.insert(lines, {text = "TABLE INFO", color = self.colors.header})

    if self.gameState.table then
        local table = self.gameState.table

        -- Playing area status
        if table.playing_area then
            table.insert(lines, {text = "Playing Area: Detected", color = self.colors.good})

            if table.playing_area.width and table.playing_area.height then
                table.insert(lines, {
                    text = string.format("  Size: %.0f x %.0f", table.playing_area.width, table.playing_area.height),
                    color = self.colors.text
                })
            end
        else
            table.insert(lines, {text = "Playing Area: Not detected", color = self.colors.error})
        end

        -- Corner count
        if table.corners then
            local cornerCount = type(table.corners) == "table" and #table.corners or 0
            local cornerColor = cornerCount == 4 and self.colors.good or self.colors.warning
            table.insert(lines, {text = string.format("Corners: %d", cornerCount), color = cornerColor})
        end
    else
        table.insert(lines, {text = "No table data", color = self.colors.error})
    end

    return self:drawSection(x, y, lines, 300)
end

-- Draw performance section
function DebugHUD:drawPerformance(x, y)
    local lines = {}

    -- Header
    table.insert(lines, {text = "PERFORMANCE", color = self.colors.header})

    -- FPS
    local fps = love.timer.getFPS()
    local fpsColor = fps > 55 and self.colors.good or (fps > 30 and self.colors.warning or self.colors.error)
    table.insert(lines, {text = string.format("FPS: %d", fps), color = fpsColor})

    -- Backend FPS (if available)
    if self.frameStats.fps > 0 then
        local backendFpsColor = self.frameStats.fps > 25 and self.colors.good or (self.frameStats.fps > 15 and self.colors.warning or self.colors.error)
        table.insert(lines, {text = string.format("Backend FPS: %.1f", self.frameStats.fps), color = backendFpsColor})
    end

    -- Latency
    if self.frameStats.latency > 0 then
        local latencyColor = self.frameStats.latency < 50 and self.colors.good or (self.frameStats.latency < 100 and self.colors.warning or self.colors.error)
        table.insert(lines, {text = string.format("Latency: %.0fms", self.frameStats.latency), color = latencyColor})
    end

    -- Message rate
    local timeSinceMsg = love.timer.getTime() - self.networkStats.lastMessageTime
    local msgRate = timeSinceMsg < 1 and (1 / timeSinceMsg) or 0
    local msgRateColor = msgRate > 20 and self.colors.good or (msgRate > 10 and self.colors.warning or self.colors.text)
    table.insert(lines, {text = string.format("Msg Rate: %.1f/s", msgRate), color = msgRateColor})

    -- Trajectory info
    local pathCount = #self.trajectoryData.paths
    if pathCount > 0 then
        table.insert(lines, {text = string.format("Trajectories: %d", pathCount), color = self.colors.text})
    end

    return self:drawSection(x, y, lines, 280)
end

-- Draw network section
function DebugHUD:drawNetwork(x, y)
    local lines = {}

    -- Header
    table.insert(lines, {text = "NETWORK", color = self.colors.header})

    -- Connection status
    local Network = _G.Network
    if Network then
        local status = Network:getStatus()

        if status.udp then
            table.insert(lines, {text = "UDP: Connected", color = self.colors.good})
            table.insert(lines, {text = string.format("  Port: %d", status.udpPort or 9999), color = self.colors.text})
        else
            table.insert(lines, {text = "UDP: Disconnected", color = self.colors.error})
        end

        if status.websocket then
            table.insert(lines, {text = "WebSocket: Connected", color = self.colors.good})
        else
            table.insert(lines, {text = "WebSocket: N/A", color = self.colors.text})
        end
    end

    -- Total messages
    table.insert(lines, {text = string.format("Total: %d msgs", self.networkStats.messagesReceived), color = self.colors.text})

    -- Messages by type (top 3)
    local typeList = {}
    for msgType, count in pairs(self.networkStats.messagesByType) do
        table.insert(typeList, {type = msgType, count = count})
    end
    table.sort(typeList, function(a, b) return a.count > b.count end)

    for i = 1, math.min(3, #typeList) do
        table.insert(lines, {
            text = string.format("  %s: %d", typeList[i].type, typeList[i].count),
            color = self.colors.text
        })
    end

    -- Last message time
    local timeSinceMsg = love.timer.getTime() - self.networkStats.lastMessageTime
    local msgAgeColor = timeSinceMsg < 1 and self.colors.good or (timeSinceMsg < 5 and self.colors.warning or self.colors.error)
    table.insert(lines, {text = string.format("Last: %.1fs ago", timeSinceMsg), color = msgAgeColor})

    return self:drawSection(x, y, lines, 280)
end

-- Draw a section with background and lines
function DebugHUD:drawSection(x, y, lines, width)
    -- Calculate height
    local height = (#lines * self.layout.lineHeight) + (self.layout.padding * 2)

    -- Draw background
    love.graphics.setColor(self.colors.background)
    love.graphics.rectangle("fill", x, y, width, height)

    -- Draw border
    love.graphics.setColor(self.colors.text[1], self.colors.text[2], self.colors.text[3], 0.3)
    love.graphics.setLineWidth(1)
    love.graphics.rectangle("line", x, y, width, height)

    -- Draw lines
    local lineY = y + self.layout.padding
    for _, line in ipairs(lines) do
        love.graphics.setColor(line.color)
        love.graphics.print(line.text, x + self.layout.padding, lineY)
        lineY = lineY + self.layout.lineHeight
    end

    return y + height
end

-- Handle key press
function DebugHUD:onKeyPressed(key)
    if key == "h" then
        self.enabled = not self.enabled
        print(string.format("Debug HUD %s", self.enabled and "enabled" or "disabled"))
    end
end

-- Cleanup module
function DebugHUD:cleanup()
    print("Debug HUD module cleaned up")
end

-- Module configuration
function DebugHUD:configure(config)
    if config.enabled ~= nil then
        self.enabled = config.enabled
    end

    if config.transparency then
        self.transparency = config.transparency
    end

    if config.sections then
        for key, value in pairs(config.sections) do
            if self.sections[key] ~= nil then
                self.sections[key] = value
            end
        end
    end
end

return DebugHUD
