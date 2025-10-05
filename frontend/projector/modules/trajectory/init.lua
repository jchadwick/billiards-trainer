-- Trajectory Visualization Module
-- Renders ball trajectories, collisions, and aiming guides

local Trajectory = {
    name = "trajectory",
    priority = 100,  -- Draw order (lower = earlier)
    enabled = true,

    -- Trajectory data
    paths = {},
    collisions = {},
    ghostBalls = {},
    aimLine = nil,

    -- Visual settings
    fadeTime = 2.0,  -- Time for trajectory to fade out
    lineWidth = 3,
    showGhostBalls = true,
    showCollisionMarkers = true,
    showSpinIndicators = true,

    -- Colors
    colors = {
        primary = {0.3, 0.8, 0.3, 0.8},      -- Green trajectory
        collision = {1, 0.8, 0, 0.8},        -- Yellow collision
        ghost = {1, 1, 1, 0.3},              -- White ghost ball
        spin = {0, 0.5, 1, 0.6},             -- Blue spin indicator
        aimLine = {1, 1, 1, 0.5}             -- White aim line
    }
}

-- Initialize module
function Trajectory:init()
    print("Trajectory module initialized")
end

-- Handle incoming messages
function Trajectory:onMessage(messageType, data)
    if messageType == "trajectory" then
        self:updateTrajectory(data)
    elseif messageType == "collision" then
        self:addCollision(data)
    elseif messageType == "aim" then
        self:updateAimLine(data)
    elseif messageType == "clear" then
        self:clear()
    end
end

-- Update trajectory data
function Trajectory:updateTrajectory(data)
    if not data then return end

    -- Store new trajectory
    self.paths = {}

    if data.paths then
        for _, path in ipairs(data.paths) do
            table.insert(self.paths, {
                points = path.points or {},
                ballType = path.ballType or "cue",
                timestamp = love.timer.getTime(),
                confidence = path.confidence or 1.0
            })
        end
    end

    -- Update collisions
    if data.collisions then
        self.collisions = {}
        for _, collision in ipairs(data.collisions) do
            table.insert(self.collisions, {
                x = collision.x,
                y = collision.y,
                type = collision.type,  -- "ball", "cushion", "pocket"
                timestamp = love.timer.getTime()
            })
        end
    end

    -- Update ghost balls
    if data.ghostBalls then
        self.ghostBalls = data.ghostBalls
    end
end

-- Add a collision marker
function Trajectory:addCollision(data)
    if not data or not data.x or not data.y then return end

    table.insert(self.collisions, {
        x = data.x,
        y = data.y,
        type = data.type or "ball",
        timestamp = love.timer.getTime()
    })
end

-- Update aim line
function Trajectory:updateAimLine(data)
    if not data then
        self.aimLine = nil
        return
    end

    self.aimLine = {
        x1 = data.x1,
        y1 = data.y1,
        x2 = data.x2,
        y2 = data.y2,
        timestamp = love.timer.getTime()
    }
end

-- Clear all trajectories
function Trajectory:clear()
    self.paths = {}
    self.collisions = {}
    self.ghostBalls = {}
    self.aimLine = nil
end

-- Update module (called every frame)
function Trajectory:update(dt)
    local currentTime = love.timer.getTime()

    -- Remove old trajectories
    for i = #self.paths, 1, -1 do
        if currentTime - self.paths[i].timestamp > self.fadeTime then
            table.remove(self.paths, i)
        end
    end

    -- Remove old collisions
    for i = #self.collisions, 1, -1 do
        if currentTime - self.collisions[i].timestamp > self.fadeTime then
            table.remove(self.collisions, i)
        end
    end

    -- Remove old aim line
    if self.aimLine and currentTime - self.aimLine.timestamp > self.fadeTime then
        self.aimLine = nil
    end
end

-- Draw module
function Trajectory:draw()
    local currentTime = love.timer.getTime()
    local Renderer = _G.Renderer
    local Calibration = _G.Calibration

    if not Renderer or not Calibration then
        return
    end

    -- DEBUG: Draw a test circle to verify draw is being called
    love.graphics.setColor(1, 0, 0, 1)  -- Red
    love.graphics.circle("fill", 200, 200, 50)

    -- DEBUG: Draw a simple test line using calibration
    local x1, y1 = Calibration:transform(0.2, 0.3)
    local x2, y2 = Calibration:transform(0.8, 0.7)

    -- Show transformed coordinates
    love.graphics.setColor(1, 1, 1, 1)
    love.graphics.print(string.format("Transform test: (0.2,0.3)->(%d,%d) (0.8,0.7)->(%d,%d)",
        math.floor(x1), math.floor(y1), math.floor(x2), math.floor(y2)), 10, 210)

    love.graphics.setColor(0, 1, 0, 1)  -- Green
    love.graphics.setLineWidth(5)
    love.graphics.line(x1, y1, x2, y2)

    -- Draw aim line
    if self.aimLine then
        local alpha = 1.0 - (currentTime - self.aimLine.timestamp) / self.fadeTime
        if alpha > 0 then
            local x1, y1 = Calibration:transform(self.aimLine.x1, self.aimLine.y1)
            local x2, y2 = Calibration:transform(self.aimLine.x2, self.aimLine.y2)

            local color = {
                self.colors.aimLine[1],
                self.colors.aimLine[2],
                self.colors.aimLine[3],
                self.colors.aimLine[4] * alpha
            }

            Renderer:drawDashedLine(x1, y1, x2, y2, 20, 10, color)
        end
    end

    -- Draw trajectories
    for _, path in ipairs(self.paths) do
        local alpha = 1.0 - (currentTime - path.timestamp) / self.fadeTime
        if alpha > 0 and #path.points > 1 then
            -- Transform points to screen coordinates
            local screenPoints = {}
            for _, point in ipairs(path.points) do
                local x, y = Calibration:transform(point.x, point.y)
                table.insert(screenPoints, {x = x, y = y})
            end

            -- Draw with fade
            local startColor = {
                self.colors.primary[1],
                self.colors.primary[2],
                self.colors.primary[3],
                self.colors.primary[4] * alpha * path.confidence
            }

            local endColor = {
                self.colors.primary[1] * 0.5,
                self.colors.primary[2] * 0.5,
                self.colors.primary[3] * 0.5,
                self.colors.primary[4] * alpha * 0.3 * path.confidence
            }

            Renderer:drawTrajectory(screenPoints, startColor, endColor, self.lineWidth)
        end
    end

    -- Draw collision markers
    if self.showCollisionMarkers then
        for _, collision in ipairs(self.collisions) do
            local alpha = 1.0 - (currentTime - collision.timestamp) / self.fadeTime
            if alpha > 0 then
                local x, y = Calibration:transform(collision.x, collision.y)

                local color = {
                    self.colors.collision[1],
                    self.colors.collision[2],
                    self.colors.collision[3],
                    self.colors.collision[4] * alpha
                }

                -- Draw collision circle
                love.graphics.setColor(color)
                love.graphics.circle("line", x, y, 10)

                -- Draw collision type icon
                if collision.type == "ball" then
                    love.graphics.circle("fill", x, y, 5)
                elseif collision.type == "cushion" then
                    love.graphics.rectangle("fill", x - 5, y - 5, 10, 10)
                elseif collision.type == "pocket" then
                    love.graphics.circle("line", x, y, 15)
                end
            end
        end
    end

    -- Draw ghost balls
    if self.showGhostBalls then
        for _, ghost in ipairs(self.ghostBalls) do
            local x, y = Calibration:transform(ghost.x, ghost.y)
            Renderer:drawCircle(x, y, ghost.radius or 10, self.colors.ghost, false)

            -- Draw ghost ball number if provided
            if ghost.number then
                love.graphics.setColor(self.colors.ghost)
                love.graphics.print(tostring(ghost.number), x - 5, y - 7)
            end
        end
    end

    -- Reset graphics state
    Renderer:reset()
end

-- Cleanup module
function Trajectory:cleanup()
    self:clear()
    print("Trajectory module cleaned up")
end

-- Module configuration
function Trajectory:configure(config)
    if config.fadeTime then self.fadeTime = config.fadeTime end
    if config.lineWidth then self.lineWidth = config.lineWidth end
    if config.showGhostBalls ~= nil then self.showGhostBalls = config.showGhostBalls end
    if config.showCollisionMarkers ~= nil then self.showCollisionMarkers = config.showCollisionMarkers end
    if config.colors then
        for key, color in pairs(config.colors) do
            if self.colors[key] then
                self.colors[key] = color
            end
        end
    end
end

return Trajectory
