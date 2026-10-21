-- Trajectory Visualization Module
-- Renders ball trajectories, collisions, and aiming guides

local Colors = require("modules.colors.init")
local Coordinates = require("core.coordinates")

local function world_point(point, fallbackScale, fallbackSpace)
    if type(point) ~= "table" or point.coordinate_scale ~= nil
        or point.units ~= nil then
        return nil
    end
    local x = point.x
    local y = point.y
    local unit = point.unit
    local space = point.coordinate_space or fallbackSpace
    if unit == "normalized_projector" or unit == "normalized_table" then
        space = space or unit
    end
    if unit == "meter" or unit == "meters" or space == "world_meters" then
        if (space ~= nil and space ~= "world_meters")
            or not Coordinates.isFiniteNumber(x) or not Coordinates.isFiniteNumber(y) then
            return nil
        end
        local copy = {}
        for key, value in pairs(point) do copy[key] = value end
        copy.x = x
        copy.y = y
        copy.unit = "meters"
        return copy
    end

    if space == "normalized_projector" or space == "normalized_table" then
        local scale = point.scale or fallbackScale
        if not Coordinates.validScale(scale) or scale[1] ~= 1 or scale[2] ~= 1
            or not Coordinates.isFiniteNumber(x) or not Coordinates.isFiniteNumber(y)
            or (unit ~= nil and unit ~= space) then
            return nil
        end
        local copy = {}
        for key, value in pairs(point) do copy[key] = value end
        copy.x = x
        copy.y = y
        copy.scale = {1, 1}
        copy.unit = space
        copy.coordinate_space = space
        return copy
    end
    return nil
end

local function normalize_point(point, fallbackScale, fallbackSpace)
    if type(point) == "table" and type(point.position) == "table" then
        if point.coordinate_scale ~= nil or point.units ~= nil then return nil end
        fallbackScale = point.scale or fallbackScale
        fallbackSpace = point.coordinate_space or fallbackSpace
        point = point.position
    end
    return world_point(point, fallbackScale, fallbackSpace)
        or Coordinates.toCanonicalPoint(point, fallbackScale, fallbackSpace)
end

local function normalized_line(line, data)
    if type(line) ~= "table" or line.coordinate_scale ~= nil
        or line.units ~= nil then
        return nil
    end
    local scale = line.scale or data.scale
    local space = line.coordinate_space or data.coordinate_space
    local startPoint = normalize_point(line.start, scale, space)
    local endPoint = normalize_point(line["end"], scale, space)
    if not startPoint or not endPoint then return nil end
    return {
        points = {startPoint, endPoint},
        ballType = line.type or "primary",
        timestamp = love.timer.getTime(),
        confidence = line.confidence or 1.0
    }
end

local function normalize_collision(collision, data)
    if type(collision) ~= "table" or type(collision.position) ~= "table"
        or collision.coordinate_scale ~= nil or collision.units ~= nil then
        return nil
    end
    local collisionScale = collision.scale or data.scale
    local collisionSpace = collision.coordinate_space or data.coordinate_space
    local point = normalize_point(collision.position, collisionScale, collisionSpace)
    if not point then return nil end
    return {
        position = point,
        type = collision.type or "ball",
        timestamp = love.timer.getTime()
    }
end

local function project_point(point, calibration)
    if point.unit == "normalized_projector" or point.unit == "normalized_table" then
        if not Coordinates.validScale(point.scale)
            or point.scale[1] ~= 1 or point.scale[2] ~= 1 then
            return nil, nil
        end
        return calibration:transform(point.x, point.y)
    end
    if point.unit == "meters" then
        local stateManager = _G.StateManager
        local tableState = stateManager and stateManager:getTable()
        local u, v = Coordinates.toPhysicalTablePlane(point, tableState)
        if not u then return nil, nil end
        return calibration:transform(u, v)
    end
    local stateManager = _G.StateManager
    local tableState = stateManager and stateManager:getTable()
    local u, v = Coordinates.toTablePlane(
        point, tableState and tableState.playing_area_corners
    )
    if not u then return nil, nil end
    return calibration:transform(u, v)
end

local Trajectory = {
    name = "trajectory",
    priority = 100,  -- Draw order (lower = earlier)
    enabled = true,

    -- Trajectory data
    paths = {},
    collisions = {},
    aimLine = nil,

    -- Visual settings
    fadeTime = 2.0,  -- Time for trajectory to fade out
    lineWidth = 3,
    showCollisionMarkers = true,
    showSpinIndicators = true,

    -- Colors now provided by adaptive color management system (Task Group 3)
    -- Falls back to hardcoded cyan if color module not available
    colors = nil  -- Will be set by init()
}

-- Keep the coordinate boundary directly testable without a renderer or LÖVE
-- window; production drawing uses the same function below.
Trajectory.projectPoint = project_point

-- Initialize module
function Trajectory:init()
    -- Get adaptive colors from color management system
    if Colors then
        self.colors = Colors:getTrajectoryColors()
    else
        -- Fallback to hardcoded cyan if colors module not loaded
        self.colors = {
            primary = {0, 1, 1, 0.9},
            collision = {1, 0.8, 0, 0.8},
            ghost = {1, 1, 1, 0.3},
            spin = {0, 0.5, 1, 0.6},
            aimLine = {1, 1, 1, 0.5}
        }
    end
    print("Trajectory module initialized")
end

-- Handle the optional aiming overlay message path.
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
    if type(data) ~= "table" then return end

    -- Store new trajectory
    self.paths = {}
    self.collisions = {}
    if data.coordinate_scale ~= nil or data.units ~= nil then return end

    if type(data.lines) == "table" then
        -- The backend contract contains independently tagged scaled segments.
        for _, line in ipairs(data.lines) do
            local normalized = normalized_line(line, data)
            if normalized then table.insert(self.paths, normalized) end
        end
    end

    -- Update collisions
    if type(data.collisions) == "table" then
        for _, collision in ipairs(data.collisions) do
            local normalized = normalize_collision(collision, data)
            if normalized then table.insert(self.collisions, normalized) end
        end
    end
end

function Trajectory:addCollision(data)
    local normalized = normalize_collision(data, {})
    if normalized then table.insert(self.collisions, normalized) end
end

function Trajectory:updateAimLine(data)
    if not data then
        self.aimLine = nil
        return
    end
    if data.coordinate_scale ~= nil or data.units ~= nil then
        self.aimLine = nil
        return
    end
    local scale = data.scale
    local space = data.coordinate_space
    local point1 = normalize_point(data.start, scale, space)
    local point2 = normalize_point(data.finish or data["end"], scale, space)
    self.aimLine = point1 and point2 and {
        start = point1,
        finish = point2,
        timestamp = love.timer.getTime()
    } or nil
end

-- Clear all trajectories
function Trajectory:clear()
    self.paths = {}
    self.collisions = {}
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

    if self.aimLine then
        local alpha = 1.0 - (currentTime - self.aimLine.timestamp) / self.fadeTime
        if alpha > 0 then
            local x1, y1 = project_point(self.aimLine.start, Calibration)
            local x2, y2 = project_point(self.aimLine.finish, Calibration)
            if x1 and y1 and x2 and y2 then
                local color = {
                    self.colors.aimLine[1],
                    self.colors.aimLine[2],
                    self.colors.aimLine[3],
                    self.colors.aimLine[4] * alpha
                }
                Renderer:drawDashedLine(x1, y1, x2, y2, 20, 10, color)
            end
        end
    end

    -- Draw trajectories
    for _, path in ipairs(self.paths) do
        local alpha = 1.0 - (currentTime - path.timestamp) / self.fadeTime
        if alpha > 0 and #path.points > 1 then
            -- Transform points to screen coordinates
            local screenPoints = {}
            for _, point in ipairs(path.points) do
                local x, y = project_point(point, Calibration)
                if not x or not y then
                    screenPoints = {}
                    break
                end
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
            local x, y = project_point(collision.position, Calibration)

            if x and y then
                local color = {
                    self.colors.collision[1],
                    self.colors.collision[2],
                    self.colors.collision[3],
                    self.colors.collision[4] * alpha
                }

                -- Marker sizes are projector UI sizes; all transported positions
                -- were scaled before reaching this rendering boundary.
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
    if config.showCollisionMarkers ~= nil then self.showCollisionMarkers = config.showCollisionMarkers end
    if config.colors then
        for key, color in pairs(config.colors) do
            if self.colors[key] then
                self.colors[key] = color
            end
        end
    end

    -- If table felt color changes, get new colors
    if config.table_felt then
        if Colors then
            Colors:setTableFeltColor(config.table_felt[1]/255, config.table_felt[2]/255, config.table_felt[3]/255)
            self.colors = Colors:getTrajectoryColors()
        end
    end
end

return Trajectory
