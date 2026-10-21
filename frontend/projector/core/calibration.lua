-- Calibration Module
-- Handles perspective transformation and calibration for projector alignment

local CoordinateContract = require "core.coordinate_contract"

local function defaultCorners()
    return {
        {x = 0.1, y = 0.1},
        {x = 0.9, y = 0.1},
        {x = 0.9, y = 0.9},
        {x = 0.1, y = 0.9}
    }
end

local Calibration = {
    -- The four output boundary corners are always normalized projector points.
    -- Their enclosing coordinate metadata is persisted with the calibration.
    coordinate_space = "normalized_projector",
    coordinate_scale = {1, 1},
    corners = defaultCorners(),

    -- Source rectangle (table dimensions in world units)
    sourceRect = {
        {x = 0, y = 0},
        {x = 1, y = 0},
        {x = 1, y = 1},
        {x = 0, y = 1}
    },

    -- Transformation matrix
    matrix = nil,
    transformMatrix = nil,  -- LÖVE2D Transform object

    -- Calibration state
    selectedCorner = 1,
    isDirty = false,

    -- Visual settings
    cornerSize = 10,
    lineWidth = 2,
    gridLines = 10,

    -- Save file
    saveFile = "calibration.json"
}

-- Initialize calibration
function Calibration:init()
    self:calculateTransform()
end

-- Load calibration from file
function Calibration:load()
    local json = require("lib.json")

    -- Check if save file exists
    if love.filesystem.getInfo(self.saveFile) then
        local contents = love.filesystem.read(self.saveFile)
        if contents then
            local success, data = pcall(json.decode, contents)
            local valid = success
                and type(data) == "table"
                and CoordinateContract.validateProjectorCorners(
                    data.corners,
                    data.coordinate_space,
                    data.coordinate_scale
                )
            if valid then
                self.corners = data.corners
                self:calculateTransform()
                print("Calibration loaded from " .. self.saveFile)
                return true
            end
        end
    end

    -- Use default calibration
    self.corners = defaultCorners()
    self:calculateTransform()
    print("Using default calibration")
    return false
end

-- Save calibration to file
function Calibration:save()
    local json = require("lib.json")
    local valid, validationError = CoordinateContract.validateProjectorCorners(
        self.corners,
        self.coordinate_space,
        self.coordinate_scale
    )
    if not valid then
        print("Calibration was not saved: " .. validationError)
        return false
    end

    local data = {
        coordinate_space = self.coordinate_space,
        coordinate_scale = self.coordinate_scale,
        corners = self.corners,
        timestamp = os.time()
    }

    local jsonStr = json.encode(data)
    local success = love.filesystem.write(self.saveFile, jsonStr)

    if success then
        print("Calibration saved to " .. self.saveFile)
    else
        print("Failed to save calibration")
    end

    return success
end

-- Calculate perspective transformation
function Calibration:calculateTransform()
    local valid = CoordinateContract.validateProjectorCorners(
        self.corners,
        self.coordinate_space,
        self.coordinate_scale
    )
    if not valid then
        self.transformMatrix = nil
        return false
    end

    -- For now, we'll use a simple linear transformation
    -- In a full implementation, this would calculate a proper perspective matrix

    -- Create a LÖVE2D Transform object
    self.transformMatrix = love.math.newTransform()

    -- Get screen dimensions
    local sw, sh = love.graphics.getDimensions()

    -- Calculate average position and scale
    local avgX = (self.corners[1].x + self.corners[2].x + self.corners[3].x + self.corners[4].x) / 4
    local avgY = (self.corners[1].y + self.corners[2].y + self.corners[3].y + self.corners[4].y) / 4

    local width = math.abs(self.corners[2].x - self.corners[1].x)
    local height = math.abs(self.corners[3].y - self.corners[1].y)

    -- Apply transformation
    self.transformMatrix:translate(self.corners[1].x * sw, self.corners[1].y * sh)
    self.transformMatrix:scale(width * sw, height * sh)

    self.isDirty = false
    return true
end

-- Get the transformation for rendering
function Calibration:getTransform()
    if self.isDirty then
        self:calculateTransform()
    end
    return self.transformMatrix
end

-- Transform a point from world to screen coordinates
function Calibration:transform(worldX, worldY)
    if not CoordinateContract.isUnitCoordinate(worldX)
        or not CoordinateContract.isUnitCoordinate(worldY) then
        return nil, nil
    end

    if not self.transformMatrix or self.isDirty then
        local calculated = self:calculateTransform()
        if not calculated then
            return nil, nil
        end
    end

    local sw, sh = love.graphics.getDimensions()

    -- Simple linear interpolation for now
    local screenX = self.corners[1].x * sw + worldX * (self.corners[2].x - self.corners[1].x) * sw
    local screenY = self.corners[1].y * sh + worldY * (self.corners[3].y - self.corners[1].y) * sh

    return screenX, screenY
end

-- Adjust selected corner
function Calibration:adjustCorner(dx, dy)
    if not CoordinateContract.isFiniteNumber(dx) or not CoordinateContract.isFiniteNumber(dy) then
        return false
    end
    local sw, sh = love.graphics.getDimensions()
    if not CoordinateContract.isFiniteNumber(sw) or not CoordinateContract.isFiniteNumber(sh) or sw <= 0 or sh <= 0 then
        return false
    end
    self.corners[self.selectedCorner].x = self.corners[self.selectedCorner].x + (dx / sw)
    self.corners[self.selectedCorner].y = self.corners[self.selectedCorner].y + (dy / sh)

    -- Clamp to screen bounds
    self.corners[self.selectedCorner].x = math.max(0, math.min(1, self.corners[self.selectedCorner].x))
    self.corners[self.selectedCorner].y = math.max(0, math.min(1, self.corners[self.selectedCorner].y))

    self.isDirty = true
    return true
end

-- Select next corner
function Calibration:nextCorner()
    self.selectedCorner = (self.selectedCorner % 4) + 1
end

-- Select corner by position
function Calibration:selectCorner(x, y)
    if not CoordinateContract.isFiniteNumber(x) or not CoordinateContract.isFiniteNumber(y) then
        return false
    end
    local sw, sh = love.graphics.getDimensions()
    if not CoordinateContract.isFiniteNumber(sw) or not CoordinateContract.isFiniteNumber(sh) or sw <= 0 or sh <= 0 then
        return false
    end
    local minDist = math.huge
    local selected = self.selectedCorner

    for i, corner in ipairs(self.corners) do
        local cx = corner.x * sw
        local cy = corner.y * sh
        local dist = math.sqrt((x - cx)^2 + (y - cy)^2)

        if dist < minDist and dist < 50 then  -- Within 50 pixels
            minDist = dist
            selected = i
        end
    end

    self.selectedCorner = selected
    return true
end

-- Drag corner to position
function Calibration:dragCorner(x, y)
    if not CoordinateContract.isFiniteNumber(x) or not CoordinateContract.isFiniteNumber(y) then
        return false
    end
    local sw, sh = love.graphics.getDimensions()
    if not CoordinateContract.isFiniteNumber(sw) or not CoordinateContract.isFiniteNumber(sh) or sw <= 0 or sh <= 0 then
        return false
    end
    self.corners[self.selectedCorner].x = math.max(0, math.min(1, x / sw))
    self.corners[self.selectedCorner].y = math.max(0, math.min(1, y / sh))

    self.isDirty = true
    return true
end

-- Reset calibration to default
function Calibration:reset()
    self.corners = defaultCorners()
    self.isDirty = true
end

-- Draw calibration overlay
function Calibration:drawOverlay()
    local sw, sh = love.graphics.getDimensions()

    love.graphics.push()
    love.graphics.origin()

    -- Draw grid
    love.graphics.setColor(0.3, 0.3, 0.3, 0.5)
    love.graphics.setLineWidth(1)

    for i = 0, self.gridLines do
        local t = i / self.gridLines

        -- Horizontal lines
        local x1, y1 = self:transform(0, t)
        local x2, y2 = self:transform(1, t)
        love.graphics.line(x1, y1, x2, y2)

        -- Vertical lines
        x1, y1 = self:transform(t, 0)
        x2, y2 = self:transform(t, 1)
        love.graphics.line(x1, y1, x2, y2)
    end

    -- Draw outline
    love.graphics.setColor(0, 1, 0, 0.8)
    love.graphics.setLineWidth(self.lineWidth)

    for i = 1, 4 do
        local j = (i % 4) + 1
        local x1 = self.corners[i].x * sw
        local y1 = self.corners[i].y * sh
        local x2 = self.corners[j].x * sw
        local y2 = self.corners[j].y * sh
        love.graphics.line(x1, y1, x2, y2)
    end

    -- Draw corners
    for i, corner in ipairs(self.corners) do
        local x = corner.x * sw
        local y = corner.y * sh

        if i == self.selectedCorner then
            love.graphics.setColor(1, 1, 0, 1)  -- Yellow for selected
            love.graphics.circle("fill", x, y, self.cornerSize)
        else
            love.graphics.setColor(0, 1, 0, 1)  -- Green for unselected
            love.graphics.circle("fill", x, y, self.cornerSize * 0.7)
        end

        -- Draw corner label
        love.graphics.setColor(1, 1, 1, 1)
        love.graphics.print(tostring(i), x + 15, y - 10)
    end

    love.graphics.pop()
end

return Calibration
