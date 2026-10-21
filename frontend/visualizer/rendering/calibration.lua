-- Calibration Module
-- Handles perspective transformation and calibration for projector alignment

local Coordinates = require("core.coordinates")

local Calibration = {
    -- Projector-output corners live in normalized screen fractions rather than
    -- camera pixels. Identity scale is explicit so they cannot be confused
    -- with untagged incoming image coordinates.
    corners = {
        {x = 0.1, y = 0.1, scale = {1, 1}, coordinate_space = "normalized_projector"},
        {x = 0.9, y = 0.1, scale = {1, 1}, coordinate_space = "normalized_projector"},
        {x = 0.9, y = 0.9, scale = {1, 1}, coordinate_space = "normalized_projector"},
        {x = 0.1, y = 0.9, scale = {1, 1}, coordinate_space = "normalized_projector"}
    },

    -- Source rectangle (table dimensions in world units)
    sourceRect = {
        {x = 0, y = 0, scale = {1, 1}, coordinate_space = "normalized_table"},
        {x = 1, y = 0, scale = {1, 1}, coordinate_space = "normalized_table"},
        {x = 1, y = 1, scale = {1, 1}, coordinate_space = "normalized_table"},
        {x = 0, y = 1, scale = {1, 1}, coordinate_space = "normalized_table"}
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
            if success and data and data.corners then
                local valid = type(data.corners) == "table" and #data.corners == 4
                local corners = {}
                if valid then
                    for index, corner in ipairs(data.corners) do
                        if type(corner) ~= "table"
                            or type(corner.x) ~= "number"
                            or type(corner.y) ~= "number"
                            or corner.x ~= corner.x or corner.y ~= corner.y
                            or math.abs(corner.x) == math.huge
                            or math.abs(corner.y) == math.huge then
                            valid = false
                            break
                        end
                        -- Existing local calibration files have always stored
                        -- normalized projector fractions. Upgrade that known
                        -- format with explicit metadata; never infer camera scale.
                        if corner.coordinate_space
                            and corner.coordinate_space ~= "normalized_projector" then
                            valid = false
                            break
                        end
                        if corner.scale and not Coordinates.validScale(corner.scale) then
                            valid = false
                            break
                        end
                        if corner.scale
                            and (corner.scale[1] ~= 1 or corner.scale[2] ~= 1) then
                            valid = false
                            break
                        end
                        if corner.x < 0 or corner.x > 1
                            or corner.y < 0 or corner.y > 1 then
                            valid = false
                            break
                        end
                        corners[index] = {
                            x = corner.x,
                            y = corner.y,
                            scale = {1, 1},
                            coordinate_space = "normalized_projector"
                        }
                    end
                end
                if valid then
                    self.corners = corners
                    self:calculateTransform()
                    print("Calibration loaded from " .. self.saveFile)
                    return true
                end
            end
        end
    end

    -- Use default calibration
    self:calculateTransform()
    print("Using default calibration")
    return false
end

-- Save calibration to file
function Calibration:save()
    local json = require("lib.json")

    local data = {
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
    if type(worldX) ~= "number" or type(worldY) ~= "number"
        or worldX ~= worldX or worldY ~= worldY
        or math.abs(worldX) == math.huge or math.abs(worldY) == math.huge then
        return nil, nil
    end

    local sw, sh = love.graphics.getDimensions()
    local corners = self.corners
    if type(corners) ~= "table" or #corners ~= 4 then return nil, nil end
    for _, corner in ipairs(corners) do
        if type(corner) ~= "table"
            or type(corner.x) ~= "number" or type(corner.y) ~= "number"
            or corner.x ~= corner.x or corner.y ~= corner.y
            or math.abs(corner.x) == math.huge or math.abs(corner.y) == math.huge
            or corner.coordinate_space ~= "normalized_projector"
            or not Coordinates.validScale(corner.scale)
            or corner.scale[1] ~= 1 or corner.scale[2] ~= 1 then
            return nil, nil
        end
    end

    -- Bilinear interpolation preserves the existing normalized calibration
    -- plane while honoring all four user-adjusted output corners. In
    -- particular, (0,0), (1,0), (1,1), and (0,1) map exactly to their paired
    -- projector corners even when the screen quadrilateral is skewed.
    local inverseU, inverseV = 1 - worldX, 1 - worldY
    local screenX = inverseU * inverseV * corners[1].x
        + worldX * inverseV * corners[2].x
        + worldX * worldY * corners[3].x
        + inverseU * worldY * corners[4].x
    local screenY = inverseU * inverseV * corners[1].y
        + worldX * inverseV * corners[2].y
        + worldX * worldY * corners[3].y
        + inverseU * worldY * corners[4].y

    return screenX * sw, screenY * sh
end

-- Transform a canonical camera/table point through calibrated playing-area
-- bounds before applying the projector calibration. Pixel scale factors must
-- already have been applied at the WebSocket boundary.
function Calibration:transformCanonical(pixelX, pixelY)
    if type(pixelX) ~= "number" or type(pixelY) ~= "number"
        or pixelX ~= pixelX or pixelY ~= pixelY
        or math.abs(pixelX) == math.huge or math.abs(pixelY) == math.huge then
        return nil, nil
    end
    local stateManager = _G.StateManager
    local tableState = stateManager and stateManager:getTable()
    local u, v = Coordinates.toTablePlane({
        x = pixelX,
        y = pixelY,
        scale = {1, 1},
        coordinate_space = Coordinates.CANONICAL_SPACE
    }, tableState and tableState.playing_area_corners)
    if not u then return nil, nil end
    return self:transform(u, v)
end

-- Size of the calibrated output plane in current window pixels. Physical and
-- canonical radii are mapped through this scale at their source unit boundary.
function Calibration:getOutputScale()
    local sw, sh = love.graphics.getDimensions()
    return {
        width = math.abs(self.corners[2].x - self.corners[1].x) * sw,
        height = math.abs(self.corners[4].y - self.corners[1].y) * sh
    }
end

-- Adjust selected corner
function Calibration:adjustCorner(dx, dy)
    local sw, sh = love.graphics.getDimensions()
    self.corners[self.selectedCorner].x = self.corners[self.selectedCorner].x + (dx / sw)
    self.corners[self.selectedCorner].y = self.corners[self.selectedCorner].y + (dy / sh)
    self.corners[self.selectedCorner].scale = {1, 1}
    self.corners[self.selectedCorner].coordinate_space = "normalized_projector"

    -- Clamp to screen bounds
    self.corners[self.selectedCorner].x = math.max(0, math.min(1, self.corners[self.selectedCorner].x))
    self.corners[self.selectedCorner].y = math.max(0, math.min(1, self.corners[self.selectedCorner].y))

    self.isDirty = true
end

-- Select next corner
function Calibration:nextCorner()
    self.selectedCorner = (self.selectedCorner % 4) + 1
end

-- Select corner by position
function Calibration:selectCorner(x, y)
    local sw, sh = love.graphics.getDimensions()
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
end

-- Drag corner to position
function Calibration:dragCorner(x, y)
    local sw, sh = love.graphics.getDimensions()
    self.corners[self.selectedCorner].x = x / sw
    self.corners[self.selectedCorner].y = y / sh
    self.corners[self.selectedCorner].scale = {1, 1}
    self.corners[self.selectedCorner].coordinate_space = "normalized_projector"

    -- Clamp to screen bounds
    self.corners[self.selectedCorner].x = math.max(0, math.min(1, self.corners[self.selectedCorner].x))
    self.corners[self.selectedCorner].y = math.max(0, math.min(1, self.corners[self.selectedCorner].y))

    self.isDirty = true
end

-- Reset calibration to default
function Calibration:reset()
    self.corners = {
        {x = 0.1, y = 0.1, scale = {1, 1}, coordinate_space = "normalized_projector"},
        {x = 0.9, y = 0.1, scale = {1, 1}, coordinate_space = "normalized_projector"},
        {x = 0.9, y = 0.9, scale = {1, 1}, coordinate_space = "normalized_projector"},
        {x = 0.1, y = 0.9, scale = {1, 1}, coordinate_space = "normalized_projector"}
    }
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
