-- Core Renderer Module
-- Provides drawing primitives for visualizer modules
-- Uses LOVE2D graphics API with proper state management

local Renderer = {}

-- Initialize renderer
function Renderer:init()
    -- Store default graphics state
    self.defaultLineWidth = 1
    self.defaultFont = love.graphics.getFont()

    -- Graphics state stack for nested operations
    self.stateStack = {}

    print("Renderer initialized")
end

-- Push current graphics state onto stack
function Renderer:pushState()
    table.insert(self.stateStack, {
        color = {love.graphics.getColor()},
        lineWidth = love.graphics.getLineWidth(),
        font = love.graphics.getFont()
    })
end

-- Pop graphics state from stack
function Renderer:popState()
    if #self.stateStack > 0 then
        local state = table.remove(self.stateStack)
        love.graphics.setColor(unpack(state.color))
        love.graphics.setLineWidth(state.lineWidth)
        love.graphics.setFont(state.font)
    end
    -- If stack is empty, just do nothing (defensive programming)
end

-- Reset graphics state to defaults
function Renderer:reset()
    love.graphics.setColor(1, 1, 1, 1)
    love.graphics.setLineWidth(self.defaultLineWidth)
    love.graphics.setFont(self.defaultFont)
    self.stateStack = {}
end

-- Draw a simple line
-- @param x1, y1: Start position
-- @param x2, y2: End position
-- @param color: {r, g, b, a} table
-- @param width: Line width (optional, default 1)
function Renderer:drawLine(x1, y1, x2, y2, color, width)
    self:pushState()

    love.graphics.setColor(unpack(color))
    love.graphics.setLineWidth(width or 1)
    love.graphics.line(x1, y1, x2, y2)

    self:popState()
end

-- Draw a dashed line
-- @param x1, y1: Start position
-- @param x2, y2: End position
-- @param dashLen: Length of each dash
-- @param gapLen: Length of gaps between dashes
-- @param color: {r, g, b, a} table
-- @param width: Line width (optional, default 1)
function Renderer:drawDashedLine(x1, y1, x2, y2, dashLen, gapLen, color, width)
    self:pushState()

    love.graphics.setColor(unpack(color))
    love.graphics.setLineWidth(width or 1)

    -- Calculate line direction and length
    local dx = x2 - x1
    local dy = y2 - y1
    local length = math.sqrt(dx * dx + dy * dy)

    if length == 0 then
        self:popState()
        return
    end

    -- Normalize direction
    local dirX = dx / length
    local dirY = dy / length

    -- Draw dashes along the line
    local pos = 0
    local segmentLen = dashLen + gapLen

    while pos < length do
        local dashEnd = math.min(pos + dashLen, length)
        local startX = x1 + dirX * pos
        local startY = y1 + dirY * pos
        local endX = x1 + dirX * dashEnd
        local endY = y1 + dirY * dashEnd

        love.graphics.line(startX, startY, endX, endY)
        pos = pos + segmentLen
    end

    self:popState()
end

-- Draw a circle
-- @param x, y: Center position
-- @param radius: Circle radius
-- @param color: {r, g, b, a} table
-- @param filled: true for filled circle, false for outline (optional, default false)
-- @param segments: Number of segments for smoothness (optional)
function Renderer:drawCircle(x, y, radius, color, filled, segments)
    self:pushState()

    love.graphics.setColor(unpack(color))

    local mode = filled and "fill" or "line"
    local segs = segments or math.max(16, math.floor(radius * 2))

    love.graphics.circle(mode, x, y, radius, segs)

    self:popState()
end

-- Draw a trajectory with gradient color from start to end
-- @param points: Array of {x, y} coordinate pairs
-- @param startColor: {r, g, b, a} color at trajectory start
-- @param endColor: {r, g, b, a} color at trajectory end
-- @param width: Line width (optional, default 2)
function Renderer:drawTrajectory(points, startColor, endColor, width)
    if not points or #points < 2 then
        return
    end

    self:pushState()

    love.graphics.setLineWidth(width or 2)

    -- Draw segments with interpolated colors
    for i = 1, #points - 1 do
        local p1 = points[i]
        local p2 = points[i + 1]

        -- Calculate color interpolation factor (0 to 1)
        local t = (i - 1) / (#points - 1)

        -- Interpolate between start and end colors
        local r = startColor[1] + (endColor[1] - startColor[1]) * t
        local g = startColor[2] + (endColor[2] - startColor[2]) * t
        local b = startColor[3] + (endColor[3] - startColor[3]) * t
        local a = startColor[4] + (endColor[4] - startColor[4]) * t

        love.graphics.setColor(r, g, b, a)
        love.graphics.line(p1[1], p1[2], p2[1], p2[2])
    end

    self:popState()
end

-- Draw text
-- @param text: String to display
-- @param x, y: Position
-- @param font: LOVE2D font object (optional, uses current font)
-- @param color: {r, g, b, a} table (optional, default white)
-- @param align: "left", "center", or "right" (optional, default "left")
function Renderer:drawText(text, x, y, font, color, align)
    self:pushState()

    if font then
        love.graphics.setFont(font)
    end

    if color then
        love.graphics.setColor(unpack(color))
    else
        love.graphics.setColor(1, 1, 1, 1)
    end

    local alignX = x
    if align == "center" then
        local textWidth = love.graphics.getFont():getWidth(text)
        alignX = x - textWidth / 2
    elseif align == "right" then
        local textWidth = love.graphics.getFont():getWidth(text)
        alignX = x - textWidth
    end

    love.graphics.print(text, alignX, y)

    self:popState()
end

-- Draw a rectangle
-- @param x, y: Top-left position
-- @param width, height: Rectangle dimensions
-- @param color: {r, g, b, a} table
-- @param filled: true for filled, false for outline (optional, default false)
function Renderer:drawRectangle(x, y, width, height, color, filled)
    self:pushState()

    love.graphics.setColor(unpack(color))

    local mode = filled and "fill" or "line"
    love.graphics.rectangle(mode, x, y, width, height)

    self:popState()
end

-- Draw a polygon
-- @param vertices: Flat array of coordinates {x1, y1, x2, y2, ...}
-- @param color: {r, g, b, a} table
-- @param filled: true for filled, false for outline (optional, default false)
function Renderer:drawPolygon(vertices, color, filled)
    if not vertices or #vertices < 6 then
        return -- Need at least 3 points (6 values)
    end

    self:pushState()

    love.graphics.setColor(unpack(color))

    local mode = filled and "fill" or "line"
    love.graphics.polygon(mode, vertices)

    self:popState()
end

-- Draw an arc
-- @param x, y: Center position
-- @param radius: Arc radius
-- @param angle1, angle2: Start and end angles in radians
-- @param color: {r, g, b, a} table
-- @param filled: true for filled (pie slice), false for outline (optional, default false)
function Renderer:drawArc(x, y, radius, angle1, angle2, color, filled)
    self:pushState()

    love.graphics.setColor(unpack(color))

    local arcType = filled and "pie" or "open"
    local segments = math.max(16, math.floor(radius * 2))

    love.graphics.arc(arcType, x, y, radius, angle1, angle2, segments)

    self:popState()
end

-- Helper: Interpolate between two colors
-- @param color1, color2: {r, g, b, a} tables
-- @param t: Interpolation factor (0 to 1)
-- @return: Interpolated color {r, g, b, a}
function Renderer:interpolateColor(color1, color2, t)
    return {
        color1[1] + (color2[1] - color1[1]) * t,
        color1[2] + (color2[2] - color1[2]) * t,
        color1[3] + (color2[3] - color1[3]) * t,
        color1[4] + (color2[4] - color1[4]) * t
    }
end

-- Helper: Calculate distance between two points
function Renderer:distance(x1, y1, x2, y2)
    local dx = x2 - x1
    local dy = y2 - y1
    return math.sqrt(dx * dx + dy * dy)
end

return Renderer
