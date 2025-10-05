-- Renderer Module
-- Core rendering utilities and helpers

local Renderer = {
    -- Default colors
    colors = {
        white = {1, 1, 1, 1},
        black = {0, 0, 0, 1},
        red = {1, 0, 0, 1},
        green = {0, 1, 0, 1},
        blue = {0, 0, 1, 1},
        yellow = {1, 1, 0, 1},
        cyan = {0, 1, 1, 1},
        magenta = {1, 0, 1, 1},

        -- Billiards specific
        tableGreen = {0.1, 0.5, 0.2, 1},
        cueBall = {0.95, 0.95, 0.9, 1},
        eightBall = {0.1, 0.1, 0.1, 1},
        trajectory = {0.3, 0.8, 0.3, 0.7},
        collision = {1, 0.8, 0, 0.8},
        ghost = {1, 1, 1, 0.3}
    },

    -- Line styles
    lineWidth = 2,
    defaultFont = nil
}

-- Initialize renderer
function Renderer:init()
    -- Set default line width
    love.graphics.setLineWidth(self.lineWidth)

    -- Create default font
    self.defaultFont = love.graphics.getFont()

    print("Renderer initialized")
end

-- Draw a line with optional gradient
function Renderer:drawLine(x1, y1, x2, y2, color1, color2)
    color1 = color1 or self.colors.white
    color2 = color2 or color1

    if color1 == color2 then
        -- Simple line
        love.graphics.setColor(color1)
        love.graphics.line(x1, y1, x2, y2)
    else
        -- Gradient line (simple approximation)
        local steps = 10
        for i = 0, steps - 1 do
            local t1 = i / steps
            local t2 = (i + 1) / steps

            local r = color1[1] * (1 - t1) + color2[1] * t1
            local g = color1[2] * (1 - t1) + color2[2] * t1
            local b = color1[3] * (1 - t1) + color2[3] * t1
            local a = (color1[4] or 1) * (1 - t1) + (color2[4] or 1) * t1

            love.graphics.setColor(r, g, b, a)

            local px1 = x1 * (1 - t1) + x2 * t1
            local py1 = y1 * (1 - t1) + y2 * t1
            local px2 = x1 * (1 - t2) + x2 * t2
            local py2 = y1 * (1 - t2) + y2 * t2

            love.graphics.line(px1, py1, px2, py2)
        end
    end
end

-- Draw a dashed line
function Renderer:drawDashedLine(x1, y1, x2, y2, dashLength, gapLength, color)
    color = color or self.colors.white
    dashLength = dashLength or 10
    gapLength = gapLength or 5

    love.graphics.setColor(color)

    local dx = x2 - x1
    local dy = y2 - y1
    local distance = math.sqrt(dx * dx + dy * dy)
    local dashCount = math.floor(distance / (dashLength + gapLength))

    local dashX = dx / distance * dashLength
    local dashY = dy / distance * dashLength
    local gapX = dx / distance * gapLength
    local gapY = dy / distance * gapLength

    local currentX = x1
    local currentY = y1

    for i = 1, dashCount do
        love.graphics.line(currentX, currentY, currentX + dashX, currentY + dashY)
        currentX = currentX + dashX + gapX
        currentY = currentY + dashY + gapY
    end

    -- Draw remaining segment
    if currentX < x2 or currentY < y2 then
        love.graphics.line(currentX, currentY, x2, y2)
    end
end

-- Draw an arrow
function Renderer:drawArrow(x1, y1, x2, y2, headSize, color)
    color = color or self.colors.white
    headSize = headSize or 15

    love.graphics.setColor(color)
    love.graphics.line(x1, y1, x2, y2)

    -- Calculate arrowhead
    local angle = math.atan2(y2 - y1, x2 - x1)
    local arrowAngle = math.pi / 6  -- 30 degrees

    local ax1 = x2 - headSize * math.cos(angle - arrowAngle)
    local ay1 = y2 - headSize * math.sin(angle - arrowAngle)
    local ax2 = x2 - headSize * math.cos(angle + arrowAngle)
    local ay2 = y2 - headSize * math.sin(angle + arrowAngle)

    love.graphics.polygon("fill", x2, y2, ax1, ay1, ax2, ay2)
end

-- Draw a circle
function Renderer:drawCircle(x, y, radius, color, filled)
    color = color or self.colors.white
    love.graphics.setColor(color)

    if filled then
        love.graphics.circle("fill", x, y, radius)
    else
        love.graphics.circle("line", x, y, radius)
    end
end

-- Draw a ball (billiards ball with number)
function Renderer:drawBall(x, y, radius, number, color, stripeColor)
    -- Draw main ball
    self:drawCircle(x, y, radius, color, true)

    -- Draw stripe if provided
    if stripeColor then
        love.graphics.setColor(stripeColor)
        love.graphics.arc("fill", x, y, radius, -math.pi/3, math.pi/3)
        love.graphics.arc("fill", x, y, radius, 2*math.pi/3, 4*math.pi/3)
    end

    -- Draw number
    if number then
        love.graphics.setColor(0, 0, 0, 1)
        local text = tostring(number)
        local font = love.graphics.getFont()
        local tw = font:getWidth(text)
        local th = font:getHeight()
        love.graphics.print(text, x - tw/2, y - th/2)
    end
end

-- Draw a rectangle
function Renderer:drawRectangle(x, y, width, height, color, filled)
    color = color or self.colors.white
    love.graphics.setColor(color)

    if filled then
        love.graphics.rectangle("fill", x, y, width, height)
    else
        love.graphics.rectangle("line", x, y, width, height)
    end
end

-- Draw text with shadow
function Renderer:drawText(text, x, y, color, shadowColor, shadowOffset)
    color = color or self.colors.white
    shadowColor = shadowColor or {0, 0, 0, 0.5}
    shadowOffset = shadowOffset or 2

    -- Draw shadow
    if shadowColor and shadowOffset > 0 then
        love.graphics.setColor(shadowColor)
        love.graphics.print(text, x + shadowOffset, y + shadowOffset)
    end

    -- Draw text
    love.graphics.setColor(color)
    love.graphics.print(text, x, y)
end

-- Draw a trajectory path
function Renderer:drawTrajectory(points, startColor, endColor, lineWidth)
    if #points < 2 then return end

    startColor = startColor or self.colors.trajectory
    endColor = endColor or startColor
    lineWidth = lineWidth or self.lineWidth

    love.graphics.setLineWidth(lineWidth)

    for i = 1, #points - 1 do
        local t = (i - 1) / (#points - 1)
        local r = startColor[1] * (1 - t) + endColor[1] * t
        local g = startColor[2] * (1 - t) + endColor[2] * t
        local b = startColor[3] * (1 - t) + endColor[3] * t
        local a = (startColor[4] or 1) * (1 - t) + (endColor[4] or 1) * t

        love.graphics.setColor(r, g, b, a)
        love.graphics.line(points[i].x, points[i].y, points[i+1].x, points[i+1].y)
    end

    love.graphics.setLineWidth(self.lineWidth)  -- Reset
end

-- Set blend mode for effects
function Renderer:setBlendMode(mode)
    if mode == "add" then
        love.graphics.setBlendMode("add")
    elseif mode == "multiply" then
        love.graphics.setBlendMode("multiply")
    else
        love.graphics.setBlendMode("alpha")
    end
end

-- Reset graphics state
function Renderer:reset()
    love.graphics.setColor(1, 1, 1, 1)
    love.graphics.setLineWidth(self.lineWidth)
    love.graphics.setBlendMode("alpha")
end

return Renderer
