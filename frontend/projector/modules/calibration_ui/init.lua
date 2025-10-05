-- Calibration UI Module
-- Interactive calibration interface

local CalibrationUI = {
    name = "calibration_ui",
    priority = 500,  -- Draw on top of everything
    enabled = false,  -- Only enabled during calibration

    -- UI state
    active = false,
    selectedCorner = 1,

    -- Visual settings
    colors = {
        grid = {0.3, 0.3, 0.3, 0.5},
        outline = {0, 1, 0, 0.8},
        corner = {0, 1, 0, 1},
        selectedCorner = {1, 1, 0, 1},
        text = {1, 1, 1, 1},
        help = {0.8, 0.8, 0.8, 0.8}
    }
}

-- Initialize module
function CalibrationUI:init()
    print("Calibration UI module initialized")
end

-- Handle calibration start
function CalibrationUI:onCalibrationStart()
    self.enabled = true
    self.active = true
    print("Calibration UI activated")
end

-- Handle calibration end
function CalibrationUI:onCalibrationEnd()
    self.enabled = false
    self.active = false
    print("Calibration UI deactivated")
end

-- Update module
function CalibrationUI:update(dt)
    -- Nothing to update
end

-- Draw calibration UI
function CalibrationUI:draw()
    if not self.active then return end

    local Calibration = _G.Calibration
    if not Calibration then return end

    -- Draw calibration overlay
    Calibration:drawOverlay()

    -- Draw help text
    self:drawHelpText()

    -- Draw corner coordinates
    self:drawCornerInfo()
end

-- Draw help text
function CalibrationUI:drawHelpText()
    love.graphics.push()
    love.graphics.origin()

    -- Background for help text
    love.graphics.setColor(0, 0, 0, 0.7)
    love.graphics.rectangle("fill", 10, love.graphics.getHeight() - 200, 400, 190)

    -- Help text
    love.graphics.setColor(self.colors.help)
    local helpText = {
        "CALIBRATION MODE",
        "",
        "Arrow Keys: Move selected corner",
        "Shift+Arrow: Move faster (10px)",
        "Tab: Select next corner",
        "Mouse: Click to select, drag to move",
        "R: Reset to default",
        "C: Save and exit",
        "",
        "Align the green outline with the table edges"
    }

    local y = love.graphics.getHeight() - 190
    for _, line in ipairs(helpText) do
        love.graphics.print(line, 20, y)
        y = y + 16
    end

    love.graphics.pop()
end

-- Draw corner information
function CalibrationUI:drawCornerInfo()
    local Calibration = _G.Calibration
    if not Calibration or not Calibration.corners then return end

    love.graphics.push()
    love.graphics.origin()

    -- Background for corner info
    love.graphics.setColor(0, 0, 0, 0.7)
    love.graphics.rectangle("fill", love.graphics.getWidth() - 250, 10, 240, 140)

    -- Corner coordinates
    love.graphics.setColor(self.colors.text)
    love.graphics.print("CORNER POSITIONS", love.graphics.getWidth() - 240, 20)

    local cornerNames = {"Top-Left", "Top-Right", "Bottom-Right", "Bottom-Left"}
    local sw, sh = love.graphics.getDimensions()

    for i, corner in ipairs(Calibration.corners) do
        local y = 40 + (i - 1) * 25

        if i == Calibration.selectedCorner then
            love.graphics.setColor(self.colors.selectedCorner)
            love.graphics.print("> ", love.graphics.getWidth() - 240, y)
        else
            love.graphics.setColor(self.colors.text)
        end

        local text = string.format("%d. %s: (%d, %d)",
            i,
            cornerNames[i],
            math.floor(corner.x * sw),
            math.floor(corner.y * sh)
        )
        love.graphics.print(text, love.graphics.getWidth() - 225, y)
    end

    love.graphics.pop()
end

-- Handle key press
function CalibrationUI:onKeyPressed(key)
    if not self.active then return end

    local Calibration = _G.Calibration
    if not Calibration then return end

    -- Already handled by main.lua, but we can add module-specific handling here
    if key == "space" then
        -- Quick test: draw test pattern
        self:toggleTestPattern()
    end
end

-- Toggle test pattern
function CalibrationUI:toggleTestPattern()
    -- This would show a test pattern to verify calibration
    print("Test pattern toggled")
end

-- Cleanup
function CalibrationUI:cleanup()
    self.active = false
    self.enabled = false
end

return CalibrationUI
