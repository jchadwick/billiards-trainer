-- Chessboard Calibration Module
-- Displays a chessboard pattern for camera calibration
-- Can be triggered via network message: {"type": "show_calibration_chessboard", "data": {...}}

local ChessboardCalibration = {
    name = "chessboard_calibration",
    priority = 1000,  -- Draw on top of everything
    enabled = false,

    -- Chessboard configuration
    rows = 9,  -- Default OpenCV calibration: 9 rows (8 internal corners)
    cols = 6,  -- Default OpenCV calibration: 6 cols (5 internal corners)
    squareSize = 80,  -- Pixel size of each square

    -- Display state
    visible = false,
    centered = true,  -- Whether to center the chessboard on screen
    offsetX = 0,
    offsetY = 0,

    -- Colors
    color1 = {0, 0, 0, 1},  -- Black
    color2 = {1, 1, 1, 1},  -- White

    -- Border/margin
    margin = 40
}

-- Initialize module
function ChessboardCalibration:init()
    -- Register network message handler
    if _G.Network then
        _G.Network:registerCallback("show_calibration_chessboard", function(data)
            self:handleShowChessboard(data)
        end)

        _G.Network:registerCallback("hide_calibration_chessboard", function(data)
            self:handleHideChessboard(data)
        end)
    end

    print("Chessboard Calibration module initialized")
end

-- Handle show chessboard command
function ChessboardCalibration:handleShowChessboard(data)
    print("Showing calibration chessboard")

    -- Parse configuration if provided
    if data then
        if data.rows then self.rows = data.rows end
        if data.cols then self.cols = data.cols end
        if data.squareSize then self.squareSize = data.squareSize end
        if data.centered ~= nil then self.centered = data.centered end
        if data.offsetX then self.offsetX = data.offsetX end
        if data.offsetY then self.offsetY = data.offsetY end
    end

    self.visible = true
    self.enabled = true

    print(string.format("Chessboard config: %dx%d grid, %dpx squares",
        self.rows, self.cols, self.squareSize))
end

-- Handle hide chessboard command
function ChessboardCalibration:handleHideChessboard(data)
    print("Hiding calibration chessboard")
    self.visible = false
    self.enabled = false
end

-- Update module
function ChessboardCalibration:update(dt)
    -- Nothing to update
end

-- Draw chessboard pattern
function ChessboardCalibration:draw()
    if not self.visible then return end

    -- Get screen dimensions
    local screenWidth = love.graphics.getWidth()
    local screenHeight = love.graphics.getHeight()

    -- Calculate chessboard dimensions
    local boardWidth = self.cols * self.squareSize
    local boardHeight = self.rows * self.squareSize

    -- Calculate position
    local startX, startY
    if self.centered then
        startX = (screenWidth - boardWidth) / 2 + self.offsetX
        startY = (screenHeight - boardHeight) / 2 + self.offsetY
    else
        startX = self.margin + self.offsetX
        startY = self.margin + self.offsetY
    end

    -- Draw chessboard
    for row = 0, self.rows - 1 do
        for col = 0, self.cols - 1 do
            -- Determine square color (alternating pattern)
            local isWhite = (row + col) % 2 == 0
            local color = isWhite and self.color2 or self.color1

            love.graphics.setColor(color)

            local x = startX + col * self.squareSize
            local y = startY + row * self.squareSize

            love.graphics.rectangle("fill", x, y, self.squareSize, self.squareSize)
        end
    end

    -- Draw info overlay
    self:drawInfo(startX, startY, boardWidth, boardHeight)
end

-- Draw information overlay
function ChessboardCalibration:drawInfo(startX, startY, boardWidth, boardHeight)
    love.graphics.push()
    love.graphics.origin()

    -- Background for info
    love.graphics.setColor(0, 0, 0, 0.8)
    love.graphics.rectangle("fill", 10, 10, 400, 100)

    -- Info text
    love.graphics.setColor(0, 1, 0, 1)
    love.graphics.print("CALIBRATION CHESSBOARD", 20, 20)

    love.graphics.setColor(1, 1, 1, 1)
    love.graphics.print(string.format("Grid: %d x %d", self.rows, self.cols), 20, 40)
    love.graphics.print(string.format("Square Size: %d px", self.squareSize), 20, 60)
    love.graphics.print(string.format("Position: (%.0f, %.0f)", startX, startY), 20, 80)

    -- Draw corner markers for easier identification
    self:drawCornerMarkers(startX, startY, boardWidth, boardHeight)

    love.graphics.pop()
end

-- Draw corner markers to help with identification
function ChessboardCalibration:drawCornerMarkers(startX, startY, boardWidth, boardHeight)
    local markerSize = 20
    local markerColor = {1, 0, 0, 1}  -- Red

    love.graphics.setColor(markerColor)
    love.graphics.setLineWidth(3)

    -- Draw cross markers at corners
    local corners = {
        {x = startX, y = startY},  -- Top-left
        {x = startX + boardWidth, y = startY},  -- Top-right
        {x = startX, y = startY + boardHeight},  -- Bottom-left
        {x = startX + boardWidth, y = startY + boardHeight}  -- Bottom-right
    }

    for _, corner in ipairs(corners) do
        -- Draw + marker
        love.graphics.line(
            corner.x - markerSize, corner.y,
            corner.x + markerSize, corner.y
        )
        love.graphics.line(
            corner.x, corner.y - markerSize,
            corner.x, corner.y + markerSize
        )
    end

    love.graphics.setLineWidth(1)
end

-- Handle network messages
function ChessboardCalibration:onMessage(messageType, data)
    if messageType == "show_calibration_chessboard" then
        self:handleShowChessboard(data)
    elseif messageType == "hide_calibration_chessboard" then
        self:handleHideChessboard(data)
    end
end

-- Handle key press
function ChessboardCalibration:onKeyPressed(key)
    if not self.visible then return end

    -- Allow hiding with escape
    if key == "escape" or key == "c" then
        self:handleHideChessboard()
    end

    -- Allow adjusting position with arrow keys
    if key == "left" then
        self.offsetX = self.offsetX - 10
    elseif key == "right" then
        self.offsetX = self.offsetX + 10
    elseif key == "up" then
        self.offsetY = self.offsetY - 10
    elseif key == "down" then
        self.offsetY = self.offsetY + 10
    elseif key == "r" then
        -- Reset position
        self.offsetX = 0
        self.offsetY = 0
    elseif key == "space" then
        -- Toggle centered mode
        self.centered = not self.centered
    end
end

-- Cleanup
function ChessboardCalibration:cleanup()
    self.visible = false
    self.enabled = false
end

return ChessboardCalibration
