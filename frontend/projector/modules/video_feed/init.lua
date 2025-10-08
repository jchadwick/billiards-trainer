-- Video Feed Module
-- Displays live video feed from the backend as a background layer

local VideoFeed = {
    name = "video_feed",
    priority = 1,  -- Background layer (lowest z-index)
    enabled = false,  -- Disabled by default

    -- Configuration
    opacity = 50,  -- 0-100 opacity percentage

    -- Frame data
    currentFrame = nil,
    currentImage = nil,
    frameTimestamp = nil,
    frameSequence = 0,

    -- Frame metrics
    lastFrameTime = 0,
    frameCount = 0,
    fps = 0,
    fpsUpdateTimer = 0,
    frameLatency = 0,

    -- Settings
    maxFrameAge = 2.0,  -- Maximum age of frame in seconds before discarding
    fpsUpdateInterval = 1.0,  -- Update FPS counter every second

    -- Transform cache
    transformMatrix = nil,
    lastTransformUpdate = 0,
    transformUpdateInterval = 0.1,  -- Update transform every 100ms
}

-- Initialize module
function VideoFeed:init()
    print("Video Feed module initialized")
    print("  Priority: " .. self.priority .. " (background layer)")
    print("  Opacity: " .. self.opacity .. "%")
    print("  Max frame age: " .. self.maxFrameAge .. "s")
end

-- Handle incoming messages
function VideoFeed:onMessage(messageType, data)
    if messageType == "frame" then
        self:processFrame(data)
    end
end

-- Process incoming frame data
function VideoFeed:processFrame(data)
    if not data or not data.image then
        return
    end

    local success, result = pcall(function()
        -- Decode base64 image data
        local imageData = love.data.decode("string", "base64", data.image)

        if not imageData then
            error("Failed to decode base64 image data")
        end

        -- Create LÖVE ImageData from decoded data
        local format = data.format or "jpeg"
        local imageDataObj

        if format == "jpeg" or format == "jpg" then
            -- For JPEG, we need to use love.image.newImageData
            local fileData = love.filesystem.newFileData(imageData, "frame.jpg")
            imageDataObj = love.image.newImageData(fileData)
        elseif format == "png" then
            -- For PNG, same approach
            local fileData = love.filesystem.newFileData(imageData, "frame.png")
            imageDataObj = love.image.newImageData(fileData)
        else
            error("Unsupported image format: " .. format)
        end

        -- Create LÖVE Image object from ImageData
        local image = love.graphics.newImage(imageDataObj)

        -- Release old image to free memory
        if self.currentImage then
            self.currentImage:release()
        end

        -- Store new frame data
        self.currentFrame = data
        self.currentImage = image
        self.frameTimestamp = love.timer.getTime()
        self.frameSequence = data.sequence or 0

        -- Calculate frame latency if timestamp is provided
        if data.timestamp then
            -- Parse ISO 8601 timestamp
            local now = os.time()
            -- Simple latency tracking (would need proper ISO parsing for accuracy)
            self.frameLatency = 0  -- Placeholder
        end

        -- Update FPS counter
        self.frameCount = self.frameCount + 1

        return true
    end)

    if not success then
        print("Error processing frame: " .. tostring(result))
    end
end

-- Update module (called every frame)
function VideoFeed:update(dt)
    -- Update FPS counter
    self.fpsUpdateTimer = self.fpsUpdateTimer + dt
    if self.fpsUpdateTimer >= self.fpsUpdateInterval then
        self.fps = self.frameCount / self.fpsUpdateTimer
        self.frameCount = 0
        self.fpsUpdateTimer = 0
    end

    -- Check if current frame is too old
    if self.currentFrame and self.frameTimestamp then
        local age = love.timer.getTime() - self.frameTimestamp
        if age > self.maxFrameAge then
            -- Clear old frame
            if self.currentImage then
                self.currentImage:release()
                self.currentImage = nil
            end
            self.currentFrame = nil
            self.frameTimestamp = nil
        end
    end

    -- Update transform matrix periodically
    local currentTime = love.timer.getTime()
    if currentTime - self.lastTransformUpdate > self.transformUpdateInterval then
        self:updateTransform()
        self.lastTransformUpdate = currentTime
    end
end

-- Update transformation matrix
function VideoFeed:updateTransform()
    local Calibration = _G.Calibration
    if not Calibration then
        return
    end

    self.transformMatrix = Calibration:getTransform()
end

-- Calculate scaling to fit projection area while maintaining aspect ratio
function VideoFeed:calculateScale()
    if not self.currentImage then
        return 1, 1, 0, 0
    end

    local Calibration = _G.Calibration
    if not Calibration or not Calibration.corners then
        return 1, 1, 0, 0
    end

    local sw, sh = love.graphics.getDimensions()

    -- Get calibration area dimensions
    local x1 = Calibration.corners[1].x * sw
    local y1 = Calibration.corners[1].y * sh
    local x2 = Calibration.corners[2].x * sw
    local y2 = Calibration.corners[3].y * sh

    local calibWidth = math.abs(x2 - x1)
    local calibHeight = math.abs(y2 - y1)

    -- Get image dimensions
    local imgWidth = self.currentImage:getWidth()
    local imgHeight = self.currentImage:getHeight()

    -- Calculate scale to fit while maintaining aspect ratio
    local scaleX = calibWidth / imgWidth
    local scaleY = calibHeight / imgHeight
    local scale = math.min(scaleX, scaleY)

    -- Calculate centering offset
    local scaledWidth = imgWidth * scale
    local scaledHeight = imgHeight * scale
    local offsetX = x1 + (calibWidth - scaledWidth) / 2
    local offsetY = y1 + (calibHeight - scaledHeight) / 2

    return scale, scale, offsetX, offsetY
end

-- Draw module
function VideoFeed:draw()
    if not self.currentImage or not self.currentFrame then
        return
    end

    local Calibration = _G.Calibration
    if not Calibration then
        return
    end

    local success, result = pcall(function()
        -- Save graphics state
        love.graphics.push()
        love.graphics.origin()

        -- Calculate opacity (0-100 to 0-1)
        local alpha = self.opacity / 100

        -- Get calibration corners
        local sw, sh = love.graphics.getDimensions()
        local x1 = Calibration.corners[1].x * sw
        local y1 = Calibration.corners[1].y * sh
        local x2 = Calibration.corners[2].x * sw
        local y3 = Calibration.corners[3].y * sh

        local calibWidth = math.abs(x2 - x1)
        local calibHeight = math.abs(y3 - y1)

        -- Get image dimensions
        local imgWidth = self.currentImage:getWidth()
        local imgHeight = self.currentImage:getHeight()

        -- Calculate scale to fit while maintaining aspect ratio
        local scaleX = calibWidth / imgWidth
        local scaleY = calibHeight / imgHeight
        local scale = math.min(scaleX, scaleY)

        -- Calculate centering offset
        local scaledWidth = imgWidth * scale
        local scaledHeight = imgHeight * scale
        local offsetX = x1 + (calibWidth - scaledWidth) / 2
        local offsetY = y1 + (calibHeight - scaledHeight) / 2

        -- Apply opacity
        love.graphics.setColor(1, 1, 1, alpha)

        -- Draw the image
        love.graphics.draw(
            self.currentImage,
            offsetX,
            offsetY,
            0,  -- rotation
            scale,
            scale
        )

        -- Reset color
        love.graphics.setColor(1, 1, 1, 1)

        -- Restore graphics state
        love.graphics.pop()
    end)

    if not success then
        print("Error drawing video feed: " .. tostring(result))
    end
end

-- Cleanup module
function VideoFeed:cleanup()
    -- Release image resources
    if self.currentImage then
        self.currentImage:release()
        self.currentImage = nil
    end

    self.currentFrame = nil
    self.frameTimestamp = nil

    print("Video Feed module cleaned up")
end

-- Module configuration
function VideoFeed:configure(config)
    if config.enabled ~= nil then
        self.enabled = config.enabled
    end

    if config.opacity then
        self.opacity = math.max(0, math.min(100, config.opacity))
    end

    if config.priority then
        self.priority = config.priority
    end

    if config.maxFrameAge then
        self.maxFrameAge = config.maxFrameAge
    end
end

-- Get module status
function VideoFeed:getStatus()
    return {
        enabled = self.enabled,
        hasFrame = self.currentImage ~= nil,
        frameSequence = self.frameSequence,
        fps = math.floor(self.fps * 10) / 10,
        frameAge = self.frameTimestamp and (love.timer.getTime() - self.frameTimestamp) or 0,
        opacity = self.opacity,
    }
end

return VideoFeed
