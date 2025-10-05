-- Network Module
-- Handles UDP and WebSocket communication with backend

local socket = require "socket"  -- Built into LÖVE2D
local json = require "lib.json"  -- We'll add this library

local Network = {
    udp = nil,
    udpPort = 9999,
    udpHost = "127.0.0.1",  -- Listen on localhost for testing
    websocket = nil,
    websocketUrl = "ws://localhost:8000/ws",
    status = {
        udp = false,
        websocket = false,
        udpPort = 9999,
        lastMessage = nil,
        messagesReceived = 0
    },
    messageQueue = {},
    callbacks = {}
}

-- Initialize network connections
function Network:init()
    -- Initialize UDP socket
    self.udp = socket.udp()

    if not self.udp then
        print("Failed to create UDP socket")
        self.status.udp = false
        return
    end

    local success, err = self.udp:setsockname(self.udpHost, self.udpPort)

    if not success then
        print(string.format("Failed to bind UDP socket: %s", tostring(err)))
        self.status.udp = false
        return
    end

    self.udp:settimeout(0)  -- Non-blocking, critical for game loop
    self.status.udp = true

    -- Get actual bound address
    local ip, port = self.udp:getsockname()
    print(string.format("UDP socket bound to %s:%d", ip, port))

    -- WebSocket initialization would go here
    -- For now, we'll focus on UDP which is simpler and lower latency
    self.status.websocket = false
end

-- Update network (called every frame)
function Network:update(dt)
    if not self.udp then return end

    -- Receive all pending UDP messages
    local data, ip, port = self.udp:receivefrom()

    if data then
        -- Process message
        local success, msg = pcall(json.decode, data)
        if success and msg then
            self:processMessage(msg, ip, port)

            -- Check for more messages
            local moreData, moreIp, morePort = self.udp:receivefrom()
            while moreData do
                local moreSuccess, moreMsg = pcall(json.decode, moreData)
                if moreSuccess and moreMsg then
                    self:processMessage(moreMsg, moreIp, morePort)
                else
                    print(string.format("Invalid JSON from %s:%d: %s", moreIp or "unknown", morePort or 0, moreData))
                end
                moreData, moreIp, morePort = self.udp:receivefrom()
            end
        else
            print(string.format("Invalid JSON from %s:%d: %s", ip or "unknown", port or 0, data))
        end
    end

    -- Process message queue
    self:processQueue(dt)
end

-- Process incoming message
function Network:processMessage(msg, ip, port)
    self.status.lastMessage = love.timer.getTime()
    self.status.messagesReceived = self.status.messagesReceived + 1

    -- Validate message structure
    if type(msg) ~= "table" or not msg.type then
        return
    end

    -- Add to queue with metadata
    table.insert(self.messageQueue, {
        message = msg,
        timestamp = love.timer.getTime(),
        ip = ip,
        port = port
    })
end

-- Process message queue
function Network:processQueue(dt)
    while #self.messageQueue > 0 do
        local item = table.remove(self.messageQueue, 1)
        local msg = item.message

        -- Send to module manager
        if _G.ModuleManager then
            _G.ModuleManager:sendMessage(msg.type, msg.data)
        end

        -- Call any registered callbacks
        if self.callbacks[msg.type] then
            for _, callback in ipairs(self.callbacks[msg.type]) do
                local success, err = pcall(callback, msg.data)
                if not success then
                    print(string.format("Callback error for %s: %s", msg.type, err))
                end
            end
        end
    end
end

-- Send UDP message (for responses if needed)
function Network:sendUDP(targetIP, targetPort, messageType, data)
    if not self.udp then return false end

    local msg = {
        type = messageType,
        timestamp = love.timer.getTime(),
        data = data
    }

    local jsonStr = json.encode(msg)
    local bytes, err = self.udp:sendto(jsonStr, targetIP, targetPort)

    if not bytes then
        print(string.format("UDP send error: %s", err))
        return false
    end

    return true
end

-- Register callback for message type
function Network:registerCallback(messageType, callback)
    if not self.callbacks[messageType] then
        self.callbacks[messageType] = {}
    end
    table.insert(self.callbacks[messageType], callback)
end

-- Get network status
function Network:getStatus()
    return self.status
end

-- Cleanup network connections
function Network:cleanup()
    if self.udp then
        self.udp:close()
        self.udp = nil
    end

    if self.websocket then
        -- Close WebSocket
        self.websocket = nil
    end

    self.status.udp = false
    self.status.websocket = false
end

return Network
