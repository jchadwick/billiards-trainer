-- Network Module
-- Provides network communication abstraction layer for the visualizer
-- Manages WebSocket connections and message routing to visualization modules

local WebSocket = require("modules.network.websocket")
local Connection = require("modules.network.connection")
local json = require("lib.json")

local Network = {
    name = "network",
    priority = 0,  -- Highest priority (processes first)
    enabled = true,

    -- Network configuration (will be loaded from Config module)
    config = {
        host = "localhost",
        port = 8080,
        protocol = "ws",
        reconnectInterval = 5.0,  -- Seconds between reconnection attempts
        heartbeatInterval = 30.0,  -- Seconds between heartbeat pings
        connectionTimeout = 10.0,  -- Seconds before connection timeout
        autoConnect = false,  -- Auto-connect on init
    },

    -- Internal state
    websocket = nil,
    connection = nil,
    messageQueue = {},
    messageHandlers = {},
    messageHandler = nil,  -- Reference to core MessageHandler
}

--- Initialize the network module
-- Sets up WebSocket client and connection state management
function Network:init()
    print("Network module initializing...")

    -- Load configuration from global Config module
    if _G.Config then
        local config_instance = _G.Config

        -- Get websocket URL (e.g., "ws://localhost:8000/api/v1/ws")
        local websocket_url = config_instance:get("network.websocket_url")
        if websocket_url then
            -- Store full URL, will be parsed by WebSocket module
            self.config.websocket_url = websocket_url
            print("Loaded WebSocket URL: " .. websocket_url)
        end

        -- Load other network settings
        self.config.autoConnect = config_instance:get("network.auto_connect") or false
        self.config.reconnectInterval = config_instance:get("network.reconnect_delay") or 5000
        self.config.heartbeatInterval = config_instance:get("network.heartbeat_interval") or 30000

        print("Network configuration loaded from Config module")
    else
        print("Warning: Global Config module not found, using defaults")
    end

    -- Get reference to global MessageHandler
    if _G.MessageHandler then
        self.messageHandler = _G.MessageHandler
        print("Connected to global MessageHandler")
    else
        print("Warning: Global MessageHandler not found, message routing will be limited")
    end

    -- Initialize connection manager with loaded config
    self.connection = Connection.new({
        reconnectInterval = self.config.reconnectInterval,
        heartbeatInterval = self.config.heartbeatInterval,
        connectionTimeout = self.config.connectionTimeout,
        autoReconnect = _G.Config and _G.Config:get("network.reconnect_enabled") or true,
    })

    -- Initialize WebSocket client with callbacks
    local ws_config = {
        onmessage = function(message) self:handleMessage(message) end,
        onopen = function() self:onConnect() end,
        onclose = function(code, reason) self:onDisconnect(reason) end,
        onerror = function(error) self:onError(error) end,
        reconnect_enabled = _G.Config and _G.Config:get("network.reconnect_enabled") or true,
        reconnect_delay = self.config.reconnectInterval or 1000,  -- milliseconds
        max_reconnect_delay = _G.Config and _G.Config:get("network.max_reconnect_delay") or 30000,
        max_reconnect_attempts = 0,  -- Unlimited
    }

    -- Use full URL if available, otherwise use host/port/path
    if self.config.websocket_url then
        ws_config.url = self.config.websocket_url
    else
        ws_config.host = self.config.host
        ws_config.port = self.config.port
        ws_config.path = "/"
    end

    self.websocket = WebSocket.new(ws_config)

    -- Initialize statistics tracking
    self.stats = {
        messagesReceived = 0,
        messagesByType = {},
        lastMessageTime = nil,
    }

    -- Auto-connect if configured
    if self.config.autoConnect then
        print("Auto-connect enabled, connecting...")
        self:connect()
    end

    print("Network module initialized")
end

--- Handle incoming messages from the network
-- Routes messages to appropriate handlers and modules
-- @param message string|table Either JSON string or already parsed message object
function Network:handleMessage(message)
    -- Parse JSON if needed
    local parsedMessage = message
    if type(message) == "string" then
        -- Try to parse JSON
        local success, decoded = pcall(json.decode, message)
        if not success then
            print("Error: Failed to parse JSON message: " .. tostring(decoded))
            if self.connection then
                self.connection:onError("JSON parse error")
            end
            return
        end
        parsedMessage = decoded
    end

    -- Validate message structure
    if not parsedMessage or type(parsedMessage) ~= "table" then
        print("Warning: Invalid message received (not a table)")
        return
    end

    if not parsedMessage.type then
        print("Warning: Invalid message received (missing type field)")
        return
    end

    -- Update connection state (mark activity)
    if self.connection then
        self.connection:onMessageReceived()
    end

    -- Track message statistics
    if not self.stats then
        self.stats = {
            messagesReceived = 0,
            messagesByType = {},
            lastMessageTime = nil,
        }
    end
    self.stats.messagesReceived = self.stats.messagesReceived + 1
    self.stats.lastMessageTime = love.timer.getTime()

    local msgType = parsedMessage.type
    if not self.stats.messagesByType[msgType] then
        self.stats.messagesByType[msgType] = 0
    end
    self.stats.messagesByType[msgType] = self.stats.messagesByType[msgType] + 1

    -- First, pass to core MessageHandler if available
    -- This handles state, motion, trajectory messages through the standard pipeline
    if self.messageHandler and type(message) == "string" then
        -- MessageHandler expects JSON string
        local success, err = pcall(function()
            self.messageHandler:handleMessage(message)
        end)
        if not success then
            print("Error in MessageHandler: " .. tostring(err))
        end
    end

    -- Also route to any registered handlers for this message type
    -- This allows modules to subscribe to specific message types
    if self.messageHandlers[msgType] then
        for _, handler in ipairs(self.messageHandlers[msgType]) do
            local success, err = pcall(handler, parsedMessage.data)
            if not success then
                print(string.format("Error in handler for message type '%s': %s", msgType, tostring(err)))
            end
        end
    end
end

--- Register a message handler for a specific message type
-- @param messageType string The type of message to handle
-- @param handler function The handler function(data)
function Network:registerHandler(messageType, handler)
    if not self.messageHandlers[messageType] then
        self.messageHandlers[messageType] = {}
    end
    table.insert(self.messageHandlers[messageType], handler)
end

--- Unregister a message handler
-- @param messageType string The type of message
-- @param handler function The handler function to remove
function Network:unregisterHandler(messageType, handler)
    if not self.messageHandlers[messageType] then return end

    for i, h in ipairs(self.messageHandlers[messageType]) do
        if h == handler then
            table.remove(self.messageHandlers[messageType], i)
            return
        end
    end
end

--- Send a message through the network
-- @param messageType string The type of message
-- @param data table The message payload
-- @return boolean Success status
function Network:send(messageType, data)
    if not self.websocket or not self.connection:isConnected() then
        print("Warning: Cannot send message - not connected")
        table.insert(self.messageQueue, {type = messageType, data = data})
        return false
    end

    local message = {
        type = messageType,
        data = data,
        timestamp = love.timer.getTime()
    }

    return self.websocket:send(message)

    -- TODO: Implement message queuing for offline mode
    -- TODO: Add message acknowledgment/confirmation
    -- TODO: Implement retry logic for failed sends
end

--- Connect to the server
-- @return boolean Success status
function Network:connect()
    if not self.websocket then
        print("Error: WebSocket not initialized")
        return false
    end

    print(string.format("Connecting to %s://%s:%d",
        self.config.protocol, self.config.host, self.config.port))

    return self.websocket:connect()
end

--- Disconnect from the server
function Network:disconnect()
    if self.websocket then
        self.websocket:disconnect()
    end
end

--- Callback when connection is established
function Network:onConnect()
    print("Network connected")

    if self.connection then
        self.connection:onConnected()
    end

    -- Send queued messages
    self:flushMessageQueue()

    -- TODO: Implement connection handshake/authentication
    -- TODO: Send initial state sync request
end

--- Callback when connection is lost
-- @param reason string The disconnection reason
function Network:onDisconnect(reason)
    print("Network disconnected: " .. (reason or "unknown"))

    if self.connection then
        self.connection:onDisconnected(reason)
    end

    -- TODO: Implement automatic reconnection logic
    -- TODO: Clear or persist message queue based on config
end

--- Callback when network error occurs
-- @param error string The error message
function Network:onError(error)
    print("Network error: " .. (error or "unknown"))

    if self.connection then
        self.connection:onError(error)
    end

    -- TODO: Implement error recovery strategies
    -- TODO: Add error logging and reporting
end

--- Flush queued messages
function Network:flushMessageQueue()
    if #self.messageQueue == 0 then return end

    print(string.format("Flushing %d queued messages", #self.messageQueue))

    for _, message in ipairs(self.messageQueue) do
        self:send(message.type, message.data)
    end

    self.messageQueue = {}
end

--- Update the network module (called every frame)
-- @param dt number Delta time in seconds
function Network:update(dt)
    -- Update WebSocket
    if self.websocket then
        self.websocket:update(dt)
    end

    -- Update connection state
    if self.connection then
        self.connection:update(dt)

        -- Handle reconnection logic
        if self.connection:shouldReconnect() then
            self:connect()
        end
    end

    -- TODO: Implement bandwidth monitoring
    -- TODO: Add latency tracking
    -- TODO: Implement message rate limiting
end

--- Draw network status (for debugging)
function Network:draw()
    -- This module typically doesn't draw anything
    -- Drawing is handled by diagnostic_hud or other UI modules

    -- TODO: Consider adding debug overlay showing connection status
end

--- Cleanup the network module
function Network:cleanup()
    print("Network module cleaning up")

    -- Disconnect cleanly
    self:disconnect()

    -- Clear handlers and queues
    self.messageHandlers = {}
    self.messageQueue = {}

    print("Network module cleaned up")
end

--- Configure the network module
-- @param config table Configuration options
function Network:configure(config)
    -- Handle websocket_url format (e.g., "ws://localhost:8080")
    if config.websocket_url then
        local protocol, host, port = config.websocket_url:match("^(%w+)://([^:]+):(%d+)$")
        if protocol and host and port then
            self.config.protocol = protocol
            self.config.host = host
            self.config.port = tonumber(port)
            print(string.format("Updated WebSocket config: %s://%s:%d", protocol, host, port))
        else
            print("Warning: Invalid websocket_url format: " .. config.websocket_url)
        end
    end

    -- Handle individual configuration options
    if config.host then self.config.host = config.host end
    if config.port then self.config.port = config.port end
    if config.protocol then self.config.protocol = config.protocol end
    if config.reconnectInterval then self.config.reconnectInterval = config.reconnectInterval end
    if config.heartbeatInterval then self.config.heartbeatInterval = config.heartbeatInterval end
    if config.connectionTimeout then self.config.connectionTimeout = config.connectionTimeout end
    if config.autoConnect ~= nil then self.config.autoConnect = config.autoConnect end

    -- Reconfigure connection manager if already initialized
    if self.connection then
        self.connection:configure({
            reconnectInterval = self.config.reconnectInterval,
            heartbeatInterval = self.config.heartbeatInterval,
            connectionTimeout = self.config.connectionTimeout,
        })
    end

    -- Note: WebSocket reconfiguration requires reconnection
    -- If WebSocket is already initialized and connected, we need to disconnect and reconnect
    if self.websocket and self.websocket:isConnected() then
        print("Warning: WebSocket configuration changed, reconnection required")
        self:disconnect()
        self.websocket = nil
        -- Re-initialize with new config on next connect() call
    end

    print("Network configuration updated")
end

--- Get network status information
-- @return table Status information
function Network:getStatus()
    local status = {
        connected = false,
        host = self.config.host,
        port = self.config.port,
        queuedMessages = #self.messageQueue,
        messages_received = 0,
        messagesByType = {},
        lastMessageTime = nil,
    }

    -- Add connection state
    if self.connection then
        status.connected = self.connection:isConnected()
        status.state = self.connection:getState()
        status.uptime = self.connection:getUptime()
        status.lastActivity = self.connection:getLastActivity()
    end

    -- Add message statistics
    if self.stats then
        status.messages_received = self.stats.messagesReceived or 0
        status.messagesByType = self.stats.messagesByType or {}
        status.lastMessageTime = self.stats.lastMessageTime
    end

    return status
end

return Network
