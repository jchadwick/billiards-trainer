-- WebSocket Client
-- Provides WebSocket communication for real-time bidirectional data transfer
-- Handles connection lifecycle, message encoding/decoding, error handling, and auto-reconnect
-- Implements exponential backoff reconnection strategy (FR-VIS-040A)

local websocket_lib = require("lib.websocket")
local json = require("lib.json")

local WebSocket = {}
WebSocket.__index = WebSocket

-- Connection states
local STATE = {
    DISCONNECTED = "DISCONNECTED",
    CONNECTING = "CONNECTING",
    CONNECTED = "CONNECTED",
    RECONNECTING = "RECONNECTING",
    ERROR = "ERROR",
}

--- Parse WebSocket URL (ws://host:port/path)
-- @param url string The WebSocket URL
-- @return table|nil Parsed components (host, port, path) or nil on error
local function parseURL(url)
    if not url then return nil end

    -- Match ws://host:port/path
    local protocol, host, port, path = url:match("^(ws)://([^:]+):(%d+)(.*)$")
    if protocol and host and port then
        return {
            host = host,
            port = tonumber(port),
            path = path ~= "" and path or "/",
        }
    end

    -- Match ws://host/path (default port 80)
    protocol, host, path = url:match("^(ws)://([^/]+)(.*)$")
    if protocol and host then
        return {
            host = host,
            port = 80,
            path = path ~= "" and path or "/",
        }
    end

    return nil
end

--- Create a new WebSocket client
-- @param config table Configuration options
--   - url: string WebSocket URL (ws://host:port/path)
--   OR
--   - host: string Server hostname
--   - port: number Server port
--   - path: string Server path (default: "/")
--
--   - onopen: function Callback when connected
--   - onmessage: function Callback for incoming messages (receives decoded JSON table)
--   - onerror: function Callback for errors
--   - onclose: function Callback when disconnected
--
--   - reconnect_enabled: boolean Enable auto-reconnect (default: true)
--   - reconnect_delay: number Initial reconnect delay in ms (default: 1000)
--   - max_reconnect_delay: number Maximum reconnect delay in ms (default: 30000)
--   - max_reconnect_attempts: number Max reconnect attempts, 0 = unlimited (default: 0)
-- @return table WebSocket instance
function WebSocket.new(config)
    local self = setmetatable({}, WebSocket)

    -- Parse URL or use host/port/path
    local parsed
    if config.url then
        parsed = parseURL(config.url)
        if not parsed then
            error("Invalid WebSocket URL: " .. config.url)
        end
    else
        parsed = {
            host = config.host or "localhost",
            port = config.port or 8080,
            path = config.path or "/",
        }
    end

    -- Connection configuration
    self.host = parsed.host
    self.port = parsed.port
    self.path = parsed.path

    -- Callbacks
    self.onopen = config.onopen or function() end
    self.onmessage = config.onmessage or function() end
    self.onerror = config.onerror or function() end
    self.onclose = config.onclose or function() end

    -- Reconnect configuration
    self.reconnect_enabled = config.reconnect_enabled ~= false  -- default true
    self.reconnect_delay = config.reconnect_delay or 1000  -- 1 second
    self.max_reconnect_delay = config.max_reconnect_delay or 30000  -- 30 seconds
    self.max_reconnect_attempts = config.max_reconnect_attempts or 0  -- unlimited

    -- State
    self.state = STATE.DISCONNECTED
    self.client = nil
    self.manual_disconnect = false

    -- Reconnect state
    self.current_reconnect_delay = self.reconnect_delay
    self.reconnect_timer = 0
    self.reconnect_attempts = 0

    -- Statistics
    self.stats = {
        messages_sent = 0,
        messages_received = 0,
        bytes_sent = 0,
        bytes_received = 0,
        connection_time = 0,
        uptime = 0,
        reconnect_attempts = 0,
    }

    return self
end

--- Connect to the WebSocket server
-- @return boolean True if connection attempt started
function WebSocket:connect()
    if self.state == STATE.CONNECTED or self.state == STATE.CONNECTING then
        print("[WebSocket] Warning: Already connected or connecting")
        return false
    end

    self.state = STATE.CONNECTING
    self.manual_disconnect = false

    -- Create underlying websocket client
    self.client = websocket_lib.new(self.host, self.port, self.path)

    -- Set up callbacks on the underlying client
    local original_onopen = self.client.onopen
    self.client.onopen = function(client)
        self:_handleOpen()
        if original_onopen then original_onopen(client) end
    end

    local original_onmessage = self.client.onmessage
    self.client.onmessage = function(client, message)
        self:_handleMessage(message)
        if original_onmessage then original_onmessage(client, message) end
    end

    local original_onerror = self.client.onerror
    self.client.onerror = function(client, err)
        self:_handleError(err)
        if original_onerror then original_onerror(client, err) end
    end

    local original_onclose = self.client.onclose
    self.client.onclose = function(client, code, reason)
        self:_handleClose(code, reason)
        if original_onclose then original_onclose(client, code, reason) end
    end

    print(string.format("[WebSocket] Connecting to ws://%s:%d%s", self.host, self.port, self.path))
    return true
end

--- Disconnect from the WebSocket server
-- @param code number Optional close code (default: 1000)
-- @param reason string Optional close reason
function WebSocket:disconnect(code, reason)
    if self.state == STATE.DISCONNECTED then
        return
    end

    self.manual_disconnect = true

    if self.client then
        self.client:close(code or 1000, reason or "client disconnect")
    end

    self.state = STATE.DISCONNECTED
    self.client = nil

    print("[WebSocket] Disconnected")
end

--- Send a message through the WebSocket
-- @param message table The message to send (will be JSON encoded)
-- @return boolean Success status
function WebSocket:send(message)
    if self.state ~= STATE.CONNECTED then
        print("[WebSocket] Warning: Cannot send - not connected")
        return false
    end

    if not self.client then
        print("[WebSocket] Warning: Cannot send - no client")
        return false
    end

    -- Encode message as JSON
    local ok, encoded = pcall(json.encode, message)
    if not ok then
        print("[WebSocket] Error encoding message: " .. tostring(encoded))
        self.onerror("JSON encoding failed: " .. tostring(encoded))
        return false
    end

    -- Send through underlying client
    local success, err = pcall(function()
        self.client:send(encoded)
    end)

    if not success then
        print("[WebSocket] Error sending message: " .. tostring(err))
        self.onerror("Send failed: " .. tostring(err))
        return false
    end

    -- Update statistics
    self.stats.messages_sent = self.stats.messages_sent + 1
    self.stats.bytes_sent = self.stats.bytes_sent + #encoded

    return true
end

--- Update the WebSocket (called every frame)
-- @param dt number Delta time in seconds
function WebSocket:update(dt)
    -- Update connection uptime
    if self.state == STATE.CONNECTED then
        self.stats.uptime = self.stats.uptime + dt
    end

    -- Update underlying client if connected
    if self.client then
        local success, err = pcall(function()
            self.client:update()
        end)

        if not success then
            print("[WebSocket] Error in client update: " .. tostring(err))
            self:_handleError(tostring(err))
        end
    end

    -- Handle reconnection logic
    if self.state == STATE.RECONNECTING then
        self.reconnect_timer = self.reconnect_timer + dt

        if self.reconnect_timer >= (self.current_reconnect_delay / 1000) then
            self.reconnect_timer = 0
            self:_attemptReconnect()
        end
    end
end

--- Get current connection state
-- @return string Current state (DISCONNECTED, CONNECTING, CONNECTED, RECONNECTING, ERROR)
function WebSocket:getState()
    return self.state
end

--- Check if connected
-- @return boolean Connection status
function WebSocket:isConnected()
    return self.state == STATE.CONNECTED
end

--- Get connection URL
-- @return string The WebSocket URL
function WebSocket:getURL()
    return string.format("ws://%s:%d%s", self.host, self.port, self.path)
end

--- Get statistics
-- @return table Statistics data
function WebSocket:getStats()
    return {
        messages_sent = self.stats.messages_sent,
        messages_received = self.stats.messages_received,
        bytes_sent = self.stats.bytes_sent,
        bytes_received = self.stats.bytes_received,
        connection_time = self.stats.connection_time,
        uptime = self.stats.uptime,
        reconnect_attempts = self.stats.reconnect_attempts,
    }
end

--- Reset statistics
function WebSocket:resetStats()
    self.stats = {
        messages_sent = 0,
        messages_received = 0,
        bytes_sent = 0,
        bytes_received = 0,
        connection_time = 0,
        uptime = 0,
        reconnect_attempts = 0,
    }
end

--- Internal: Handle connection open
function WebSocket:_handleOpen()
    self.state = STATE.CONNECTED
    self.stats.connection_time = love.timer.getTime()

    -- Reset reconnect parameters on successful connection
    self.current_reconnect_delay = self.reconnect_delay
    self.reconnect_attempts = 0

    print("[WebSocket] Connected to " .. self:getURL())
    self.onopen()
end

--- Internal: Handle incoming message
-- @param message string Raw message (JSON string)
function WebSocket:_handleMessage(message)
    -- Update statistics
    self.stats.messages_received = self.stats.messages_received + 1
    self.stats.bytes_received = self.stats.bytes_received + #message

    -- Decode JSON
    local ok, decoded = pcall(json.decode, message)
    if not ok then
        print("[WebSocket] Error decoding message: " .. tostring(decoded))
        self.onerror("JSON decoding failed: " .. tostring(decoded))
        return
    end

    -- Call user callback with decoded message
    self.onmessage(decoded)
end

--- Internal: Handle error
-- @param err string Error message
function WebSocket:_handleError(err)
    print("[WebSocket] Error: " .. tostring(err))
    self.state = STATE.ERROR
    self.onerror(err)

    -- Trigger reconnect if enabled and not manually disconnected
    if self.reconnect_enabled and not self.manual_disconnect then
        self:_scheduleReconnect()
    end
end

--- Internal: Handle connection close
-- @param code number Close code
-- @param reason string Close reason
function WebSocket:_handleClose(code, reason)
    print(string.format("[WebSocket] Closed: code=%d, reason=%s", code or 0, reason or ""))

    local was_connected = (self.state == STATE.CONNECTED)
    self.state = STATE.DISCONNECTED
    self.client = nil

    -- Call user callback
    self.onclose(code, reason)

    -- Trigger reconnect if enabled and not manually disconnected
    if self.reconnect_enabled and not self.manual_disconnect and was_connected then
        self:_scheduleReconnect()
    end
end

--- Internal: Schedule a reconnection attempt
function WebSocket:_scheduleReconnect()
    -- Check if we've exceeded max attempts
    if self.max_reconnect_attempts > 0 and self.reconnect_attempts >= self.max_reconnect_attempts then
        print(string.format("[WebSocket] Max reconnect attempts (%d) reached", self.max_reconnect_attempts))
        self.state = STATE.ERROR
        self.onerror("Max reconnect attempts reached")
        return
    end

    self.state = STATE.RECONNECTING
    self.reconnect_timer = 0

    print(string.format("[WebSocket] Reconnecting in %.1f seconds (attempt %d)",
        self.current_reconnect_delay / 1000, self.reconnect_attempts + 1))
end

--- Internal: Attempt to reconnect
function WebSocket:_attemptReconnect()
    self.reconnect_attempts = self.reconnect_attempts + 1
    self.stats.reconnect_attempts = self.stats.reconnect_attempts + 1

    print(string.format("[WebSocket] Reconnect attempt %d", self.reconnect_attempts))

    -- Exponential backoff: double the delay, up to max
    self.current_reconnect_delay = math.min(
        self.current_reconnect_delay * 2,
        self.max_reconnect_delay
    )

    -- Attempt connection
    self:connect()
end

return WebSocket
