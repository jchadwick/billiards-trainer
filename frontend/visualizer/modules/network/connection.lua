-- Connection State Manager
-- Manages connection lifecycle, reconnection logic, and health monitoring
-- Tracks connection state, uptime, and activity for network reliability

local Connection = {}
Connection.__index = Connection

-- Connection states
Connection.States = {
    DISCONNECTED = "disconnected",
    CONNECTING = "connecting",
    CONNECTED = "connected",
    RECONNECTING = "reconnecting",
    ERROR = "error",
}

-- Valid state transitions
-- Defines which state transitions are allowed to prevent invalid state changes
local VALID_TRANSITIONS = {
    disconnected = {connecting = true, error = true},
    connecting = {connected = true, error = true, disconnected = true},
    connected = {disconnected = true, reconnecting = true, error = true},
    reconnecting = {connecting = true, connected = true, error = true, disconnected = true},
    error = {disconnected = true, reconnecting = true, connecting = true},
}

--- Create a new Connection state manager
-- @param config table Configuration options
--   - reconnectInitialDelay: number Initial delay before first reconnection attempt (default: 1.0s)
--   - reconnectMaxDelay: number Maximum delay between reconnection attempts (default: 30.0s)
--   - reconnectBackoffFactor: number Exponential backoff multiplier (default: 2.0)
--   - maxReconnectAttempts: number Maximum reconnection attempts, 0 = unlimited (default: 0)
--   - heartbeatTimeout: number Seconds before considering connection dead (default: 60.0s)
--   - heartbeatInterval: number Seconds between heartbeat checks (default: 30.0s)
--   - enablePingPong: boolean Whether to send ping/expect pong frames (default: false)
--   - autoReconnect: boolean Whether to automatically reconnect on disconnect (default: true)
--   - onStateChange: function Callback for state changes function(newState, oldState, reason)
--   - onReconnecting: function Callback when reconnection starts function(attempt, delay)
-- @return table Connection instance
function Connection.new(config)
    config = config or {}
    local self = setmetatable({}, Connection)

    -- Reconnection configuration with exponential backoff
    self.reconnectInitialDelay = config.reconnectInitialDelay or 1.0
    self.reconnectMaxDelay = config.reconnectMaxDelay or 30.0
    self.reconnectBackoffFactor = config.reconnectBackoffFactor or 2.0
    self.maxReconnectAttempts = config.maxReconnectAttempts or 0  -- 0 = infinite
    self.autoReconnect = config.autoReconnect ~= false  -- Default true

    -- Heartbeat configuration
    self.heartbeatTimeout = config.heartbeatTimeout or 60.0
    self.heartbeatInterval = config.heartbeatInterval or 30.0
    self.enablePingPong = config.enablePingPong or false

    -- State callbacks
    self.onStateChangeCallback = config.onStateChange
    self.onReconnectingCallback = config.onReconnecting

    -- Current state
    self.state = Connection.States.DISCONNECTED
    self.previousState = nil

    -- Error tracking
    self.lastError = nil
    self.lastDisconnectReason = nil

    -- Timing
    self.connectedAt = nil
    self.disconnectedAt = nil
    self.lastActivityAt = nil
    self.lastHeartbeatSentAt = nil
    self.lastPongReceivedAt = nil

    -- Timers
    self.reconnectTimer = 0
    self.heartbeatCheckTimer = 0
    self.currentReconnectDelay = self.reconnectInitialDelay

    -- Reconnection state
    self.reconnectAttempts = 0
    self.isReconnecting = false

    -- Statistics tracking
    self.stats = {
        totalConnections = 0,
        totalSuccessfulConnections = 0,
        totalFailedConnections = 0,
        totalDisconnects = 0,
        totalReconnects = 0,
        totalErrors = 0,
        totalUptime = 0,
        totalConnectionTime = 0,  -- Cumulative time connected across all connections
        lastDisconnectReason = nil,
        lastError = nil,
    }

    return self
end

--- Set the connection state with validation
-- Validates state transitions and triggers callbacks
-- @param newState string The new state to transition to
-- @param reason string Optional reason for the state change
-- @return boolean True if state change was successful
function Connection:setState(newState, reason)
    -- Validate state exists
    local validState = false
    for _, state in pairs(Connection.States) do
        if state == newState then
            validState = true
            break
        end
    end

    if not validState then
        print(string.format("Warning: Invalid state '%s'", tostring(newState)))
        return false
    end

    -- Check if transition is valid
    local currentState = self.state
    if currentState ~= newState then
        local transitions = VALID_TRANSITIONS[currentState]
        if not transitions or not transitions[newState] then
            print(string.format("Warning: Invalid state transition from '%s' to '%s'",
                currentState, newState))
            return false
        end
    end

    -- Perform state transition
    local oldState = self.state
    self.previousState = oldState
    self.state = newState

    -- Log state change
    if oldState ~= newState then
        local reasonStr = reason and (" - " .. reason) or ""
        print(string.format("Connection state: %s -> %s%s", oldState, newState, reasonStr))

        -- Trigger state change callback
        if self.onStateChangeCallback then
            self.onStateChangeCallback(newState, oldState, reason)
        end
    end

    return true
end

--- Update the connection state (called every frame)
-- Handles heartbeat monitoring and reconnection timing
-- @param dt number Delta time in seconds
function Connection:update(dt)
    -- Update connection uptime and total connection time
    if self.state == Connection.States.CONNECTED then
        self.stats.totalUptime = self.stats.totalUptime + dt
        self.stats.totalConnectionTime = self.stats.totalConnectionTime + dt

        -- Update heartbeat check timer
        self.heartbeatCheckTimer = self.heartbeatCheckTimer + dt

        -- Check for heartbeat timeout (inactivity)
        if self.lastActivityAt then
            local timeSinceActivity = love.timer.getTime() - self.lastActivityAt
            if timeSinceActivity >= self.heartbeatTimeout then
                print(string.format("Heartbeat timeout: %.1fs since last activity", timeSinceActivity))
                self:onDisconnected("heartbeat timeout")
                return
            end
        end

        -- Periodic heartbeat check
        if self.heartbeatCheckTimer >= self.heartbeatInterval then
            self:checkHeartbeat()
            self.heartbeatCheckTimer = 0
        end
    end

    -- Update reconnection timer
    if self.state == Connection.States.RECONNECTING then
        self.reconnectTimer = self.reconnectTimer + dt
    end
end

--- Callback when connection is established
-- Called when a connection is successfully established
function Connection:onConnected()
    -- Update state with validation
    self:setState(Connection.States.CONNECTED, "connection established")

    -- Record connection time
    local now = love.timer.getTime()
    self.connectedAt = now
    self.lastActivityAt = now
    self.lastHeartbeatSentAt = now
    self.lastPongReceivedAt = now

    -- Update statistics
    self.stats.totalConnections = self.stats.totalConnections + 1
    self.stats.totalSuccessfulConnections = self.stats.totalSuccessfulConnections + 1

    -- Reset reconnection state on successful connection
    if self.reconnectAttempts > 0 then
        print(string.format("Successfully reconnected after %d attempt(s)", self.reconnectAttempts))
    end
    self.reconnectAttempts = 0
    self.reconnectTimer = 0
    self.currentReconnectDelay = self.reconnectInitialDelay
    self.isReconnecting = false

    print(string.format("Connection established (total connections: %d)",
        self.stats.totalSuccessfulConnections))
end

--- Callback when connection is lost
-- Handles disconnection and initiates reconnection if enabled
-- @param reason string Disconnection reason
function Connection:onDisconnected(reason)
    -- Don't re-process if already disconnected
    if self.state == Connection.States.DISCONNECTED then
        return
    end

    -- Record disconnect time and reason
    self.disconnectedAt = love.timer.getTime()
    self.lastDisconnectReason = reason or "unknown"

    -- Update statistics
    self.stats.totalDisconnects = self.stats.totalDisconnects + 1
    self.stats.lastDisconnectReason = self.lastDisconnectReason

    -- Update state with validation
    self:setState(Connection.States.DISCONNECTED, self.lastDisconnectReason)

    print(string.format("Connection lost: %s (total disconnects: %d)",
        self.lastDisconnectReason, self.stats.totalDisconnects))

    -- Start reconnection if enabled
    if self.autoReconnect then
        self:startReconnect()
    end
end

--- Callback when an error occurs
-- Handles connection errors and tracks error statistics
-- @param error string Error message
function Connection:onError(error)
    -- Record error
    self.lastError = error or "unknown error"
    self.stats.totalErrors = self.stats.totalErrors + 1
    self.stats.lastError = self.lastError

    -- Only transition to error state if not already disconnected/reconnecting
    if self.state == Connection.States.CONNECTED or
       self.state == Connection.States.CONNECTING then
        self:setState(Connection.States.ERROR, self.lastError)
    end

    -- Track failed connection attempts
    if self.state == Connection.States.CONNECTING or
       self.state == Connection.States.RECONNECTING then
        self.stats.totalFailedConnections = self.stats.totalFailedConnections + 1
    end

    print(string.format("Connection error: %s (total errors: %d)",
        self.lastError, self.stats.totalErrors))

    -- Treat errors during connection attempts as disconnection
    if self.state == Connection.States.CONNECTING or
       self.state == Connection.States.RECONNECTING then
        if self.autoReconnect then
            self:startReconnect()
        else
            self:setState(Connection.States.DISCONNECTED, "error: " .. self.lastError)
        end
    end
end

--- Callback when a message is received
-- Updates activity timestamp to prevent timeout
function Connection:onMessageReceived()
    self.lastActivityAt = love.timer.getTime()
end

--- Callback when a pong frame is received
-- Used for ping/pong heartbeat mechanism
function Connection:onPongReceived()
    self.lastPongReceivedAt = love.timer.getTime()
    self.lastActivityAt = self.lastPongReceivedAt
end

--- Calculate exponential backoff delay
-- Formula: min(initial * factor^attempts, max_delay)
-- @return number Delay in seconds before next reconnection attempt
function Connection:calculateReconnectDelay()
    local delay = self.reconnectInitialDelay * math.pow(self.reconnectBackoffFactor, self.reconnectAttempts)
    return math.min(delay, self.reconnectMaxDelay)
end

--- Start reconnection process with exponential backoff
-- Initiates reconnection attempt or schedules next attempt
function Connection:startReconnect()
    -- Check max reconnect attempts (0 = unlimited)
    if self.maxReconnectAttempts > 0 and
       self.reconnectAttempts >= self.maxReconnectAttempts then
        print(string.format("Max reconnection attempts reached (%d/%d)",
            self.reconnectAttempts, self.maxReconnectAttempts))
        self:setState(Connection.States.DISCONNECTED, "max reconnection attempts reached")
        return
    end

    -- Transition to reconnecting state
    if not self:setState(Connection.States.RECONNECTING, "starting reconnection") then
        return
    end

    -- Increment attempt counter and calculate delay
    self.reconnectAttempts = self.reconnectAttempts + 1
    self.currentReconnectDelay = self:calculateReconnectDelay()
    self.reconnectTimer = 0
    self.isReconnecting = true

    -- Update statistics
    self.stats.totalReconnects = self.stats.totalReconnects + 1

    print(string.format("Reconnection attempt %d scheduled in %.1fs (backoff: %.1fx)",
        self.reconnectAttempts,
        self.currentReconnectDelay,
        self.reconnectBackoffFactor))

    -- Trigger callback
    if self.onReconnectingCallback then
        self.onReconnectingCallback(self.reconnectAttempts, self.currentReconnectDelay)
    end
end

--- Check if should attempt reconnection
-- Returns true when enough time has passed for the next reconnection attempt
-- @return boolean True if should reconnect now
function Connection:shouldReconnect()
    if self.state ~= Connection.States.RECONNECTING then
        return false
    end

    -- Check if enough time has passed since last attempt
    return self.reconnectTimer >= self.currentReconnectDelay
end

--- Check heartbeat/keepalive
-- Monitors connection health and optionally sends ping frames
-- @param websocket table Optional WebSocket instance for sending pings
function Connection:checkHeartbeat(websocket)
    if not self.lastActivityAt then
        return
    end

    local now = love.timer.getTime()
    local timeSinceActivity = now - self.lastActivityAt

    -- Log heartbeat check
    print(string.format("Heartbeat check: %.1fs since last activity", timeSinceActivity))

    -- Send ping if ping/pong is enabled and we have a websocket
    if self.enablePingPong and websocket then
        -- Check if we should send a ping
        local timeSinceLastPing = self.lastHeartbeatSentAt and (now - self.lastHeartbeatSentAt) or math.huge

        if timeSinceLastPing >= self.heartbeatInterval then
            print("Sending heartbeat ping")
            if websocket:ping() then
                self.lastHeartbeatSentAt = now
            else
                print("Warning: Failed to send ping")
            end
        end

        -- Check if we haven't received a pong recently
        if self.lastPongReceivedAt then
            local timeSinceLastPong = now - self.lastPongReceivedAt
            if timeSinceLastPong >= self.heartbeatTimeout then
                print(string.format("Pong timeout: %.1fs since last pong", timeSinceLastPong))
                self:onDisconnected("pong timeout")
            end
        end
    end
end

--- Check if currently connected
-- @return boolean Connection status
function Connection:isConnected()
    return self.state == Connection.States.CONNECTED
end

--- Get current connection state
-- @return string Connection state
function Connection:getState()
    return self.state
end

--- Get connection uptime
-- @return number Uptime in seconds (or nil if not connected)
function Connection:getUptime()
    if not self.connectedAt then
        return nil
    end

    if self.state == Connection.States.CONNECTED then
        return love.timer.getTime() - self.connectedAt
    else
        return (self.disconnectedAt or love.timer.getTime()) - self.connectedAt
    end
end

--- Get time since last activity
-- @return number Seconds since last activity (or nil if never connected)
function Connection:getLastActivity()
    if not self.lastActivityAt then
        return nil
    end

    return love.timer.getTime() - self.lastActivityAt
end

--- Get comprehensive statistics
-- @return table Statistics data including uptime, attempts, and error tracking
function Connection:getStats()
    return {
        -- Connection counts
        totalConnections = self.stats.totalConnections,
        totalSuccessfulConnections = self.stats.totalSuccessfulConnections,
        totalFailedConnections = self.stats.totalFailedConnections,
        totalDisconnects = self.stats.totalDisconnects,
        totalReconnects = self.stats.totalReconnects,
        totalErrors = self.stats.totalErrors,

        -- Time tracking
        totalUptime = self.stats.totalUptime,
        totalConnectionTime = self.stats.totalConnectionTime,
        currentUptime = self:getUptime(),
        lastActivity = self:getLastActivity(),

        -- Current state
        reconnectAttempts = self.reconnectAttempts,
        currentReconnectDelay = self.currentReconnectDelay,
        state = self.state,
        previousState = self.previousState,

        -- Error tracking
        lastDisconnectReason = self.stats.lastDisconnectReason,
        lastError = self.stats.lastError,
    }
end

--- Reset statistics and reconnection state
-- Clears all counters and tracking data
function Connection:reset()
    self.stats = {
        totalConnections = 0,
        totalSuccessfulConnections = 0,
        totalFailedConnections = 0,
        totalDisconnects = 0,
        totalReconnects = 0,
        totalErrors = 0,
        totalUptime = 0,
        totalConnectionTime = 0,
        lastDisconnectReason = nil,
        lastError = nil,
    }

    -- Reset reconnection state
    self.reconnectAttempts = 0
    self.currentReconnectDelay = self.reconnectInitialDelay
    self.isReconnecting = false

    -- Clear error tracking
    self.lastError = nil
    self.lastDisconnectReason = nil

    print("Connection statistics and state reset")
end

--- Configure the connection manager
-- Updates configuration parameters at runtime
-- @param config table Configuration options
function Connection:configure(config)
    if config.reconnectInitialDelay then
        self.reconnectInitialDelay = config.reconnectInitialDelay
    end
    if config.reconnectMaxDelay then
        self.reconnectMaxDelay = config.reconnectMaxDelay
    end
    if config.reconnectBackoffFactor then
        self.reconnectBackoffFactor = config.reconnectBackoffFactor
    end
    if config.maxReconnectAttempts ~= nil then
        self.maxReconnectAttempts = config.maxReconnectAttempts
    end
    if config.heartbeatTimeout then
        self.heartbeatTimeout = config.heartbeatTimeout
    end
    if config.heartbeatInterval then
        self.heartbeatInterval = config.heartbeatInterval
    end
    if config.enablePingPong ~= nil then
        self.enablePingPong = config.enablePingPong
    end
    if config.autoReconnect ~= nil then
        self.autoReconnect = config.autoReconnect
    end
    if config.onStateChange then
        self.onStateChangeCallback = config.onStateChange
    end
    if config.onReconnecting then
        self.onReconnectingCallback = config.onReconnecting
    end

    -- Legacy compatibility
    if config.reconnectInterval then
        self.reconnectInitialDelay = config.reconnectInterval
    end
    if config.connectionTimeout then
        self.heartbeatTimeout = config.connectionTimeout
    end
end

--- Enable auto-reconnect
function Connection:enableAutoReconnect()
    self.autoReconnect = true
end

--- Disable auto-reconnect
function Connection:disableAutoReconnect()
    self.autoReconnect = false
end

--- Reset reconnection attempts counter
-- Resets attempts and reconnect delay to initial values
function Connection:resetReconnectAttempts()
    self.reconnectAttempts = 0
    self.currentReconnectDelay = self.reconnectInitialDelay
    self.isReconnecting = false
    print("Reconnection attempts reset")
end

return Connection
