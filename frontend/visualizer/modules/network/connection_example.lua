-- Example usage of Connection state manager
-- This file demonstrates how to use the Connection class for managing
-- WebSocket connection state, reconnection, and health monitoring

local Connection = require("modules.network.connection")

-- Example 1: Basic connection with default settings
local function example_basic()
    local connection = Connection.new({})

    -- Connection starts in DISCONNECTED state
    print("Initial state:", connection:getState())

    -- Simulate connection established
    connection:onConnected()
    print("State after connect:", connection:getState())

    -- Simulate receiving a message (updates activity timestamp)
    connection:onMessageReceived()

    -- Check connection status
    print("Is connected?", connection:isConnected())
    print("Uptime:", connection:getUptime(), "seconds")
end

-- Example 2: Connection with exponential backoff reconnection
local function example_reconnection()
    local connection = Connection.new({
        reconnectInitialDelay = 1.0,     -- Start with 1 second delay
        reconnectMaxDelay = 30.0,         -- Cap at 30 seconds
        reconnectBackoffFactor = 2.0,     -- Double delay each attempt
        maxReconnectAttempts = 5,         -- Try 5 times before giving up
        autoReconnect = true,             -- Enable auto-reconnect
    })

    -- Simulate connection and disconnection
    connection:onConnected()
    connection:onDisconnected("network error")

    -- Check if we should reconnect
    if connection:shouldReconnect() then
        print("Attempting reconnection...")
    end

    -- Get reconnection delay
    local delay = connection:calculateReconnectDelay()
    print("Next reconnect in:", delay, "seconds")
end

-- Example 3: Connection with heartbeat monitoring and ping/pong
local function example_heartbeat(websocket)
    local connection = Connection.new({
        heartbeatTimeout = 60.0,      -- Consider dead after 60s of inactivity
        heartbeatInterval = 30.0,     -- Check every 30s
        enablePingPong = true,        -- Enable ping/pong frames
    })

    -- Simulate connection
    connection:onConnected()

    -- Update loop (called every frame)
    local dt = 1.0  -- 1 second frame
    connection:update(dt)

    -- Perform heartbeat check (optionally passing websocket for ping)
    connection:checkHeartbeat(websocket)

    -- When pong is received
    connection:onPongReceived()
end

-- Example 4: Connection with state change callbacks
local function example_callbacks()
    local connection = Connection.new({
        -- State change callback
        onStateChange = function(newState, oldState, reason)
            print(string.format("State changed: %s -> %s (%s)",
                oldState, newState, reason or ""))
        end,

        -- Reconnection callback
        onReconnecting = function(attempt, delay)
            print(string.format("Reconnecting... attempt %d in %.1fs",
                attempt, delay))
        end,
    })

    -- State changes will trigger callbacks
    connection:onConnected()
    connection:onDisconnected("timeout")
end

-- Example 5: Connection statistics tracking
local function example_statistics()
    local connection = Connection.new({})

    -- Simulate some activity
    connection:onConnected()
    connection:onMessageReceived()
    connection:onDisconnected("error")

    -- Get comprehensive statistics
    local stats = connection:getStats()
    print("Total connections:", stats.totalSuccessfulConnections)
    print("Total disconnects:", stats.totalDisconnects)
    print("Total errors:", stats.totalErrors)
    print("Total uptime:", stats.totalUptime, "seconds")
    print("Total connection time:", stats.totalConnectionTime, "seconds")
    print("Last disconnect reason:", stats.lastDisconnectReason)

    -- Reset statistics
    connection:reset()
end

-- Example 6: Dynamic configuration
local function example_configuration()
    local connection = Connection.new({})

    -- Update configuration at runtime
    connection:configure({
        reconnectInitialDelay = 2.0,
        heartbeatTimeout = 120.0,
        enablePingPong = true,
        autoReconnect = false,
    })

    -- Enable/disable auto-reconnect
    connection:enableAutoReconnect()
    connection:disableAutoReconnect()

    -- Reset reconnection attempts
    connection:resetReconnectAttempts()
end

-- Example 7: Integration with Network module
local function example_integration()
    local connection = Connection.new({
        reconnectInitialDelay = 1.0,
        reconnectMaxDelay = 30.0,
        heartbeatTimeout = 60.0,
        heartbeatInterval = 30.0,
        autoReconnect = true,
    })

    -- In your update loop:
    local function update(dt)
        -- Update connection state
        connection:update(dt)

        -- Check if reconnection is needed
        if connection:shouldReconnect() then
            -- Call your websocket connect function
            -- websocket:connect()
            print("Attempting reconnection...")
        end

        -- Periodic heartbeat check
        -- connection:checkHeartbeat(websocket)
    end

    -- Handle WebSocket callbacks:
    local function onWebSocketOpen()
        connection:onConnected()
    end

    local function onWebSocketClose(reason)
        connection:onDisconnected(reason)
    end

    local function onWebSocketError(error)
        connection:onError(error)
    end

    local function onWebSocketMessage(message)
        connection:onMessageReceived()
    end
end

-- Export examples (for documentation/testing)
return {
    basic = example_basic,
    reconnection = example_reconnection,
    heartbeat = example_heartbeat,
    callbacks = example_callbacks,
    statistics = example_statistics,
    configuration = example_configuration,
    integration = example_integration,
}
