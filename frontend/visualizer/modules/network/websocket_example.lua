-- WebSocket Client Usage Example
-- Demonstrates all features of the WebSocket implementation

local WebSocket = require("modules.network.websocket")

-- Example 1: Basic Usage
print("=== Example 1: Basic Usage ===")

local ws1 = WebSocket.new({
    url = "ws://localhost:8000/api/v1/websocket/ws",

    onopen = function()
        print("[ws1] Connected!")
    end,

    onmessage = function(message)
        print("[ws1] Received message:")
        print("  Type:", message.type)
        print("  Data:", message.data)
    end,

    onerror = function(err)
        print("[ws1] Error:", err)
    end,

    onclose = function(code, reason)
        print("[ws1] Closed:", code, reason or "")
    end,
})

-- Connect to server
ws1:connect()

print()

-- Example 2: Host/Port/Path Configuration
print("=== Example 2: Host/Port/Path Configuration ===")

local ws2 = WebSocket.new({
    host = "127.0.0.1",
    port = 9000,
    path = "/api/v1/websocket/ws",

    onopen = function()
        print("[ws2] Connected to", ws2:getURL())
    end,

    onmessage = function(message)
        print("[ws2] Message:", message.type)
    end,
})

print()

-- Example 3: Custom Reconnection Settings
print("=== Example 3: Custom Reconnection Settings ===")

local ws3 = WebSocket.new({
    url = "ws://localhost:8000/api/v1/websocket/ws",

    -- Start with 2 second delay
    reconnect_delay = 2000,

    -- Cap at 1 minute
    max_reconnect_delay = 60000,

    -- Try maximum 10 times
    max_reconnect_attempts = 10,

    onopen = function()
        print("[ws3] Connected - reconnect attempts will reset")
    end,

    onerror = function(err)
        print("[ws3] Error:", err)
        print("[ws3] Current state:", ws3:getState())
    end,
})

print()

-- Example 4: Disabling Auto-Reconnect
print("=== Example 4: Disabling Auto-Reconnect ===")

local ws4 = WebSocket.new({
    url = "ws://localhost:8000/api/v1/websocket/ws",
    reconnect_enabled = false,

    onclose = function(code, reason)
        print("[ws4] Closed and will NOT auto-reconnect")
    end,
})

print()

-- Example 5: Sending Messages
print("=== Example 5: Sending Messages ===")

local ws5
ws5 = WebSocket.new({
    url = "ws://localhost:8000/api/v1/websocket/ws",

    onopen = function()
        print("[ws5] Connected, sending messages...")

        -- Image points below are 1080p camera pixels. Each per-axis scale maps
        -- them to canonical 4K pixels; the space tag names the source domain.
        ws5:send({
            type = "state",
            data = {
                coordinate_space = "pixel",
                balls = {
                    {
                        id = 1,
                        position = {x = 480, y = 270, scale = {2, 2},
                            coordinate_space = "pixel"},
                        velocity = {x = 0, y = 0, scale = {2, 2},
                            coordinate_space = "pixel"}
                    },
                    {
                        id = 2,
                        position = {x = 1440, y = 540, scale = {2, 2},
                            coordinate_space = "pixel"},
                        velocity = {x = 0, y = 0, scale = {2, 2},
                            coordinate_space = "pixel"}
                    }
                },
                table = {
                    coordinate_space = "pixel",
                    width = 3072,
                    height = 1536,
                    dimensions_unit = "canonical_pixels",
                    dimensions_scale = {1, 1},
                    playing_area_corners = {
                        {x = 192, y = 156, scale = {2, 2},
                            coordinate_space = "pixel"},
                        {x = 1728, y = 156, scale = {2, 2},
                            coordinate_space = "pixel"},
                        {x = 1728, y = 924, scale = {2, 2},
                            coordinate_space = "pixel"},
                        {x = 192, y = 924, scale = {2, 2},
                            coordinate_space = "pixel"}
                    },
                    bounds = {
                        x = 192, y = 156, width = 1536, height = 768,
                        scale = {2, 2}, coordinate_space = "pixel"
                    },
                    pocket_positions = {
                        {x = 192, y = 156, scale = {2, 2},
                            coordinate_space = "pixel"},
                        {x = 1728, y = 924, scale = {2, 2},
                            coordinate_space = "pixel"}
                    },
                    real_world_dimensions = {
                        width = 2.1, height = 1.05, unit = "meters"
                    }
                }
            }
        })

        -- Send a trajectory update with the same declared source image space.
        ws5:send({
            type = "trajectory",
            data = {
                coordinate_space = "pixel",
                line_count = 2,
                collision_count = 1,
                lines = {
                    {
                        start = {x = 480, y = 270, scale = {2, 2},
                            coordinate_space = "pixel"},
                        ["end"] = {x = 960, y = 540, scale = {2, 2},
                            coordinate_space = "pixel"},
                        type = "primary",
                        ball_id = "1"
                    },
                    {
                        start = {x = 960, y = 540, scale = {2, 2},
                            coordinate_space = "pixel"},
                        ["end"] = {x = 1440, y = 270, scale = {2, 2},
                            coordinate_space = "pixel"},
                        type = "primary",
                        ball_id = "1"
                    }
                },
                collisions = {{
                    position = {x = 960, y = 540, scale = {2, 2},
                        coordinate_space = "pixel"},
                    ball1_id = "1",
                    ball2_id = "2",
                    type = "ball",
                    angle = 0
                }}
            }
        })

        -- Send a configuration change
        ws5:send({
            type = "config",
            data = {
                debug_hud = {enabled = true}
            }
        })
    end,
})
ws5:connect()

print()

-- Example 6: Statistics Tracking
print("=== Example 6: Statistics Tracking ===")

local ws6
ws6 = WebSocket.new({
    url = "ws://localhost:8000/api/v1/websocket/ws",

    onopen = function()
        -- Send some test messages
        for i = 1, 5 do
            ws6:send({type = "test", data = {count = i}})
        end

        -- Print statistics
        local stats = ws6:getStats()
        print("[ws6] Statistics:")
        print("  Messages sent:", stats.messages_sent)
        print("  Messages received:", stats.messages_received)
        print("  Bytes sent:", stats.bytes_sent)
        print("  Bytes received:", stats.bytes_received)
        print("  Uptime:", stats.uptime, "seconds")
        print("  Reconnect attempts:", stats.reconnect_attempts)
    end,

    onmessage = function(message)
        -- Print updated statistics after each message
        local stats = ws6:getStats()
        print("[ws6] Received message #" .. stats.messages_received)
    end,
})
ws6:connect()

print()

-- Example 7: State Management
print("=== Example 7: State Management ===")

local ws7 = WebSocket.new({
    url = "ws://localhost:8000/api/v1/websocket/ws",
})

-- Check state before connection
print("[ws7] Initial state:", ws7:getState())
print("[ws7] Is connected?", ws7:isConnected())

-- Connect
ws7:connect()
print("[ws7] After connect():", ws7:getState())

-- Manually disconnect
-- ws7:disconnect(1000, "Example finished")
-- print("[ws7] After disconnect():", ws7:getState())

print()

-- Example 8: Integration Pattern (typical usage in modules)
print("=== Example 8: Integration Pattern ===")

local NetworkModule = {
    websocket = nil,
    messageHandlers = {},
}

function NetworkModule:init()
    self.websocket = WebSocket.new({
        url = "ws://localhost:8000/api/v1/websocket/ws",

        onopen = function()
            self:onConnect()
        end,

        onmessage = function(message)
            self:handleMessage(message)
        end,

        onerror = function(err)
            self:onError(err)
        end,

        onclose = function(code, reason)
            self:onDisconnect(code, reason)
        end,

        reconnect_enabled = true,
        reconnect_delay = 1000,
        max_reconnect_delay = 30000,
    })

    self.websocket:connect()
    print("[Module] WebSocket initialized")
end

function NetworkModule:onConnect()
    print("[Module] Connected to server")

    -- Send initial handshake
    self.websocket:send({
        type = "handshake",
        data = {
            client = "visualizer",
            version = "1.0.0"
        }
    })
end

function NetworkModule:handleMessage(message)
    print("[Module] Handling message:", message.type)

    -- Route to appropriate handler
    if self.messageHandlers[message.type] then
        self.messageHandlers[message.type](message.data)
    end
end

function NetworkModule:onError(err)
    print("[Module] Error:", err)
end

function NetworkModule:onDisconnect(code, reason)
    print("[Module] Disconnected:", code, reason or "")
end

function NetworkModule:registerHandler(messageType, handler)
    self.messageHandlers[messageType] = handler
end

function NetworkModule:update(dt)
    if self.websocket then
        self.websocket:update(dt)
    end
end

-- Initialize the module
NetworkModule:init()

-- Register a handler
NetworkModule:registerHandler("state", function(data)
    print("[Handler] Processing state update")
end)

print()

-- Example 9: Error Handling
print("=== Example 9: Error Handling ===")

local ws9 = WebSocket.new({
    url = "ws://invalid-host:9999",

    onerror = function(err)
        print("[ws9] Handling error:", err)

        -- Common errors and how to handle them:
        if string.find(err, "connection refused") then
            print("[ws9] Server is not running")
        elseif string.find(err, "timeout") then
            print("[ws9] Connection timed out")
        elseif string.find(err, "JSON") then
            print("[ws9] Message encoding/decoding error")
        elseif string.find(err, "Max reconnect attempts") then
            print("[ws9] Giving up after too many attempts")
        end
    end,
})

-- This will fail and trigger the error handler
ws9:connect()

print()
print("=== Examples Complete ===")
print()
print("Note: Most examples require a WebSocket server running.")
print("Connect to ws://localhost:8000/api/v1/websocket/ws to test connections.")
