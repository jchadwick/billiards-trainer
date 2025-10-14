# WebSocket Client Implementation

## Overview

Complete WebSocket client implementation for the Billiards Visualizer using the `love2d-lua-websocket` library. Provides auto-reconnect with exponential backoff, comprehensive statistics tracking, and robust error handling.

## Features

- **Automatic JSON encoding/decoding** - Messages are automatically serialized and deserialized
- **Auto-reconnect with exponential backoff** - Configurable reconnection strategy (FR-VIS-040A)
- **Connection state tracking** - DISCONNECTED, CONNECTING, CONNECTED, RECONNECTING, ERROR
- **Comprehensive statistics** - Messages, bytes, uptime, reconnection attempts
- **URL parsing** - Supports both URL format and host/port/path format
- **Event-driven callbacks** - onopen, onmessage, onerror, onclose
- **Graceful error handling** - All callbacks wrapped in pcall for safety

## Usage

### Basic Example

```lua
local WebSocket = require("modules.network.websocket")

-- Create WebSocket client
local ws = WebSocket.new({
    url = "ws://localhost:8080/api",

    onopen = function()
        print("Connected!")
    end,

    onmessage = function(message)
        -- message is already decoded from JSON
        print("Received:", message.type)
        print("Data:", message.data)
    end,

    onerror = function(err)
        print("Error:", err)
    end,

    onclose = function(code, reason)
        print("Closed:", code, reason)
    end,
})

-- Connect
ws:connect()

-- In love.update()
function love.update(dt)
    ws:update(dt)
end

-- Send message (will be JSON encoded automatically)
ws:send({
    type = "state_update",
    data = {
        position = {x = 100, y = 200}
    }
})

-- Check connection
if ws:isConnected() then
    print("Connected to:", ws:getURL())
end

-- Get statistics
local stats = ws:getStats()
print("Messages sent:", stats.messages_sent)
print("Messages received:", stats.messages_received)
print("Uptime:", stats.uptime, "seconds")
```

### Configuration Options

#### Connection Configuration

```lua
local ws = WebSocket.new({
    -- Option 1: Use URL format
    url = "ws://localhost:8080/api",

    -- Option 2: Use host/port/path
    host = "localhost",
    port = 8080,
    path = "/api",

    -- Callbacks
    onopen = function() end,
    onmessage = function(msg) end,
    onerror = function(err) end,
    onclose = function(code, reason) end,

    -- Reconnection settings
    reconnect_enabled = true,           -- Enable auto-reconnect (default: true)
    reconnect_delay = 1000,             -- Initial delay in ms (default: 1000)
    max_reconnect_delay = 30000,        -- Max delay in ms (default: 30000)
    max_reconnect_attempts = 0,         -- Max attempts, 0 = unlimited (default: 0)
})
```

#### URL Formats Supported

- `ws://hostname:port/path` - Full URL with port and path
- `ws://hostname/path` - URL with default port (80) and path
- `ws://hostname:port` - URL with port, default path (/)
- `ws://hostname` - URL with defaults (port 80, path /)

#### Reconnection Strategy

The client implements exponential backoff for reconnection:

1. **Initial delay**: Starts at `reconnect_delay` (default 1 second)
2. **Exponential backoff**: Doubles delay after each failed attempt
3. **Max delay cap**: Never exceeds `max_reconnect_delay` (default 30 seconds)
4. **Max attempts**: Stops after `max_reconnect_attempts` (0 = unlimited)
5. **Delay reset**: Resets to initial delay on successful connection

Example progression with defaults:
- Attempt 1: Wait 1 second
- Attempt 2: Wait 2 seconds
- Attempt 3: Wait 4 seconds
- Attempt 4: Wait 8 seconds
- Attempt 5: Wait 16 seconds
- Attempt 6: Wait 30 seconds (capped)
- Attempt 7+: Wait 30 seconds (capped)

## API Reference

### Constructor

#### `WebSocket.new(config)`

Creates a new WebSocket client instance.

**Parameters:**
- `config` (table): Configuration options (see Configuration Options above)

**Returns:**
- WebSocket instance

### Methods

#### `connect()`

Initiates connection to the WebSocket server.

**Returns:**
- `boolean`: True if connection attempt started, false if already connected/connecting

#### `disconnect([code, reason])`

Closes the WebSocket connection.

**Parameters:**
- `code` (number, optional): WebSocket close code (default: 1000)
- `reason` (string, optional): Close reason (default: "client disconnect")

#### `send(message)`

Sends a message through the WebSocket. Message is automatically JSON encoded.

**Parameters:**
- `message` (table): Message to send (must be JSON-serializable)

**Returns:**
- `boolean`: True if sent successfully, false otherwise

**Example:**
```lua
ws:send({
    type = "trajectory_update",
    data = {
        ball_id = 1,
        path = {{x=0, y=0}, {x=100, y=100}}
    }
})
```

#### `update(dt)`

Updates the WebSocket client. Call this every frame.

**Parameters:**
- `dt` (number): Delta time in seconds

**Example:**
```lua
function love.update(dt)
    ws:update(dt)
end
```

#### `isConnected()`

Checks if the WebSocket is currently connected.

**Returns:**
- `boolean`: True if connected, false otherwise

#### `getState()`

Gets the current connection state.

**Returns:**
- `string`: One of "DISCONNECTED", "CONNECTING", "CONNECTED", "RECONNECTING", "ERROR"

#### `getURL()`

Gets the WebSocket URL.

**Returns:**
- `string`: The WebSocket URL (e.g., "ws://localhost:8080/api")

#### `getStats()`

Gets connection and message statistics.

**Returns:**
- `table`: Statistics object with fields:
  - `messages_sent` (number): Total messages sent
  - `messages_received` (number): Total messages received
  - `bytes_sent` (number): Total bytes sent
  - `bytes_received` (number): Total bytes received
  - `connection_time` (number): Time when last connected
  - `uptime` (number): Total time connected in seconds
  - `reconnect_attempts` (number): Total reconnection attempts

#### `resetStats()`

Resets all statistics to zero.

## Connection States

### DISCONNECTED
- Not connected to server
- No active connection attempt
- Safe to call `connect()`

### CONNECTING
- Connection attempt in progress
- Waiting for server response
- Will transition to CONNECTED or ERROR

### CONNECTED
- Successfully connected to server
- Can send and receive messages
- Statistics are being tracked

### RECONNECTING
- Waiting to attempt reconnection
- Auto-reconnect is enabled
- Will attempt connection after delay

### ERROR
- Connection error occurred
- May transition to RECONNECTING if auto-reconnect is enabled
- Check error callback for details

## Error Handling

All callbacks are wrapped in `pcall` to prevent crashes:

```lua
local ws = WebSocket.new({
    url = "ws://localhost:8080",

    onerror = function(err)
        -- Handle errors gracefully
        print("WebSocket error:", err)

        -- Common errors:
        -- - "connection refused" - Server not running
        -- - "connection timeout" - Server not responding
        -- - "JSON encoding failed" - Invalid message format
        -- - "JSON decoding failed" - Invalid message received
        -- - "Max reconnect attempts reached" - Too many reconnection failures
    end,
})
```

## Integration with Network Module

The WebSocket client is used by the Network module (`modules/network/init.lua`):

```lua
-- In Network:init()
self.websocket = WebSocket.new({
    host = self.config.host,
    port = self.config.port,
    path = "/",

    onmessage = function(message)
        self:handleMessage(message)
    end,

    onopen = function()
        self:onConnect()
    end,

    onclose = function(code, reason)
        self:onDisconnect(reason)
    end,

    onerror = function(error)
        self:onError(error)
    end,

    reconnect_enabled = true,
    reconnect_delay = 1000,
    max_reconnect_delay = 30000,
})

-- In Network:update(dt)
if self.websocket then
    self.websocket:update(dt)
end
```

## Testing

### Manual Testing

1. **Start a WebSocket server** on `ws://localhost:8080`
2. **Run the visualizer**: `love /path/to/visualizer`
3. **Check console output** for connection messages
4. **Monitor HUD** for connection status (F1 to toggle)

### Test Server (Node.js)

```javascript
const WebSocket = require('ws');
const wss = new WebSocket.Server({ port: 8080 });

wss.on('connection', (ws) => {
    console.log('Client connected');

    // Send test message
    ws.send(JSON.stringify({
        type: 'test',
        data: { message: 'Hello from server!' }
    }));

    // Echo messages back
    ws.on('message', (data) => {
        console.log('Received:', data);
        ws.send(data);
    });

    ws.on('close', () => {
        console.log('Client disconnected');
    });
});
```

### Expected Behavior

1. **Connection**: Should connect within 1 second
2. **Auto-reconnect**: Should reconnect if server restarts
3. **Message flow**: Should send/receive JSON messages correctly
4. **Exponential backoff**: Reconnection delays should increase (1s, 2s, 4s, ...)
5. **Statistics**: Should track all messages and bytes accurately

## Troubleshooting

### Connection Refused

```
Error: connection refused
```

**Solution**: Ensure WebSocket server is running on the configured host/port.

### JSON Encoding/Decoding Errors

```
Error: JSON encoding failed
```

**Solution**: Ensure messages are valid Lua tables (not functions, userdata, etc).

### Max Reconnect Attempts Reached

```
Error: Max reconnect attempts reached
```

**Solution**: Increase `max_reconnect_attempts` or set to 0 for unlimited attempts.

### Connection Hangs

**Check**:
1. Server is running and accessible
2. No firewall blocking connection
3. Correct host/port configuration
4. Server implements WebSocket protocol correctly

## Performance Considerations

- **Update frequency**: Call `update(dt)` every frame (typically 60 FPS)
- **Message size**: Keep messages small for low latency
- **Send rate**: Avoid sending too many messages per second
- **Statistics overhead**: Minimal, tracked with simple counters
- **Memory usage**: Buffers are managed by underlying websocket library

## Future Enhancements

Potential improvements for future versions:

- [ ] Message queuing for offline mode
- [ ] Message acknowledgment/confirmation
- [ ] Compression support (gzip, deflate)
- [ ] Binary message support (MessagePack, Protocol Buffers)
- [ ] Connection quality metrics (latency, packet loss)
- [ ] Automatic bandwidth throttling
- [ ] WSS (secure WebSocket) support
- [ ] Message priority queuing
- [ ] Heartbeat/ping-pong implementation

## References

- **Underlying library**: `lib/websocket.lua` (love2d-lua-websocket)
- **Specification**: FR-VIS-040A (Auto-reconnect with exponential backoff)
- **Integration**: `modules/network/init.lua` (Network module)
- **Documentation**: `/frontend/visualizer/SPECS.md`

## License

Part of the Billiards Trainer project.
