# Projector Application Specification

## Purpose

The Projector application is responsible for rendering and displaying augmented reality overlays directly onto the pool table surface via a physical projector. It receives trajectory and game state data from the backend, performs geometric corrections for proper alignment, and generates real-time visual feedback that enhances the playing experience.

The application must be architected in such a way that it is easy to extend via addons/modules/plugins to add new features by providing a core API for addons/modules/plugins to interact with the main application. The application should be modular, with clear separation of concerns between components such as rendering, calibration, and network communication. The application should also include comprehensive logging and error handling to facilitate debugging and maintenance.

The first module shall be a calibration module to allow the user to calibrate the projector to the table. The second module shall be a trajectory rendering module to render the trajectories and other visual aids on the table.

The application should be primarily designed to operated with no user interaction, expecting all of the commands to arrive from commands over the network (originating from the API, triggered by various things including the user's interaction with the web interface).

## Deployment Options

The projector application supports two deployment modes:

### Native Desktop Application (LÖVE2D)
- Runs as a native application using LÖVE2D framework
- Supports UDP datagram sockets for low-latency communication
- Full hardware acceleration and GPU access
- Recommended for production use on dedicated projector hardware
- Supports all LÖVE2D features including native socket libraries

### Web Application (love.js)
- Compiled to WebAssembly for browser deployment
- Runs in any modern web browser
- **Network Protocol**: WebSocket (UDP not available in browsers due to security restrictions)
- Easier deployment and updates via web server
- Portable across devices without installation
- Limited to browser-based networking (WebSocket only)
- Requires WebSocket server endpoint from API

## Functional Requirements

### 1. Display Management

- **FR-PROJ-001**: Initialize projector display output (fullscreen)
- **FR-PROJ-002**: Detect and configure available display devices
- **FR-PROJ-003**: Support multiple display resolutions (720p, 1080p, 4K)

### 2. Geometric Calibration

#### 2.1 Perspective Correction

- **FR-PROJ-011**: Perform 4-point perspective transformation
- **FR-PROJ-012**: Support keystone correction for angled projection
- **FR-PROJ-013**: Compensate for barrel/pincushion distortion
- **FR-PROJ-014**: Handle non-perpendicular projector positioning
- **FR-PROJ-015**: Save and load calibration profiles

#### 2.2 Alignment Calibration

- **FR-PROJ-016**: Display calibration grid pattern
- **FR-PROJ-017**: Allow manual corner point adjustment
- **FR-PROJ-018**: Provide automatic alignment detection
- **FR-PROJ-019**: Support fine-tuning with arrow keys
- **FR-PROJ-020**: Validate calibration accuracy with test patterns

### 3. Visual Rendering

#### 3.1 Trajectory Visualization

- **FR-PROJ-021**: Draw primary ball trajectory lines
- **FR-PROJ-022**: Show reflection paths off cushions
- **FR-PROJ-023**: Display collision transfer trajectories
- **FR-PROJ-024**: Indicate pocket entry paths
- **FR-PROJ-025**: Show spin effect curves

#### 3.2 Visual Effects

- **FR-PROJ-026**: Apply gradient colors to indicate force/speed
- **FR-PROJ-027**: Animate trajectory appearance/disappearance
- **FR-PROJ-028**: Pulse or highlight recommended shots
- **FR-PROJ-029**: Show ghost balls for aiming reference
- **FR-PROJ-030**: Display impact zones with transparency

#### 3.3 Information Overlays

- **FR-PROJ-031**: Show shot difficulty indicators
- **FR-PROJ-032**: Display angle measurements
- **FR-PROJ-033**: Present force/power recommendations
- **FR-PROJ-034**: Show success probability percentages
- **FR-PROJ-035**: Display system messages and alerts

### 4. Real-time Updates

#### 4.1 Data Reception

- **FR-PROJ-036**: Connect to backend via network protocol (UDP for native, WebSocket for web)
- **FR-PROJ-036A**: Detect deployment mode (native vs web) and select appropriate protocol
- **FR-PROJ-036B**: Support UDP datagram sockets in native LÖVE2D deployment
- **FR-PROJ-036C**: Support WebSocket connection in web/browser deployment
- **FR-PROJ-037**: Receive trajectory updates in real-time
- **FR-PROJ-038**: Handle game state changes (periodic updates every 500ms)
- **FR-PROJ-039**: Process configuration updates
- **FR-PROJ-040**: Manage connection failures and reconnection
- **FR-PROJ-040A**: Implement WebSocket reconnection logic with exponential backoff
- **FR-PROJ-041A**: Receive ball motion events immediately when balls move
- **FR-PROJ-041B**: Process ball state messages with unique ball IDs
- **FR-PROJ-041C**: Handle optional ball number/type metadata
- **FR-PROJ-131**: Support subscription management for video feed data
- **FR-PROJ-132**: Send "subscribe" message for frame data when video feed enabled
- **FR-PROJ-133**: Send "unsubscribe" message when video feed disabled
- **FR-PROJ-134**: Handle subscription confirmation from backend
- **FR-PROJ-135**: Automatically resubscribe after reconnection

#### 4.2 Synchronization

- **FR-PROJ-041**: Synchronize display with camera frame rate
- **FR-PROJ-042**: Maintain smooth animation at 60 FPS
- **FR-PROJ-043**: Minimize latency between detection and display
- **FR-PROJ-044**: Handle frame dropping gracefully
- **FR-PROJ-045**: Interpolate between updates for smoothness

### 5. Customization

#### 5.1 Visual Preferences

- **FR-PROJ-046**: Configure line thickness and styles
- **FR-PROJ-047**: Adjust color schemes and themes
- **FR-PROJ-048**: Set transparency/opacity levels
- **FR-PROJ-049**: Choose animation speeds
- **FR-PROJ-050**: Select information display modes

#### 5.2 Assistance Levels

- **FR-PROJ-051**: Show/hide different trajectory types
- **FR-PROJ-052**: Adjust assistance based on skill level
- **FR-PROJ-053**: Toggle specific visual aids on/off
- **FR-PROJ-054**: Configure information density
- **FR-PROJ-055**: Support practice vs competition modes

#### 5.3 Debug and Development Options

- **FR-PROJ-136**: Enable/disable video feed display
- **FR-PROJ-137**: Toggle debug HUD overlay visibility
- **FR-PROJ-138**: Configure video feed display position (fullscreen/inset/overlay)
- **FR-PROJ-139**: Set video feed display size and opacity
- **FR-PROJ-140**: Select debug HUD information categories
- **FR-PROJ-141**: Configure debug HUD position and layout
- **FR-PROJ-142**: Set debug HUD text size and colors
- **FR-PROJ-143**: Toggle individual HUD sections on/off
- **FR-PROJ-144**: Enable/disable performance metrics overlay
- **FR-PROJ-145**: Configure debug hotkeys and shortcuts

### 6. Adaptive Color Management

#### 6.1 Table Surface Color Detection

- **FR-PROJ-056**: Detect and store base table felt color
- **FR-PROJ-057**: Support manual table color configuration
- **FR-PROJ-058**: Auto-detect color from calibration process
- **FR-PROJ-059**: Handle multiple table color profiles
- **FR-PROJ-060**: Persist table color settings per calibration profile

#### 6.2 Adaptive Color Selection

- **FR-PROJ-061**: Generate high-contrast colors based on table felt color
- **FR-PROJ-062**: Avoid colors similar to table felt (e.g., green on green)
- **FR-PROJ-063**: Provide complementary color palettes for visibility
- **FR-PROJ-064**: Calculate color visibility scores against table background
- **FR-PROJ-065**: Automatically adjust RGB values for maximum contrast

#### 6.3 Color Palette Generation

- **FR-PROJ-066**: Generate primary trajectory color (high contrast)
- **FR-PROJ-067**: Generate secondary/tertiary colors for multi-path visualization
- **FR-PROJ-068**: Create gradient colors that maintain visibility
- **FR-PROJ-069**: Generate text/overlay colors with readability guarantees
- **FR-PROJ-070**: Support color-blind friendly palette generation

### 7. Video Feed Display

#### 7.1 Video Stream Reception

- **FR-PROJ-071**: Receive video frames via WebSocket frame messages
- **FR-PROJ-072**: Decode base64-encoded JPEG frame data
- **FR-PROJ-073**: Parse frame metadata (timestamp, frame_number, dimensions)
- **FR-PROJ-074**: Handle frame data validation and error detection
- **FR-PROJ-075**: Maintain frame buffer for smooth playback
- **FR-PROJ-076**: Detect and handle frame sequence gaps
- **FR-PROJ-077**: Process frame rate metadata from backend
- **FR-PROJ-078**: Handle variable bitrate and quality settings

#### 7.2 Video Display Rendering

- **FR-PROJ-079**: Render video feed in configurable display modes (fullscreen, inset, overlay)
- **FR-PROJ-080**: Support video frame scaling and aspect ratio preservation
- **FR-PROJ-081**: Apply video feed opacity/transparency settings
- **FR-PROJ-082**: Position video feed according to configuration (corners, center, custom)
- **FR-PROJ-083**: Layer video feed appropriately (behind/above AR overlays)
- **FR-PROJ-084**: Apply geometric calibration transform to video feed
- **FR-PROJ-085**: Support video feed rotation and mirroring

#### 7.3 Video Performance Management

- **FR-PROJ-086**: Monitor video frame reception rate
- **FR-PROJ-087**: Implement frame dropping strategy for performance
- **FR-PROJ-088**: Track video decode performance metrics
- **FR-PROJ-089**: Adjust video quality based on system performance
- **FR-PROJ-090**: Manage video frame memory efficiently

### 8. Debug HUD Overlay

#### 8.1 HUD Information Display

- **FR-PROJ-091**: Display connection status (connected, disconnected, reconnecting)
- **FR-PROJ-092**: Show network protocol information (UDP/WebSocket)
- **FR-PROJ-093**: Display backend endpoint and port
- **FR-PROJ-094**: Show message reception statistics (total, rate, errors)
- **FR-PROJ-095**: Display current sequence number and gaps detected
- **FR-PROJ-096**: Show timestamp of last received message
- **FR-PROJ-097**: Display calibration status and profile name
- **FR-PROJ-098**: Show current display resolution and mode
- **FR-PROJ-099**: Display application version and build info
- **FR-PROJ-100**: Show system time and uptime

#### 8.2 Ball Tracking Information

- **FR-PROJ-101**: Display count of detected balls
- **FR-PROJ-102**: Show individual ball positions (x, y coordinates)
- **FR-PROJ-103**: Display ball velocities and movement status
- **FR-PROJ-104**: Show ball IDs and numbers (when available)
- **FR-PROJ-105**: Highlight cue ball separately
- **FR-PROJ-106**: Display ball tracking confidence scores
- **FR-PROJ-107**: Show ball state history (last N positions)

#### 8.3 Cue Stick Debug Information

- **FR-PROJ-108**: Display cue stick detection status
- **FR-PROJ-109**: Show cue stick position and angle
- **FR-PROJ-110**: Display cue tip position
- **FR-PROJ-111**: Show predicted strike direction
- **FR-PROJ-112**: Display cue stick velocity (when moving)
- **FR-PROJ-113**: Show cue stick tracking confidence

#### 8.4 Table Geometry Debug

- **FR-PROJ-114**: Display table corner positions
- **FR-PROJ-115**: Show table dimensions (width x height)
- **FR-PROJ-116**: Display pocket positions
- **FR-PROJ-117**: Show cushion rail positions
- **FR-PROJ-118**: Display calibration transform matrix

#### 8.5 Performance Metrics Display

- **FR-PROJ-119**: Display current FPS (frames per second)
- **FR-PROJ-120**: Show frame render time (milliseconds)
- **FR-PROJ-121**: Display memory usage (MB)
- **FR-PROJ-122**: Show GPU usage percentage
- **FR-PROJ-123**: Display network latency measurements
- **FR-PROJ-124**: Show video frame decode time (when applicable)

#### 8.6 HUD Toggle Controls

- **FR-PROJ-125**: Support keyboard shortcut to toggle HUD visibility
- **FR-PROJ-126**: Allow toggling individual HUD sections
- **FR-PROJ-127**: Support HUD opacity adjustment
- **FR-PROJ-128**: Enable/disable HUD via network command
- **FR-PROJ-129**: Save HUD configuration preferences
- **FR-PROJ-130**: Support HUD layout presets (minimal, standard, detailed)

## Non-Functional Requirements

### Visual Quality Requirements

- **NFR-PROJ-001**: Anti-aliased line rendering
- **NFR-PROJ-002**: Smooth gradient transitions
- **NFR-PROJ-003**: No visible flicker or tearing
- **NFR-PROJ-004**: Consistent brightness across projection
- **NFR-PROJ-005**: Color accuracy within 10% of target
- **NFR-PROJ-007**: Minimum contrast ratio of 4.5:1 for all overlays against table surface

### Video Feed Requirements

- **NFR-PROJ-014**: Video feed display latency < 100ms from capture to display
- **NFR-PROJ-015**: Support minimum video resolution of 640x480
- **NFR-PROJ-016**: Support video frame rates from 15-60 FPS
- **NFR-PROJ-017**: Video frame decode time < 16ms (60 FPS target)
- **NFR-PROJ-018**: Video display maintains 60 FPS overall rendering

### Debug HUD Requirements

- **NFR-PROJ-019**: HUD rendering overhead < 2ms per frame
- **NFR-PROJ-020**: HUD text legible at all supported resolutions
- **NFR-PROJ-021**: HUD update rate at least 10 Hz for dynamic information

### Reliability Requirements

- **NFR-PROJ-006**: No display crashes during operation
- **NFR-PROJ-008**: Graceful degradation under load
- **NFR-PROJ-009**: Maintain calibration across restarts
- **NFR-PROJ-010**: Handle projector power cycles

### Compatibility Requirements

- **NFR-PROJ-011**: Support Ubuntu server
- **NFR-PROJ-012**: Leverage integrated VAAPI/iHD GPU
- **NFR-PROJ-013**: Allow for support of multiple GPU configurations

## Success Criteria

### Display Success Criteria

1. **Visual Quality**

   - Smooth, anti-aliased line rendering
   - No visible flicker or tearing
   - Consistent color reproduction
   - Clear text legibility at all sizes

2. **Alignment Accuracy**

   - Projection aligned within 5mm of physical table
   - Stable calibration without drift
   - Correct perspective at all table positions
   - Accurate trajectory overlay on actual paths

3. **Real-time Performance**
   - Consistent 60 FPS rendering
   - < 16ms frame rendering time
   - Smooth animations without stuttering
   - No dropped frames during normal operation

### Functional Success Criteria

1. **Trajectory Display**

   - All trajectory types rendered correctly
   - Collision points clearly visible
   - Smooth trajectory updates during aiming
   - Correct fade-in/fade-out effects

2. **Calibration**

   - Calibration completes in < 2 minutes
   - Calibration persists across restarts
   - Easy fine-tuning adjustments
   - Visual confirmation of accuracy

3. **Customization**
   - All visual preferences apply immediately
   - Settings persist between sessions
   - Assistance levels work as configured
   - Color schemes apply correctly

4. **Video Feed Display**
   - Video feed displays without stuttering
   - Video maintains aspect ratio and quality
   - Video feed opacity and positioning work correctly
   - Video display does not impact AR overlay performance

5. **Debug HUD**
   - All HUD information displays accurately
   - HUD updates in real-time
   - HUD toggle controls work reliably
   - HUD remains legible in all display modes

### Performance Success Criteria

1. **Rendering Performance**

   - 60 FPS with full visual effects
   - < 500MB GPU memory usage
   - < 30% GPU utilization typical
   - Smooth degradation under load

2. **Latency**

   - < 50ms total system latency
   - < 16ms rendering latency
   - Immediate response to updates
   - No perceptible lag in animations

3. **Stability**
   - No crashes during 24-hour operation
   - Automatic recovery from errors
   - Graceful handling of disconnections
   - Consistent performance over time

## Testing Requirements

### Unit Testing

- Test geometric transformation functions
- Validate calibration calculations
- Test rendering primitive functions
- Verify color space conversions
- Test adaptive color generation algorithms
- Validate contrast ratio calculations
- Test video frame decoding functions
- Test HUD rendering components
- Coverage target: 80%

### Integration Testing

- Test complete rendering pipeline
- Verify datagram sockets data reception
- Test calibration procedures
- Validate configuration updates
- Test error recovery
- Test video feed subscription and streaming
- Verify HUD data collection and display
- Test video feed and AR overlay layering

### Visual Testing

- Verify rendering quality
- Test all visual effects
- Validate color accuracy
- Check text rendering
- Test animation smoothness
- Verify video feed display quality
- Validate HUD legibility and layout
- Test video feed opacity and blending

### Performance Testing

- Benchmark frame rates
- Measure rendering latency
- Test GPU memory usage
- Profile CPU/GPU usage
- Stress test with complex scenes
- Measure video decode performance
- Test HUD rendering overhead
- Profile video feed + AR overlay + HUD combined performance

### Hardware Testing

- Test with different projectors
- Verify multiple resolutions
- Test various GPU configurations
- Validate on all platforms
- Test with different cable types
- Test video streaming at different network conditions
- Verify video display on different display sizes

## Network Communication Protocol

The projector receives real-time updates via network sockets. Messages are JSON-encoded and follow a consistent format regardless of transport protocol.

### Transport Protocols

#### Native Application (LÖVE2D)
- **Protocol**: UDP datagram sockets
- **Default listening port**: 5005
- **Advantages**: Lower latency, connectionless
- **Library**: LuaSocket (included with LÖVE2D)

#### Web Application (love.js)
- **Protocol**: WebSocket
- **Default endpoint**: `ws://[api-host]:8000/api/v1/game/state/ws`
- **Advantages**: Bi-directional, browser-compatible, automatic reconnection
- **Library**: Pure Lua WebSocket client (love2d-lua-websocket or löve-ws)
- **Note**: UDP sockets are not available in browsers due to security restrictions

### Message Format

```javascript
// Base message structure
{
  "type": "state|motion|trajectory|alert|config|frame|subscribe|unsubscribe",
  "timestamp": "ISO 8601 timestamp",
  "sequence": 12345,
  "data": { /* type-specific payload */ }
}

// State message - periodic update of all ball positions (every 500ms default)
{
  "type": "state",
  "timestamp": "2024-01-15T10:30:45.123Z",
  "sequence": 100,
  "data": {
    "balls": [
      {
        "id": "ball_001",        // Unique persistent ID
        "position": [100, 200],  // [x, y] coordinates
        "velocity": [0, 0],      // [vx, vy]
        "is_moving": false,
        "number": 1,             // Optional: ball number (1-15) or null
        "is_cue_ball": false
      }
    ],
    "cue": { /* cue stick data */ },
    "table": { /* table geometry */ }
  }
}

// Motion event - immediate update when ball starts moving
{
  "type": "motion",
  "timestamp": "2024-01-15T10:30:45.123Z",
  "sequence": 101,
  "data": {
    "ball_id": "ball_001",     // Which ball is moving
    "position": [105, 205],
    "velocity": [10.5, 5.2],
    "is_moving": true,
    "number": 1                // Optional
  }
}

// Trajectory message
{
  "type": "trajectory",
  "timestamp": "2024-01-15T10:30:45.123Z",
  "sequence": 102,
  "data": {
    "lines": [
      {
        "start": [100, 200],
        "end": [300, 400],
        "type": "primary|reflection|collision"
      }
    ],
    "collisions": [ /* collision data */ ]
  }
}

// Frame message - video feed data
{
  "type": "frame",
  "timestamp": "2024-01-15T10:30:45.123Z",
  "sequence": 103,
  "data": {
    "frame_number": 12345,
    "timestamp": "2024-01-15T10:30:45.123Z",
    "width": 1920,
    "height": 1080,
    "format": "jpeg",
    "data": "base64_encoded_jpeg_data...",
    "fps": 30.0,
    "quality": 85
  }
}

// Subscribe message - request video feed subscription
{
  "type": "subscribe",
  "timestamp": "2024-01-15T10:30:45.123Z",
  "sequence": 104,
  "data": {
    "stream_type": "frames",
    "options": {
      "fps": 30,
      "quality": 85,
      "format": "jpeg"
    }
  }
}

// Unsubscribe message - cancel video feed subscription
{
  "type": "unsubscribe",
  "timestamp": "2024-01-15T10:30:45.123Z",
  "sequence": 105,
  "data": {
    "stream_type": "frames"
  }
}

// Subscription confirmation
{
  "type": "subscription_confirmed",
  "timestamp": "2024-01-15T10:30:45.123Z",
  "sequence": 106,
  "data": {
    "stream_type": "frames",
    "status": "active",
    "fps": 30,
    "quality": 85
  }
}
```

### Protocol Configuration

#### UDP Configuration (Native)
- **Default listening port**: 5005
- **Message encoding**: UTF-8 JSON
- **Max packet size**: 65507 bytes
- **Protocol**: UDP (connectionless, no acknowledgment)
- **Message ordering**: Use sequence number for ordering
- **Duplicate detection**: Use sequence number to detect duplicates

#### WebSocket Configuration (Web)
- **Default endpoint**: `ws://[api-host]:8000/api/v1/game/state/ws`
- **Message encoding**: UTF-8 JSON (text frames)
- **Reconnection**: Automatic with exponential backoff
- **Heartbeat**: Ping/pong every 30 seconds
- **Max reconnect delay**: 30 seconds
- **Initial retry delay**: 1 second

## Configuration Schema

The projector application uses a JSON configuration file with the following structure:

```json
{
  "display": {
    "resolution": "1920x1080",
    "fullscreen": true,
    "vsync": true
  },
  "network": {
    "protocol": "auto",
    "udp_port": 5005,
    "websocket_url": "ws://localhost:8000/api/v1/game/state/ws",
    "reconnect": true,
    "reconnect_delay": 1000,
    "max_reconnect_delay": 30000
  },
  "calibration": {
    "profile": "default",
    "auto_load": true,
    "corner_points": [[0,0], [1920,0], [1920,1080], [0,1080]]
  },
  "rendering": {
    "fps_target": 60,
    "line_thickness": 3,
    "anti_aliasing": true,
    "theme": "default"
  },
  "video_feed": {
    "enabled": false,
    "display_mode": "inset",
    "position": "bottom_right",
    "size": "25%",
    "opacity": 0.8,
    "layer": "below_overlays",
    "subscribe_on_start": false,
    "quality": 85,
    "fps": 30
  },
  "debug_hud": {
    "enabled": false,
    "position": "top_left",
    "opacity": 0.9,
    "font_size": 14,
    "color": [255, 255, 255],
    "background": [0, 0, 0, 128],
    "sections": {
      "connection": true,
      "balls": true,
      "cue": true,
      "table": false,
      "performance": true
    },
    "layout": "compact",
    "update_rate": 10
  }
}
```

## Implementation Guidelines

### Code Structure

```lua
projector/                    # LÖVE2D Application
├── main.lua                 # Main entry point
├── conf.lua                 # LÖVE configuration
├── conf_windowed.lua        # Windowed mode config (development)
├── core/
│   ├── init.lua            # Core initialization
│   ├── network.lua         # Network abstraction layer
│   ├── udp_client.lua      # UDP client (native only)
│   └── websocket_client.lua # WebSocket client (web only)
├── modules/
│   ├── calibration/
│   │   ├── init.lua       # Calibration module
│   │   └── interactive.lua # Interactive calibration
│   ├── trajectory/
│   │   ├── init.lua       # Trajectory rendering module
│   │   └── effects.lua    # Visual effects
│   ├── video_feed/
│   │   ├── init.lua       # Video feed display module
│   │   ├── decoder.lua    # Frame decoding
│   │   └── renderer.lua   # Video rendering
│   ├── debug_hud/
│   │   ├── init.lua       # Debug HUD module
│   │   ├── sections.lua   # HUD section components
│   │   └── metrics.lua    # Performance metrics
│   └── network_status/
│       └── init.lua       # Network status display
├── lib/
│   ├── json.lua           # JSON encoding/decoding
│   ├── websocket.lua      # WebSocket library (for web builds)
│   └── base64.lua         # Base64 decoding for video frames
├── config/
│   └── default.json       # Default configuration
├── build_html.sh          # Build script for web version
├── deploy.sh              # Deploy to target environment
└── test_local.sh          # Local testing script
```

### Build & Deployment

#### Native Application
```bash
# Run locally
love .

# Deploy to target
./deploy.sh deploy
./deploy.sh run
```

#### Web Application
```bash
# Build web version
./build_html.sh

# Serve locally for testing
cd build/web
python3 -m http.server 8080

# Deploy to target environment
rsync -av build/web/ jchadwick@192.168.1.31:/opt/billiards-trainer/frontend/projector/web/
```

### Key Dependencies

#### Native (LÖVE2D)
- **LÖVE2D**: Game framework (v11.4+)
- **LuaSocket**: Network sockets (included with LÖVE)
- **JSON library**: Message parsing
- **Base64 library**: Video frame decoding

#### Web (love.js)
- **love.js**: LÖVE to WebAssembly compiler
- **WebSocket library**: Browser networking (pure Lua implementation)
- **JSON library**: Message parsing
- **Base64 library**: Video frame decoding

### Development Priorities

1. ✅ Implement basic display output
2. ✅ Add simple line rendering
3. ✅ Implement network communication (UDP for native)
4. ✅ Add trajectory rendering
5. Add WebSocket support for web builds
6. Implement video feed display module
7. Implement debug HUD overlay module
8. Implement calibration system
9. Add visual effects
10. Optimize performance
11. Add advanced features

### Network Protocol Detection

The application should detect its runtime environment and automatically select the appropriate network protocol:

```lua
-- Pseudo-code for protocol detection
function detectNetworkProtocol()
  if love.system.getOS() == "Web" then
    -- Browser environment - use WebSocket
    return NetworkProtocol.WebSocket
  else
    -- Native LÖVE2D - use UDP
    return NetworkProtocol.UDP
  end
end
```

### WebSocket Integration (Web Builds)

For web deployments, the projector needs:

1. **WebSocket Library**: Pure Lua implementation compatible with love.js
   - Recommended: `love2d-lua-websocket` (https://github.com/flaribbit/love2d-lua-websocket)
   - Alternative: `löve-ws` (C++ based, may require compilation)

2. **API WebSocket Endpoint**: The backend API must provide a WebSocket endpoint
   - Endpoint: `ws://[api-host]:8000/api/v1/game/state/ws`
   - Same message format as UDP (JSON encoded)
   - Server-side implementation in FastAPI

3. **Connection Management**:
   - Auto-connect on startup
   - Reconnect on disconnect with exponential backoff
   - Display connection status in UI
   - Graceful degradation when disconnected

4. **Message Handling**: Identical to UDP implementation
   - Parse JSON messages
   - Route by message type (state, motion, trajectory, config, alert, frame)
   - Update visual displays accordingly

### Video Feed Integration

The video feed module provides real-time display of the camera feed:

1. **Subscription Management**:
   - Send subscribe message when video feed enabled
   - Handle subscription confirmation
   - Automatically resubscribe after reconnection
   - Send unsubscribe message when video feed disabled

2. **Frame Decoding**:
   - Decode base64-encoded JPEG data
   - Validate frame integrity
   - Handle decode errors gracefully
   - Maintain frame buffer for smooth playback

3. **Display Rendering**:
   - Support multiple display modes (fullscreen, inset, overlay)
   - Apply geometric calibration transforms
   - Handle aspect ratio preservation
   - Layer appropriately with AR overlays

4. **Performance Management**:
   - Monitor frame reception rate
   - Implement frame dropping when necessary
   - Track decode performance
   - Optimize memory usage

### Debug HUD Integration

The debug HUD provides real-time development and diagnostic information:

1. **Information Collection**:
   - Gather connection status and metrics
   - Collect ball tracking data
   - Monitor cue stick information
   - Track table geometry
   - Measure performance metrics

2. **Display Rendering**:
   - Render text overlays with high contrast
   - Support multiple layout modes
   - Apply opacity and background styling
   - Update at configurable rate (10 Hz default)

3. **Toggle Controls**:
   - Keyboard shortcuts (e.g., F3 for HUD toggle)
   - Network commands for remote control
   - Individual section toggles
   - Layout preset switching

4. **Performance Optimization**:
   - Minimize rendering overhead (< 2ms target)
   - Cache text rendering when possible
   - Update only changed information
   - Batch draw calls efficiently
