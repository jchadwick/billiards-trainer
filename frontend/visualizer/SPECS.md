# Visualizer Core Specification

## Purpose

The Visualizer is the **core visualization engine** responsible for rendering training visualizations via both augmented-reality physical pool table projection and web interface. It provides modular, reusable components for trajectory visualization, calibration, real-time updates, and customization. The visualizer is designed to be embedded in different wrapper applications (native projector wrapper or web application wrapper).

The visualizer is architected as an extensible, modular system with clear separation of concerns between components such as rendering, calibration, and network communication. It includes comprehensive logging and error handling to facilitate debugging and maintenance.

The visualizer operates primarily through network commands (from the backend API) with minimal direct user interaction required.

## Deployment Modes

The visualizer core supports two deployment contexts:

### 1. Native Projector Wrapper
- Embedded in LÖVE2D native application (`frontend/projector`)
- Full hardware acceleration and GPU access
- **Does NOT display video feed** (dedicated projector only shows AR overlays)
- Optimized for low-latency, high-performance rendering

### 2. Web Application Wrapper
- Embedded in browser-based application (`frontend/web`)
- WebGL rendering via browser
- **ALWAYS displays video feed** (users need to see the table in browser)
- Provides remote access and monitoring capabilities

This core visualizer code does NOT implement any of the wrapper-specific logic (e.g. window management, input handling, etc). That is handled by the respective wrapper applications. The visualizer core provides a clean API for embedding in either context.

## Modular Architecture

The core components it implements are to be written with a plugin/module architecture so that it is very easy to add new functionality (training/visualization modes) in the future by quickly and easily building on top of the existing functionality.

At the minimum, the visualizer core includes the following modules:

1. **Core API**: Provides hooks and interfaces for modules to interact with the main application
2. **Calibration Module**: Handles geometric correction and alignment
3. **Trajectory Module**: Renders trajectories and visual aids
4. **Debug HUD Module**: Shows diagnostic information
5. **Network Module**: Handles real-time data reception

## Functional Requirements

### 1. Geometric Calibration

These features ensure the projected visuals align correctly with the physical table surface.  Though they will likely only be used to calibrate the physical projector wrapper, they are implemented in the core visualizer so that they can be rendered in the web wrapper for development/debugging purposes.

#### 1.1 Perspective Correction
- **FR-VIS-011**: Perform 4-point perspective transformation
- **FR-VIS-012**: Support keystone correction for angled projection
- **FR-VIS-013**: Compensate for barrel/pincushion distortion
- **FR-VIS-014**: Handle non-perpendicular positioning
- **FR-VIS-015**: Save and load calibration profiles

#### 1.2 Alignment Calibration
- **FR-VIS-016**: Display calibration grid pattern
- **FR-VIS-017**: Allow manual corner point adjustment
- **FR-VIS-018**: Provide automatic alignment detection
- **FR-VIS-019**: Support fine-tuning with arrow keys
- **FR-VIS-020**: Validate calibration accuracy with test patterns

### 2. Visual Rendering

#### 2.1 Trajectory Visualization
- **FR-VIS-021**: Draw primary ball trajectory lines
- **FR-VIS-022**: Show reflection paths off cushions
- **FR-VIS-023**: Display collision transfer trajectories
- **FR-VIS-024**: Indicate pocket entry paths
- **FR-VIS-025**: (FUTURE) Show spin effect curves

#### 2.2 Visual Effects
- **FR-VIS-026**: (FUTURE) Apply gradient colors to indicate force/speed
- **FR-VIS-027**: Animate trajectory appearance/disappearance
- **FR-VIS-028**: Pulse or highlight recommended shots
- **FR-VIS-029**: Show ghost balls for aiming reference
- **FR-VIS-030**: Display impact zones with transparency

#### 2.3 Information Overlays
- **FR-VIS-031**: Show shot difficulty indicators
- **FR-VIS-032**: Display angle measurements
- **FR-VIS-033**: Present force/power recommendations
- **FR-VIS-034**: Show success probability percentages
- **FR-VIS-035**: Display system messages and alerts

### 3. Real-time Updates

#### 3.1 Data Reception
- **FR-VIS-036**: Connect to backend via WebSocket
- **FR-VIS-037**: Receive trajectory updates in real-time
- **FR-VIS-038**: Handle game state changes (periodic updates every 500ms)
- **FR-VIS-039**: Process configuration updates
- **FR-VIS-040**: Manage connection failures and reconnection
- **FR-VIS-040A**: Implement WebSocket reconnection logic with exponential backoff
- **FR-VIS-041A**: Receive and process ball motion events immediately when balls move
- **FR-VIS-041B**: Process ball state messages with unique ball IDs
- **FR-VIS-041C**: (FUTURE) Handle optional ball number/type metadata, when provided by the backend

#### 3.2 Synchronization
- **FR-VIS-041**: Synchronize display with backend updates
- **FR-VIS-042**: Maintain smooth animation at 60 FPS
- **FR-VIS-043**: Minimize latency between detection and display
- **FR-VIS-044**: Handle frame dropping gracefully
- **FR-VIS-045**: Interpolate between updates for smoothness

### 4. Customization

#### 4.1 Visual Preferences
- **FR-VIS-046**: Configure line thickness and styles
- **FR-VIS-047**: Adjust color schemes and themes
- **FR-VIS-048**: Set transparency/opacity levels
- **FR-VIS-049**: Choose animation speeds
- **FR-VIS-050**: Select information display modes

#### 4.3 Debug and Development Options
- **FR-VIS-057**: Toggle debug HUD overlay visibility
- **FR-VIS-060**: Select debug HUD information categories
- **FR-VIS-061**: Configure debug HUD position and layout
- **FR-VIS-062**: Set debug HUD text size and colors
- **FR-VIS-063**: Toggle individual HUD sections on/off
- **FR-VIS-064**: Enable/disable performance metrics overlay
- **FR-VIS-065**: Configure debug hotkeys and shortcuts

### 5. Adaptive Color Management

#### 5.1 Table Surface Color Detection
- **FR-VIS-070**: Detect and store base table felt color
- **FR-VIS-071**: Support manual table color configuration
- **FR-VIS-072**: Auto-detect color from calibration process
- **FR-VIS-074**: Persist table color settings per calibration profile

#### 5.2 Adaptive Color Selection
- **FR-VIS-075**: Generate high-contrast colors based on table felt color
- **FR-VIS-076**: Avoid colors similar to table felt (e.g., green on green)
- **FR-VIS-077**: Provide complementary color palettes for visibility
- **FR-VIS-078**: Calculate color visibility scores against table background
- **FR-VIS-079**: Automatically adjust RGB values for maximum contrast

#### 5.3 Color Palette Generation
- **FR-VIS-080**: Generate primary trajectory color (high contrast)
- **FR-VIS-081**: Generate secondary/tertiary colors for multi-path visualization
- **FR-VIS-082**: Create gradient colors that maintain visibility
- **FR-VIS-083**: Generate text/overlay colors with readability guarantees
- **FR-VIS-084**: Support color-blind friendly palette generation

### 7. Debug HUD Overlay

#### 7.1 HUD Information Display
- **FR-VIS-150**: Display connection status (connected, disconnected, reconnecting)
- **FR-VIS-151**: Show network protocol information (WebSocket)
- **FR-VIS-152**: Display backend endpoint and port
- **FR-VIS-153**: Show message reception statistics (total, rate, errors)
- **FR-VIS-154**: Display current sequence number and gaps detected
- **FR-VIS-155**: Show timestamp of last received message
- **FR-VIS-156**: Display calibration status and profile name
- **FR-VIS-157**: Show current display resolution and mode
- **FR-VIS-158**: Display application version and build info
- **FR-VIS-159**: Show system time and uptime

#### 7.2 Ball Tracking Information
- **FR-VIS-160**: Display count of detected balls
- **FR-VIS-161**: Show individual ball positions (x, y coordinates)
- **FR-VIS-162**: Display ball velocities and movement status
- **FR-VIS-163**: Show ball IDs and numbers (when available)
- **FR-VIS-164**: Highlight cue ball separately
- **FR-VIS-165**: Display ball tracking confidence scores
- **FR-VIS-166**: Show ball state history (last N positions)

#### 7.3 Cue Stick Debug Information
- **FR-VIS-173**: Show predicted strike direction
- **FR-VIS-175**: Show cue stick tracking confidence

#### 7.4 Table Geometry Debug
- **FR-VIS-180**: Display table corner positions
- **FR-VIS-181**: Show table dimensions (width x height)
- **FR-VIS-182**: Display pocket positions
- **FR-VIS-183**: Show cushion rail positions
- **FR-VIS-184**: Display calibration transform matrix

#### 7.5 Performance Metrics Display
- **FR-VIS-190**: Display current FPS (frames per second)
- **FR-VIS-191**: Show frame render time (milliseconds)
- **FR-VIS-192**: Display memory usage (MB)
- **FR-VIS-193**: Show GPU usage percentage (when available)
- **FR-VIS-194**: Display network latency measurements

#### 7.6 HUD Toggle Controls
- **FR-VIS-200**: Support keyboard shortcut to toggle HUD visibility
- **FR-VIS-201**: Allow toggling individual HUD sections
- **FR-VIS-202**: Support HUD opacity adjustment
- **FR-VIS-203**: Enable/disable HUD via network command
- **FR-VIS-204**: Save HUD configuration preferences
- **FR-VIS-205**: Support HUD layout presets (minimal, standard, detailed)

## Non-Functional Requirements

### Visual Quality Requirements
- **NFR-VIS-001**: Anti-aliased line rendering
- **NFR-VIS-002**: Smooth gradient transitions
- **NFR-VIS-003**: No visible flicker or tearing
- **NFR-VIS-004**: Consistent brightness across display
- **NFR-VIS-005**: Color accuracy within 10% of target
- **NFR-VIS-006**: Minimum contrast ratio of 4.5:1 for all overlays against table surface

### Performance Requirements
- **NFR-VIS-010**: Maintain 60 FPS rendering
- **NFR-VIS-011**: Frame render time < 16ms
- **NFR-VIS-012**: Memory footprint < 500MB
- **NFR-VIS-013**: < 30% GPU utilization typical
- **NFR-VIS-014**: Network latency < 50ms typical

### Debug HUD Requirements
- **NFR-VIS-030**: HUD rendering overhead < 2ms per frame
- **NFR-VIS-031**: HUD text legible at all supported resolutions
- **NFR-VIS-032**: HUD update rate at least 10 Hz for dynamic information

### Reliability Requirements
- **NFR-VIS-040**: No crashes during operation
- **NFR-VIS-041**: Graceful degradation under load
- **NFR-VIS-042**: Maintain calibration across restarts
- **NFR-VIS-043**: Handle network disconnection gracefully
- **NFR-VIS-044**: Automatic reconnection with exponential backoff

## Network Communication Protocol

The visualizer receives real-time updates via WebSocket. Messages are JSON-encoded.

### WebSocket Configuration
- **Default endpoint**: `ws://[api-host]:8000/api/v1/game/state/ws`
- **Message encoding**: UTF-8 JSON (text frames)
- **Reconnection**: Automatic with exponential backoff
- **Heartbeat**: Ping/pong every 30 seconds
- **Max reconnect delay**: 30 seconds
- **Initial retry delay**: 1 second

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

## Configuration Schema

The visualizer uses a JSON configuration file:

```json
{
  "network": {
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
    "opacity": 1.0,
    "layer": "background",
    "subscribe_on_start": true,
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
  },
  "colors": {
    "table_felt": [34, 139, 34],
    "auto_adapt": true,
    "trajectory_primary": null,
    "trajectory_secondary": null
  }
}
```

## Implementation Structure

```
visualizer/                     # Core Visualizer Module
├── init.lua                   # Module entry point and API
├── core/
│   ├── renderer.lua          # Core rendering engine
│   ├── state.lua             # State management
│   └── config.lua            # Configuration management
├── modules/
│   ├── calibration/
│   │   ├── init.lua         # Calibration module
│   │   ├── transform.lua    # Geometric transformations
│   │   └── interactive.lua  # Interactive calibration UI
│   ├── trajectory/
│   │   ├── init.lua         # Trajectory rendering module
│   │   ├── renderer.lua     # Trajectory drawing
│   │   └── effects.lua      # Visual effects
│   ├── debug_hud/
│   │   ├── init.lua         # Debug HUD module
│   │   ├── sections.lua     # HUD section components
│   │   └── metrics.lua      # Performance metrics
│   ├── network/
│   │   ├── init.lua         # Network abstraction
│   │   ├── websocket.lua    # WebSocket client
│   │   └── messages.lua     # Message parsing
│   └── colors/
│       ├── init.lua         # Color management
│       └── adaptive.lua     # Adaptive color generation
├── lib/
│   ├── json.lua            # JSON encoding/decoding
│   ├── websocket.lua       # WebSocket library
│   └── base64.lua          # Base64 decoding
└── config/
    └── default.json        # Default configuration
```

## Module API

The visualizer provides a clean API for embedding:

```lua
-- Initialize visualizer
local visualizer = require("visualizer")
visualizer.init(config)

-- Update loop (called every frame)
function love.update(dt)
  visualizer.update(dt)
end

-- Draw loop (called every frame)
function love.draw()
  visualizer.draw()
end

-- Cleanup
function love.quit()
  visualizer.cleanup()
end

-- Event handlers
function love.keypressed(key)
  visualizer.keypressed(key)
end
```

## Testing Requirements

### Unit Testing
- Test geometric transformation functions
- Validate calibration calculations
- Test rendering primitive functions
- Verify color space conversions
- Test adaptive color generation algorithms
- Validate contrast ratio calculations
- Test HUD rendering components
- Coverage target: 80%

### Integration Testing
- Test complete rendering pipeline
- Verify WebSocket data reception
- Test calibration procedures
- Validate configuration updates
- Test error recovery
- Verify HUD data collection and display

### Visual Testing
- Verify rendering quality
- Test all visual effects
- Validate color accuracy
- Check text rendering
- Test animation smoothness
- Validate HUD legibility and layout

### Performance Testing
- Benchmark frame rates
- Measure rendering latency
- Test GPU memory usage
- Profile CPU/GPU usage
- Stress test with complex scenes
- Test HUD rendering overhead

## Success Criteria

### Display Success Criteria
1. **Visual Quality**
   - Smooth, anti-aliased line rendering
   - No visible flicker or tearing
   - Consistent color reproduction
   - Clear text legibility at all sizes

2. **Alignment Accuracy**
   - Correct perspective at all positions
   - Accurate trajectory overlay on actual paths
   - Stable calibration without drift

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

4. **Debug HUD**
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

## Development Priorities

1. ✅ Implement basic rendering engine
2. ✅ Add line rendering primitives
3. ✅ Implement WebSocket communication
4. ✅ Add trajectory rendering
5. Implement calibration system
6. Add visual effects
8. Implement debug HUD overlay
9. Add adaptive color management
10. Optimize performance
11. Add advanced features
