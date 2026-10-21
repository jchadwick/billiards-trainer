# Billiards Visualizer

A data-driven LÖVE2D visualizer for billiards state and trajectory overlays. It receives supported game-state data over WebSocket; camera video is a separate delivery path.

## Status

The WebSocket client, state manager, trajectory normalization, and diagnostic HUD are implemented. Ball, cue, table, and video-feed rendering modules remain placeholders. Backend WebSocket video delivery is disabled, so this app does not receive camera frames over its WebSocket connection.

## Architecture

### What the Visualizer DOES:
- Render trajectory overlays from backend updates
- Show diagnostic HUD with connection/performance info
- Receive supported game-state and trajectory messages via WebSocket (no direct camera access)

### What the Visualizer DOES NOT DO:
- ❌ Render ball, cue, table, or camera-video modules; those modules are placeholders
- ❌ Access camera directly (backend handles all video)
- ❌ Recreate the scene from scratch

## Data Flow

```
Backend Vision System
    ↓ (WebSocket state and trajectory messages; no video frames)
MessageHandler (routes messages)
    ↓
StateManager (maintains game state)
    ↓
Modules (trajectory, overlays, HUD)
    ↓
LOVE2D Rendering
```

## Directory Structure

```
frontend/visualizer/
├── main.lua                     # Main entry point
├── conf.lua                     # LOVE2D configuration (windowed)
├── core/
│   ├── state_manager.lua        # Game state tracking from WebSocket
│   └── message_handler.lua      # WebSocket message routing
├── lib/
│   └── json.lua                 # JSON parsing
├── rendering/
│   └── calibration.lua          # Perspective transformation
└── modules/
    ├── trajectory/              # Trajectory visualization
    ├── table_overlay/           # Table boundaries and guides (stub)
    ├── diagnostic_hud/          # Comprehensive HUD (stub)
    ├── video_feed/              # Video feed display (stub)
    ├── ball_display/            # Ball highlighting helper (stub)
    └── cue_display/             # Cue stick display (stub)
```

## Running the Visualizer

```bash
# From the visualizer directory
cd frontend/visualizer
love .
```

### Controls

- **F1**: Toggle HUD visibility
- **F2**: Toggle debug mode
- **E**: Clear error messages
- **ESC**: Quit application

## Implementation Status

### ✅ Task Group 1: Core Setup (COMPLETE)
1. ✅ Main entry point (main.lua)
2. ✅ Configuration (conf.lua)
3. ✅ State manager (tracks balls, cue, table)
4. ✅ Message handler (routes WebSocket messages)
5. ✅ JSON library
6. ✅ Calibration module
7. ✅ Trajectory module (basic)

### 🔜 Task Group 2: WebSocket Integration (NEXT - 6-8 hours)
- Add WebSocket library (love2d-lua-websocket)
- Implement websocket_client.lua with auto-reconnect
- Integrate with MessageHandler
- Test connection with backend

### ⏳ Task Group 3: Visualization Modules (8-10 hours)
- Update trajectory module for AR overlays
- Create table_overlay module
- Update video_feed module

### ⏳ Task Group 4: Diagnostic HUD (10-12 hours)
- Implement comprehensive HUD with connection status
- Ball tracking display
- Cue detection display
- Performance metrics

### ⏳ Task Group 5: Video Feed Module (4-6 hours)
- WebSocket frame reception (base64 JPEG)
- Display modes (fullscreen/inset/overlay)

### ⏳ Task Groups 6-7: Projector Wrapper & Testing (7-10 hours)
- Native projector wrapper
- Integration testing
- Performance optimization

## WebSocket Message Protocol

The visualizer expects these message types from the backend:

### From Backend → Visualizer:
- `state`: Periodic ball positions (500ms intervals)
- `motion`: Immediate ball motion events
- `trajectory`: Trajectory predictions for AR overlay
- `frame`: Video frame data (base64 JPEG)
- `alert`: System alerts and warnings
- `config`: Configuration updates

### From Visualizer → Backend:
- `subscribe`: Request video feed subscription
- `unsubscribe`: Cancel video feed subscription

## Development Notes

- The visualizer is designed to run in windowed mode (1440x810) by default
- All game state comes from the backend - this is a pure visualization layer
- Module system allows easy addition of new overlays and visualizations
- Uses LOVE2D 11.5+ (cross-platform: Windows, Mac, Linux)

## Testing

### Basic Functionality Test (Task Group 1)
Run the visualizer and verify:
- ✅ Window opens at 1440x810 (windowed)
- ✅ Placeholder content displays
- ✅ HUD shows connection status (disconnected)
- ✅ FPS counter visible
- ✅ F1 toggles HUD
- ✅ ESC quits cleanly

### WebSocket Integration Test (Task Group 2+)
With backend running:
- WebSocket connects successfully
- State messages update ball positions
- Motion messages trigger immediate updates
- Trajectory messages display on overlay
- No sequence gaps in message stream

## Next Steps

1. **Implement WebSocket Client** (Task Group 2)
   - Add lua-websockets library
   - Create websocket_client.lua with auto-reconnect logic
   - Wire MessageHandler to receive and parse messages

2. **Connect to Backend** (Task Group 2)
   - Configure backend WebSocket URL
   - Test connection with real game state
   - Verify message routing works correctly

3. **Enhance Modules** (Task Group 3+)
   - Update trajectory module for AR overlay rendering
   - Create table overlay with boundaries and guides
   - Implement diagnostic HUD with comprehensive stats

## License

Part of the Billiards Trainer project.
