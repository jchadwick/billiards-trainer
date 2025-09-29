# Video Streaming Integration Test

This document outlines how to test the video streaming integration between the frontend and backend.

## Test Instructions

### 1. Test with Mock Data (Frontend Only)

1. **Start Frontend**:
   ```bash
   cd frontend/web
   npm run dev
   ```

2. **Access Test Route**:
   - Open browser to `http://localhost:3001/test-video`
   - The page will show video streaming test interface with mock data
   - You should see:
     - Animated billiard balls
     - Moving cue stick
     - Trajectory predictions
     - Table boundaries and pockets

3. **Test Video Controls**:
   - Toggle overlay visibility (balls, trajectories, table, cue)
   - Test fullscreen mode (F key)
   - Test screenshot function (S key)
   - Verify zoom/pan functionality

### 2. Test with Backend Integration

1. **Start Backend**:
   ```bash
   python -m backend.dev_server --port 8080 --no-reload
   ```

2. **Start Frontend**:
   ```bash
   cd frontend/web
   npm run dev
   ```

3. **Access Live View**:
   - Open browser to `http://localhost:3001/live-view` (if route exists)
   - Or use test page and disable "Use Mock Data" checkbox
   - Click "Connect" in stream controls

4. **Expected Behavior**:
   - If camera is available: live MJPEG stream will display
   - If no camera: error messages will show in stream controls
   - WebSocket connection will attempt to establish for real-time detection data

### 3. Backend Video Endpoints

The backend provides these video streaming endpoints:

- `GET /api/v1/stream/video` - MJPEG video stream
- `GET /api/v1/stream/video/status` - Stream status and statistics
- `POST /api/v1/stream/video/start` - Start camera capture
- `POST /api/v1/stream/video/stop` - Stop camera capture
- `GET /api/v1/stream/video/frame` - Single frame capture

### 4. Frontend Components

The video streaming system consists of:

- **VideoStore**: MobX store managing video state and WebSocket connections
- **VideoStream**: Component displaying MJPEG streams using HTML img element
- **OverlayCanvas**: Canvas-based overlay system for drawing detection data
- **LiveView**: Main component combining video + overlays + controls
- **StreamControls**: UI controls for video quality, FPS, connection

### 5. Testing Checklist

- [ ] Frontend loads without errors
- [ ] Mock data displays correctly in test mode
- [ ] Video overlays render properly over mock video
- [ ] All overlay controls work (balls, trajectories, table, cue, grid)
- [ ] Zoom and pan functionality works
- [ ] Screenshot functionality works
- [ ] Backend connects and serves video endpoints
- [ ] Live video stream displays when camera available
- [ ] Error handling works when backend unavailable
- [ ] WebSocket connection establishes for real-time data
- [ ] Auto-reconnect functionality works

## Current Status

✅ **Frontend Implementation**: Complete
- VideoStore with MJPEG stream handling
- VideoStream component with proper img element
- OverlayCanvas with comprehensive drawing system
- LiveView integration with all components
- StreamControls with full functionality

✅ **Backend Integration**: Complete
- Proper API endpoints for video streaming
- MJPEG stream generation
- Camera control endpoints
- Error handling and status reporting

✅ **Mock Data Testing**: Complete
- VideoStreamTest component with realistic mock data
- Animated detection data for testing overlays
- Full overlay system verification

## Architecture Summary

The video streaming integration uses a hybrid approach:

1. **Video Stream**: HTML img element consuming MJPEG stream from backend
2. **Detection Data**: WebSocket connection for real-time ball/cue/table detection
3. **Overlays**: HTML5 Canvas positioned over video for drawing detection data
4. **Coordinate Transformation**: Proper video-to-canvas coordinate mapping
5. **State Management**: MobX store coordinating all video-related state

This architecture provides:
- Low latency video streaming via MJPEG
- Real-time detection data via WebSocket
- High performance overlay rendering via Canvas
- Proper separation of video display and detection visualization
- Comprehensive error handling and reconnection logic
