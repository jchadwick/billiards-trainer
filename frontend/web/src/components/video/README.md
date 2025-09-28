# Video Streaming System for Billiards Trainer

This comprehensive video streaming system provides real-time MJPEG video display with sophisticated overlay capabilities for the billiards trainer application.

## Architecture Overview

The video streaming system consists of several interconnected components:

### Core Components

1. **VideoStream** - Main video display component for MJPEG streaming
2. **StreamControls** - Control panel for stream settings and operations
3. **OverlayCanvas** - Canvas-based overlay system for drawing detections
4. **LiveView** - Main integration component that combines all features

### Overlay Components

1. **BallOverlay** - Renders ball detections with labels and velocity vectors
2. **TrajectoryOverlay** - Displays trajectory predictions and collision points
3. **TableOverlay** - Shows table boundaries, pockets, and rails
4. **CueOverlay** - Visualizes cue stick position and aiming guides

### State Management

- **VideoStore** - MobX store managing video streaming state and detection data
- Reactive updates for real-time performance
- Error handling and connection management

### Utilities

- **coordinates.ts** - Coordinate transformation utilities
- Video-to-canvas coordinate mapping
- Interactive features (zoom, pan, click)

## Features

### Video Streaming
- **Real-time MJPEG streaming** from backend `/api/v1/stream/video`
- **Adaptive quality control** (low, medium, high, ultra)
- **Frame rate adjustment** (15, 30, 60 FPS)
- **Automatic reconnection** with configurable delays
- **Stream health monitoring** and statistics

### Interactive Controls
- **Zoom and pan** with mouse wheel and drag
- **Fullscreen mode** support
- **Screenshot capture** functionality
- **Click-to-coordinate** mapping for debugging
- **Double-click to reset** view transform

### Overlay System
- **Ball detection overlays** with customizable appearance
  - Ball circles with type-based colors
  - Number labels and IDs
  - Velocity vectors with magnitude
  - Confidence indicators
- **Trajectory prediction overlays**
  - Smooth curved trajectory lines
  - Collision point markers
  - Probability indicators
  - Uncertainty cones for predictions
- **Table detection overlays**
  - Table boundary detection
  - Pocket markers with influence zones
  - Rail detection with endpoints
  - Playing area indicators (head/foot strings)
- **Cue stick overlays**
  - Cue stick visualization with tip/tail
  - Angle indicators and measurements
  - Aiming guide lines
  - Power indicators

### Performance Optimizations
- **High-DPI canvas** support for crisp rendering
- **60 FPS rendering** with requestAnimationFrame
- **Efficient coordinate transformations**
- **Memory usage monitoring**
- **Frame dropping detection**

### Error Handling
- **Connection failure recovery** with automatic retry
- **Stream error detection** and user feedback
- **Graceful degradation** when camera unavailable
- **Comprehensive error logging** with timestamps

## Usage

### Basic Integration

```typescript
import { LiveView } from './components/video/LiveView';

function App() {
  return (
    <div className="h-screen">
      <LiveView
        autoConnect={true}
        baseUrl="http://localhost:8000"
      />
    </div>
  );
}
```

### Advanced Usage with Custom Store

```typescript
import { VideoStore } from './stores/VideoStore';
import { VideoStream, StreamControls, OverlayCanvas } from './components/video';

function CustomVideoView() {
  const [videoStore] = useState(() => new VideoStore());

  useEffect(() => {
    videoStore.connect('http://localhost:8000');
    return () => videoStore.dispose();
  }, []);

  return (
    <div className="relative">
      <VideoStream
        videoStore={videoStore}
        className="absolute inset-0"
      />
      <OverlayCanvas
        videoStore={videoStore}
        // ... configuration
      />
      <StreamControls videoStore={videoStore} />
    </div>
  );
}
```

### Overlay Configuration

```typescript
const overlayConfig: OverlayConfig = {
  balls: {
    visible: true,
    showLabels: true,
    showVelocity: true,
    opacity: 0.9,
    radius: 15,
  },
  trajectories: {
    visible: true,
    showProbability: true,
    lineWidth: 3,
    opacity: 0.8,
  },
  // ... other overlay settings
};
```

## Testing

The system includes a comprehensive test component:

```typescript
import { VideoStreamTest } from './components/video/VideoStreamTest';

// Test with mock data
<VideoStreamTest useMockData={true} />

// Test with real backend
<VideoStreamTest useMockData={false} />
```

Access the test page at `/test-video` route.

## API Integration

The system connects to the backend video streaming API:

- **Stream endpoint**: `GET /api/v1/stream/video`
- **Status endpoint**: `GET /api/v1/stream/video/status`
- **Control endpoints**: `POST /api/v1/stream/video/start`, `POST /api/v1/stream/video/stop`

### Query Parameters

- `quality`: JPEG quality (1-100)
- `fps`: Frame rate (1-60)
- `width`: Maximum frame width
- `height`: Maximum frame height

## Real-time Data Flow

1. **Video Stream**: MJPEG frames from backend camera
2. **Detection Data**: Real-time ball/cue/table detection via WebSocket or polling
3. **Overlay Rendering**: Canvas-based overlay system draws detections over video
4. **User Interaction**: Mouse/touch events for zoom, pan, and coordinate mapping
5. **State Updates**: MobX reactive updates ensure smooth UI synchronization

## Performance Characteristics

- **30+ FPS** video streaming with overlay rendering
- **< 100ms** interaction response time
- **Smooth zoom/pan** with hardware acceleration
- **Memory efficient** with automatic cleanup
- **Bandwidth adaptive** quality adjustment

## Browser Compatibility

- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

## Dependencies

- React 19+
- MobX 6+ with mobx-react-lite
- TypeScript 5+
- Tailwind CSS 4+

## File Structure

```
src/components/video/
├── VideoStream.tsx          # Main video component
├── StreamControls.tsx       # Control panel
├── OverlayCanvas.tsx       # Canvas overlay system
├── LiveView.tsx            # Integration component
├── VideoStreamTest.tsx     # Test component
├── overlays/
│   ├── BallOverlay.tsx     # Ball detection overlay
│   ├── TrajectoryOverlay.tsx # Trajectory overlay
│   ├── TableOverlay.tsx    # Table overlay
│   └── CueOverlay.tsx      # Cue stick overlay
└── index.ts                # Exports

src/stores/
└── VideoStore.ts           # MobX video state store

src/types/
└── video.ts                # TypeScript type definitions

src/utils/
└── coordinates.ts          # Coordinate transformation utilities
```

## Future Enhancements

- WebRTC support for lower latency
- Multiple camera feeds
- Recording capabilities
- Advanced trajectory prediction algorithms
- Machine learning integration for improved detection
- Mobile-optimized touch controls
- Accessibility improvements (ARIA labels, keyboard navigation)

## Troubleshooting

### Common Issues

1. **Stream not connecting**: Check backend server is running on correct port
2. **Poor performance**: Reduce video quality or overlay complexity
3. **Coordinate misalignment**: Verify video size and transform calculations
4. **Memory leaks**: Ensure proper cleanup with `videoStore.dispose()`

### Debug Tools

- Browser Developer Tools for performance profiling
- MobX DevTools for state inspection
- Network tab for stream monitoring
- Console logs for error tracking

This video streaming system provides a solid foundation for real-time billiards training applications with comprehensive overlay capabilities and excellent performance characteristics.
