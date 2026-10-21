# Backend Integration Documentation

This document describes the frontend-backend integration implementation for the Billiards Trainer application.

## Overview

The frontend stores have been updated to replace mock data implementations with real backend API integration. This provides:

- Real-time system health monitoring
- Configuration persistence to backend
- Live video streaming from cameras, with optional direct MJPEG delivery
- WebSocket connections for real-time updates
- Proper error handling and fallback mechanisms

## Integration Components

### 1. API Client (`src/api/client.ts`)

A centralized HTTP client that handles all backend communication:

```typescript
import { apiClient } from '../api/client';

// Health checks
const health = await apiClient.getHealth(true, true);

// Configuration management
const config = await apiClient.getConfiguration();
await apiClient.updateConfiguration(configData);

// Video streaming
const streamUrl = apiClient.getVideoStreamUrl(quality, fps);
const status = await apiClient.getStreamStatus();
```

**Features:**
- Automatic error handling with proper response types
- WebSocket connection creation
- Video streaming URL generation

### 2. Store Integration

#### SystemStore
- **Real Health Monitoring**: `refreshMetrics()` and `getHealthStatus()` now call `/api/v1/health/`
- **Live Metrics**: `refreshMetrics()` fetches real CPU, memory, and network data
- **WebSocket Support**: Real WebSocket connections with message routing

#### ConfigStore
- **Backend Persistence**: `saveToBackend()` and `loadFromBackend()` methods
- **Import/Export**: Real file upload/download through backend APIs
- **Auto-sync**: Configuration changes automatically saved with debouncing
- **Note**: Profile management is handled client-side in browser localStorage (not synced to backend)

#### VideoStore
- **Real Streaming**: Existing consumers can connect to `/api/v1/stream/video`
- **Camera Control**: Start/stop video capture through backend
- **Stream Status**: Live monitoring of camera and streaming health
- **Quality Control**: Dynamic quality and FPS adjustment

#### ConnectionStore
- **Health Checks**: Tests actual backend connectivity
- **Retry Logic**: Smart reconnection with exponential backoff
- **Metrics**: Real connection performance data

### 3. WebSocket Message Routing

The RootStore now handles WebSocket message routing between stores:

```typescript
// Message types automatically routed:
- 'game_update' → GameStore
- 'detection_frame' → VisionStore + VideoStore
- 'config_update' → ConfigStore (reload)
- 'system_alert' → UIStore (notifications)
- 'vision_status' → VisionStore
```

### 4. Integration Testing

Use the `IntegrationTestPanel` component for testing:

```typescript
import IntegrationTestPanel from './components/debug/IntegrationTestPanel';

// In your development/debug page:
<IntegrationTestPanel className="mb-6" />
```

**Test Coverage:**
- API health checks
- System metrics retrieval
- Configuration access
- Video stream status
- WebSocket connections
- Store integration validation

## Backend API Endpoints Used

| Endpoint | Purpose | Store |
|----------|---------|--------|
| `GET /api/v1/health` | System health and status | SystemStore |
| `GET /api/v1/health/metrics` | Performance metrics | SystemStore |
| `GET /api/v1/config` | Load configuration | ConfigStore |
| `PUT /api/v1/config` | Save configuration | ConfigStore |
| `POST /api/v1/config/reset` | Reset to defaults | ConfigStore |
| `GET /api/v1/config/export` | Export configuration | ConfigStore |
| `POST /api/v1/config/import` | Import configuration | ConfigStore |
| `GET /api/v1/stream/video` | Video stream (MJPEG) | VideoStore |
| `GET /api/v1/stream/video/status` | Stream status | VideoStore |
| `POST /api/v1/stream/video/start` | Start capture | VideoStore |
| `POST /api/v1/stream/video/stop` | Stop capture | VideoStore |
| `GET /api/v1/stream/video/frame` | Single frame | VideoStore |
| `WS /api/v1/websocket/ws` | WebSocket connection | SystemStore |

## Error Handling

All API calls include proper error handling:

1. **Network Errors**: Caught and displayed with user-friendly messages
2. **Backend Errors**: HTTP error responses properly parsed and handled
3. **Fallback Data**: Mock/cached data used when backend unavailable
4. **Retry Logic**: Automatic reconnection for WebSocket and critical operations

## Configuration

Set the required `VITE_API_BASE_URL` environment variable to
`http://localhost:8000` for local development. The WebSocket base URL is
derived from this value, and clients connect at `/api/v1/websocket/ws`.
Set the optional runtime `api.stream.viewer_url` to an HTTP(S) MJPEG URL
reachable by viewer browsers. Live View loads that image feed directly; scene
and detection state continue over the API/WebSocket path. The URL must not
contain credentials. This can differ from the detector's `vision.camera.device_id`
URI, which may use loopback from the backend host. For example, set the detector
source to `http://127.0.0.1:8080/stream` and the viewer URL to the target's
LAN-reachable address, such as `http://<target-host>:8080/stream`. The detector
URI takes precedence over the legacy recorded-video path; clear that path when
using a numeric device index for live capture.

```typescript
// Custom base URL (if needed)
const customClient = new ApiClient(
  'http://your-backend:8000',
  'ws://your-backend:8000'
);
```

## Development Testing

1. **Start the backend server** on `localhost:8000`
2. **Run the frontend** with `npm run dev`
3. **Open the diagnostics page** to see the IntegrationTestPanel
4. **Click "Run Tests"** to verify all connections work

## Monitoring and Debugging

### Store State Inspection

All stores expose their connection state:

```typescript
// Check system health
console.log(rootStore.system.isHealthy);
console.log(rootStore.system.status);

// Check configuration sync
console.log(rootStore.config.hasUnsavedChanges);
console.log(rootStore.config.isValid);

// Check video connection
console.log(rootStore.vision.isConnected);
console.log(rootStore.vision.status);
```

### WebSocket Message Debugging

All WebSocket messages are logged in development:

```typescript
// Enable debug logging
localStorage.setItem('debug', 'billiards:*');
```

### Integration Test Results

The integration tester provides detailed results:

```typescript
import { integrationTester } from '../utils/integration-test';

const results = await integrationTester.runAllTests();
integrationTester.printResults();
```

## Troubleshooting

### Common Issues

1. **Backend Not Running**: Stores will show "disconnected" status and use fallback data
2. **CORS Issues**: Ensure backend has proper CORS configuration for `localhost:3000`
3. **WebSocket Failures**: Check firewall and proxy settings

### Debug Steps

1. **Check Network Tab**: Verify API calls are reaching the backend
2. **Check Console**: Look for error messages and failed requests
3. **Run Integration Tests**: Use the test panel to isolate issues
4. **Check Store State**: Inspect store status in browser dev tools

## Future Enhancements

- **Offline Mode**: Cache data for offline operation
- **Real-time Sync**: Conflict resolution for concurrent edits
- **Performance Optimization**: Request batching and caching
