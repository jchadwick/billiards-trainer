# Billiards Trainer API Integration

This directory contains the comprehensive WebSocket and REST API client integration for the Billiards Trainer frontend. The implementation provides real-time communication with the backend, state management, error handling, and connection monitoring.

## Architecture Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   React App     │    │   MobX Stores    │    │  API Services   │
│                 │◄──►│                  │◄──►│                 │
│ - Components    │    │ - AuthStore      │    │ - ApiClient     │
│ - Routes        │    │ - GameStateStore │    │ - WebSocketClient│
│ - UI Elements   │    │ - ConfigStore    │    │ - AuthService   │
└─────────────────┘    │ - SystemStore    │    │ - DataProcessor │
                       │ - UIStore        │    │ - ErrorHandler  │
                       └──────────────────┘    └─────────────────┘
                                ▲                        ▲
                                │                        │
                                ▼                        ▼
                       ┌──────────────────┐    ┌─────────────────┐
                       │   Root Store     │    │ Backend API     │
                       │                  │    │                 │
                       │ - Store Factory  │    │ - REST API      │
                       │ - Store Context  │    │ - WebSocket     │
                       │ - Global State   │    │ - Authentication│
                       └──────────────────┘    └─────────────────┘
```

## Core Components

### 1. API Service (`api-service.ts`)

The high-level service that orchestrates all API interactions.

**Features:**

- Unified interface for REST and WebSocket APIs
- Request caching and deduplication
- Loading state management
- Error handling and retry logic
- Authentication integration
- Connection monitoring

**Usage:**

```typescript
import { createApiService } from "./services/api-service";

const apiService = createApiService({
  apiBaseUrl: "http://localhost:8000",
  wsBaseUrl: "ws://localhost:8000/ws",
  enableCaching: true,
  autoConnectWebSocket: true,
});

// Authentication
await apiService.login({ username: "user", password: "pass" });

// Real-time data
apiService.onGameStateData((state) => {
  console.log("Game state update:", state);
});

// REST API calls
const health = await apiService.getHealth();
```

### 2. WebSocket Client (`websocket-client.ts`)

Advanced WebSocket client with auto-reconnection and message handling.

**Features:**

- Auto-reconnection with exponential backoff
- Message queuing during disconnection
- Heartbeat/ping-pong mechanism
- Connection quality monitoring
- Type-safe message handling
- Subscription management

**Usage:**

```typescript
import { createWebSocketClient } from "./services/websocket-client";

const wsClient = createWebSocketClient({
  url: "ws://localhost:8000/ws",
  token: "jwt-token",
  autoReconnect: true,
  maxReconnectAttempts: 10,
});

// Connect and subscribe
await wsClient.connect();
wsClient.subscribe(["frame", "state", "trajectory"]);

// Handle messages
wsClient.on("frame", (message) => {
  console.log("Frame received:", message.data);
});
```

### 3. REST API Client (`api-client.ts`)

Comprehensive REST API client with authentication and error handling.

**Features:**

- Automatic token management and refresh
- Request/response interceptors
- Retry logic with exponential backoff
- Request timeout handling
- Type-safe API responses
- Error classification

**Usage:**

```typescript
import { apiClient } from "./services/api-client";

// Set authentication
apiClient.setAuthToken("jwt-token");

// Make API calls
const config = await apiClient.getConfig();
const gameState = await apiClient.getCurrentGameState();

// Handle errors
try {
  await apiClient.updateConfig({ values: { camera: { enabled: true } } });
} catch (error) {
  if (isApiError(error)) {
    console.log("API Error:", error.code, error.message);
  }
}
```

### 4. Authentication Service (`auth-service.ts`)

Complete authentication and session management.

**Features:**

- JWT token management
- Automatic token refresh
- Session persistence
- Activity tracking
- Role-based permissions
- Logout handling

**Usage:**

```typescript
import { createAuthService } from "./services/auth-service";

const authService = createAuthService(apiClient, {
  persistAuth: true,
  autoRefresh: true,
  inactivityTimeout: 3600,
});

// Login
const user = await authService.login({
  username: "admin",
  password: "password",
});

// Check permissions
if (authService.hasPermission("config:write")) {
  // User can modify configuration
}
```

### 5. Data Processing Service (`data-handlers.ts`)

Real-time data processing and transformation.

**Features:**

- Frame rate limiting and quality filtering
- Ball tracking and movement analysis
- Trajectory smoothing and prediction
- Game state interpolation
- Alert processing and categorization
- Performance optimization

**Usage:**

```typescript
import { createDataProcessingService } from "./services/data-handlers";

const dataProcessor = createDataProcessingService({
  frameProcessing: {
    maxFps: 30,
    qualityThreshold: 50,
  },
  stateProcessing: {
    enablePrediction: true,
    confidenceThreshold: 0.7,
  },
});

// Handle processed data
dataProcessor.onGameState((processedState) => {
  console.log("Processed game state:", processedState);
  console.log("Ball movements:", processedState.changesSinceLastFrame);
});
```

### 6. Error Handling Service (`error-handler.ts`)

Comprehensive error handling and connection monitoring.

**Features:**

- Global error catching and reporting
- Error categorization and severity assessment
- Connection health monitoring
- Retry mechanisms
- Error statistics and analysis
- User-friendly error presentation

**Usage:**

```typescript
import { errorHandler, reportError } from "./services/error-handler";

// Report errors
reportError(new Error("Something went wrong"), {
  component: "game-view",
  action: "load_state",
});

// Monitor connection health
errorHandler.onConnectionHealth((health) => {
  if (health.status === "poor") {
    showConnectionWarning();
  }
});

// Retry failed operations
const result = await errorHandler.withRetry(() => {
  return apiClient.getGameState();
});
```

## MobX Store Integration

### Store Structure

```typescript
// Root store that contains all other stores
interface RootStore {
  authStore: AuthStore; // Authentication state
  gameStateStore: GameStateStore; // Real-time game data
  configStore: ConfigStore; // Configuration management
  calibrationStore: CalibrationStore; // Calibration process
  systemStore: SystemStore; // System health and monitoring
  uiStore: UIStore; // UI state and preferences
}
```

### Usage with React Components

```typescript
import { observer } from 'mobx-react-lite';
import { useStore } from './stores';

const GameView = observer(() => {
  const { gameStateStore, authStore } = useStore();

  // Reactive data access
  const currentState = gameStateStore.currentState;
  const isAuthenticated = authStore.isAuthenticated;

  // Actions
  const handleLogin = async () => {
    await authStore.login({ username, password });
  };

  return (
    <div>
      {isAuthenticated ? (
        <GameDisplay state={currentState} />
      ) : (
        <LoginForm onLogin={handleLogin} />
      )}
    </div>
  );
});
```

## Configuration

### Environment Variables

```env
# API Configuration
VITE_API_BASE_URL=http://localhost:8000

# Feature Flags
VITE_ENABLE_REAL_TIME=true
VITE_ENABLE_CACHING=true
VITE_DEBUG_MODE=false
```

### Service Configuration

```typescript
const config = {
  // API Service
  apiBaseUrl: "http://localhost:8000",
  wsBaseUrl: "ws://localhost:8000/ws",
  enableCaching: true,
  cacheTimeout: 300000,
  autoConnectWebSocket: true,
  defaultStreamSubscriptions: ["frame", "state", "trajectory", "alert"],

  // WebSocket Client
  maxReconnectAttempts: 10,
  reconnectDelay: 1000,
  maxReconnectDelay: 30000,
  heartbeatInterval: 30000,

  // Authentication
  persistAuth: true,
  autoRefresh: true,
  refreshThreshold: 300,
  inactivityTimeout: 3600,

  // Data Processing
  frameProcessing: {
    maxFps: 30,
    qualityThreshold: 50,
    enableCompression: true,
  },
  stateProcessing: {
    smoothingFactor: 0.3,
    confidenceThreshold: 0.7,
    enablePrediction: true,
  },
};
```

## Error Handling Patterns

### API Error Handling

```typescript
try {
  const result = await apiService.updateConfig(changes);
} catch (error) {
  if (isApiError(error)) {
    switch (error.status) {
      case 401:
        // Redirect to login
        authStore.logout();
        break;
      case 403:
        // Show permission error
        uiStore.showError("Insufficient permissions");
        break;
      case 422:
        // Show validation errors
        showValidationErrors(error.details);
        break;
      default:
        // Generic error
        uiStore.showError(error.message);
    }
  }
}
```

### Connection Error Recovery

```typescript
// Automatic retry with exponential backoff
const robustApiCall = async () => {
  return errorHandler.withRetry(() => apiService.getHealth(), {
    maxAttempts: 3,
    initialDelay: 1000,
    backoffFactor: 2,
    retryCondition: (error) => error.status >= 500,
  });
};
```

## Real-time Data Flow

### Message Processing Pipeline

```
WebSocket Message → Data Processor → MobX Store → React Component
      ↓                   ↓              ↓             ↓
  Raw JSON         Processed Data   Reactive State   UI Update
```

### Example: Frame Processing

```typescript
// 1. Raw WebSocket message
{
  type: 'frame',
  timestamp: '2024-01-01T12:00:00Z',
  data: {
    image: 'base64-encoded-jpeg',
    width: 1920,
    height: 1080,
    fps: 30
  }
}

// 2. Processed frame data
{
  imageUrl: 'data:image/jpeg;base64,...',
  displayWidth: 1920,
  displayHeight: 1080,
  aspectRatio: 1.78,
  processedAt: Date
}

// 3. Store update triggers React re-render
gameStateStore.currentFrame = processedFrame;
```

## Performance Considerations

### Optimization Strategies

1. **Frame Rate Limiting**: Limit processing to target FPS
2. **Quality Filtering**: Skip low-quality frames
3. **Request Deduplication**: Avoid duplicate API calls
4. **Response Caching**: Cache API responses with TTL
5. **Connection Pooling**: Reuse WebSocket connections
6. **Error Rate Limiting**: Prevent error spam

### Memory Management

```typescript
// Automatic cleanup of old data
const maxHistoryLength = 100;
const maxFrameBuffer = 10;

// Periodic cleanup
setInterval(() => {
  gameStateStore.cleanupOldData();
  errorHandler.clearResolvedErrors();
}, 60000);
```

## Testing

### Unit Tests

```bash
npm test services/
```

### Integration Tests

```bash
npm test services/__tests__/api-integration.test.ts
```

### Manual Testing

```typescript
import { runApiIntegrationDemo } from "./examples/api-integration-demo";

// Run comprehensive demo
runApiIntegrationDemo();
```

## Development Workflow

### Setup

1. Install dependencies: `npm install`
2. Configure environment variables
3. Start development server: `npm run dev`
4. Initialize API services in your app

### Adding New Features

1. Define TypeScript interfaces in `types/api.ts`
2. Implement API client methods
3. Add WebSocket message handlers
4. Create or update MobX stores
5. Add error handling
6. Write tests
7. Update documentation

### Debugging

```typescript
// Enable debug mode
if (import.meta.env.DEV) {
  // Global store access for debugging
  window.__MOBX_STORES__ = stores;

  // Enable detailed logging
  console.log("API Services initialized");
}
```

## Deployment Considerations

### Production Configuration

- Enable authentication persistence
- Set appropriate cache timeouts
- Configure retry limits
- Enable error reporting
- Set up monitoring alerts

### Security

- Use HTTPS/WSS in production
- Validate all user inputs
- Implement rate limiting
- Secure token storage
- Regular security audits

## Support and Maintenance

### Monitoring

- Connection health metrics
- Error rates and types
- Performance statistics
- User activity tracking

### Troubleshooting

1. Check browser console for errors
2. Verify network connectivity
3. Validate authentication tokens
4. Review WebSocket connection state
5. Check API service health

For additional support, see the main project documentation or contact the development team.
