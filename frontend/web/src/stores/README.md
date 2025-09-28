# MobX State Management System

This document provides comprehensive documentation for the MobX-based state management system in the Billiards Trainer frontend application.

## Overview

The state management system is built around MobX and provides reactive state management for the entire application. It's organized into specialized stores that handle different aspects of the application:

- **SystemStore** - Overall system state, WebSocket connections, health monitoring
- **GameStore** - Game state, ball tracking, shot history
- **VisionStore** - Camera management, detection, calibration
- **ConfigStore** - Application configuration and user preferences
- **AuthStore** - Authentication and user management
- **UIStore** - UI state, modals, notifications, loading states

## Architecture

```
RootStore
├── SystemStore (system health, websocket)
├── GameStore (game state, balls, table)
├── VisionStore (camera, detection, calibration)
├── ConfigStore (settings, profiles)
├── AuthStore (authentication, users)
└── UIStore (UI state, notifications)
```

## Quick Start

### 1. Basic Setup

The stores are automatically initialized in `main.tsx` and available throughout the application via React context:

```tsx
import { useGameStore, useAuth, useNotifications } from './stores';

function MyComponent() {
  const gameStore = useGameStore();
  const { isAuthenticated, user } = useAuth();
  const { showSuccess } = useNotifications();

  // Your component logic here
}
```

### 2. Using Observers

Components that need to react to store changes should be wrapped with the `observer` HOC:

```tsx
import { observer } from 'mobx-react-lite';
import { useGameStore } from './stores';

const GameStatus = observer(() => {
  const { gameState, balls, isGameActive } = useGameStore();

  return (
    <div>
      <h2>Game Status: {isGameActive ? 'Active' : 'Inactive'}</h2>
      <p>Balls on table: {balls.filter(b => !b.isPocketed).length}</p>
      <p>Current player: {gameState.currentPlayer}</p>
    </div>
  );
});
```

## Store Usage Examples

### SystemStore Examples

```tsx
import { useSystemStatus } from './stores';

// Monitor system health
const SystemHealth = observer(() => {
  const { isHealthy, status, errors, connect, disconnect } = useSystemStatus();

  return (
    <div>
      <div className={`status ${isHealthy ? 'healthy' : 'unhealthy'}`}>
        Status: {status.websocketStatus}
      </div>

      {!status.isConnected && (
        <button onClick={() => connect('ws://localhost:8080/ws')}>
          Connect to Backend
        </button>
      )}

      {errors.length > 0 && (
        <div className="errors">
          {errors.map(error => (
            <div key={error.id} className={`error ${error.level}`}>
              {error.message}
            </div>
          ))}
        </div>
      )}
    </div>
  );
});
```

### GameStore Examples

```tsx
import { useGameState } from './stores';

// Game controls
const GameControls = observer(() => {
  const {
    isGameActive,
    gameState,
    startNewGame,
    pauseGame,
    resumeGame,
    endGame,
    resetTable
  } = useGameState();

  const handleStartGame = async () => {
    const result = await startNewGame('eightball', [
      { name: 'Player 1' },
      { name: 'Player 2' }
    ]);

    if (result.success) {
      console.log('Game started successfully');
    }
  };

  return (
    <div className="game-controls">
      {!isGameActive ? (
        <button onClick={handleStartGame}>Start New Game</button>
      ) : (
        <>
          {gameState.isPaused ? (
            <button onClick={resumeGame}>Resume</button>
          ) : (
            <button onClick={pauseGame}>Pause</button>
          )}
          <button onClick={endGame}>End Game</button>
        </>
      )}
      <button onClick={resetTable}>Reset Table</button>
    </div>
  );
});

// Ball tracking
const BallTracker = observer(() => {
  const { balls, activeBalls, cueBall } = useGameState();

  return (
    <div className="ball-tracker">
      <h3>Ball Status</h3>
      <div>Active balls: {activeBalls.length}</div>
      <div>Cue ball position: ({cueBall?.position.x}, {cueBall?.position.y})</div>

      <div className="ball-list">
        {balls.map(ball => (
          <div key={ball.id} className={`ball ${ball.type} ${ball.isPocketed ? 'pocketed' : ''}`}>
            Ball {ball.id}: {ball.isPocketed ? 'Pocketed' : 'On table'}
          </div>
        ))}
      </div>
    </div>
  );
});
```

### VisionStore Examples

```tsx
import { useVision } from './stores';

// Camera controls
const CameraControls = observer(() => {
  const {
    cameras,
    selectedCamera,
    isConnected,
    isDetecting,
    discoverCameras,
    selectCamera,
    startDetection,
    stopDetection
  } = useVision();

  useEffect(() => {
    discoverCameras();
  }, []);

  return (
    <div className="camera-controls">
      <h3>Camera Controls</h3>

      <select
        value={selectedCamera?.id || ''}
        onChange={(e) => selectCamera(e.target.value)}
      >
        <option value="">Select Camera</option>
        {cameras.map(camera => (
          <option key={camera.id} value={camera.id}>
            {camera.name} {camera.isConnected ? '✓' : '✗'}
          </option>
        ))}
      </select>

      {isConnected && (
        <div>
          {isDetecting ? (
            <button onClick={stopDetection}>Stop Detection</button>
          ) : (
            <button onClick={startDetection}>Start Detection</button>
          )}
        </div>
      )}
    </div>
  );
});

// Calibration interface
const CalibrationInterface = observer(() => {
  const {
    isCalibrated,
    isCalibrating,
    calibrationData,
    startCalibration,
    cancelCalibration
  } = useVision();

  return (
    <div className="calibration">
      <h3>Camera Calibration</h3>

      <div className={`status ${isCalibrated ? 'calibrated' : 'not-calibrated'}`}>
        {isCalibrated ? 'Camera is calibrated' : 'Camera needs calibration'}
      </div>

      {!isCalibrating ? (
        <button onClick={startCalibration}>Start Calibration</button>
      ) : (
        <button onClick={cancelCalibration}>Cancel Calibration</button>
      )}

      {calibrationData && (
        <div className="calibration-info">
          <p>Calibrated: {calibrationData.timestamp.toLocaleString()}</p>
          <p>Accuracy: {(calibrationData.isValid ? 'Valid' : 'Invalid')}</p>
        </div>
      )}
    </div>
  );
});
```

### ConfigStore Examples

```tsx
import { useConfig } from './stores';

// Settings panel
const SettingsPanel = observer(() => {
  const {
    config,
    profiles,
    currentProfile,
    hasUnsavedChanges,
    updateCameraConfig,
    updateGameConfig,
    saveProfile,
    loadProfile
  } = useConfig();

  return (
    <div className="settings-panel">
      <h3>Settings</h3>

      {hasUnsavedChanges && (
        <div className="unsaved-warning">
          You have unsaved changes
        </div>
      )}

      {/* Profile selection */}
      <div className="profile-section">
        <label>Profile:</label>
        <select
          value={currentProfile}
          onChange={(e) => loadProfile(e.target.value)}
        >
          {profiles.map(profile => (
            <option key={profile} value={profile}>{profile}</option>
          ))}
        </select>
      </div>

      {/* Camera settings */}
      <div className="camera-settings">
        <h4>Camera Settings</h4>
        <label>
          FPS:
          <input
            type="number"
            value={config.camera.fps}
            onChange={(e) => updateCameraConfig({ fps: parseInt(e.target.value) })}
          />
        </label>
        <label>
          Brightness:
          <input
            type="range"
            min="0"
            max="1"
            step="0.1"
            value={config.camera.brightness}
            onChange={(e) => updateCameraConfig({ brightness: parseFloat(e.target.value) })}
          />
        </label>
      </div>

      {/* Game settings */}
      <div className="game-settings">
        <h4>Game Settings</h4>
        <label>
          Default Game Type:
          <select
            value={config.game.defaultGameType}
            onChange={(e) => updateGameConfig({ defaultGameType: e.target.value as any })}
          >
            <option value="practice">Practice</option>
            <option value="eightball">8-Ball</option>
            <option value="nineball">9-Ball</option>
          </select>
        </label>
      </div>

      <button onClick={() => saveProfile('custom')}>
        Save as Custom Profile
      </button>
    </div>
  );
});
```

### AuthStore Examples

```tsx
import { useAuth } from './stores';

// Login form
const LoginForm = observer(() => {
  const { login, isLoading, error } = useAuth();
  const [credentials, setCredentials] = useState({ username: '', password: '' });

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const result = await login(credentials);
    if (result.success) {
      console.log('Login successful');
    }
  };

  return (
    <form onSubmit={handleSubmit} className="login-form">
      <h3>Login</h3>

      {error && <div className="error">{error}</div>}

      <input
        type="text"
        placeholder="Username"
        value={credentials.username}
        onChange={(e) => setCredentials(prev => ({ ...prev, username: e.target.value }))}
      />

      <input
        type="password"
        placeholder="Password"
        value={credentials.password}
        onChange={(e) => setCredentials(prev => ({ ...prev, password: e.target.value }))}
      />

      <button type="submit" disabled={isLoading}>
        {isLoading ? 'Logging in...' : 'Login'}
      </button>
    </form>
  );
});

// User profile
const UserProfile = observer(() => {
  const { isAuthenticated, user, logout } = useAuth();

  if (!isAuthenticated || !user) {
    return <div>Not logged in</div>;
  }

  return (
    <div className="user-profile">
      <h3>Welcome, {user.username}</h3>
      <p>Email: {user.email}</p>
      <p>Roles: {user.roles.join(', ')}</p>
      <p>Last login: {user.lastLogin.toLocaleString()}</p>

      <button onClick={logout}>Logout</button>
    </div>
  );
});
```

### UIStore Examples

```tsx
import { useNotifications, useModals, useLoading } from './stores';

// Notification system
const NotificationCenter = observer(() => {
  const {
    notifications,
    unreadCount,
    markAsRead,
    markAllAsRead,
    remove,
    clearAll
  } = useNotifications();

  return (
    <div className="notification-center">
      <div className="notification-header">
        <h3>Notifications ({unreadCount})</h3>
        <button onClick={markAllAsRead}>Mark All Read</button>
        <button onClick={clearAll}>Clear All</button>
      </div>

      <div className="notification-list">
        {notifications.map(notification => (
          <div
            key={notification.id}
            className={`notification ${notification.type} ${notification.isRead ? 'read' : 'unread'}`}
          >
            <div className="notification-content">
              <h4>{notification.title}</h4>
              <p>{notification.message}</p>
              <small>{notification.timestamp.toLocaleString()}</small>
            </div>
            <div className="notification-actions">
              {!notification.isRead && (
                <button onClick={() => markAsRead(notification.id)}>
                  Mark Read
                </button>
              )}
              <button onClick={() => remove(notification.id)}>
                Remove
              </button>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
});

// Modal system
const ModalManager = observer(() => {
  const { modals, close } = useModals();

  return (
    <>
      {modals.settings && (
        <div className="modal-overlay" onClick={() => close('settings')}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <h2>Settings</h2>
            {/* Settings content */}
            <button onClick={() => close('settings')}>Close</button>
          </div>
        </div>
      )}

      {modals.calibration && (
        <div className="modal-overlay" onClick={() => close('calibration')}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <h2>Camera Calibration</h2>
            {/* Calibration content */}
            <button onClick={() => close('calibration')}>Close</button>
          </div>
        </div>
      )}
    </>
  );
});

// Loading indicators
const LoadingManager = observer(() => {
  const { loading, isGlobalLoading } = useLoading();

  return (
    <>
      {isGlobalLoading && (
        <div className="global-loading">
          <div className="spinner">Loading...</div>
        </div>
      )}

      {loading.calibration && (
        <div className="calibration-loading">
          Calibrating camera...
        </div>
      )}

      {loading.detection && (
        <div className="detection-loading">
          Starting detection...
        </div>
      )}
    </>
  );
});
```

## Advanced Usage

### Custom Hooks

You can create custom hooks that combine multiple stores:

```tsx
import { useGameStore, useVisionStore, useSystemStore } from './stores';

// Custom hook for game session management
export function useGameSession() {
  const gameStore = useGameStore();
  const visionStore = useVisionStore();
  const systemStore = useSystemStore();

  const startGameSession = async () => {
    // Start vision detection
    await visionStore.startDetection();

    // Start a new game
    const result = await gameStore.startNewGame('eightball', [
      { name: 'Player 1' },
      { name: 'Player 2' }
    ]);

    return result;
  };

  const endGameSession = async () => {
    // Stop detection
    visionStore.stopDetection();

    // End game
    gameStore.endGame();
  };

  return {
    isSessionActive: gameStore.gameState.isActive && visionStore.isDetecting,
    startSession: startGameSession,
    endSession: endGameSession,
    gameState: gameStore.gameState,
    detectionRate: visionStore.detectionRate
  };
}
```

### Error Handling

```tsx
import { useErrorHandler } from './stores';

function MyComponent() {
  const { handleError, handleWarning } = useErrorHandler();

  const riskyOperation = async () => {
    try {
      // Some risky operation
      await someAsyncOperation();
    } catch (error) {
      handleError(error as Error, 'MyComponent.riskyOperation');
    }
  };

  return (
    <button onClick={riskyOperation}>
      Do Risky Thing
    </button>
  );
}
```

### Performance Monitoring

```tsx
import { usePerformance } from './stores';

const PerformanceMonitor = observer(() => {
  const {
    systemUptime,
    detectionRate,
    averageProcessingTime,
    frameCount
  } = usePerformance();

  return (
    <div className="performance-monitor">
      <h3>Performance</h3>
      <div>System Uptime: {Math.round(systemUptime / 1000)}s</div>
      <div>Detection Rate: {detectionRate.toFixed(1)} FPS</div>
      <div>Avg Processing Time: {averageProcessingTime.toFixed(1)}ms</div>
      <div>Total Frames: {frameCount}</div>
    </div>
  );
});
```

## Development Tools

In development mode, several debugging tools are available:

```javascript
// Available in browser console
__MOBX_DEBUG__.inspect()          // Inspect all store states
__MOBX_DEBUG__.reset()            // Reset all stores
__MOBX_DEBUG__.export()           // Export current state
__MOBX_DEBUG__.simulate.login()   // Simulate user login
__MOBX_DEBUG__.simulate.gameStart() // Simulate game start
__MOBX_DEBUG__.trace()            // Start MobX tracing

// Direct store access
__MOBX_STORES__.game.startNewGame('practice', [])
__MOBX_STORES__.ui.showSuccess('Test', 'Message')
```

## Best Practices

### 1. Use Observer Components

Always wrap components that read from stores with `observer`:

```tsx
// ✅ Good
const GameStatus = observer(() => {
  const { isGameActive } = useGameStore();
  return <div>Game is {isGameActive ? 'active' : 'inactive'}</div>;
});

// ❌ Bad - won't react to changes
const GameStatus = () => {
  const { isGameActive } = useGameStore();
  return <div>Game is {isGameActive ? 'active' : 'inactive'}</div>;
};
```

### 2. Keep Store Logic Simple

Store actions should be focused and atomic:

```tsx
// ✅ Good
const updateBallPosition = (ballId: number, position: Point2D) => {
  const ball = balls.find(b => b.id === ballId);
  if (ball) {
    ball.position = position;
  }
};

// ❌ Bad - too complex
const updateGameState = (data: any) => {
  // Complex logic mixing concerns
};
```

### 3. Use Computed Values

For derived state, use computed values:

```tsx
// ✅ Good
get activeBalls(): Ball[] {
  return this.balls.filter(ball => !ball.isPocketed);
}

// ❌ Bad - recalculates every time
const getActiveBalls = () => {
  return balls.filter(ball => !ball.isPocketed);
};
```

### 4. Handle Async Operations Properly

```tsx
// ✅ Good
async login(credentials: LoginCredentials): Promise<ActionResult> {
  runInAction(() => {
    this.isLoggingIn = true;
    this.loginError = null;
  });

  try {
    const result = await api.login(credentials);
    runInAction(() => {
      this.user = result.user;
      this.isAuthenticated = true;
      this.isLoggingIn = false;
    });
    return { success: true };
  } catch (error) {
    runInAction(() => {
      this.loginError = error.message;
      this.isLoggingIn = false;
    });
    return { success: false, error: error.message };
  }
}
```

## Testing

For testing components that use stores:

```tsx
import { render } from '@testing-library/react';
import { StoreProvider } from './stores';
import { RootStore } from './stores/RootStore';

// Create test store
const createTestStore = () => new RootStore();

// Test wrapper
const TestWrapper = ({ children, store }: { children: React.ReactNode; store?: RootStore }) => (
  <StoreProvider store={store || createTestStore()}>
    {children}
  </StoreProvider>
);

// Test example
test('game status displays correctly', () => {
  const testStore = createTestStore();
  testStore.game.startNewGame('practice', [{ name: 'Test Player' }]);

  render(<GameStatus />, { wrapper: (props) => <TestWrapper {...props} store={testStore} /> });

  expect(screen.getByText(/Game is active/)).toBeInTheDocument();
});
```

## Troubleshooting

### Common Issues

1. **Component not updating**: Make sure it's wrapped with `observer`
2. **State not persisting**: Check browser's localStorage quota
3. **WebSocket connection fails**: Verify backend is running and URL is correct
4. **Performance issues**: Use MobX DevTools to trace unnecessary re-renders

### Debug Commands

```javascript
// Check store health
__MOBX_DEBUG__.inspect()

// Monitor performance
__MOBX_DEBUG__.performance()

// Reset everything
__MOBX_DEBUG__.reset()

// Export state for debugging
console.log(__MOBX_DEBUG__.export())
```

## Migration Guide

If migrating from another state management solution:

1. Identify your current state structure
2. Map state to appropriate stores
3. Convert actions to store methods
4. Wrap components with `observer`
5. Replace selectors with computed values
6. Test thoroughly with the dev tools

This documentation provides a comprehensive guide to using the MobX state management system in the Billiards Trainer application. For more specific examples or troubleshooting, refer to the individual store files or use the development tools.
