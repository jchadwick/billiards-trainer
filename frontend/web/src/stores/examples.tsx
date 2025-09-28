/**
 * Example components demonstrating MobX store usage
 * These are complete, working examples that can be used as reference
 */

import React, { useState, useEffect } from 'react';
import { observer } from 'mobx-react-lite';
import {
  useAuth,
  useGameState,
  useVision,
  useConfig,
  useNotifications,
  useModals,
  useSystemStatus,
  useLoading,
  useResponsive
} from './context';

// 1. Authentication Example
export const LoginExample = observer(() => {
  const { isAuthenticated, user, login, logout, isLoading, error } = useAuth();
  const [credentials, setCredentials] = useState({ username: '', password: '' });

  if (isAuthenticated && user) {
    return (
      <div className="p-4 bg-green-50 border border-green-200 rounded">
        <h3 className="text-green-800 font-bold">Welcome, {user.username}!</h3>
        <p className="text-green-600">Email: {user.email}</p>
        <p className="text-green-600">Roles: {user.roles.join(', ')}</p>
        <button
          onClick={() => logout()}
          className="mt-2 px-4 py-2 bg-red-500 text-white rounded hover:bg-red-600"
        >
          Logout
        </button>
      </div>
    );
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    await login(credentials);
  };

  return (
    <form onSubmit={handleSubmit} className="p-4 space-y-4 bg-white border rounded">
      <h3 className="text-lg font-bold">Login</h3>

      {error && (
        <div className="p-3 bg-red-50 border border-red-200 rounded text-red-700">
          {error}
        </div>
      )}

      <input
        type="text"
        placeholder="Username (try: admin/admin or user/user)"
        value={credentials.username}
        onChange={(e) => setCredentials(prev => ({ ...prev, username: e.target.value }))}
        className="w-full p-2 border rounded"
        disabled={isLoading}
      />

      <input
        type="password"
        placeholder="Password"
        value={credentials.password}
        onChange={(e) => setCredentials(prev => ({ ...prev, password: e.target.value }))}
        className="w-full p-2 border rounded"
        disabled={isLoading}
      />

      <button
        type="submit"
        disabled={isLoading}
        className="w-full px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 disabled:opacity-50"
      >
        {isLoading ? 'Logging in...' : 'Login'}
      </button>
    </form>
  );
});

// 2. Game State Example
export const GameStateExample = observer(() => {
  const {
    gameState,
    isGameActive,
    balls,
    activeBalls,
    cueBall,
    startNewGame,
    pauseGame,
    resumeGame,
    endGame,
    resetTable
  } = useGameState();

  const { showSuccess, showInfo } = useNotifications();

  const handleStartGame = async () => {
    const result = await startNewGame('eightball', [
      { name: 'Player 1' },
      { name: 'Player 2' }
    ]);

    if (result.success) {
      showSuccess('Game Started', 'New 8-ball game started successfully!');
    }
  };

  const handleResetTable = () => {
    resetTable();
    showInfo('Table Reset', 'Billiard table has been reset to initial position');
  };

  return (
    <div className="p-4 space-y-4 bg-white border rounded">
      <h3 className="text-lg font-bold">Game State Management</h3>

      {/* Game Status */}
      <div className={`p-3 rounded ${isGameActive ? 'bg-green-50 border-green-200' : 'bg-gray-50 border-gray-200'} border`}>
        <h4 className="font-semibold">Status: {isGameActive ? 'Active' : 'Inactive'}</h4>
        {isGameActive && (
          <div className="mt-2 space-y-1 text-sm">
            <p>Game Type: {gameState.gameType}</p>
            <p>Current Player: {gameState.currentPlayer + 1}</p>
            <p>Shot Count: {gameState.shotCount}</p>
            <p>Paused: {gameState.isPaused ? 'Yes' : 'No'}</p>
          </div>
        )}
      </div>

      {/* Ball Information */}
      <div className="p-3 bg-blue-50 border border-blue-200 rounded">
        <h4 className="font-semibold">Ball Status</h4>
        <div className="mt-2 space-y-1 text-sm">
          <p>Total balls: {balls.length}</p>
          <p>Active balls: {activeBalls.length}</p>
          <p>Cue ball position: {cueBall ? `(${cueBall.position.x.toFixed(0)}, ${cueBall.position.y.toFixed(0)})` : 'Not found'}</p>
        </div>
      </div>

      {/* Game Controls */}
      <div className="space-y-2">
        {!isGameActive ? (
          <button
            onClick={handleStartGame}
            className="px-4 py-2 bg-green-500 text-white rounded hover:bg-green-600"
          >
            Start New Game
          </button>
        ) : (
          <div className="space-x-2">
            {gameState.isPaused ? (
              <button
                onClick={resumeGame}
                className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
              >
                Resume
              </button>
            ) : (
              <button
                onClick={pauseGame}
                className="px-4 py-2 bg-yellow-500 text-white rounded hover:bg-yellow-600"
              >
                Pause
              </button>
            )}
            <button
              onClick={endGame}
              className="px-4 py-2 bg-red-500 text-white rounded hover:bg-red-600"
            >
              End Game
            </button>
          </div>
        )}
        <button
          onClick={handleResetTable}
          className="block px-4 py-2 bg-gray-500 text-white rounded hover:bg-gray-600"
        >
          Reset Table
        </button>
      </div>
    </div>
  );
});

// 3. Vision System Example
export const VisionExample = observer(() => {
  const {
    cameras,
    selectedCamera,
    isConnected,
    isDetecting,
    isCalibrated,
    detectionRate,
    frameCount,
    discoverCameras,
    selectCamera,
    startDetection,
    stopDetection,
    startCalibration
  } = useVision();

  const { showInfo, showWarning } = useNotifications();

  useEffect(() => {
    discoverCameras();
  }, [discoverCameras]);

  const handleCameraSelect = async (cameraId: string) => {
    const result = await selectCamera(cameraId);
    if (result.success) {
      showInfo('Camera Selected', `Camera "${cameras.find(c => c.id === cameraId)?.name}" selected successfully`);
    }
  };

  const handleStartDetection = () => {
    startDetection();
    showInfo('Detection Started', 'Ball detection has been started');
  };

  const handleStopDetection = () => {
    stopDetection();
    showInfo('Detection Stopped', 'Ball detection has been stopped');
  };

  const handleStartCalibration = () => {
    if (!isConnected) {
      showWarning('Camera Required', 'Please select and connect a camera first');
      return;
    }
    startCalibration();
    showInfo('Calibration Started', 'Camera calibration process started');
  };

  return (
    <div className="p-4 space-y-4 bg-white border rounded">
      <h3 className="text-lg font-bold">Vision System</h3>

      {/* Camera Selection */}
      <div className="p-3 bg-gray-50 border rounded">
        <h4 className="font-semibold mb-2">Camera Selection</h4>
        <select
          value={selectedCamera?.id || ''}
          onChange={(e) => handleCameraSelect(e.target.value)}
          className="w-full p-2 border rounded"
        >
          <option value="">Select a camera...</option>
          {cameras.map(camera => (
            <option key={camera.id} value={camera.id}>
              {camera.name} {camera.isConnected ? '✓' : '✗'}
            </option>
          ))}
        </select>
      </div>

      {/* Status Indicators */}
      <div className="grid grid-cols-3 gap-4">
        <div className={`p-3 rounded text-center ${isConnected ? 'bg-green-50 text-green-700' : 'bg-red-50 text-red-700'}`}>
          <div className="font-semibold">Connected</div>
          <div>{isConnected ? 'Yes' : 'No'}</div>
        </div>
        <div className={`p-3 rounded text-center ${isCalibrated ? 'bg-green-50 text-green-700' : 'bg-yellow-50 text-yellow-700'}`}>
          <div className="font-semibold">Calibrated</div>
          <div>{isCalibrated ? 'Yes' : 'No'}</div>
        </div>
        <div className={`p-3 rounded text-center ${isDetecting ? 'bg-blue-50 text-blue-700' : 'bg-gray-50 text-gray-700'}`}>
          <div className="font-semibold">Detecting</div>
          <div>{isDetecting ? 'Yes' : 'No'}</div>
        </div>
      </div>

      {/* Performance Metrics */}
      {isDetecting && (
        <div className="p-3 bg-blue-50 border border-blue-200 rounded">
          <h4 className="font-semibold">Performance</h4>
          <div className="mt-2 space-y-1 text-sm">
            <p>Detection Rate: {detectionRate.toFixed(1)} FPS</p>
            <p>Frames Processed: {frameCount}</p>
          </div>
        </div>
      )}

      {/* Controls */}
      <div className="space-y-2">
        <button
          onClick={() => discoverCameras()}
          className="px-4 py-2 bg-gray-500 text-white rounded hover:bg-gray-600"
        >
          Refresh Cameras
        </button>

        {isConnected && (
          <>
            <div className="space-x-2">
              {!isDetecting ? (
                <button
                  onClick={handleStartDetection}
                  className="px-4 py-2 bg-green-500 text-white rounded hover:bg-green-600"
                >
                  Start Detection
                </button>
              ) : (
                <button
                  onClick={handleStopDetection}
                  className="px-4 py-2 bg-red-500 text-white rounded hover:bg-red-600"
                >
                  Stop Detection
                </button>
              )}

              {!isCalibrated && (
                <button
                  onClick={handleStartCalibration}
                  className="px-4 py-2 bg-orange-500 text-white rounded hover:bg-orange-600"
                >
                  Start Calibration
                </button>
              )}
            </div>
          </>
        )}
      </div>
    </div>
  );
});

// 4. Configuration Example
export const ConfigExample = observer(() => {
  const {
    config,
    profiles,
    currentProfile,
    hasUnsavedChanges,
    isValid,
    validationErrors,
    updateCameraConfig,
    updateGameConfig,
    updateUIConfig,
    saveProfile,
    loadProfile,
    resetToDefaults
  } = useConfig();

  const { showSuccess, showError } = useNotifications();

  const handleSaveProfile = async () => {
    const result = await saveProfile(`profile_${Date.now()}`);
    if (result.success) {
      showSuccess('Profile Saved', 'Configuration profile saved successfully');
    } else {
      showError('Save Failed', result.error || 'Failed to save profile');
    }
  };

  const handleLoadProfile = async (profileName: string) => {
    const result = await loadProfile(profileName);
    if (result.success) {
      showSuccess('Profile Loaded', `Profile "${profileName}" loaded successfully`);
    }
  };

  return (
    <div className="p-4 space-y-4 bg-white border rounded">
      <h3 className="text-lg font-bold">Configuration Management</h3>

      {/* Status */}
      <div className="flex items-center space-x-4">
        <div className={`px-3 py-1 rounded text-sm ${hasUnsavedChanges ? 'bg-yellow-100 text-yellow-800' : 'bg-green-100 text-green-800'}`}>
          {hasUnsavedChanges ? 'Unsaved Changes' : 'All Changes Saved'}
        </div>
        <div className={`px-3 py-1 rounded text-sm ${isValid ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'}`}>
          {isValid ? 'Valid Configuration' : 'Invalid Configuration'}
        </div>
      </div>

      {/* Validation Errors */}
      {validationErrors.length > 0 && (
        <div className="p-3 bg-red-50 border border-red-200 rounded">
          <h4 className="font-semibold text-red-800">Validation Errors:</h4>
          <ul className="mt-2 space-y-1">
            {validationErrors.map((error, index) => (
              <li key={index} className="text-sm text-red-700">• {error}</li>
            ))}
          </ul>
        </div>
      )}

      {/* Profile Management */}
      <div className="p-3 bg-gray-50 border rounded">
        <h4 className="font-semibold mb-2">Profile Management</h4>
        <div className="space-y-2">
          <select
            value={currentProfile}
            onChange={(e) => handleLoadProfile(e.target.value)}
            className="w-full p-2 border rounded"
          >
            {profiles.map(profile => (
              <option key={profile} value={profile}>{profile}</option>
            ))}
          </select>
          <div className="space-x-2">
            <button
              onClick={handleSaveProfile}
              className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
            >
              Save as New Profile
            </button>
            <button
              onClick={() => resetToDefaults()}
              className="px-4 py-2 bg-gray-500 text-white rounded hover:bg-gray-600"
            >
              Reset to Defaults
            </button>
          </div>
        </div>
      </div>

      {/* Camera Configuration */}
      <div className="p-3 bg-blue-50 border border-blue-200 rounded">
        <h4 className="font-semibold mb-2">Camera Settings</h4>
        <div className="space-y-3">
          <div>
            <label className="block text-sm font-medium">FPS: {config.camera.fps}</label>
            <input
              type="range"
              min="15"
              max="60"
              value={config.camera.fps}
              onChange={(e) => updateCameraConfig({ fps: parseInt(e.target.value) })}
              className="w-full"
            />
          </div>
          <div>
            <label className="block text-sm font-medium">Brightness: {config.camera.brightness.toFixed(2)}</label>
            <input
              type="range"
              min="0"
              max="1"
              step="0.01"
              value={config.camera.brightness}
              onChange={(e) => updateCameraConfig({ brightness: parseFloat(e.target.value) })}
              className="w-full"
            />
          </div>
        </div>
      </div>

      {/* Game Configuration */}
      <div className="p-3 bg-green-50 border border-green-200 rounded">
        <h4 className="font-semibold mb-2">Game Settings</h4>
        <div className="space-y-3">
          <div>
            <label className="block text-sm font-medium">Default Game Type</label>
            <select
              value={config.game.defaultGameType}
              onChange={(e) => updateGameConfig({ defaultGameType: e.target.value as any })}
              className="w-full p-2 border rounded"
            >
              <option value="practice">Practice</option>
              <option value="eightball">8-Ball</option>
              <option value="nineball">9-Ball</option>
              <option value="straight">Straight Pool</option>
            </select>
          </div>
          <div>
            <label className="block text-sm font-medium">Shot Timeout: {config.game.shotTimeout}s</label>
            <input
              type="range"
              min="30"
              max="300"
              value={config.game.shotTimeout}
              onChange={(e) => updateGameConfig({ shotTimeout: parseInt(e.target.value) })}
              className="w-full"
            />
          </div>
        </div>
      </div>

      {/* UI Configuration */}
      <div className="p-3 bg-purple-50 border border-purple-200 rounded">
        <h4 className="font-semibold mb-2">UI Settings</h4>
        <div className="space-y-3">
          <div>
            <label className="block text-sm font-medium">Theme</label>
            <select
              value={config.ui.theme}
              onChange={(e) => updateUIConfig({ theme: e.target.value as any })}
              className="w-full p-2 border rounded"
            >
              <option value="light">Light</option>
              <option value="dark">Dark</option>
              <option value="auto">Auto (System)</option>
            </select>
          </div>
          <div className="flex items-center space-x-2">
            <input
              type="checkbox"
              checked={config.ui.showDebugInfo}
              onChange={(e) => updateUIConfig({ showDebugInfo: e.target.checked })}
            />
            <label className="text-sm">Show Debug Information</label>
          </div>
        </div>
      </div>
    </div>
  );
});

// 5. Notification System Example
export const NotificationExample = observer(() => {
  const {
    notifications,
    unreadCount,
    showSuccess,
    showError,
    showWarning,
    showInfo,
    markAsRead,
    markAllAsRead,
    remove,
    clearAll
  } = useNotifications();

  const triggerTestNotifications = () => {
    showInfo('Info', 'This is an info notification');
    setTimeout(() => showSuccess('Success', 'This is a success notification'), 500);
    setTimeout(() => showWarning('Warning', 'This is a warning notification'), 1000);
    setTimeout(() => showError('Error', 'This is an error notification'), 1500);
  };

  return (
    <div className="p-4 space-y-4 bg-white border rounded">
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-bold">Notifications ({unreadCount} unread)</h3>
        <div className="space-x-2">
          <button
            onClick={triggerTestNotifications}
            className="px-3 py-1 text-sm bg-blue-500 text-white rounded hover:bg-blue-600"
          >
            Test Notifications
          </button>
          <button
            onClick={markAllAsRead}
            className="px-3 py-1 text-sm bg-gray-500 text-white rounded hover:bg-gray-600"
          >
            Mark All Read
          </button>
          <button
            onClick={clearAll}
            className="px-3 py-1 text-sm bg-red-500 text-white rounded hover:bg-red-600"
          >
            Clear All
          </button>
        </div>
      </div>

      <div className="space-y-2 max-h-96 overflow-y-auto">
        {notifications.length === 0 ? (
          <div className="text-center text-gray-500 py-8">
            No notifications yet. Click "Test Notifications" to see examples.
          </div>
        ) : (
          notifications.map(notification => (
            <div
              key={notification.id}
              className={`p-3 border rounded-lg ${
                notification.type === 'success' ? 'bg-green-50 border-green-200' :
                notification.type === 'error' ? 'bg-red-50 border-red-200' :
                notification.type === 'warning' ? 'bg-yellow-50 border-yellow-200' :
                'bg-blue-50 border-blue-200'
              } ${!notification.isRead ? 'border-l-4' : ''}`}
            >
              <div className="flex items-start justify-between">
                <div className="flex-1">
                  <div className="flex items-center space-x-2">
                    <h4 className="font-semibold">{notification.title}</h4>
                    {!notification.isRead && (
                      <span className="w-2 h-2 bg-blue-500 rounded-full"></span>
                    )}
                  </div>
                  <p className="text-sm text-gray-600 mt-1">{notification.message}</p>
                  <p className="text-xs text-gray-400 mt-2">{notification.timestamp.toLocaleString()}</p>
                </div>
                <div className="flex space-x-1 ml-4">
                  {!notification.isRead && (
                    <button
                      onClick={() => markAsRead(notification.id)}
                      className="text-xs px-2 py-1 bg-blue-500 text-white rounded hover:bg-blue-600"
                    >
                      Mark Read
                    </button>
                  )}
                  <button
                    onClick={() => remove(notification.id)}
                    className="text-xs px-2 py-1 bg-gray-500 text-white rounded hover:bg-gray-600"
                  >
                    Remove
                  </button>
                </div>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
});

// 6. System Status Example
export const SystemStatusExample = observer(() => {
  const {
    status,
    isHealthy,
    isConnected,
    errors,
    criticalErrors,
    connect,
    disconnect,
    clearErrors
  } = useSystemStatus();

  const { showInfo, showError } = useNotifications();

  const handleConnect = async () => {
    const result = await connect('ws://localhost:8080/ws');
    if (result.success) {
      showInfo('Connected', 'Successfully connected to backend system');
    } else {
      showError('Connection Failed', result.error || 'Failed to connect to backend');
    }
  };

  const handleDisconnect = () => {
    disconnect();
    showInfo('Disconnected', 'Disconnected from backend system');
  };

  return (
    <div className="p-4 space-y-4 bg-white border rounded">
      <h3 className="text-lg font-bold">System Status</h3>

      {/* Health Indicator */}
      <div className={`p-4 rounded-lg text-center ${isHealthy ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'}`}>
        <div className="text-2xl font-bold">
          {isHealthy ? '✅ System Healthy' : '❌ System Issues'}
        </div>
        <div className="text-sm mt-1">
          WebSocket: {status.websocketStatus} |
          Backend: {status.backendVersion || 'Unknown'} |
          Last heartbeat: {status.lastHeartbeat?.toLocaleTimeString() || 'Never'}
        </div>
      </div>

      {/* Connection Controls */}
      <div className="flex space-x-2">
        {!isConnected ? (
          <button
            onClick={handleConnect}
            className="px-4 py-2 bg-green-500 text-white rounded hover:bg-green-600"
          >
            Connect to Backend
          </button>
        ) : (
          <button
            onClick={handleDisconnect}
            className="px-4 py-2 bg-red-500 text-white rounded hover:bg-red-600"
          >
            Disconnect
          </button>
        )}

        {errors.length > 0 && (
          <button
            onClick={clearErrors}
            className="px-4 py-2 bg-gray-500 text-white rounded hover:bg-gray-600"
          >
            Clear Errors ({errors.length})
          </button>
        )}
      </div>

      {/* Error Display */}
      {criticalErrors.length > 0 && (
        <div className="p-3 bg-red-50 border border-red-200 rounded">
          <h4 className="font-semibold text-red-800">Critical Errors:</h4>
          <div className="mt-2 space-y-2">
            {criticalErrors.map(error => (
              <div key={error.id} className="text-sm">
                <div className="font-medium">[{error.component}] {error.message}</div>
                <div className="text-gray-500">{error.timestamp.toLocaleString()}</div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Recent Errors */}
      {errors.length > 0 && criticalErrors.length === 0 && (
        <div className="p-3 bg-yellow-50 border border-yellow-200 rounded">
          <h4 className="font-semibold text-yellow-800">Recent Errors ({errors.length}):</h4>
          <div className="mt-2 space-y-1 max-h-32 overflow-y-auto">
            {errors.slice(0, 5).map(error => (
              <div key={error.id} className="text-sm text-yellow-700">
                [{error.level}] {error.message}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
});

// 7. Responsive Design Example
export const ResponsiveExample = observer(() => {
  const { windowWidth, windowHeight, isMobile, isTablet, isDesktop } = useResponsive();

  return (
    <div className="p-4 space-y-4 bg-white border rounded">
      <h3 className="text-lg font-bold">Responsive Information</h3>

      <div className="grid grid-cols-2 gap-4">
        <div className="p-3 bg-gray-50 rounded">
          <h4 className="font-semibold">Viewport</h4>
          <p className="text-sm">Width: {windowWidth}px</p>
          <p className="text-sm">Height: {windowHeight}px</p>
        </div>

        <div className="p-3 bg-gray-50 rounded">
          <h4 className="font-semibold">Device Type</h4>
          <div className="space-y-1 text-sm">
            <div className={`${isMobile ? 'text-green-600 font-bold' : 'text-gray-400'}`}>
              📱 Mobile {isMobile && '(Current)'}
            </div>
            <div className={`${isTablet ? 'text-green-600 font-bold' : 'text-gray-400'}`}>
              📟 Tablet {isTablet && '(Current)'}
            </div>
            <div className={`${isDesktop ? 'text-green-600 font-bold' : 'text-gray-400'}`}>
              🖥️ Desktop {isDesktop && '(Current)'}
            </div>
          </div>
        </div>
      </div>

      <div className="p-3 bg-blue-50 border border-blue-200 rounded">
        <h4 className="font-semibold">Responsive Behavior</h4>
        <p className="text-sm mt-1">
          The UI automatically adapts based on screen size. Sidebar collapses on mobile/tablet,
          modals stack appropriately, and touch targets are optimized for each device type.
        </p>
      </div>
    </div>
  );
});

// 8. Combined Example Dashboard
export const ExampleDashboard = observer(() => {
  const { isAuthenticated } = useAuth();
  const { open, close, modals } = useModals();

  if (!isAuthenticated) {
    return (
      <div className="min-h-screen bg-gray-100 p-8">
        <div className="max-w-md mx-auto">
          <h1 className="text-3xl font-bold text-center mb-8">Billiards Trainer</h1>
          <LoginExample />
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-100 p-4">
      <div className="max-w-7xl mx-auto">
        <h1 className="text-3xl font-bold mb-8">Billiards Trainer - MobX Examples</h1>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <LoginExample />
          <SystemStatusExample />
          <GameStateExample />
          <VisionExample />
          <ConfigExample />
          <NotificationExample />
          <ResponsiveExample />
        </div>

        {/* Modal Examples */}
        <div className="mt-8 p-4 bg-white border rounded">
          <h3 className="text-lg font-bold mb-4">Modal System</h3>
          <div className="space-x-2">
            <button
              onClick={() => open('settings')}
              className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
            >
              Open Settings Modal
            </button>
            <button
              onClick={() => open('calibration')}
              className="px-4 py-2 bg-green-500 text-white rounded hover:bg-green-600"
            >
              Open Calibration Modal
            </button>
          </div>
        </div>

        {/* Modal overlays */}
        {modals.settings && (
          <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
            <div className="bg-white p-6 rounded-lg max-w-md w-full mx-4">
              <h2 className="text-xl font-bold mb-4">Settings Modal</h2>
              <p className="text-gray-600 mb-4">
                This is an example settings modal. In a real application, this would contain
                configuration options and settings forms.
              </p>
              <button
                onClick={() => close('settings')}
                className="px-4 py-2 bg-gray-500 text-white rounded hover:bg-gray-600"
              >
                Close
              </button>
            </div>
          </div>
        )}

        {modals.calibration && (
          <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
            <div className="bg-white p-6 rounded-lg max-w-lg w-full mx-4">
              <h2 className="text-xl font-bold mb-4">Calibration Modal</h2>
              <p className="text-gray-600 mb-4">
                This is an example calibration modal. In a real application, this would contain
                camera calibration controls and visual feedback.
              </p>
              <button
                onClick={() => close('calibration')}
                className="px-4 py-2 bg-gray-500 text-white rounded hover:bg-gray-600"
              >
                Close
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
});

export default ExampleDashboard;
