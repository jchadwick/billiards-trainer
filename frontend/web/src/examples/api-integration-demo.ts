/**
 * Comprehensive example demonstrating the billiards trainer API integration
 * This file shows how to use all the major components together
 */

import { createApiService } from '../services/api-service';
import { errorHandler } from '../services/error-handler';
import { createRootStore } from '../stores';

// =============================================================================
// Basic Setup and Configuration
// =============================================================================

async function initializeApiIntegration() {
  console.log('🚀 Initializing Billiards Trainer API Integration...');

  // Create the API service with configuration
  const apiService = createApiService({
    apiBaseUrl: import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000',
    wsBaseUrl: import.meta.env.VITE_WS_BASE_URL || 'ws://localhost:8000/ws',
    enableCaching: true,
    cacheTimeout: 300000, // 5 minutes
    autoConnectWebSocket: true,
    defaultStreamSubscriptions: ['frame', 'state', 'trajectory', 'alert'],
  });

  // Create MobX store with API service
  const rootStore = createRootStore({ apiService });

  // Setup error handling
  errorHandler.onError((error) => {
    console.error('🔥 Error reported:', error);

    // Show error notification in UI
    rootStore.uiStore.showError(error.message, error.code);
  });

  errorHandler.onConnectionHealth((health) => {
    console.log('📊 Connection health:', health);

    // Update UI based on connection health
    if (health.status === 'poor' || health.status === 'disconnected') {
      rootStore.uiStore.showWarning('Connection issues detected');
    }
  });

  return { apiService, rootStore };
}

// =============================================================================
// Authentication Example
// =============================================================================

async function authenticateUser(rootStore: any) {
  console.log('🔐 Demonstrating authentication...');

  try {
    // Login with credentials
    await rootStore.authStore.login({
      username: 'demo_user',
      password: 'demo_password'
    });

    console.log('✅ Login successful!');
    console.log('User:', rootStore.authStore.user);
    console.log('Permissions:', rootStore.authStore.user?.permissions);

    // Check permissions
    if (rootStore.authStore.hasPermission('stream:frame')) {
      console.log('✅ User has permission to access frame stream');
    }

    if (rootStore.authStore.isAdmin) {
      console.log('✅ User has admin privileges');
    }

  } catch (error) {
    console.error('❌ Authentication failed:', error);
    errorHandler.reportAuthError(error, {
      component: 'demo',
      action: 'login_attempt',
    });
  }
}

// =============================================================================
// WebSocket Real-time Data Example
// =============================================================================

function setupRealtimeDataHandling(apiService: any, rootStore: any) {
  console.log('📡 Setting up real-time data handling...');

  // Handle frame data (video stream)
  apiService.onFrameData((frame) => {
    console.log('📽️ Frame received:', {
      width: frame.displayWidth,
      height: frame.displayHeight,
      fps: frame.originalData.fps,
      quality: frame.originalData.quality,
      size: frame.originalData.size_bytes
    });

    // Frame data is automatically processed and stored in game state store
    // Access via: rootStore.gameStateStore.currentFrame
  });

  // Handle game state updates
  apiService.onGameStateData((gameState) => {
    console.log('🎱 Game state update:', {
      ballCount: gameState.balls.length,
      confidence: gameState.confidence,
      isValid: gameState.isValid,
      movingBalls: gameState.balls.filter(ball => ball.isMoving).length,
      changes: gameState.changesSinceLastFrame
    });

    // Check for specific game events
    if (gameState.changesSinceLastFrame.length > 0) {
      console.log('🎯 Game state changes detected:', gameState.changesSinceLastFrame);
    }

    // Access individual balls
    const cueBall = gameState.balls.find(ball => ball.id === 'cue' || ball.is_cue_ball);
    if (cueBall && cueBall.isMoving) {
      console.log('🎱 Cue ball is moving at speed:', cueBall.speed);
    }
  });

  // Handle trajectory predictions
  apiService.onTrajectoryData((trajectory) => {
    console.log('📈 Trajectory prediction:', {
      lineCount: trajectory.smoothedLines.length,
      collisionCount: trajectory.collisionPredictions.length,
      successProbability: trajectory.successProbability,
      recommendation: trajectory.recommendation
    });

    // Show trajectory recommendations
    if (trajectory.recommendation) {
      rootStore.uiStore.showInfo(trajectory.recommendation, 'Shot Analysis');
    }
  });

  // Handle system alerts
  apiService.onAlertData((alert) => {
    console.log('🚨 System alert:', alert);

    // Show appropriate UI notification based on alert level
    switch (alert.level) {
      case 'critical':
        rootStore.uiStore.showError(alert.message, 'Critical Alert');
        break;
      case 'error':
        rootStore.uiStore.showError(alert.message);
        break;
      case 'warning':
        rootStore.uiStore.showWarning(alert.message);
        break;
      case 'info':
        rootStore.uiStore.showInfo(alert.message);
        break;
    }
  });
}

// =============================================================================
// Configuration Management Example
// =============================================================================

async function demonstrateConfigManagement(rootStore: any) {
  console.log('⚙️ Demonstrating configuration management...');

  try {
    // Load all configuration
    await rootStore.configStore.loadAllConfig();
    console.log('✅ Configuration loaded');
    console.log('Sections:', rootStore.configStore.configSectionNames);

    // Load specific section
    await rootStore.configStore.loadConfigSection('camera');
    console.log('✅ Camera configuration loaded');

    // Update a configuration value
    rootStore.configStore.updateConfigValue('camera', 'enabled', true);
    rootStore.configStore.updateConfigValue('camera', 'resolution', '1920x1080');

    console.log('📝 Pending changes:', rootStore.configStore.pendingChanges.length);

    // Validate configuration before saving
    const validationErrors = await rootStore.configStore.validateConfig({
      camera: { enabled: true, resolution: '1920x1080' }
    });

    if (validationErrors.length === 0) {
      // Save changes
      await rootStore.configStore.saveAllChanges();
      console.log('✅ Configuration saved successfully');
    } else {
      console.log('❌ Validation errors:', validationErrors);
    }

  } catch (error) {
    console.error('❌ Configuration management failed:', error);
    errorHandler.reportError(error, {
      component: 'demo',
      action: 'config_management',
    });
  }
}

// =============================================================================
// Calibration Process Example
// =============================================================================

async function demonstrateCalibration(rootStore: any) {
  console.log('📐 Demonstrating calibration process...');

  try {
    // Start calibration session
    await rootStore.calibrationStore.startCalibration({
      calibration_type: 'table_corners',
      table_type: 'standard',
      timeout: 300 // 5 minutes
    });

    console.log('✅ Calibration session started');
    console.log('Instructions:', rootStore.calibrationStore.sessionInstructions);

    // Simulate capturing calibration points
    const samplePoints = [
      { screenX: 100, screenY: 100, tableX: 0, tableY: 0 },      // Top-left
      { screenX: 1820, screenY: 100, tableX: 2.84, tableY: 0 },  // Top-right
      { screenX: 1820, screenY: 980, tableX: 2.84, tableY: 1.42 }, // Bottom-right
      { screenX: 100, screenY: 980, tableX: 0, tableY: 1.42 },   // Bottom-left
    ];

    for (const point of samplePoints) {
      await rootStore.calibrationStore.capturePoint(
        point.screenX,
        point.screenY,
        point.tableX,
        point.tableY
      );

      console.log(`✅ Point captured: (${point.screenX}, ${point.screenY})`);
      console.log('Progress:', rootStore.calibrationStore.progress);
    }

    // Apply calibration if ready
    if (rootStore.calibrationStore.isReadyToApply) {
      await rootStore.calibrationStore.applyCalibration();
      console.log('✅ Calibration applied successfully');
      console.log('Final accuracy:', rootStore.calibrationStore.stats.averageAccuracy);
    }

  } catch (error) {
    console.error('❌ Calibration failed:', error);
    errorHandler.reportError(error, {
      component: 'demo',
      action: 'calibration',
    });
  }
}

// =============================================================================
// Game State Monitoring Example
// =============================================================================

function demonstrateGameStateMonitoring(rootStore: any) {
  console.log('🎮 Demonstrating game state monitoring...');

  // Monitor game state changes
  setInterval(() => {
    const gameStore = rootStore.gameStateStore;

    if (gameStore.currentState) {
      const stats = gameStore.statistics;

      console.log('📊 Game Statistics:', {
        totalFrames: stats.totalFrames,
        frameRate: stats.frameRate.toFixed(1),
        ballsDetected: stats.ballsDetected,
        averageConfidence: (stats.averageConfidence * 100).toFixed(1) + '%',
        ballMovements: stats.ballMovements,
        connectionHealth: gameStore.streamHealth.status
      });

      // Check for specific game conditions
      if (gameStore.isGameActive) {
        console.log('🎯 Game is active - balls are moving');
      }

      // Analyze ball positions
      const movingBalls = gameStore.movingBalls;
      if (movingBalls.length > 0) {
        console.log('🎱 Moving balls:', movingBalls.map(ball => ({
          id: ball.id,
          position: ball.position,
          speed: ball.speed
        })));
      }

      // Check cue stick detection
      if (gameStore.cueStick?.detected) {
        console.log('🎯 Cue stick detected:', {
          angle: gameStore.cueStick.angle.toFixed(1) + '°',
          aimingAccuracy: (gameStore.cueStick.aimingAccuracy * 100).toFixed(1) + '%'
        });
      }
    }
  }, 5000); // Every 5 seconds
}

// =============================================================================
// Error Handling and Recovery Example
// =============================================================================

function demonstrateErrorHandling(apiService: any, rootStore: any) {
  console.log('🛡️ Demonstrating error handling and recovery...');

  // Example of handling API errors with retry
  async function robustApiCall() {
    try {
      return await errorHandler.withRetry(
        () => apiService.getHealth(),
        {
          maxAttempts: 3,
          initialDelay: 1000,
          maxDelay: 5000,
          backoffFactor: 2,
          retryCondition: (error) => {
            // Retry on network errors but not on auth errors
            return !error.message.includes('401') && !error.message.includes('403');
          }
        },
        {
          component: 'demo',
          action: 'health_check_with_retry'
        }
      );
    } catch (error) {
      console.error('❌ Health check failed after retries:', error);
      rootStore.uiStore.showError('Unable to connect to server after multiple attempts');
    }
  }

  // Monitor connection health
  setInterval(async () => {
    const health = errorHandler.connectionHealthStatus;

    if (health.status === 'poor' || health.status === 'disconnected') {
      console.log('⚠️ Poor connection detected, attempting recovery...');

      // Attempt to reconnect WebSocket
      try {
        await apiService.connectWebSocket();
        console.log('✅ WebSocket reconnected');
      } catch (error) {
        console.error('❌ WebSocket reconnection failed:', error);
      }

      // Test API connectivity
      await robustApiCall();
    }
  }, 30000); // Every 30 seconds

  // Get error statistics periodically
  setInterval(() => {
    const stats = errorHandler.getErrorStats();
    if (stats.total > 0) {
      console.log('📈 Error Statistics:', stats);

      const criticalErrors = errorHandler.getCriticalErrors();
      if (criticalErrors.length > 0) {
        console.log('🚨 Critical errors detected:', criticalErrors);
        rootStore.uiStore.showError(
          `${criticalErrors.length} critical errors require attention`,
          'System Status'
        );
      }
    }
  }, 60000); // Every minute
}

// =============================================================================
// Performance Monitoring Example
// =============================================================================

function demonstratePerformanceMonitoring(rootStore: any) {
  console.log('📊 Demonstrating performance monitoring...');

  setInterval(() => {
    const systemStore = rootStore.systemStore;

    if (systemStore.metrics) {
      console.log('🖥️ System Performance:', {
        cpu: systemStore.metrics.cpu_usage.toFixed(1) + '%',
        memory: systemStore.metrics.memory_usage.toFixed(1) + '%',
        disk: systemStore.metrics.disk_usage.toFixed(1) + '%',
        apiLatency: systemStore.metrics.average_response_time.toFixed(1) + 'ms',
        wsConnections: systemStore.metrics.websocket_connections
      });

      // Check for performance issues
      if (systemStore.metrics.cpu_usage > 80) {
        console.log('⚠️ High CPU usage detected');
        rootStore.uiStore.showWarning('System performance may be affected by high CPU usage');
      }

      if (systemStore.metrics.memory_usage > 85) {
        console.log('⚠️ High memory usage detected');
        rootStore.uiStore.showWarning('High memory usage detected');
      }
    }

    // Check system health status
    const healthStatus = systemStore.systemStatus;
    console.log('🏥 System Health:', healthStatus.overall, '-', healthStatus.details);

    if (healthStatus.issues.length > 0) {
      console.log('⚠️ System Issues:', healthStatus.issues);
    }
  }, 10000); // Every 10 seconds
}

// =============================================================================
// Main Demo Function
// =============================================================================

export async function runApiIntegrationDemo() {
  console.log('🎱 Starting Billiards Trainer API Integration Demo');
  console.log('================================================');

  try {
    // Initialize the API integration
    const { apiService, rootStore } = await initializeApiIntegration();

    // Wait a moment for initialization
    await new Promise(resolve => setTimeout(resolve, 1000));

    // Demonstrate authentication
    await authenticateUser(rootStore);

    // Setup real-time data handling
    setupRealtimeDataHandling(apiService, rootStore);

    // Demonstrate configuration management
    await demonstrateConfigManagement(rootStore);

    // Demonstrate calibration process
    await demonstrateCalibration(rootStore);

    // Start monitoring systems
    demonstrateGameStateMonitoring(rootStore);
    demonstrateErrorHandling(apiService, rootStore);
    demonstratePerformanceMonitoring(rootStore);

    console.log('✅ Demo setup complete! Monitor the console for real-time updates.');
    console.log('🎮 The system is now actively monitoring game state, performance, and errors.');

    // Return the instances for further use
    return { apiService, rootStore };

  } catch (error) {
    console.error('❌ Demo initialization failed:', error);
    errorHandler.reportError(error, {
      component: 'demo',
      action: 'initialization',
    }, 'critical');
  }
}

// =============================================================================
// Utility Functions for Testing
// =============================================================================

export async function testBasicConnectivity(apiService: any) {
  console.log('🔍 Testing basic connectivity...');

  try {
    // Test REST API
    const health = await apiService.getHealth();
    console.log('✅ REST API connected:', health.status);

    // Test WebSocket
    await apiService.connectWebSocket();
    console.log('✅ WebSocket connected');

    // Test stream subscription
    await apiService.subscribeToStreams(['frame', 'state']);
    console.log('✅ Subscribed to streams');

    return true;
  } catch (error) {
    console.error('❌ Connectivity test failed:', error);
    return false;
  }
}

export function simulateGameActivity(rootStore: any) {
  console.log('🎮 Simulating game activity...');

  // Simulate some ball movements
  const mockBalls = [
    { id: 'cue', position: [100, 200], isMoving: true, speed: 5.2 },
    { id: '1', position: [300, 400], isMoving: false, speed: 0 },
    { id: '2', position: [500, 300], isMoving: true, speed: 3.1 },
  ];

  // This would normally come from real WebSocket data
  console.log('🎱 Mock ball positions:', mockBalls);
}

// Auto-run demo if this file is executed directly
if (import.meta.env.DEV) {
  // Uncomment to auto-run demo in development
  // runApiIntegrationDemo().catch(console.error);
}
