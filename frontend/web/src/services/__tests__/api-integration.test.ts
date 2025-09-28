/**
 * Integration tests for API service functionality
 */

import { describe, it, expect, beforeEach, afterEach, vi, Mock } from 'vitest';
import { ApiService, createApiService } from '../api-service';
import { ApiClient } from '../api-client';
import { WebSocketClient } from '../websocket-client';
import { AuthService } from '../auth-service';
import { DataProcessingService } from '../data-handlers';

// Mock implementations
vi.mock('../api-client');
vi.mock('../websocket-client');
vi.mock('../auth-service');
vi.mock('../data-handlers');

describe('API Integration Tests', () => {
  let apiService: ApiService;
  let mockApiClient: Mock;
  let mockWsClient: Mock;
  let mockAuthService: Mock;
  let mockDataProcessor: Mock;

  beforeEach(() => {
    // Reset all mocks
    vi.clearAllMocks();

    // Create mock instances
    mockApiClient = vi.mocked(ApiClient).prototype;
    mockWsClient = vi.mocked(WebSocketClient).prototype;
    mockAuthService = vi.mocked(AuthService).prototype;
    mockDataProcessor = vi.mocked(DataProcessingService).prototype;

    // Setup default mock implementations
    mockApiClient.getHealth = vi.fn().mockResolvedValue({
      status: 'healthy',
      timestamp: new Date().toISOString(),
      uptime: 3600,
      version: '1.0.0',
      components: {},
    });

    mockWsClient.connect = vi.fn().mockResolvedValue(undefined);
    mockWsClient.subscribe = vi.fn().mockReturnValue(true);
    mockWsClient.disconnect = vi.fn();
    mockWsClient.onConnectionState = vi.fn();
    mockWsClient.on = vi.fn();

    mockAuthService.onStateChange = vi.fn();
    mockAuthService.isAuthenticated = vi.fn().mockReturnValue(false);
    mockAuthService.login = vi.fn();
    mockAuthService.logout = vi.fn();

    mockDataProcessor.processMessage = vi.fn();
    mockDataProcessor.onFrame = vi.fn();
    mockDataProcessor.onGameState = vi.fn();

    // Create API service instance
    apiService = createApiService({
      apiBaseUrl: 'http://localhost:8000',
      wsBaseUrl: 'ws://localhost:8000/ws',
      autoConnectWebSocket: false, // Don't auto-connect in tests
    });
  });

  afterEach(() => {
    apiService?.destroy();
  });

  describe('Service Initialization', () => {
    it('should create service with correct configuration', () => {
      expect(apiService).toBeDefined();
      expect(mockAuthService.onStateChange).toHaveBeenCalled();
    });

    it('should setup WebSocket event handlers', () => {
      expect(mockWsClient.onConnectionState).toHaveBeenCalled();
    });
  });

  describe('Authentication Integration', () => {
    it('should handle successful login', async () => {
      const credentials = { username: 'test', password: 'password' };
      const mockUser = {
        id: '1',
        username: 'test',
        role: 'admin',
        permissions: ['admin:*'],
        loginTimestamp: new Date(),
        lastActivity: new Date(),
      };

      mockAuthService.login.mockResolvedValue(mockUser);

      await apiService.login(credentials);

      expect(mockAuthService.login).toHaveBeenCalledWith(credentials);
    });

    it('should handle authentication state changes', () => {
      // Simulate auth state change
      const authStateHandler = mockAuthService.onStateChange.mock.calls[0][0];

      authStateHandler({
        isAuthenticated: true,
        user: { id: '1', username: 'test' },
        accessToken: 'test-token',
        refreshToken: 'refresh-token',
        expiresAt: new Date(Date.now() + 3600000),
        isRefreshing: false,
        lastError: null,
      });

      expect(mockWsClient.updateToken).toHaveBeenCalledWith('test-token');
    });

    it('should disconnect WebSocket on logout', async () => {
      await apiService.logout();

      expect(mockAuthService.logout).toHaveBeenCalled();
      expect(mockWsClient.disconnect).toHaveBeenCalled();
    });
  });

  describe('WebSocket Integration', () => {
    it('should connect to WebSocket successfully', async () => {
      await apiService.connectWebSocket();

      expect(mockWsClient.connect).toHaveBeenCalled();
    });

    it('should subscribe to default streams on connection', () => {
      // Simulate WebSocket connection
      const connectionHandler = mockWsClient.onConnectionState.mock.calls[0][0];
      connectionHandler('connected');

      expect(mockWsClient.subscribe).toHaveBeenCalledWith([
        'frame', 'state', 'trajectory', 'alert'
      ]);
    });

    it('should handle WebSocket messages through data processor', () => {
      // Find the message handler for frame messages
      const messageHandlerCall = mockWsClient.on.mock.calls.find(
        call => call[0] === 'frame'
      );

      expect(messageHandlerCall).toBeDefined();

      const messageHandler = messageHandlerCall[1];
      const testMessage = {
        type: 'frame',
        timestamp: new Date().toISOString(),
        data: {
          image: 'base64data',
          width: 1920,
          height: 1080,
          format: 'jpeg',
          quality: 85,
          compressed: false,
          fps: 30,
          size_bytes: 1024,
        },
      };

      messageHandler(testMessage);

      expect(mockDataProcessor.processMessage).toHaveBeenCalledWith(testMessage);
    });
  });

  describe('REST API Integration', () => {
    it('should fetch health status', async () => {
      const health = await apiService.getHealth();

      expect(mockApiClient.getHealth).toHaveBeenCalled();
      expect(health.status).toBe('healthy');
    });

    it('should handle API errors gracefully', async () => {
      const error = new Error('API Error');
      mockApiClient.getHealth.mockRejectedValue(error);

      await expect(apiService.getHealth()).rejects.toThrow('API Error');
    });

    it('should cache API responses', async () => {
      // First call
      await apiService.getHealth();

      // Second call should use cache
      await apiService.getHealth();

      // API should only be called once due to caching
      expect(mockApiClient.getHealth).toHaveBeenCalledTimes(1);
    });
  });

  describe('Configuration Management', () => {
    it('should load configuration', async () => {
      const mockConfig = {
        timestamp: new Date().toISOString(),
        values: { camera: { enabled: true } },
        schema_version: '1.0',
        last_modified: new Date().toISOString(),
        is_valid: true,
        validation_errors: [],
      };

      mockApiClient.getConfig = vi.fn().mockResolvedValue(mockConfig);

      const config = await apiService.getConfig();

      expect(mockApiClient.getConfig).toHaveBeenCalled();
      expect(config.values).toEqual({ camera: { enabled: true } });
    });

    it('should update configuration', async () => {
      const updateRequest = {
        section: 'camera',
        values: { enabled: false },
      };

      const mockResponse = {
        success: true,
        updated_fields: ['enabled'],
        validation_errors: [],
        warnings: [],
        rollback_available: true,
        restart_required: false,
      };

      mockApiClient.updateConfig = vi.fn().mockResolvedValue(mockResponse);

      const result = await apiService.updateConfig(updateRequest);

      expect(mockApiClient.updateConfig).toHaveBeenCalledWith(updateRequest);
      expect(result.success).toBe(true);
    });
  });

  describe('Game State Management', () => {
    it('should fetch current game state', async () => {
      const mockGameState = {
        timestamp: new Date().toISOString(),
        frame_number: 123,
        balls: [],
        table: {
          width: 2.84,
          height: 1.42,
          pocket_positions: [],
          pocket_radius: 0.06,
          surface_friction: 0.1,
        },
        game_type: 'eight_ball',
        is_valid: true,
        confidence: 0.95,
        events: [],
      };

      mockApiClient.getCurrentGameState = vi.fn().mockResolvedValue(mockGameState);

      const gameState = await apiService.getCurrentGameState();

      expect(mockApiClient.getCurrentGameState).toHaveBeenCalled();
      expect(gameState.confidence).toBe(0.95);
    });

    it('should reset game state', async () => {
      const mockResponse = {
        success: true,
        message: 'Game state reset successfully',
        timestamp: new Date().toISOString(),
      };

      mockApiClient.resetGameState = vi.fn().mockResolvedValue(mockResponse);

      await apiService.resetGameState();

      expect(mockApiClient.resetGameState).toHaveBeenCalled();
    });
  });

  describe('Real-time Data Handling', () => {
    it('should process frame data', () => {
      const frameHandler = vi.fn();
      apiService.onFrameData(frameHandler);

      expect(mockDataProcessor.onFrame).toHaveBeenCalledWith(frameHandler);
    });

    it('should process game state data', () => {
      const stateHandler = vi.fn();
      apiService.onGameStateData(stateHandler);

      expect(mockDataProcessor.onGameState).toHaveBeenCalledWith(stateHandler);
    });

    it('should provide access to current processed state', () => {
      const mockProcessedState = {
        timestamp: new Date(),
        balls: [],
        confidence: 0.95,
        isValid: true,
        changesSinceLastFrame: [],
      };

      mockDataProcessor.getCurrentGameState = vi.fn().mockReturnValue(mockProcessedState);

      const currentState = apiService.getCurrentProcessedGameState();

      expect(currentState).toEqual(mockProcessedState);
    });
  });

  describe('Error Handling', () => {
    it('should handle network errors', async () => {
      const networkError = new Error('Network unreachable');
      mockApiClient.getHealth.mockRejectedValue(networkError);

      await expect(apiService.getHealth()).rejects.toThrow('Network unreachable');
    });

    it('should handle WebSocket connection errors', () => {
      const connectionHandler = mockWsClient.onConnectionState.mock.calls[0][0];
      const error = new Error('WebSocket connection failed');

      connectionHandler('error', error);

      // Connection status should be updated
      const status = apiService.getConnectionStatus();
      expect(status.websocket).toBe('error');
    });
  });

  describe('Caching and Performance', () => {
    it('should respect cache timeout', async () => {
      vi.useFakeTimers();

      // First call
      await apiService.getHealth();

      // Advance time beyond cache timeout
      vi.advanceTimersByTime(350000); // 5 minutes and 50 seconds

      // Second call should bypass cache
      await apiService.getHealth();

      expect(mockApiClient.getHealth).toHaveBeenCalledTimes(2);

      vi.useRealTimers();
    });

    it('should deduplicate concurrent requests', async () => {
      // Make multiple concurrent requests
      const promises = [
        apiService.getHealth(),
        apiService.getHealth(),
        apiService.getHealth(),
      ];

      await Promise.all(promises);

      // Should only make one actual API call
      expect(mockApiClient.getHealth).toHaveBeenCalledTimes(1);
    });
  });

  describe('Stream Management', () => {
    it('should subscribe to specific streams', async () => {
      const streams = ['frame', 'state'];
      await apiService.subscribeToStreams(streams);

      expect(mockWsClient.subscribe).toHaveBeenCalledWith(streams);
    });

    it('should unsubscribe from streams', async () => {
      const streams = ['frame'];
      await apiService.unsubscribeFromStreams(streams);

      expect(mockWsClient.unsubscribe).toHaveBeenCalledWith(streams);
    });

    it('should filter invalid stream types', async () => {
      const streams = ['frame', 'invalid_stream', 'state'];
      await apiService.subscribeToStreams(streams);

      // Should only subscribe to valid streams
      expect(mockWsClient.subscribe).toHaveBeenCalledWith(['frame', 'state']);
    });
  });

  describe('Service Lifecycle', () => {
    it('should clean up resources on destroy', () => {
      apiService.destroy();

      expect(mockWsClient.destroy).toHaveBeenCalled();
      expect(mockAuthService.destroy).toHaveBeenCalled();
      expect(mockDataProcessor.destroy).toHaveBeenCalled();
    });
  });
});

describe('API Service Factory', () => {
  it('should create service with minimal configuration', () => {
    const service = createApiService({
      apiBaseUrl: 'http://localhost:8000',
      wsBaseUrl: 'ws://localhost:8000/ws',
    });

    expect(service).toBeInstanceOf(ApiService);
    service.destroy();
  });

  it('should create service with full configuration', () => {
    const service = createApiService({
      apiBaseUrl: 'http://localhost:8000',
      wsBaseUrl: 'ws://localhost:8000/ws',
      enableCaching: true,
      cacheTimeout: 300000,
      autoConnectWebSocket: true,
      defaultStreamSubscriptions: ['frame', 'state'],
    });

    expect(service).toBeInstanceOf(ApiService);
    service.destroy();
  });
});

// Mock environment variable for testing
Object.defineProperty(import.meta, 'env', {
  value: {
    VITE_API_BASE_URL: 'http://localhost:8000',
    VITE_WS_BASE_URL: 'ws://localhost:8000/ws',
  },
  writable: true,
});
