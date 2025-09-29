/**
 * High-level API service layer that combines REST and WebSocket clients
 * with state management, caching, and request deduplication
 */

import { ApiClient } from './api-client';
import { WebSocketClient, ConnectionState } from './websocket-client';
import { AuthService } from './auth-service';
import { DataProcessingService, ProcessedGameState, ProcessedFrameData, ProcessedTrajectory, ProcessedAlert } from './data-handlers';
import {
  HealthResponse,
  ConfigResponse,
  ConfigUpdateRequest,
  ConfigUpdateResponse,
  GameStateResponse,
  CalibrationStartRequest,
  CalibrationStartResponse,
  CalibrationPointRequest,
  CalibrationPointResponse,
  CalibrationApplyResponse,
  LoginRequest,
  VALID_STREAM_TYPES,
} from '../types/api';
import type { MessageType } from '../types/api';

export interface ApiServiceConfig {
  apiBaseUrl: string;
  wsBaseUrl: string;
  enableCaching?: boolean;
  cacheTimeout?: number;
  enableRequestDeduplication?: boolean;
  maxConcurrentRequests?: number;
  autoConnectWebSocket?: boolean;
  defaultStreamSubscriptions?: string[];
}

export interface CacheEntry<T = any> {
  data: T;
  timestamp: Date;
  expiresAt: Date;
}

export interface LoadingState {
  isLoading: boolean;
  error: string | null;
  lastUpdate: Date | null;
}

export interface ConnectionStatus {
  api: 'connected' | 'disconnected' | 'error';
  websocket: ConnectionState;
  lastApiCall: Date | null;
  lastWebSocketMessage: Date | null;
}

export type HealthCheckHandler = (health: HealthResponse) => void;
export type ConnectionStatusHandler = (status: ConnectionStatus) => void;

export class ApiService {
  private apiClient: ApiClient;
  private wsClient: WebSocketClient;
  private authService: AuthService;
  private dataProcessor: DataProcessingService;
  private config: Required<ApiServiceConfig>;

  // Caching and deduplication
  private cache = new Map<string, CacheEntry>();
  private pendingRequests = new Map<string, Promise<any>>();
  private loadingStates = new Map<string, LoadingState>();

  // Event handlers
  private healthCheckHandlers = new Set<HealthCheckHandler>();
  private connectionStatusHandlers = new Set<ConnectionStatusHandler>();

  // Timers
  private cacheCleanupTimer: NodeJS.Timeout | null = null;
  private healthCheckTimer: NodeJS.Timeout | null = null;

  // Status tracking
  private connectionStatus: ConnectionStatus = {
    api: 'disconnected',
    websocket: 'disconnected',
    lastApiCall: null,
    lastWebSocketMessage: null,
  };

  constructor(
    apiClient: ApiClient,
    wsClient: WebSocketClient,
    authService: AuthService,
    dataProcessor: DataProcessingService,
    config: ApiServiceConfig
  ) {
    this.apiClient = apiClient;
    this.wsClient = wsClient;
    this.authService = authService;
    this.dataProcessor = dataProcessor;

    this.config = {
      enableCaching: true,
      cacheTimeout: 300000, // 5 minutes
      enableRequestDeduplication: true,
      maxConcurrentRequests: 10,
      autoConnectWebSocket: true,
      defaultStreamSubscriptions: ['frame', 'state', 'trajectory', 'alert'],
      ...config,
    };

    this.initializeServices();
  }

  // =============================================================================
  // Initialization
  // =============================================================================

  private initializeServices(): void {
    // Setup auth state changes
    this.authService.onStateChange((authState) => {
      if (authState.isAuthenticated && authState.accessToken) {
        // Update WebSocket token
        this.wsClient.updateToken(authState.accessToken);

        // Auto-connect WebSocket if enabled
        if (this.config.autoConnectWebSocket) {
          this.connectWebSocket();
        }
      } else {
        // Disconnect WebSocket on logout
        this.wsClient.disconnect();
      }
    });

    // Setup WebSocket connection state changes
    this.wsClient.onConnectionState((state, error) => {
      this.updateConnectionStatus({ websocket: state });

      if (state === 'connected') {
        // Subscribe to default streams
        this.subscribeToStreams(this.config.defaultStreamSubscriptions);
      }
    });

    // Setup WebSocket message handling
    this.setupWebSocketHandlers();

    // Setup API client interceptors
    this.setupApiInterceptors();

    // Start periodic tasks
    this.startCacheCleanup();
    this.startHealthCheck();
  }

  private setupWebSocketHandlers(): void {
    // Handle all message types through data processor
    const messageTypes: MessageType[] = [
      'frame', 'state', 'trajectory', 'alert', 'config', 'metrics',
      'connection', 'ping', 'pong', 'subscribe', 'unsubscribe',
      'subscribed', 'unsubscribed', 'status', 'error'
    ];

    messageTypes.forEach(messageType => {
      this.wsClient.on(messageType, (message) => {
        this.updateConnectionStatus({ lastWebSocketMessage: new Date() });
        this.dataProcessor.processMessage(message);
      });
    });
  }

  private setupApiInterceptors(): void {
    // Add request tracking
    this.apiClient.addRequestInterceptor((config) => {
      this.updateConnectionStatus({ lastApiCall: new Date() });
      return config;
    });

    // Add response tracking
    this.apiClient.addResponseInterceptor((response) => {
      this.updateConnectionStatus({
        api: response.ok ? 'connected' : 'error'
      });
      return response;
    });
  }

  // =============================================================================
  // Connection Management
  // =============================================================================

  async connectWebSocket(): Promise<void> {
    try {
      await this.wsClient.connect();
    } catch (error) {
      console.error('WebSocket connection failed:', error);
      throw error;
    }
  }

  disconnectWebSocket(): void {
    this.wsClient.disconnect();
  }

  async subscribeToStreams(streams: string[]): Promise<void> {
    const validStreams = streams.filter(stream =>
      VALID_STREAM_TYPES.includes(stream as any)
    );

    if (validStreams.length > 0) {
      this.wsClient.subscribe(validStreams);
    }
  }

  async unsubscribeFromStreams(streams: string[]): Promise<void> {
    this.wsClient.unsubscribe(streams);
  }

  // =============================================================================
  // Authentication Methods
  // =============================================================================

  async login(credentials: LoginRequest): Promise<void> {
    await this.authService.login(credentials);
  }

  async logout(): Promise<void> {
    await this.authService.logout();
  }

  isAuthenticated(): boolean {
    return this.authService.isAuthenticated();
  }

  getCurrentUser() {
    return this.authService.getUser();
  }

  hasPermission(permission: string): boolean {
    return this.authService.hasPermission(permission);
  }

  // =============================================================================
  // Health and System Methods
  // =============================================================================

  async getHealth(): Promise<HealthResponse> {
    return this.withCaching('health', async () => {
      return this.apiClient.getHealth();
    });
  }

  async getSystemVersion(): Promise<{ version: string; build_date: string }> {
    return this.withCaching('version', async () => {
      return this.apiClient.getVersion();
    });
  }

  // =============================================================================
  // Configuration Methods
  // =============================================================================

  async getConfig(section?: string): Promise<ConfigResponse> {
    const cacheKey = `config_${section || 'all'}`;
    return this.withCaching(cacheKey, async () => {
      return this.apiClient.getConfig(section);
    });
  }

  async updateConfig(request: ConfigUpdateRequest): Promise<ConfigUpdateResponse> {
    const result = await this.withLoading(`config_update_${request.section || 'all'}`, async () => {
      return this.apiClient.updateConfig(request);
    });

    // Invalidate config cache after update
    this.invalidateCache(`config_${request.section || 'all'}`);

    return result;
  }

  async validateConfig(values: Record<string, any>): Promise<ConfigUpdateResponse> {
    return this.apiClient.validateConfig(values);
  }

  async exportConfig(format: string = 'json'): Promise<Blob> {
    return this.apiClient.exportConfig(format);
  }

  // =============================================================================
  // Game State Methods
  // =============================================================================

  async getCurrentGameState(): Promise<GameStateResponse> {
    return this.withCaching('current_game_state', async () => {
      return this.apiClient.getCurrentGameState();
    }, 5000); // Shorter cache for game state
  }

  async getGameHistory(options: {
    limit?: number;
    offset?: number;
    startTime?: Date;
    endTime?: Date;
  } = {}): Promise<GameStateResponse[]> {
    const cacheKey = `game_history_${JSON.stringify(options)}`;
    return this.withCaching(cacheKey, async () => {
      const { limit, offset, startTime, endTime } = options;
      return this.apiClient.getGameHistory(limit, offset, startTime, endTime);
    });
  }

  async resetGameState(): Promise<void> {
    await this.withLoading('reset_game_state', async () => {
      const result = await this.apiClient.resetGameState();
      // Invalidate game state cache
      this.invalidateCache('current_game_state');
      return result;
    });
  }

  // =============================================================================
  // Calibration Methods
  // =============================================================================

  async startCalibration(request: CalibrationStartRequest): Promise<CalibrationStartResponse> {
    return this.withLoading('start_calibration', async () => {
      return this.apiClient.startCalibration(request);
    });
  }

  async captureCalibrationPoint(
    sessionId: string,
    point: CalibrationPointRequest
  ): Promise<CalibrationPointResponse> {
    return this.withLoading(`calibration_point_${sessionId}`, async () => {
      return this.apiClient.captureCalibrationPoint(sessionId, point);
    });
  }

  async applyCalibration(sessionId: string): Promise<CalibrationApplyResponse> {
    return this.withLoading(`apply_calibration_${sessionId}`, async () => {
      const result = await this.apiClient.applyCalibration(sessionId);
      // Invalidate config cache since calibration affects configuration
      this.invalidateCache('config_all');
      return result;
    });
  }

  async cancelCalibration(sessionId: string): Promise<void> {
    await this.withLoading(`cancel_calibration_${sessionId}`, async () => {
      return this.apiClient.cancelCalibration(sessionId);
    });
  }

  // =============================================================================
  // Real-time Data Access
  // =============================================================================

  getCurrentProcessedGameState(): ProcessedGameState | null {
    return this.dataProcessor.getCurrentGameState();
  }

  getGameStateHistory(): ProcessedGameState[] {
    return this.dataProcessor.getGameStateHistory();
  }

  getFrameBuffer(): Map<string, ProcessedFrameData> {
    return this.dataProcessor.getFrameBuffer();
  }

  // Event handlers for real-time data
  onFrameData(handler: (frame: ProcessedFrameData) => void): void {
    this.dataProcessor.onFrame(handler);
  }

  onGameStateData(handler: (state: ProcessedGameState) => void): void {
    this.dataProcessor.onGameState(handler);
  }

  onTrajectoryData(handler: (trajectory: ProcessedTrajectory) => void): void {
    this.dataProcessor.onTrajectory(handler);
  }

  onAlertData(handler: (alert: ProcessedAlert) => void): void {
    this.dataProcessor.onAlert(handler);
  }

  // =============================================================================
  // Hardware Control Methods
  // =============================================================================

  async getCameraStatus(): Promise<Record<string, any>> {
    return this.withCaching('camera_status', async () => {
      return this.apiClient.getCameraStatus();
    }, 10000); // 10 second cache
  }

  async setCameraSettings(settings: Record<string, any>): Promise<void> {
    await this.withLoading('camera_settings', async () => {
      const result = await this.apiClient.setCameraSettings(settings);
      this.invalidateCache('camera_status');
      return result;
    });
  }

  async getProjectorStatus(): Promise<Record<string, any>> {
    return this.withCaching('projector_status', async () => {
      return this.apiClient.getProjectorStatus();
    }, 10000);
  }

  async setProjectorSettings(settings: Record<string, any>): Promise<void> {
    await this.withLoading('projector_settings', async () => {
      const result = await this.apiClient.setProjectorSettings(settings);
      this.invalidateCache('projector_status');
      return result;
    });
  }

  // =============================================================================
  // WebSocket Management Methods
  // =============================================================================

  async getWebSocketConnections(): Promise<Record<string, any>> {
    return this.apiClient.getWebSocketConnections();
  }

  async getWebSocketHealth(): Promise<Record<string, any>> {
    return this.apiClient.getWebSocketHealth();
  }

  async broadcastTestFrame(width?: number, height?: number): Promise<void> {
    await this.apiClient.broadcastTestFrame(width, height);
  }

  async broadcastTestAlert(level?: string, message?: string): Promise<void> {
    await this.apiClient.broadcastTestAlert(level, message);
  }

  // =============================================================================
  // Status and Monitoring
  // =============================================================================

  getConnectionStatus(): ConnectionStatus {
    return { ...this.connectionStatus };
  }

  getLoadingState(key: string): LoadingState | null {
    return this.loadingStates.get(key) || null;
  }

  isLoading(key: string): boolean {
    const state = this.getLoadingState(key);
    return state?.isLoading || false;
  }

  onHealthCheck(handler: HealthCheckHandler): void {
    this.healthCheckHandlers.add(handler);
  }

  offHealthCheck(handler: HealthCheckHandler): void {
    this.healthCheckHandlers.delete(handler);
  }

  onConnectionStatus(handler: ConnectionStatusHandler): void {
    this.connectionStatusHandlers.add(handler);
  }

  offConnectionStatus(handler: ConnectionStatusHandler): void {
    this.connectionStatusHandlers.delete(handler);
  }

  // =============================================================================
  // Caching and Request Management
  // =============================================================================

  private async withCaching<T>(
    key: string,
    operation: () => Promise<T>,
    customTimeout?: number
  ): Promise<T> {
    if (!this.config.enableCaching) {
      return operation();
    }

    // Check cache first
    const cached = this.cache.get(key);
    if (cached && cached.expiresAt > new Date()) {
      return cached.data;
    }

    // Execute operation and cache result
    const result = await this.withRequestDeduplication(key, operation);

    const timeout = customTimeout || this.config.cacheTimeout;
    this.cache.set(key, {
      data: result,
      timestamp: new Date(),
      expiresAt: new Date(Date.now() + timeout),
    });

    return result;
  }

  private async withRequestDeduplication<T>(
    key: string,
    operation: () => Promise<T>
  ): Promise<T> {
    if (!this.config.enableRequestDeduplication) {
      return operation();
    }

    // Check if request is already pending
    const pending = this.pendingRequests.get(key);
    if (pending) {
      return pending;
    }

    // Execute operation and store promise
    const promise = operation();
    this.pendingRequests.set(key, promise);

    try {
      const result = await promise;
      return result;
    } finally {
      this.pendingRequests.delete(key);
    }
  }

  private async withLoading<T>(
    key: string,
    operation: () => Promise<T>
  ): Promise<T> {
    this.setLoadingState(key, { isLoading: true, error: null, lastUpdate: null });

    try {
      const result = await operation();
      this.setLoadingState(key, {
        isLoading: false,
        error: null,
        lastUpdate: new Date()
      });
      return result;
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      this.setLoadingState(key, {
        isLoading: false,
        error: errorMessage,
        lastUpdate: new Date()
      });
      throw error;
    }
  }

  private setLoadingState(key: string, state: LoadingState): void {
    this.loadingStates.set(key, state);
  }

  private invalidateCache(pattern: string): void {
    const keysToDelete: string[] = [];

    for (const key of this.cache.keys()) {
      if (key.startsWith(pattern) || key.includes(pattern)) {
        keysToDelete.push(key);
      }
    }

    keysToDelete.forEach(key => this.cache.delete(key));
  }

  private updateConnectionStatus(updates: Partial<ConnectionStatus>): void {
    const previousStatus = { ...this.connectionStatus };
    this.connectionStatus = { ...this.connectionStatus, ...updates };

    // Notify handlers if status changed
    if (JSON.stringify(previousStatus) !== JSON.stringify(this.connectionStatus)) {
      this.connectionStatusHandlers.forEach(handler => {
        try {
          handler(this.connectionStatus);
        } catch (error) {
          console.error('Error in connection status handler:', error);
        }
      });
    }
  }

  // =============================================================================
  // Periodic Tasks
  // =============================================================================

  private startCacheCleanup(): void {
    this.cacheCleanupTimer = setInterval(() => {
      const now = new Date();
      const keysToDelete: string[] = [];

      for (const [key, entry] of this.cache.entries()) {
        if (entry.expiresAt <= now) {
          keysToDelete.push(key);
        }
      }

      keysToDelete.forEach(key => this.cache.delete(key));
    }, 60000); // Run every minute
  }

  private startHealthCheck(): void {
    this.healthCheckTimer = setInterval(async () => {
      try {
        const health = await this.getHealth();
        this.healthCheckHandlers.forEach(handler => {
          try {
            handler(health);
          } catch (error) {
            console.error('Error in health check handler:', error);
          }
        });
      } catch (error) {
        console.error('Health check failed:', error);
      }
    }, 30000); // Run every 30 seconds
  }

  // =============================================================================
  // Cleanup
  // =============================================================================

  destroy(): void {
    // Clear timers
    if (this.cacheCleanupTimer) {
      clearInterval(this.cacheCleanupTimer);
    }
    if (this.healthCheckTimer) {
      clearInterval(this.healthCheckTimer);
    }

    // Clear caches and handlers
    this.cache.clear();
    this.pendingRequests.clear();
    this.loadingStates.clear();
    this.healthCheckHandlers.clear();
    this.connectionStatusHandlers.clear();

    // Destroy services
    this.wsClient.destroy();
    this.authService.destroy();
    this.dataProcessor.destroy();
  }
}

// =============================================================================
// Factory Function
// =============================================================================

export function createApiService(config: ApiServiceConfig): ApiService {
  const apiClient = new ApiClient({ baseUrl: config.apiBaseUrl });

  const wsClient = new WebSocketClient({
    url: config.wsBaseUrl,
    autoReconnect: true,
  });

  const authService = new AuthService(apiClient, {
    persistAuth: true,
    autoRefresh: true,
  });

  const dataProcessor = new DataProcessingService({
    frameProcessing: {
      maxFps: 30,
      enableCompression: true,
    },
    stateProcessing: {
      enablePrediction: true,
      confidenceThreshold: 0.7,
    },
  });

  return new ApiService(apiClient, wsClient, authService, dataProcessor, config);
}
