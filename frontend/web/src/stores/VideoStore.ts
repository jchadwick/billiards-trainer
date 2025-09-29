/**
 * MobX store for managing video streaming state and data
 */

import { makeAutoObservable, action, computed, runInAction } from 'mobx';
import { WebSocketClient, createWebSocketClient } from '../services/websocket-client';
import type {
  VideoStreamConfig,
  VideoStreamStatus,
  DetectionFrame,
  Ball,
  CueStick,
  Table,
  Trajectory,
  VideoQuality,
  VideoError,
  PerformanceMetrics,
} from '../types/video';
import type {
  WebSocketMessage,
  GameStateData,
  TrajectoryData,
  BallData,
  CueData,
  TableData,
  isGameStateMessage,
  isTrajectoryMessage,
} from '../types/api';

export class VideoStore {
  // Stream configuration
  config: VideoStreamConfig = {
    quality: 'medium',
    fps: 30,
    autoReconnect: true,
    reconnectDelay: 2000,
  };

  // Stream status
  status: VideoStreamStatus = {
    connected: false,
    streaming: false,
    fps: 0,
    quality: 'medium',
    latency: 0,
    errors: 0,
    lastFrameTime: 0,
  };

  // Current detection data
  currentFrame: DetectionFrame | null = null;

  // Stream URL and connection
  private streamUrl = '';
  private reconnectTimeout: NodeJS.Timeout | null = null;
  private performanceInterval: NodeJS.Timeout | null = null;

  // WebSocket connection for real-time detection data
  private wsClient: WebSocketClient | null = null;
  private isWebSocketConnected = false;

  // Performance tracking
  performance: PerformanceMetrics = {
    renderFPS: 0,
    droppedFrames: 0,
    avgRenderTime: 0,
    memoryUsage: 0,
    cpuUsage: 0,
  };

  // Error handling
  errors: VideoError[] = [];
  lastError: VideoError | null = null;

  // UI state
  isFullscreen = false;
  isLoading = false;

  constructor() {
    makeAutoObservable(this, {
      // Actions
      setConfig: action,
      setStatus: action,
      setCurrentFrame: action,
      addError: action,
      clearErrors: action,
      setFullscreen: action,
      setLoading: action,
      updatePerformance: action,

      // Computed
      isConnected: computed,
      isStreaming: computed,
      currentBalls: computed,
      currentCue: computed,
      currentTable: computed,
      currentTrajectories: computed,
      hasErrors: computed,
      latestError: computed,
    });

    // Start performance monitoring
    this.startPerformanceMonitoring();
  }

  // Computed properties
  get isConnected(): boolean {
    return this.status.connected;
  }

  get isStreaming(): boolean {
    return this.status.streaming;
  }

  get currentBalls(): Ball[] {
    return this.currentFrame?.balls || [];
  }

  get currentCue(): CueStick | null {
    return this.currentFrame?.cue || null;
  }

  get currentTable(): Table | null {
    return this.currentFrame?.table || null;
  }

  get currentTrajectories(): Trajectory[] {
    return this.currentFrame?.trajectories || [];
  }

  get hasErrors(): boolean {
    return this.errors.length > 0;
  }

  get latestError(): VideoError | null {
    return this.errors.length > 0 ? this.errors[this.errors.length - 1] : null;
  }

  // Actions
  setConfig(config: Partial<VideoStreamConfig>): void {
    this.config = { ...this.config, ...config };
  }

  setStatus(status: Partial<VideoStreamStatus>): void {
    this.status = { ...this.status, ...status };
  }

  setCurrentFrame(frame: DetectionFrame): void {
    this.currentFrame = frame;
    this.status.lastFrameTime = Date.now();

    // Update FPS calculation
    this.updateFPSCalculation();
  }

  addError(error: VideoError): void {
    this.errors.push(error);
    this.lastError = error;
    this.status.errors = this.errors.length;

    // Keep only last 50 errors
    if (this.errors.length > 50) {
      this.errors = this.errors.slice(-50);
    }
  }

  clearErrors(): void {
    this.errors = [];
    this.lastError = null;
    this.status.errors = 0;
  }

  setFullscreen(isFullscreen: boolean): void {
    this.isFullscreen = isFullscreen;
  }

  setLoading(isLoading: boolean): void {
    this.isLoading = isLoading;
  }

  updatePerformance(metrics: Partial<PerformanceMetrics>): void {
    this.performance = { ...this.performance, ...metrics };
  }

  // Stream control methods
  async connect(baseUrl: string): Promise<void> {
    if (this.status.connected) {
      return;
    }

    this.setLoading(true);
    this.clearErrors();

    try {
      const { apiClient } = await import('../api/client');

      // Update API client base URL if needed
      if (baseUrl && baseUrl !== 'http://localhost:8080') {
        // Would need to create a new client instance or update the existing one
        console.log('Custom base URL requested:', baseUrl);
      }

      // Check stream status first
      const statusResponse = await apiClient.getStreamStatus();
      if (!statusResponse.success) {
        throw new Error(`Failed to get stream status: ${statusResponse.error}`);
      }

      // Start video capture if not already running
      const startResponse = await apiClient.startVideoCapture();
      if (!startResponse.success) {
        console.warn('Video capture start failed, but continuing:', startResponse.error);
      }

      // Set stream URL using API client
      this.streamUrl = apiClient.getVideoStreamUrl(
        this.getQualityValue(this.config.quality),
        this.config.fps
      );

      // Initialize WebSocket connection for real-time detection data
      await this.connectWebSocket(baseUrl);

      runInAction(() => {
        this.status.connected = true;
        this.status.streaming = true;
        this.setLoading(false);
      });

    } catch (error) {
      const videoError: VideoError = {
        code: 'CONNECTION_FAILED',
        message: error instanceof Error ? error.message : 'Unknown connection error',
        timestamp: Date.now(),
        recoverable: true,
      };

      runInAction(() => {
        this.addError(videoError);
        this.status.connected = false;
        this.status.streaming = false;
        this.setLoading(false);
      });

      if (this.config.autoReconnect) {
        this.scheduleReconnect();
      }

      throw error;
    }
  }

  async disconnect(): Promise<void> {
    try {
      const { apiClient } = await import('../api/client');
      await apiClient.stopVideoCapture();
    } catch (error) {
      console.warn('Failed to stop video capture:', error);
    }

    // Disconnect WebSocket
    this.disconnectWebSocket();

    runInAction(() => {
      this.status.connected = false;
      this.status.streaming = false;
      this.currentFrame = null;
    });

    if (this.reconnectTimeout) {
      clearTimeout(this.reconnectTimeout);
      this.reconnectTimeout = null;
    }
  }

  setQuality(quality: VideoQuality): void {
    this.config.quality = quality;
    this.status.quality = quality;
  }

  setFPS(fps: number): void {
    this.config.fps = Math.max(1, Math.min(60, fps));

    // Update stream URL if connected
    if (this.status.connected) {
      this.updateStreamUrl();
    }
  }

  async refreshStreamStatus(): Promise<void> {
    try {
      const { apiClient } = await import('../api/client');
      const response = await apiClient.getStreamStatus();

      if (response.success && response.data) {
        runInAction(() => {
          this.status.connected = response.data.camera?.connected || false;
          this.status.streaming = response.data.streaming?.active_streams > 0 || false;
          this.status.fps = response.data.vision?.processing_fps || 0;
          this.status.latency = response.data.streaming?.avg_fps || 0;
        });
      }
    } catch (error) {
      const videoError: VideoError = {
        code: 'STATUS_REFRESH_FAILED',
        message: error instanceof Error ? error.message : 'Failed to refresh status',
        timestamp: Date.now(),
        recoverable: true,
      };
      this.addError(videoError);
    }
  }

  async captureFrame(): Promise<string | null> {
    try {
      const { apiClient } = await import('../api/client');
      const frameUrl = apiClient.getSingleFrameUrl(
        this.getQualityValue(this.config.quality)
      );

      return frameUrl;
    } catch (error) {
      const videoError: VideoError = {
        code: 'FRAME_CAPTURE_FAILED',
        message: error instanceof Error ? error.message : 'Failed to capture frame',
        timestamp: Date.now(),
        recoverable: true,
      };
      this.addError(videoError);
      return null;
    }
  }

  private updateStreamUrl(): void {
    if (this.status.connected) {
      try {
        const { apiClient } = require('../api/client');
        this.streamUrl = apiClient.getVideoStreamUrl(
          this.getQualityValue(this.config.quality),
          this.config.fps
        );
      } catch (error) {
        console.error('Failed to update stream URL:', error);
      }
    }
  }

  // Internal methods
  private updateFPSCalculation(): void {
    // Simple FPS calculation based on frame timestamps
    const now = Date.now();

    if (this.status.lastFrameTime > 0) {
      const deltaTime = (now - this.status.lastFrameTime) / 1000;
      if (deltaTime > 0) {
        const instantFPS = 1 / deltaTime;
        // Use exponential moving average for smooth FPS display
        this.status.fps = this.status.fps * 0.9 + instantFPS * 0.1;
      }
    }
  }

  private scheduleReconnect(): void {
    if (this.reconnectTimeout) {
      clearTimeout(this.reconnectTimeout);
    }

    this.reconnectTimeout = setTimeout(() => {
      if (!this.status.connected && this.streamUrl) {
        const baseUrl = this.streamUrl.replace('/api/v1/stream/video', '');
        this.connect(baseUrl).catch(console.error);
      }
    }, this.config.reconnectDelay);
  }

  private startPerformanceMonitoring(): void {
    this.performanceInterval = setInterval(() => {
      // Update performance metrics

      // Simple memory usage estimation (if available)
      if ('memory' in performance) {
        const memory = (performance as any).memory;
        this.updatePerformance({
          memoryUsage: memory.usedJSHeapSize / 1024 / 1024, // MB
        });
      }

      // Calculate render FPS (simplified)
      this.updatePerformance({
        renderFPS: this.status.fps,
      });

    }, 1000);
  }

  // WebSocket connection management
  private async connectWebSocket(baseUrl: string): Promise<void> {
    try {
      // Create WebSocket URL from base URL
      const wsUrl = baseUrl.replace(/^http/, 'ws') + '/api/v1/ws';

      // Initialize WebSocket client
      this.wsClient = createWebSocketClient({
        url: wsUrl,
        autoReconnect: true,
        maxReconnectAttempts: 10,
        reconnectDelay: 1000,
        heartbeatInterval: 30000,
      });

      // Set up message handlers
      this.wsClient.on('state', this.handleGameStateMessage.bind(this));
      this.wsClient.on('trajectory', this.handleTrajectoryMessage.bind(this));
      this.wsClient.onConnectionState(this.handleWebSocketStateChange.bind(this));

      // Connect and subscribe to real-time data streams
      await this.wsClient.connect();

      // Subscribe to detection data streams
      this.wsClient.subscribe(['state', 'trajectory'], {
        quality: 'high',
        frame_rate: this.config.fps,
      });

      runInAction(() => {
        this.isWebSocketConnected = true;
      });

      console.log('WebSocket connected for real-time detection data');

    } catch (error) {
      console.error('Failed to connect WebSocket:', error);
      const videoError: VideoError = {
        code: 'WEBSOCKET_CONNECTION_FAILED',
        message: `WebSocket connection failed: ${error instanceof Error ? error.message : 'Unknown error'}`,
        timestamp: Date.now(),
        recoverable: true,
      };
      this.addError(videoError);
    }
  }

  private disconnectWebSocket(): void {
    if (this.wsClient) {
      this.wsClient.disconnect();
      this.wsClient = null;
    }

    runInAction(() => {
      this.isWebSocketConnected = false;
    });
  }

  private handleWebSocketStateChange(state: string, error?: Error): void {
    runInAction(() => {
      this.isWebSocketConnected = state === 'connected';

      if (error) {
        const videoError: VideoError = {
          code: 'WEBSOCKET_ERROR',
          message: `WebSocket error: ${error.message}`,
          timestamp: Date.now(),
          recoverable: true,
        };
        this.addError(videoError);
      }
    });
  }

  private handleGameStateMessage(message: WebSocketMessage): void {
    if (!isGameStateMessage(message)) return;

    const gameState = message.data as GameStateData;

    // Convert WebSocket ball data to frontend ball format
    const balls: Ball[] = gameState.balls.map((ballData: BallData) => ({
      id: ballData.id,
      position: { x: ballData.position[0], y: ballData.position[1] },
      radius: ballData.radius,
      type: this.inferBallType(ballData.id, ballData.color),
      number: this.inferBallNumber(ballData.id, ballData.color),
      velocity: ballData.velocity ? { x: ballData.velocity[0], y: ballData.velocity[1] } : { x: 0, y: 0 },
      confidence: ballData.confidence,
      color: ballData.color,
    }));

    // Convert cue data
    let cue: CueStick | null = null;
    if (gameState.cue && gameState.cue.detected) {
      cue = {
        tipPosition: { x: gameState.cue.position[0], y: gameState.cue.position[1] },
        tailPosition: gameState.cue.tip_position
          ? { x: gameState.cue.tip_position[0], y: gameState.cue.tip_position[1] }
          : { x: gameState.cue.position[0] - Math.cos(gameState.cue.angle) * (gameState.cue.length || 100),
              y: gameState.cue.position[1] - Math.sin(gameState.cue.angle) * (gameState.cue.length || 100) },
        angle: gameState.cue.angle,
        elevation: 0,
        detected: gameState.cue.detected,
        confidence: gameState.cue.confidence,
        length: gameState.cue.length || 100,
      };
    }

    // Convert table data
    let table: Table | null = null;
    if (gameState.table && gameState.table.calibrated) {
      table = {
        corners: gameState.table.corners.map(corner => ({ x: corner[0], y: corner[1] })),
        pockets: gameState.table.pockets.map(pocket => ({ x: pocket[0], y: pocket[1] })),
        bounds: { x: 0, y: 0, width: 0, height: 0 }, // Will be calculated
        rails: [], // Will be populated if available
        detected: gameState.table.calibrated,
        confidence: 0.9, // Default high confidence for calibrated table
      };

      // Calculate table bounds from corners
      if (table.corners.length >= 4) {
        const xs = table.corners.map(c => c.x);
        const ys = table.corners.map(c => c.y);
        table.bounds = {
          x: Math.min(...xs),
          y: Math.min(...ys),
          width: Math.max(...xs) - Math.min(...xs),
          height: Math.max(...ys) - Math.min(...ys),
        };
      }
    }

    // Create detection frame
    const detectionFrame: DetectionFrame = {
      balls,
      cue,
      table,
      trajectories: [], // Will be updated by trajectory messages
      timestamp: Date.now(),
      frameNumber: gameState.frame_number || 0,
      processingTime: 0, // Not provided in game state
    };

    runInAction(() => {
      this.setCurrentFrame(detectionFrame);
    });
  }

  private handleTrajectoryMessage(message: WebSocketMessage): void {
    if (!isTrajectoryMessage(message)) return;

    const trajectoryData = message.data as TrajectoryData;

    // Convert trajectory data to frontend format
    const trajectories: Trajectory[] = trajectoryData.lines.map((line, index) => ({
      ballId: `trajectory_${index}`,
      points: [
        { x: line.start[0], y: line.start[1] },
        { x: line.end[0], y: line.end[1] },
      ],
      collisions: trajectoryData.collisions.map(collision => ({
        position: { x: collision.position[0], y: collision.position[1] },
        type: collision.ball_id ? 'ball' : 'rail' as 'ball' | 'rail' | 'pocket',
        targetId: collision.ball_id,
        angle: collision.angle,
        impulse: 0, // Not provided
      })),
      type: line.type,
      probability: line.confidence,
      color: this.getTrajectoryColor(line.type),
    }));

    // Update current frame with trajectory data
    runInAction(() => {
      if (this.currentFrame) {
        this.currentFrame.trajectories = trajectories;
      }
    });
  }

  private inferBallType(id: string, color: string): Ball['type'] {
    if (id.toLowerCase().includes('cue') || id === '0') return 'cue';
    if (id === '8' || id === 'eight') return 'eight';

    // Check if it's a stripe ball (numbers 9-15)
    const ballNumber = parseInt(id);
    if (!isNaN(ballNumber) && ballNumber >= 9 && ballNumber <= 15) return 'stripe';

    // Default to solid for other numbered balls
    if (!isNaN(ballNumber) && ballNumber >= 1 && ballNumber <= 7) return 'solid';

    return 'solid'; // Default fallback
  }

  private inferBallNumber(id: string, color: string): number | undefined {
    const ballNumber = parseInt(id);
    if (!isNaN(ballNumber)) return ballNumber;

    if (id.toLowerCase().includes('cue')) return 0;
    if (id.toLowerCase().includes('eight')) return 8;

    return undefined;
  }

  private getTrajectoryColor(type: string): string {
    const colors = {
      primary: '#00FF00',
      reflection: '#0080FF',
      collision: '#FF8000',
    };
    return colors[type as keyof typeof colors] || '#00FF00';
  }

  // Cleanup
  dispose(): void {
    this.disconnect();

    if (this.performanceInterval) {
      clearInterval(this.performanceInterval);
      this.performanceInterval = null;
    }

    if (this.reconnectTimeout) {
      clearTimeout(this.reconnectTimeout);
      this.reconnectTimeout = null;
    }
  }

  // Utility methods for external components
  getStreamUrl(params?: Record<string, string | number>): string {
    if (!this.streamUrl) {
      // If no stream URL is set, try to generate one using the API client
      try {
        const { apiClient } = require('../api/client');
        return apiClient.getVideoStreamUrl(
          this.getQualityValue(this.config.quality),
          this.config.fps
        );
      } catch {
        return '';
      }
    }

    const url = new URL(this.streamUrl);

    // Add quality and FPS parameters
    url.searchParams.set('quality', this.getQualityValue(this.config.quality).toString());
    url.searchParams.set('fps', this.config.fps.toString());

    // Add any additional parameters
    if (params) {
      Object.entries(params).forEach(([key, value]) => {
        url.searchParams.set(key, value.toString());
      });
    }

    return url.toString();
  }

  private getQualityValue(quality: VideoQuality): number {
    const qualityMap = {
      low: 50,
      medium: 70,
      high: 85,
      ultra: 95,
    };
    return qualityMap[quality];
  }

  // Ball finding utilities
  findBallById(id: string): Ball | undefined {
    return this.currentBalls.find(ball => ball.id === id);
  }

  findBallByNumber(number: number): Ball | undefined {
    return this.currentBalls.find(ball => ball.number === number);
  }

  findCueBall(): Ball | undefined {
    return this.currentBalls.find(ball => ball.type === 'cue');
  }

  getBallsByType(type: Ball['type']): Ball[] {
    return this.currentBalls.filter(ball => ball.type === type);
  }
}
