/**
 * MobX store for managing video streaming state and data
 */

import { makeAutoObservable, action, computed, runInAction } from 'mobx';
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
    if (this.streamUrl) {
      const { apiClient } = require('../api/client');
      this.streamUrl = apiClient.getVideoStreamUrl(
        this.getQualityValue(this.config.quality),
        this.config.fps
      );
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
    if (!this.streamUrl) return '';

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
