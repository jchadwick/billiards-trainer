/**
 * MobX store for managing video streaming state and data
 */

import { action, computed, makeAutoObservable, runInAction } from "mobx";
import {
  createWebSocketClient,
  type WebSocketClient,
} from "../services/websocket-client";
import type {
  BallData,
  CueData,
  GameStateData,
  PhysicalTableDimensions,
  PositionWithScale,
  TableData,
  TablePocketData,
  TrajectoryData,
  WebSocketMessage,
} from "../types/api";
import {
  isGameStateMessage,
  isTrajectoryMessage,
} from "../types/api";
import type {
  Ball,
  CalibrationData,
  CueStick,
  DetectionFrame,
  PerformanceMetrics,
  Table,
  Trajectory,
  VideoError,
  VideoQuality,
  VideoStreamConfig,
  VideoStreamStatus,
  ObservedBallTrail,
  ObservedBallTrailPoint,
} from "../types/video";
import { observedPathFallbacks } from "../config/vision";
import {
  normalizeImageExtentToCanonical,
  normalizeImageLengthToCanonical,
  normalizeImagePointToCanonical,
  normalizeTablePixelExtentsToCanonical,
} from "../utils/coordinates";

function normalizePoint(point: unknown): { x: number; y: number } | null {
  return normalizeImagePointToCanonical(point);
}

function scaleForExtent(scale: unknown): [number, number] | null {
  if (
    !Array.isArray(scale) ||
    scale.length !== 2 ||
    !scale.every((factor) => typeof factor === "number" && Number.isFinite(factor) && factor > 0)
  ) return null;
  return [scale[0], scale[1]];
}

function physicalTableDimensions(value: unknown): PhysicalTableDimensions | undefined {
  if (!value || typeof value !== "object") return undefined;
  const dimensions = value as Record<string, unknown>;
  if (
    dimensions.unit !== "meters" ||
    typeof dimensions.width !== "number" || !Number.isFinite(dimensions.width) || dimensions.width <= 0 ||
    typeof dimensions.height !== "number" || !Number.isFinite(dimensions.height) || dimensions.height <= 0
  ) return undefined;
  return {
    width: dimensions.width,
    height: dimensions.height,
    unit: "meters",
  };
}

export class VideoStore {
  // Stream configuration
  config: VideoStreamConfig = {
    quality: "medium",
    fps: 30,
    autoReconnect: true,
    reconnectDelay: 2000,
  };

  // Stream status
  status: VideoStreamStatus = {
    connected: false,
    streaming: false,
    fps: 0,
    quality: "medium",
    latency: 0,
    errors: 0,
    lastFrameTime: 0,
  };

  // Current detection data
  currentFrame: DetectionFrame | null = null;
  /** Original state payload retained for the live overlay and external consumers. */
  latestGameState: GameStateData | null = null;
  currentFrameImage: string | null = null; // For WebSocket frame streaming
  calibrationData: CalibrationData | null = null;
  /** Latest vision phase metadata, when supplied by the backend. */
  visionPhase: string | null = null;
  calibrationRevision: string | number | null = null;
  mapRevision: string | number | null = null;
  mapTrusted: boolean | null = null;
  aimCalculationEligible: boolean | null = null;
  movingBallCount: number | null = null;
  /** Actual measured ball positions, kept separate from predicted trajectories. */
  observedBallTrails: ObservedBallTrail[] = [];
  latestObservationStatus: GameStateData["observation_status"] | null = null;
  private latestObservationIdentity: string | null = null;
  private latestStateEpochIdentity: string | null = null;
  private latestStateTransportSequence: number | null = null;
  private latestStateSnapshotRevision: number | null = null;
  private latestStateSourceFrameNumber: number | null = null;
  private pendingObservedTrailGaps = new Set<string>();

  // Stream URL and connection
  private streamUrl = "";
  private reconnectTimeout: NodeJS.Timeout | null = null;
  private performanceInterval: NodeJS.Timeout | null = null;

  // WebSocket connection for real-time detection data
  private wsClient: WebSocketClient | null = null;
  private readonly injectedWebSocketClient: WebSocketClient | null;
  private registeredWebSocketClient: WebSocketClient | null = null;
  private ownsWebSocketClient = false;
  private isWebSocketConnected = false;
  private readonly frameMessageHandler = (message: WebSocketMessage) => this.handleFrameMessage(message);
  private readonly stateMessageHandler = (message: WebSocketMessage) => this.handleGameStateMessage(message);
  private readonly trajectoryMessageHandler = (message: WebSocketMessage) => this.handleTrajectoryMessage(message);
  private readonly metricsMessageHandler = (message: WebSocketMessage) => this.handleMetricsMessage(message);
  private readonly alertMessageHandler = (message: WebSocketMessage) => this.handleAlertMessage(message);
  private readonly connectionStateHandler = (state: string, error?: Error) => this.handleWebSocketStateChange(state, error);

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

  constructor(websocketClient?: WebSocketClient) {
    this.injectedWebSocketClient = websocketClient ?? null;
    this.wsClient = websocketClient ?? null;
    makeAutoObservable(this, {
      injectedWebSocketClient: false,
      wsClient: false,
      registeredWebSocketClient: false,
      ownsWebSocketClient: false,
      frameMessageHandler: false,
      stateMessageHandler: false,
      trajectoryMessageHandler: false,
      metricsMessageHandler: false,
      alertMessageHandler: false,
      connectionStateHandler: false,
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
      hasCalibration: computed,
    } as any);

    if (websocketClient) this.registerWebSocketHandlers(websocketClient);

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
    const trajectories = this.currentFrame?.trajectories || [];
    return trajectories.filter((trajectory) =>
      this.isTrajectorySpatiallyValid(trajectory)
    );
  }

  get hasErrors(): boolean {
    return this.errors.length > 0;
  }

  get latestError(): VideoError | null {
    return this.errors.length > 0 ? this.errors[this.errors.length - 1] : null;
  }

  get hasCalibration(): boolean {
    return this.calibrationData !== null && this.calibrationData.isValid;
  }

  /** Live transport state, independent of whether capture has been started. */
  get isRealtimeConnected(): boolean {
    return this.isWebSocketConnected;
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
  async connect(): Promise<void> {
    if (this.status.connected) {
      return;
    }

    this.setLoading(true);
    this.clearErrors();

    try {
      const { apiClient } = await import("../api/client");

      // Check stream status first
      const statusResponse = await apiClient.getStreamStatus();
      if (!statusResponse.success) {
        throw new Error(`Failed to get stream status: ${statusResponse.error}`);
      }

      // Start video capture if not already running
      const startResponse = await apiClient.startVideoCapture();
      if (!startResponse.success) {
        console.warn(
          "Video capture start failed, but continuing:",
          startResponse.error
        );
      }

      // Set stream URL using API client
      this.streamUrl = apiClient.getVideoStreamUrl(
        this.getQualityValue(this.config.quality),
        this.config.fps
      );

      // Initialize WebSocket connection for real-time detection data
      await this.connectWebSocket();

      // Fetch calibration data if available
      await this.fetchCalibrationData();

      runInAction(() => {
        this.status.connected = true;
        this.status.streaming = true;
        this.setLoading(false);
      });
    } catch (error) {
      const videoError: VideoError = {
        code: "CONNECTION_FAILED",
        message:
          error instanceof Error ? error.message : "Unknown connection error",
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
    // Release this store's subscriptions immediately; waiting for the capture
    // API call would leave auxiliary streams active during a slow shutdown.
    this.disconnectWebSocket();

    try {
      const { apiClient } = await import("../api/client");
      await apiClient.stopVideoCapture();
    } catch (error) {
      console.warn("Failed to stop video capture:", error);
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
      const { apiClient } = await import("../api/client");
      const response = await apiClient.getStreamStatus();

      if (response.success && response.data) {
        runInAction(() => {
          this.status.connected = response.data.camera?.connected || false;
          this.status.streaming =
            response.data.streaming?.active_streams > 0 || false;
          this.status.fps = response.data.vision?.processing_fps || 0;
          this.status.latency = response.data.streaming?.avg_fps || 0;
        });
      }
    } catch (error) {
      const videoError: VideoError = {
        code: "STATUS_REFRESH_FAILED",
        message:
          error instanceof Error ? error.message : "Failed to refresh status",
        timestamp: Date.now(),
        recoverable: true,
      };
      this.addError(videoError);
    }
  }

  async fetchCalibrationData(): Promise<void> {
    try {
      const { apiClient } = await import("../api/client");
      const response = await apiClient.getCalibrationData();

      if (response.success && response.data) {
        // Transform API response to CalibrationData format
        const apiData = response.data;
        const sourceResolution = this.calibrationSourceResolution(apiData);
        const corners = (apiData.corners || [])
          .map((corner: unknown, index: number) => {
            const record = corner && typeof corner === "object" && !Array.isArray(corner)
              ? corner as Record<string, unknown>
              : {};
            const imagePoint = normalizeImagePointToCanonical(
              record.image_point ?? corner,
              sourceResolution,
            );
            if (!imagePoint) return null;
            const world = record.world_position ?? record.world;
            const worldPosition = world && typeof world === "object"
              ? world as { x?: unknown; y?: unknown; unit?: unknown }
              : undefined;
            return {
              id: `corner-${index}`,
              screenPosition: imagePoint,
              ...(worldPosition?.unit === "meters" &&
                typeof worldPosition.x === "number" && Number.isFinite(worldPosition.x) &&
                typeof worldPosition.y === "number" && Number.isFinite(worldPosition.y)
                ? { worldPosition: { x: worldPosition.x, y: worldPosition.y, unit: "meters" as const } }
                : {}),
              timestamp: Date.now(),
              confidence: apiData.accuracy || 1.0,
            };
          })
          .filter((corner: unknown): corner is NonNullable<typeof corner> => corner !== null);

        const calibrationData: CalibrationData = {
          corners,
          transformationMatrix: apiData.transformation_matrix,
          calibratedAt: apiData.calibrated_at ? new Date(apiData.calibrated_at).getTime() : Date.now(),
          accuracy: apiData.accuracy,
          isValid: apiData.is_valid !== false && corners.length >= 4,
          tableType: apiData.table_type,
          dimensions: apiData.dimensions && {
            width: apiData.dimensions.width,
            height: apiData.dimensions.height,
            ...(apiData.dimensions.unit === "meters" ? { unit: "meters" as const } : {}),
          },
        };

        runInAction(() => {
          this.calibrationData = calibrationData;
        });
      } else {
        // No calibration data available or invalid response
        runInAction(() => {
          this.calibrationData = null;
        });
      }
    } catch (error) {
      console.warn("Failed to fetch calibration data:", error);
      // Don't add error since calibration is optional
      runInAction(() => {
        this.calibrationData = null;
      });
    }
  }

  private calibrationSourceResolution(data: Record<string, any>): { width: number; height: number } | undefined {
    const resolution = data.source_resolution ?? data.calibration_resolution;
    if (resolution && typeof resolution === "object" && !Array.isArray(resolution)) {
      const width = resolution.width;
      const height = resolution.height;
      if (Number.isFinite(width) && width > 0 && Number.isFinite(height) && height > 0) {
        return { width, height };
      }
    }
    if (Array.isArray(resolution) && resolution.length === 2) {
      const [width, height] = resolution;
      if (Number.isFinite(width) && width > 0 && Number.isFinite(height) && height > 0) {
        return { width, height };
      }
    }
    const width = data.calibration_resolution_width ?? data.resolution_width;
    const height = data.calibration_resolution_height ?? data.resolution_height;
    return Number.isFinite(width) && width > 0 && Number.isFinite(height) && height > 0
      ? { width, height }
      : undefined;
  }

  async captureFrame(): Promise<string | null> {
    try {
      const { apiClient } = await import("../api/client");
      const frameUrl = apiClient.getSingleFrameUrl(
        this.getQualityValue(this.config.quality)
      );

      return frameUrl;
    } catch (error) {
      const videoError: VideoError = {
        code: "FRAME_CAPTURE_FAILED",
        message:
          error instanceof Error ? error.message : "Failed to capture frame",
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
        const { apiClient } = require("../api/client");
        this.streamUrl = apiClient.getVideoStreamUrl(
          this.getQualityValue(this.config.quality),
          this.config.fps
        );
      } catch (error) {
        console.error("Failed to update stream URL:", error);
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
      if (!this.status.connected) {
        // Reconnect using the already configured base URL
        this.connect().catch(console.error);
      }
    }, this.config.reconnectDelay);
  }

  private startPerformanceMonitoring(): void {
    this.performanceInterval = setInterval(() => {
      // Update performance metrics

      // Simple memory usage estimation (if available)
      if ("memory" in performance) {
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
  private async connectWebSocket(): Promise<void> {
    try {
      const axiosClient = (await import("../api/axios-client")).default;

      // Get WebSocket URL from axios client
      const wsUrl = axiosClient.getWebSocketUrl("/api/v1/websocket/ws");

      // Production injects the RootStore's shared client. A standalone store
      // retains its historical ability to create its own client for tests and
      // isolated use, while never creating a second one when injected.
      if (!this.wsClient) {
        this.wsClient = createWebSocketClient({ url: wsUrl });
        this.ownsWebSocketClient = true;
      }
      this.wsClient.updateConfig({ url: wsUrl, autoReconnect: true });
      this.registerWebSocketHandlers(this.wsClient);

      // SystemStore owns the state subscription. This store requests only the
      // auxiliary streams it consumes when shared. Standalone stores have no
      // SystemStore, so retain their previous self-contained state behavior.
      const streams = this.ownsWebSocketClient
        ? ["state", "frame", "trajectory", "metrics", "alert"]
        : ["frame", "trajectory", "metrics", "alert"];
      this.wsClient.subscribe(
        streams,
        { quality: this.config.quality, frame_rate: this.config.fps },
      );
      await this.wsClient.connect();

      runInAction(() => {
        this.isWebSocketConnected = true;
      });

      console.log("WebSocket connected for real-time detection data");
    } catch (error) {
      console.error("Failed to connect WebSocket:", error);
      const videoError: VideoError = {
        code: "WEBSOCKET_CONNECTION_FAILED",
        message: `WebSocket connection failed: ${error instanceof Error ? error.message : "Unknown error"}`,
        timestamp: Date.now(),
        recoverable: true,
      };
      this.addError(videoError);
    }
  }

  private disconnectWebSocket(): void {
    if (this.wsClient) {
      this.wsClient.unsubscribe(["frame", "trajectory", "metrics", "alert"]);
      if (this.ownsWebSocketClient) {
        this.unregisterWebSocketHandlers();
        this.wsClient.destroy();
        this.wsClient = null;
        this.ownsWebSocketClient = false;
      }
    }

    runInAction(() => {
      this.isWebSocketConnected = false;
    });
  }

  private registerWebSocketHandlers(client: WebSocketClient): void {
    if (this.registeredWebSocketClient === client) return;
    this.unregisterWebSocketHandlers();
    client.on("frame", this.frameMessageHandler);
    client.on("state", this.stateMessageHandler);
    client.on("trajectory", this.trajectoryMessageHandler);
    client.on("metrics", this.metricsMessageHandler);
    client.on("alert", this.alertMessageHandler);
    client.onConnectionState(this.connectionStateHandler);
    this.registeredWebSocketClient = client;
  }

  private unregisterWebSocketHandlers(): void {
    if (!this.registeredWebSocketClient) return;
    this.registeredWebSocketClient.off("frame", this.frameMessageHandler);
    this.registeredWebSocketClient.off("state", this.stateMessageHandler);
    this.registeredWebSocketClient.off("trajectory", this.trajectoryMessageHandler);
    this.registeredWebSocketClient.off("metrics", this.metricsMessageHandler);
    this.registeredWebSocketClient.off("alert", this.alertMessageHandler);
    this.registeredWebSocketClient.offConnectionState(this.connectionStateHandler);
    this.registeredWebSocketClient = null;
  }

  private handleWebSocketStateChange(state: string, error?: Error): void {
    runInAction(() => {
      this.isWebSocketConnected = state === "connected";

      if (error) {
        const videoError: VideoError = {
          code: "WEBSOCKET_ERROR",
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
    const observationStatus = gameState.observation_status;
    const sourceIdentity = observationStatus?.observation_identity;
    const producerEngineId = sourceIdentity?.engine_instance_id
      ?? observationStatus?.engine_instance_id
      ?? observationStatus?.vision_instance_id;
    const reportedSessionId = sourceIdentity?.session_id
      ?? observationStatus?.source_session_id
      ?? observationStatus?.source_process_id;
    const producerSessionId = typeof reportedSessionId === 'string'
      || typeof reportedSessionId === 'number'
      ? String(reportedSessionId)
      : undefined;
    if (
      !this.acceptStateObservation(
        observationStatus,
        gameState.frame_number,
        producerEngineId,
        producerSessionId,
      )
    ) return;
    const trailIdentity = typeof producerEngineId === 'string'
      && producerSessionId !== undefined
      ? `${producerEngineId}\u0000${producerSessionId}`
      : null;
    if (trailIdentity !== null && trailIdentity !== this.latestObservationIdentity) {
      // Track IDs are only unique within an engine/session. A restarted source
      // may reuse the same ID and reset its timestamp or frame counter.
      this.observedBallTrails = [];
      this.pendingObservedTrailGaps.clear();
    }
    if (trailIdentity !== null) this.latestObservationIdentity = trailIdentity;
    const phase = observationStatus?.phase ?? gameState.phase;
    const previousPhase = this.visionPhase;
    const calibrationRevision =
      observationStatus?.calibration_revision ?? gameState.calibration_revision;

    if (phase !== undefined) this.visionPhase = phase;
    if (
      this.isMotionOrRecoveryPhase(phase) &&
      !this.isMotionOrRecoveryPhase(previousPhase) &&
      this.isMotionPhase(phase)
    ) {
      // Start a distinct actual path for each new shot while retaining the
      // frozen prediction that the path will be compared against.
      this.observedBallTrails = [];
      this.pendingObservedTrailGaps.clear();
    }
    this.latestObservationStatus = observationStatus ?? null;
    if (calibrationRevision !== undefined) {
      this.calibrationRevision = calibrationRevision;
    }
    const mapRevision = observationStatus?.map_revision ?? gameState.map_revision;
    if (mapRevision !== undefined) this.mapRevision = mapRevision;
    const mapTrusted = observationStatus?.map_trusted ?? gameState.map_trusted;
    if (mapTrusted !== undefined) this.mapTrusted = mapTrusted;
    const aimCalculationEligible =
      observationStatus?.aim_calculation_eligible ?? gameState.aim_calculation_eligible;
    if (aimCalculationEligible !== undefined) {
      this.aimCalculationEligible = aimCalculationEligible;
    }
    const movingBallCount =
      observationStatus?.moving_ball_count ?? gameState.moving_ball_count;
    if (movingBallCount !== undefined) this.movingBallCount = movingBallCount;

    if (this.isMotionOrRecoveryPhase(phase)) {
      // A moving or recovering map cannot authorize new aim guidance. Keep these
      // false until a later state message explicitly reports trusted eligibility.
      this.mapTrusted = false;
      this.aimCalculationEligible = false;
      this.freezeCurrentPrediction();
    }

    // Normalize all image geometry once at the transport boundary. Radius and
    // velocity each carry independent source-to-canonical scales.
    const balls: Ball[] = gameState.balls.flatMap((ballData: BallData) => {
      if (
        typeof phase === "string" && phase.startsWith("rust_scene_") &&
        ballData.image_space !== "raw"
      ) {
        return [];
      }
      if (ballData.radius_unit !== undefined && ballData.radius_unit !== "canonical_pixels") {
        return [];
      }
      const position = normalizePoint(ballData.position);
      const radiusScale = scaleForExtent(ballData.radius_scale ?? (
        ballData.radius_unit === "canonical_pixels" ? [1, 1] : ballData.position.scale
      ));
      const radiusExtent = ballData.radius_extent
        ? normalizePoint(ballData.radius_extent)
        : radiusScale
          ? normalizeImageExtentToCanonical(ballData.radius, radiusScale)
          : null;
      if (!position || !radiusExtent) return [];

      const velocity = ballData.velocity
        ? normalizePoint(ballData.velocity)
        : { x: 0, y: 0 };
      if (!velocity) return [];
      const number = this.inferBallNumber(
        ballData.id,
        ballData.number,
        ballData.is_cue_ball === true,
      );
      return [{
        id: ballData.id,
        position,
        radius: Math.sqrt(radiusExtent.x * radiusExtent.y),
        radiusX: radiusExtent.x,
        radiusY: radiusExtent.y,
        type: this.inferBallType(
          ballData.id,
          ballData.number,
          ballData.is_cue_ball === true,
        ),
        number,
        identityStatus: ballData.identity_status,
        identityConfidence: ballData.identity_confidence,
        identityEvidence: ballData.identity_evidence,
        velocity,
        confidence: ballData.confidence,
        confidenceKind: ballData.confidence_kind,
        lastObservationConfidence: ballData.last_observation_confidence,
        color: ballData.color ?? "",
        isObserved: ballData.is_observed !== false,
        trackingStatus: ballData.tracking_status,
        positionKind: ballData.position_kind,
        tablePositionMm: ballData.table_position
          ? { x: ballData.table_position.x, y: ballData.table_position.y }
          : undefined,
        sourceObservation: ballData.source_observation,
      }];
    });

    this.recordObservedBallPositions(gameState, balls);

    // Cue endpoints are normalized independently. A pixel length is used only
    // when the backend has not supplied both endpoints and carries its own scale.
    let cue: CueStick | null = null;
    if (gameState.cue && gameState.cue.detected) {
      cue = this.normalizeCue(gameState.cue);
    }

    // Normalize every boundary point and pocket center before deriving bounds.
    let table: Table | null = null;
    if (gameState.table && gameState.table.calibrated !== false) {
      table = this.normalizeTable(gameState.table);
    }

    // Create detection frame
    const detectionFrame: DetectionFrame = {
      balls,
      cue,
      table,
      // A state message carries observed ball state, not a new prediction. Keep the
      // last prediction visible so observed movement can be compared against it.
      trajectories: this.currentFrame?.trajectories ?? [],
      timestamp: Date.now(),
      frameNumber: gameState.frame_number || 0,
      processingTime: 0, // Not provided in game state
    };

    runInAction(() => {
      this.latestGameState = gameState;
      this.setCurrentFrame(detectionFrame);
    });
  }

  /**
   * Reject out-of-order states only when the producer epoch and ordering
   * watermark are explicit. A producer restart creates a new sequence space;
   * legacy payloads without comparable provenance keep their existing contract.
   */
  private acceptStateObservation(
    status: GameStateData["observation_status"],
    frameNumber?: number,
    engineId?: unknown,
    sessionId?: string,
  ): boolean {
    if (typeof engineId !== "string" || typeof sessionId !== "string") return true;

    const epochIdentity = `${engineId}\u0000${sessionId}`;
    if (epochIdentity !== this.latestStateEpochIdentity) {
      this.latestStateEpochIdentity = epochIdentity;
      this.latestStateTransportSequence = null;
      this.latestStateSnapshotRevision = null;
      this.latestStateSourceFrameNumber = null;
    }

    const transportSequence = status?.transport_sequence;
    if (typeof transportSequence === "number" && Number.isSafeInteger(transportSequence)) {
      if (
        this.latestStateTransportSequence !== null
        && transportSequence <= this.latestStateTransportSequence
      ) return false;
      this.latestStateTransportSequence = transportSequence;
      return true;
    }

    const snapshotRevision = status?.observation_identity?.snapshot_revision
      ?? status?.snapshot_revision;
    if (typeof snapshotRevision === "number" && Number.isSafeInteger(snapshotRevision)) {
      if (
        this.latestStateSnapshotRevision !== null
        && snapshotRevision <= this.latestStateSnapshotRevision
      ) return false;
      this.latestStateSnapshotRevision = snapshotRevision;
      return true;
    }

    const sourceFrameNumber = status?.source_frame_number ?? frameNumber;
    if (typeof sourceFrameNumber === "number" && Number.isSafeInteger(sourceFrameNumber)) {
      if (
        this.latestStateSourceFrameNumber !== null
        && sourceFrameNumber < this.latestStateSourceFrameNumber
      ) return false;
      this.latestStateSourceFrameNumber = sourceFrameNumber;
    }
    return true;
  }

  private normalizeCue(cueData: CueData): CueStick | null {
    const tip = normalizePoint(cueData.tip_position ?? cueData.position);
    if (!tip) return null;
    const tailInput = cueData.butt_position ??
      (cueData.position && cueData.tip_position ? cueData.position : undefined);
    let tail = tailInput ? normalizePoint(tailInput) : null;
    if (tailInput && !tail) return null;

    const angle = cueData.angle * Math.PI / 180;
    let length = tail ? Math.hypot(tip.x - tail.x, tip.y - tail.y) : 0;
    if (!tail && cueData.length !== undefined) {
      const scale = scaleForExtent(
        cueData.length_scale ?? (cueData.tip_position ?? cueData.position)?.scale,
      );
      const canonicalLength = scale
        ? normalizeImageLengthToCanonical(cueData.length, scale, angle)
        : null;
      if (canonicalLength === null) return null;
      length = canonicalLength;
      tail = {
        x: tip.x - Math.cos(angle) * length,
        y: tip.y - Math.sin(angle) * length,
      };
    }
    if (!tail) return null;

    return {
      tipPosition: tip,
      tailPosition: tail,
      angle,
      elevation: 0,
      detected: cueData.detected === true,
      confidence: cueData.confidence,
      length,
    };
  }

  private normalizeTable(tableData: TableData): Table | null {
    const sourcePockets: Array<PositionWithScale | TablePocketData> = tableData.pockets?.length
      ? tableData.pockets
      : (tableData.pocket_positions ?? []).map((position) => ({
        position,
        radius: tableData.pocket_radius,
        radius_scale: tableData.pocket_radius_scale,
        radius_unit: tableData.pocket_radius_unit,
      }));
    const pockets = (sourcePockets ?? []).map((pocket) => {
      const position = pocket && typeof pocket === "object" && "position" in pocket
        ? pocket.position
        : pocket as PositionWithScale;
      return normalizePoint(position);
    });
    if (pockets.some((pocket) => pocket === null)) return null;

    const sourceCorners = (tableData.playing_area_corners ?? tableData.corners ?? []).map(normalizePoint);
    if (sourceCorners.some((corner) => corner === null)) return null;
    const corners = sourceCorners as Array<{ x: number; y: number }>;

    const rails: Array<Array<{ x: number; y: number }>> = [];
    for (const rail of tableData.rails ?? []) {
      let sourcePoints: unknown[];
      if (Array.isArray(rail)) {
        sourcePoints = rail;
      } else {
        const segment = rail as Record<string, unknown>;
        if (!segment.start || !segment.end) return null;
        sourcePoints = [segment.start, segment.end];
      }
      const normalized = sourcePoints.map(normalizePoint);
      if (normalized.length < 2 || normalized.some((point) => point === null)) return null;
      rails.push(normalized as Array<{ x: number; y: number }>);
    }

    const pocketRadii: Array<{ x: number; y: number }> = [];
    for (const pocket of sourcePockets ?? []) {
      if (!(pocket && typeof pocket === "object" && "position" in pocket) || pocket.radius === undefined) {
        pocketRadii.push({ x: 0, y: 0 });
        continue;
      }
      const radiusUnit = pocket.radius_unit ?? tableData.pocket_radius_unit;
      if (radiusUnit !== undefined && radiusUnit !== "canonical_pixels") {
        pocketRadii.push({ x: 0, y: 0 });
        continue;
      }
      const scale = scaleForExtent(pocket.radius_scale ?? (
        radiusUnit === "canonical_pixels"
          ? tableData.pocket_radius_scale ?? [1, 1]
          : pocket.position.scale
      ));
      const extent = pocket.radius_extent
        ? normalizePoint(pocket.radius_extent)
        : scale
          ? normalizeImageExtentToCanonical(pocket.radius, scale)
          : null;
      if (!extent) return null;
      pocketRadii.push(extent);
    }

    const validPockets = pockets as Array<{ x: number; y: number }>;
    const xs = corners.map((corner) => corner.x);
    const ys = corners.map((corner) => corner.y);
    const pixelExtents = normalizeTablePixelExtentsToCanonical(tableData);
    const physicalDimensions = physicalTableDimensions(tableData.real_world_dimensions) ??
      physicalTableDimensions(tableData.dimensions);
    return {
      corners,
      pockets: validPockets,
      ...(corners.length >= 4 ? {
        bounds: {
          x: Math.min(...xs),
          y: Math.min(...ys),
          width: Math.max(...xs) - Math.min(...xs),
          height: Math.max(...ys) - Math.min(...ys),
        },
      } : {}),
      rails,
      pocketRadii,
      width: pixelExtents.width,
      height: pixelExtents.height,
      dimensions: physicalDimensions,
      detected: tableData.calibrated !== false,
      confidence: 0.9,
    };
  }

  /**
   * Append only full successful observations. Cached positions remain useful
   * for rendering the stable map, but must never be mistaken for measurements.
   */
  private recordObservedBallPositions(gameState: GameStateData, balls: Ball[]): void {
    const observation = gameState.observation_status;
    if (observation?.ball_positions_status !== "observed") {
      this.observedBallTrails.forEach((trail) =>
        this.pendingObservedTrailGaps.add(trail.ballId)
      );
      return;
    }

    const timestamp = this.toTimestampSeconds(
      observation.ball_observation_source_timestamp ??
        observation.source_timestamp ??
        observation.ball_observation_timestamp,
      observation.ball_observation_source_timestamp_unit ??
        observation.source_timestamp_unit,
      observation.phase === "rust_scene_observation" ||
        observation.phase === "rust_scene_health",
    );
    const sourceFrameNumber =
      observation.ball_observation_source_frame_number ?? gameState.frame_number;
    const timestampKind =
      observation.ball_observation_source_timestamp_kind ??
      (observation.ball_observation_timestamp !== undefined
        ? "vision_monotonic_observation"
        : "unknown_source_timestamp");

    if (timestamp === null) {
      this.observedBallTrails.forEach((trail) =>
        this.pendingObservedTrailGaps.add(trail.ballId)
      );
      return;
    }

    const maximumGapSeconds = this.positiveSetting(
      observation.maximum_ball_observation_gap_seconds,
      observedPathFallbacks.maximumGapSeconds
    );
    const maximumPoints = Math.floor(
      this.positiveSetting(
        observation.actual_path_history_max_points,
        observedPathFallbacks.maximumPointsPerBall
      )
    );
    const presentIds = new Set(balls.map((ball) => ball.id));
    this.observedBallTrails.forEach((trail) => {
      if (!presentIds.has(trail.ballId)) {
        this.pendingObservedTrailGaps.add(trail.ballId);
      }
    });

    const trailsById = new Map(
      this.observedBallTrails.map((trail) => [trail.ballId, trail])
    );
    balls.forEach((ball) => {
      const trail = trailsById.get(ball.id) ?? { ballId: ball.id, points: [] };
      const previous = trail.points[trail.points.length - 1];
      if (
        previous &&
        sourceFrameNumber !== undefined &&
        previous.sourceFrameNumber !== undefined &&
        sourceFrameNumber <= previous.sourceFrameNumber
      ) {
        if (sourceFrameNumber < previous.sourceFrameNumber) {
          this.pendingObservedTrailGaps.add(ball.id);
        }
        return;
      }
      if (
        previous &&
        previous.timestampKind === timestampKind &&
        timestamp <= previous.timestamp
      ) {
        if (timestamp < previous.timestamp) {
          this.pendingObservedTrailGaps.add(ball.id);
        }
        return;
      }

      const gapByTime = Boolean(
        previous &&
          (previous.timestampKind !== timestampKind ||
            timestamp - previous.timestamp > maximumGapSeconds)
      );
      const gapPending = this.pendingObservedTrailGaps.delete(ball.id);
      const point: ObservedBallTrailPoint = {
        x: ball.position.x,
        y: ball.position.y,
        timestamp,
        timestampKind,
        sourceFrameNumber,
        gapBefore: gapByTime || gapPending,
      };
      trail.points.push(point);
      if (trail.points.length > maximumPoints) {
        trail.points.splice(0, trail.points.length - maximumPoints);
      }
      trailsById.set(ball.id, trail);
    });
    this.observedBallTrails = Array.from(trailsById.values());
  }

  private toTimestampSeconds(
    value: unknown,
    unit?: string,
    requireExplicitUnit = false,
  ): number | null {
    if (unit !== undefined) {
      const numericValue =
        typeof value === "number" || typeof value === "string"
          ? Number(value)
          : Number.NaN;
      if (!Number.isFinite(numericValue) || numericValue < 0) return null;
      switch (unit) {
        case "nanoseconds":
          return numericValue / 1_000_000_000;
        case "milliseconds":
          return numericValue / 1_000;
        case "seconds":
          return numericValue;
        default:
          return null;
      }
    }
    if (requireExplicitUnit) return null;
    if (typeof value === "number" && Number.isFinite(value)) {
      return value > 1e12 ? value / 1000 : value;
    }
    if (typeof value === "string") {
      const numericValue = Number(value);
      if (Number.isFinite(numericValue)) {
        return numericValue > 1e12 ? numericValue / 1000 : numericValue;
      }
      const parsed = Date.parse(value);
      return Number.isFinite(parsed) ? parsed / 1000 : null;
    }
    return null;
  }

  private positiveSetting(value: unknown, fallback: number): number {
    const numericValue = Number(value);
    return Number.isFinite(numericValue) && numericValue > 0
      ? numericValue
      : fallback;
  }

  private handleTrajectoryMessage(message: WebSocketMessage): void {
    if (!isTrajectoryMessage(message)) return;

    const trajectoryData = message.data as TrajectoryData;

    if (trajectoryData.spatial_valid === false) {
      // Treat an explicit spatial invalidation as authoritative even when either
      // side lacks a calibration revision to compare.
      runInAction(() => {
        if (this.currentFrame) {
          this.currentFrame.trajectories = this.currentFrame.trajectories.map(
            (trajectory) => ({ ...trajectory, spatialValid: false })
          );
        }
      });
      return;
    }
    if (!this.isValidTrajectoryMessage(trajectoryData)) return;
    const predictionStatus = trajectoryData.prediction_status;
    if (predictionStatus !== "live" && predictionStatus !== "frozen_reference") return;

    // A live aim prediction is only eligible to replace a retained reference while
    // aiming is valid. Frozen references are accepted through motion/recovery.
    const predictionPhase = trajectoryData.phase ?? this.visionPhase;
    if (
      predictionStatus === "live" &&
      (this.isMotionOrRecoveryPhase(predictionPhase) ||
        (!!predictionPhase && !this.isAimingPhase(predictionPhase)) ||
        this.aimCalculationEligible === false ||
        this.mapTrusted === false)
    ) {
      return;
    }

    // Collision metadata can identify an otherwise unlabelled trajectory, while
    // producers without either identity get a stable ID based on their line order.
    const collisionBallId = trajectoryData.collisions[0]?.ball1_id;

    // Convert trajectory data to frontend format
    // Note: API now sends positions as {x, y, scale} dicts instead of [x, y] arrays
    const trajectories: Trajectory[] = trajectoryData.lines.map(
      (line, index) => ({
        ballId: line.ball_id || collisionBallId || `trajectory_${index}`,
        points: [
          normalizePoint(line.start)!,
          normalizePoint(line.end)!,
        ],
        collisions: trajectoryData.collisions.map((collision) => ({
          position: normalizePoint(collision.position)!,
          // Use ball2_id to determine collision type (if ball2_id exists, it's a ball collision)
          type: collision.ball2_id
            ? "ball"
            : (collision.type === "ball_pocket" ? "pocket" : "rail") as "ball" | "rail" | "pocket",
          targetId: collision.ball2_id || undefined,  // Target ball ID (ball being hit)
          angle: collision.angle,
          impulse: 0, // Not provided
        })),
        type: line.type,
        probability: line.confidence,
        color: this.getTrajectoryColor(line.type),
        predictionStatus,
        predictionId: trajectoryData.prediction_id,
        sourceFrameNumber: trajectoryData.source_frame_number,
        sourceTimestamp: trajectoryData.source_timestamp,
        sourceMapRevision: trajectoryData.map_revision,
        calibrationRevision: trajectoryData.calibration_revision,
        sourceObservations: trajectoryData.source_observations,
        spatialValid: trajectoryData.spatial_valid,
      })
    );

    // Update current frame with trajectory data
    runInAction(() => {
      if (this.currentFrame) {
        this.currentFrame.trajectories = trajectories;
      }
    });
  }

  private isMotionOrRecoveryPhase(phase: string | undefined | null): boolean {
    if (!phase) return false;
    const normalized = phase.toLowerCase().replace(/[\s-]+/g, "_");
    return ["limited_motion", "many_moving", "many_balls_moving", "strike", "recovering", "recovery", "uncertain"].some(
      (phaseName) => normalized.includes(phaseName)
    );
  }

  private isMotionPhase(phase: string | undefined | null): boolean {
    if (!phase) return false;
    const normalized = phase.toLowerCase().replace(/[\s-]+/g, "_");
    return ["limited_motion", "many_moving", "many_balls_moving", "strike"].some(
      (phaseName) => normalized.includes(phaseName)
    );
  }

  private isAimingPhase(phase: string): boolean {
    const normalized = phase.toLowerCase().replace(/[\s-]+/g, "_");
    return normalized.includes("aiming") || normalized.includes("setup");
  }

  private freezeCurrentPrediction(): void {
    if (!this.currentFrame?.trajectories.length) return;
    this.currentFrame.trajectories = this.currentFrame.trajectories.map((trajectory) => ({
      ...trajectory,
      predictionStatus: "frozen_reference",
    }));
  }

  private isValidTrajectoryMessage(trajectoryData: TrajectoryData): boolean {
    if (
      !Array.isArray(trajectoryData.lines) ||
      trajectoryData.lines.length === 0 ||
      !Array.isArray(trajectoryData.collisions)
    ) {
      return false;
    }
    return trajectoryData.lines.every((line) => {
      const start = line?.start;
      const end = line?.end;
      return (
        normalizePoint(start) !== null &&
        normalizePoint(end) !== null
      );
    }) && trajectoryData.collisions.every((collision) =>
      normalizePoint(collision?.position) !== null
    );
  }

  private isTrajectorySpatiallyValid(trajectory: Trajectory): boolean {
    if (trajectory.spatialValid === false) return false;
    return !(
      this.calibrationRevision !== null &&
      trajectory.calibrationRevision !== undefined &&
      String(trajectory.calibrationRevision) !== String(this.calibrationRevision)
    );
  }

  private handleFrameMessage(message: WebSocketMessage): void {
    if (!message || message.type !== "frame") return;

    try {
      const frameData = message.data as any;

      // Decode base64 image and create data URL
      const imageUrl = `data:image/${frameData.format || "jpeg"};base64,${frameData.image}`;

      // Update frame image for rendering
      runInAction(() => {
        this.currentFrameImage = imageUrl;
        this.status.fps = frameData.fps || this.status.fps;
        this.status.lastFrameTime = Date.now();

        // Update performance metrics
        this.updateFPSCalculation();
      });
    } catch (error) {
      console.error("Failed to process frame message:", error);
    }
  }

  private handleMetricsMessage(message: WebSocketMessage): void {
    if (!message || message.type !== "metrics") return;

    try {
      const metricsData = message.data as any;

      // Update performance metrics from server
      runInAction(() => {
        if (metricsData.fps !== undefined) this.status.fps = metricsData.fps;
        if (metricsData.latency_ms !== undefined)
          this.status.latency = metricsData.latency_ms;
        if (metricsData.dropped_frames !== undefined)
          this.status.droppedFrames = metricsData.dropped_frames;
      });
    } catch (error) {
      console.error("Failed to process metrics message:", error);
    }
  }

  private handleAlertMessage(message: WebSocketMessage): void {
    if (!message || message.type !== "alert") return;

    try {
      const alertData = message.data as any;

      // Handle alerts by converting to errors if needed
      if (alertData.level === "error" || alertData.level === "critical") {
        const videoError: VideoError = {
          code: alertData.code || "SYSTEM_ALERT",
          message: alertData.message,
          timestamp: Date.now(),
          recoverable: alertData.level !== "critical",
        };
        this.addError(videoError);
      }

      console.log(`Alert received: ${alertData.level} - ${alertData.message}`);
    } catch (error) {
      console.error("Failed to process alert message:", error);
    }
  }

  private inferBallType(
    id: string,
    explicitNumber?: number | null,
    isCueBall = false,
  ): Ball["type"] {
    if (isCueBall || explicitNumber === 0) return "cue";
    if (explicitNumber !== undefined) {
      if (explicitNumber === null) return "unknown";
      return this.ballTypeForNumber(explicitNumber);
    }

    const legacyId = id.trim().toLowerCase();
    if (["cue", "cue-ball", "cue_ball", "0"].includes(legacyId)) return "cue";
    const number = this.inferBallNumber(id);
    return number === undefined ? "unknown" : this.ballTypeForNumber(number);
  }

  private inferBallNumber(
    id: string,
    explicitNumber?: number | null,
    isCueBall = false,
  ): number | undefined {
    if (isCueBall) return 0;
    if (explicitNumber !== undefined) return explicitNumber ?? undefined;

    const legacyId = id.trim().toLowerCase();
    if (["cue", "cue-ball", "cue_ball"].includes(legacyId)) return 0;
    if (legacyId === "eight") return 8;
    return /^(?:0|[1-9]|1[0-5])$/.test(legacyId)
      ? Number(legacyId)
      : undefined;
  }

  private ballTypeForNumber(number: number): Ball["type"] {
    if (number === 0) return "cue";
    if (number === 8) return "eight";
    if (number >= 9 && number <= 15) return "stripe";
    if (number >= 1 && number <= 7) return "solid";
    return "unknown";
  }

  private getTrajectoryColor(type: string): string {
    const colors = {
      primary: "#00FF00",
      reflection: "#0080FF",
      collision: "#FF8000",
    };
    return colors[type as keyof typeof colors] || "#00FF00";
  }

  // Cleanup
  dispose(): void {
    this.unregisterWebSocketHandlers();
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
        const { apiClient } = require("../api/client");
        return apiClient.getVideoStreamUrl(
          this.getQualityValue(this.config.quality),
          this.config.fps
        );
      } catch {
        return "";
      }
    }

    const url = new URL(this.streamUrl);

    // Add quality and FPS parameters
    url.searchParams.set(
      "quality",
      this.getQualityValue(this.config.quality).toString()
    );
    url.searchParams.set("fps", this.config.fps.toString());

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
    return this.currentBalls.find((ball) => ball.id === id);
  }

  findBallByNumber(number: number): Ball | undefined {
    return this.currentBalls.find((ball) => ball.number === number);
  }

  findCueBall(): Ball | undefined {
    return this.currentBalls.find((ball) => ball.type === "cue");
  }

  getBallsByType(type: Ball["type"]): Ball[] {
    return this.currentBalls.filter((ball) => ball.type === type);
  }
}
