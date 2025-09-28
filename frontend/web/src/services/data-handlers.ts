/**
 * Real-time data handlers for game state, vision data, and other WebSocket streams
 */

import {
  WebSocketMessage,
  FrameData,
  GameStateData,
  TrajectoryData,
  AlertData,
  ConfigData,
  MetricsData,
  StatusData,
  BallData,
  CueData,
  TableData,
  isFrameMessage,
  isGameStateMessage,
  isTrajectoryMessage,
  isAlertMessage,
} from '../types/api';

export interface DataProcessingOptions {
  frameProcessing?: {
    maxFps?: number;
    qualityThreshold?: number;
    enableCompression?: boolean;
    resizeToFit?: boolean;
    targetWidth?: number;
    targetHeight?: number;
  };
  stateProcessing?: {
    smoothingFactor?: number;
    confidenceThreshold?: number;
    enablePrediction?: boolean;
    maxHistoryLength?: number;
  };
  trajectoryProcessing?: {
    enableSmoothing?: boolean;
    confidenceThreshold?: number;
    maxPredictionTime?: number;
  };
}

export interface ProcessedFrameData {
  imageUrl: string;
  originalData: FrameData;
  processedAt: Date;
  displayWidth: number;
  displayHeight: number;
  aspectRatio: number;
  isCompressed: boolean;
}

export interface ProcessedGameState {
  timestamp: Date;
  balls: ProcessedBallData[];
  cue?: ProcessedCueData;
  table?: ProcessedTableData;
  confidence: number;
  isValid: boolean;
  frameNumber?: number;
  changesSinceLastFrame: string[];
}

export interface ProcessedBallData extends BallData {
  interpolatedPosition?: [number, number];
  predictedPosition?: [number, number];
  movementDirection?: number; // angle in degrees
  speed?: number;
  isMoving: boolean;
  hasPositionChanged: boolean;
}

export interface ProcessedCueData extends CueData {
  predictedTrajectory?: [number, number][];
  aimingAccuracy?: number;
  recommendedAdjustment?: {
    angle: number;
    force: number;
  };
}

export interface ProcessedTableData extends TableData {
  scaleFactor?: number;
  centerPoint?: [number, number];
  bounds?: {
    minX: number;
    maxX: number;
    minY: number;
    maxY: number;
  };
}

export interface ProcessedTrajectory {
  originalData: TrajectoryData;
  smoothedLines: Array<{
    start: [number, number];
    end: [number, number];
    confidence: number;
  }>;
  collisionPredictions: Array<{
    position: [number, number];
    ballId: string;
    probability: number;
    timeToCollision: number;
  }>;
  successProbability?: number;
  recommendation?: string;
}

export type AlertHandler = (alert: ProcessedAlert) => void;
export type FrameHandler = (frame: ProcessedFrameData) => void;
export type GameStateHandler = (state: ProcessedGameState) => void;
export type TrajectoryHandler = (trajectory: ProcessedTrajectory) => void;
export type ConfigHandler = (config: ConfigData) => void;
export type MetricsHandler = (metrics: MetricsData) => void;
export type StatusHandler = (status: StatusData) => void;

export interface ProcessedAlert {
  level: AlertData['level'];
  message: string;
  code: string;
  timestamp: Date;
  details: Record<string, any>;
  isActionable: boolean;
  autoCloseDelay?: number;
}

export class DataProcessingService {
  private options: Required<DataProcessingOptions>;
  private lastGameState: ProcessedGameState | null = null;
  private gameStateHistory: ProcessedGameState[] = [];
  private frameBuffer: Map<string, ProcessedFrameData> = new Map();
  private lastFrameTime = 0;

  // Event handlers
  private alertHandlers = new Set<AlertHandler>();
  private frameHandlers = new Set<FrameHandler>();
  private gameStateHandlers = new Set<GameStateHandler>();
  private trajectoryHandlers = new Set<TrajectoryHandler>();
  private configHandlers = new Set<ConfigHandler>();
  private metricsHandlers = new Set<MetricsHandler>();
  private statusHandlers = new Set<StatusHandler>();

  constructor(options: DataProcessingOptions = {}) {
    this.options = {
      frameProcessing: {
        maxFps: 30,
        qualityThreshold: 50,
        enableCompression: true,
        resizeToFit: true,
        targetWidth: 1920,
        targetHeight: 1080,
        ...options.frameProcessing,
      },
      stateProcessing: {
        smoothingFactor: 0.3,
        confidenceThreshold: 0.7,
        enablePrediction: true,
        maxHistoryLength: 100,
        ...options.stateProcessing,
      },
      trajectoryProcessing: {
        enableSmoothing: true,
        confidenceThreshold: 0.5,
        maxPredictionTime: 5.0,
        ...options.trajectoryProcessing,
      },
    };
  }

  // =============================================================================
  // Event Handler Management
  // =============================================================================

  onAlert(handler: AlertHandler): void {
    this.alertHandlers.add(handler);
  }

  onFrame(handler: FrameHandler): void {
    this.frameHandlers.add(handler);
  }

  onGameState(handler: GameStateHandler): void {
    this.gameStateHandlers.add(handler);
  }

  onTrajectory(handler: TrajectoryHandler): void {
    this.trajectoryHandlers.add(handler);
  }

  onConfig(handler: ConfigHandler): void {
    this.configHandlers.add(handler);
  }

  onMetrics(handler: MetricsHandler): void {
    this.metricsHandlers.add(handler);
  }

  onStatus(handler: StatusHandler): void {
    this.statusHandlers.add(handler);
  }

  offAlert(handler: AlertHandler): void {
    this.alertHandlers.delete(handler);
  }

  offFrame(handler: FrameHandler): void {
    this.frameHandlers.delete(handler);
  }

  offGameState(handler: GameStateHandler): void {
    this.gameStateHandlers.delete(handler);
  }

  offTrajectory(handler: TrajectoryHandler): void {
    this.trajectoryHandlers.delete(handler);
  }

  offConfig(handler: ConfigHandler): void {
    this.configHandlers.delete(handler);
  }

  offMetrics(handler: MetricsHandler): void {
    this.metricsHandlers.delete(handler);
  }

  offStatus(handler: StatusHandler): void {
    this.statusHandlers.delete(handler);
  }

  // =============================================================================
  // Message Processing
  // =============================================================================

  processMessage(message: WebSocketMessage): void {
    try {
      switch (message.type) {
        case 'frame':
          if (isFrameMessage(message)) {
            this.processFrame(message.data);
          }
          break;

        case 'state':
          if (isGameStateMessage(message)) {
            this.processGameState(message.data);
          }
          break;

        case 'trajectory':
          if (isTrajectoryMessage(message)) {
            this.processTrajectory(message.data);
          }
          break;

        case 'alert':
          if (isAlertMessage(message)) {
            this.processAlert(message.data);
          }
          break;

        case 'config':
          this.processConfig(message.data as ConfigData);
          break;

        case 'metrics':
          this.processMetrics(message.data as MetricsData);
          break;

        case 'status':
          this.processStatus(message.data as StatusData);
          break;

        default:
          console.debug('Unhandled message type:', message.type);
      }
    } catch (error) {
      console.error('Error processing message:', error, message);
    }
  }

  // =============================================================================
  // Frame Processing
  // =============================================================================

  private processFrame(frameData: FrameData): void {
    // Implement frame rate limiting
    const now = Date.now();
    const minInterval = 1000 / this.options.frameProcessing.maxFps;
    if (now - this.lastFrameTime < minInterval) {
      return; // Skip frame to maintain FPS limit
    }
    this.lastFrameTime = now;

    // Skip low-quality frames
    if (frameData.quality < this.options.frameProcessing.qualityThreshold) {
      return;
    }

    const processedFrame = this.createProcessedFrame(frameData);

    // Store in buffer (limited size)
    const frameId = `frame_${now}`;
    this.frameBuffer.set(frameId, processedFrame);

    // Cleanup old frames
    if (this.frameBuffer.size > 10) {
      const oldestKey = this.frameBuffer.keys().next().value;
      this.frameBuffer.delete(oldestKey);
    }

    // Notify handlers
    this.frameHandlers.forEach(handler => {
      try {
        handler(processedFrame);
      } catch (error) {
        console.error('Error in frame handler:', error);
      }
    });
  }

  private createProcessedFrame(frameData: FrameData): ProcessedFrameData {
    let displayWidth = frameData.width;
    let displayHeight = frameData.height;

    // Resize to fit target dimensions while maintaining aspect ratio
    if (this.options.frameProcessing.resizeToFit) {
      const aspectRatio = frameData.width / frameData.height;
      const targetAspectRatio = this.options.frameProcessing.targetWidth! / this.options.frameProcessing.targetHeight!;

      if (aspectRatio > targetAspectRatio) {
        displayWidth = this.options.frameProcessing.targetWidth!;
        displayHeight = Math.round(displayWidth / aspectRatio);
      } else {
        displayHeight = this.options.frameProcessing.targetHeight!;
        displayWidth = Math.round(displayHeight * aspectRatio);
      }
    }

    // Create data URL for display
    const imageUrl = `data:image/${frameData.format};base64,${frameData.image}`;

    return {
      imageUrl,
      originalData: frameData,
      processedAt: new Date(),
      displayWidth,
      displayHeight,
      aspectRatio: frameData.width / frameData.height,
      isCompressed: frameData.compressed,
    };
  }

  // =============================================================================
  // Game State Processing
  // =============================================================================

  private processGameState(stateData: GameStateData): void {
    const processedState = this.createProcessedGameState(stateData);

    // Add to history
    this.gameStateHistory.push(processedState);
    if (this.gameStateHistory.length > this.options.stateProcessing.maxHistoryLength) {
      this.gameStateHistory.shift();
    }

    this.lastGameState = processedState;

    // Notify handlers
    this.gameStateHandlers.forEach(handler => {
      try {
        handler(processedState);
      } catch (error) {
        console.error('Error in game state handler:', error);
      }
    });
  }

  private createProcessedGameState(stateData: GameStateData): ProcessedGameState {
    const previousState = this.lastGameState;
    const changesSinceLastFrame: string[] = [];

    // Process balls with smoothing and prediction
    const processedBalls = stateData.balls.map(ball => {
      const prevBall = previousState?.balls.find(b => b.id === ball.id);

      let interpolatedPosition = ball.position;
      let predictedPosition: [number, number] | undefined;
      let isMoving = false;
      let hasPositionChanged = false;
      let movementDirection: number | undefined;
      let speed: number | undefined;

      if (prevBall && this.options.stateProcessing.enablePrediction) {
        // Check if position changed
        const deltaX = ball.position[0] - prevBall.position[0];
        const deltaY = ball.position[1] - prevBall.position[1];
        const distance = Math.sqrt(deltaX * deltaX + deltaY * deltaY);

        hasPositionChanged = distance > 1; // 1 pixel threshold
        isMoving = distance > 2; // 2 pixel threshold for movement

        if (hasPositionChanged) {
          changesSinceLastFrame.push(`Ball ${ball.id} moved`);

          // Calculate movement direction and speed
          movementDirection = Math.atan2(deltaY, deltaX) * (180 / Math.PI);
          speed = distance; // pixels per frame

          // Apply smoothing
          const smoothing = this.options.stateProcessing.smoothingFactor;
          interpolatedPosition = [
            prevBall.position[0] + deltaX * smoothing,
            prevBall.position[1] + deltaY * smoothing,
          ] as [number, number];

          // Predict future position
          if (ball.velocity) {
            predictedPosition = [
              ball.position[0] + ball.velocity[0] * 2, // 2 frames ahead
              ball.position[1] + ball.velocity[1] * 2,
            ] as [number, number];
          }
        }
      }

      return {
        ...ball,
        interpolatedPosition,
        predictedPosition,
        movementDirection,
        speed,
        isMoving,
        hasPositionChanged,
      } as ProcessedBallData;
    });

    // Process cue data
    let processedCue: ProcessedCueData | undefined;
    if (stateData.cue) {
      processedCue = {
        ...stateData.cue,
        // Add trajectory prediction logic here
        predictedTrajectory: this.calculateCueTrajectory(stateData.cue, stateData.table),
        aimingAccuracy: this.calculateAimingAccuracy(stateData.cue, processedBalls),
      };
    }

    // Process table data
    let processedTable: ProcessedTableData | undefined;
    if (stateData.table) {
      processedTable = {
        ...stateData.table,
        centerPoint: this.calculateTableCenter(stateData.table),
        bounds: this.calculateTableBounds(stateData.table),
      };
    }

    // Calculate overall confidence
    const ballConfidences = processedBalls.map(b => b.confidence);
    const avgBallConfidence = ballConfidences.length > 0
      ? ballConfidences.reduce((a, b) => a + b, 0) / ballConfidences.length
      : 0;

    const cueConfidence = stateData.cue?.confidence || 1;
    const overallConfidence = (avgBallConfidence + cueConfidence) / 2;

    return {
      timestamp: new Date(),
      balls: processedBalls,
      cue: processedCue,
      table: processedTable,
      confidence: overallConfidence,
      isValid: overallConfidence >= this.options.stateProcessing.confidenceThreshold,
      frameNumber: stateData.frame_number,
      changesSinceLastFrame,
    };
  }

  private calculateCueTrajectory(cue: CueData, table?: TableData): [number, number][] {
    // Simplified trajectory calculation
    const points: [number, number][] = [];
    const angleRad = (cue.angle * Math.PI) / 180;
    const distance = cue.length || 100;

    for (let i = 0; i <= 10; i++) {
      const factor = (i / 10) * distance;
      const x = cue.position[0] + Math.cos(angleRad) * factor;
      const y = cue.position[1] + Math.sin(angleRad) * factor;
      points.push([x, y]);
    }

    return points;
  }

  private calculateAimingAccuracy(cue: CueData, balls: ProcessedBallData[]): number {
    // Simplified aiming accuracy calculation
    // This would need more sophisticated physics calculations
    const cueBall = balls.find(b => b.id === 'cue' || b.is_cue_ball);
    if (!cueBall) return 0;

    // Calculate if cue is aimed at any ball
    const angleRad = (cue.angle * Math.PI) / 180;
    const aimDirection = [Math.cos(angleRad), Math.sin(angleRad)];

    let bestAccuracy = 0;
    balls.forEach(ball => {
      if (ball.id === cueBall.id) return;

      const toBall = [
        ball.position[0] - cue.position[0],
        ball.position[1] - cue.position[1],
      ];
      const distance = Math.sqrt(toBall[0] * toBall[0] + toBall[1] * toBall[1]);
      const normalized = [toBall[0] / distance, toBall[1] / distance];

      const dotProduct = aimDirection[0] * normalized[0] + aimDirection[1] * normalized[1];
      const accuracy = Math.max(0, dotProduct);

      if (accuracy > bestAccuracy) {
        bestAccuracy = accuracy;
      }
    });

    return bestAccuracy;
  }

  private calculateTableCenter(table: TableData): [number, number] {
    const avgX = table.corners.reduce((sum, corner) => sum + corner[0], 0) / table.corners.length;
    const avgY = table.corners.reduce((sum, corner) => sum + corner[1], 0) / table.corners.length;
    return [avgX, avgY];
  }

  private calculateTableBounds(table: TableData): { minX: number; maxX: number; minY: number; maxY: number } {
    const xs = table.corners.map(corner => corner[0]);
    const ys = table.corners.map(corner => corner[1]);

    return {
      minX: Math.min(...xs),
      maxX: Math.max(...xs),
      minY: Math.min(...ys),
      maxY: Math.max(...ys),
    };
  }

  // =============================================================================
  // Trajectory Processing
  // =============================================================================

  private processTrajectory(trajectoryData: TrajectoryData): void {
    if (trajectoryData.confidence < this.options.trajectoryProcessing.confidenceThreshold) {
      return; // Skip low-confidence trajectories
    }

    const processedTrajectory = this.createProcessedTrajectory(trajectoryData);

    this.trajectoryHandlers.forEach(handler => {
      try {
        handler(processedTrajectory);
      } catch (error) {
        console.error('Error in trajectory handler:', error);
      }
    });
  }

  private createProcessedTrajectory(trajectoryData: TrajectoryData): ProcessedTrajectory {
    // Apply smoothing if enabled
    let smoothedLines = trajectoryData.lines.map(line => ({
      start: line.start,
      end: line.end,
      confidence: line.confidence,
    }));

    if (this.options.trajectoryProcessing.enableSmoothing) {
      smoothedLines = this.smoothTrajectoryLines(smoothedLines);
    }

    // Process collision predictions
    const collisionPredictions = trajectoryData.collisions.map(collision => ({
      position: collision.position,
      ballId: collision.ball_id,
      probability: Math.min(1, collision.angle / 90), // Simplified probability
      timeToCollision: collision.time_to_collision || 0,
    }));

    // Calculate success probability
    const successProbability = this.calculateSuccessProbability(trajectoryData);

    return {
      originalData: trajectoryData,
      smoothedLines,
      collisionPredictions,
      successProbability,
      recommendation: this.generateTrajectoryRecommendation(trajectoryData),
    };
  }

  private smoothTrajectoryLines(lines: Array<{ start: [number, number]; end: [number, number]; confidence: number }>): typeof lines {
    // Simple line smoothing - in practice, you'd use more sophisticated algorithms
    return lines.map((line, index) => {
      if (index === 0 || index === lines.length - 1) {
        return line; // Don't smooth first and last lines
      }

      const prev = lines[index - 1];
      const next = lines[index + 1];

      const smoothedStart: [number, number] = [
        (prev.end[0] + line.start[0] + next.start[0]) / 3,
        (prev.end[1] + line.start[1] + next.start[1]) / 3,
      ];

      const smoothedEnd: [number, number] = [
        (line.end[0] + next.start[0] + next.end[0]) / 3,
        (line.end[1] + next.start[1] + next.end[1]) / 3,
      ];

      return {
        ...line,
        start: smoothedStart,
        end: smoothedEnd,
      };
    });
  }

  private calculateSuccessProbability(trajectoryData: TrajectoryData): number {
    // Simplified success probability calculation
    let probability = trajectoryData.confidence;

    // Reduce probability for complex trajectories
    if (trajectoryData.line_count > 3) {
      probability *= 0.8;
    }

    // Reduce probability for many collisions
    if (trajectoryData.collision_count > 2) {
      probability *= 0.7;
    }

    return Math.max(0, Math.min(1, probability));
  }

  private generateTrajectoryRecommendation(trajectoryData: TrajectoryData): string {
    if (trajectoryData.confidence < 0.5) {
      return "Low confidence trajectory - consider adjusting aim";
    }

    if (trajectoryData.collision_count === 0) {
      return "Direct shot - good aim";
    }

    if (trajectoryData.collision_count === 1) {
      return "One collision expected - moderate difficulty";
    }

    return "Complex shot with multiple collisions - high difficulty";
  }

  // =============================================================================
  // Alert Processing
  // =============================================================================

  private processAlert(alertData: AlertData): void {
    const processedAlert: ProcessedAlert = {
      ...alertData,
      timestamp: new Date(),
      isActionable: this.isActionableAlert(alertData),
      autoCloseDelay: this.getAutoCloseDelay(alertData),
    };

    this.alertHandlers.forEach(handler => {
      try {
        handler(processedAlert);
      } catch (error) {
        console.error('Error in alert handler:', error);
      }
    });
  }

  private isActionableAlert(alert: AlertData): boolean {
    // Determine if alert requires user action
    const actionableCodes = ['HW_CAMERA_UNAVAILABLE', 'HW_PROJECTOR_UNAVAILABLE', 'HW_CALIBRATION_FAILED'];
    return actionableCodes.includes(alert.code);
  }

  private getAutoCloseDelay(alert: AlertData): number | undefined {
    switch (alert.level) {
      case 'info':
        return 5000; // 5 seconds
      case 'warning':
        return 10000; // 10 seconds
      case 'error':
      case 'critical':
        return undefined; // Don't auto-close
      default:
        return 5000;
    }
  }

  // =============================================================================
  // Other Message Processing
  // =============================================================================

  private processConfig(configData: ConfigData): void {
    this.configHandlers.forEach(handler => {
      try {
        handler(configData);
      } catch (error) {
        console.error('Error in config handler:', error);
      }
    });
  }

  private processMetrics(metricsData: MetricsData): void {
    this.metricsHandlers.forEach(handler => {
      try {
        handler(metricsData);
      } catch (error) {
        console.error('Error in metrics handler:', error);
      }
    });
  }

  private processStatus(statusData: StatusData): void {
    this.statusHandlers.forEach(handler => {
      try {
        handler(statusData);
      } catch (error) {
        console.error('Error in status handler:', error);
      }
    });
  }

  // =============================================================================
  // State Access Methods
  // =============================================================================

  getCurrentGameState(): ProcessedGameState | null {
    return this.lastGameState;
  }

  getGameStateHistory(): ProcessedGameState[] {
    return [...this.gameStateHistory];
  }

  getFrameBuffer(): Map<string, ProcessedFrameData> {
    return new Map(this.frameBuffer);
  }

  // =============================================================================
  // Configuration Updates
  // =============================================================================

  updateOptions(newOptions: Partial<DataProcessingOptions>): void {
    this.options = {
      frameProcessing: { ...this.options.frameProcessing, ...newOptions.frameProcessing },
      stateProcessing: { ...this.options.stateProcessing, ...newOptions.stateProcessing },
      trajectoryProcessing: { ...this.options.trajectoryProcessing, ...newOptions.trajectoryProcessing },
    };
  }

  // =============================================================================
  // Cleanup
  // =============================================================================

  destroy(): void {
    this.alertHandlers.clear();
    this.frameHandlers.clear();
    this.gameStateHandlers.clear();
    this.trajectoryHandlers.clear();
    this.configHandlers.clear();
    this.metricsHandlers.clear();
    this.statusHandlers.clear();
    this.frameBuffer.clear();
    this.gameStateHistory.length = 0;
    this.lastGameState = null;
  }
}

// =============================================================================
// Factory Function
// =============================================================================

export function createDataProcessingService(options?: DataProcessingOptions): DataProcessingService {
  return new DataProcessingService(options);
}
