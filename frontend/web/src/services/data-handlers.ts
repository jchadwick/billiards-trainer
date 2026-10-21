/**
 * Real-time data handlers for game state, vision data, and other WebSocket streams
 */

import {
  isAlertMessage,
  isFrameMessage,
  isGameStateMessage,
  isTrajectoryMessage,
  type AlertData,
  type BallData,
  type ConfigData,
  type CueData,
  type FrameData,
  type GameStateData,
  type MetricsData,
  type PositionWithScale,
  type StatusData,
  type TableData,
  type TablePocketData,
  type TrajectoryData,
  type WebSocketMessage,
} from "../types/api";
import {
  normalizeImageExtentToCanonical,
  normalizeImageLengthToCanonical,
  normalizeImagePointToCanonical,
  normalizeTablePixelExtentsToCanonical,
} from "../utils/coordinates";

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

type ResolvedDataProcessingOptions = {
  frameProcessing: Required<NonNullable<DataProcessingOptions["frameProcessing"]>>;
  stateProcessing: Required<NonNullable<DataProcessingOptions["stateProcessing"]>>;
  trajectoryProcessing: Required<NonNullable<DataProcessingOptions["trajectoryProcessing"]>>;
};

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

// Processed positions are always canonical 4K pixels. Their scale is therefore
// identity; source-to-canonical scaling happens once at the input boundary.
export interface Position2D {
  x: number;
  y: number;
  scale: [1, 1];
}

export interface ProcessedBallData extends BallData {
  interpolatedPosition?: Position2D;
  predictedPosition?: Position2D;
  movementDirection?: number; // angle in degrees
  speed?: number;
  radiusX?: number;
  radiusY?: number;
  isMoving: boolean;
  hasPositionChanged: boolean;
}

export interface ProcessedCueData extends CueData {
  predictedTrajectory?: Position2D[];
  aimingAccuracy?: number;
  recommendedAdjustment?: {
    angle: number;
    force: number;
  };
}

export interface ProcessedTableData extends TableData {
  scaleFactor?: number;
  centerPoint?: Position2D;
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
    start: Position2D;
    end: Position2D;
    confidence: number;
  }>;
  collisionPredictions: Array<{
    position: Position2D;
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
  level: AlertData["level"];
  message: string;
  code: string;
  timestamp: Date;
  details: Record<string, any>;
  isActionable: boolean;
  autoCloseDelay?: number;
}

// =============================================================================
// Helper Functions
// =============================================================================

/**
 * Convert a transport point into the frontend's canonical image plane. Array
 * compatibility is intentionally omitted here because state messages do not
 * carry a frame resolution from which an unscaled legacy point could be read.
 */
function toPosition2D(pos: PositionWithScale | Position2D): Position2D {
  const normalized = normalizeImagePointToCanonical(pos);
  if (!normalized) {
    throw new RangeError("Image point is missing finite coordinates or valid scale metadata");
  }
  return { ...normalized, scale: [1, 1] };
}

function physicalDimensions(
  dimensions: unknown,
): TableData["dimensions"] | undefined {
  if (!dimensions || typeof dimensions !== "object") return undefined;
  const value = dimensions as Record<string, unknown>;
  if (
    value.unit !== "meters" ||
    typeof value.width !== "number" || !Number.isFinite(value.width) || value.width <= 0 ||
    typeof value.height !== "number" || !Number.isFinite(value.height) || value.height <= 0
  ) return undefined;
  return { width: value.width, height: value.height, unit: "meters" };
}

function normalizeTableData(table: TableData): TableData {
  const sourcePockets: Array<PositionWithScale | TablePocketData> = table.pockets?.length
    ? table.pockets
    : (table.pocket_positions ?? []).map((position) => ({
      position,
      radius: table.pocket_radius,
      radius_scale: table.pocket_radius_scale,
      radius_unit: table.pocket_radius_unit,
    }));
  const pockets = (sourcePockets ?? []).map((pocket) => {
    if (pocket && typeof pocket === "object" && "position" in pocket) {
      const position = toPosition2D(pocket.position);
      const radiusUnit = pocket.radius_unit ?? table.pocket_radius_unit;
      const supportsRadiusUnit = radiusUnit === undefined || radiusUnit === "canonical_pixels";
      const radiusScale = pocket.radius_scale ?? (
        radiusUnit === "canonical_pixels"
          ? table.pocket_radius_scale ?? [1, 1]
          : pocket.position.scale
      );
      const radius = pocket.radius === undefined || !supportsRadiusUnit
        ? undefined
        : normalizeImageExtentToCanonical(pocket.radius, radiusScale);
      return {
        ...pocket,
        position,
        radius: radius ? Math.sqrt(radius.x * radius.y) : undefined,
        radius_unit: radius ? "canonical_pixels" as const : undefined,
        radius_scale: radius ? [1, 1] as [number, number] : undefined,
        radius_extent: radius ? { x: radius.x, y: radius.y, scale: [1, 1] as [1, 1] } : undefined,
      };
    }
    return toPosition2D(pocket as PositionWithScale);
  });
  const corners = (table.playing_area_corners ?? table.corners ?? [])
    .map((corner) => toPosition2D(corner));

  const rails = table.rails?.map((rail) => {
    if (Array.isArray(rail)) return rail.map((point) => toPosition2D(point as PositionWithScale));
    const segment = rail as Record<string, unknown>;
    if (segment.start && segment.end) {
      return {
        ...segment,
        start: toPosition2D(segment.start as PositionWithScale),
        end: toPosition2D(segment.end as PositionWithScale),
      };
    }
    throw new RangeError("Table rail is missing scaled endpoints");
  });

  const pixelExtents = normalizeTablePixelExtentsToCanonical(table);
  const dimensions = physicalDimensions(table.real_world_dimensions) ??
    physicalDimensions(table.dimensions);
  const pocketRadiusUnit = table.pocket_radius_unit;
  const supportsPocketRadiusUnit =
    pocketRadiusUnit === undefined || pocketRadiusUnit === "canonical_pixels";
  const firstPocket = sourcePockets[0];
  const firstPocketPosition = firstPocket && typeof firstPocket === "object" && "position" in firstPocket
    ? firstPocket.position
    : firstPocket as PositionWithScale | undefined;
  const pocketRadiusScale = table.pocket_radius_scale ?? (
    pocketRadiusUnit === "canonical_pixels"
      ? [1, 1]
      : firstPocketPosition?.scale
  );
  const pocketRadius = table.pocket_radius === undefined || !supportsPocketRadiusUnit
    ? undefined
    : normalizeImageExtentToCanonical(table.pocket_radius, pocketRadiusScale);

  return {
    ...table,
    corners,
    playing_area_corners: corners,
    pockets,
    rails,
    width: pixelExtents.width,
    width_scale: pixelExtents.width === undefined ? undefined : [1, 1],
    height: pixelExtents.height,
    height_scale: pixelExtents.height === undefined ? undefined : [1, 1],
    dimensions_scale: pixelExtents.width === undefined && pixelExtents.height === undefined
      ? undefined
      : [1, 1],
    dimensions_unit: pixelExtents.width === undefined && pixelExtents.height === undefined
      ? undefined
      : "canonical_pixels",
    dimensions,
    real_world_dimensions: dimensions,
    pocket_radius: pocketRadius ? Math.sqrt(pocketRadius.x * pocketRadius.y) : undefined,
    pocket_radius_scale: pocketRadius ? [1, 1] : undefined,
    pocket_radius_unit: pocketRadius ? "canonical_pixels" : undefined,
  };
}

function normalizeCueData(cue: CueData): CueData {
  const primaryPosition = cue.tip_position ?? cue.position;
  if (!primaryPosition) throw new RangeError("Cue is missing scaled image position");
  const position = toPosition2D(primaryPosition);
  const tip = cue.tip_position ? toPosition2D(cue.tip_position) : position;
  const butt = cue.butt_position
    ? toPosition2D(cue.butt_position)
    : cue.position && cue.tip_position
      ? toPosition2D(cue.position)
      : undefined;
  const angleRadians = cue.angle * Math.PI / 180;
  const inputScale = cue.length_scale ?? primaryPosition.scale;
  const normalizedLength = cue.length === undefined
    ? undefined
    : normalizeImageLengthToCanonical(cue.length, inputScale, angleRadians);
  if (cue.length !== undefined && normalizedLength === null) {
    throw new RangeError("Cue length is invalid or missing scale metadata");
  }
  const endpointLength = butt
    ? Math.hypot(tip.x - butt.x, tip.y - butt.y)
    : undefined;
  const length = endpointLength ?? normalizedLength ?? undefined;

  return {
    ...cue,
    position,
    tip_position: tip,
    butt_position: butt,
    length,
    length_scale: length === undefined ? undefined : [1, 1],
  };
}

export class DataProcessingService {
  private options: ResolvedDataProcessingOptions;
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
        case "frame":
          if (isFrameMessage(message)) {
            this.processFrame(message.data);
          }
          break;

        case "state":
          if (isGameStateMessage(message)) {
            this.processGameState(message.data);
          }
          break;

        case "trajectory":
          if (isTrajectoryMessage(message)) {
            this.processTrajectory(message.data);
          }
          break;

        case "alert":
          if (isAlertMessage(message)) {
            this.processAlert(message.data);
          }
          break;

        case "config":
          this.processConfig(message.data as ConfigData);
          break;

        case "metrics":
          this.processMetrics(message.data as MetricsData);
          break;

        case "status":
          this.processStatus(message.data as StatusData);
          break;

        default:
          console.debug("Unhandled message type:", message.type);
      }
    } catch (error) {
      console.error("Error processing message:", error, message);
    }
  }

  // =============================================================================
  // Frame Processing
  // =============================================================================

  private processFrame(frameData: FrameData): void {
    if(!this.options.frameProcessing) return;

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
      if (oldestKey !== undefined) this.frameBuffer.delete(oldestKey);
    }

    // Notify handlers
    this.frameHandlers.forEach((handler) => {
      try {
        handler(processedFrame);
      } catch (error) {
        console.error("Error in frame handler:", error);
      }
    });
  }

  private createProcessedFrame(frameData: FrameData): ProcessedFrameData {
    let displayWidth = frameData.width;
    let displayHeight = frameData.height;

    // Resize to fit target dimensions while maintaining aspect ratio
    if (this.options.frameProcessing.resizeToFit) {
      const aspectRatio = frameData.width / frameData.height;
      const targetAspectRatio =
        this.options.frameProcessing.targetWidth! /
        this.options.frameProcessing.targetHeight!;

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
    if (
      this.gameStateHistory.length >
      this.options.stateProcessing.maxHistoryLength
    ) {
      this.gameStateHistory.shift();
    }

    this.lastGameState = processedState;

    // Notify handlers
    this.gameStateHandlers.forEach((handler) => {
      try {
        handler(processedState);
      } catch (error) {
        console.error("Error in game state handler:", error);
      }
    });
  }

  private createProcessedGameState(
    stateData: GameStateData
  ): ProcessedGameState {
    const previousState = this.lastGameState;
    const changesSinceLastFrame: string[] = [];
    const normalizedTable = stateData.table
      ? normalizeTableData(stateData.table)
      : undefined;
    const normalizedCue = stateData.cue
      ? normalizeCueData(stateData.cue)
      : undefined;

    // Process balls with smoothing and prediction
    const processedBalls = stateData.balls.map((ball) => {
      if (ball.radius_unit !== undefined && ball.radius_unit !== "canonical_pixels") {
        return null;
      }
      const prevBall = previousState?.balls.find((b) => b.id === ball.id);

      // Convert all source pixel geometry once, before comparisons or derived
      // velocities. Previous processed state is already canonical `[1, 1]`.
      const currentPos = toPosition2D(ball.position);
      const velocity = ball.velocity
        ? toPosition2D(ball.velocity)
        : { x: 0, y: 0, scale: [1, 1] as [1, 1] };
      const radiusScale = ball.radius_scale ?? (
        ball.radius_unit === "canonical_pixels" ? [1, 1] : ball.position.scale
      );
      const radius = ball.radius_extent
        ? normalizeImagePointToCanonical(ball.radius_extent)
        : normalizeImageExtentToCanonical(ball.radius, radiusScale);
      if (!radius) return null;
      let interpolatedPosition: Position2D | undefined;
      let predictedPosition: Position2D | undefined;
      let isMoving = false;
      let hasPositionChanged = false;
      let movementDirection: number | undefined;
      let speed: number | undefined;

      if (prevBall && this.options.stateProcessing.enablePrediction) {
        // Check if position changed
        const prevPos = toPosition2D(prevBall.position);
        const deltaX = currentPos.x - prevPos.x;
        const deltaY = currentPos.y - prevPos.y;
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
          interpolatedPosition = {
            x: prevPos.x + deltaX * smoothing,
            y: prevPos.y + deltaY * smoothing,
            scale: currentPos.scale, // Preserve scale metadata
          };

          // Predict future position
          predictedPosition = {
            x: currentPos.x + velocity.x * 2, // 2 frames ahead
            y: currentPos.y + velocity.y * 2,
            scale: [1, 1],
          };
        }
      }

      return {
        ...ball,
        position: currentPos,
        velocity,
        radius: Math.sqrt(radius.x * radius.y),
        radius_scale: [1, 1],
        radius_extent: { x: radius.x, y: radius.y, scale: [1, 1] },
        radiusX: radius.x,
        radiusY: radius.y,
        interpolatedPosition,
        predictedPosition,
        movementDirection,
        speed,
        isMoving,
        hasPositionChanged,
      } as ProcessedBallData;
    }).filter((ball): ball is ProcessedBallData => ball !== null);

    // Process cue data
    let processedCue: ProcessedCueData | undefined;
    if (normalizedCue) {
      processedCue = {
        ...normalizedCue,
        // Add trajectory prediction logic here
        predictedTrajectory: this.calculateCueTrajectory(
          normalizedCue,
          normalizedTable
        ),
        aimingAccuracy: this.calculateAimingAccuracy(
          normalizedCue,
          processedBalls
        ),
      };
    }

    // Process table data
    let processedTable: ProcessedTableData | undefined;
    if (normalizedTable) {
      const centerPoint = this.calculateTableCenter(normalizedTable);
      const bounds = this.calculateTableBounds(normalizedTable);
      processedTable = {
        ...normalizedTable,
        ...(centerPoint ? { centerPoint } : {}),
        ...(bounds ? { bounds } : {}),
      };
    }

    // Calculate overall confidence
    const ballConfidences = processedBalls.map((b) => b.confidence);
    const avgBallConfidence =
      ballConfidences.length > 0
        ? ballConfidences.reduce((a, b) => a + b, 0) / ballConfidences.length
        : 0;

    const cueConfidence = normalizedCue?.confidence || 1;
    const overallConfidence = (avgBallConfidence + cueConfidence) / 2;

    return {
      timestamp: new Date(),
      balls: processedBalls,
      cue: processedCue,
      table: processedTable,
      confidence: overallConfidence,
      isValid:
        overallConfidence >= this.options.stateProcessing.confidenceThreshold,
      frameNumber: stateData.frame_number,
      changesSinceLastFrame,
    };
  }

  private calculateCueTrajectory(
    cue: CueData,
    table?: TableData
  ): Position2D[] {
    // Simplified trajectory calculation
    const points: Position2D[] = [];
    const angleRad = (cue.angle * Math.PI) / 180;
    const distance = cue.length || 100;
    const anchor = cue.tip_position ?? cue.position;
    if (!anchor) throw new RangeError("Cue trajectory requires a scaled anchor point");
    const cuePos = toPosition2D(anchor);

    for (let i = 0; i <= 10; i++) {
      const factor = (i / 10) * distance;
      const x = cuePos.x + Math.cos(angleRad) * factor;
      const y = cuePos.y + Math.sin(angleRad) * factor;
      points.push({ x, y, scale: [1, 1] });
    }

    return points;
  }

  private calculateAimingAccuracy(
    cue: CueData,
    balls: ProcessedBallData[]
  ): number {
    // Simplified aiming accuracy calculation
    // This would need more sophisticated physics calculations
    const cueBall = balls.find((b) => b.id === "cue" || b.id === "0");
    if (!cueBall) return 0;

    // Calculate if cue is aimed at any ball
    const angleRad = (cue.angle * Math.PI) / 180;
    const aimDirection = { x: Math.cos(angleRad), y: Math.sin(angleRad) };
    const anchor = cue.tip_position ?? cue.position;
    if (!anchor) throw new RangeError("Aiming accuracy requires a scaled cue point");
    const cuePos = toPosition2D(anchor);

    let bestAccuracy = 0;
    balls.forEach((ball) => {
      if (ball.id === cueBall.id) return;

      const ballPos = toPosition2D(ball.position);
      const toBall = {
        x: ballPos.x - cuePos.x,
        y: ballPos.y - cuePos.y,
      };
      const distance = Math.sqrt(toBall.x * toBall.x + toBall.y * toBall.y);
      if (distance === 0) return;
      const normalized = { x: toBall.x / distance, y: toBall.y / distance };

      const dotProduct =
        aimDirection.x * normalized.x + aimDirection.y * normalized.y;
      const accuracy = Math.max(0, dotProduct);

      if (accuracy > bestAccuracy) {
        bestAccuracy = accuracy;
      }
    });

    return bestAccuracy;
  }

  private calculateTableCenter(table: TableData): Position2D | undefined {
    // Convert corners to Position2D and calculate center
    const corners = (table.playing_area_corners ?? table.corners ?? []).map(toPosition2D);
    if (corners.length < 4) return undefined;
    const avgX = corners.reduce((sum, corner) => sum + corner.x, 0) / corners.length;
    const avgY = corners.reduce((sum, corner) => sum + corner.y, 0) / corners.length;

    return { x: avgX, y: avgY, scale: [1, 1] };
  }

  private calculateTableBounds(table: TableData): {
    minX: number;
    maxX: number;
    minY: number;
    maxY: number;
  } | undefined {
    // Convert corners to Position2D and calculate bounds
    const corners = (table.playing_area_corners ?? table.corners ?? []).map(toPosition2D);
    if (corners.length < 4) return undefined;
    const xs = corners.map((corner) => corner.x);
    const ys = corners.map((corner) => corner.y);

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
    if (
      trajectoryData.confidence <
      this.options.trajectoryProcessing.confidenceThreshold
    ) {
      return; // Skip low-confidence trajectories
    }

    const processedTrajectory = this.createProcessedTrajectory(trajectoryData);

    this.trajectoryHandlers.forEach((handler) => {
      try {
        handler(processedTrajectory);
      } catch (error) {
        console.error("Error in trajectory handler:", error);
      }
    });
  }

  private createProcessedTrajectory(
    trajectoryData: TrajectoryData
  ): ProcessedTrajectory {
    // Apply smoothing if enabled
    // Convert trajectory lines to Position2D format
    let smoothedLines = trajectoryData.lines.map((line) => ({
      start: toPosition2D(line.start),
      end: toPosition2D(line.end),
      confidence: line.confidence,
    }));

    if (this.options.trajectoryProcessing.enableSmoothing) {
      smoothedLines = this.smoothTrajectoryLines(smoothedLines);
    }

    // Process collision predictions with Position2D format
    const collisionPredictions = trajectoryData.collisions.map((collision) => ({
      position: toPosition2D(collision.position),
      ballId: collision.ball2_id || '',
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

  private smoothTrajectoryLines(
    lines: Array<{
      start: Position2D;
      end: Position2D;
      confidence: number;
    }>
  ): typeof lines {
    // Simple line smoothing - in practice, you'd use more sophisticated algorithms
    return lines.map((line, index) => {
      if (index === 0 || index === lines.length - 1) {
        return line; // Don't smooth first and last lines
      }

      const prev = lines[index - 1];
      const next = lines[index + 1];

      const smoothedStart: Position2D = {
        x: (prev.end.x + line.start.x + next.start.x) / 3,
        y: (prev.end.y + line.start.y + next.start.y) / 3,
        scale: line.start.scale, // Preserve scale metadata
      };

      const smoothedEnd: Position2D = {
        x: (line.end.x + next.start.x + next.end.x) / 3,
        y: (line.end.y + next.start.y + next.end.y) / 3,
        scale: line.end.scale, // Preserve scale metadata
      };

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

  private generateTrajectoryRecommendation(
    trajectoryData: TrajectoryData
  ): string {
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

    this.alertHandlers.forEach((handler) => {
      try {
        handler(processedAlert);
      } catch (error) {
        console.error("Error in alert handler:", error);
      }
    });
  }

  private isActionableAlert(alert: AlertData): boolean {
    // Determine if alert requires user action
    const actionableCodes = [
      "HW_CAMERA_UNAVAILABLE",
      "HW_PROJECTOR_UNAVAILABLE",
      "HW_CALIBRATION_FAILED",
    ];
    return actionableCodes.includes(alert.code);
  }

  private getAutoCloseDelay(alert: AlertData): number | undefined {
    switch (alert.level) {
      case "info":
        return 5000; // 5 seconds
      case "warning":
        return 10000; // 10 seconds
      case "error":
      case "critical":
        return undefined; // Don't auto-close
      default:
        return 5000;
    }
  }

  // =============================================================================
  // Other Message Processing
  // =============================================================================

  private processConfig(configData: ConfigData): void {
    this.configHandlers.forEach((handler) => {
      try {
        handler(configData);
      } catch (error) {
        console.error("Error in config handler:", error);
      }
    });
  }

  private processMetrics(metricsData: MetricsData): void {
    this.metricsHandlers.forEach((handler) => {
      try {
        handler(metricsData);
      } catch (error) {
        console.error("Error in metrics handler:", error);
      }
    });
  }

  private processStatus(statusData: StatusData): void {
    this.statusHandlers.forEach((handler) => {
      try {
        handler(statusData);
      } catch (error) {
        console.error("Error in status handler:", error);
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
      frameProcessing: {
        ...this.options.frameProcessing,
        ...newOptions.frameProcessing,
      },
      stateProcessing: {
        ...this.options.stateProcessing,
        ...newOptions.stateProcessing,
      },
      trajectoryProcessing: {
        ...this.options.trajectoryProcessing,
        ...newOptions.trajectoryProcessing,
      },
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

export function createDataProcessingService(
  options?: DataProcessingOptions
): DataProcessingService {
  return new DataProcessingService(options);
}
