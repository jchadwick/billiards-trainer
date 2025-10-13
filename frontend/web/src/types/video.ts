/**
 * Video streaming and overlay types for the billiards trainer frontend
 */

export interface Point2D {
  x: number;
  y: number;
}

export interface Vector2D {
  x: number;
  y: number;
}

export interface Size2D {
  width: number;
  height: number;
}

export interface BoundingBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

// Video streaming types
export interface VideoStreamConfig {
  quality: VideoQuality;
  fps: number;
  resolution?: Size2D;
  autoReconnect: boolean;
  reconnectDelay: number;
}

export type VideoQuality = 'low' | 'medium' | 'high' | 'ultra';

export interface VideoStreamStatus {
  connected: boolean;
  streaming: boolean;
  fps: number;
  quality: VideoQuality;
  latency: number;
  errors: number;
  lastFrameTime: number;
}

// Detection data types
export interface Ball {
  id: string;
  position: Point2D;
  radius: number;
  type: 'cue' | 'solid' | 'stripe' | 'eight';
  number?: number;
  velocity: Vector2D;
  confidence: number;
  color?: string;
}

export interface CueStick {
  tipPosition: Point2D;
  tailPosition: Point2D;
  angle: number;
  elevation: number;
  detected: boolean;
  confidence: number;
  length: number;
}

export interface Table {
  corners: Point2D[];
  pockets: Point2D[];
  bounds: BoundingBox;
  rails: Point2D[][];
  detected: boolean;
  confidence: number;
}

export interface Trajectory {
  ballId: string;
  points: Point2D[];
  collisions: CollisionPoint[];
  type: 'primary' | 'reflection' | 'collision';
  probability: number;
  color: string;
}

export interface CollisionPoint {
  position: Point2D;
  type: 'ball' | 'rail' | 'pocket';
  targetId?: string;
  angle: number;
  impulse: number;
}

export interface DetectionFrame {
  balls: Ball[];
  cue: CueStick | null;
  table: Table | null;
  trajectories: Trajectory[];
  timestamp: number;
  frameNumber: number;
  processingTime: number;
}

// Overlay types
export interface OverlayConfig {
  balls: {
    visible: boolean;
    showLabels: boolean;
    showIds: boolean;
    showVelocity: boolean;
    showConfidence: boolean;
    radius: number;
    opacity: number;
  };
  trajectories: {
    visible: boolean;
    showProbability: boolean;
    lineWidth: number;
    opacity: number;
    maxLength: number;
  };
  table: {
    visible: boolean;
    showPockets: boolean;
    showRails: boolean;
    lineWidth: number;
    opacity: number;
  };
  cue: {
    visible: boolean;
    showAngle: boolean;
    showGuideLines: boolean;
    lineWidth: number;
    opacity: number;
  };
  grid: {
    visible: boolean;
    spacing: number;
    opacity: number;
  };
  calibration: {
    visible: boolean;
    showPoints: boolean;
    showLines: boolean;
    showLabels: boolean;
    showAccuracy: boolean;
    lineWidth: number;
    opacity: number;
  };
}

// Interaction types
export interface ViewportTransform {
  x: number;
  y: number;
  scale: number;
  rotation: number;
}

export interface InteractionState {
  isDragging: boolean;
  isZooming: boolean;
  lastPointerPosition: Point2D | null;
  transform: ViewportTransform;
  bounds: BoundingBox;
}

// Stream controls
export interface StreamControls {
  play: () => void;
  pause: () => void;
  stop: () => void;
  setQuality: (quality: VideoQuality) => void;
  setFPS: (fps: number) => void;
  toggleFullscreen: () => void;
  takeScreenshot: () => Promise<Blob>;
}

// Performance metrics
export interface PerformanceMetrics {
  renderFPS: number;
  droppedFrames: number;
  avgRenderTime: number;
  memoryUsage: number;
  cpuUsage: number;
}

// Error types
export interface VideoError {
  code: string;
  message: string;
  timestamp: number;
  recoverable: boolean;
}

// Events
export interface VideoEvents {
  'stream-connected': () => void;
  'stream-disconnected': () => void;
  'stream-error': (error: VideoError) => void;
  'frame-received': (frame: DetectionFrame) => void;
  'quality-changed': (quality: VideoQuality) => void;
  'fullscreen-changed': (isFullscreen: boolean) => void;
  'screenshot-taken': (blob: Blob) => void;
}

// Canvas rendering context
export interface CanvasContext {
  canvas: HTMLCanvasElement;
  ctx: CanvasRenderingContext2D;
  size: Size2D;
  devicePixelRatio: number;
}

// Coordinate transformation
export interface CoordinateTransform {
  videoToCanvas: (point: Point2D) => Point2D;
  canvasToVideo: (point: Point2D) => Point2D;
  screenToCanvas: (point: Point2D) => Point2D;
  canvasToScreen: (point: Point2D) => Point2D;
}

// Calibration data types
export interface CalibrationPoint {
  id: string;
  screenPosition: Point2D;
  worldPosition: Point2D;
  timestamp: number;
  confidence?: number;
}

export interface CalibrationData {
  corners: CalibrationPoint[];
  transformationMatrix?: number[][];
  calibratedAt: number;
  accuracy?: number;
  isValid: boolean;
  tableType?: string;
  dimensions?: {
    width: number;
    height: number;
  };
}
