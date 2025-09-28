// Core domain types for the billiards trainer application

export interface Point2D {
  x: number;
  y: number;
}

export interface Point3D extends Point2D {
  z: number;
}

export interface Rectangle {
  x: number;
  y: number;
  width: number;
  height: number;
}

// Ball types
export interface Ball {
  id: number;
  type: 'solid' | 'stripe' | 'cue' | 'eight';
  color: string;
  position: Point2D;
  velocity: Point2D;
  isVisible: boolean;
  isPocketed: boolean;
  confidence: number; // Detection confidence 0-1
}

export interface Table {
  width: number;
  height: number;
  pockets: Point2D[];
  rails: Rectangle[];
  playArea: Rectangle;
}

export interface Cue {
  position: Point2D;
  angle: number; // Radians
  power: number; // 0-1
  isVisible: boolean;
  tipPosition: Point2D;
}

// Vision and detection types
export interface CameraInfo {
  id: string;
  name: string;
  resolution: { width: number; height: number };
  fps: number;
  isConnected: boolean;
  isCalibrated: boolean;
}

export interface CalibrationData {
  tableCorners: Point2D[];
  homographyMatrix: number[][];
  distortionCoefficients: number[];
  isValid: boolean;
  timestamp: Date;
}

export interface DetectionFrame {
  timestamp: Date;
  frameNumber: number;
  balls: Ball[];
  cue: Cue | null;
  confidence: number;
  processingTimeMs: number;
}

// Game state types
export interface GameState {
  gameType: 'eightball' | 'nineball' | 'straight' | 'practice';
  currentPlayer: number;
  players: Player[];
  isActive: boolean;
  isPaused: boolean;
  startTime: Date | null;
  lastShotTime: Date | null;
  shotCount: number;
}

export interface Player {
  id: string;
  name: string;
  ballGroup: 'solid' | 'stripe' | null;
  score: number;
  isActive: boolean;
}

export interface Shot {
  id: string;
  timestamp: Date;
  playerId: string;
  cueBallPosition: Point2D;
  targetBall: number | null;
  contactPoint: Point2D | null;
  result: ShotResult;
  ballsBeforeShot: Ball[];
  ballsAfterShot: Ball[];
}

export interface ShotResult {
  ballsPocketed: number[];
  foulCommitted: boolean;
  foulType: string | null;
  isSuccessful: boolean;
  points: number;
}

// System and connection types
export interface SystemStatus {
  isConnected: boolean;
  lastHeartbeat: Date | null;
  backendVersion: string | null;
  websocketStatus: 'connected' | 'connecting' | 'disconnected' | 'error';
  errors: SystemError[];
}

export interface SystemError {
  id: string;
  timestamp: Date;
  level: 'info' | 'warning' | 'error' | 'critical';
  message: string;
  component: string;
  details?: any;
}

// Configuration types
export interface AppConfig {
  camera: CameraConfig;
  detection: DetectionConfig;
  game: GameConfig;
  ui: UIConfig;
}

export interface CameraConfig {
  selectedCameraId: string | null;
  resolution: { width: number; height: number };
  fps: number;
  autoExposure: boolean;
  exposure: number;
  brightness: number;
  contrast: number;
}

export interface DetectionConfig {
  ballDetectionThreshold: number;
  cueDetectionThreshold: number;
  motionDetectionThreshold: number;
  stabilizationFrames: number;
  enableTracking: boolean;
  enablePrediction: boolean;
}

export interface GameConfig {
  defaultGameType: GameState['gameType'];
  autoStartGames: boolean;
  enableFoulDetection: boolean;
  shotTimeout: number; // seconds
  enableShotHistory: boolean;
  maxHistorySize: number;
}

export interface UIConfig {
  theme: 'light' | 'dark' | 'auto';
  showDebugInfo: boolean;
  enableNotifications: boolean;
  animationSpeed: 'slow' | 'normal' | 'fast';
  language: string;
}

// Authentication types
export interface User {
  id: string;
  username: string;
  email: string;
  roles: string[];
  preferences: UserPreferences;
  lastLogin: Date;
}

export interface UserPreferences {
  theme: UIConfig['theme'];
  language: string;
  notifications: {
    gameEvents: boolean;
    systemAlerts: boolean;
    achievements: boolean;
  };
}

export interface AuthState {
  isAuthenticated: boolean;
  user: User | null;
  token: string | null;
  refreshToken: string | null;
  expiresAt: Date | null;
}

// UI state types
export interface UIState {
  modals: {
    calibration: boolean;
    settings: boolean;
    gameSetup: boolean;
    shotHistory: boolean;
    help: boolean;
  };
  notifications: Notification[];
  loading: {
    global: boolean;
    calibration: boolean;
    detection: boolean;
    gameStart: boolean;
  };
  activeTab: string;
  sidebarOpen: boolean;
}

export interface Notification {
  id: string;
  type: 'info' | 'success' | 'warning' | 'error';
  title: string;
  message: string;
  timestamp: Date;
  isRead: boolean;
  autoHide: boolean;
  duration?: number; // ms
}

// WebSocket message types
export interface WebSocketMessage {
  type: string;
  timestamp: Date;
  data: any;
}

export interface GameUpdateMessage extends WebSocketMessage {
  type: 'game_update';
  data: {
    balls: Ball[];
    cue: Cue | null;
    gameState: Partial<GameState>;
  };
}

export interface SystemUpdateMessage extends WebSocketMessage {
  type: 'system_update';
  data: {
    status: Partial<SystemStatus>;
    errors?: SystemError[];
  };
}

// Store persistence types
export interface PersistedState {
  config: AppConfig;
  auth: Pick<AuthState, 'token' | 'refreshToken' | 'expiresAt'>;
  ui: Pick<UIState, 'activeTab' | 'sidebarOpen'>;
  version: string;
}

// Action result types
export interface ActionResult<T = any> {
  success: boolean;
  data?: T;
  error?: string;
  timestamp: Date;
}

// API response types
export interface ApiResponse<T = any> {
  success: boolean;
  data: T;
  message?: string;
  error?: string;
  timestamp: string;
}

export interface PaginatedResponse<T> extends ApiResponse<T[]> {
  pagination: {
    page: number;
    limit: number;
    total: number;
    pages: number;
  };
}
