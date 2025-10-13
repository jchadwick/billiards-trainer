/**
 * TypeScript interfaces for Billiards Trainer API
 * Generated from backend Pydantic models
 */

// =============================================================================
// Base Types and Enums
// =============================================================================

export type HealthStatus = 'healthy' | 'degraded' | 'unhealthy';
export type StatusCode = 'success' | 'partial_success' | 'failure' | 'pending' | 'timeout' | 'cancelled';
export type LogLevel = 'CRITICAL' | 'ERROR' | 'WARNING' | 'INFO' | 'DEBUG';

export interface BaseResponse {
  [key: string]: any;
}

export interface ErrorResponse extends BaseResponse {
  error: string;
  message: string;
  code: string;
  details?: Record<string, any>;
  timestamp: string;
  request_id?: string;
}

export interface SuccessResponse extends BaseResponse {
  success: boolean;
  message: string;
  timestamp: string;
  data?: Record<string, any>;
}

// =============================================================================
// WebSocket Message Types
// =============================================================================

export type MessageType =
  | 'frame'
  | 'state'
  | 'trajectory'
  | 'alert'
  | 'config'
  | 'metrics'
  | 'connection'
  | 'ping'
  | 'pong'
  | 'subscribe'
  | 'unsubscribe'
  | 'subscribed'
  | 'unsubscribed'
  | 'status'
  | 'error';

export type AlertLevel = 'info' | 'warning' | 'error' | 'critical';
export type QualityLevel = 'low' | 'medium' | 'high' | 'auto';

export interface WebSocketMessage {
  type: MessageType;
  timestamp: string;
  sequence?: number;
  data: Record<string, any>;
}

// Frame Data
export interface FrameData {
  image: string; // Base64 encoded image
  width: number;
  height: number;
  format: string;
  quality: number;
  compressed: boolean;
  fps: number;
  size_bytes: number;
}

// Ball and Game State Data
export interface BallData {
  id: string;
  position: [number, number];
  radius: number;
  color: string;
  velocity?: [number, number];
  confidence: number;
  visible: boolean;
}

export interface CueData {
  angle: number;
  position: [number, number];
  detected: boolean;
  confidence: number;
  length?: number;
  tip_position?: [number, number];
}

export interface TableData {
  corners: [number, number][];
  pockets: [number, number][];
  rails?: Record<string, any>[];
  calibrated: boolean;
  dimensions?: Record<string, number>;
}

export interface GameStateData {
  balls: BallData[];
  cue?: CueData;
  table?: TableData;
  ball_count: number;
  frame_number?: number;
}

// Trajectory Data
export interface TrajectoryLine {
  start: [number, number];
  end: [number, number];
  type: 'primary' | 'reflection' | 'collision';
  confidence: number;
}

export interface CollisionData {
  position: [number, number];
  ball_id?: string;  // Target ball ID (for backwards compatibility)
  ball1_id?: string;  // Moving ball ID (e.g., cue ball)
  ball2_id?: string;  // Target ball ID (ball being hit, None for cushion/pocket)
  type?: string;  // Collision type (ball_ball, ball_cushion, ball_pocket)
  angle: number;
  velocity_before?: [number, number];
  velocity_after?: [number, number];
  time_to_collision?: number;
}

export interface TrajectoryData {
  ball_id?: string;  // ID of the moving ball (typically cue ball)
  lines: TrajectoryLine[];
  collisions: CollisionData[];
  confidence: number;
  calculation_time_ms: number;
  line_count: number;
  collision_count: number;
}

// Alert Data
export interface AlertData {
  level: AlertLevel;
  message: string;
  code: string;
  details: Record<string, any>;
}

// Configuration Data
export interface ConfigData {
  section: string;
  config: Record<string, any>;
  change_summary?: string;
}

// Connection and Status Data
export interface ConnectionData {
  client_id: string;
  status: 'connected' | 'reconnecting' | 'disconnected';
  timestamp: string;
}

export interface StatusData {
  client_id: string;
  user_id?: string;
  connected_at: string;
  uptime: number;
  subscriptions: string[];
  message_count: number;
  bytes_sent: number;
  bytes_received: number;
  quality_score: number;
  last_ping_latency: number;
  is_alive: boolean;
}

export interface ErrorData {
  code: string;
  message: string;
  details?: Record<string, any>;
}

export interface MetricsData {
  broadcast_stats: Record<string, number | boolean>;
  frame_metrics: Record<string, number>;
  connection_stats: Record<string, any>;
}

// =============================================================================
// REST API Response Types
// =============================================================================

// Health Check
export interface ComponentHealth extends BaseResponse {
  name: string;
  status: HealthStatus;
  message?: string;
  last_check: string;
  uptime?: number;
  errors: string[];
}

export interface SystemMetrics extends BaseResponse {
  cpu_usage: number;
  memory_usage: number;
  disk_usage: number;
  network_io: Record<string, number>;
  api_requests_per_second: number;
  websocket_connections: number;
  average_response_time: number;
}

export interface HealthResponse extends BaseResponse {
  status: HealthStatus;
  timestamp: string;
  uptime: number;
  version: string;
  components: Record<string, ComponentHealth>;
  metrics?: SystemMetrics;
}

// Configuration
export interface ConfigValidationError extends BaseResponse {
  field: string;
  message: string;
  current_value: any;
  expected_type: string;
}

export interface ConfigResponse extends BaseResponse {
  timestamp: string;
  values: Record<string, any>;
  schema_version: string;
  last_modified: string;
  is_valid: boolean;
  validation_errors: ConfigValidationError[];
}

export interface ConfigUpdateResponse extends BaseResponse {
  success: boolean;
  updated_fields: string[];
  validation_errors: ConfigValidationError[];
  warnings: string[];
  rollback_available: boolean;
  restart_required: boolean;
}

// Authentication
export interface LoginResponse extends BaseResponse {
  success: boolean;
  access_token?: string;
  refresh_token?: string;
  expires_in?: number;
  user_id?: string;
  role?: string;
  permissions: string[];
  timestamp: string;
}

export interface TokenRefreshResponse extends BaseResponse {
  success: boolean;
  access_token?: string;
  expires_in?: number;
  timestamp: string;
}

export interface UserCreateResponse extends BaseResponse {
  success: boolean;
  user_id: string;
  username: string;
  role: string;
  created_at: string;
  validation_errors: string[];
}

// Game State
export interface BallInfo extends BaseResponse {
  id: string;
  number?: number;
  position: [number, number];
  velocity: [number, number];
  is_cue_ball: boolean;
  is_pocketed: boolean;
  confidence: number;
  last_update: string;
}

export interface CueInfo extends BaseResponse {
  tip_position: [number, number];
  angle: number;
  elevation: number;
  estimated_force: number;
  is_visible: boolean;
  confidence: number;
}

export interface TableInfo extends BaseResponse {
  width: number;
  height: number;
  pocket_positions: [number, number][];
  pocket_radius: number;
  surface_friction: number;
}

export interface GameEvent extends BaseResponse {
  timestamp: string;
  event_type: string;
  description: string;
  data: Record<string, any>;
}

export interface GameStateResponse extends BaseResponse {
  timestamp: string;
  frame_number: number;
  balls: BallInfo[];
  cue?: CueInfo;
  table: TableInfo;
  game_type: string;
  is_valid: boolean;
  confidence: number;
  events: GameEvent[];
}

// Calibration
export interface CalibrationSession extends BaseResponse {
  session_id: string;
  calibration_type: string;
  status: string;
  created_at: string;
  expires_at: string;
  points_captured: number;
  points_required: number;
  accuracy?: number;
  errors: string[];
}

export interface CalibrationStartResponse extends BaseResponse {
  session: CalibrationSession;
  instructions: string[];
  expected_points: number;
  timeout: number;
}

export interface CalibrationPointResponse extends BaseResponse {
  success: boolean;
  point_id: string;
  accuracy: number;
  total_points: number;
  remaining_points: number;
  can_proceed: boolean;
}

export interface CalibrationApplyResponse extends BaseResponse {
  success: boolean;
  accuracy: number;
  backup_created: boolean;
  applied_at: string;
  transformation_matrix: number[][];
  errors: string[];
}

// WebSocket Management
export interface WebSocketConnectionResponse extends BaseResponse {
  success: boolean;
  connection_id: string;
  available_streams: string[];
  supported_qualities: string[];
  max_frame_rate: number;
  timestamp: string;
}

export interface WebSocketSubscriptionResponse extends BaseResponse {
  success: boolean;
  subscribed_streams: string[];
  failed_streams: string[];
  quality: string;
  frame_rate: number;
  timestamp: string;
}

// =============================================================================
// Request Types for API Calls
// =============================================================================

export interface LoginRequest {
  username: string;
  password: string;
}

export interface ConfigUpdateRequest {
  section?: string;
  values: Record<string, any>;
  validate_only?: boolean;
}

export interface SubscribeRequest {
  streams: string[];
  filters?: Record<string, any>;
}

export interface UnsubscribeRequest {
  streams: string[];
}

export interface CalibrationStartRequest {
  calibration_type: string;
  table_type?: string;
  timeout?: number;
}

export interface CalibrationPointRequest {
  x: number;
  y: number;
  screen_x: number;
  screen_y: number;
}

// =============================================================================
// Constants
// =============================================================================

export const VALID_STREAM_TYPES = [
  'frame',
  'state',
  'trajectory',
  'alert',
  'config'
] as const;

export const VALID_ALERT_LEVELS = [
  'info',
  'warning',
  'error',
  'critical'
] as const;

export const VALID_QUALITY_LEVELS = [
  'low',
  'medium',
  'high',
  'auto'
] as const;

// =============================================================================
// Type Guards
// =============================================================================

export function isErrorResponse(response: any): response is ErrorResponse {
  return response && typeof response.error === 'string' && typeof response.code === 'string';
}

export function isSuccessResponse(response: any): response is SuccessResponse {
  return response && typeof response.success === 'boolean';
}

export function isWebSocketMessage(message: any): message is WebSocketMessage {
  return message &&
         typeof message.type === 'string' &&
         typeof message.timestamp === 'string' &&
         typeof message.data === 'object';
}

export function isFrameMessage(message: WebSocketMessage): message is WebSocketMessage & { data: FrameData } {
  return message.type === 'frame' &&
         typeof message.data.image === 'string' &&
         typeof message.data.width === 'number' &&
         typeof message.data.height === 'number';
}

export function isGameStateMessage(message: WebSocketMessage): message is WebSocketMessage & { data: GameStateData } {
  return message.type === 'state' &&
         Array.isArray(message.data.balls);
}

export function isTrajectoryMessage(message: WebSocketMessage): message is WebSocketMessage & { data: TrajectoryData } {
  return message.type === 'trajectory' &&
         Array.isArray(message.data.lines);
}

export function isAlertMessage(message: WebSocketMessage): message is WebSocketMessage & { data: AlertData } {
  return message.type === 'alert' &&
         typeof message.data.level === 'string' &&
         typeof message.data.message === 'string';
}
