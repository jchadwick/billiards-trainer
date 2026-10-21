/**
 * TypeScript interfaces for Billiards Trainer API
 * Generated from backend Pydantic models
 */

// =============================================================================
// Base Types and Enums
// =============================================================================

/**
 * Position with scale metadata for coordinate system tracking.
 *
 * `x` and `y` are in the source image's pixel units. `scale` contains the
 * independent x/y multipliers that map those values into canonical 4K pixels.
 * A canonical point therefore has scale `[1, 1]`; the tuple is not a frame
 * resolution.
 */
export interface PositionWithScale {
  x: number;
  y: number;
  scale: [number, number];
}

export type HealthStatus = 'healthy' | 'degraded' | 'unhealthy';
export type StatusCode = 'success' | 'partial_success' | 'failure' | 'pending' | 'timeout' | 'cancelled';
export type LogLevel = 'CRITICAL' | 'ERROR' | 'WARNING' | 'INFO' | 'DEBUG';
export type GameType = 'practice' | '8ball' | '9ball' | 'straight';

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
  /** Null means the source explicitly reports unknown identity. */
  number?: number | null;
  identity_status?: 'identified' | 'unknown';
  identity_confidence?: number | null;
  identity_evidence?: Array<Record<string, unknown>>;
  is_cue_ball?: boolean;
  type?: 'cue' | 'solid' | 'stripe' | 'eight' | 'unknown';
  /** False when the tracker is extrapolating from a missed observation. */
  is_observed?: boolean;
  is_pocketed?: boolean;
  position: PositionWithScale;
  /** Radius in the source pixel units described by radius_scale. */
  radius: number;
  radius_scale?: [number, number];
  radius_unit?: 'canonical_pixels';
  radius_extent?: PositionWithScale;
  color?: string;
  velocity?: PositionWithScale;
  confidence: number;
  confidence_kind?: 'detector_observation' | 'unavailable';
  last_observation_confidence?: number | null;
  visible?: boolean;
  /** Canonical rendering radius may be an assumption rather than a measurement. */
  radius_source?: string;
  tracking_status?: 'observed' | 'tracked' | 'occluded' | 'lost';
  position_kind?: 'measured' | 'filtered' | 'extrapolated' | 'last_observed';
  table_position?: {
    x: number;
    y: number;
    unit: 'millimeters';
    origin: 'playing_surface_center';
  };
  source_observation?: {
    engine_instance_id?: string;
    session_id?: string;
    frame_id?: string;
    timestamp?: Record<string, unknown>;
  };
  /** This public state stream is rendered over raw camera frames. */
  image_space?: 'raw';
  image_width_px?: number;
  image_height_px?: number;
}

export interface CueData {
  /** Angle in degrees in canonical image coordinates. */
  angle: number;
  /** Preferred cue endpoint. Older payloads may send only `position`. */
  position?: PositionWithScale;
  detected?: boolean;
  is_visible?: boolean;
  confidence: number;
  length?: number;
  length_scale?: [number, number];
  tip_position?: PositionWithScale;
  butt_position?: PositionWithScale;
}

export interface PhysicalTableDimensions {
  width: number;
  height: number;
  unit: 'meters';
}

export interface TablePocketData {
  position: PositionWithScale;
  radius?: number;
  radius_scale?: [number, number];
  radius_unit?: 'canonical_pixels';
  radius_extent?: PositionWithScale;
}

export interface TableData {
  /** Vision payload boundaries; API TableStateData may provide pocket_positions instead. */
  corners?: PositionWithScale[];
  playing_area_corners?: PositionWithScale[];
  pockets?: Array<PositionWithScale | TablePocketData>;
  /** Core/API table shape, whose geometry is positioned by pocket centers. */
  pocket_positions?: PositionWithScale[];
  pocket_radius?: number;
  pocket_radius_scale?: [number, number];
  pocket_radius_unit?: 'canonical_pixels';
  rails?: Array<Record<string, unknown> | PositionWithScale[]>;
  calibrated?: boolean;
  /** Image-space extents are pixels and carry their own scale. */
  width?: number;
  width_scale?: [number, number];
  height?: number;
  height_scale?: [number, number];
  /** Shared x/y scale for width and height in the API/Core table payload. */
  dimensions_scale?: [number, number];
  dimensions_unit?: 'canonical_pixels';
  /** Real table measurements retain their physical units. */
  dimensions?: PhysicalTableDimensions;
  real_world_dimensions?: PhysicalTableDimensions;
}

/** Unit-bearing API/Core table payload used by HTTP and WebSocket state models. */
export interface TableStateData extends TableData {
  width: number;
  height: number;
  dimensions_scale: [number, number];
  dimensions_unit: 'canonical_pixels';
  pocket_positions: PositionWithScale[];
  pocket_radius: number;
  pocket_radius_scale: [number, number];
  pocket_radius_unit: 'canonical_pixels';
  real_world_dimensions?: PhysicalTableDimensions;
}

export interface GameStateData {
  balls: BallData[];
  cue?: CueData;
  table?: TableData;
  ball_count: number;
  /** Current game mode selected in the core state. */
  game_type?: GameType;
  frame_number?: number;
  /** Vision scheduler metadata; optional for older servers. */
  phase?: string;
  calibration_revision?: string | number;
  map_revision?: string | number;
  map_trusted?: boolean;
  aim_calculation_eligible?: boolean;
  moving_ball_count?: number;
  observation_status?: {
    phase?: string;
    calibration_revision?: string | number;
    map_revision?: string | number;
    map_trusted?: boolean;
    aim_calculation_eligible?: boolean;
    moving_ball_count?: number;
    source_frame_number?: number;
    source_timestamp?: number | string;
    source_timestamp_kind?: string;
    source_timestamp_unit?: 'nanoseconds' | 'milliseconds' | 'seconds';
    source_timestamp_clock?: 'monotonic' | 'unix_epoch' | 'sensor';
    source_timestamp_domain_id?: string;
    source_timestamp_is_exposure?: boolean;
    source_session_id?: string;
    source_frame_id?: string;
    source_image_space?: 'raw' | 'rectified';
    source_width_px?: number;
    source_height_px?: number;
    engine_instance_id?: string;
    snapshot_revision?: number;
    transport_sequence?: number;
    configuration_revision?: number;
    rectification_revision?: number;
    calibration_state?: 'valid' | 'invalid' | 'unavailable';
    ball_observation_source_timestamp?: number | string;
    ball_observation_source_timestamp_unit?: 'nanoseconds' | 'milliseconds' | 'seconds';
    ball_observation_source_timestamp_kind?: string;
    frame_publication_age_seconds?: number;
    ball_positions_status?: 'observed' | 'cached' | 'unavailable';
    ball_detection_status?: 'observed' | 'degraded' | 'skipped' | 'failed';
    health_reason?: string | null;
    freshness_status?: 'source_silent';
    observation_identity?: {
      engine_instance_id?: string;
      session_id?: string;
      frame_id?: string;
      snapshot_revision?: number;
    };
    ball_observation_source_frame_number?: number;
    ball_observation_timestamp?: number;
    maximum_ball_observation_gap_seconds?: number;
    actual_path_history_max_points?: number;
    [key: string]: unknown;
  };
}

// Trajectory Data
export interface TrajectoryLine {
  /** Ball whose path this segment represents, when provided by the producer. */
  ball_id?: string;
  start: PositionWithScale;
  end: PositionWithScale;
  type: 'primary' | 'reflection' | 'collision';
  confidence: number;
}

export interface CollisionData {
  position: PositionWithScale;
  ball1_id?: string;  // Moving ball ID (e.g., cue ball)
  ball2_id?: string;  // Target ball ID (ball being hit, None for cushion/pocket)
  type?: string;  // Collision type (ball_ball, ball_cushion, ball_pocket)
  angle: number;
  velocity_before?: PositionWithScale;
  velocity_after?: PositionWithScale;
  time_to_collision?: number;
}

export interface TrajectoryData {
  lines: TrajectoryLine[];
  collisions: CollisionData[];
  confidence: number;
  calculation_time_ms: number;
  line_count: number;
  collision_count: number;
  /** `live` is current aiming guidance; `frozen_reference` is a retained prior prediction. */
  prediction_status: 'live' | 'frozen_reference';
  prediction_id?: string;
  source_frame_number?: number;
  source_timestamp?: number | string;
  map_revision?: string | number;
  calibration_revision?: string | number;
  source_observations?: Array<Record<string, unknown>>;
  /** False means the prediction must not be drawn in the current calibrated view. */
  spatial_valid?: boolean;
  phase?: string;
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

// Game State
export interface BallInfo extends BaseResponse {
  id: string;
  number?: number;
  position: PositionWithScale;
  velocity: PositionWithScale;
  is_cue_ball: boolean;
  is_pocketed: boolean;
  is_observed?: boolean;
  confidence: number;
  last_update: string;
}

export interface CueInfo extends BaseResponse {
  tip_position: PositionWithScale;
  angle: number;
  elevation: number;
  estimated_force: number;
  is_visible: boolean;
  confidence: number;
}

export interface TableInfo extends BaseResponse, TableStateData {
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

/** Rust localization response; unknown kinematics and metadata stay explicit. */
export interface SceneBallInfo extends Omit<BallInfo, 'velocity' | 'is_cue_ball' | 'is_pocketed'> {
  velocity: PositionWithScale | null;
  is_cue_ball: boolean | null;
  is_pocketed: boolean | null;
  is_observed: boolean;
  velocity_status: 'unavailable';
  cue_ball_status: 'unavailable';
  pocket_status: 'unavailable';
  confidence_kind: 'detector_observation' | 'unavailable';
  last_observation_confidence?: number | null;
  identity_status: 'identified' | 'unknown';
  identity_confidence?: number | null;
  identity_evidence: Array<Record<string, unknown>>;
  ball_type: 'solid' | 'stripe' | 'eight' | 'unknown';
  tracking_status: 'observed' | 'tracked' | 'occluded' | 'lost';
  position_kind: 'measured' | 'filtered' | 'extrapolated' | 'last_observed';
  table_position?: {
    x: number;
    y: number;
    unit: 'millimeters';
    origin: 'playing_surface_center';
  } | null;
  source_observation: {
    engine_instance_id: string;
    session_id: string;
    frame_id: string;
    timestamp: Record<string, unknown>;
  };
  radius: number;
  radius_unit: 'canonical_pixels';
  radius_source: 'legacy_standard_pool_ball_assumption';
}

/** Scene-localized variant of GET /api/v1/game/state. */
export interface SceneGameStateResponse extends BaseResponse {
  timestamp: string;
  frame_number: number | null;
  balls: SceneBallInfo[];
  cue: CueInfo | null;
  table: TableInfo | null;
  game_type: string | null;
  is_valid: boolean;
  confidence: number | null;
  confidence_status: 'mean_detector_confidence' | 'unavailable';
  core_metadata_status: 'available' | 'unavailable';
  core_metadata_timestamp: string | null;
  table_metadata_status: 'matched' | 'unverified' | 'mismatch' | 'unavailable';
  scene_calibration_revision: number | null;
  table_calibration_revision: number | null;
  events: GameEvent[];
  observation_status: Record<string, unknown>;
  perception_source: 'rust_scene';
}

/** HTTP state endpoint can return either retained Core or scene localization. */
export type ApiGameStateResponse = GameStateResponse | SceneGameStateResponse;

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
         Array.isArray(message.data.lines) &&
         Array.isArray(message.data.collisions) &&
         (message.data.prediction_status === 'live' ||
          message.data.prediction_status === 'frozen_reference');
}

export function isAlertMessage(message: WebSocketMessage): message is WebSocketMessage & { data: AlertData } {
  return message.type === 'alert' &&
         typeof message.data.level === 'string' &&
         typeof message.data.message === 'string';
}
