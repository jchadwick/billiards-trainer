/** Wire contracts required by the live camera and state client. */

export type MessageType =
  | 'state'
  | 'ping'
  | 'pong'
  | 'subscribe'
  | 'unsubscribe'
  | 'subscribed'
  | 'unsubscribed'
  | 'status'
  | 'error';

export interface WebSocketMessage {
  /** Keep the wire type open so unknown server messages can be ignored safely. */
  type: string;
  timestamp: string;
  data: Record<string, any>;
}

export interface SubscribeRequest {
  streams: string[];
  filters?: Record<string, any>;
}

export interface UnsubscribeRequest {
  streams: string[];
}

export type GameType = 'practice' | '8ball' | '9ball' | 'straight';

export interface PositionWithScale {
  x: number;
  y: number;
  /** Per-axis source pixel to canonical 4K pixel scale. */
  scale: [number, number];
}

export interface ImageOutline {
  points: { x: number; y: number }[];
  scale: [number, number];
}

export interface BallData {
  outline?: ImageOutline;
  source_observation?: ObservationIdentity;
  id: string | number;
  number?: number | null;
  type?: 'cue' | 'solid' | 'stripe' | 'eight' | 'unknown';
  is_cue_ball?: boolean;
  is_observed?: boolean;
  position_kind?: 'measured' | 'filtered' | 'extrapolated' | 'last_observed';
  confidence: number;
  confidence_kind?: 'detector_observation' | 'unavailable';
  position: PositionWithScale;
  radius: number;
  radius_scale?: [number, number];
  radius_extent?: PositionWithScale;
}

export interface CueData {
  tip_position?: PositionWithScale;
  butt_position?: PositionWithScale;
  detected?: boolean;
  is_visible?: boolean;
  confidence?: number;
}

export interface ObservationIdentity {
  engine_instance_id?: string;
  session_id?: string | number;
  frame_id?: string;
  snapshot_revision?: number;
  configuration_revision?: number;
  calibration_revision?: number | string;
  rectification_revision?: number;
}

export type PerceptionPhase = 'recovering' | 'pre_shot' | 'aiming' | 'shot_underway';

/**
 * Freshness and eligibility from the low-FPS perception coordinator. These
 * ages describe separate evidence sources and must not renew ball positions.
 */
export interface PerceptionState {
  event_sequence: number;
  image_width_px?: number | null;
  image_height_px?: number | null;
  phase: PerceptionPhase;
  layout_revision: number;
  layout_trusted: boolean;
  /** Explicit proof that a complete empty-table layout was verified. */
  empty_layout_verified?: boolean;
  guidance_valid: boolean;
  trust_reason: string;
  last_ball_observation_age_ms?: number | null;
  latest_motion_check_age_ms?: number | null;
  cue_evidence_age_ms?: number | null;
  motion_evidence_max_age_ms: number;
  cue_evidence_max_age_ms: number;
  layout_verification_max_age_ms: number;
  cue?: {
    shaft_start_px: [number, number];
    shaft_end_px: [number, number];
    axis_unit: [number, number];
    confidence: number;
    observation_age_ms: number;
    signed_direction?: 'toward_start' | 'toward_end' | null;
    source_frame_id?: string | null;
  } | null;
  /** Frame whose pixels were used for this latest live decision. */
  decision_frame_id?: string | null;
  decision_calibration_revision?: number | null;
  decision_rectification_revision?: number | null;
  table_visibility?: {
    status: 'visible' | 'not_visible' | 'unknown';
    reason: string;
    observation_frame_id: string;
    observation_age_ms: number;
  } | null;
}

export interface ObservationStatus {
  observation_identity?: ObservationIdentity;
  engine_instance_id?: string;
  vision_instance_id?: string;
  source_session_id?: string | number;
  source_process_id?: string | number;
  source_frame_id?: string;
  source_frame_number?: number;
  source_timestamp?: number | string;
  snapshot_revision?: number;
  transport_sequence?: number;
  frame_publication_age_seconds?: number;
  freshness_status?: 'source_silent';
  ball_positions_status?: 'observed' | 'cached' | 'unavailable';
  ball_detection_status?: 'observed' | 'degraded' | 'skipped' | 'failed';
  position_retention?: PositionRetentionStatus;
  perception?: PerceptionState;
  [key: string]: unknown;
}

/** Optional per-WebSocket retention metadata for consumer-safe brief gaps. */
export interface PositionRetentionStatus {
  retained: boolean;
  /** Age of the held layout, independent from the current unavailable interval. */
  retained_age_ms?: number | null;
  /** Age since the first unavailable state in the current gap. */
  unavailable_age_ms?: number | null;
  grace_ms: number;
  current_status?: string | null;
  original_status?: string | null;
  source_identity?: Record<string, unknown> | null;
}

export interface GameStateData {
  balls: BallData[];
  ball_count?: number;
  cue?: CueData | null;
  game_type?: GameType;
  frame_number?: number;
  observation_status?: ObservationStatus;
  [key: string]: unknown;
}

export function isWebSocketMessage(message: unknown): message is WebSocketMessage {
  if (!message || typeof message !== 'object') return false;
  const candidate = message as Record<string, unknown>;
  return typeof candidate.type === 'string'
    && typeof candidate.timestamp === 'string'
    && typeof candidate.data === 'object'
    && candidate.data !== null
    && !Array.isArray(candidate.data);
}

export function isGameStateMessage(
  message: WebSocketMessage,
): message is WebSocketMessage & { data: GameStateData } {
  return message.type === 'state'
    && Array.isArray((message.data as Record<string, unknown>).balls);
}
