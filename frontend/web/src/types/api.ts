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

export interface BallData {
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
  [key: string]: unknown;
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
