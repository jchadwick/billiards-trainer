export type DiagnosticsConnectionState =
  | "connecting"
  | "connected"
  | "reconnecting"
  | "disconnected"
  | "error";

export type DiagnosticsStageName =
  | "gateway_service"
  | "source_lifecycle"
  | "capture_progress"
  | "model_initialization"
  | "scheduled_perception_work"
  | "scene_delivery"
  | "localization";

export type DiagnosticsStageStatus =
  | "starting"
  | "healthy"
  | "idle"
  | "degraded"
  | "stalled"
  | "failed"
  | "stopped"
  | "unknown";

export type ScheduledWorkState =
  | "unknown"
  | "idle"
  | "phase_gated"
  | "due"
  | "in_flight"
  | "result_ready"
  | "failed";

export type ScheduledJobKind =
  | "motion_assessment"
  | "ball_detection"
  | "cue_geometry";

export interface BuildInfo {
  package_name: string;
  package_version: string;
  git_revision: string | null;
  dirty: boolean | null;
}

export interface ModelClass {
  class_id: number;
  class_name: string;
}

export interface ActiveModel {
  identity: {
    artifact_sha256: string;
    source_checkpoint_sha256: string | null;
    classes: ModelClass[];
  };
  input_width_px: number;
  input_height_px: number;
  requested_provider: string;
  execution_provider: string | null;
  execution_device: string | null;
  execution_proof: string | null;
}

export interface ProducerMetadata {
  build: BuildInfo;
  model_state: "loading" | "active" | "failed" | "stopped";
  model: ActiveModel | null;
  engine_instance_id: string | null;
  capture_session_id: string | null;
  failure_code: string | null;
}

export interface ProducerStageObservation {
  stage: "source_lifecycle" | "capture_progress" | "model_initialization" | "scheduled_perception_work";
  status: DiagnosticsStageStatus;
  reason_code: string;
  detail: string | null;
  last_progress_age_ms: number | null;
  in_flight: boolean;
  work_started_age_ms: number | null;
  expected_jobs: number | null;
  completed_jobs: number | null;
  failed_jobs: number | null;
  scheduler_state: ScheduledWorkState | null;
  active_job_kind: ScheduledJobKind | null;
  last_completion_age_ms: number | null;
  last_failure_code: string | null;
  last_failure_age_ms: number | null;
  deadline_ms: number | null;
}

export interface ProducerDiagnosticsReport {
  schema_version: 1;
  generation_id: string;
  process_id: string;
  sequence: number;
  metadata: ProducerMetadata;
  stages: ProducerStageObservation[];
}

export type ProducerExitReason =
  | "graceful_stop"
  | "process_exited_successfully"
  | "process_exited_nonzero"
  | "process_wait_failed"
  | "spawn_failed"
  | "launch_request_failed"
  | "forced_termination"
  | "unknown";

export interface ProducerLastKnownSnapshot {
  report: ProducerDiagnosticsReport;
  report_age_ms: number;
}

export interface ProducerReportSnapshot {
  state: "unknown" | "fresh" | "expired" | "stopped";
  generation_id: string | null;
  report_sequence: number | null;
  report_age_ms: number | null;
  expires_in_ms: number | null;
  metadata: ProducerMetadata | null;
  exit_reason: ProducerExitReason | null;
  last_known: ProducerLastKnownSnapshot | null;
}

export interface BrowserDiagnosticsStage {
  stage: DiagnosticsStageName;
  status: DiagnosticsStageStatus;
  reason_code: string;
  source: "gateway_runtime" | "supervisor_registry" | "producer_report" | "scene_lifecycle";
  detail: string | null;
  last_progress_age_ms: number | null;
  in_flight: boolean;
  work_started_age_ms: number | null;
  expected_jobs: number | null;
  completed_jobs: number | null;
  failed_jobs: number | null;
  scheduler_state: ScheduledWorkState | null;
  active_job_kind: ScheduledJobKind | null;
  last_completion_age_ms: number | null;
  last_failure_code: string | null;
  last_failure_age_ms: number | null;
  deadline_ms: number | null;
}

export interface BrowserDiagnosticsSnapshot {
  schema_version: 1;
  gateway: {
    instance_id: string;
    revision: number;
    assessed_at_utc: string;
    uptime_ms: number;
    build: BuildInfo;
  };
  producer: ProducerReportSnapshot;
  stream: { heartbeat_interval_ms: number; stale_after_ms: number };
  stages: BrowserDiagnosticsStage[];
}

const stageNames: DiagnosticsStageName[] = [
  "gateway_service",
  "source_lifecycle",
  "capture_progress",
  "model_initialization",
  "scheduled_perception_work",
  "scene_delivery",
  "localization",
];
const stageStatuses: DiagnosticsStageStatus[] = [
  "starting",
  "healthy",
  "idle",
  "degraded",
  "stalled",
  "failed",
  "stopped",
  "unknown",
];
const scheduledStates: ScheduledWorkState[] = [
  "unknown",
  "idle",
  "phase_gated",
  "due",
  "in_flight",
  "result_ready",
  "failed",
];
const scheduledJobKinds: ScheduledJobKind[] = [
  "motion_assessment",
  "ball_detection",
  "cue_geometry",
];
const producerExitReasons: ProducerExitReason[] = [
  "graceful_stop",
  "process_exited_successfully",
  "process_exited_nonzero",
  "process_wait_failed",
  "spawn_failed",
  "launch_request_failed",
  "forced_termination",
  "unknown",
];
const evidenceSources = [
  "gateway_runtime",
  "supervisor_registry",
  "producer_report",
  "scene_lifecycle",
] as const;

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value);
const isSafeInteger = (value: unknown, minimum = 0): value is number =>
  typeof value === "number" && Number.isSafeInteger(value) && value >= minimum;
const isNullableText = (value: unknown): value is string | null =>
  value === null || typeof value === "string";
const isNullableInteger = (value: unknown): value is number | null =>
  value === null || isSafeInteger(value);
const isEnum = <T extends string>(value: unknown, options: readonly T[]): value is T =>
  typeof value === "string" && options.includes(value as T);

function validBuild(value: unknown): value is BuildInfo {
  return isRecord(value)
    && typeof value.package_name === "string"
    && typeof value.package_version === "string"
    && isNullableText(value.git_revision)
    && (value.dirty === null || typeof value.dirty === "boolean");
}

function validModel(value: unknown): value is ActiveModel {
  if (!isRecord(value) || !isRecord(value.identity)) return false;
  const identity = value.identity;
  return typeof identity.artifact_sha256 === "string"
    && isNullableText(identity.source_checkpoint_sha256)
    && Array.isArray(identity.classes)
    && identity.classes.every((item) => isRecord(item)
      && isSafeInteger(item.class_id)
      && typeof item.class_name === "string")
    && isSafeInteger(value.input_width_px, 1)
    && isSafeInteger(value.input_height_px, 1)
    && typeof value.requested_provider === "string"
    && isNullableText(value.execution_provider)
    && isNullableText(value.execution_device)
    && isNullableText(value.execution_proof);
}

function validMetadata(value: unknown): value is ProducerMetadata {
  if (!isRecord(value)) return false;
  const modelState = value.model_state;
  const validModelState = ["loading", "active", "failed", "stopped"].includes(String(modelState));
  return validBuild(value.build)
    && validModelState
    && (value.model === null || validModel(value.model))
    && ((modelState === "active") === (value.model !== null))
    && isNullableText(value.engine_instance_id)
    && isNullableText(value.capture_session_id)
    && isNullableText(value.failure_code);
}

function validProducerStage(value: unknown): value is ProducerStageObservation {
  if (!isRecord(value)) return false;
  const stages = ["source_lifecycle", "capture_progress", "model_initialization", "scheduled_perception_work"];
  const workTimerConsistent = (value.work_started_age_ms === null) === (value.deadline_ms === null);
  const scheduled = value.stage === "scheduled_perception_work";
  const inFlightValid = value.in_flight === false
    ? value.active_job_kind === null && value.scheduler_state !== "in_flight"
    : scheduled && value.scheduler_state === "in_flight" && isEnum(value.active_job_kind, scheduledJobKinds);
  return isEnum(value.stage, stages)
    && isEnum(value.status, stageStatuses)
    && typeof value.reason_code === "string"
    && isNullableText(value.detail)
    && isNullableInteger(value.last_progress_age_ms)
    && typeof value.in_flight === "boolean"
    && isNullableInteger(value.work_started_age_ms)
    && isNullableInteger(value.deadline_ms)
    && workTimerConsistent
    && isNullableInteger(value.expected_jobs)
    && isNullableInteger(value.completed_jobs)
    && isNullableInteger(value.failed_jobs)
    && (value.scheduler_state === null || isEnum(value.scheduler_state, scheduledStates))
    && (value.active_job_kind === null || isEnum(value.active_job_kind, scheduledJobKinds))
    && isNullableInteger(value.last_completion_age_ms)
    && isNullableText(value.last_failure_code)
    && isNullableInteger(value.last_failure_age_ms)
    && inFlightValid
    && (scheduled || [value.expected_jobs, value.completed_jobs, value.failed_jobs, value.scheduler_state, value.active_job_kind, value.last_completion_age_ms, value.last_failure_code, value.last_failure_age_ms].every((item) => item === null));
}

function validProducerReport(value: unknown): value is ProducerDiagnosticsReport {
  if (!isRecord(value) || !Array.isArray(value.stages) || value.stages.length > 16) {
    return false;
  }
  const stages = value.stages;
  return value.schema_version === 1
    && typeof value.generation_id === "string"
    && typeof value.process_id === "string"
    && isSafeInteger(value.sequence, 1)
    && validMetadata(value.metadata)
    && stages.every(validProducerStage)
    && new Set(stages.filter(validProducerStage).map((stage) => stage.stage)).size === stages.length;
}

function validProducer(value: unknown): value is ProducerReportSnapshot {
  if (!isRecord(value)) return false;
  const state = value.state;
  const validState = ["unknown", "fresh", "expired", "stopped"].includes(String(state));
  const lastKnown = value.last_known;
  const validLastKnown = lastKnown === null || (isRecord(lastKnown)
    && validProducerReport(lastKnown.report)
    && isSafeInteger(lastKnown.report_age_ms));
  return validState
    && isNullableText(value.generation_id)
    && isNullableInteger(value.report_sequence)
    && isNullableInteger(value.report_age_ms)
    && isNullableInteger(value.expires_in_ms)
    && (value.metadata === null || validMetadata(value.metadata))
    && ((state === "fresh") === (value.metadata !== null))
    && isNullableText(value.exit_reason) && (value.exit_reason === null || isEnum(value.exit_reason, producerExitReasons))
    && validLastKnown
    && ((state === "stopped") === (value.exit_reason !== null))
    && (state === "fresh" ? isSafeInteger(value.report_sequence, 1) && isSafeInteger(value.report_age_ms) : value.metadata === null)
    && (state === "fresh" ? lastKnown === null : true);
}

function validStage(value: unknown): value is BrowserDiagnosticsStage {
  if (!isRecord(value)) return false;
  const scheduled = value.stage === "scheduled_perception_work";
  const timerConsistent = (value.work_started_age_ms === null) === (value.deadline_ms === null);
  const inFlightValid = value.in_flight === false
    ? value.active_job_kind === null && value.scheduler_state !== "in_flight"
    : scheduled && value.scheduler_state === "in_flight" && isEnum(value.active_job_kind, scheduledJobKinds);
  return isEnum(value.stage, stageNames)
    && isEnum(value.status, stageStatuses)
    && typeof value.reason_code === "string"
    && isEnum(value.source, evidenceSources)
    && isNullableText(value.detail)
    && isNullableInteger(value.last_progress_age_ms)
    && typeof value.in_flight === "boolean"
    && isNullableInteger(value.work_started_age_ms)
    && isNullableInteger(value.expected_jobs)
    && isNullableInteger(value.completed_jobs)
    && isNullableInteger(value.failed_jobs)
    && (value.scheduler_state === null || isEnum(value.scheduler_state, scheduledStates))
    && (value.active_job_kind === null || isEnum(value.active_job_kind, scheduledJobKinds))
    && isNullableInteger(value.last_completion_age_ms)
    && isNullableText(value.last_failure_code)
    && isNullableInteger(value.last_failure_age_ms)
    && isNullableInteger(value.deadline_ms)
    && timerConsistent
    && inFlightValid
    && (scheduled || [value.expected_jobs, value.completed_jobs, value.failed_jobs, value.scheduler_state, value.active_job_kind, value.last_completion_age_ms, value.last_failure_code, value.last_failure_age_ms].every((item) => item === null));
}

/** Validate the raw, versioned JSON sent on `/api/v1/diagnostics/ws`. */
export function isBrowserDiagnosticsSnapshot(value: unknown): value is BrowserDiagnosticsSnapshot {
  if (!isRecord(value) || value.schema_version !== 1 || !isRecord(value.gateway) || !isRecord(value.stream)) return false;
  const gateway = value.gateway;
  const stream = value.stream;
  if (!Array.isArray(value.stages)) return false;
  const stages = value.stages;
  return typeof gateway.instance_id === "string"
    && gateway.instance_id.length > 0
    && isSafeInteger(gateway.revision, 1)
    && typeof gateway.assessed_at_utc === "string"
    && Number.isFinite(Date.parse(gateway.assessed_at_utc))
    && isSafeInteger(gateway.uptime_ms)
    && validBuild(gateway.build)
    && validProducer(value.producer)
    && isSafeInteger(stream.heartbeat_interval_ms, 1)
    && isSafeInteger(stream.stale_after_ms, 1)
    && stream.heartbeat_interval_ms <= 300_000
    && stream.stale_after_ms <= 300_000
    && stream.stale_after_ms > stream.heartbeat_interval_ms
    && stages.length === stageNames.length
    && stages.every(validStage)
    && new Set(stages.filter(validStage).map((stage) => stage.stage)).size === stageNames.length
    && stageNames.every((name) => stages.some((stage) => isRecord(stage) && stage.stage === name));
}

export interface DiagnosticsTransportCallbacks {
  onPayload(payload: unknown): void;
  onConnectionState(state: DiagnosticsConnectionState): void;
}

export interface DiagnosticsTransport {
  connect(callbacks: DiagnosticsTransportCallbacks): void;
  destroy(): void;
}

export interface DiagnosticsReconnectPolicy {
  initialDelayMs: number;
  maximumDelayMs: number;
}

const DEFAULT_RECONNECT_POLICY: DiagnosticsReconnectPolicy = {
  initialDelayMs: 1_000,
  maximumDelayMs: 30_000,
};

export function getDiagnosticsReconnectPolicy(
  env: Record<string, unknown> = import.meta.env,
): DiagnosticsReconnectPolicy {
  const initial = Number(env.VITE_DIAGNOSTICS_RECONNECT_INITIAL_MS ?? DEFAULT_RECONNECT_POLICY.initialDelayMs);
  const maximum = Number(env.VITE_DIAGNOSTICS_RECONNECT_MAX_MS ?? DEFAULT_RECONNECT_POLICY.maximumDelayMs);
  if (!Number.isSafeInteger(initial) || initial < 100 || initial > 30_000
    || !Number.isSafeInteger(maximum) || maximum < initial || maximum > 120_000) {
    return { ...DEFAULT_RECONNECT_POLICY };
  }
  return { initialDelayMs: initial, maximumDelayMs: maximum };
}

type SocketLike = Pick<WebSocket, "close"> & {
  onopen: ((event: Event) => void) | null;
  onmessage: ((event: MessageEvent) => void) | null;
  onerror: ((event: Event) => void) | null;
  onclose: ((event: CloseEvent) => void) | null;
};
type WebSocketFactory = (url: string) => SocketLike;

/** One reconnecting raw-snapshot WebSocket. It sends no requests or polling traffic. */
export class BrowserDiagnosticsWebSocket implements DiagnosticsTransport {
  private callbacks: DiagnosticsTransportCallbacks | null = null;
  private socket: SocketLike | null = null;
  private reconnectTimer: number | null = null;
  private retryCount = 0;
  private destroyed = false;
  private readonly policy: DiagnosticsReconnectPolicy;
  private readonly createSocket: WebSocketFactory;

  constructor(
    private readonly url: string,
    policy: DiagnosticsReconnectPolicy = getDiagnosticsReconnectPolicy(),
    createSocket?: WebSocketFactory,
  ) {
    this.policy = policy;
    this.createSocket = createSocket ?? ((target) => new WebSocket(target));
  }

  connect(callbacks: DiagnosticsTransportCallbacks): void {
    if (this.destroyed) return;
    this.callbacks = callbacks;
    this.openSocket();
  }

  destroy(): void {
    this.destroyed = true;
    this.callbacks = null;
    if (this.reconnectTimer !== null && typeof window !== "undefined") {
      window.clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    const socket = this.socket;
    this.socket = null;
    socket?.close(1000, "Application stopped");
  }

  private openSocket(): void {
    if (this.destroyed || this.socket) return;
    this.callbacks?.onConnectionState(this.retryCount === 0 ? "connecting" : "reconnecting");
    try {
      const socket = this.createSocket(this.url);
      this.socket = socket;
      socket.onopen = () => {
        if (this.socket !== socket || this.destroyed) return;
        this.retryCount = 0;
        this.callbacks?.onConnectionState("connected");
      };
      socket.onmessage = (event) => {
        if (this.socket !== socket || this.destroyed) return;
        if (typeof event.data !== "string") return;
        try {
          this.callbacks?.onPayload(JSON.parse(event.data) as unknown);
        } catch {
          this.callbacks?.onPayload(undefined);
        }
      };
      socket.onerror = () => {
        if (this.socket === socket && !this.destroyed) {
          this.callbacks?.onConnectionState("error");
        }
      };
      socket.onclose = () => {
        if (this.socket !== socket) return;
        this.socket = null;
        if (this.destroyed) return;
        this.callbacks?.onConnectionState("disconnected");
        this.scheduleReconnect();
      };
    } catch {
      this.callbacks?.onConnectionState("error");
      this.scheduleReconnect();
    }
  }

  private scheduleReconnect(): void {
    if (this.destroyed || typeof window === "undefined" || this.reconnectTimer !== null) return;
    const delay = Math.min(
      this.policy.initialDelayMs * 2 ** Math.min(this.retryCount, 16),
      this.policy.maximumDelayMs,
    );
    this.retryCount += 1;
    this.reconnectTimer = window.setTimeout(() => {
      this.reconnectTimer = null;
      this.openSocket();
    }, delay);
  }
}
