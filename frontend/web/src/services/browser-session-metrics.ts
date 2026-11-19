import type { GameStateData } from "../types/api";
import { advanceEvidenceAgeMs } from "../components/video/freshness";

export interface BrowserSessionMetricsPolicy {
  /** Receipt window for browser-observed state and source-frame rates. */
  rateWindowMs: number;
  /** Local aggregation cadence. This timer never performs network work. */
  sampleIntervalMs: number;
  /** Maximum number of one-second samples retained in this browser session. */
  retentionBuckets: number;
  /** Bounds both rate receipt queues and recent identity de-duplication. */
  maximumRateEvents: number;
}

export const DEFAULT_BROWSER_SESSION_METRICS_POLICY: Readonly<BrowserSessionMetricsPolicy> =
  Object.freeze({
    rateWindowMs: 10_000,
    sampleIntervalMs: 1_000,
    retentionBuckets: 900,
    maximumRateEvents: 50_000,
  });

const MAXIMUM_RETENTION_BUCKETS = 3_600;
const MAXIMUM_RATE_EVENTS = 100_000;

function validatedNumber(
  raw: unknown,
  fallback: number,
  minimum: number,
  maximum: number,
): number {
  if (typeof raw !== "string" || raw.trim() === "") return fallback;
  const candidate = Number(raw);
  return Number.isFinite(candidate) && Number.isInteger(candidate)
    && candidate >= minimum && candidate <= maximum
    ? candidate
    : fallback;
}

/** Read each policy value once and fall back safely when an env value is invalid. */
export function getBrowserSessionMetricsPolicy(
  env: Record<string, unknown> = import.meta.env,
): BrowserSessionMetricsPolicy {
  const defaults = DEFAULT_BROWSER_SESSION_METRICS_POLICY;
  const rateWindowMs = validatedNumber(
    env.VITE_HEALTH_RATE_WINDOW_MS,
    defaults.rateWindowMs,
    1_000,
    60_000,
  );
  const retentionBuckets = validatedNumber(
    env.VITE_HEALTH_SAMPLE_RETENTION_BUCKETS,
    defaults.retentionBuckets,
    1,
    MAXIMUM_RETENTION_BUCKETS,
  );
  const maximumRateEvents = validatedNumber(
    env.VITE_HEALTH_MAX_RATE_EVENTS,
    defaults.maximumRateEvents,
    100,
    MAXIMUM_RATE_EVENTS,
  );
  const sampleIntervalMs = validatedNumber(
    env.VITE_HEALTH_SAMPLE_INTERVAL_MS,
    defaults.sampleIntervalMs,
    250,
    5_000,
  );

  return {
    rateWindowMs,
    sampleIntervalMs: Math.min(sampleIntervalMs, rateWindowMs),
    retentionBuckets,
    maximumRateEvents,
  };
}

export interface SourceFrameIdentity {
  engineInstanceId: string;
  sessionId: string | number;
  frameId: string | number;
}

export interface ObservationAges {
  ballAgeMs: number | null;
  cueAgeMs: number | null;
  motionAgeMs: number | null;
  ballEvidenceId: string | null;
  cueEvidenceId: string | null;
  motionEvidenceId: string | null;
}

export interface BrowserSessionSample {
  /** Wall-clock label for the sampled bucket; elapsed calculations use monotonic time. */
  sampledAt: string;
  monotonicMs: number;
  receivedStateMessagesPerSecond: number | null;
  distinctObservedSourceFramesPerSecond: number | null;
  ballObservationAgeMs: number | null;
  cueObservationAgeMs: number | null;
  motionObservationAgeMs: number | null;
  gapReason?: "disconnected" | "suspended" | "producer_changed";
  frameRateUnknown?: "missing_identity" | "retention_limit";
}

interface TimedKey {
  key: string;
  at: number;
}

interface AgeClock {
  ageAtReceiptMs: number;
  receivedAtMs: number;
  evidenceId: string | null;
}

type AgeName = "ball" | "cue" | "motion";

function validAge(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) && value >= 0
    ? value
    : null;
}

function compositeEvidenceIdentity(value: unknown): string | null {
  if (!value || typeof value !== "object") return null;
  const identity = value as Record<string, unknown>;
  const engine = identity.engine_instance_id;
  const session = identity.session_id;
  const frame = identity.frame_id ?? identity.frame_number;
  if (
    typeof engine !== "string" || engine.length === 0
    || (typeof session !== "string" && typeof session !== "number")
    || (typeof frame !== "string" && typeof frame !== "number")
  ) return null;
  return JSON.stringify([engine, session, frame]);
}

/** Extract only observation ages and evidence identities present in the state API. */
export function extractObservationAges(state: GameStateData): ObservationAges {
  const status = state.observation_status;
  const perception = status?.perception;
  const observedBall = state.balls.find((ball) =>
    ball.is_observed === true || ball.position_kind === "measured",
  );
  const ballEvidenceId = compositeEvidenceIdentity(
    observedBall?.source_observation,
  ) ?? (
    status?.ball_positions_status === "observed"
      ? compositeEvidenceIdentity({
        engine_instance_id:
          status.observation_identity?.engine_instance_id
          ?? status.engine_instance_id
          ?? status.vision_instance_id,
        session_id:
          status.observation_identity?.session_id
          ?? status.source_session_id
          ?? status.source_process_id,
        frame_id:
          status.observation_identity?.frame_id ?? status.source_frame_id,
      })
      : null
  );
  const cueEvidenceId = perception?.cue?.source_frame_id
    ? compositeEvidenceIdentity({
      engine_instance_id:
        status?.observation_identity?.engine_instance_id
        ?? status?.engine_instance_id
        ?? status?.vision_instance_id,
      session_id:
        status?.observation_identity?.session_id
        ?? status?.source_session_id
        ?? status?.source_process_id,
      frame_id: perception.cue.source_frame_id,
    })
    : null;
  const motionSourceFrameId =
    perception?.table_visibility?.observation_frame_id
    ?? perception?.decision_frame_id;
  const motionEvidenceId = typeof motionSourceFrameId === "string"
    ? compositeEvidenceIdentity({
      engine_instance_id:
        status?.observation_identity?.engine_instance_id
        ?? status?.engine_instance_id
        ?? status?.vision_instance_id,
      session_id:
        status?.observation_identity?.session_id
        ?? status?.source_session_id
        ?? status?.source_process_id,
      frame_id: motionSourceFrameId,
    })
    : null;

  return {
    ballAgeMs: validAge(perception?.last_ball_observation_age_ms),
    cueAgeMs: validAge(
      perception?.cue_evidence_age_ms ?? perception?.cue?.observation_age_ms,
    ),
    motionAgeMs: validAge(perception?.latest_motion_check_age_ms),
    ballEvidenceId,
    cueEvidenceId,
    motionEvidenceId,
  };
}

function validFrameIdentity(value: SourceFrameIdentity | null): value is SourceFrameIdentity {
  return value !== null
    && value.engineInstanceId.length > 0
    && (typeof value.sessionId === "number"
      ? Number.isFinite(value.sessionId)
      : value.sessionId.length > 0)
    && (typeof value.frameId === "number"
      ? Number.isFinite(value.frameId)
      : value.frameId.length > 0);
}

function sourceEpoch(identity: SourceFrameIdentity): string {
  return JSON.stringify([identity.engineInstanceId, identity.sessionId]);
}

function frameKey(identity: SourceFrameIdentity): string {
  return JSON.stringify([
    identity.engineInstanceId,
    identity.sessionId,
    identity.frameId,
  ]);
}

/**
 * Browser-only, bounded session telemetry. Receipt events are supplied by the
 * existing state WebSocket; `sample` only aggregates already observed values.
 */
export class BrowserSessionMetrics {
  readonly policy: Readonly<BrowserSessionMetricsPolicy>;
  private connected = false;
  private connectedAtMs: number | null = null;
  private firstSourceFrameAtMs: number | null = null;
  private lastSampleAtMs: number | null = null;
  private producerEpoch: string | null = null;
  private producerChangedAtMs: number | null = null;
  private lastMissingIdentityAtMs: number | null = null;
  private messageReceipts: number[] = [];
  private frameReceipts: TimedKey[] = [];
  private seenFrameKeys = new Map<string, number>();
  private rateHistoryOverflowAtMs: number | null = null;
  private frameIdentityOverflowAtMs: number | null = null;
  private readonly ages: Record<AgeName, AgeClock | null> = {
    ball: null,
    cue: null,
    motion: null,
  };

  constructor(policy = getBrowserSessionMetricsPolicy()) {
    this.policy = Object.freeze({ ...policy });
  }

  setConnected(connected: boolean, nowMs: number): void {
    if (connected === this.connected) return;
    this.connected = connected;
    this.lastSampleAtMs = nowMs;
    if (connected) {
      this.connectedAtMs = nowMs;
      this.firstSourceFrameAtMs = null;
      return;
    }
    this.connectedAtMs = null;
    this.firstSourceFrameAtMs = null;
    this.messageReceipts = [];
    this.frameReceipts = [];
    this.seenFrameKeys.clear();
    this.lastMissingIdentityAtMs = null;
    this.rateHistoryOverflowAtMs = null;
    this.frameIdentityOverflowAtMs = null;
    this.producerChangedAtMs = null;
    this.resetAges();
  }

  recordStateMessageReceipt(nowMs: number): void {
    if (!this.connected) return;
    this.prune(nowMs);
    if (this.messageReceipts.length >= this.policy.maximumRateEvents) {
      this.messageReceipts.shift();
      this.rateHistoryOverflowAtMs = nowMs;
    }
    this.messageReceipts.push(nowMs);
  }

  /** Call after ordering logic has accepted a new source observation. */
  recordAcceptedSourceFrame(
    identity: SourceFrameIdentity | null,
    nowMs: number,
  ): void {
    if (!this.connected) return;
    this.prune(nowMs);
    if (!validFrameIdentity(identity)) {
      this.lastMissingIdentityAtMs = nowMs;
      return;
    }
    const epoch = sourceEpoch(identity);
    if (this.producerEpoch !== epoch) {
      const hadPreviousEpoch = this.producerEpoch !== null;
      this.producerEpoch = epoch;
      if (hadPreviousEpoch) this.producerChangedAtMs = nowMs;
      this.firstSourceFrameAtMs = null;
      this.frameReceipts = [];
      this.seenFrameKeys.clear();
      this.lastMissingIdentityAtMs = null;
      this.frameIdentityOverflowAtMs = null;
      this.resetAges();
    }

    const key = frameKey(identity);
    if (this.seenFrameKeys.has(key)) return;
    if (this.seenFrameKeys.size >= this.policy.maximumRateEvents) {
      const oldest = this.seenFrameKeys.keys().next().value as string | undefined;
      if (oldest !== undefined) this.seenFrameKeys.delete(oldest);
      this.frameIdentityOverflowAtMs = nowMs;
    }
    if (this.frameReceipts.length >= this.policy.maximumRateEvents) {
      this.frameReceipts.shift();
      this.frameIdentityOverflowAtMs = nowMs;
    }
    this.seenFrameKeys.set(key, nowMs);
    this.frameReceipts.push({ key, at: nowMs });
    this.firstSourceFrameAtMs ??= nowMs;
  }

  /** A producer/session replacement creates a chart discontinuity for ages. */
  recordProducerEpoch(epoch: string | null, nowMs: number): void {
    if (!epoch || epoch === this.producerEpoch) return;
    const hadPreviousEpoch = this.producerEpoch !== null;
    this.producerEpoch = epoch;
    if (hadPreviousEpoch) this.producerChangedAtMs = nowMs;
    this.firstSourceFrameAtMs = null;
    this.frameReceipts = [];
    this.seenFrameKeys.clear();
    this.lastMissingIdentityAtMs = null;
    this.frameIdentityOverflowAtMs = null;
    this.resetAges();
  }

  recordObservationAges(observation: ObservationAges, nowMs: number): void {
    if (!this.connected) return;
    this.updateAge("ball", observation.ballAgeMs, observation.ballEvidenceId, nowMs);
    this.updateAge("cue", observation.cueAgeMs, observation.cueEvidenceId, nowMs);
    this.updateAge("motion", observation.motionAgeMs, observation.motionEvidenceId, nowMs);
  }

  sample(nowMs: number, wallTime: Date): BrowserSessionSample {
    this.prune(nowMs);
    const base = {
      sampledAt: wallTime.toISOString(),
      monotonicMs: nowMs,
      receivedStateMessagesPerSecond: null,
      distinctObservedSourceFramesPerSecond: null,
      ballObservationAgeMs: null,
      cueObservationAgeMs: null,
      motionObservationAgeMs: null,
    } satisfies Omit<BrowserSessionSample, "gapReason" | "frameRateUnknown">;

    if (!this.connected) {
      this.lastSampleAtMs = nowMs;
      return { ...base, gapReason: "disconnected" };
    }
    if (
      this.lastSampleAtMs !== null
      && nowMs - this.lastSampleAtMs > this.policy.sampleIntervalMs * 2
    ) {
      this.lastSampleAtMs = nowMs;
      this.resetRateWindowAfterSuspension(nowMs);
      this.producerChangedAtMs = null;
      return { ...base, gapReason: "suspended" };
    }
    this.lastSampleAtMs = nowMs;
    if (this.producerChangedAtMs !== null) {
      this.producerChangedAtMs = null;
      return { ...base, gapReason: "producer_changed" };
    }

    const messageDuration = this.observedDuration(nowMs, this.connectedAtMs);
    const frameDuration = this.observedDuration(nowMs, this.firstSourceFrameAtMs);
    const messageRate = this.rateHistoryOverflowAtMs !== null
      && nowMs - this.rateHistoryOverflowAtMs < this.policy.rateWindowMs
      ? null
      : this.rate(this.messageReceipts, nowMs, messageDuration);
    const hasMissingIdentity = this.lastMissingIdentityAtMs !== null
      && nowMs - this.lastMissingIdentityAtMs <= this.policy.rateWindowMs;
    const frameIdentityOverflow = this.frameIdentityOverflowAtMs !== null
      && nowMs - this.frameIdentityOverflowAtMs < this.policy.rateWindowMs;
    const frameRateUnknown = hasMissingIdentity
      ? "missing_identity"
      : frameIdentityOverflow
        ? "retention_limit"
        : undefined;
    const frameRate = this.firstSourceFrameAtMs === null
      || frameRateUnknown
      ? null
      : this.rate(this.frameReceipts.map((receipt) => receipt.at), nowMs, frameDuration);

    return {
      ...base,
      receivedStateMessagesPerSecond: messageRate,
      distinctObservedSourceFramesPerSecond: frameRate,
      ballObservationAgeMs: this.currentAge("ball", nowMs),
      cueObservationAgeMs: this.currentAge("cue", nowMs),
      motionObservationAgeMs: this.currentAge("motion", nowMs),
      ...(frameRateUnknown ? { frameRateUnknown } : {}),
    };
  }

  private observedDuration(nowMs: number, startedAtMs: number | null): number {
    if (startedAtMs === null) return 0;
    return Math.min(this.policy.rateWindowMs, Math.max(0, nowMs - startedAtMs));
  }

  private rate(receipts: number[], nowMs: number, durationMs: number): number | null {
    if (durationMs <= 0) return null;
    const windowStartMs = nowMs - Math.min(this.policy.rateWindowMs, durationMs);
    const count = receipts.reduce(
      (total, receivedAt) => total + Number(receivedAt >= windowStartMs && receivedAt <= nowMs),
      0,
    );
    return count / (Math.min(this.policy.rateWindowMs, durationMs) / 1_000);
  }

  private prune(nowMs: number): void {
    const historyWindowMs = Math.max(
      this.policy.rateWindowMs,
      this.policy.sampleIntervalMs * (this.policy.retentionBuckets + 1),
    );
    const oldestRetainedMs = nowMs - historyWindowMs;
    this.messageReceipts = this.messageReceipts.filter((at) => at >= nowMs - this.policy.rateWindowMs);
    this.frameReceipts = this.frameReceipts.filter((receipt) =>
      receipt.at >= nowMs - this.policy.rateWindowMs,
    );
    if (
      this.lastMissingIdentityAtMs !== null
      && this.lastMissingIdentityAtMs < nowMs - this.policy.rateWindowMs
    ) this.lastMissingIdentityAtMs = null;
    for (const [key, at] of this.seenFrameKeys) {
      if (at < oldestRetainedMs) this.seenFrameKeys.delete(key);
    }
  }

  private resetRateWindowAfterSuspension(nowMs: number): void {
    // The browser did not observe the interval between the previous sample
    // and this wakeup. Begin new rate windows now; do not include receipts or
    // identities whose timing could have crossed the suspended interval.
    this.connectedAtMs = nowMs;
    this.messageReceipts = this.messageReceipts.filter((at) => at >= nowMs);
    this.frameReceipts = this.frameReceipts.filter((receipt) => receipt.at >= nowMs);
    this.firstSourceFrameAtMs = this.frameReceipts[0]?.at ?? null;
    this.seenFrameKeys = new Map(
      [...this.seenFrameKeys].filter(([, at]) => at >= nowMs),
    );
    if (this.lastMissingIdentityAtMs !== null && this.lastMissingIdentityAtMs < nowMs)
      this.lastMissingIdentityAtMs = null;
    this.rateHistoryOverflowAtMs = null;
    this.frameIdentityOverflowAtMs = null;
  }

  private updateAge(name: AgeName, rawAgeMs: number | null, evidenceId: string | null, nowMs: number): void {
    if (rawAgeMs === null) {
      this.ages[name] = null;
      return;
    }
    const previous = this.ages[name];
    const currentAge = previous
      ? advanceEvidenceAgeMs(
        rawAgeMs,
        previous.ageAtReceiptMs,
        previous.receivedAtMs,
        nowMs,
      )
      : null;
    const identityChanged = evidenceId !== null
      && previous !== null
      && previous.evidenceId !== evidenceId;
    // Without an API evidence/source-frame identity, keep age monotonic. A
    // lower reported age alone cannot prove that repeated state renewed its
    // underlying evidence.
    const ageAtReceiptMs = !previous || identityChanged
      ? rawAgeMs
      : Math.max(rawAgeMs, currentAge ?? rawAgeMs);
    this.ages[name] = {
      ageAtReceiptMs,
      receivedAtMs: nowMs,
      evidenceId: evidenceId ?? previous?.evidenceId ?? null,
    };
  }

  private currentAge(name: AgeName, nowMs: number): number | null {
    const evidence = this.ages[name];
    return evidence
      ? advanceEvidenceAgeMs(
        evidence.ageAtReceiptMs,
        evidence.ageAtReceiptMs,
        evidence.receivedAtMs,
        nowMs,
      )
      : null;
  }

  private resetAges(): void {
    this.ages.ball = null;
    this.ages.cue = null;
    this.ages.motion = null;
  }
}
