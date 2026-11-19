import {
  extractObservationAges,
  type ObservationAges,
} from "../../services/browser-session-metrics";
import type { GameStateData, ObservationStatus } from "../../types/api";
import {
  advanceEvidenceAgeMs,
  observationIsStale,
  positiveFreshnessLimit,
} from "../video/freshness";
import type { BrowserEvent } from "./event-model";

export type BrowserEventDraft = Omit<BrowserEvent, "id" | "timestamp">;
type FreshnessStatus = "unknown" | "fresh" | "stale";
type EvidenceName = "ball" | "cue" | "motion";

interface EvidenceTrack {
  status: FreshnessStatus;
  evidenceId: string | null;
  ageAtReceiptMs: number | null;
  reportedAgeMs: number | null;
  receivedAtMs: number;
  maximumAgeMs: number | null;
  hardStale: boolean;
  staleReason: string | null;
  acrossGap: boolean;
}

interface EvidenceInput {
  name: EvidenceName;
  ageMs: number | null;
  evidenceId: string | null;
  maximumAgeMs: number | null;
  hardStale: boolean;
  staleReason: string | null;
}

function scalar(value: unknown): string | number | undefined {
  return typeof value === "string" || (typeof value === "number" && Number.isFinite(value))
    ? value
    : undefined;
}

function sourceSessionIdentity(status?: ObservationStatus): string | null {
  const identity = status?.observation_identity;
  const engine = identity?.engine_instance_id ?? status?.engine_instance_id ?? status?.vision_instance_id;
  const session = identity?.session_id ?? status?.source_session_id ?? status?.source_process_id;
  if (
    typeof engine !== "string" || engine.length === 0
    || (typeof session !== "string" && typeof session !== "number")
  ) return null;
  return JSON.stringify([engine, session]);
}

function evidenceInputs(
  state: GameStateData,
  maximumObservationAgeSeconds: number,
  ages: ObservationAges,
): EvidenceInput[] {
  const status = state.observation_status;
  const perception = status?.perception;
  const ballMaximumAgeMs = typeof perception?.layout_verification_max_age_ms === "number"
    && Number.isFinite(perception.layout_verification_max_age_ms)
    && perception.layout_verification_max_age_ms > 0
    ? perception.layout_verification_max_age_ms
    : positiveFreshnessLimit(maximumObservationAgeSeconds) * 1_000;
  const cueMaximumAgeMs = validMaximum(perception?.cue_evidence_max_age_ms);
  const motionMaximumAgeMs = validMaximum(perception?.motion_evidence_max_age_ms);

  const ballHardStale = observationIsStale(
    ages.ballAgeMs === null ? null : ages.ballAgeMs / 1_000,
    ballMaximumAgeMs / 1_000,
    status,
  );
  const cueAgeMs = finiteAge(perception?.cue_evidence_age_ms ?? perception?.cue?.observation_age_ms);
  const motionAgeMs = finiteAge(perception?.latest_motion_check_age_ms);
  const inputs: EvidenceInput[] = [
    {
      name: "ball",
      ageMs: ages.ballAgeMs,
      evidenceId: ages.ballEvidenceId,
      maximumAgeMs: ballMaximumAgeMs,
      hardStale: ballHardStale,
      staleReason: ballHardStale ? ballStatusReason(status) : null,
    },
    {
      name: "cue",
      ageMs: cueAgeMs,
      evidenceId: ages.cueEvidenceId,
      maximumAgeMs: cueMaximumAgeMs,
      hardStale: false,
      staleReason: null,
    },
    {
      name: "motion",
      ageMs: motionAgeMs,
      evidenceId: ages.motionEvidenceId,
      maximumAgeMs: motionMaximumAgeMs,
      hardStale: false,
      staleReason: null,
    },
  ];
  return inputs;
}

function validMaximum(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) && value > 0
    ? value
    : null;
}

function finiteAge(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) && value >= 0
    ? value
    : null;
}

function ballStatusReason(status?: ObservationStatus): string {
  if (status?.freshness_status === "source_silent") return "source_silent";
  if (status?.ball_positions_status === "cached") return "ball_positions_cached";
  if (status?.ball_positions_status === "unavailable") return "ball_positions_unavailable";
  if (status?.ball_detection_status === "failed") return "ball_detection_failed";
  return "age_limit_exceeded";
}

function valueChange(
  component: string,
  type: string,
  label: string,
  previous: string | number,
  current: string | number,
  acrossGap: boolean,
): BrowserEventDraft {
  return {
    severity: "Info",
    component,
    type,
    origin: "Browser observed",
    message: `${label} changed: ${String(previous)} → ${String(current)}${acrossGap ? " across a delivery gap" : ""}`,
    payload: {
      previous,
      current,
      across_gap: acrossGap,
    },
  };
}

/**
 * Tracks delivered state transitions and independently aging evidence. It
 * never reconstructs intermediate backend events that the latest-state socket
 * may have coalesced.
 */
export class BrowserStateChangeObserver {
  private readonly values: Partial<Record<"phase" | "session" | "configuration", string | number>> = {};
  private readonly valueGaps: Record<"phase" | "session" | "configuration", boolean> = {
    phase: false,
    session: false,
    configuration: false,
  };
  private pendingDeliveryGap = false;
  private readonly evidence: Record<EvidenceName, EvidenceTrack | null> = {
    ball: null,
    cue: null,
    motion: null,
  };

  markDeliveryGap(): void {
    this.pendingDeliveryGap = true;
    for (const item of Object.values(this.evidence)) {
      if (item) item.acrossGap = true;
    }
  }

  observeAcceptedState(
    state: GameStateData,
    nowMs: number,
    maximumObservationAgeSeconds: number,
  ): BrowserEventDraft[] {
    const events: BrowserEventDraft[] = [];
    const status = state.observation_status;
    const phase = status?.perception?.phase;
    const session = sourceSessionIdentity(status);
    const revision = scalar(
      status?.configuration_revision ?? status?.observation_identity?.configuration_revision,
    );

    this.observeValue("phase", phase, "Perception", "Phase", "Perception phase", events);
    this.observeValue("session", session ?? undefined, "Capture", "Session", "Capture session identity", events);
    this.observeValue("configuration", revision, "Configuration", "Revision", "Configuration revision", events);

    const ages = extractObservationAges(state);
    for (const input of evidenceInputs(state, maximumObservationAgeSeconds, ages)) {
      this.updateEvidence(input, nowMs, events);
    }
    this.pendingDeliveryGap = false;
    return events;
  }

  evaluateFreshness(nowMs: number): BrowserEventDraft[] {
    const events: BrowserEventDraft[] = [];
    for (const name of ["ball", "cue", "motion"] as const) {
      const track = this.evidence[name];
      if (!track) continue;
      const ageMs = this.currentAge(track, nowMs);
      const status = this.statusFor(ageMs, track.maximumAgeMs, track.hardStale);
      this.transitionEvidence(name, track, status, ageMs, nowMs, events, null, track.evidenceId);
    }
    return events;
  }

  nextFreshnessDeadline(nowMs: number): number | null {
    let next: number | null = null;
    for (const track of Object.values(this.evidence)) {
      if (!track || track.status !== "fresh" || track.ageAtReceiptMs === null || track.maximumAgeMs === null) continue;
      const deadline = track.receivedAtMs + Math.max(0, track.maximumAgeMs - track.ageAtReceiptMs);
      if (deadline <= nowMs) return nowMs;
      if (next === null || deadline < next) next = deadline;
    }
    return next;
  }

  private observeValue(
    key: "phase" | "session" | "configuration",
    value: string | number | undefined,
    component: string,
    type: string,
    label: string,
    events: BrowserEventDraft[],
  ): void {
    if (value === undefined) {
      this.valueGaps[key] = true;
      return;
    }
    const previous = this.values[key];
    if (previous !== undefined && previous !== value) {
      events.push(valueChange(
        component,
        type,
        label,
        previous,
        value,
        this.pendingDeliveryGap || this.valueGaps[key],
      ));
    }
    this.values[key] = value;
    this.valueGaps[key] = false;
  }

  private updateEvidence(input: EvidenceInput, nowMs: number, events: BrowserEventDraft[]): void {
    const previous = this.evidence[input.name];
    const evidenceChanged = previous !== null
      && input.evidenceId !== null
      && input.evidenceId !== previous.evidenceId;
    const ageAtReceiptMs = input.ageMs === null
      ? null
      : previous && !evidenceChanged && previous.ageAtReceiptMs !== null
        ? advanceEvidenceAgeMs(input.ageMs, previous.ageAtReceiptMs, previous.receivedAtMs, nowMs)
        : input.ageMs;
    const status = this.statusFor(ageAtReceiptMs, input.maximumAgeMs, input.hardStale);
    const track: EvidenceTrack = {
      status: previous?.status ?? "unknown",
      evidenceId: input.evidenceId ?? previous?.evidenceId ?? null,
      ageAtReceiptMs,
      reportedAgeMs: input.ageMs,
      receivedAtMs: nowMs,
      maximumAgeMs: input.maximumAgeMs,
      hardStale: input.hardStale,
      staleReason: input.staleReason,
      acrossGap: this.pendingDeliveryGap || (previous?.acrossGap ?? false),
    };
    if (previous) {
      this.transitionEvidence(
        input.name,
        track,
        status,
        ageAtReceiptMs,
        nowMs,
        events,
        evidenceChanged ? input.evidenceId : null,
        previous?.evidenceId ?? null,
      );
    } else {
      // The first value is a baseline, not a fictional stale/recovered event.
      track.status = status;
    }
    this.evidence[input.name] = track;
  }

  private transitionEvidence(
    name: EvidenceName,
    track: EvidenceTrack,
    currentStatus: FreshnessStatus,
    ageMs: number | null,
    nowMs: number,
    events: BrowserEventDraft[],
    changedEvidenceId: string | null,
    previousEvidenceId: string | null,
  ): void {
    const previousStatus = track.status;
    // A recovery needs a newly identified observation. A numeric age reset
    // without new evidence cannot make an old measurement look recovered.
    const nextStatus = previousStatus === "stale" && currentStatus === "fresh"
      && changedEvidenceId === null
      ? "unknown"
      : currentStatus;
    if (previousStatus !== nextStatus) {
      const label = name === "ball" ? "Ball observations" : name === "cue" ? "Cue evidence" : "Motion evidence";
      const acrossGap = this.pendingDeliveryGap || track.acrossGap;
      if (nextStatus === "stale") {
        events.push({
          severity: "Warning",
          component: "Perception",
          type: "Freshness",
          origin: "Browser observed",
          message: `${label} exceeded the reported freshness limit${acrossGap ? " across a delivery gap" : ""}`,
          payload: {
            evidence: name,
            previous_status: previousStatus,
            current_status: "stale",
            evidence_identity: track.evidenceId,
            reported_age_ms: track.reportedAgeMs,
            browser_estimated_age_ms: ageMs,
            maximum_age_ms: track.maximumAgeMs,
            reason: track.staleReason ?? "age_limit_exceeded",
            age_origin: ageMs !== null && track.reportedAgeMs !== null && ageMs > track.reportedAgeMs
              ? "Browser estimated"
              : "API reported",
            across_gap: acrossGap,
          },
        });
      } else if (nextStatus === "fresh" && previousStatus === "stale") {
        events.push({
          severity: "Info",
          component: "Perception",
          type: "Freshness",
          origin: "Browser observed",
          message: `${label} recovered on a new observation${acrossGap ? " across a delivery gap" : ""}`,
          payload: {
            evidence: name,
            previous_status: "stale",
            current_status: "fresh",
            previous_evidence_identity: previousEvidenceId,
            evidence_identity: changedEvidenceId,
            reported_age_ms: track.reportedAgeMs,
            browser_estimated_age_ms: ageMs,
            maximum_age_ms: track.maximumAgeMs,
            age_origin: "API reported",
            across_gap: acrossGap,
          },
        });
      } else if (nextStatus === "unknown" && previousStatus !== "unknown") {
        events.push({
          severity: "Info",
          component: "Perception",
          type: "Freshness",
          origin: "Browser observed",
          message: `${label} freshness is unknown${acrossGap ? " across a delivery gap" : ""}`,
          payload: {
            evidence: name,
            previous_status: previousStatus,
            current_status: "unknown",
            evidence_identity: track.evidenceId,
            reported_age_ms: track.reportedAgeMs,
            browser_estimated_age_ms: ageMs,
            maximum_age_ms: track.maximumAgeMs,
            reason: track.ageAtReceiptMs === null ? "evidence_missing" : "new_evidence_identity_unavailable",
            across_gap: acrossGap,
          },
        });
      }
    }
    track.status = nextStatus;
    track.ageAtReceiptMs = ageMs;
    track.receivedAtMs = nowMs;
    if (changedEvidenceId !== null) {
      track.evidenceId = changedEvidenceId;
      track.acrossGap = this.pendingDeliveryGap;
    }
  }

  private currentAge(track: EvidenceTrack, nowMs: number): number | null {
    if (track.ageAtReceiptMs === null) return null;
    return advanceEvidenceAgeMs(
      track.ageAtReceiptMs,
      track.ageAtReceiptMs,
      track.receivedAtMs,
      nowMs,
    );
  }

  private statusFor(ageMs: number | null, maximumAgeMs: number | null, hardStale: boolean): FreshnessStatus {
    if (hardStale) return "stale";
    if (ageMs === null || maximumAgeMs === null) return "unknown";
    return ageMs >= maximumAgeMs ? "stale" : "fresh";
  }
}
