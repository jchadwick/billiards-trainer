import { describe, expect, it } from "vitest";
import { BrowserStateChangeObserver } from "./state-change-observer";
import type { GameStateData } from "../../types/api";

function state(overrides: {
  phase?: string | null;
  session?: string;
  configurationRevision?: number | null;
  ballAge?: number | null;
  ballLimit?: number;
  cueAge?: number | null;
  motionAge?: number | null;
  cueFrame?: string | null;
  ballFrame?: string | null;
  motionFrame?: string | null;
} = {}): GameStateData {
  const {
    phase = "pre_shot",
    session = "session-a",
    configurationRevision = 4,
    ballAge = 1,
    ballLimit = 200,
    cueAge = 10,
    motionAge = 20,
    cueFrame = "cue-frame-1",
    ballFrame = "ball-frame-1",
    motionFrame = "motion-frame-1",
  } = overrides;
  return {
    balls: ballFrame === null
      ? []
      : [{
        id: "cue-ball",
        is_observed: true,
        confidence: 1,
        position: { x: 0.25, y: 0.5, scale: [1, 1] },
        radius: 1,
        source_observation: {
          engine_instance_id: "engine-a",
          session_id: session,
          frame_id: ballFrame,
        },
      }],
    observation_status: {
      observation_identity: {
        engine_instance_id: "engine-a",
        session_id: session,
        frame_id: "source-frame-1",
        configuration_revision: configurationRevision ?? undefined,
      },
      ball_positions_status: "observed",
      ...(phase === null ? {} : {
        perception: {
          event_sequence: 1,
          phase: phase as "pre_shot",
          layout_revision: 1,
          layout_trusted: true,
          guidance_valid: true,
          trust_reason: "test evidence",
          last_ball_observation_age_ms: ballAge,
          layout_verification_max_age_ms: ballLimit,
          cue_evidence_age_ms: cueAge,
          cue_evidence_max_age_ms: 150,
          latest_motion_check_age_ms: motionAge,
          motion_evidence_max_age_ms: 250,
          cue: cueFrame === null ? null : {
            shaft_start_px: [0, 0],
            shaft_end_px: [1, 1],
            axis_unit: [1, 1],
            confidence: 1,
            observation_age_ms: cueAge ?? 0,
            source_frame_id: cueFrame,
          },
          table_visibility: motionFrame === null ? null : {
            status: "visible",
            reason: "test evidence",
            observation_frame_id: motionFrame,
            observation_age_ms: motionAge ?? 0,
          },
        },
      }),
    },
  };
}

describe("browser-observed state changes", () => {
  it("uses initial values as baselines, ignores repeats, and labels changes after missing values or a reconnect gap", () => {
    const observer = new BrowserStateChangeObserver();
    expect(observer.observeAcceptedState(state(), 0, 3)).toEqual([]);
    expect(observer.observeAcceptedState(state(), 20, 3)).toEqual([]);

    const first = observer.observeAcceptedState(state({
      phase: "aiming",
      configurationRevision: 5,
    }), 40, 3);
    expect(first.map((event) => event.type)).toEqual(["Phase", "Revision"]);
    expect(first.every((event) => (event.payload as any).across_gap === false)).toBe(true);
    expect(observer.observeAcceptedState(state({ phase: "aiming", configurationRevision: 5 }), 60, 3)).toEqual([]);

    observer.observeAcceptedState(state({ phase: null, configurationRevision: null }), 80, 3);
    observer.markDeliveryGap();
    const resumed = observer.observeAcceptedState(state({
      phase: "shot_underway",
      session: "session-b",
      configurationRevision: 6,
    }), 100, 3);
    expect(resumed.map((event) => event.type)).toEqual(["Phase", "Session", "Revision"]);
    expect(resumed.every((event) => (event.payload as any).across_gap === true)).toBe(true);
  });

  it("emits freshness deadlines once with reported and browser-estimated ages", () => {
    const observer = new BrowserStateChangeObserver();
    expect(observer.observeAcceptedState(state(), 0, 3)).toEqual([]);
    expect(observer.nextFreshnessDeadline(0)).toBe(140);

    const expired = observer.evaluateFreshness(140);
    expect(expired).toHaveLength(1);
    expect(expired[0].message).toContain("Cue evidence exceeded");
    expect(expired[0].payload).toMatchObject({
      evidence: "cue",
      previous_status: "fresh",
      current_status: "stale",
      reported_age_ms: 10,
      browser_estimated_age_ms: 150,
      maximum_age_ms: 150,
      age_origin: "Browser estimated",
    });
    expect(observer.evaluateFreshness(140)).toEqual([]);
  });

  it("requires a new evidence identity to recover and marks absent age unknown", () => {
    const observer = new BrowserStateChangeObserver();
    observer.observeAcceptedState(state({ ballLimit: 1_000 }), 0, 3);
    const cueStale = observer.evaluateFreshness(140);
    expect(cueStale).toHaveLength(1);

    // Re-publication of the same cue frame cannot renew or recover it.
    expect(observer.observeAcceptedState(state({ cueAge: 1, ballLimit: 1_000 }), 200, 3)).toEqual([]);
    const recovered = observer.observeAcceptedState(state({ cueAge: 2, cueFrame: "cue-frame-2", ballLimit: 1_000 }), 210, 3);
    expect(recovered).toHaveLength(1);
    expect(recovered[0].payload).toMatchObject({
      evidence: "cue",
      previous_status: "stale",
      current_status: "fresh",
      previous_evidence_identity: JSON.stringify(["engine-a", "session-a", "cue-frame-1"]),
      evidence_identity: JSON.stringify(["engine-a", "session-a", "cue-frame-2"]),
    });

    const unknown = observer.observeAcceptedState(state({ cueAge: null, cueFrame: null, ballLimit: 1_000 }), 220, 3);
    expect(unknown).toHaveLength(1);
    expect(unknown[0].payload).toMatchObject({
      evidence: "cue",
      previous_status: "fresh",
      current_status: "unknown",
      reason: "evidence_missing",
    });
  });
});
