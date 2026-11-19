import { afterEach, describe, expect, it, vi } from "vitest";
import { runInAction } from "mobx";
import type {
  ConnectionStateHandler,
  MessageHandler,
  WebSocketClient,
} from "../services/websocket-client";
import { RootStore } from "./RootStore";
import { createNoopDiagnosticsTransport } from "../tests/diagnostics-test-helpers";

function createTransport() {
  const handlers = new Map<string, MessageHandler>();
  let connectionHandler!: ConnectionStateHandler;
  const transport = {
    onConnectionState: (handler: ConnectionStateHandler) => { connectionHandler = handler; },
    on: (type: string, handler: MessageHandler) => handlers.set(type, handler),
    subscribe: vi.fn(),
    connect: vi.fn().mockResolvedValue(undefined),
    destroy: vi.fn(),
  };
  return {
    client: transport as unknown as WebSocketClient,
    handlers,
    connection: (state: Parameters<ConnectionStateHandler>[0]) => connectionHandler(state),
  };
}

function stateMessage(input: {
  frame: number;
  session?: string;
  phase?: "pre_shot" | "aiming" | "shot_underway";
  eventSequence?: number;
  configurationRevision?: number;
}) {
  const {
    frame,
    session = "capture-a",
    phase = "pre_shot",
    eventSequence = frame,
    configurationRevision = 1,
  } = input;
  return {
    type: "state",
    timestamp: new Date(0).toISOString(),
    data: {
      balls: [],
      observation_status: {
        observation_identity: {
          engine_instance_id: "engine-a",
          session_id: session,
          frame_id: `frame-${frame}`,
          configuration_revision: configurationRevision,
        },
        transport_sequence: frame,
        perception: {
          event_sequence: eventSequence,
          phase,
          layout_revision: frame,
          layout_trusted: true,
          guidance_valid: true,
          trust_reason: "measured",
        },
      },
    },
  };
}

afterEach(() => vi.useRealTimers());

describe("RootStore browser-observed events", () => {
  it("records accepted phase/session/revision changes, suppresses repeats and rejects old snapshots", async () => {
    const transport = createTransport();
    const store = new RootStore(transport.client, undefined, createNoopDiagnosticsTransport());
    const publish = (message: ReturnType<typeof stateMessage>) => transport.handlers.get("state")!(message);
    try {
      transport.connection("connected");
      publish(stateMessage({ frame: 1, phase: "pre_shot", eventSequence: 1 }));
      expect(store.events.filter((event) => event.component !== "State delivery")).toEqual([]);

      publish(stateMessage({ frame: 1, phase: "pre_shot", eventSequence: 1 }));
      publish(stateMessage({ frame: 1, phase: "aiming", eventSequence: 2 }));
      publish(stateMessage({ frame: 2, phase: "aiming", eventSequence: 2, configurationRevision: 2 }));
      publish(stateMessage({ frame: 1, phase: "pre_shot", eventSequence: 1, configurationRevision: 1 }));
      let changes = store.events.filter((event) => event.component !== "State delivery");
      expect(changes.map((event) => event.type)).toEqual(["Revision", "Phase"]);
      expect(changes[0].payload).toMatchObject({ previous: 1, current: 2, across_gap: false });

      transport.connection("disconnected");
      transport.connection("connected");
      publish(stateMessage({ frame: 1, session: "capture-b", phase: "shot_underway", eventSequence: 1, configurationRevision: 3 }));
      changes = store.events.filter((event) => event.component !== "State delivery");
      expect(changes.map((event) => event.type)).toEqual(["Revision", "Session", "Phase", "Revision", "Phase"]);
      expect(changes.slice(0, 3).every((event) => (event.payload as any).across_gap === true)).toBe(true);
      expect(changes[1].payload).toMatchObject({ across_gap: true });
    } finally {
      await store.shutdown();
    }
  });

  it("expires freshness on a local monotonic deadline without a network request", async () => {
    vi.useFakeTimers();
    let nowMs = 0;
    const transport = createTransport();
    const store = new RootStore(transport.client, {
      monotonicNowMs: () => nowMs,
      wallTime: () => new Date(1_800_000_000_000 + nowMs),
    }, createNoopDiagnosticsTransport());
    try {
      transport.connection("connected");
      transport.handlers.get("state")!({
        ...stateMessage({ frame: 1 }),
        data: {
          ...stateMessage({ frame: 1 }).data,
          observation_status: {
            ...stateMessage({ frame: 1 }).data.observation_status,
            ball_positions_status: "observed",
            perception: {
              ...stateMessage({ frame: 1 }).data.observation_status.perception,
              last_ball_observation_age_ms: 10,
              layout_verification_max_age_ms: 10_000,
              cue_evidence_age_ms: 20,
              cue_evidence_max_age_ms: 100,
              cue: {
                shaft_start_px: [0, 0],
                shaft_end_px: [1, 1],
                axis_unit: [1, 1],
                confidence: 1,
                observation_age_ms: 20,
                source_frame_id: "cue-frame-1",
              },
              latest_motion_check_age_ms: 30,
              motion_evidence_max_age_ms: 20_000,
              table_visibility: {
                status: "visible",
                reason: "measured",
                observation_frame_id: "motion-frame-1",
                observation_age_ms: 30,
              },
            },
          },
        },
      });
      expect(store.events.some((event) => event.type === "Freshness")).toBe(false);

      nowMs = 80;
      vi.advanceTimersByTime(80);
      const freshness = store.events.filter((event) => event.type === "Freshness");
      expect(freshness).toHaveLength(1);
      expect(freshness[0].message).toContain("Cue evidence exceeded");
      expect(freshness[0].payload).toMatchObject({
        reported_age_ms: 20,
        browser_estimated_age_ms: 100,
        maximum_age_ms: 100,
        age_origin: "Browser estimated",
      });
      expect(transport.handlers.size).toBeGreaterThan(0);
      expect(transport.client).toHaveProperty("subscribe");
    } finally {
      await store.shutdown();
    }
  });

  it("recomputes an already scheduled stale deadline when the loaded freshness limit changes", async () => {
    vi.useFakeTimers();
    let nowMs = 0;
    const transport = createTransport();
    const store = new RootStore(transport.client, {
      monotonicNowMs: () => nowMs,
      wallTime: () => new Date(1_800_000_000_000 + nowMs),
    }, createNoopDiagnosticsTransport());
    try {
      transport.connection("connected");
      const initial = stateMessage({ frame: 1 });
      transport.handlers.get("state")!({
        ...initial,
        data: {
          ...initial.data,
          observation_status: {
            ...initial.data.observation_status,
            ball_positions_status: "observed",
            perception: {
              ...initial.data.observation_status.perception,
              last_ball_observation_age_ms: 0,
            },
          },
        },
      });

      // The state arrived before configuration, so RootStore initially used
      // its fallback of three seconds. The loaded config shortens it to 100ms.
      runInAction(() => { store.videoStore.maximumObservationAgeSeconds = 0.1; });
      nowMs = 99;
      vi.advanceTimersByTime(99);
      expect(store.events.filter((event) => event.type === "Freshness")).toHaveLength(0);

      nowMs = 100;
      vi.advanceTimersByTime(1);
      const freshness = store.events.filter((event) => event.type === "Freshness");
      expect(freshness).toHaveLength(1);
      expect(freshness[0].message).toContain("Ball observations exceeded");
      expect((freshness[0].payload as any).maximum_age_ms).toBe(100);
    } finally {
      await store.shutdown();
    }
  });

  it("records source-scoped video failures, recovery, replacement and confirmed interface actions without URLs or values", async () => {
    const transport = createTransport();
    const store = new RootStore(transport.client, undefined, createNoopDiagnosticsTransport());
    const updateDelivery = (input: Partial<typeof store.videoStore.delivery>) => runInAction(() => {
      store.videoStore.delivery = { ...store.videoStore.delivery, ...input };
    });
    try {
      updateDelivery({ state: "error", mode: "identified", configured: true, sourceGeneration: 1, error: "http://person:secret@camera/live" });
      updateDelivery({ state: "connecting" });
      updateDelivery({ state: "receiving" });
      updateDelivery({ state: "stalled" });
      updateDelivery({ state: "connecting" });
      updateDelivery({ state: "receiving" });
      updateDelivery({ state: "stalled" });
      updateDelivery({ state: "connecting", sourceGeneration: 2 });
      updateDelivery({ state: "receiving" });
      updateDelivery({ state: "error", mode: "native" });
      updateDelivery({ state: "loaded" });
      store.recordConfigurationApplied({
        changedFieldPaths: ["/camera/viewer_url"],
        restartRequired: true,
        previousRevision: 7,
      });

      const videoEvents = store.events.filter((event) => event.component === "Video");
      expect(videoEvents.map((event) => event.message)).toEqual([
        "Video receiver failed to load the configured source",
        "Video source replaced",
        "Identified video delivery stalled",
        "Identified video delivery recovered",
        "Identified video delivery stalled",
        "Identified video delivery recovered",
        "Video receiver failed to load the configured source",
        "Video source configured",
      ]);
      expect(videoEvents.filter((event) => event.message.includes("recovered"))).toHaveLength(2);
      const action = store.events.find((event) => event.origin === "Interface action");
      expect(action?.payload).toMatchObject({
        changed_field_paths: ["/camera/viewer_url"],
        restart_required: true,
        previous_revision: 7,
        applied_revision: "unknown until canonical reload",
      });
      expect(JSON.stringify(store.events)).not.toContain("secret");
      expect(JSON.stringify(store.events)).not.toContain("http://person");
    } finally {
      await store.shutdown();
    }
  });
});
