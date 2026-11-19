import { afterEach, describe, expect, it, vi } from "vitest";
import noSceneStartup from "../../../../perception/contracts/browser-diagnostics/v1/no-scene-startup.json";
import type {
  ConnectionStateHandler,
  MessageHandler,
  WebSocketClient,
} from "../services/websocket-client";
import type { BrowserDiagnosticsSnapshot } from "../services/browser-diagnostics";
import { createDiagnosticsTransportHarness } from "../tests/diagnostics-test-helpers";
import { RootStore } from "./RootStore";

const http = vi.hoisted(() => ({
  get: vi.fn(),
  getWebSocketUrl: vi.fn((path: string) => "ws://localhost" + path),
}));
vi.mock("../api/axios-client", () => ({ default: http }));

function createStateTransport() {
  const handlers = new Map<string, MessageHandler>();
  let connectionHandler!: ConnectionStateHandler;
  const client = {
    onConnectionState: (handler: ConnectionStateHandler) => {
      connectionHandler = handler;
    },
    on: (type: string, handler: MessageHandler) => handlers.set(type, handler),
    subscribe: vi.fn(),
    connect: vi.fn().mockResolvedValue(undefined),
    destroy: vi.fn(),
  };
  return {
    client: client as unknown as WebSocketClient,
    handlers,
    connection: (state: Parameters<ConnectionStateHandler>[0]) => connectionHandler(state),
  };
}

function snapshot(overrides: Partial<BrowserDiagnosticsSnapshot> = {}): BrowserDiagnosticsSnapshot {
  return {
    ...structuredClone(noSceneStartup),
    ...overrides,
  } as BrowserDiagnosticsSnapshot;
}

function stateMessage() {
  return {
    type: "state",
    timestamp: new Date(0).toISOString(),
    data: {
      balls: [],
      observation_status: {
        observation_identity: {
          engine_instance_id: "engine-a",
          session_id: "capture-a",
          frame_id: "frame-1",
          configuration_revision: 1,
        },
        transport_sequence: 1,
        perception: {
          event_sequence: 1,
          phase: "pre_shot",
          layout_revision: 1,
          layout_trusted: true,
          guidance_valid: true,
          trust_reason: "measured",
        },
      },
    },
  };
}

afterEach(() => {
  vi.useRealTimers();
  vi.clearAllMocks();
});

describe("root-owned pushed diagnostics", () => {
  it("orders revisions per gateway instance and rejects invalid or old snapshots", async () => {
    const state = createStateTransport();
    const diagnostics = createDiagnosticsTransportHarness();
    const store = new RootStore(state.client, undefined, diagnostics.client);
    try {
      const first = snapshot();
      diagnostics.publish(first);
      expect(store.diagnosticsStore.snapshot?.gateway.revision).toBe(1);

      diagnostics.publish(snapshot({
        gateway: { ...first.gateway, revision: 3 },
      }));
      diagnostics.publish(snapshot({
        gateway: { ...first.gateway, revision: 2 },
      }));
      expect(store.diagnosticsStore.snapshot?.gateway.revision).toBe(3);

      diagnostics.publish(snapshot({
        gateway: { ...first.gateway, instance_id: "gateway-restarted", revision: 1 },
      }));
      expect(store.diagnosticsStore.snapshot?.gateway).toMatchObject({
        instance_id: "gateway-restarted",
        revision: 1,
      });

      const accepted = store.diagnosticsStore.snapshot;
      diagnostics.publish({ ...snapshot(), schema_version: 0 });
      expect(store.diagnosticsStore.snapshot).toBe(accepted);
      expect(store.diagnosticsStore.protocolError).toMatch(/invalid diagnostics snapshot/i);
      expect(store.diagnosticsStore.isStale).toBe(true);
    } finally {
      await store.shutdown();
    }
  });

  it("marks silence and disconnect stale without coupling state/video channels or polling HTTP", async () => {
    vi.useFakeTimers();
    const state = createStateTransport();
    const diagnostics = createDiagnosticsTransportHarness();
    const store = new RootStore(state.client, undefined, diagnostics.client);
    try {
      diagnostics.connection("connected");
      diagnostics.publish(snapshot());
      state.connection("connected");
      state.handlers.get("state")!(stateMessage());

      expect(state.client.connect).toHaveBeenCalledOnce();
      expect(state.client.subscribe).toHaveBeenCalledExactlyOnceWith(["state"]);
      expect(diagnostics.client.connect).toHaveBeenCalledOnce();
      expect(store.system.status.isConnected).toBe(true);
      expect(store.videoStore.delivery.state).toBe("unconfigured");

      // Processing a state message schedules its own local freshness deadline.
      // That deadline must leave the separate diagnostics silence timer intact.
      vi.advanceTimersByTime(3_000);
      expect(store.diagnosticsStore.isStale).toBe(true);
      expect(store.system.status.isConnected).toBe(true);
      expect(store.videoStore.delivery.state).toBe("unconfigured");
      expect(http.get).not.toHaveBeenCalled();

      diagnostics.publish(snapshot({
        gateway: { ...snapshot().gateway, revision: 2 },
      }));
      diagnostics.connection("disconnected");
      expect(store.diagnosticsStore.isStale).toBe(true);
      expect(store.system.status.isConnected).toBe(true);
      expect(state.client.connect).toHaveBeenCalledOnce();
      expect(diagnostics.client.connect).toHaveBeenCalledOnce();
      expect(http.get).not.toHaveBeenCalled();
    } finally {
      await store.shutdown();
    }
  });

  it("clears the diagnostics stale timer on shutdown", async () => {
    vi.useFakeTimers();
    const state = createStateTransport();
    const diagnostics = createDiagnosticsTransportHarness();
    const store = new RootStore(state.client, undefined, diagnostics.client);
    diagnostics.publish(snapshot());
    await store.shutdown();
    vi.advanceTimersByTime(3_000);
    expect(store.diagnosticsStore.isStale).toBe(false);
    expect(diagnostics.client.destroy).toHaveBeenCalledOnce();
  });
});
