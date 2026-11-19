import { afterEach, describe, expect, it, vi } from "vitest";
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
    onConnectionState: (handler: ConnectionStateHandler) => {
      connectionHandler = handler;
    },
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

function stateMessage({
  frameId,
  sequence,
  sessionId = "session-a",
  eventSequence,
}: {
  frameId: string;
  sequence: number;
  sessionId?: string;
  eventSequence?: number;
}) {
  return {
    type: "state",
    timestamp: new Date(0).toISOString(),
    data: {
      balls: [],
      frame_number: sequence,
      observation_status: {
        observation_identity: {
          engine_instance_id: "engine-a",
          session_id: sessionId,
          frame_id: frameId,
        },
        transport_sequence: sequence,
        ...(eventSequence === undefined
          ? {}
          : { perception: { event_sequence: eventSequence } }),
      },
    },
  };
}

afterEach(() => vi.useRealTimers());

describe("RootStore browser session telemetry", () => {
  it("counts valid receipts before ordering and distinct accepted source frames after it", async () => {
    vi.useFakeTimers();
    let nowMs = 0;
    const transport = createTransport();
    const store = new RootStore(transport.client, {
      monotonicNowMs: () => nowMs,
      wallTime: () => new Date(1_800_000_000_000 + nowMs),
    }, createNoopDiagnosticsTransport());
    try {
      transport.connection("connected");
      const publish = (message: ReturnType<typeof stateMessage>, receivedAt: number) => {
        nowMs = receivedAt;
        transport.handlers.get("state")!(message);
      };

      publish(stateMessage({ frameId: "frame-1", sequence: 1 }), 100);
      publish(stateMessage({ frameId: "frame-1", sequence: 1 }), 200); // repeated frame
      publish(stateMessage({ frameId: "frame-0", sequence: 0 }), 300); // rejected old state
      publish(stateMessage({
        frameId: "frame-1",
        sequence: 1,
        eventSequence: 1,
      }), 400); // phase-only publication
      publish(stateMessage({ frameId: "frame-2", sequence: 2 }), 500);

      nowMs = 1_000;
      vi.advanceTimersByTime(1_000);

      expect(store.browserMetrics.latest?.receivedStateMessagesPerSecond).toBe(5);
      expect(store.browserMetrics.latest?.distinctObservedSourceFramesPerSecond)
        .toBeCloseTo(2.2222222);
      expect(store.browserMetrics.samples).toHaveLength(1);
    } finally {
      await store.shutdown();
    }
  });

  it("bounds the root-owned history to the configured retention count", async () => {
    vi.useFakeTimers();
    let nowMs = 0;
    const transport = createTransport();
    const store = new RootStore(transport.client, {
      monotonicNowMs: () => nowMs,
      wallTime: () => new Date(1_800_000_000_000 + nowMs),
    }, createNoopDiagnosticsTransport());
    try {
      transport.connection("connected");
      for (let index = 1; index <= 901; index += 1) {
        nowMs = index * 1_000;
        vi.advanceTimersByTime(1_000);
      }
      expect(store.browserMetrics.samples).toHaveLength(900);
      expect(store.browserMetrics.samples[0].monotonicMs).toBe(2_000);
      expect(store.browserMetrics.latest?.receivedStateMessagesPerSecond).toBe(0);
    } finally {
      await store.shutdown();
    }
  });
});
