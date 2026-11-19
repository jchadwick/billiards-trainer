import { describe, expect, it, vi } from "vitest";
import { RootStore } from "../../stores/RootStore";
import type {
  WebSocketClient,
  ConnectionStateHandler,
  MessageHandler,
} from "../../services/websocket-client";
import { createNoopDiagnosticsTransport } from "../../tests/diagnostics-test-helpers";

describe("shared browser event retention", () => {
  it("records real transport evidence, excludes routine states and bounds retained records", async () => {
    const handlers = new Map<string, MessageHandler>();
    let onConnection!: ConnectionStateHandler;
    const transport = {
      onConnectionState: (handler: ConnectionStateHandler) => {
        onConnection = handler;
      },
      on: (type: string, handler: MessageHandler) =>
        handlers.set(type, handler),
      subscribe: vi.fn(),
      connect: vi.fn().mockResolvedValue(undefined),
      destroy: vi.fn(),
    };
    const store = new RootStore(transport as unknown as WebSocketClient, undefined, createNoopDiagnosticsTransport());
    try {
      onConnection("connected");
      expect(store.events[0].origin).toBe("Browser observed");
      expect(store.events[0].message).toBe("State stream connected");
      handlers.get("state")!({
        type: "state",
        timestamp: new Date().toISOString(),
        data: { balls: [] },
      });
      expect(store.events).toHaveLength(1);
      handlers.get("error")!({
        type: "error",
        timestamp: new Date().toISOString(),
        data: { message: "Unsupported stream" },
      });
      expect(store.events[0].origin).toBe("WebSocket control response");
      expect(transport.subscribe).toHaveBeenCalledExactlyOnceWith(["state"]);
      expect(transport.connect).toHaveBeenCalledOnce();
      const beforeDuplicate = store.events.length;
      onConnection("reconnecting");
      onConnection("reconnecting");
      expect(store.events.length).toBe(beforeDuplicate + 1);
      for (let index = 0; index < store.eventHistoryLimit; index++) {
        onConnection("connected");
        onConnection("reconnecting");
      }
      expect(store.events).toHaveLength(store.eventHistoryLimit);
      expect(store.events[0]?.message).toBe("State stream reconnecting");
      expect(new Set(store.events.map((event) => event.id)).size).toBe(
        store.eventHistoryLimit,
      );
    } finally {
      await store.shutdown();
    }
  });
});
