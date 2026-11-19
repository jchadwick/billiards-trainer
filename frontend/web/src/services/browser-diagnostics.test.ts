import { afterEach, describe, expect, it, vi } from "vitest";
import noSceneStartup from "../../../../perception/contracts/browser-diagnostics/v1/no-scene-startup.json";
import lastKnownExpired from "../../../../perception/contracts/browser-diagnostics/v1/last-known-expired.json";
import {
  BrowserDiagnosticsWebSocket,
  isBrowserDiagnosticsSnapshot,
} from "./browser-diagnostics";

class MockSocket {
  onopen: ((event: Event) => void) | null = null;
  onmessage: ((event: MessageEvent) => void) | null = null;
  onerror: ((event: Event) => void) | null = null;
  onclose: ((event: CloseEvent) => void) | null = null;
  close = vi.fn();

  open() {
    this.onopen?.(new Event("open"));
  }

  sendPayload(payload: unknown) {
    this.onmessage?.({ data: JSON.stringify(payload) } as MessageEvent);
  }

  closeConnection() {
    this.onclose?.({ code: 1006, reason: "network interruption" } as CloseEvent);
  }
}

afterEach(() => vi.useRealTimers());

describe("browser diagnostics v1", () => {
  it("accepts current and last-known contract fixtures and rejects malformed snapshots", () => {
    expect(isBrowserDiagnosticsSnapshot(noSceneStartup)).toBe(true);
    expect(isBrowserDiagnosticsSnapshot(lastKnownExpired)).toBe(true);

    const missingStage = structuredClone(noSceneStartup);
    missingStage.stages.pop();
    expect(isBrowserDiagnosticsSnapshot(missingStage)).toBe(false);

    const unsupportedVersion = { ...noSceneStartup, schema_version: 2 };
    expect(isBrowserDiagnosticsSnapshot(unsupportedVersion)).toBe(false);

    const invalidScheduler = structuredClone(noSceneStartup);
    invalidScheduler.stages[4].active_job_kind = "not-a-job";
    expect(isBrowserDiagnosticsSnapshot(invalidScheduler)).toBe(false);
  });

  it("uses one raw-snapshot socket, reconnects after close and sends no polling traffic", () => {
    vi.useFakeTimers();
    const sockets: MockSocket[] = [];
    const states: string[] = [];
    const payloads: unknown[] = [];
    const client = new BrowserDiagnosticsWebSocket(
      "ws://gateway/api/v1/diagnostics/ws",
      { initialDelayMs: 100, maximumDelayMs: 200 },
      () => {
        const socket = new MockSocket();
        sockets.push(socket);
        return socket;
      },
    );
    client.connect({
      onConnectionState: (state) => states.push(state),
      onPayload: (payload) => payloads.push(payload),
    });
    expect(sockets).toHaveLength(1);

    sockets[0].open();
    sockets[0].sendPayload(noSceneStartup);
    expect(states).toEqual(["connecting", "connected"]);
    expect(payloads).toEqual([noSceneStartup]);
    expect("send" in sockets[0]).toBe(false);

    vi.advanceTimersByTime(60_000);
    expect(sockets).toHaveLength(1);
    sockets[0].closeConnection();
    expect(states.at(-1)).toBe("disconnected");
    vi.advanceTimersByTime(99);
    expect(sockets).toHaveLength(1);
    vi.advanceTimersByTime(1);
    expect(sockets).toHaveLength(2);

    client.destroy();
    vi.advanceTimersByTime(60_000);
    expect(sockets).toHaveLength(2);
    expect(sockets[1].close).toHaveBeenCalledOnce();
  });
});
