import { vi } from "vitest";
import type {
  DiagnosticsTransport,
  DiagnosticsTransportCallbacks,
} from "../services/browser-diagnostics";

export function createDiagnosticsTransportHarness() {
  let callbacks: DiagnosticsTransportCallbacks | null = null;
  const client: DiagnosticsTransport = {
    connect: vi.fn((handlers: DiagnosticsTransportCallbacks) => {
      callbacks = handlers;
    }),
    destroy: vi.fn(),
  };
  return {
    client,
    callbacks: () => callbacks,
    publish: (payload: unknown) => callbacks?.onPayload(payload),
    connection: (state: Parameters<DiagnosticsTransportCallbacks["onConnectionState"]>[0]) =>
      callbacks?.onConnectionState(state),
  };
}

export function createNoopDiagnosticsTransport(): DiagnosticsTransport {
  return { connect: vi.fn(), destroy: vi.fn() };
}
