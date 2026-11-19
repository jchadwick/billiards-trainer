import {
  act,
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import lastKnownExpired from "../../../../../perception/contracts/browser-diagnostics/v1/last-known-expired.json";
import { HealthPage } from "./HealthPage";
const fixture = vi.hoisted(() => ({
  get: vi.fn(),
  state: null as any,
  connected: true,
  browserMetrics: { samples: [] as any[], latest: null as any },
  diagnosticsStore: {
    connection: "connecting",
    snapshot: null,
    receivedAt: null,
    isStale: false,
    protocolError: null,
  },
}));
vi.mock("../../api/axios-client", () => ({ default: { get: fixture.get } }));
vi.mock("../../stores/context", () => ({
  useRootStore: () => ({
    system: { status: { isConnected: fixture.connected } },
    videoStore: { latestGameState: fixture.state },
    browserMetrics: fixture.browserMetrics,
    diagnosticsStore: fixture.diagnosticsStore,
  }),
}));
beforeEach(() => {
  fixture.get.mockReset();
  fixture.state = null;
  fixture.connected = true;
  fixture.browserMetrics = { samples: [], latest: null };
  fixture.diagnosticsStore = {
    connection: "connecting",
    snapshot: null,
    receivedAt: null,
    isStale: false,
    protocolError: null,
  };
  fixture.get.mockImplementation((path: string) =>
    Promise.resolve({
      status: 200,
      data: { status: path.endsWith("/live") ? "alive" : "ready" },
    }),
  );
});
afterEach(() => {
  cleanup();
  vi.useRealTimers();
});
describe("System Health evidence", () => {
  it("loads health snapshots only on entry or explicit refresh, never polling", async () => {
    render(<HealthPage />);
    await waitFor(() => expect(screen.getByText("Alive")).toBeTruthy());
    expect(screen.getByText("Ready")).toBeTruthy();
    expect(fixture.get).toHaveBeenCalledTimes(2);
    vi.useFakeTimers();
    await act(async () => {
      vi.advanceTimersByTime(60_000);
    });
    expect(fixture.get).toHaveBeenCalledTimes(2);
    vi.useRealTimers();
    fireEvent.click(screen.getByText("Refresh health"));
    await waitFor(() => expect(fixture.get).toHaveBeenCalledTimes(4));
  });
  it("preserves a real 503 readiness assessment and keeps connectivity separate", async () => {
    fixture.get.mockImplementation((path: string) =>
      path.endsWith("/ready")
        ? Promise.reject({
            isAxiosError: true,
            response: {
              status: 503,
              data: {
                status: "not_ready",
                localization_available: false,
                freshness: "stale",
              },
            },
          })
        : Promise.resolve({ status: 200, data: { status: "alive" } }),
    );
    render(<HealthPage />);
    await waitFor(() => expect(screen.getByText("Not ready")).toBeTruthy());
    expect(screen.getByText("Alive")).toBeTruthy();
    expect(screen.getAllByText("Connected").length).toBeGreaterThan(0);
    expect(screen.getByText(/No valid snapshot received/)).toBeTruthy();
  });
  it("does not call a transport failure a dead server and maps pushed runtime evidence", async () => {
    fixture.get.mockRejectedValue(new Error("Connection unavailable"));
    fixture.state = {
      balls: [],
      observation_status: {
        engine_instance_id: "engine-test",
        source_session_id: "capture-test",
        source_frame_id: "frame-test",
        configuration_revision: 8,
        perception: {
          layout_revision: 4,
          layout_trusted: true,
          guidance_valid: true,
          cue_evidence_age_ms: 12,
          last_ball_observation_age_ms: 34,
        },
      },
    };
    render(<HealthPage />);
    await waitFor(() =>
      expect(screen.getByText(/Last checked .*HTTP snapshot/)).toBeTruthy(),
    );
    expect(screen.queryByText("Alive")).toBeNull();
    expect(screen.getByText("engine-test")).toBeTruthy();
    expect(screen.getByText("12 ms")).toBeTruthy();
    expect(screen.getByText("Connected")).toBeTruthy();
  });

  it("renders expired producer build and model data as inactive last-known evidence", async () => {
    fixture.diagnosticsStore = {
      connection: "connected",
      snapshot: structuredClone(lastKnownExpired),
      receivedAt: "2026-11-02T12:00:00.000Z",
      isStale: false,
      protocolError: null,
    };
    render(<HealthPage />);
    await waitFor(() => expect(screen.getByText("Refresh health")).toBeTruthy());
    fireEvent.click(screen.getByText(/Runtime details/));
    const inactiveBuild = screen.getAllByText(/Inactive · last-known report/)[0];
    expect(inactiveBuild).toBeTruthy();
    expect(screen.getByText(/Inactive · last-known report model/)).toBeTruthy();
    expect(screen.getByText(/no retained metric history/).closest(".highlight-issue"))
      .toBeTruthy();
    expect(inactiveBuild.closest(".highlight-issue")).toBeNull();
  });

  it("uses fresh producer metadata for active build and model details", async () => {
    const fresh = structuredClone(lastKnownExpired);
    fresh.producer.state = "fresh";
    fresh.producer.report_sequence = fresh.producer.last_known.report.sequence;
    fresh.producer.report_age_ms = 0;
    fresh.producer.expires_in_ms = 2_000;
    fresh.producer.metadata = fresh.producer.last_known.report.metadata;
    fresh.producer.last_known = null;
    fixture.diagnosticsStore = {
      connection: "connected",
      snapshot: fresh,
      receivedAt: "2026-11-02T12:00:00.000Z",
      isStale: false,
      protocolError: null,
    };
    render(<HealthPage />);
    await waitFor(() => expect(screen.getByText("Refresh health")).toBeTruthy());
    fireEvent.click(screen.getByText(/Runtime details/));
    expect(screen.getByText(/Active · billiards-perception/)).toBeTruthy();
    expect(screen.getByText(/1280×1280 · cue, ball · openvino/)).toBeTruthy();
  });
});
