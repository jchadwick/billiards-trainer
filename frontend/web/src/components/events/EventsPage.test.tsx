import {
  act,
  cleanup,
  fireEvent,
  render,
  screen,
} from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { EventsPage } from "./EventsPage";
const mocks = vi.hoisted(() => ({
  get: vi.fn(),
  store: {
    eventSessionStartedAt: "2026-10-31T05:00:00Z",
    eventHistoryLimit: 200,
    events: [
      {
        id: "browser-1",
        timestamp: "2026-10-31T05:00:01Z",
        severity: "Info",
        component: "State delivery",
        type: "Connection",
        origin: "Browser observed",
        message: "State stream connected",
        payload: { connection_state: "connected" },
      },
    ],
    videoStore: {
      latestGameState: {
        balls: [],
        observation_status: { engine_instance_id: "engine-real" },
      },
    },
    diagnosticsStore: {
      connection: "connecting",
      snapshot: null,
      receivedAt: null,
      isStale: false,
      protocolError: null,
    },
  },
}));
vi.mock("../../stores/context", () => ({ useRootStore: () => mocks.store }));
vi.mock("../../api/axios-client", () => ({ default: { get: mocks.get } }));
vi.mock("@tanstack/react-router", () => ({
  Link: ({ children, ...props }: any) => <a {...props}>{children}</a>,
}));
afterEach(() => {
  cleanup();
  vi.clearAllMocks();
  vi.useRealTimers();
});
describe("EventsPage", () => {
  it("filters real browser records and keeps prototype gaps separate", () => {
    render(<EventsPage />);
    expect(
      screen.getAllByText("State stream connected").length,
    ).toBeGreaterThan(0);
    expect(
      screen
        .getAllByText(/Cue observation exceeded its freshness limit/)[0]
        .closest(".highlight-issue"),
    ).toBeTruthy();
    fireEvent.change(screen.getByLabelText("Search events"), {
      target: { value: "no matching event" },
    });
    expect(screen.getByText("No events match these filters")).toBeTruthy();
    expect(
      screen.getByText("Select an event to inspect its evidence."),
    ).toBeTruthy();
    expect(
      screen.getAllByText(/Cue observation exceeded its freshness limit/)[0],
    ).toBeTruthy();
    expect(mocks.get).not.toHaveBeenCalled();
  });
  it("collects explicit health snapshots once, treats503 as evidence and excludes prototype logs", async () => {
    mocks.get.mockImplementation(async (path: string) => ({
      status: path.endsWith("ready") ? 503 : 200,
      data: { status: path.endsWith("ready") ? "not_ready" : "alive" },
    }));
    render(<EventsPage />);
    fireEvent.click(screen.getByText("Collect evidence"));
    expect(mocks.get).not.toHaveBeenCalled();
    await act(async () => {
      fireEvent.click(screen.getByText("Prepare review"));
    });
    expect(mocks.get).toHaveBeenCalledTimes(2);
    const bundle = JSON.parse(
      document.querySelector('[aria-label="Review diagnostic evidence"] pre')!
        .textContent!,
    );
    expect(bundle.health.checks[1].http_status).toBe(503);
    expect(bundle.events.items.map((item: { id: string }) => item.id)).toEqual([
      "browser-1",
    ]);
    expect(bundle.diagnostics).toMatchObject({
      source: "Single root-owned pushed diagnostics WebSocket snapshot",
      assessment: "unknown",
      producer_metadata_assessment: "unknown",
      snapshot: null,
    });
    expect(JSON.stringify(bundle)).not.toContain("evt-0208");
    expect(bundle.missing).toContain("Server logs");
    expect(
      (screen.getByText("Download reviewed JSON") as HTMLButtonElement)
        .disabled,
    ).toBe(false);
  });
  it("does not issue background HTTP requests while idle", () => {
    vi.useFakeTimers();
    render(<EventsPage />);
    act(() => {
      vi.advanceTimersByTime(60_000);
    });
    expect(mocks.get).not.toHaveBeenCalled();
  });
});
