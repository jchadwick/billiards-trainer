import { render, screen, fireEvent, cleanup } from "@testing-library/react";
import { afterEach, describe, it, expect, vi } from "vitest";
import { Diagnostics } from "./Diagnostics";
vi.mock("@tanstack/react-router", () => ({
  Link: ({ children, ...props }: any) => <a {...props}>{children}</a>,
}));
vi.mock("./video/LiveView", () => ({
  LiveView: ({ presentation }: { presentation?: string }) => (
    <div data-testid="diagnostics-camera" data-presentation={presentation}>
      Camera fixture
    </div>
  ),
}));
const fixture = vi.hoisted(() => ({ current: null as any, mode: "native" }));
vi.mock("../stores/context", () => ({
  useRootStore: () => ({
    system: { status: { isConnected: true } },
    videoStore: { latestGameState: fixture.current, delivery: { mode: fixture.mode } },
  }),
}));
afterEach(() => {
  cleanup();
  fixture.current = null;
});
describe("Diagnostics evidence", () => {
  it("inventories unsupported ages separately from live observations", () => {
    render(<Diagnostics />);
    expect(screen.getByTestId("diagnostics-camera").getAttribute("data-presentation")).toBe("diagnostics");
    expect(screen.getByText("No ball observations available.")).toBeTruthy();
    const overlays = screen.getByRole("checkbox", { name: "Detection overlays" }) as HTMLInputElement;
    expect(overlays.disabled).toBe(false);
    fireEvent.click(overlays);
    expect(overlays.checked).toBe(false);
    expect(screen.getByText("Ball centers show trusted or last-known table positions.")).toBeTruthy();
    expect(
      screen.getByText("84 ms · 420 ms · 980 ms · 1.84 s").className,
    ).toContain("highlight-issue");
    fireEvent.click(screen.getByText("Pause inspection"));
    expect(screen.getByRole("status").textContent).toContain(
      "Wait for a camera image",
    );
  });
  it("shows API coordinates, identity and confidence and filters retained evidence", () => {
    fixture.current = {
      balls: [
        {
          id: "live-a",
          position: { x: 22, y: 33, scale: [2, 2] },
          number: 7,
          confidence: 0.88,
          is_observed: true,
        },
        {
          id: "live-b",
          position: { x: 3, y: 4, scale: [1, 1] },
          confidence: 0,
          confidence_kind: "unavailable",
          is_observed: false,
        },
      ],
      observation_status: {
        configuration_revision: 12,
        perception: {
          phase: "aiming",
          layout_trusted: true,
          trust_reason: "verified",
          guidance_valid: false,
          last_ball_observation_age_ms: 100,
          cue_evidence_age_ms: 500,
        },
      },
    };
    render(<Diagnostics />);
    expect(screen.getByText("live-a")).toBeTruthy();
    expect(screen.getByText("88%")).toBeTruthy();
    fireEvent.change(screen.getByLabelText("Filter ball observations"), {
      target: { value: "stale" },
    });
    expect(screen.queryByText("live-a")).toBeNull();
    expect(screen.getByText("live-b")).toBeTruthy();
  });
});
