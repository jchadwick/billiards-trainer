import { describe, expect, it } from "vitest";
import { DEFAULT_EVENT_HISTORY_LIMIT, filterEvents, getEventHistoryLimit, redactEvidence, type BrowserEvent } from "./event-model";

describe("browser event evidence", () => {
  it("bounds the configurable in-memory retention limit", () => {
    expect(getEventHistoryLimit({})).toBe(DEFAULT_EVENT_HISTORY_LIMIT);
    expect(getEventHistoryLimit({ VITE_EVENT_HISTORY_LIMIT: "500" })).toBe(500);
    expect(getEventHistoryLimit({ VITE_EVENT_HISTORY_LIMIT: "0" })).toBe(DEFAULT_EVENT_HISTORY_LIMIT);
    expect(getEventHistoryLimit({ VITE_EVENT_HISTORY_LIMIT: "1000000" })).toBe(DEFAULT_EVENT_HISTORY_LIMIT);
  });
  it("redacts nested secrets and URL credentials without changing the source", () => {
    const source = {
      settings: [
        {
          api_key: "secret",
          viewer:
            "http://person:password@camera.local/live?token=private&mode=live",
        },
      ],
      phase: "aiming",
    };
    const result = redactEvidence(source) as typeof source;
    expect(result.settings[0].api_key).toBe("[REDACTED]");
    expect(result.settings[0].viewer).toBe(
      "http://REDACTED:REDACTED@camera.local/live?token=REDACTED&mode=live",
    );
    expect(result.phase).toBe("aiming");
    expect(source.settings[0].api_key).toBe("secret");
  });
  it("combines search, severity, component, type and wall-clock time filters", () => {
    const now = Date.parse("2026-10-31T05:00:00Z");
    const event: BrowserEvent = {
      id: "browser-1",
      timestamp: new Date(now - 20_000).toISOString(),
      severity: "Warning",
      component: "State delivery",
      type: "Connection",
      origin: "Browser observed",
      message: "State stream reconnecting",
      payload: {},
    };
    const events = [
      event,
      {
        ...event,
        id: "browser-2",
        timestamp: new Date(now - 90_000).toISOString(),
      },
    ];
    const filters = {
      search: "RECONNECT",
      severity: "Warning",
      component: "State delivery",
      type: "Connection",
      range: "1m",
    };
    expect(filterEvents(events, filters, now).map((item) => item.id)).toEqual([
      "browser-1",
    ]);
    expect(
      filterEvents(events, { ...filters, severity: "Error" }, now),
    ).toEqual([]);
  });
});
