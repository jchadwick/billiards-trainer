/** Session evidence is browser-owned; it is never presented as a server log. */
export interface BrowserEvent {
  id: string;
  timestamp: string;
  severity: "Info" | "Warning" | "Error";
  component: string;
  type: string;
  origin:
    | "Browser observed"
    | "WebSocket control response"
    | "Interface action";
  message: string;
  payload: unknown;
}
export const DEFAULT_EVENT_HISTORY_LIMIT = 200;
export const MAX_EVENT_HISTORY_LIMIT = 10_000;

export function getEventHistoryLimit(
  env: Record<string, unknown> = import.meta.env,
): number {
  const raw = env.VITE_EVENT_HISTORY_LIMIT;
  if (typeof raw !== "string" || raw.trim() === "") {
    return DEFAULT_EVENT_HISTORY_LIMIT;
  }
  const configured = Number(raw);
  return Number.isSafeInteger(configured)
    && configured >= 1
    && configured <= MAX_EVENT_HISTORY_LIMIT
    ? configured
    : DEFAULT_EVENT_HISTORY_LIMIT;
}

export const eventHistoryLimit = getEventHistoryLimit();
export const missingEvidence = [
  "Authoritative backend event feed and occurrence timestamps",
  "Persistent backend event history",
  "Server logs",
  "Retained historical camera frames",
  "Resource monitoring metrics",
  "Server-generated diagnostic bundle",
];

/** Exports exclude known secret fields and credentials carried inside URLs. */
export function redactEvidence(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(redactEvidence);
  if (value && typeof value === "object")
    return Object.fromEntries(
      Object.entries(value).map(([key, item]) => [
        key,
        /password|secret|token|authorization|api.?key|credential/i.test(key)
          ? "[REDACTED]"
          : redactEvidence(item),
      ]),
    );
  if (typeof value === "string" && /^[a-z][a-z\d+.-]*:\/\//i.test(value)) {
    try {
      const url = new URL(value);
      if (url.username) url.username = "REDACTED";
      if (url.password) url.password = "REDACTED";
      for (const key of [...url.searchParams.keys()])
        if (/token|key|secret|password|credential/i.test(key))
          url.searchParams.set(key, "REDACTED");
      return url.toString();
    } catch {
      return value;
    }
  }
  return value;
}
export function filterEvents(
  events: BrowserEvent[],
  filters: {
    search: string;
    severity: string;
    component: string;
    type: string;
    range: string;
  },
  now = Date.now(),
) {
  const maximumAge =
    filters.range === "1m"
      ? 60_000
      : filters.range === "5m"
        ? 300_000
        : Infinity;
  return events.filter(
    (event) =>
      (filters.severity === "all" || event.severity === filters.severity) &&
      (filters.component === "all" || event.component === filters.component) &&
      (filters.type === "all" || event.type === filters.type) &&
      now - Date.parse(event.timestamp) <= maximumAge &&
      [event.id, event.message, event.component, event.type, event.origin]
        .join(" ")
        .toLowerCase()
        .includes(filters.search.toLowerCase()),
  );
}
