import { useEffect, useState } from "react";
import { observer } from "mobx-react-lite";
import { toJS } from "mobx";
import { Link } from "@tanstack/react-router";
import { Download, Search, X, Copy } from "lucide-react";
import { useRootStore } from "../../stores/context";
import client from "../../api/axios-client";
import { Button } from "../ui/button";
import { Card, CardHeader, CardTitle, CardContent } from "../ui/card";
import { filterEvents, missingEvidence, redactEvidence } from "./event-model";
import { prototypeEvents } from "./prototype-events";

const initialFilters = {
  search: "",
  severity: "all",
  component: "all",
  type: "all",
  range: "session",
};
const pretty = (value: unknown) => JSON.stringify(value, null, 2);
/** Local session evidence never claims to provide persistent or lossless server history. */
export const EventsPage = observer(() => {
  const store = useRootStore();
  const [filters, setFilters] = useState(initialFilters);
  const [follow, setFollow] = useState(true);
  const [activeId, setActiveId] = useState<string | null>(
    store.events[0]?.id ?? null,
  );
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [collecting, setCollecting] = useState(false);
  const [include, setInclude] = useState({
    events: true,
    health: true,
    state: true,
    runtime: true,
    diagnostics: true,
    configuration: false,
  });
  const [bundle, setBundle] = useState<unknown>();
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState("");
  const [reviewedNewest, setReviewedNewest] = useState(store.events[0]?.id);
  const filtered = filterEvents(store.events, filters);
  const newestFilteredId = filtered[0]?.id;
  useEffect(() => {
    if (follow) {
      setActiveId(newestFilteredId ?? null);
      setReviewedNewest(store.events[0]?.id);
    }
  }, [follow, newestFilteredId, store.events[0]?.id]);
  const active = store.events.find((event) => event.id === activeId);
  const diagnosticsSnapshot = store.diagnosticsStore.snapshot;
  const diagnosticsFresh = diagnosticsSnapshot !== null
    && !store.diagnosticsStore.isStale
    && store.diagnosticsStore.connection === "connected";
  const diagnosticsProducerAssessment = !diagnosticsSnapshot
    ? "Unknown · no valid snapshot"
    : diagnosticsSnapshot.producer.state === "fresh" && diagnosticsFresh
      ? "Active · fresh producer metadata"
      : diagnosticsSnapshot.producer.last_known || diagnosticsSnapshot.producer.metadata
        ? "Inactive · last-known producer metadata"
        : "No active producer metadata";
  const unseenIndex = store.events.findIndex(
    (event) => event.id === reviewedNewest,
  );
  const unseen = follow
    ? 0
    : unseenIndex < 0
      ? store.events.length
      : unseenIndex;
  const selectedEvents = selected.size
    ? store.events.filter((event) => selected.has(event.id))
    : filtered;
  function toggle(id: string, checked: boolean) {
    setSelected((prior) => {
      const next = new Set(prior);
      checked ? next.add(id) : next.delete(id);
      return next;
    });
  }
  async function collect() {
    setBusy(true);
    setNotice("");
    setBundle(undefined);
    // These are user-requested snapshots, never periodic health/configuration polling.
    const requestedAt = new Date().toISOString();
    const state = store.videoStore.latestGameState
      ? toJS(store.videoStore.latestGameState)
      : null;
    const status = state?.observation_status;
    const events = toJS(selectedEvents);
    const diagnostics = toJS(store.diagnosticsStore);
    const producerSnapshot = diagnostics.snapshot?.producer;
    const producerMetadataIsFresh = producerSnapshot?.state === "fresh"
      && !diagnostics.isStale
      && diagnostics.connection === "connected";
    try {
      const [health, configuration] = await Promise.all([
        include.health
          ? Promise.all(
              ["/api/v1/health/live", "/api/v1/health/ready"].map(
                async (path) => {
                  try {
                    const response = await client.get(path, {
                      validateStatus: () => true,
                    });
                    return {
                      path,
                      checked_at: new Date().toISOString(),
                      http_status: response.status,
                      response: response.data,
                    };
                  } catch (error) {
                    return {
                      path,
                      checked_at: new Date().toISOString(),
                      error:
                        error instanceof Error ? error.message : String(error),
                    };
                  }
                }
              ),
            )
          : undefined,
        include.configuration
          ? client
              .get<{ values: unknown }>("/api/v1/config")
              .then((response) => ({
                source: "GET /api/v1/config; active settings, draft excluded",
                checked_at: new Date().toISOString(),
                values: response.data.values,
              }))
          : undefined,
      ]);
      setBundle(
        redactEvidence({
          format: "billiards-trainer-browser-evidence-v1",
          collected_at: requestedAt,
          evidence_scope:
            "Current browser session; not a server diagnostic bundle",
          session_started_at: store.eventSessionStartedAt,
          missing: missingEvidence,
          ...(include.events
            ? {
                events: {
                  source:
                    "Browser-observed state, freshness and video transitions; confirmed interface actions; WebSocket transport/control responses",
                  retained_limit: store.eventHistoryLimit,
                  items: events,
                },
              }
            : {}),
          ...(include.health
            ? {
                health: {
                  source: "Explicit HTTP snapshot; not continuous monitoring",
                  checks: health,
                },
              }
            : {}),
          ...(include.state
            ? {
                state: {
                  source:
                    "Latest accepted WebSocket snapshot at collection start; not historical event state",
                  snapshot: state,
                },
              }
            : {}),
          ...(include.runtime
            ? {
                runtime: {
                  source: "Latest accepted state provenance",
                  observation_identity: status?.observation_identity,
                  engine_instance_id: status?.engine_instance_id,
                  source_session_id: status?.source_session_id,
                  configuration_revision: status?.configuration_revision,
                  calibration_revision: status?.calibration_revision,
                  rectification_revision: status?.rectification_revision,
                },
              }
            : {}),
          ...(include.diagnostics
            ? {
                diagnostics: {
                  source: "Single root-owned pushed diagnostics WebSocket snapshot",
                  assessment: diagnostics.snapshot
                    ? diagnostics.isStale || diagnostics.connection !== "connected"
                      ? "last_known_stale"
                      : "current"
                    : "unknown",
                    connection: diagnostics.connection,
                    received_at: diagnostics.receivedAt,
                    protocol_error: diagnostics.protocolError,
                    producer_metadata_assessment: diagnostics.snapshot?.producer.state === "fresh"
                      && producerMetadataIsFresh
                      ? "active_fresh"
                      : diagnostics.snapshot?.producer.last_known
                        || diagnostics.snapshot?.producer.metadata
                        ? "inactive_last_known"
                        : "unknown",
                    active_producer_metadata: producerMetadataIsFresh
                      ? producerSnapshot?.metadata ?? null
                      : null,
                    inactive_last_known_metadata: producerSnapshot?.metadata
                      && !producerMetadataIsFresh
                      ? {
                          assessment: "inactive_last_known_snapshot_metadata",
                          metadata: producerSnapshot.metadata,
                        }
                      : null,
                    inactive_last_known_report: diagnostics.snapshot?.producer.last_known
                      ? {
                          assessment: "inactive_last_known",
                          report_age_ms: diagnostics.snapshot.producer.last_known.report_age_ms,
                          report: diagnostics.snapshot.producer.last_known.report,
                        }
                      : null,
                    snapshot: diagnostics.snapshot,
                  },
              }
            : {}),
          ...(include.configuration ? { configuration } : {}),
        }),
      );
      setCollecting(true);
    } catch (error) {
      setNotice(
        `Evidence could not be collected: ${error instanceof Error ? error.message : String(error)}. Nothing downloaded.`,
      );
    } finally {
      setBusy(false);
    }
  }
  function download() {
    const url = URL.createObjectURL(
      new Blob([pretty(bundle)], { type: "application/json" }),
    );
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = "billiards-trainer-browser-evidence.json";
    anchor.click();
    URL.revokeObjectURL(url);
    setCollecting(false);
    setNotice("Reviewed browser evidence downloaded.");
  }
  return (
    <div className="space-y-5">
      <div className="page-heading">
        <h1>Events &amp; debugging</h1>
        <Button
          disabled={busy}
          onClick={() => {
            setCollecting(true);
            setBundle(undefined);
          }}
        >
          <Download />
          Collect evidence
        </Button>
      </div>
      <div className="flex flex-wrap justify-between gap-3 text-sm text-muted-foreground">
        <span>
          <strong className="text-foreground">Current browser session</strong> ·
          Started {new Date(store.eventSessionStartedAt).toLocaleString()} ·{" "}
          {store.events.length} retained events · limit{" "}
          {store.eventHistoryLimit}
        </span>
        <span>
          Persistent history &amp; server logs{" "}
          <span className="highlight-issue">Not available</span>
        </span>
      </div>
      <p className="text-xs text-muted-foreground">
        Browser-observed state changes, freshness crossings, video receiver
        transitions, confirmed configuration actions, and WebSocket transport
        evidence. Repeated state snapshots are not events; delivery failures do
        not prove a backend crash or create an authoritative backend timeline.
      </p>
      {notice && (
        <p role="status" className="notice">
          {notice}
        </p>
      )}
      <Card>
        <CardContent className="space-y-4">
          <div className="flex flex-wrap items-center justify-between gap-4">
            <label className="flex flex-1 items-center gap-2">
              <Search size={16} />
              <span className="sr-only">Search events</span>
              <input
                className="w-full rounded-md border bg-background p-2 text-sm"
                aria-label="Search events"
                placeholder="Search messages, components, or event IDs…"
                value={filters.search}
                onChange={(event) =>
                  setFilters({ ...filters, search: event.target.value })
                }
              />
            </label>
            <label className="flex items-center gap-2 text-sm">
              <input
                type="checkbox"
                checked={follow}
                onChange={(event) => setFollow(event.target.checked)}
              />
              Follow live
            </label>
          </div>
          <div className="flex flex-wrap items-center gap-4 text-xs">
            {(["severity", "component", "type", "range"] as const).map(
              (field) => (
                <label key={field} className="flex items-center gap-2">
                  {
                    {
                      severity: "Severity",
                      component: "Component",
                      type: "Event type",
                      range: "Time range",
                    }[field]
                  }
                  <select
                    aria-label={
                      field === "type"
                        ? "Event type"
                        : field === "range"
                          ? "Time range"
                          : field === "severity"
                            ? "Severity"
                            : "Component"
                    }
                    value={filters[field]}
                    onChange={(event) =>
                      setFilters({ ...filters, [field]: event.target.value })
                    }
                  >
                    {field === "range" ? (
                      <>
                        <option value="session">This session</option>
                        <option value="1m">Last 1 minute</option>
                        <option value="5m">Last 5 minutes</option>
                      </>
                    ) : (
                      <>
                        <option value="all">
                          All{" "}
                          {field === "severity"
                            ? "severities"
                            : field === "component"
                              ? "components"
                              : "types"}
                        </option>
                        {(field === "severity"
                          ? ["Info", "Warning", "Error"]
                          : [
                              ...new Set(
                                store.events.map((event) => event[field]),
                              ),
                            ]
                        ).map((value) => (
                          <option key={value}>{value}</option>
                        ))}
                      </>
                    )}
                  </select>
                </label>
              ),
            )}
            <span>
              {filtered.length} shown · {selectedEvents.length}{" "}
              {selected.size ? "selected" : "filtered"}
            </span>
            <Button size="sm" onClick={() => setFilters(initialFilters)}>
              Clear filters
            </Button>
          </div>
          {!follow && unseen > 0 && (
            <Button size="sm" onClick={() => setFollow(true)}>
              {unseen} new events · Return to latest
            </Button>
          )}
        </CardContent>
        <div className="grid border-t lg:grid-cols-[minmax(0,1fr)_360px]">
          <div className="min-w-0">
            <div className="flex items-center gap-3 border-b px-4 py-3 text-xs text-muted-foreground">
              <input
                aria-label="Select all filtered events"
                type="checkbox"
                checked={
                  filtered.length > 0 &&
                  filtered.every((event) => selected.has(event.id))
                }
                onChange={(event) => {
                  const checked = event.target.checked;
                  setSelected((prior) => {
                    const next = new Set(prior);
                    filtered.forEach((item) =>
                      checked ? next.add(item.id) : next.delete(item.id),
                    );
                    return next;
                  });
                }}
              />
              <span>TIME · browser locale</span>
              <span>EVENT</span>
            </div>
            <div className="max-h-[60vh] overflow-auto">
              {filtered.map((event) => (
                <div
                  key={event.id}
                  className={`flex items-start gap-3 border-b px-4 py-3 ${activeId === event.id ? "bg-accent" : ""}`}
                >
                  <input
                    type="checkbox"
                    aria-label={`Select event ${event.id}`}
                    checked={selected.has(event.id)}
                    onChange={(input) => toggle(event.id, input.target.checked)}
                  />
                  <button
                    className="flex min-w-0 flex-1 gap-4 text-left"
                    onClick={() => {
                      setActiveId(event.id);
                      setFollow(false);
                    }}
                  >
                    <time
                      className="shrink-0 font-mono text-xs text-muted-foreground"
                      dateTime={event.timestamp}
                    >
                      {new Date(event.timestamp).toLocaleTimeString()}
                    </time>
                    <div className="min-w-0 space-y-1">
                      <div className="text-xs text-muted-foreground">
                        <span
                          className={
                            event.severity === "Error"
                              ? "text-red-300"
                              : event.severity === "Warning"
                                ? "text-amber-300"
                                : ""
                          }
                        >
                          {event.severity}
                        </span>{" "}
                        · {event.component}
                      </div>
                      <strong className="block text-sm">{event.message}</strong>
                      <span className="text-xs text-muted-foreground">
                        {event.origin}
                      </span>
                    </div>
                  </button>
                </div>
              ))}
              {!filtered.length && (
                <div className="p-8 text-sm text-muted-foreground">
                  <h2 className="mb-2 font-semibold text-foreground">
                    No events match these filters
                  </h2>
                  <p>Try a broader time range or clear the search.</p>
                </div>
              )}
            </div>
            <p className="p-4 text-xs text-muted-foreground">
              Beginning of retained browser-session evidence. Older records are
              evicted; older server history is not connected.
            </p>
          </div>
          <aside className="min-w-0 border-t lg:border-l lg:border-t-0">
            <div className="flex items-center justify-between border-b p-4">
              <h2 className="text-sm font-semibold">Event detail</h2>
              {active && (
                <Button
                  size="icon"
                  variant="ghost"
                  aria-label="Close event detail"
                  onClick={() => {
                    setActiveId(null);
                    setFollow(false);
                  }}
                >
                  <X />
                </Button>
              )}
            </div>
            {active ? (
              <div className="space-y-4 p-4 text-sm">
                <div className="font-mono text-xs text-muted-foreground">
                  {active.id} · {active.severity}
                </div>
                <h3 className="font-semibold">{active.message}</h3>
                <dl className="detail-rows">
                  {[
                    ["Timestamp", active.timestamp],
                    ["Evidence origin", active.origin],
                    ["Component", active.component],
                    ["Type", active.type],
                  ].map(([label, value]) => (
                    <div key={label}>
                      <dt>{label}</dt>
                      <dd>{value}</dd>
                    </div>
                  ))}
                </dl>
                <Button
                  size="sm"
                  onClick={() => {
                    void navigator.clipboard
                      .writeText(pretty(redactEvidence(active)))
                      .then(
                        () => setNotice("Copied redacted event."),
                        () =>
                          setNotice(
                            "Clipboard unavailable. Select the detail text to copy.",
                          ),
                      );
                  }}
                >
                  <Copy />
                  Copy
                </Button>
                <pre className="json-preview">
                  {pretty(redactEvidence(active.payload))}
                </pre>
                <details className="rounded-md border p-3">
                  <summary className="cursor-pointer text-sm font-medium">
                    Current pushed diagnostics snapshot
                    <span className="ml-2 text-xs text-muted-foreground">
                      {store.diagnosticsStore.snapshot
                        ? store.diagnosticsStore.isStale || store.diagnosticsStore.connection !== "connected"
                          ? "Last known · stale"
                          : "Current"
                        : "Unknown"}
                    </span>
                  </summary>
                  <p className="my-2 text-xs text-muted-foreground">
                    This is the latest shared snapshot, not evidence captured at the event time.
                    {" "}{diagnosticsProducerAssessment}.
                  </p>
                  <pre className="json-preview">{pretty(redactEvidence({
                    connection: store.diagnosticsStore.connection,
                    received_at: store.diagnosticsStore.receivedAt,
                    protocol_error: store.diagnosticsStore.protocolError,
                    snapshot: store.diagnosticsStore.snapshot,
                  }))}</pre>
                </details>
                <p className="text-xs text-muted-foreground">
                  No historical state snapshot or camera image is retained.
                  Interface-action records include changed field paths, never
                  configuration values. Each record identifies its evidence
                  origin.
                </p>
                <div className="flex flex-wrap gap-2">
                  <Button asChild size="sm">
                    <Link to="/live">Live diagnostics</Link>
                  </Button>
                  <Button asChild size="sm">
                    <Link to="/health">System health</Link>
                  </Button>
                </div>
              </div>
            ) : (
              <p className="p-4 text-sm text-muted-foreground">
                Select an event to inspect its evidence.
              </p>
            )}
          </aside>
        </div>
      </Card>
      <Card>
        <details>
          <summary className="cursor-pointer p-4 text-sm font-semibold">
            Prototype event data awaiting authoritative backend integration
          </summary>
          <CardContent>
            <p className="mb-4 text-xs text-muted-foreground">
              Verbatim illustrative records, excluded from live filtering and
              evidence downloads. Similar browser-observed transitions above
              are separate evidence and do not establish backend occurrence
              times, event IDs, or a complete history.
            </p>
            <div className="space-y-3">
              {prototypeEvents.map((event) => (
                <details
                  key={event.id}
                  className="highlight-issue rounded-md border p-3"
                >
                  <summary className="cursor-pointer text-sm">
                    {event.time} · {event.severity} · {event.component} ·{" "}
                    {event.message}
                  </summary>
                  <pre className="json-preview mt-3">{pretty(event)}</pre>
                </details>
              ))}
            </div>
          </CardContent>
        </details>
      </Card>
      {collecting && (
        <Card role="region" aria-label="Review diagnostic evidence">
          <CardHeader>
            <CardTitle>Review diagnostic evidence</CardTitle>
            <Button
              variant="ghost"
              size="icon"
              aria-label="Close evidence review"
              onClick={() => setCollecting(false)}
            >
              <X />
            </Button>
          </CardHeader>
          <CardContent className="space-y-4">
            <p className="text-sm text-muted-foreground">
              Download only the information currently available to this browser.
              Optional HTTP checks run once when you prepare the review.
            </p>
            <div className="flex flex-wrap gap-4">
              {(
                [
                  "events",
                  "health",
                  "state",
                  "runtime",
                  "diagnostics",
                  "configuration",
                ] as const
              ).map((section) => (
                <label
                  className="flex items-center gap-2 text-sm"
                  key={section}
                >
                  <input
                    type="checkbox"
                    checked={include[section]}
                    disabled={busy}
                    onChange={(event) => {
                      setInclude({
                        ...include,
                        [section]: event.target.checked,
                      });
                      setBundle(undefined);
                    }}
                  />
                  {
                    {
                      events: "Selected / filtered events",
                      health: "Current health evidence",
                      state: "Latest accepted state",
                      runtime: "Runtime & revisions",
                      diagnostics: "Pushed backend diagnostics snapshot",
                      configuration: "Active configuration · redacted",
                    }[section]
                  }
                </label>
              ))}
            </div>
            <p className="text-xs text-muted-foreground">
              Secret fields and URL credentials are redacted. Review the JSON
              before downloading. Active configuration is fetched explicitly;
              unsaved drafts are excluded.
            </p>
            <div className="flex flex-wrap gap-2">
              <Button
                disabled={busy || !Object.values(include).some(Boolean)}
                onClick={() => void collect()}
              >
                {busy ? "Collecting…" : "Prepare review"}
              </Button>
              <Button disabled={!bundle || busy} onClick={download}>
                <Download />
                Download reviewed JSON
              </Button>
            </div>
            {!!bundle && (
              <pre className="json-preview max-h-[50vh]">{pretty(bundle)}</pre>
            )}
            <p className="text-xs text-muted-foreground">
              Missing: {missingEvidence.join(" · ")}.
            </p>
          </CardContent>
        </Card>
      )}
    </div>
  );
});
