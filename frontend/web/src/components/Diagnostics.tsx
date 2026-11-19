import { useEffect, useRef, useState } from "react";
import { observer } from "mobx-react-lite";
import { toJS } from "mobx";
import {
  Camera,
  Code2,
  Download,
  Pause,
  Play,
  Maximize2,
  Activity,
  Settings2,
} from "lucide-react";
import { Link } from "@tanstack/react-router";
import { useRootStore } from "../stores/context";
import { LiveView } from "./video/LiveView";
import { Button } from "./ui/button";
import { Card, CardHeader, CardTitle, CardContent } from "./ui/card";
import type { GameStateData } from "../types/api";

// Browser presentation cadence is configurable independently of producer sampling.
const requestedRefreshMs = Number(import.meta.env.VITE_DIAGNOSTICS_REFRESH_MS);
const diagnosticRefreshMs =
  Number.isFinite(requestedRefreshMs) && requestedRefreshMs > 0
    ? requestedRefreshMs
    : 250;

const display = (value: unknown): string =>
  value == null
    ? "Not available"
    : typeof value === "object"
      ? JSON.stringify(value)
      : String(value);
const age = (value: number | null | undefined) =>
  value == null ? "Not available" : `${Math.round(value)} ms`;
function Rows({ rows }: { rows: [string, unknown][] }) {
  return (
    <dl className="detail-rows">
      {rows.map(([label, value]) => (
        <div key={label}>
          <dt>{label}</dt>
          <dd>{display(value)}</dd>
        </div>
      ))}
    </dl>
  );
}
function Issue({ children }: { children: React.ReactNode }) {
  return (
    <span
      className="highlight-issue"
      title="Prototype value: no matching API data"
    >
      {children}
    </span>
  );
}

/** Browser-only inspection keeps a captured image and state together; streams continue underneath. */
export const Diagnostics = observer(() => {
  const store = useRootStore();
  const current = store.videoStore.latestGameState;
  const [frozen, setFrozen] = useState<{
    state: GameStateData | null;
    image: string;
    at: string;
    elapsed: number;
  } | null>(null);
  const [error, setError] = useState("");
  const [overlays, setOverlays] = useState(true);
  const [expanded, setExpanded] = useState(false);
  const [filter, setFilter] = useState("all");
  const preview = useRef<HTMLDivElement>(null);
  const state = frozen ? frozen.state : current;
  const status = state?.observation_status;
  const perception = status?.perception;
  const identity = status?.observation_identity;
  const [elapsed, setElapsed] = useState(0);
  const receipt = useRef(performance.now());
  useEffect(() => {
    receipt.current = performance.now();
    setElapsed(0);
  }, [current]);
  useEffect(() => {
    // This clock only ages pushed evidence locally; it must never fetch state.
    const timer = window.setInterval(
      () => setElapsed(performance.now() - receipt.current),
      diagnosticRefreshMs,
    );
    return () => clearInterval(timer);
  }, []);
  const evidenceAge = (value: number | null | undefined) =>
    value == null ? undefined : value + (frozen ? frozen.elapsed : elapsed);
  const cueAge = evidenceAge(perception?.cue_evidence_age_ms);
  const cueStatus = cueAge == null || !perception || cueAge > perception.cue_evidence_max_age_ms
    ? "No recent observation"
    : perception.cue ? "Observed" : "Not detected";
  function pause() {
    if (frozen) {
      setFrozen(null);
      return;
    }
    const image = preview.current?.querySelector("img");
    const overlay = preview.current?.querySelector("canvas");
    if (!image?.naturalWidth) {
      setError("Wait for a camera image before pausing inspection.");
      return;
    }
    try {
      const canvas = document.createElement("canvas");
      canvas.width = image.naturalWidth;
      canvas.height = image.naturalHeight;
      const context = canvas.getContext("2d");
      if (!context) throw new Error("Image capture is unavailable.");
      context.drawImage(image, 0, 0);
      if (overlays && overlay && Number(overlay.style.opacity) !== 0)
        context.drawImage(overlay, 0, 0, canvas.width, canvas.height);
      setFrozen({
        state: current ? toJS(current) : null,
        image: canvas.toDataURL("image/png"),
        at: new Date().toLocaleTimeString(),
        elapsed,
      });
      setError("");
    } catch {
      setError(
        "This camera does not permit browser image capture. Inspection remains live.",
      );
    }
  }
  function download() {
    const url = URL.createObjectURL(
      new Blob([JSON.stringify(state, null, 2)], { type: "application/json" }),
    );
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = "inspection-state.json";
    anchor.click();
    URL.revokeObjectURL(url);
  }
  const balls = (state?.balls ?? []).filter(
    (ball) =>
      filter === "all" ||
      (filter === "stale"
        ? ball.is_observed === false
        : ball.is_observed !== false),
  );
  return (
    <>
      <div className="page-heading">
        <h1>Live diagnostics</h1>
        <div className="flex gap-2">
          <Button asChild><Link to="/events">View events</Link></Button>
          <Button onClick={pause}>
            {frozen ? <Play /> : <Pause />}
            {frozen ? "Return to live" : "Pause inspection"}
          </Button>
        </div>
      </div>
      <div className="freshness-strip">
        <div>
          <span>Latest state received</span>
          <strong>{current ? age(elapsed) + " ago" : "Not available"}</strong>
        </div>
        <div>
          <span>Ball observation age</span>
          <strong>
            {age(evidenceAge(perception?.last_ball_observation_age_ms))}
          </strong>
        </div>
        <div>
          <span>Cue observation age</span>
          <strong>{age(evidenceAge(perception?.cue_evidence_age_ms))}</strong>
        </div>
        <Link to="/configuration" className="flex items-center gap-3">
          <Settings2 size={16} />
          <div>
            <span>Active limits</span>
            <strong>{display(status?.configuration_revision)}</strong>
          </div>
        </Link>
      </div>
      {error && (
        <p role="status" className="notice">
          {error}
        </p>
      )}
      {frozen && (
        <p role="status" className="notice">
          Inspection paused · Image and diagnostic snapshot frozen at{" "}
          {frozen.at}. Backend processing continues.
        </p>
      )}
      <div className={`diagnostics-grid ${expanded ? "camera-expanded" : ""}`}>
        <Card className="camera-panel">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Camera size={16} />
              Camera preview
            </CardTitle>
            <Button
              size="icon"
              variant="ghost"
              aria-label="Expand camera"
              onClick={() => setExpanded(!expanded)}
            >
              <Maximize2 />
            </Button>
          </CardHeader>
          <div
            className={`diagnostics-preview ${!overlays ? "overlays-hidden" : ""}`}
            ref={preview}
          >
            <LiveView presentation="diagnostics" />
            {frozen && (
              <img
                className="frozen-preview"
                src={frozen.image}
                alt="Paused camera inspection"
              />
            )}
          </div>
          <div className="camera-toolbar">
            <label className="flex items-center gap-2">
              <input
                type="checkbox"
                checked={overlays}
                disabled={!!frozen}
                onChange={(event) => setOverlays(event.target.checked)}
              />
              Detection overlays
            </label>
            <span className="text-muted-foreground">
              Ball centers show trusted or last-known table positions.
            </span>
          </div>
        </Card>
        <Card className="interpretation-panel">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Activity size={16} />
              Current interpretation
            </CardTitle>
          </CardHeader>
          <CardContent>
            <Rows
              rows={[
                ["Perception phase", perception?.phase],
                [
                  "Layout trust",
                  perception
                    ? perception.layout_trusted
                      ? "Usable"
                      : "Not usable"
                    : undefined,
                ],
                ["Perception reason", perception?.trust_reason],
                ["Table visibility", perception?.table_visibility?.status],
                ["Visibility reason", perception?.table_visibility?.reason],
                [
                  "Guidance validity",
                  perception
                    ? perception.guidance_valid
                      ? "Valid"
                      : "Not valid"
                    : undefined,
                ],
                [
                  "Motion observation age",
                  age(evidenceAge(perception?.latest_motion_check_age_ms)),
                ],
                [
                  "Ball freshness limit",
                  age(perception?.layout_verification_max_age_ms),
                ],
                [
                  "Cue freshness limit",
                  age(perception?.cue_evidence_max_age_ms),
                ],
                [
                  "Motion freshness limit",
                  age(perception?.motion_evidence_max_age_ms),
                ],
              ]}
            />
            <p className="text-xs text-muted-foreground mt-4">
              New state messages do not renew retained measurements.
            </p>
          </CardContent>
        </Card>
        <Card className="observations-panel">
          <CardHeader>
            <CardTitle>
              Ball observations{" "}
              <span className="text-muted-foreground">{balls.length}</span>
            </CardTitle>
            <select
              aria-label="Filter ball observations"
              value={filter}
              onChange={(event) => setFilter(event.target.value)}
            >
              <option value="all">All observations</option>
              <option value="observed">Observed</option>
              <option value="stale">Retained</option>
            </select>
          </CardHeader>
          <div className="overflow-auto">
            <table>
              <thead>
                <tr>
                  <th>Track</th>
                  <th>Position</th>
                  <th>Identity</th>
                  <th>Kind</th>
                  <th>Confidence</th>
                  <th>Source frame</th>
                </tr>
              </thead>
              <tbody>
                {balls.map((ball) => (
                  <tr key={ball.id}>
                    <td className="font-mono">{ball.id}</td>
                    <td className="font-mono">
                      {ball.position.x}, {ball.position.y}{" "}
                      <span className="text-muted-foreground">
                        × {ball.position.scale.join(", ")}
                      </span>
                    </td>
                    <td>
                      {ball.is_cue_ball
                        ? "Cue ball"
                        : ball.number != null
                          ? ball.number
                          : (ball.type ?? "Unknown")}
                    </td>
                    <td>
                      {ball.position_kind ??
                        (ball.is_observed === false ? "Retained" : "Observed")}
                    </td>
                    <td>
                      {ball.confidence_kind === "unavailable"
                        ? "Not available"
                        : `${Math.round(ball.confidence * 100)}%`}
                    </td>
                    <td className="font-mono">
                      {display(ball.source_observation?.frame_id)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            {!balls.length && (
              <p className="empty-message">
                {filter !== "all"
                  ? "No ball observations match this filter."
                  : perception?.empty_layout_verified && perception.layout_trusted
                    ? "The detector verified an empty table."
                    : "No ball observations available."}
              </p>
            )}
          </div>
          <details className="p-4 border-t">
            <summary>Prototype data awaiting API support</summary>
            <p className="mt-3">
              Per-track observation age:{" "}
              <Issue>84 ms · 420 ms · 980 ms · 1.84 s</Issue>
            </p>
          </details>
        </Card>
        <Card className="cue-panel">
          <CardHeader>
            <CardTitle>Cue observation</CardTitle>
          </CardHeader>
          <CardContent>
            <Rows
              rows={[
                ["Status", cueStatus],
                [
                  "Observation age",
                  age(evidenceAge(perception?.cue_evidence_age_ms)),
                ],
                ["Source frame", perception?.cue?.source_frame_id],
                ["Shaft start", perception?.cue?.shaft_start_px],
                ["Shaft end", perception?.cue?.shaft_end_px],
                ["Axis unit", perception?.cue?.axis_unit],
                ["Signed direction", perception?.cue?.signed_direction],
                ["Confidence", perception?.cue?.confidence],
              ]}
            />
            <div className="mt-4 text-xs">
              <p>
                Tip{" "}
                {display(
                  state?.cue?.tip_position
                    ? [state.cue.tip_position.x, state.cue.tip_position.y]
                    : perception?.cue?.signed_direction === "toward_start"
                      ? perception.cue.shaft_start_px
                      : perception?.cue?.signed_direction === "toward_end"
                        ? perception.cue.shaft_end_px
                        : undefined,
                )}
              </p>
              <p>
                Direction unit{" "}
                {perception?.cue?.signed_direction ? (
                  display(
                    perception.cue.axis_unit.map(
                      (value) =>
                        value *
                        (perception.cue?.signed_direction === "toward_start"
                          ? -1
                          : 1),
                    ),
                  )
                ) : (
                  <Issue>0.96, −0.28</Issue>
                )}
              </p>
            </div>
          </CardContent>
        </Card>
      </div>
      <Card className="mt-5">
        <details>
          <summary className="p-4 flex items-center gap-2 cursor-pointer">
            <Code2 size={16} />
            Technical inspector
          </summary>
          <div className="p-4 border-t">
            <Rows
              rows={[
                [
                  "Producer",
                  identity?.engine_instance_id ??
                    status?.engine_instance_id ??
                    status?.vision_instance_id,
                ],
                [
                  "Capture session",
                  identity?.session_id ?? status?.source_session_id,
                ],
                ["Source frame", identity?.frame_id ?? status?.source_frame_id],
                ["Configuration revision", status?.configuration_revision],
                ["Layout revision", perception?.layout_revision],
                ["Calibration revision", status?.calibration_revision],
                ["Rectification revision", status?.rectification_revision],
                [
                  "Snapshot revision",
                  identity?.snapshot_revision ?? status?.snapshot_revision,
                ],
                ["Transport sequence", status?.transport_sequence],
              ]}
            />
            <Button onClick={download} disabled={!state} className="my-4">
              <Download />
              Download inspected state
            </Button>
            <pre className="json-preview">{JSON.stringify(state, null, 2)}</pre>
          </div>
        </details>
      </Card>
    </>
  );
});
