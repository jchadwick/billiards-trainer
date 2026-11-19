import { useEffect, useRef, useState } from "react";
import { observer } from "mobx-react-lite";
import axios from "axios";
import { Activity, Camera, HeartPulse, RefreshCw, Radio } from "lucide-react";
import {
  CartesianGrid,
  Line,
  LineChart,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import client from "../../api/axios-client";
import type { BrowserSessionSample } from "../../services/browser-session-metrics";
import { useRootStore } from "../../stores/context";
import { Button } from "../ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "../ui/card";
import {
  ChartContainer,
  ChartTooltipContent,
  type ChartSeries,
} from "../ui/chart";

type Evidence = {
  body?: Record<string, unknown>;
  status?: number;
  error?: string;
};
type HealthSnapshot = { live: Evidence; ready: Evidence; checkedAt: string };
const show = (value: unknown): string =>
  value == null
    ? "Not available"
    : typeof value === "object"
      ? JSON.stringify(value)
      : String(value);
const age = (value: number | null | undefined) =>
  value == null ? "Not available" : `${Math.round(value)} ms`;
const RATE_SERIES: ChartSeries[] = [
  {
    key: "receivedStateMessagesPerSecond",
    label: "Received state messages/s",
    color: "var(--chart-1)",
  },
  {
    key: "distinctObservedSourceFramesPerSecond",
    label: "Distinct observed source frames/s",
    color: "var(--chart-2)",
  },
];
const AGE_SERIES: ChartSeries[] = [
  { key: "ballObservationAgeMs", label: "Ball evidence age", color: "var(--chart-1)" },
  { key: "cueObservationAgeMs", label: "Cue evidence age", color: "var(--chart-2)" },
  { key: "motionObservationAgeMs", label: "Motion evidence age", color: "var(--chart-3)" },
];
const RANGE_MS: Record<string, number> = {
  "1m": 60_000,
  "5m": 5 * 60_000,
  "15m": 15 * 60_000,
};

function chartSamples(samples: BrowserSessionSample[], range: string) {
  const end = samples.at(-1)?.monotonicMs;
  if (end === undefined) return [];
  const start = end - (RANGE_MS[range] ?? RANGE_MS["5m"]);
  return samples
    .filter((sample) => sample.monotonicMs >= start)
    .map((sample) => ({ ...sample, time: sample.sampledAt }));
}

function formatRate(value: unknown): string {
  return typeof value === "number" && Number.isFinite(value)
    ? value.toFixed(1)
    : "Not available";
}

function formatAge(value: unknown): string {
  return typeof value === "number" && Number.isFinite(value)
    ? `${Math.round(value)} ms`
    : "Not available";
}

function RateChart({ data }: { data: ReturnType<typeof chartSamples> }) {
  return (
    <ChartContainer label="Browser-observed rates over time" series={RATE_SERIES}>
      <LineChart data={data} accessibilityLayer margin={{ top: 10, right: 8, bottom: 0, left: -12 }}>
        <CartesianGrid stroke="var(--chart-grid)" strokeDasharray="3 3" vertical={false} />
        <XAxis
          dataKey="time"
          tickFormatter={(value: string) => new Date(value).toLocaleTimeString([], { minute: "2-digit", second: "2-digit" })}
          minTickGap={28}
          tickLine={false}
          axisLine={false}
          tick={{ fill: "var(--chart-muted)", fontSize: 11 }}
        />
        <YAxis
          allowDecimals
          width={44}
          tickLine={false}
          axisLine={false}
          tick={{ fill: "var(--chart-muted)", fontSize: 11 }}
        />
        <Tooltip content={<ChartTooltipContent series={RATE_SERIES} formatValue={formatRate} />} />
        {RATE_SERIES.map((series) => (
          <Line
            key={series.key}
            dataKey={series.key}
            name={series.key}
            stroke={series.color}
            strokeWidth={2}
            dot={false}
            activeDot={{ r: 4 }}
            connectNulls={false}
            isAnimationActive={false}
          />
        ))}
      </LineChart>
    </ChartContainer>
  );
}

function AgeChart({ data }: { data: ReturnType<typeof chartSamples> }) {
  return (
    <ChartContainer label="Ball, cue and motion evidence age over time" series={AGE_SERIES}>
      <LineChart data={data} accessibilityLayer margin={{ top: 10, right: 8, bottom: 0, left: -12 }}>
        <CartesianGrid stroke="var(--chart-grid)" strokeDasharray="3 3" vertical={false} />
        <XAxis
          dataKey="time"
          tickFormatter={(value: string) => new Date(value).toLocaleTimeString([], { minute: "2-digit", second: "2-digit" })}
          minTickGap={28}
          tickLine={false}
          axisLine={false}
          tick={{ fill: "var(--chart-muted)", fontSize: 11 }}
        />
        <YAxis
          allowDecimals={false}
          width={52}
          tickFormatter={(value: number) => `${Math.round(value)} ms`}
          tickLine={false}
          axisLine={false}
          tick={{ fill: "var(--chart-muted)", fontSize: 11 }}
        />
        <Tooltip content={<ChartTooltipContent series={AGE_SERIES} formatValue={formatAge} />} />
        {AGE_SERIES.map((series) => (
          <Line
            key={series.key}
            dataKey={series.key}
            name={series.key}
            stroke={series.color}
            strokeWidth={2}
            dot={false}
            activeDot={{ r: 4 }}
            connectNulls={false}
            isAnimationActive={false}
          />
        ))}
      </LineChart>
    </ChartContainer>
  );
}

function Prototype({ children }: { children: React.ReactNode }) {
  return (
    <span
      className="highlight-issue"
      title="Prototype content; no matching API evidence"
    >
      {children}
    </span>
  );
}
function Fields({ values }: { values: [string, unknown][] }) {
  return (
    <dl className="detail-rows">
      {values.map(([key, value]) => (
        <div key={key}>
          <dt>{key}</dt>
          <dd>{show(value)}</dd>
        </div>
      ))}
    </dl>
  );
}

/** HTTP checks are snapshots, never inferred from socket connectivity or repeatedly polled. */
async function check(path: string, signal: AbortSignal): Promise<Evidence> {
  try {
    const response = await client.get<Record<string, unknown>>(path, {
      signal,
    });
    return { body: response.data, status: response.status };
  } catch (error) {
    // A 503 readiness response contains authoritative evidence of not-ready,
    // whereas a transport failure cannot establish whether the server is alive.
    if (axios.isAxiosError(error) && error.response)
      return { body: error.response.data, status: error.response.status };
    return {
      error:
        error instanceof Error ? error.message : "Health check unavailable",
    };
  }
}
export const HealthPage = observer(() => {
  const store = useRootStore();
  const state = store.videoStore.latestGameState;
  const status = state?.observation_status;
  const perception = status?.perception;
  const identity = status?.observation_identity;
  const diagnostics = store.diagnosticsStore;
  const diagnosticsSnapshot = diagnostics.snapshot;
  const diagnosticsFresh = diagnosticsSnapshot !== null
    && !diagnostics.isStale
    && diagnostics.connection === "connected";
  const activeProducerMetadata = diagnosticsFresh
    && diagnosticsSnapshot?.producer.state === "fresh"
    ? diagnosticsSnapshot.producer.metadata
    : null;
  const lastKnownProducerMetadata = diagnosticsSnapshot?.producer.last_known?.report.metadata
    ?? (!diagnosticsFresh ? diagnosticsSnapshot?.producer.metadata ?? null : null);
  const lastKnownProducerMetadataSource = diagnosticsSnapshot?.producer.last_known
    ? "last-known report"
    : "stale diagnostics snapshot";
  const producerMetadataEvidence = activeProducerMetadata
    ? { assessment: "active_fresh", metadata: activeProducerMetadata }
    : lastKnownProducerMetadata
      ? {
          assessment: diagnosticsSnapshot?.producer.last_known
            ? "inactive_last_known_report"
            : "inactive_stale_snapshot",
          report_age_ms: diagnosticsSnapshot?.producer.last_known?.report_age_ms,
          metadata: lastKnownProducerMetadata,
        }
      : { assessment: "unknown", metadata: null };
  const [snapshot, setSnapshot] = useState<HealthSnapshot>();
  const [loading, setLoading] = useState(false);
  const [range, setRange] = useState("5m");
  const [copied, setCopied] = useState("");
  const request = useRef<AbortController | null>(null);
  const mounted = useRef(false);
  async function refresh() {
    request.current?.abort();
    const controller = new AbortController();
    request.current = controller;
    setLoading(true);
    const [live, ready] = await Promise.all([
      check("/api/v1/health/live", controller.signal),
      check("/api/v1/health/ready", controller.signal),
    ]);
    if (!mounted.current || controller.signal.aborted) return;
    setSnapshot({ live, ready, checkedAt: new Date().toLocaleTimeString() });
    setLoading(false);
  }
  useEffect(() => {
    mounted.current = true;
    void refresh();
    return () => {
      mounted.current = false;
      request.current?.abort();
    };
  }, []);
  const connection = store.system.status.isConnected
    ? "Connected"
    : "Disconnected";
  const video = store.videoStore.delivery;
  const rateSamples = chartSamples(store.browserMetrics.samples, range);
  const latestMetrics = store.browserMetrics.latest;
  const rateUnavailable = latestMetrics?.gapReason != null;
  const receivedRate = rateUnavailable
    ? "Unobserved"
    : latestMetrics?.receivedStateMessagesPerSecond == null
      ? store.system.status.isConnected ? "Collecting" : "Unobserved"
      : formatRate(latestMetrics.receivedStateMessagesPerSecond);
  const distinctFrameRate = rateUnavailable
    ? "Unobserved"
    : latestMetrics?.frameRateUnknown
      ? latestMetrics.frameRateUnknown === "missing_identity"
        ? "Unknown identity"
        : "Retention limit"
      : latestMetrics?.distinctObservedSourceFramesPerSecond == null
        ? store.system.status.isConnected ? "Collecting" : "Unobserved"
        : formatRate(latestMetrics.distinctObservedSourceFramesPerSecond);
  const videoLabel = video?.state === "receiving"
    ? "Receiving identified frames"
    : video?.state === "loaded"
      ? "Image loaded · freshness unavailable"
      : video?.state === "stalled"
        ? "Stalled"
        : video?.state === "error"
          ? "Unavailable"
          : video?.state === "connecting"
            ? "Connecting"
            : video?.state === "unconfigured"
              ? "Not configured"
              : "Unknown";
  const runtime = {
    configuration_revision:
      status?.configuration_revision ?? identity?.configuration_revision,
    producer:
      identity?.engine_instance_id ??
      status?.engine_instance_id ??
      status?.vision_instance_id,
    capture_session: identity?.session_id ?? status?.source_session_id,
    layout_revision: perception?.layout_revision,
    calibration_revision:
      identity?.calibration_revision ?? status?.calibration_revision,
    rectification_revision:
      identity?.rectification_revision ?? status?.rectification_revision,
    diagnostics: {
      connection: diagnostics.connection,
      assessment: diagnosticsSnapshot
        ? diagnosticsFresh ? "current" : "last known · stale"
        : "unknown",
      received_at: diagnostics.receivedAt,
      gateway_instance_id: diagnosticsSnapshot?.gateway.instance_id ?? null,
      gateway_revision: diagnosticsSnapshot?.gateway.revision ?? null,
      snapshot: diagnosticsSnapshot,
    },
  };
  const liveLabel =
    snapshot?.live.body?.status === "alive"
      ? "Alive"
      : snapshot?.live.error
        ? "Unknown"
        : show(snapshot?.live.body?.status);
  const readyLabel =
    snapshot?.ready.body?.status === "ready"
      ? "Ready"
      : snapshot?.ready.body?.status === "not_ready"
        ? "Not ready"
        : "Unknown";
  const freshness = !perception
    ? "Unknown"
    : perception.layout_trusted && perception.guidance_valid
      ? "Usable"
      : "Degraded";
  const cards = [
    {
      title: "Server liveness",
      value: liveLabel,
      source: "Backend · liveness response",
      description: "Does the server respond?",
      icon: HeartPulse,
      detail: snapshot?.live,
    },
    {
      title: "Server readiness",
      value: readyLabel,
      source: "Backend · readiness response",
      description: "Is its localized state source ready?",
      icon: Activity,
      detail: snapshot?.ready,
    },
    {
      title: "Video availability",
      value: videoLabel,
      source: video?.mode === "identified" ? "Browser · identified frame receiver" : video?.mode === "native" ? "Browser · native image load" : "Browser · shared video receiver",
      description: video?.mode === "native"
        ? "The image has loaded, but ongoing frame freshness is unavailable."
        : "Image delivery is independent of the state connection.",
      icon: Camera,
      detail: video ? {
        state: video.state,
        mode: video.mode,
        configured: video.configured,
        started_at: video.startedAt,
        last_frame_receipt_at: video.lastFrameReceiptAt,
        last_frame_identity: video.lastFrameIdentity,
        error: video.error,
      } : undefined,
    },
    {
      title: "State delivery",
      value: connection,
      source: "Browser · WebSocket connection",
      description:
        "A connected socket does not establish measurement freshness.",
      icon: Radio,
      detail: {
        connected: store.system.status.isConnected,
        has_snapshot: !!state,
      },
    },
    {
      title: "Observation freshness",
      value: freshness,
      source: "State · layout trust and guidance validity",
      description:
        "Last received assessment; inspect its reported measurement ages below.",
      icon: Activity,
      detail: perception,
    },
  ];
  async function copyRuntime() {
    try {
      await navigator.clipboard.writeText(JSON.stringify(runtime, null, 2));
      setCopied("Runtime metadata copied");
    } catch {
      setCopied(
        "Copy unavailable. Select the runtime JSON below to copy it manually.",
      );
    }
  }
  return (
    <div className="space-y-5">
      <div className="page-heading">
        <h1>System health</h1>
        <div className="flex flex-wrap items-center gap-4">
          <span className="text-xs text-muted-foreground">
            Last checked {snapshot?.checkedAt ?? "Not checked"} · HTTP snapshot
          </span>
          <Button disabled={loading} onClick={() => void refresh()}>
            <RefreshCw />
            {loading ? "Checking health…" : "Refresh health"}
          </Button>
        </div>
      </div>
      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-5">
        {cards.map((card) => (
          <Card key={card.title}>
            <CardHeader>
              <CardTitle className="flex gap-2 items-center">
                <card.icon size={16} />
                {card.title}
              </CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-lg font-semibold">{card.value}</p>
              <p className="my-3 text-xs text-muted-foreground">
                {card.description}
              </p>
              <p className="text-xs text-muted-foreground">{card.source}</p>
              {card.detail && (
                <details className="mt-3 text-xs">
                  <summary className="cursor-pointer">Inspect detail</summary>
                  <pre className="json-preview mt-2">
                    {JSON.stringify(card.detail, null, 2)}
                  </pre>
                </details>
              )}
            </CardContent>
          </Card>
        ))}
      </div>
      <p className="notice">
        Liveness answers “Does the server respond?” Readiness answers “Is its
        state source ready?” Neither proves that every camera or perception
        stage is healthy. HTTP checks run on entry and Refresh health; state
        evidence arrives through the existing WebSocket.
      </p>
      <Card>
        <CardHeader>
          <div>
            <CardTitle>Backend diagnostic stages</CardTitle>
            <p className="text-xs text-muted-foreground mt-1">
              Complete pushed assessment from the diagnostics WebSocket. A
              stale snapshot is shown as last known evidence.
            </p>
          </div>
        </CardHeader>
        <CardContent>
          <div className="mb-4 flex flex-wrap items-center justify-between gap-3 text-xs text-muted-foreground">
            <span>
              Stream {diagnostics.connection} · {diagnosticsSnapshot
                ? `${diagnosticsFresh ? "Current" : "Last known · stale"} · revision ${diagnosticsSnapshot.gateway.revision}`
                : "No valid snapshot received"}
            </span>
            {diagnostics.protocolError && <span className="text-amber-300">{diagnostics.protocolError}</span>}
          </div>
          {diagnosticsSnapshot ? (
          <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
            {diagnosticsSnapshot.stages.map((stage, index) => (
              <div
                key={stage.stage}
                className="rounded-md border border-border p-4"
              >
                <span className="text-xs text-muted-foreground font-mono">
                  0{index + 1}
                </span>
                <h3 className="my-2 text-sm font-semibold">{stage.stage.replaceAll("_", " ")}</h3>
                <p className={`text-sm font-medium ${["failed", "stalled", "degraded"].includes(stage.status) ? "text-amber-300" : ""}`}>
                  {diagnosticsFresh ? stage.status : `Last known · ${stage.status}`}
                </p>
                <p className="mt-2 text-xs text-muted-foreground break-words">
                  {stage.reason_code}{stage.detail ? ` · ${stage.detail}` : ""}
                </p>
                <p className="mt-2 text-xs text-muted-foreground">
                  {stage.source} · progress age {age(stage.last_progress_age_ms)}
                </p>
                {stage.stage === "scheduled_perception_work" && (
                  <p className="mt-2 text-xs text-muted-foreground">
                    Scheduler {show(stage.scheduler_state)} · job {show(stage.active_job_kind)} · counts {show(stage.completed_jobs)}/{show(stage.expected_jobs)} complete, {show(stage.failed_jobs)} failed · last completion {age(stage.last_completion_age_ms)} · last failure {stage.last_failure_code ?? "None"} ({age(stage.last_failure_age_ms)})
                  </p>
                )}
                <details className="mt-3 text-xs">
                  <summary className="cursor-pointer">Inspect stage evidence</summary>
                  <pre className="json-preview mt-2">{JSON.stringify(stage, null, 2)}</pre>
                </details>
              </div>
            ))}
          </div>
          ) : (
            <p className="rounded-md border border-border p-4 text-sm text-muted-foreground">
              Backend stage assessments are unknown until a valid pushed snapshot arrives. Connection silence does not establish gateway failure.
            </p>
          )}
          <p className="mt-4 text-xs text-muted-foreground">
            Browser state and video delivery are reported in the shared evidence cards above. They remain independent from this backend stage assessment.
          </p>
        </CardContent>
      </Card>
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h2 className="font-semibold">Delivery &amp; observation trends</h2>
          <p className="text-xs text-muted-foreground">
            Browser-observed state delivery, distinct source-frame identities,
            and independently advancing evidence ages.
          </p>
        </div>
        <div className="flex gap-2" role="group" aria-label="Chart time range">
          {["1m", "5m", "15m"].map((value) => (
            <Button
              key={value}
              variant={range === value ? "secondary" : "outline"}
              aria-pressed={range === value}
              onClick={() => setRange(value)}
            >
              {value}
            </Button>
          ))}
        </div>
      </div>
      <div className="grid gap-5 md:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>State delivery rates</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <p className="text-2xl font-semibold tabular-nums">{receivedRate}</p>
                <p className="mt-1 text-xs text-muted-foreground">Received state messages/s</p>
              </div>
              <div>
                <p className="text-2xl font-semibold tabular-nums">{distinctFrameRate}</p>
                <p className="mt-1 text-xs text-muted-foreground">Distinct observed source frames/s</p>
              </div>
            </div>
            <div className="mt-4">
              <RateChart data={rateSamples} />
            </div>
            <p className="mt-2 text-xs text-muted-foreground">
              Rates use a rolling 10-second receipt window. Message counts
              include repeated or delayed state messages; source-frame rate
              counts only distinct accepted engine/session/frame identities.
            </p>
            {latestMetrics?.frameRateUnknown && (
              <p role="status" className="mt-2 text-xs text-amber-300">
                Source-frame rate is unknown because an accepted state lacked
                a complete frame identity.
              </p>
            )}
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle>Observation age</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-3 gap-3">
              <div>
                <p className="text-lg font-semibold tabular-nums">{age(latestMetrics?.ballObservationAgeMs)}</p>
                <p className="mt-1 text-xs text-muted-foreground">Ball</p>
              </div>
              <div>
                <p className="text-lg font-semibold tabular-nums">{age(latestMetrics?.cueObservationAgeMs)}</p>
                <p className="mt-1 text-xs text-muted-foreground">Cue</p>
              </div>
              <div>
                <p className="text-lg font-semibold tabular-nums">{age(latestMetrics?.motionObservationAgeMs)}</p>
                <p className="mt-1 text-xs text-muted-foreground">Motion</p>
              </div>
            </div>
            <div className="mt-4">
              <AgeChart data={rateSamples} />
            </div>
            <Fields
              values={[
                [
                  "Balls · last reported",
                  age(perception?.last_ball_observation_age_ms),
                ],
                ["Cue · last reported", age(perception?.cue_evidence_age_ms)],
                [
                  "Motion · last reported",
                  age(perception?.latest_motion_check_age_ms),
                ],
                [
                  "Ball freshness limit",
                  age(perception?.layout_verification_max_age_ms),
                ],
                [
                  "Cue freshness limit",
                  age(perception?.cue_evidence_max_age_ms),
                ],
              ]}
            />
            <p className="text-xs mt-3 text-muted-foreground">
              Ages advance locally from the last API evidence. Repeated
              publications do not renew unchanged evidence; disconnected and
              unobserved intervals remain chart gaps.
            </p>
          </CardContent>
        </Card>
      </div>
      <div className="grid gap-5 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Performance details</CardTitle>
          </CardHeader>
          <div className="overflow-auto">
            <table>
              <thead>
                <tr>
                  <th>Metric</th>
                  <th>Value / unit</th>
                  <th>Window</th>
                  <th>Evidence source</th>
                </tr>
              </thead>
              <tbody>
                {[
                  [
                    "Unique observed source frames",
                    "Live chart above",
                    "Rolling 10 s",
                    "Browser · accepted engine/session/frame identities",
                  ],
                  [
                    "Perception processing time",
                    "Not available",
                    "Not available",
                    "No backend timing instrumentation",
                  ],
                  [
                    "Dropped / replaced / skipped frames",
                    "Not available",
                    "Not available",
                    "Counters not exposed by backend",
                  ],
                  [
                    "Camera-to-display latency",
                    "Not available",
                    "Not available",
                    "Requires synchronized end-to-end timestamps",
                  ],
                ].map((row) => (
                  <tr key={row[0]}>
                    <td>{row[0]}</td>
                    {row.slice(1).map((value, index) => (
                      <td key={index}>
                        {row[0] === "Unique observed source frames"
                          ? value
                          : <Prototype>{value}</Prototype>}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="p-4 text-xs text-muted-foreground">
            Prototype inventory. Message age and processing duration are not
            end-to-end camera latency.
          </p>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle>Resource monitoring</CardTitle>
            <span className="text-xs text-muted-foreground">
              Backend integration
            </span>
          </CardHeader>
          <CardContent>
            {[
              ["CPU utilization", "%"],
              ["Memory usage", "MiB"],
              ["GPU utilization", "%"],
            ].map(([name, unit]) => (
              <div
                key={name}
                className="flex items-center justify-between gap-3 border-b border-border py-4 first:pt-0"
              >
                <div>
                  <h3 className="text-sm">
                    {name}{" "}
                    <span className="text-muted-foreground">({unit})</span>
                  </h3>
                  <p className="mt-1 text-xs text-muted-foreground">
                    <Prototype>
                      No metric endpoint · window unavailable
                    </Prototype>
                  </p>
                </div>
                <Prototype>Not available</Prototype>
              </div>
            ))}
            <p className="mt-3 text-xs text-muted-foreground">
              Resource metrics and retained history need additional backend
              support. Zero is not a substitute for missing data.
            </p>
          </CardContent>
        </Card>
      </div>
      <Card>
        <details>
          <summary className="p-4 cursor-pointer font-semibold text-sm">
            Runtime details{" "}
            <span className="font-normal text-muted-foreground">
              · Identities, revisions &amp; instrumentation
            </span>
          </summary>
          <CardContent className="border-t">
            <Fields values={Object.entries(runtime)} />
            <div className="detail-rows">
              {[
                [
                  "Gateway build",
                  diagnosticsSnapshot
                    ? `${diagnosticsFresh ? "Current" : "Inactive · last-known snapshot"} · ${diagnosticsSnapshot.gateway.build.package_name} ${diagnosticsSnapshot.gateway.build.package_version} · revision ${diagnosticsSnapshot.gateway.build.git_revision ?? "unknown"} · dirty ${diagnosticsSnapshot.gateway.build.dirty ?? "unknown"}`
                    : "Unknown · no valid diagnostics snapshot",
                ],
                [
                  "Producer build",
                  activeProducerMetadata
                    ? `Active · ${activeProducerMetadata.build.package_name} ${activeProducerMetadata.build.package_version} · revision ${activeProducerMetadata.build.git_revision ?? "unknown"}`
                    : lastKnownProducerMetadata
                      ? "Inactive · " + lastKnownProducerMetadataSource + " · " + lastKnownProducerMetadata.build.package_name + " " + lastKnownProducerMetadata.build.package_version + (diagnosticsSnapshot?.producer.last_known ? " · age " + age(diagnosticsSnapshot.producer.last_known.report_age_ms) : "")
                      : diagnosticsSnapshot?.producer.state === "fresh"
                        ? "Active producer metadata has no build details"
                        : "Unknown · no producer build metadata",
                ],
                [
                  "Active model",
                  activeProducerMetadata?.model
                    ? activeProducerMetadata.model.identity.artifact_sha256 + " · " + activeProducerMetadata.model.input_width_px + "×" + activeProducerMetadata.model.input_height_px + " · " + activeProducerMetadata.model.identity.classes.map((item) => item.class_name).join(", ") + " · " + activeProducerMetadata.model.requested_provider + " → " + (activeProducerMetadata.model.execution_provider ?? "execution provider unknown")
                    : lastKnownProducerMetadata?.model
                      ? "Inactive · " + lastKnownProducerMetadataSource + " model · " + lastKnownProducerMetadata.model.identity.artifact_sha256 + " · " + lastKnownProducerMetadata.model.identity.classes.map((item) => item.class_name).join(", ") + (diagnosticsSnapshot?.producer.last_known ? " · report age " + age(diagnosticsSnapshot.producer.last_known.report_age_ms) : "")
                      : "Unknown · no active or last-known model metadata",
                ],
                [
                  "Backend metric retention",
                  "Not available · no retained metric history exposed",
                ],
              ].map(([name, value]) => (
                <div key={name}>
                  <span>{name}</span>
                  {name === "Backend metric retention"
                    ? <Prototype>{value}</Prototype>
                    : <span>{value}</span>}
                </div>
              ))}
            </div>
            <details className="mb-3 text-xs">
              <summary className="cursor-pointer">Inspect producer build and model metadata</summary>
              <p className="my-2 text-muted-foreground">
                Active producer metadata is present only in fresh reports. Last-known data is historical and inactive.
              </p>
              <pre className="json-preview">{JSON.stringify(producerMetadataEvidence, null, 2)}</pre>
            </details>
            <Button className="my-4" onClick={() => void copyRuntime()}>
              Copy runtime metadata
            </Button>
            {copied && (
              <p role="status" className="text-xs mb-3">
                {copied}
              </p>
            )}
            <pre className="json-preview">
              {JSON.stringify(runtime, null, 2)}
            </pre>
          </CardContent>
        </details>
      </Card>
    </div>
  );
});
