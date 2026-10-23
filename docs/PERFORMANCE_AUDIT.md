# Repeatable real-time performance audit

> **Rust migration status:** `backend.main` and `backend.dev_server` no longer
> launch an application server. `scripts/performance-audit-live.sh` builds the
> native audit runner, Rust gateway, live detector, and GStreamer capture
> helper, launches the gateway as a child, waits
> for a fresh observed state, then runs bounded warm-up and measurement windows.
> The Python collector remains only as an off-device reference for legacy
> trajectory, viewer, and compatibility scenarios; the state-only live path
> does not need Python.

This workflow measures fresh, correct state delivery on one development
machine. It separates vision and state production, trajectory work, and video
viewer interference. The measurements support a ranked investigation backlog;
they do not automatically change production settings or select an optimization.

State correctness is the first gate, followed by fresh-state latency and unique
source-observation throughput. The configured 50 ms value is a reference
deadline for reporting misses, not a result or a guarantee.

The legacy `backend/tools/quick_perf_check.py` has been retired. Its synthetic
all-zero image and Python detector timings did not qualify the supported Rust
runtime. Use the managed live audit below to measure the complete Rust service;
an isolated detector microbenchmark remains a separate future tool.

## Development-machine integration smoke

The managed Rust live path was exercised against the target machine's remote
MJPEG stream from the development Mac. The latest five-second window received
24 complete state messages: 19 fresh, unique observed frames and five repeated
states. It had no WebSocket sequence gaps; the source frame counter reported
three gaps totaling 21 skipped IDs. Read-completion-to-publication age was
95.9/98.0/98.4 ms at p50/p95/p99; local receipt intervals were 103/499/658 ms.
First observed state arrived after 3.84 seconds, and the managed gateway
exited with code zero without forced termination. The recorded run is
[20261023-023743-3b43f629](../thoughts/performance-audits/20261023-023743-3b43f629/report.md).

This short run verifies end-to-end integration only. It is not a performance
baseline: it has one five-second window, zero warm-up, no verified accuracy
labels, and no detector identity or camera-content hash. Do not use its
percentiles to rank implementation changes.

## Opt in

Audit collection is off by default. The former Python server flag
`python -m backend.main --performance-audit` is retired. To run the full Rust
gateway and live detector as one supervised audit, pass the gateway's normal
CLI arguments as repeated `--server-argument` values. The runner does not
start warm-up until it sees a state with a fresh observation identity and
`ball_positions_status: observed`; it writes a failure artifact and stops the
server if that state does not arrive before the configured startup timeout.

```bash
scripts/performance-audit-live.sh \
  --url ws://127.0.0.1:8001/api/v1/websocket/ws \
  --server-argument=--config \
  --server-argument=/path/to/live-config.json \
  --server-argument=--project-root \
  --server-argument=/path/to/project \
  --server-argument=--http-port \
  --server-argument=8001 \
  --server-argument=--start-live \
  --warmup-seconds 0 --duration-seconds 30 --repetitions 1
```

`--server-argument` forwards one literal argument per occurrence; no shell
string is evaluated. Set `--startup-timeout-seconds` if the selected model or
input needs longer to produce its first state. On Unix the runner starts the
gateway in a private process group, sends `SIGTERM` to that group after the
audit, waits for the gateway and its detector/capture descendants to exit, and
sends `SIGKILL` to the group only if the configured shutdown timeout expires.
It then reaps the gateway process. Managed runs require a loopback WebSocket
URL.

For `--start-live` managed-server runs, the versioned report records the
effective configuration and capture-settings hashes, camera-profile identity,
ONNX model and manifest hashes, source-checkpoint hash, live detector and
capture-helper executable hashes, configured provider, model size, thresholds,
and ball/cue class mapping. It writes hashes rather than raw config, camera
URLs, or model contents. Provider placement and dynamic runtime version remain
unavailable. Input-video hashing is unavailable because the receiver handles
state messages rather than video bytes; detector accuracy is unavailable
without temporally aligned labels.

This remains a measurement harness, not an application service. The legacy
Python compatibility commands below are retained as differential references;
they do not turn on instrumentation in the supported Rust gateway.
The live wrapper builds with `camera-gstreamer-source`; the host needs the
GStreamer development libraries and the `soup`, `multipart`, `jpeg`, `app`,
`videoconvertscale`, and `videorate` runtime plugins required by the remote
MJPEG helper.

The native `billiards-performance-audit-run` command owns the state-only
repeatable workflow. It defaults to a 30-second warm-up followed by three
120-second measured windows; each connection is closed between windows. Use
`--soak` for one 20-minute bounded-retention run after the same warm-up. It
writes a schema-versioned `summary.json`, a Markdown report, one JSON report and
bounded JSONL detail file per measured window, and a `failure.json` if a
connection fails after the run folder is created. The report records revision,
dirty-worktree fingerprint, operating system, architecture, CPU count, host,
kernel, memory, and effective limits. A managed `--start-live` run adds the
model, runtime-binary, capture-helper, effective-configuration, and camera
profile identities described above. External state streams do not disclose
detector identity; network input-video hashing and accuracy evidence remain
unavailable in either mode.

For a managed Rust live server, `--live-pipeline-timing-audit` also enables a
bounded loopback side channel from the detector. The runner binds the receiver
before starting the server and supplies the ephemeral endpoint and validated
limits to the child. The option requires `--server-executable` and a forwarded
`--start-live`. With the option omitted, neither process opens the timing
socket or allocates its timing-retention queue. The shell entry point accepts
the same Rust CLI options:

```bash
scripts/performance-audit-live.sh \
  --url ws://127.0.0.1:8001/api/v1/websocket/ws \
  --server-argument=--config \
  --server-argument=/path/to/live-config.json \
  --server-argument=--project-root \
  --server-argument=/path/to/project \
  --server-argument=--http-port \
  --server-argument=8001 \
  --server-argument=--start-live \
  --live-pipeline-timing-audit \
  --warmup-seconds 0 --duration-seconds 30 --repetitions 1
```

The producer queue defaults to 2,048 records and 64 KiB per record. The
runner retains at most 200,000 records or 128 MiB across the run; the receiver
accepts one loopback producer. Producer connect and write deadlines default to
1,000 ms. The queue, record, retention, and deadline bounds are configurable
with the corresponding `--live-audit-*` options. The receiver uses its own
monotonic clock domain and records the exact measurement-window boundaries;
producer timestamps are retained as separate fields and never subtracted from
state-receiver timestamps. Every measured window gets a timing summary and
bounded JSONL detail file under `runs/`. Missing producer footer, producer or
receiver loss, or incomplete state/timing identity joins marks evidence
partial. These records measure detector stages, but do not supply temporal
ground truth or camera exposure timestamps.

Run a short smoke check or the standard repeated audit with:

```bash
scripts/performance-audit-run.sh \
  --url ws://127.0.0.1:8080/api/v1/websocket/ws \
  --warmup-seconds 0 --duration-seconds 10 --repetitions 1

scripts/performance-audit-run.sh \
  --url ws://127.0.0.1:8080/api/v1/websocket/ws \
  --output-root thoughts/performance-audits
```

Add `--resource-process-id <pid>` to sample one process on a separate worker;
the interval defaults to 1,000 ms and respects the sampler's minimum refresh
period. The receiver supports both the application WebSocket contract (welcome,
explicit `state` subscription, and JSON heartbeats) and the Rust gateway's
direct pushed-state contract. Its heartbeat interval defaults to 20 seconds,
separate from WebSocket control-frame pings. To run one receiver window without
warm-up, repetitions, or report-directory orchestration, use the thin receiver
entry point:

```bash
scripts/performance-audit-state.sh \
  --url ws://127.0.0.1:8001/api/v1/websocket/ws \
  --duration-seconds 120 \
  --output-json /tmp/live-audit.json
```

To include one process's resources, add `--resource-process-id <pid>` and,
optionally, `--resource-sample-interval-ms <milliseconds>` (default `1000`,
minimum `200` to respect the process-CPU sampler's update interval).
Resource sampling is opt-in, runs on its own worker thread, binds to the
observed process start-time value, and retains no more than the configured
sample limit. It reports per-core CPU percentage, accumulated CPU
milliseconds, resident-memory bytes, unavailable samples, and retention loss.
The receiver reports complete/incomplete and uncorrelated states, component
status totals, unique observed frames, repeats, frame gaps, source restarts,
and out-of-order frames. Schema version 4 adds this source and completeness
evidence. Resource-enabled reports use the same schema version. The
`billiards-performance-audit-server-summary` binary reduces retained server
JSONL events for vision, integration, delivery, video, trajectory, process
resources, and viewers. The Python collector now calls that Rust reducer after
applying its existing comparable-clock measurement window. A shared golden
fixture checks its output against the former Python reducer. Its input defaults
are 1 MiB per line, 200,000 events, and 128 MiB total; boolean values are not
accepted as numeric measurements. Inspect its configurable limits with:

```bash
cargo run --locked --manifest-path perception/Cargo.toml \
  --bin billiards-performance-audit-server-summary -- --help
```

The `billiards-performance-audit-delivery-summary` command separately computes
state and trajectory publication-to-receipt, socket-send-to-receipt, and
component-age summaries. It only subtracts producer and receiver timestamps
when each side reports exactly one matching verified OS monotonic clock
domain. Its JSON Lines protocol starts with a config record, then accepts
`client` or `server` event records; it uses the same documented input bounds.
The Python collector keeps collection-window selection and delegates these
joins and percentiles to the Rust command.

The `billiards-performance-audit-overhead-summary` command compares matched
instrumentation-off and state-only runs. The thin shell entry point accepts its
bounded JSON comparison document on stdin:

```bash
scripts/performance-audit-overhead-summary.sh \
  < /path/to/overhead-comparison.json
```

It checks workload, machine, revision, model, input, relevant configuration,
collector settings, repetition count, and delivery sample count before it
reports whether instrumentation stays within budget. The Python collector
still selects the baseline report and assembles the comparison document, then
delegates the decision to Rust.

When the collector subscribes to `trajectory`, it forwards each receipt to a
long-lived Rust summary process through a bounded, nonblocking queue. The
default queue capacity comes from `performance_audit.event_queue_capacity`
(2,048). The summary retains up to `performance_audit.history_size` samples
(10,000 by default, capped at 1,000,000); the collector allows the configured
event export limit plus the protocol's config/finish records, capped at
5,000,000 records. Lines default to 1 MiB, total input to 128 MiB, and summary
output to 1 MiB. Queue or record-limit losses appear in
`trajectory_bridge_dropped_records`; JSONL trace-export losses appear
separately in `trajectory_exporter_dropped_records`. Deliberate trace sampling
is not counted as loss. The report labels the selected summary engine and
compares stable Rust fields with the retained Python reference while parity is
being established. State-only runs do not start this process.

The native reducer can also be used directly with a JSON Lines file or stream:

```bash
scripts/performance-audit-trajectory-summary.sh \
  < /path/to/trajectory-receipts.jsonl
```

The protocol requires one `config` record first, zero or more
`trajectory_receipt` records containing the collector's flattened receipt
event, and one final `finish` record with duration and optional loss counts.

The compatibility collector retains Python event collection and clock-window
selection, warm-up/repetition orchestration, trajectory WebSocket subscription,
provenance, baseline selection, accuracy evaluation, and Markdown rendering
for its legacy API, trajectory, and viewer scenarios. The managed state-only
workflow, its optional detector timing receiver, correlation, provenance, and
Markdown/JSON report generation are Rust-owned. The compatibility
responsibilities remain migration work; the Rust reducers do not change their
measurement window or capture path.

| Setting | Default | Purpose |
| --- | ---: | --- |
| `performance_audit.enabled` | `false` | Global activation |
| `performance_audit.state_enabled` | `true` | Camera-to-state and state-delivery evidence |
| `performance_audit.trajectory_enabled` | `false` | Prediction queue, compute, freshness, and delivery |
| `performance_audit.trajectory_workload_enabled` | `true` | Keep or suppress trajectory admission and compute in an active audit; ignored when auditing is off |
| `performance_audit.viewers_enabled` | `false` | Per-viewer frame read, encoding, and backpressure |
| `performance_audit.warmup_seconds` | `30` | Warm-up before each measured run |
| `performance_audit.measured_seconds` | `120` | Duration of each ordinary run |
| `performance_audit.repetitions` | `3` | Ordinary measured repetitions |
| `performance_audit.soak_seconds` | `1200` | One separate 20-minute soak duration |
| `performance_audit.sample_rate` | `1.0` | Fraction of eligible audit events accepted by the exporter |
| `performance_audit.detailed_frame_sample_rate` | `0.05` | Fraction of vision frames that receive nested stage timings |
| `performance_audit.detailed_event_sample_rate` | `0.05` | Fraction of high-frequency pipeline and delivery details timed and exported |
| `performance_audit.history_size` | `10000` | In-memory timing, percentile, and source-identity deduplication window cap |
| `performance_audit.event_queue_capacity` | `2048` | Per-process asynchronous export queue cap |
| `performance_audit.event_export_limit` | `200000` | Per-process written event cap |
| `performance_audit.export_drain_timeout_seconds` | `5.0` | Maximum post-run wait for the API exporter queue to drain |
| `performance_audit.resource_sample_interval_seconds` | `1.0` | CPU, memory, event-loop, and task sample interval |
| `performance_audit.latency_reference_budget_ms` | `50.0` | Reference deadline used for miss counts |
| `performance_audit.instrumentation_overhead_budget_percent` | `2.0` | Maximum accepted throughput loss or p95 increase |
| `performance_audit.minimum_overhead_repetitions` | `3` | Minimum measured runs required in each overhead-comparison arm; maximum `10000` |
| `performance_audit.minimum_overhead_delivery_samples` | `100` | Minimum source-to-client latency samples required in every compared run; maximum `10000000` |
| `performance_audit.output_directory` | `thoughts/performance-audits` | Run reports and per-process event files |

The shared configuration validator checks these values before accepting a
configuration update. Frame detail selection happens before allocating its
timing context or recording nested spans. It uses a deterministic hash of the
frame number so separate runs sample the same frame identities without a
shared random-number lock. High-frequency event detail uses the same selector;
the audit reports candidate, selected, and sampled-out counts separately from
records lost to the bounded export queue. Correctness and lifecycle anomalies
remain unsampled. In-memory profiler counters retain exact total and failure
counts; exported stage events and server-side event summaries describe only
the selected detail sample and retained records. Their names say `recorded` or
`sampled` so they cannot be mistaken for full pipeline totals. Use the headless
subscriber's state accumulator for delivered-state throughput and repeated
publication counts; it also discloses the bounded identity-deduplication
window. The general `sample_rate` then controls exporter acceptance for the
remaining eligible events. Failed frames, source discontinuities, and the
sampling footer bypass probabilistic sampling, while the bounded queue and
export cap still apply. Queue drops, sampled-out records, export-limit drops,
and write failures appear in status and reports so missing trace data cannot
look like a performance win. A run ranks recorded vision stages by cumulative
exclusive work and reports per-stage p50/p95/p99; this attributes work but does
not predict speedup from removing it.

Video events retain the capture thread's `camera_read_seconds` for selected
frames, then time undistortion and shared-memory publication. The main loop
cannot select a frame before `get_latest_frame()` returns its identity, so
`camera_frame_queue_retrieval_seconds` is explicitly unavailable rather than
timing every empty polling iteration.

The compatibility API status endpoints are
`GET /api/v1/vision/performance/audit` and
`GET /api/v1/vision/performance`. The supported Rust gateway does not currently
expose these endpoints. Its WebSocket state stream can still be measured with
the native receiver, but server-side audit event joins and the complete
repeatable report workflow are not yet wired to the Rust gateway.

The headless collector accepts both Python state identities and native Rust
scene identities. Rust observations retain engine, session, and full frame ID;
numeric frame suffixes are used only to count ordering and gaps. Source ticks
keep their declared clock and stage and are never interpreted as wall time. In
containers without the Git checkout, revision and dirty-worktree identity are
reported as unavailable; overhead comparisons require known matching source,
worktree, model, and input hashes. Rust replay metadata uses the configured ONNX
model, source checkpoint, inference executable, and replay input directory.
The primary `unique_source_observations` and
`unique_observations_per_second` metrics count new frames whose ball positions
are explicitly marked `observed`. `unique_source_frame_updates` counts every
newly received source frame, including cached, failed, and skipped updates;
those states are counted separately so stale state cannot inflate fresh
detection throughput.

For a finite API replay, subscribe before replay begins or collect while the
source is actively running. A collector that attaches after startup can miss
early frames and include idle time after a one-shot replay ends; that result is
a startup smoke, not a steady-state throughput measurement.

For a bounded paced audit, configure `perception.rust_replay.replay_repeat_count`
to repeat the recorded PNG/JPEG corpus after each complete pass. It defaults to
`1`, preserving one-pass replay. `maximum_replay_frames` defaults to `100000`
and caps the total scheduled frames (`corpus image count × repeat count`); the
supervisor rejects a configuration that exceeds the cap before launching the
detector. The Rust replay assigns each repeated frame a new monotonically
increasing source identity, while the recorded pixels and hashes repeat. This
keeps delivery accounting well-defined, but repeated visual content is load
input only and does not count as new accuracy evidence.

Choose enough repeats to cover the collector's full warm-up and measurement
window, plus the time between backend startup and collector attachment:

```text
repeat_count >= ceil(
  replay_rate_fps × (warmup_seconds + measured_seconds × repetitions + startup_margin_seconds)
  / corpus_image_count
)
```

Keep the resulting total at or below `maximum_replay_frames`. For separate
accuracy evaluation, use temporally annotated recordings rather than looping a
short corpus.

## Run the live delivery audit

The retired compatibility-service collector uses a headless WebSocket
subscriber and defaults to the local state stream. Its defaults are 30 seconds
of warm-up and three 120-second measurements. It requires the compatibility
server's audit status endpoints and cannot collect a Rust gateway audit until
that integration is implemented. Use `--help` to inspect the available
off-device qualification modes; no Python server startup command is supported.

The following instrumentation-overhead procedure applies only to the retained
Python compatibility service. Its server-start commands are retired and it
must not be used to select optimizations for the Rust product runtime. A
matched Rust gateway baseline and instrumentation-enabled run are still
required before drawing those conclusions.

The `BT_PERFORMANCE_AUDIT_CLOCKS` instructions below apply only to the legacy
Python producer. They do not configure the Rust application.

The comparison is inconclusive unless it has at least three repetitions on the
same host and hardware, matching source revision and dirty-worktree contents,
the same model, input, and relevant settings, an instrumentation-off baseline,
and state-only instrumentation-on runs. Both runs must subscribe to state
alone, with no video viewer or artificial subscriber delay, and use the same
warm-up, measured duration, audit sampling/retention/export settings, and
collector queue/sample settings. Each run needs at least 100 age samples. A
pass requires throughput loss and p95 increase to stay within the configured
2% budget. Inconclusive results call for longer matched runs.

To add a measurement domain, edit its independent setting and restart the
backend. Keep state enabled while measuring interference so the state stream
remains the primary outcome:

```json
{
  "performance_audit": {
    "enabled": true,
    "state_enabled": true,
    "trajectory_enabled": true,
    "viewers_enabled": false
  }
}
```

Collect state and trajectory together:

```bash
uv run python backend/tools/collect_performance_audit.py \
  --streams state,trajectory --scenario trajectory-enabled
```

The trajectory-enabled setting controls measurement only. Prediction work
continues by default. To isolate prediction interference, run matched state
audits with trajectory_workload_enabled set first to false and then to true;
restart the backend between arms and keep the input, detector, calibration,
stream subscription, and measurement settings fixed. Keep the audit domains
identical between these two workload arms so the measured difference is
prediction work rather than added tracing. Set trajectory_enabled to true in
a separate trajectory-detail run when you need prediction queue and compute
timings. This control is effective only while auditing is active and preserves
phase/calibration invalidation.

For viewer interference, turn on `viewers_enabled` and add a local MJPEG
subscriber. The application video route is `/api/v1/stream/video`:

```bash
uv run python backend/tools/collect_performance_audit.py \
  --video-url http://127.0.0.1:8000/api/v1/stream/video \
  --scenario state-plus-video-viewer
```

The MJPEG subscriber starts before warm-up so its connection and encoder state
settle before measurement. Per-run viewer byte, chunk, and frame counts exclude
the warm-up interval. Detailed viewer events use deterministic sampling by
source writer and frame number on the shared-memory path, so all viewers select
the same source frames for fan-out and cross-run A/B comparisons. Frame number
is the sampling key; writer PID remains event provenance only. The legacy
raw-camera route uses a per-client sequence because it does not expose a
shared-memory source frame identity. The read span needs two monotonic clock reads before the shared-memory
frame identity is available; only selected frames export that span and receive
JPEG/send timing and paired event records. Selected detail events bypass the
second probabilistic event filter, while queue capacity and export limits still
apply. New collector summaries use schema version 6 and distinguish retained
event counts from headless-client delivery counts.

Live MJPEG routes read the newest shared-memory frame on each iteration rather
than queueing frames for a viewer. The response-writer wait is limited by
`api.stream.performance.maximum_send_backpressure_seconds` (validated default
`0.5` seconds); a client exceeding the limit is disconnected after its current
chunk and can reconnect on the latest frame. This bounds server-side backlog,
but a JPEG already accepted by TCP cannot be recalled and browser-display delay
is not measured.

The headless collector bounds source high-water identities by `history_size`.
Its state and trajectory summaries report the retained identity count, capacity,
and number of evictions. A source identity that returns after eviction is
counted as new because duplicate and ordering history for that identity is no
longer retained. Trajectory queue, executor, calculator-lock, compute, and
publication timing distributions combine published, calculation-failed,
stale-discarded, and publication-failed outcomes when that timing is available;
the outcome counts remain separate so a fast successful subset cannot hide slow
or failed work.

To exercise bounded slow-consumer behavior, add a measured delay after each
headless receive:

```bash
uv run python backend/tools/collect_performance_audit.py \
  --subscriber-delay-ms 100 --scenario slow-subscriber
```

The slow subscriber is an explicit stress scenario. Compare it with the healthy
subscriber run and verify that the existing latest-work scheduling and
per-client queue isolation remain intact.

Run the separate bounded-memory soak with:

```bash
uv run python backend/tools/collect_performance_audit.py --soak
```

The soak uses one 20-minute measured interval after the configured warm-up.
The collector retains at most `history_size` latency samples and a bounded
number of exported events. Percentiles describe the retained most-recent sample
window; the summary separately reports total observations, retained count, and
whether the sample window was truncated. Override run settings with `--warmup`,
`--duration`,
`--repetitions`, `--history-size`, `--queue-capacity`, `--sample-rate`, and
`--event-export-limit` when a test calls for different limits.

## Recorded-input detector replay

Use `backend/tools/benchmark_vision_profiles.py` to test the real configured
detector against recorded frames. It refuses to run candidate comparisons if
the configured detector model did not load. Static annotated-frame labels are
scored alongside phase, cadence, and resolution candidates.

Real-time paced replay follows recorded presentation timestamps and evaluates
freshness behavior:

```bash
uv run python backend/tools/benchmark_vision_profiles.py \
  --video /path/to/recorded-table-session.mp4 \
  --pace realtime --frames 360 --dimensions native \
  --cadence-factors 1 --crop-modes full-frame \
  --output /tmp/vision-realtime.json
```

Unpaced replay measures processing capacity, not live freshness:

```bash
uv run python backend/tools/benchmark_vision_profiles.py \
  --video /path/to/recorded-table-session.mp4 \
  --pace capacity --frames 360 --dimensions native \
  --cadence-factors 1 --crop-modes full-frame \
  --output /tmp/vision-capacity.json
```

Keep the detector, model, resolution, calibration, and effective settings fixed
when comparing pace modes or candidate profiles. Each report records the input
hash, model hash, source revision, dirty-worktree identity, host, runtime,
hardware, and relevant configuration. Capacity results must not be presented
as end-to-end live latency.

Camera frame-rate qualification is separate from detector cadence. The
`vision.camera.fps` setting requests a camera mode that must be negotiated and
verified; it does not implement software sampling. A target-Linux replay paced
unique 4K video frames at several software arrival rates through the Rust
latest-frame mailbox. It processed 3.317 observations/s from 3.333 arrivals/s
with no pending replacements; at 3.75 FPS one pending frame was replaced, and
replacement and p95 completion latency rose above that. This is a single-run
capacity screen with synthetic post-decode timestamps, no connected camera,
and no temporal accuracy labels. It does not justify a webcam setting. See the
[unique-video sampling report](../thoughts/performance-audits/software-sampling-rate-20261021/sampling-rate-comparison.md).
Use continuous-video source frame IDs and temporal labels to evaluate software
sampling; count intentional sampling skips separately from overload
replacements.

The opt-in Rust physical-camera qualification is a state-only supervisor run,
separate from the API delivery audit. It requires an explicit capture profile
and a Rust live settings file. During the measured window it samples the Rust
inference child process through the native `billiards-performance-audit-resources`
helper (CPU utilization, CPU time, and RSS) on the configured
`performance_audit.resource_sample_interval_seconds` interval. Override the
interval for this mode with `--rust-live-resource-sample-interval`. The report
retains at most 50,000 sample details per repetition and summarizes them; if
`history_size` requests more, the effective and requested retention limits are
both reported. The helper returns a bounded JSON line (16 MiB maximum). Sample
timestamps remain in its own monotonic domain and are not compared with state
receipt times. The report labels whether CPU percentiles and first/last/range
statistics cover all samples or only the retained tail. `sysinfo` exposes
process start time at one-second resolution, so PID reuse within the same second
can be indistinguishable. Missing process identity, oversized helper output, or
unsupported resource sampling is reported as unavailable. This mode does not
include the API, frontend, trajectory, or video viewers.

The same run enables receiver-side scene transport counters after warm-up and
turns them off at the end of the measured window. It reports accepted snapshot
rate, serialized payload bytes (minimum, mean, and maximum), transport sequence
gaps, and the one-slot queue's occupancy before enqueue and replacement
fraction. These scalar counters retain no per-frame history and do not split
Rust serialization, socket write, kernel buffering, or Python parsing. They
use the collector process monotonic window and do not subtract Rust timestamps.

## Temporal detector accuracy evaluation

Score a Rust scene-snapshot JSONL replay against versioned frame labels and
the exact recorded video. The Rust evaluator verifies the video SHA-256 and
geometry, uses the annotation presentation times, rejects non-raw or stale
ball detections, and reports localization, tracking, cue, freshness, source
session, and calibration metrics. Rust monotonic timestamps are not video
times and are not used to claim capture-to-result latency.

```bash
scripts/evaluate-temporal-localization.sh \
  --video /path/to/recording.mp4 \
  --annotations /path/to/temporal-labels.json \
  --scene-snapshots /path/to/rust-scenes.jsonl \
  --output thoughts/performance-audits/temporal-evaluation/report.json
```

The wrapper is portable development glue; the Rust command owns validation,
matching, scoring, and report generation. It requires `ffprobe` on `PATH` by
default; pass `--ffprobe-executable` to select another installation. The
geometry query has a 30-second timeout, captures at most 64 KiB of standard
output and retains at most 8 KiB of error details. Set
`--ffprobe-timeout-seconds` to change the timeout. The
default matching gate is half the labeled ball radius and the freshness budget
is 100 ms of recorded-video presentation time. The evaluator reads at most
128 MiB of annotations, hashes at most 8 GiB of video, and reads at most 1 GiB
of scene JSONL with a 16 MiB per-line limit. Every limit and scoring threshold
can be changed through the CLI. Results identify all input paths and hashes,
the frame-index offset, and the number of scored engine frames.

Use the same labeled clip and frame mapping when comparing detector or
scheduling changes. Ball-number identity is intentionally not scored by this
tool. The Python evaluator remains an offline parity reference until the Rust
command passes the wider temporal corpus and a recorded detector replay; do
not treat the shared golden fixture alone as qualification.

## Repeatable workload matrix

Keep scenario clips and camera settings unchanged across each comparison. Use
the live audit to record state delivery and the replay benchmark to measure
detector processing. Keep simulated detectors confined to instrumentation and
overload tests; their timing is not evidence about real inference cost.

| Workload | Input or action | Main evidence |
| --- | --- | --- |
| State-only baseline | Stable, aiming, or settled table | State correctness, freshness, and unique observations |
| Aiming | Player approaches and addresses cue ball | Cue detection quality, component reuse age, state freshness |
| Dense table | Many visible and closely spaced balls | Ball precision/recall, position error, CPU, allocations proxy |
| Motion | Break, rolling balls, and collisions | Tracking continuity, observation gaps, inference cadence |
| Settlement | Motion stops and state stabilizes | Freshness, unnecessary repeated work, map stability |
| Occlusion | Cue or balls temporarily hidden | Identity continuity, stale-state age, recovery |
| Camera interruption/restart | Stop and restore the capture process | Restart identity, rejected stale frames, recovery time |
| Calibration change | Apply a new table calibration during capture | Calibration/map revision propagation and invalidation |
| Trajectory interference | Same input with trajectory domain enabled | Prediction queue, executor, lock and compute waits; state impact |
| Video interference | Same input with a headless MJPEG viewer | Encode, read, backpressure, CPU, state impact |
| Slow subscriber | Add fixed receiver delay | Queue depth, coalesced updates, drops, healthy-client isolation |

Use at least one recorded clip per temporal workload and retain its hash in the
report. Where the current still-image label set does not cover the scenario,
mark accuracy unavailable and collect temporal annotations before making a
production recommendation.

## Read the results

Each run creates a versioned `summary.json`, a Markdown `report.md`, and a
bounded client trace under `thoughts/performance-audits/<run-id>/`. The API and
video capture processes keep separate event files under `live/` and
`video-process/`; each event includes schema version, run, process, host,
sequence, monotonic timestamp, and an OS boot clock-domain identity when the
platform exposes one. The report records trace loss and effective settings as
well as revision, dirty-worktree content, hardware, Python runtime, model and
input hashes.

The report separates:

- unique source observations processed, published, and received from repeated
  state publications and stream-sequence gaps, duplicates, or reordering;
- source-publication-to-state, source-publication-to-client-receipt, and
  state-publication-to-client-receipt p50/p95/p99 latency, sample counts, and
  deadline misses;
- source and per-component age, observed/reused/skipped/failed status, queue
  depth, overload drops, stale prediction rejection, payload size, and delivery
  fan-out;
- trajectory queue delay, executor wait, calculator-lock wait, compute,
  coalescing, invalidation, and publication;
- sampled CPU time/utilization, RSS trend, event-loop lag, and outstanding
  tasks.

The shared-memory epoch timestamp is software publication/retrieval time,
never a camera sensor exposure timestamp. End-to-end source intervals use the
optional `perf_counter_ns` timestamp carried in reserved shared-memory header
space. They are calculated only when the collector and producer report the
same verified OS boot clock domain; a matching hostname alone is insufficient.
During an instrumentation-off baseline, the status endpoint exposes this
read-only clock identity without creating an audit recorder, queue, sampler, or
exporter. The ordinary wall-clock publication-age estimate is retained as
secondary context and is not used to rank latency. For remote or unidentified
clock domains, cross-process monotonic latency is explicitly unavailable;
inspect the separately measured segments. Allocation counts, accelerator
metrics not exposed by the detector backend, browser rendering, and physical
display latency are reported as unavailable.

For the opt-in Rust live path, the audit reports parent framed-record read
duration separately from parent-read-complete-to-inference-start. When the
supervised GStreamer helper is used, it also reports helper-local appsink wait,
RGB allocation, and RGB mapping/copy durations. The helper and parent values are
elapsed segments from different processes; they are never subtracted from each
other. The parent framed-read span includes pipe transfer and producer
backpressure, but is not a separate helper write timer. Camera exposure remains
unavailable, and GStreamer PTS remains explicitly unverified as source time.
Per-repetition CPU, CPU-time, and RSS summaries identify the Rust inference
child separately from the audit runner.

Rust live-audit schema v3 adds `engine_total` to successful frame timings and
adds `inference_timings_ms` to observations rejected after inference for
freshness. Those rejected observations retain their normal identity and
nullable source timestamp, and never become scene or state publications. The
collector reports their capture, scheduling, preprocess, inference, decode,
completion-age, and configured-limit distributions separately from matched
published frames. If startup fails before a measured window exists, the report
labels these as all-retained startup diagnostics rather than a performance
baseline. The Python parser continues to accept archived schema-v1/v2 records.

On the Linux target, the illuminated-table camera run on the real 1280-input
CPU model rejected all 62 inference results against its 100 ms freshness limit:
inference p50/p95 was 267/292 ms, preprocessing 37.6/38.3 ms, and age at
inference completion 317/363 ms. The run failed before readiness and is not a
performance or accuracy result. This evidence identifies inference as the
first bottleneck to investigate; reducing capture FPS alone cannot make that
inference fit inside the current freshness budget. See the
[target startup diagnostic](../thoughts/performance-audits/target-physical-camera-qualification-20261022/20261022-040555-efa22995/report.md).

Source frame counters are tracked with their source publication timestamps. A
newer publication paired with a non-increasing counter starts a new sequence
epoch, including when a camera restarts inside the same process. Delayed
messages from an older timestamp epoch remain out of order rather than moving
the current high-water mark backward.

Accuracy is evaluated before speed. Current annotated stills support ball
precision/recall and center/radius error for their labeled cases. Temporal
identity continuity, cue position/angle quality, labeled observation
freshness, calibration-change behavior, interruption recovery, and broad
occlusion coverage need additional temporal annotations. A candidate without
accuracy evidence for its affected workload is not eligible for a production
recommendation.

## Rank follow-up experiments

Do not optimize from stage times alone. For every candidate, retain a matched
baseline and record the measured bottleneck, workload, resource evidence, and
accuracy result. Rank candidates by state correctness, then fresh-state p95 and
unique-observation throughput. Use this finding record:

| Field | Required evidence |
| --- | --- |
| Finding and rank | Named bottleneck, measured critical-path share, and affected scenario |
| Baseline | Revision, dirty-worktree identity, model/input hashes, configuration, runs, sample counts, and loss counts |
| Expected benefit | Which measured latency, throughput, CPU, memory, or isolation outcome should change |
| Accuracy implications | Ball precision/recall and error, identity continuity, cue quality, and freshness evidence or explicit gap |
| Complexity | Code ownership, operational impact, and deployment/platform cost |
| Validation experiment | A repeatable before/after run with unchanged input and correctness gates |

Investigate in this order when measurement evidence points there:

1. **Micro costs:** image copies and allocations; conversion and preprocessing;
   detector/classification work; coordinate conversion; JSON serialization;
   lock wait; and repeated work on unchanged source observations. Change one
   measured cost at a time and re-run accuracy scoring.
2. **Scheduling:** compare polling with observation-triggered updates, and
   inference cadence with state coverage, reuse age, gaps, and freshness. Keep
   the latest-work policy and prediction lifecycle unless a controlled test
   shows a better contract without stale state or lost correctness.
3. **Compute architecture:** test shared executor contention, dedicated workers,
   then process isolation only when queue, CPU, lock, and event-loop data show
   compute interference. Include serialization and copying in the comparison.
4. **Local transport:** compare the current path with direct in-process handoff
   and bounded loopback TCP using identical payloads, serialization, queue
   limits, and freshness semantics. Keep cross-platform same-machine operation
   as the default.
5. **Consumer isolation:** compare state-only, trajectory-enabled, and
   video/viewer-enabled runs; inspect per-client encoding, publish fan-out,
   slow-client drops, and healthy-client latency.

Any recommendation to change latest-work scheduling, prediction lifecycle,
slow-client isolation, or transport contracts requires measured evidence and
an Architecture Decision Record. Production optimization work is a separate
follow-up after this audit identifies a candidate and its validation test.
