# Repeatable real-time performance audit

This workflow measures fresh, correct state delivery on one development
machine. It separates vision and state production, trajectory work, and video
viewer interference. The measurements support a ranked investigation backlog;
they do not automatically change production settings or select an optimization.

State correctness is the first gate, followed by fresh-state latency and unique
source-observation throughput. The configured 50 ms value is a reference
deadline for reporting misses, not a result or a guarantee.

## Opt in

Audit collection is off by default. A normal application start creates no audit
recorder, sampling thread, bounded event queue, or exporter. Start the backend
with:

```bash
uv run python -m backend.main --performance-audit
```

The process-local switch is inherited by the video capture process and is not
saved to `config.json`. Alternatively, set `performance_audit.enabled` to
`true` in the selected configuration. The launch switch and configuration
switch activate collection; each domain switch then controls its own events.

| Setting | Default | Purpose |
| --- | ---: | --- |
| `performance_audit.enabled` | `false` | Global activation |
| `performance_audit.state_enabled` | `true` | Camera-to-state and state-delivery evidence |
| `performance_audit.trajectory_enabled` | `false` | Prediction queue, compute, freshness, and delivery |
| `performance_audit.viewers_enabled` | `false` | Per-viewer frame read, encoding, and backpressure |
| `performance_audit.warmup_seconds` | `30` | Warm-up before each measured run |
| `performance_audit.measured_seconds` | `120` | Duration of each ordinary run |
| `performance_audit.repetitions` | `3` | Ordinary measured repetitions |
| `performance_audit.soak_seconds` | `1200` | One separate 20-minute soak duration |
| `performance_audit.sample_rate` | `1.0` | Fraction of audit events retained |
| `performance_audit.history_size` | `10000` | In-memory timing and percentile sample cap |
| `performance_audit.event_queue_capacity` | `2048` | Per-process asynchronous export queue cap |
| `performance_audit.event_export_limit` | `200000` | Per-process written event cap |
| `performance_audit.export_drain_timeout_seconds` | `5.0` | Maximum post-run wait for the API exporter queue to drain |
| `performance_audit.resource_sample_interval_seconds` | `1.0` | CPU, memory, event-loop, and task sample interval |
| `performance_audit.latency_reference_budget_ms` | `50.0` | Reference deadline used for miss counts |
| `performance_audit.instrumentation_overhead_budget_percent` | `2.0` | Maximum accepted throughput loss or p95 increase |
| `performance_audit.output_directory` | `thoughts/performance-audits` | Run reports and per-process event files |

The shared configuration validator checks these values before accepting a
configuration update. Sampling and export are bounded. Queue drops, sampled-out
records, export-limit drops, and write failures appear in the report so missing
trace data cannot look like a performance win. A run ranks recorded vision
stages by cumulative exclusive work and reports per-stage p50/p95/p99; this
attributes work but does not predict speedup from removing it.

The API status endpoint is
`GET /api/v1/vision/performance/audit`. When state profiling is enabled,
`GET /api/v1/vision/performance` shows the current profiler window.

## Run the live delivery audit

The collector uses a headless WebSocket subscriber and defaults to the local
state stream. Its defaults are 30 seconds of warm-up and three 120-second
measurements:

```bash
uv run python backend/tools/collect_performance_audit.py
```

For comparable instrumentation-overhead runs, first start the backend with
audit disabled and save a baseline:

```bash
BT_PERFORMANCE_AUDIT_CLOCKS=1 uv run python -m backend.main
uv run python backend/tools/collect_performance_audit.py \
  --scenario instrumentation-off --quiet
```

Then restart with the audit enabled and point the collector at the baseline
`summary.json`:

```bash
uv run python -m backend.main --performance-audit
uv run python backend/tools/collect_performance_audit.py \
  --scenario state-audit-only \
  --compare-with thoughts/performance-audits/<baseline-run-id>/summary.json
```

For the instrumentation-off baseline, set
`BT_PERFORMANCE_AUDIT_CLOCKS=1` when starting the backend. This enables only a
monotonic publication timestamp in the existing shared-memory frame metadata;
it creates no recorder, queue, sampler, or exporter. The state audit enables
the same timestamp automatically, so the baseline and measured run use the
same source-clock path. Leave the variable unset for normal startup.

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

For viewer interference, turn on `viewers_enabled` and add a local MJPEG
subscriber. The application video route is `/api/v1/video`:

```bash
uv run python backend/tools/collect_performance_audit.py \
  --video-url http://127.0.0.1:8000/api/v1/video \
  --scenario state-plus-video-viewer
```

The MJPEG subscriber starts before warm-up so its connection and encoder state
settle before measurement. Per-run viewer byte, chunk, and frame counts exclude
the warm-up interval.

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
number of exported events. Override run settings with `--warmup`, `--duration`,
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
