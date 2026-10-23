# Performance and localization evaluation

The supported measurement path is Rust. Use the live audit for delivery speed,
freshness, skipped/repeated states, and optional detector-stage timings. Use the
temporal evaluator for accuracy against reviewed video labels. These answer
different questions and must be run on the same model, input, calibration, and
effective settings before comparing a change.

The live audit does not measure browser rendering or physical display latency,
does not capture viewer/video interference, and does not score accuracy. A
standalone recorded-video capacity/freshness benchmark is not implemented yet;
still-image inference comparisons are not a substitute. ADR-0084 proposes a
Rust video-replay tool to close that gap.

## Live state delivery

Against an already running Rust gateway, run the standard warm-up and three
measured windows:

```bash
scripts/performance-audit-run.sh \
  --url ws://127.0.0.1:8000/api/v1/websocket/ws
```

Defaults are 30 seconds of warm-up, three 120-second measured windows, and a
20-minute bounded-memory soak when `--soak` is selected. Use `--help` for
configurable windows, sample limits, optional trajectory receipts, and a
deliberately delayed subscriber. The runner reports message and unique-observed
state rates, repeated states, source and WebSocket gaps, freshness and delivery
latency distributions, and losses. It writes versioned JSON, bounded detail,
and Markdown under `thoughts/performance-audits/<run-id>/` while printing a
live terminal summary.

To supervise a local Rust server and collect correlated detector timings,
including optional detector-process CPU and memory, use:

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
  --sample-managed-live-process-resources
```

The wrapper builds the Rust server, live detector, and GStreamer helper, then
starts the Rust audit runner. Each `--server-argument` forwards one literal
argument. Timing collection is opt-in; without it, the detector does not open
the audit side channel or allocate its timing-retention queue. Timing records
and state receipts retain separate monotonic clock domains, so the audit does
not invent cross-process latency where clocks are not verified comparable.
Losses, incomplete identity joins, detector restarts, and unavailable resource
samples are disclosed in the report.

For an interference comparison, hold the same input, model, camera settings,
calibration, warm-up, and measurement windows constant. Compare state-only with
state plus `--include-trajectory`; record the exact effective configuration
for both runs. The subscriber-delay option is a stress test, not a production
client profile. The current runner does not provide a video-viewer workload, so
do not infer viewer impact from these runs.

The five-second remote-stream smoke in
[run 20261023-023743-3b43f629](../thoughts/performance-audits/20261023-023743-3b43f629/report.md)
only verified end-to-end wiring. It had no warm-up, sustained repetitions,
detector identity, or accuracy labels; its latency values are not a baseline
for ranking changes.

## Labeled temporal accuracy

The Rust evaluator scores Rust scene-snapshot JSONL against versioned temporal
labels bound to the exact input video hash and selected video-stream geometry:

```bash
scripts/evaluate-temporal-localization.sh \
  --video /path/to/recording.mp4 \
  --annotations /path/to/temporal-labels.json \
  --scene-snapshots /path/to/rust-scenes.jsonl \
  --output /path/to/report.json
```

Use `scripts/import-cvat-temporal-annotations.sh` to convert a reviewed CVAT
video export to the versioned annotation format. Supply the exact video hash
used for annotation and an explicit list of fully reviewed frame indices. The
importer uses `ffprobe` presentation timestamps and selected-stream geometry;
it rejects unreviewed or inconsistent inputs. Do not treat model-to-model
similarity, still-image scores, synthetic scenes, or repeated video as temporal
accuracy evidence.

The evaluator reports localization, continuity, cue, calibration/session
transition, observation freshness, and missing/extra detections. A production
recommendation still requires representative moving, settling, occluded,
colliding, interrupted-camera, and calibration-change recordings with reviewed
labels. Current still-image labels do not cover those temporal cases.

## Limits and interpretation

- Prior Python collector, Python detector-profile benchmark, and their Rust
  reducer/adapter commands are retired. Old reports remain historical data;
  they are not runnable workflows or current evidence contracts.
- The live audit measures the configured live system. It does not establish
  camera exposure-to-state latency when the source timestamps only shared-memory
  publication, and it does not measure browser render or display latency.
- The temporal evaluator measures accuracy against its supplied annotation
  set. It does not measure real hardware performance or guarantee the labels
  represent deployment conditions.
- Numbered-ball classification is outside the current model contract. The
  current detector distinguishes generic balls and cue only.
- Do not recommend an optimization from throughput alone. Preserve accuracy
  first, then compare fresh-state latency and unique observed-frame throughput.
