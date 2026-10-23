# Performance and localization evaluation

The supported measurement path is Rust. Use the live audit for delivery speed,
freshness, skipped/repeated states, and optional nested capture-to-publication
timings. Each audited frame reports actual read-completion-to-snapshot elapsed
time separately from per-stage inclusive and exclusive durations. Use the
temporal evaluator for accuracy against reviewed video labels. These answer
different questions and must be run on the same model, input, calibration, and
effective settings before comparing a change.

The live audit measures backend state delivery and optional pipeline timing. It
does not measure browser rendering, physical display latency, or accuracy.
Viewer interference has a separate shared-source comparison described below;
that comparison measures backend changes while a browser viewer is open, not
browser frame rate or total browser resource use.
Recorded-video replay is available for PTS-paced freshness and unpaced detector
capacity. It predecodes the selected stream into a bounded RGB spool, so it does
not measure camera or live decoder delay and must not be presented as an online
capture benchmark. Reviewed labels are required for localization accuracy.

## Live state delivery

Against an already running Rust gateway, run the standard warm-up and three
measured windows:

```bash
scripts/performance-audit-run.sh \
  --url ws://127.0.0.1:8000/api/v1/websocket/ws
```

The wrapper builds and runs the subscriber in release mode so its own parsing
and JSON work do not carry debug-build overhead into receipt measurements.

Defaults are 30 seconds of warm-up, three 120-second measured windows, and a
20-minute bounded-memory soak when `--soak` is selected. Use `--help` for
configurable windows, sample limits, and a deliberately delayed subscriber.
The runner reports message and unique-observed state rates, repeated states,
source and WebSocket gaps, freshness and delivery latency distributions, and
losses. It writes versioned JSON, bounded detail, and Markdown under
`thoughts/performance-audits/<run-id>/` while printing a live terminal summary.

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

The wrapper builds the Rust server, live detector, GStreamer helper, and audit
runner in release mode before starting the run. Each `--server-argument`
forwards one literal argument. Timing collection is opt-in; without it, the
detector does not open the audit side channel or allocate its timing-retention
queue or per-frame span contexts. Frame spans cover host read-completion,
detector preprocessing/model execution/decode, calibration, tracking and state
assembly, serialization, and scene queue submission where those stages apply.
When audit timing is active, capture records also report capacity-one receiver
mailbox residence from software read-completion until dequeue, separately from
the remaining freshness/admission/scheduling interval before inference starts.
The measured host boundary is software read-completion, never camera exposure.
Timing records and state receipts retain separate monotonic clock
domains, so the audit does not invent cross-process latency where clocks are not
verified comparable. Losses, incomplete identity joins, detector restarts, and
unavailable resource samples are disclosed in the report.

To audit the Rust gateway's per-client WebSocket selection, message assembly,
JSON serialization, and deadline-bounded socket-send call independently, add
`--gateway-websocket-timing-audit` to the same managed-server command. This
requires `--start-live` on the managed server but does not require
`--live-pipeline-timing-audit`. Gateway records use a second bounded receiver,
footer, process identity, and monotonic clock domain; detector PID/resource
sampling remains tied only to detector records. Each measured window receives
separate `run-NNN-gateway-websocket-timing.json` and `.jsonl` artifacts and a
separate Markdown report section. The socket-send duration is an application call
boundary, not proof that the subscriber or display received the state. The
runner reports publication-to-client receipt and physical display latency as
unavailable because it has no verified shared clock/receipt correlation.
Keep the input and workload fixed when comparing gateway timing on and off,
and qualify instrumentation overhead against the 2% budget before using small
stage differences to rank production changes.

The live audit subscribes to detected state only. The Rust server has no
trajectory producer, so the audit does not allocate trajectory receipt buffers
or report prediction timing; those measurements are unavailable until an
active Rust prediction workload and output contract exist. The subscriber-delay
option is a stress test, not a production client profile. The current runner
does not provide a video-viewer workload, so do not infer viewer impact from
these runs. Any future prediction-interference experiment must measure actual
prediction work and state freshness together, as recorded in
[ADR-0095](architecture/decisions/0095-defer-trajectory-interference-switch-until-rust-workload-exists.md).

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

## Recorded-video detector replay

Use this path to compare the current Rust detector on a fixed recording. The
mode is mandatory because real-time freshness and unpaced capacity are distinct
measurements:

```bash
scripts/run-rust-video-audit.sh \
  --video /path/to/recording.mp4 \
  --onnx /path/to/model.onnx \
  --source-checkpoint /path/to/model.pt \
  --provider cpu \
  --image-size 1280 \
  --confidence 0.4 \
  --iou 0.45 \
  --mode realtime-latest-frame
```

Use `--mode capacity` to process every decoded source frame as quickly as
possible. The command records exact video/model hashes, selected stream and
timestamps, runtime/provider, repository revision and worktree identity,
effective inference settings, processing/replacement counts, and source-to-
inference latency percentiles. It writes versioned JSON and Markdown under
`thoughts/performance-audits/<run-id>/`, prints a concise terminal summary, and
can write a bounded `--details-jsonl` trace. The default run is one video pass;
the standard 30-second warm-up, three 120-second live windows, and 20-minute
bounded-memory soak belong to the separate live audit workflow above.

The default video resource caps are 8 GiB encoded input, 512 MiB ffprobe
JSON, 250k frames, 64M pixels per frame, 8 GiB decoded RGB spool, and 600
seconds per ffprobe/ffmpeg subprocess. A requested detailed trace keeps at most
1,000 records by default; traces are off unless requested. Each cap is
configurable on the command line. Source PTS is normalized to the first frame
and remains media-time provenance; it is never described as camera exposure
time.

For tracking evidence, use the existing `--detection-batches-jsonl` and
`--scene-snapshots-jsonl` outputs. Detection batches require explicit engine
and source-session identities; scene snapshots require the tracking class and
all effective tracking settings. Add `--annotations` with scene snapshots to
score them against reviewed labels in the same run. The tool checks annotation
video hash and selected-stream geometry. Without labels, the report explicitly
marks accuracy unavailable. A replay with synthetic or incomplete labels is
plumbing validation only; it cannot support a production setting
recommendation.

## Network-capture fixture

For software-only capture checks and detector/viewer interference experiments,
start `scripts/serve-test-video-mjpeg.sh`. It pre-encodes the saved video before
serving, then the Rust fixture paces a preloaded JPEG corpus and shares one
capacity-one latest-frame value across HTTP multipart MJPEG clients. Point both
`vision.camera.device_id` and `api.stream.viewer_url` at the fixture's
`/stream.mjpg` URL to let GStreamer capture and the browser `<img>` consume the
same stream. Set `BILLIARDS_TEST_VIDEO` to select a recording; use
`BILLIARDS_TEST_VIDEO_FPS` to override its average frame rate.

Pre-encoding keeps FFmpeg decode/encode work outside the measured windows and
the one producer sends the exact same JPEG frames to every client. This
validates software fan-out, multipart delivery, and viewer workload
interference. It still does not represent camera exposure or driver-buffer age,
qualify physical capture, or replace recorded-video replay for controlled
detector comparisons. The Rust fixture's real-socket tests exercise multipart
parsing, two-client delivery, latest-frame coalescing, client bounds, and
slow-reader isolation, and bounded shutdown with open streams. A repeated local
Mac browser-on/off comparison is
recorded in the [viewer-interference comparison](../thoughts/performance-audits/viewer-interference-20261023/comparison.md).
It found no material backend slowdown, but does not establish browser rendering
performance or small effects below the measurement noise.

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
- Recorded-video replay excludes offline codec startup and decode time from its
  measurement window. It reports those costs separately; its PTS-paced mode
  models detector-side latest-frame replacement rather than live capture/decode
  contention.
- Numbered-ball classification is outside the current model contract. The
  current detector distinguishes generic balls and cue only.
- Do not recommend an optimization from throughput alone. Preserve accuracy
  first, then compare fresh-state latency and unique observed-frame throughput.
