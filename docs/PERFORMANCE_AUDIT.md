# Performance and localization checks

Use the Rust audit tools for current state-delivery, recorded-video, and
labeled-localization measurements. Keep the detector, input, source geometry,
calibration, and effective settings fixed when comparing runs. Speed results
without matching accuracy evidence do not qualify a production change.

Audit artifacts default to a unique directory under the system temporary
directory (`billiards-performance-audits`). Pass an explicit output path when
a result needs to be retained. Generated reports are not stored in the source
checkout by default.

## Live state delivery

To measure a running Rust gateway:

```sh
scripts/performance-audit-run.sh \
  --url ws://127.0.0.1:8001/api/v1/websocket/ws
```

Defaults are a 30-second warm-up and three 120-second measured windows. A
20-minute bounded-memory soak is available with `--soak`; `--help` lists
configurable durations, sample bounds, and the slow-subscriber stress option.
The receiver reports unique observed states, repeated publications, gaps,
freshness and delivery latency percentiles, sample counts, drops, and losses.

To supervise live capture and collect nested detector timing records, use
`scripts/performance-audit-live.sh` with `--live-pipeline-timing-audit` and a
machine-specific configuration. Add `--gateway-websocket-timing-audit` to
measure gateway selection, serialization, and bounded socket-send durations
independently. Resource sampling is opt-in. Without timing options, the live
runtime allocates no timing receiver or per-frame timing retention.

Camera timing starts at software read completion; it is not camera exposure
time. Detector and gateway audit streams use separate process identities and
monotonic clock domains. The tool does not report cross-process durations
unless clocks are verified comparable. Socket-send duration is not proof of
client receipt, browser rendering, or display latency.

## Recorded-video evaluation

Run a PTS-paced latest-frame audit for freshness or an unpaced capacity audit.
The mode is mandatory because these answer different questions:

```sh
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

Use `--mode capacity` to process decoded frames as quickly as possible. Replay
predecodes into a bounded RGB spool, so it measures detector processing and
replacement behavior rather than live camera or decoder delay. Reports record
input/model hashes, source timestamps, effective settings, unique and replaced
frames, latency distributions, and explicitly unavailable measurements.

For temporal accuracy, pass reviewed Rust annotation labels with
`--annotations` and scene-snapshot output. Labels must identify the exact input
video hash and selected-stream geometry. Use
`scripts/import-cvat-temporal-annotations.sh` to import a reviewed CVAT video
export. The evaluator reports localization error, continuity, cue quality,
missing/extra detections, and freshness. It cannot support a production
recommendation when relevant moving, settling, occlusion, collision, camera
restart, or calibration-change cases lack reviewed labels.

To score the phase scheduler's fitted cue line as well, add
`--phase-cue-geometry-jsonl cue-geometry.jsonl` to replay. The trace is bounded
by `--maximum-detail-records`; it records only scheduled cue observations. The
replay passes it to the existing scorer, which reports undirected
`axis_angle_error_degrees` modulo 180° and uses signed endpoint evidence for
tip/butt errors when available. With compatible calibration, the cue endpoint
on the playing surface is the tip side when the fitted line crosses the table
boundary; endpoint contrast is a fallback when that geometry is unavailable.
For an already-created trace, pass
`--cue-geometry-jsonl` to `billiards-temporal-evaluate`.

Phase-aware replay defaults to the live Rust schedule and uses normalized
source presentation timestamps to make rate selection reproducible. Override
the complete phase schedule with `--phase-schedule-json` when comparing one
candidate. Follow the [phase-aware perception rate procedure](PHASE_AWARE_PERCEPTION.md#selecting-rates)
to alternate candidates, score reviewed labels, and raise only the rate tied to
a measured failure.

## Shared video fixture

`scripts/serve-test-video-mjpeg.sh` serves a pre-encoded saved video through a
bounded latest-frame MJPEG source. The live detector and browser can use the
same fixture to test source fan-out and viewer interference. This is a software
integration test; it does not establish camera exposure age, physical camera
behavior, browser rendering speed, or localization accuracy.

## Current measurement limits

- The current live server produces ball locations and cue status. It does not
  produce numbered-ball identity or trajectory predictions.
- Shared-memory and stream source timestamps describe software publication or
  media time, not camera exposure.
- Browser rendering and physical display latency are not measured.
- Synthetic data, model-to-model comparisons, and unlabeled replays validate
  plumbing only. Accuracy requires reviewed representative temporal labels.
