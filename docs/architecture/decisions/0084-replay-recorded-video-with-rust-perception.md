# ADR-0084: Replay recorded video with Rust perception

- Status: accepted
- Date: 2026-10-23
- Deciders: Project maintainers
- Supersedes: None
- Superseded by: None

## Context

The Rust inference executable currently accepts still images and image
directories. Its paced directory mode uses a configured constant frame rate,
so it cannot preserve variable-frame-rate source timing or directly evaluate
the current detector against source-timed video annotations. The existing
Python profile benchmark is tied to the retired Python `VisionModule`, uses
Python-specific phase and crop settings, and processes decoded frames
sequentially when it falls behind. Its “realtime” results therefore do not
represent the Rust capture path's latest-frame behavior.

The project needs one repeatable, Rust-owned recorded-video path for detector
localization and freshness experiments. It must preserve the source video
frame identity and presentation time, use the current Rust inference/tracking
contracts, and distinguish real-time freshness from unpaced capacity. Exported
traces can perturb timing, so accuracy-artifact runs and clean performance
runs must remain distinguishable. The Rust temporal evaluator reads the
selected video stream from annotation metadata; matching geometry alone is
insufficient when a file contains multiple same-sized streams.

## Decision

Add a dedicated Rust video-replay command and a thin shell entry point. Use
`ffprobe` to obtain the selected video stream's finite, ordered presentation
timestamps and geometry, and `ffmpeg` to decode that same stream as bounded
RGB frames without scaling. Reject a decode whose frame count, geometry, or
timeline cannot be reconciled. Bound subprocess time, output, decoded frame
count, and per-frame pixel allocation; terminate and reap child processes on
failure.

The real-time mode schedules by normalized source PTS and feeds the existing
capacity-one latest-frame mailbox. If inference is late, queued old frames may
be replaced; every emitted record retains the original source-frame ordinal
so label joins expose skips rather than renumbering history. The capacity mode
processes sequentially and reports processing throughput only. Both modes
reuse the configured Rust inference engine and validated detection-batch and
scene-snapshot pipeline; they do not reproduce Python detector, phase, crop,
or candidate-profile settings.

Keep source PTS as media-time provenance separate from local monotonic runtime
timestamps. Use the Rust temporal evaluator for scored runs and bind its
geometry query to the annotation's `source_video_stream_index`. Keep the
default performance path free of detailed synchronous trace writes; an
accuracy run that writes batches/scenes must report that artifact cost.

## Consequences

### Positive

- Current Rust detector and tracker are exercised against real recorded video.
- Variable-frame-rate identity, intentional/overload skips, and media-time
  freshness remain observable without treating stale cached state as a new
  detection.
- Capacity and real-time freshness evidence cannot be conflated by one FPS
  number, and the same annotation scorer is used for replay results.

### Negative

- The developer workflow depends on compatible `ffmpeg` and `ffprobe`
  executables; their versions and selected stream must be recorded.
- Decoding, inference, and tracking are qualified, but a recorded replay does
  not by itself measure physical camera exposure-to-state or browser display
  latency.
- JSONL artifact generation can affect throughput and must be run separately
  or disclosed.

## Alternatives considered

### Port the Python profile benchmark as-is

Port the Python profile benchmark as-is. Rejected because it exercises an
obsolete detector pipeline, its candidate knobs do not configure the current
Rust runtime, and sequential processing cannot reproduce latest-frame
replacement under overload.

### Use OpenCV video timestamps

Rejected as the authoritative timeline because its container timestamp
behavior can silently degrade to constant-frame-rate estimates. `ffprobe`
provides the explicit per-frame PTS source used by temporal annotation import.

### Keep still-image replay only

Rejected because it cannot measure temporal association, frame skipping,
tracking continuity, or freshness against actual source presentation timing.

## Verification

- Deterministic tests cover stream selection, finite/strictly ordered PTS,
  invalid or missing timestamps, geometry and frame-count mismatch, and
  maximum frame/pixel bounds.
- Subprocess tests cover output limits, timeouts, nonzero exits, short/extra
  decoded output, cleanup, and child termination/reaping.
- Replay tests cover irregular-PTS scheduling, latest-frame replacement with
  preserved source ordinals, all-frame capacity mode, and keeping PTS separate
  from monotonic runtime timestamps.
- The Rust all-target suite covers timeline parsing, strict stream/frame
  geometry, irregular-PTS latest-frame selection, bounded child output and
  timeouts, short/extra decoder output, completion deadlines, and report
  accounting. Strict Clippy and formatting checks pass.
- A real recorded-video detector-path smoke completed with the current ONNX
  model and Rust inference runtime; see the [paced smoke report](../../../thoughts/performance-audits/video-replay-smoke-20261023/real-recording/video-replay-report.md)
  and [capacity smoke report](../../../thoughts/performance-audits/video-replay-smoke-20261023/capacity/video-replay-report.md).
  These development-machine runs had no reviewed temporal labels and are not
  accuracy or production qualification evidence.
- Accuracy scoring is available through the Rust temporal evaluator when
  reviewed labels are supplied. A real recorded clip with representative
  labels must be scored before candidate settings can be recommended;
  synthetic detections prove only replay plumbing and overload behavior.
- There is not yet a permanent integration test that runs external ffmpeg,
  real model inference, scene export, and temporal scoring together. The
  command's external codec dependency and model artifacts make the current
  smoke a separately recorded developer verification, not a CI guarantee.
