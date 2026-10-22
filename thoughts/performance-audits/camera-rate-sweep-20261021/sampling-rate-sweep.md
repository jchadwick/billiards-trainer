# Camera-rate capacity sweep

Captured 2026-10-21 on the available Linux development host. This is a short,
relational capacity experiment to choose the next camera test, not production
qualification for the final target machine.

## Question and conclusion

The question was whether lowering the image arrival cadence lets the detector
keep up without replacing pending observations, and what that costs in update
freshness. On this host and workload, **3 FPS is the most defensible initial
camera trial**: the repeated 120-frame run processed every scheduled image,
with no pending-frame replacements, at 3.003 processed observations/second for
3.000 observed arrivals/second. Its p95 source-arrival-to-inference-completion
was 285 ms. 3.5 FPS also processed all 120 images, but had a worse p95 of
364 ms and provides less headroom. At 4 FPS one of 90 arrivals was replaced;
at 5 FPS seven of 30 were replaced.

This establishes a **replay capacity point**, not that 3 FPS is the right
operating rate for real play. A 3 FPS camera gives the detector a new image only
every 333 ms, which can miss ball movement or collisions between observations.
The camera was disconnected, the Coral accelerator was not used, and the replay
reused a 30-image corpus with synthetic timestamps. There are no temporal
labels here to score missed motion or localization accuracy. Keep 3 FPS as a
starting point for camera qualification only; do not adopt it as a production
setting until a connected-camera test passes the accuracy and freshness gates.

## Method

- Host: Linux x86_64, Intel Core i5-8600T, 16 GiB RAM, at `192.168.1.31`.
- Runtime: Rust replay executable, native OpenVINO CPU provider. The attached
  Coral was not selected or measured.
- Input: the same 30 extracted 3840x2160 JPEGs, repeated for sustained tests.
- Detector: `billiards-yolov8n.onnx`, 1280x1280 input, confidence 0.4, IoU
  0.45, three warm-up inferences. The model emits only `cue` and `ball` classes.
- The harness paced images at the requested rate and used its latest-frame
  input path. Source arrival and detector completion timestamps came from the
  target process's monotonic clock.
- Repeated runs were 4 x 30 images at 3.0 and 3.5 FPS, and 3 x 30 at 4.0 FPS.
  Other rates were single 30-image screening runs. These are short runs, not the
  planned three 120-second measurements.

Since the same still frames were replayed with synthetic spacing, this measures
whether the detector can service an arrival cadence. It does not reproduce
actual frame skipping from a continuous video: it does not retain video PTS,
original frame IDs, or real inter-frame motion. A lower replay cadence also
does not prove that a webcam supports the requested mode.

## Results

| Pacing | Processed / scheduled | Observed → processed FPS | Pending replacements | Source → inference start p95 | Source → inference complete p95 |
|---:|---:|---:|---:|---:|---:|
| 2.5 FPS, one pass | 30 / 30 | 2.489 → 2.514 | 0 | 93 ms | 273 ms |
| 3.0 FPS, 4 repeats | 120 / 120 | 3.000 → 3.003 | 0 | 91 ms | 285 ms |
| 3.5 FPS, 4 repeats | 120 / 120 | 3.501 → 3.502 | 0 | 152 ms | 364 ms |
| 4.0 FPS, 3 repeats | 89 / 90 | 3.997 → 3.923 | 1 | 255 ms | 440 ms |
| 5.0 FPS, one pass | 23 / 30 | 4.993 → 3.647 | 7 | 322 ms | 545 ms |

At 3.0 FPS the repeated run's p50/p95/p99 source-to-completion was
252/285/307 ms. At 3.5 FPS it was 250/364/390 ms. The single 3.0 and 3.5 FPS
passes had much higher p95 completion values (439 and 443 ms), demonstrating
that single-pass tail timings vary materially on this host. The repeated
results are better evidence for a capacity comparison, but still too short to
estimate long-run tail behavior. At 4 FPS the repeated run's p99 completion
reached 641 ms despite only one replacement.

The one-pass 4 FPS screening run happened to process all 30 images because the
run could drain work at the end; the repeated run exposed a replacement. For
this reason, processed count from a short finite run alone is not a sustainable
throughput measure.

## Interpretation and recommendation

The observed boundary is between 3.5 and 4 FPS for this particular host, model,
resolution, and replay path. At 5 FPS the latest-frame path avoids unbounded
backlog by replacing pending work, but the client receives fewer unique
observations than the source schedule. Lowering the input cadence reduces
replacements; it does not make each detector result arrive sooner. At 3.5 FPS,
completion p95 already exceeds the 286 ms source interval. At 3 FPS, completion
p95 is below the 333 ms interval, making it a more conservative candidate.

The next real-camera test should:

1. Inventory the webcam's supported width, height, pixel format, and frame-rate
   modes. The live Rust camera path requests an exact mode and checks negotiated
   settings; `vision.camera.fps` is a camera-mode request, not a separate
   software sampling control. Changing the camera profile can also change its
   profile/calibration identity.
2. With resolution and all detector settings held constant, test the nearest
   supported modes around 3 FPS (or a supported higher pair such as 5 and 10
   FPS if 3 is unavailable). Record negotiated—not requested—rate, capture
   drops, mailbox replacements, processed unique source IDs, state publication
   rate, and same-host receipt latency.
3. Replay the original saved videos with source timestamps and frame IDs. Compare
   30/15/10/5/3 FPS sampling against full-rate detections on the *same original
   frames*, then use temporal labels to score position error, missed fast
   motion/collisions, and track continuity. Do not infer sampling accuracy by
   repeating still frames.
4. Run moving, settled, occluded, camera-restart, and dense-table segments. Pass
   only if accuracy remains acceptable and processed unique observations track
   emitted camera observations without growing age or replacement counts.

If the webcam cannot deliver the desired low-rate mode, a software sampling
policy is a separate design choice: capture continuously, preserve original
frame identity and capture time, and count intentional skips separately from
overload drops. That keeps the option to run a high-rate motion/event path while
limiting expensive detector inference. Because fast moving balls are central to
the use case, a dual-rate policy may ultimately be preferable to reducing the
camera's physical capture rate, but it needs motion-labelled evidence first.

## Limits and evidence

- This run does not measure camera exposure or exposure blur, camera-to-state
  end-to-end latency, browser rendering, identity classification (the current
  model distinguishes only cue versus ball), or Coral inference.
- The camera was disconnected, and this host is not the final target. Its result
  is useful for relative comparison only.
- The provided data has no temporal ground truth for ball locations between
  frames. We cannot conclude that 3 FPS retains acceptable localization or
  collision detection accuracy.
- At 3.0 FPS, processed throughput matches arrivals with no replacements in a
  40-second test, but this is not the full repeated 120-second baseline or the
  20-minute memory test from the audit plan.

Raw run artifacts are in this directory: `2_5fps.json`, `3_0fps.json`,
`3_0fps-repeat4.json`, `3_5fps.json`, `3_5fps-repeat4.json`, `4_0fps.json`,
`4_0fps-repeat3.json`, and `5_0fps.json`. The host and provider smoke context is
recorded in
[`../target-rust-replay-smoke-20261021/target-smoke-notes.md`](../target-rust-replay-smoke-20261021/target-smoke-notes.md).

## Follow-up with unique video content

The 3.5 FPS result above used a repeated 30-still corpus. For a stronger
capacity comparison, see the [60-second unique-content software sampling
run](../software-sampling-rate-20261021/sampling-rate-comparison.md), which
paces real video frames through the latest-frame mailbox and reports same-window
input/completion counts, replacement losses, and completion latency. Its highest
tested no-replacement rate was 3.333 FPS. Neither experiment qualifies a real
camera rate or motion accuracy.
