# Real-time performance audit findings

> **Architecture-review correction (2026-10-19):** the earlier MPS attribution was not verified. A new single-frame diagnostic on the same production PT path reports `detector.device=mps` while the actual predictor and model parameters are on CPU. Historical runs did not capture actual placement. Their rates/latencies remain measured results, but they cannot establish MPS performance. `model_execute` measures the Ultralytics call, not accelerator kernels alone. See the [architectural review](../architectural-review-2026-10-19.md) and [diagnostic evidence](architecture-review-diagnostics.json).

## Workload and confidence

The audit used the real YOLO model on `test_data/demo3.mp4` (3840×2160 at 30 fps) on the local M4 Max. The detector reported MPS, but actual execution placement was not recorded and is now disputed by the diagnostic above. State was received by a headless local WebSocket client. The corrected 20-minute run sent an application heartbeat every 20 seconds and received all 26,001 state messages without client-side loss. Source timestamps represent shared-memory publication, not camera exposure.

The trajectory audit domain was disabled, but the application still calculated trajectories from the integration loop during the run. Thus “state-only” means state-only instrumentation and subscriber streams; it does not isolate state processing from trajectory computation. No browser rendering or display latency was measured.

## Findings, ranked by likely effect on fresh state

### 1. 4K capture and detector execution are the main measured critical-path costs

The 300-frame 4K replay processed at 15.31 frames/s in real-time mode, with 126.2 ms p95 processing time. In the corrected live soak, retained stage records show model execution p95 rising from 93 ms in the first measured minute to about 230–238 ms from minutes 8–17. Frame completion p95 followed the same rise, while recorded vision queue drops exceeded 8,300 during the measured window. A later fresh 3×30-second run on the M4 Max measured `model_execute` at 16.7 ms p95 and total sampled vision service at 29.9 ms p95. The short run does not invalidate the long-soak degradation; it shows that sustained behavior and input/device configuration need a matched repeat. The earlier soak's final server trace was truncated at its configured export limit, so its last portion is missing.

**Candidate experiment:** compare full-frame inference with a correctly applied lower-resolution or ROI path, using identical temporal clips and calibration. The earlier 720p setting did not improve results because motion assessment still ran on the full-resolution frame before resizing. Keep the experiment gated on temporal identity, cue quality, and position accuracy; static labels alone are insufficient.

**Expected benefit:** potentially the largest throughput and freshness gain. **Complexity:** medium. **Accuracy risk:** high until temporal labels cover fast motion, occlusion, and settlement.

### 2. Processing cadence is below the capture rate and state freshness degrades

The 20-minute receiver observed 21.67 state messages/s but only 17.83 unique source observations/s, with 9,638 source-frame gaps and a p95 source-publication-to-receipt latency of 293 ms over the receiver’s bounded 10,000-sample window. Its p50 was 85.8 ms and p99 319.7 ms. The 50 ms reference budget is therefore missed in the tail. The three instrumentation-off runs delivered 24.19 unique observations/s on average, with p95 latency between 92 and 109 ms.

**Candidate experiment:** compare polling and inference cadence against observation-triggered scheduling and latest-work processing. Measure unique observations, source age, queue drops, and temporal accuracy together. Preserve the current prediction lifecycle and latest-work behavior until an experiment demonstrates a safe change.

**Expected benefit:** fewer stale results and less wasted work. **Complexity:** medium. **Accuracy risk:** medium to high if cadence misses short motion or cue events.

### 3. Repeated per-update trajectory work and verbose logs confound the state baseline

The integration loop calculated trajectory predictions repeatedly while state was being published, even though trajectory instrumentation was disabled. The live process also emitted detailed per-ball trajectory logs at high frequency. This workload can consume CPU and event-loop time, so the current audit does not separate its contribution from vision or state work.

**Candidate experiment:** the audit now has a workload-level `trajectory_workload_enabled` switch that suppresses prediction admission while retaining lifecycle invalidation. The recorded state run above predates that isolation and still calculated predictions, so it cannot estimate interference. Repeat the same input with the switch off and on, then separately enable trajectory detail and compare queue, compute, freshness, and state latency. In another matched run, suppress or sample the per-ball hot-path logs. Do not infer a trajectory optimization from the older state-only instrumentation run.

**Expected benefit:** isolates consumer interference and may reduce repeated work. **Complexity:** medium. **Accuracy risk:** medium; prediction lifecycle and settlement behavior must remain correct.

### 4. Full-rate audit detail exceeded its overhead budget; sampled detail passed a short screen

The earlier matched comparison found that full-rate state instrumentation changed unique-observation throughput by −13.94% and increased p95 source-to-client latency by 71.31%, against a 2% overhead budget. Its corrected soak accepted 216,309 API events but retained only 200,000, dropping 16,309 at the export limit. The new deterministic 10% detail sampler retained exact observation counts and produced no queue, sample, export-limit, or writer losses in the later short comparison. The matched 3×30-second runs passed the configured comparison: throughput was +2.08% with audit enabled and p95 receipt latency was −3.30%. Treat this as a short relational pass; repeat the documented 3×120-second and 20-minute checks before claiming the default audit overhead or memory bounds.

**Candidate experiment:** keep the sampled detail path and run the default-duration repeated comparison plus a bounded soak. Retain the event cap and verify memory and queue loss across all producer processes.

**Expected benefit:** trustworthy, lower-interference measurements. **Complexity:** low to medium. **Accuracy risk:** low; rare stalls, drops, failures, and restarts bypass detail sampling.

### 5. Test worker isolation only after reducing inference and audit costs

During the instrumented soak, the API process used about 2.8 CPU cores at its median and 3.1 at p95; event-loop lag reached 178 ms p95. The video process used about 0.8 cores at p95. API RSS ranged from about 668 MiB to 989 MiB and ended near 838 MiB; video RSS ranged from about 576 MiB to 738 MiB and ended near 689 MiB. This single run shows no monotonic memory growth, but does not establish a leak-free bound. These numbers are directional because the audit itself exceeds its overhead budget and accelerator utilization is unavailable.

**Candidate experiment:** after addressing audit overhead, compare the current executor with a dedicated inference worker and process isolation using identical inputs, while recording event-loop lag, queue depth, memory, and state freshness.

**Expected benefit:** protects API scheduling if inference work blocks shared execution resources. **Complexity:** high. **Accuracy risk:** low to medium, mainly around ordering, stale-result rejection, and restart handling.

### 6. Defer serializer, coordinate-conversion, and socket micro-optimizations

Measured p95 values were about 0.14 ms for coordinate conversion, 1.20 ms for Core updates, 0.15 ms for serialization, 0.50 ms for outbound queue wait, and 1.61 ms for socket writes. These are small beside detector execution and source-to-state latency. Preserve slow-client isolation; investigate transport alternatives only if a repeatable profile changes this ranking.

## Accuracy and remaining evidence gaps

The still-image labels give ball detection F1 0.972, mean center error 3.77 px, and p95 center error 9.11 px across only eight images and 52 balls. They do not cover temporal identity continuity, cue quality, or labeled freshness. No production optimization is recommended until those temporal annotations exist and candidates pass the same accuracy checks.

## 2026-10-21 follow-up: sampled real-time state audit

The matched local M4 Max run used the same 3840×2160 `demo3.mp4`, model hash,
configuration, source revision, and dirty-worktree identity in both arms. Each
arm had a 10-second warm-up and three 30-second repetitions. With audit off,
mean unique state delivery was 24.55 observations/s, p50 receipt latency was
42.17 ms, p95 was 61.83 ms, and 617/2,210 observations exceeded 50 ms. With
10% sampled audit detail, throughput was 25.07 observations/s, p50 was 41.86
ms, p95 was 60.39 ms, and 617/2,256 exceeded 50 ms. All 2,256 instrumented
state messages were complete and unique; 99 source-frame gaps occurred across
the three measured repetitions, with no restarts or out-of-order messages.
The API and video-process exporters recorded zero loss. This is a short
relational screen, not the documented 120-second repetition or 20-minute soak.

In the sampled stage records, total vision service was 22.9 ms mean / 29.9 ms
p95. `model_execute` contributed 50.6% of observed exclusive stage time
(11.7 ms mean, 16.7 ms p95); its parent detector backend contributed another
29.9% (6.9 ms mean, 9.7 ms p95 exclusive). Tracking was 1.8 ms mean / 3.0 ms
p95. Coordinate conversion p95 was 0.17 ms, Core update p95 1.32 ms, JSON
serialization p95 0.18 ms, outbound queue wait p95 0.61 ms, and socket send
p95 2.23 ms. This puts detector work first for micro-level investigation and
rules out serialization and socket write as leading causes in this workload.
Source-publication-to-state p95 was 57.79 ms, so reducing state freshness needs
to address capture age and processing cadence as well as inference. The state
stream itself delivered below the 30 fps capture rate at about 25 observations/s.

The API process used 45.1% CPU at p50 / 55.2% at p95; the video process used
77.8% / 87.7%. Event-loop lag p95 was 2 ms, with p99 75 ms. This local result
places substantial work in the capture process and leaves occasional loop
stalls to investigate. Accelerator metrics were unavailable, so the run does
not prove MPS placement or GPU acceleration. The application still calculated
trajectory predictions even though trajectory instrumentation and subscription
were off, and its per-ball logs remained high-frequency. A true state-only
workload control is needed before measuring trajectory interference.

The model reports only `cue` and generic `ball` classes, so this detector cannot
identify numbered balls such as the 6, 7, or 8. Numbered-ball recognition needs
a separately trained and temporally evaluated model; it must not delay the
location-first work. See the [instrumentation-off report](20261021-151051-91d409b4/report.md), [sampled state-audit report](20261021-151341-a0226100/report.md), and their JSON results.

## 2026-10-21 follow-up: model input-size screening

A separate, audit-only three-run screen used nine annotated 4K stills. A fixed
960×960 export reduced mean CPU inference from 43.76 ms to 24.28 ms and mean
image-to-result time from 97.74 ms to 76.54 ms, while retaining all 52 labeled
ball detections. Ball center error increased slightly from 3.81/9.23 px mean/p95
to 3.99/9.78 px. The 640×640 export missed four labeled balls (ball recall
92.3%, versus 100% for 1280 and 960), so it should not advance under the
location-first accuracy gate.

The target Linux screen processed one set of 30 sequential recorded-video
frames per provider and size. It showed lower p95 inference time at 960 on both
CPU and native OpenVINO, with OpenVINO faster than ONNX Runtime CPU in each
tested arm. This is a bounded relative comparison only: there was no API,
camera, temporal ground truth, or long soak, and the OpenVINO library was loaded
through a temporary test container with an ephemeral finder symlink.

**Updated candidate:** evaluate 960×960 on the next paced, calibrated temporal
clips. It remains an audit candidate, not a production recommendation. The
retained model still emits only `cue` and generic `ball`; it cannot identify
numbered balls. See the [full model-size screening report](model-size-screening-20261021/report.md)
and its raw runs, labels, hashes, and target input frames.

## 2026-10-21 follow-up: camera arrival-rate capacity screening

On the Linux development host, repeated 30-image replay at 3.0 FPS processed
120/120 scheduled images across four passes. It observed 3.000 arrivals/s and
processed 3.003 observations/s, with zero pending replacements and 285 ms p95
source-arrival-to-inference completion. At 3.5 FPS it also processed 120/120
with no replacements, but p95 completion rose to 364 ms. At 4 FPS the repeated
run replaced one pending frame; at 5 FPS it replaced seven of thirty. This
places this workload's short-run capacity boundary between 3.5 and 4 FPS and
makes 3 FPS the conservative next camera-mode trial.

This is replay capacity evidence only: the camera was disconnected, Coral was
not used, and the same 30 still images were repeated with synthetic timestamps.
It does not measure a continuously sampled video, sensor exposure, missed
motion, or temporal localization quality. `vision.camera.fps` requests an
exactly qualified camera mode; it is not a separate software sampling knob.
Treat 3 FPS as a test point, not a production setting, until connected-camera
and temporally labeled video runs show that movement, collisions, settlement,
and identity continuity remain accurate. Full measurements and limits are in
the [camera-rate sweep report](camera-rate-sweep-20261021/sampling-rate-sweep.md).

## Run artifacts

- Instrumentation-off matched baseline: [report](20261019-205837-3188d1ec/report.md), [results](20261019-205837-3188d1ec/summary.json).
- State instrumentation comparison: [report](20261019-210657-c97d5d51/report.md), [results](20261019-210657-c97d5d51/summary.json).
- Corrected, heartbeat-protected 20-minute soak: [report](20261019-214945-8c5f68d0/report.md), [results](20261019-214945-8c5f68d0/summary.json).
- The earlier 20-minute attempt is not comparable: the subscriber lost its stream after roughly seven minutes while the application continued publishing. Its aggregate delivery rate includes that silent tail.


## 2026-10-21 follow-up: latest-frame sampling with unique video content

A 60-second sequence of unique 4K video frames was paced at 3.0, 3.333,
3.75, 4.286, and 5 FPS through the Rust latest-work mailbox on the target
Linux development host. The first two rates had no pending replacements;
3.75 FPS replaced one frame, 4.286 FPS replaced 17/258, and 5 FPS replaced
78/300. The highest no-replacement point was 3.333 FPS, but p95 source-to-
inference-complete latency was 324 ms there. At 5 FPS, completion capacity
was 3.68/s with 530 ms p95 latency. These are single-run, software-paced,
post-decode measurements with the camera disconnected and no temporal labels;
they do not justify setting the physical webcam to 3.333 FPS. See the [full
comparison and raw reports](software-sampling-rate-20261021/sampling-rate-comparison.md).

A code-path review also found that `broadcast_frame` JPEG-encoded, zlib-tested,
base64-encoded, and retained every camera frame even though WebSocket video
delivery is disabled and the viewer has a separate HTTP stream. The callback
now records lightweight frame metadata only, with separate received-versus-
sent counters. A regression test proves the disabled path never calls the
image encoder and retains no image payload. The CPU savings are structurally
clear, but their effect on live detector throughput remains unmeasured without
a connected camera and matched live A/B run.

## 2026-10-21 consolidated priorities: freshness, capture rate, and audit cost

This section supersedes the earlier short 10% audit-overhead screen and the
repeated-still camera-rate recommendation above. The target sampling result
uses unique video content; the instrumentation comparison uses a matched
fixed-content segment and the current 5% detail-sampling default.

### Evidence and limits

- On the target Linux machine (Core i5-8600T, native OpenVINO CPU), a 60-second
  unique-content replay completed 3.317 observations/s from 3.333 arrivals/s
  with no pending replacements. At 3.75 FPS it replaced one frame; at 4.286
  FPS it replaced 17/258; at 5 FPS it replaced 78/300. The 3.333 FPS p95
  source-to-inference completion time was 324.2 ms. Decode, preprocessing, and
  inference averaged about 51 ms, 68 ms, and 190 ms near that rate. This is a
  single software-paced replay after decode, without a connected camera,
  Coral, or temporal ground truth. The 3.333 FPS point is a capacity boundary
  for this input, not a recommended camera rate.
- On the local M4 Max, a corrected matched A/B used the same fixed 60-second
  dense-table segment, real model, revision, dirty-worktree fingerprint,
  model/input hashes, settings, three 120-second repetitions, and 30-second
  warm-up in each arm. Audit off delivered 24.87–25.14 unique observations/s
  at 70.51–70.60 ms p95 publication-to-client receipt. With 5% sampled detail,
  it delivered 25.03–25.12/s at 70.73–70.93 ms p95. The comparison passes the
  2% budget: +0.223% throughput and +0.374% p95, with zero audit-export and
  client trace losses. This qualifies only this Mac workload and sampled
  configuration; it does not qualify the target's audit overhead.
- The 5% instrumented arm retained 449 frame-detail records. `model_execute`
  accounted for 90.4% of sampled exclusive vision-stage time (25.07 ms mean,
  27.16 ms p95). Source-publication-to-state p95 was 63.75 ms; publication to
  local WebSocket receipt p95 was 7.70 ms. Across the full source path p95 was
  about 70.9 ms, with 420/440 sampled source-to-state observations above the
  50 ms reference budget. Sampled state-publication-to-receipt exceeded 50 ms
  only once in 440 records. The main delay in this run was before state
  publication, not the local WebSocket write.
- An earlier matched 10% full-clip A/B exceeded the overhead budget (-2.33%
  throughput, +3.45% p95). The later 5% fixed-segment comparison passes. The
  10% result remains a caution that sampling rate and workload matching matter;
  it is not evidence that 10% is an acceptable default.
- The target run used no physical camera or Coral. The Mac runs had no
  calibrated table and no temporal labels. The model reports `cue` and generic
  `ball`; it cannot identify the 6, 7, or 8 ball. No run establishes physical
  exposure-to-display latency or accuracy during fast motion, occlusion, and
  settlement. Camera publication/read timestamps must not be described as
  sensor exposure timestamps.

### Ranked improvement backlog

1. **Prove calibrated localization correctness on representative temporal
   data.** The current live audits cannot score physical table coordinates.
   First qualify camera/table calibration, then build temporal annotations for
   aiming, rapid shots, collisions, occlusion, and settlement. Score per-ball
   recall, center error in table units, cue quality, identity continuity, and
   observation age. This is the gate for every production optimization.
2. **Reduce the target detector critical path.** The target's roughly 190 ms
   inference and 68 ms preprocessing dominate its current frame budget; the
   M4's sampled profile likewise places 90.4% of vision-stage time in model
   execution. Compare 960 input, quantization, ROI/crops, OpenVINO thread
   settings, and Coral as separate matched experiments. Preserve full-table
   coverage and motion recall. The nine-still input-size screen supports 960
   only as a candidate, not a production change. Avoid optimizing JSON,
   coordinate conversion, or socket writes before evidence moves them up the
   profile.
3. **Qualify adaptive sampling using motion accuracy, not queue balance
   alone.** Latest-wins now bounds pending work and exposes replacements. The
   target can sustain only about 3.3 arrivals/s without replacement in the
   recorded-video screen, yet still takes about 324 ms p95 to finish one
   observation. Compare rates and adaptive cadence against temporal motion
   recall, source age, gaps, and reacquisition. A rate where processed/s is no
   greater than received/s is necessary for a stable queue, but it is not
   sufficient for current state or accurate ball tracking.
4. **Measure scheduling and repeated work.** The 5% trace recorded 149
   duplicate-poll candidates, of which six fell in the selected detail sample.
   This is evidence to measure polling versus observation-triggered admission,
   not evidence that those duplicates dominate CPU. Compare unique observation
   coverage, reuse age, CPU, and state freshness while preserving the latest
   completed usable result and prediction lifecycle.
5. **Isolate secondary workloads after detector improvements.** Run matched
   trajectory-workload-off/on and video/viewer-off/on tests. Prediction is
   secondary to location state, but existing logs show that prediction and
   per-ball logging may still consume work in state-focused operation. Track
   event-loop lag and healthy-client freshness under slow consumers before
   selecting a worker/process redesign.
6. **Keep and verify best-effort latest-state semantics.** The pipeline now
   replaces pending frames and snapshots, expires stale work, and reports
   gaps/drops; frontend overlays clear when source freshness is stale or
   unknown. TCP/WebSocket remains the browser-compatible transport, so queued
   application work can be dropped but bytes already accepted by a kernel or
   remote client cannot be recalled. Test slow-network clients and reconnects;
   compare another transport only with identical snapshots and freshness
   semantics. This is a delivery contract, not proof of a hard camera-to-screen
   deadline.
7. **Treat numbered-ball identity as a separate, later model objective.** The
   current detector cannot distinguish 6/7/8. First stabilize localization and
   temporal tracks; then collect labeled ball-number imagery and evaluate a
   dedicated model/classifier without delaying location publication.

### Reproducible artifacts

- [Target unique-video rate comparison](software-sampling-rate-20261021/sampling-rate-comparison.md)
- [Matched 5% instrumentation-off run](local-sampled-instrumentation-ab/matched-5pct-config-refresh/20261021-212120-6a414523/report.md)
- [Matched 5% instrumentation-on run](local-sampled-instrumentation-ab/matched-5pct-config-refresh/20261021-212922-98f45b17/report.md)
- [Earlier 10% full-clip instrumentation-off run](local-sampled-instrumentation-ab/20261021-205220-1943b30c/report.md)
- [Earlier 10% full-clip instrumentation-on run](local-sampled-instrumentation-ab/20261021-210036-d31a23d5/report.md)
- [Latest-state freshness contract and implementation limits](../latest-state-freshness-contract.md)
