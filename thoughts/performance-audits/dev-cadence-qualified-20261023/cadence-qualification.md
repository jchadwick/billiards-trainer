# Current Rust live capture cadence qualification

This is a relational comparison on the development Mac. It does not qualify a
production camera setting or ball-localization accuracy.

## Result

**20 fps is the strongest next setting to test with the calibrated target
camera. Keep 30 fps as a comparison arm; do not change the shipped setting yet.**
At 20 fps the live service delivered a median 14.29 unique observed
ball-bearing source frames/s, with 63.07 ms p95 read-completion-to-publication
age. At 30 fps it delivered 16.62 such frames/s, but p95 age rose to 77.65 ms,
detector CPU rose from 4.07 to 4.60 cores, and the receiver saw a median 941
missing source frame IDs per 120-second window versus 45 at 20 fps. The extra
2.33 observed frames/s at 30 fps costs about 15 ms of p95 freshness and 13%
more detector CPU.

| Capture cap | State messages/s | Unique observed ball-bearing frames/s | Stale state messages/window | Missing source frame IDs/window | Read completion → publication p50 / p95 / p99 (ms) | Inference p95 (ms) | Read completion → inference start p95 (ms) | Detector CPU median (cores) | Producer shutdown |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| 15 fps | 14.95 | 10.38 | 43 | 49 | 45.04 / 56.68 / 63.82 | 45.38 | 12.17 | 3.07 | unclean; unqualified |
| 20 fps | 19.81 | 14.29 | 21 | 45 | 47.62 / 63.07 / 70.18 | 45.19 | 17.92 | 4.07 | clean |
| 30 fps | 22.26 | 16.62 | 24 | 941 | 60.36 / 77.65 / 82.46 | 45.00 | 33.06 | 4.60 | clean |

Each row is the median of three 120-second measured windows after a 30-second
warm-up. The 20/30 order was counterbalanced as 20→30, 30→20, 20→30. All arms
used the same saved 1280×720 recording looped as local HTTP MJPEG, Rust live
server and detector, CPU ONNX Runtime, model and confidence settings. The
three JSON config files produce the same SHA-256 after removing
`vision.camera.fps`, confirming that cadence was the only configuration
difference. The recording SHA-256 is
`4e09e30c49fed7e134f43f276c629394c209757e0440958a51ab148f12f6e388` and the
detector ONNX SHA-256 is
`4ceec94901553ff8a22f71d8cca684ec314a9fec8b7f219c410a5df16be4c4fa`.

All nine measured windows had complete state messages, available detector
resource samples, complete timing-audit status, and zero audit queue/export
loss. No detector restarted during a measured window. The 15 fps measurements
did not shut down cleanly: each producer footer reported `clean_shutdown=false`
and the capture helper reported SIGTERM as a read failure after measurement.
A separate 20-second 15 fps check with the MJPEG source already listening
reproduced this behavior. Its cadence numbers are useful for exploration, but
the setting is unqualified until capture shutdown is understood.

## Bottleneck interpretation

The live timing records place inference at roughly 45 ms p95 in all three
cadence arms. At 20 fps, the p95 wait from parent read completion to inference
start was 17.92 ms; it increased to 33.06 ms at 30 fps. Lowering the source cap
to 20 fps therefore reduces queue wait and source-frame gaps while retaining
most of the observed-ball update rate. Fifteen fps reduces CPU further, but
loses about 27% of observed ball-bearing frames/s versus 20 fps, reports more
stale states, and has the shutdown defect.

Inference dominates the measured detector work. In one 20 fps trace, inference
p95 was 45.19 ms while preprocessing was 1.03 ms, decode/NMS 0.10 ms,
tracking/state assembly 0.08 ms, JSON serialization 0.05 ms, and scene queue
submission 0.008 ms. This makes model/runtime execution the highest-value
performance experiment. Serializer, coordinate, and queue micro-optimizations
are unlikely to change the current latency materially without new profiling
evidence.

## What this does not establish

The capture timestamps mean host read completion, not sensor exposure. No
temporal labels or independent physical ball centers were supplied, so these
runs cannot tell us whether 15 or 20 fps misses fast motion, collisions, or
occlusions, or quantify position and identity accuracy. The 30 fps source is
a looped saved recording on an M4 Max; it is neither the physical webcam nor
the target Linux machine. Trajectory workload and browser/video viewers were
off, so their interference remains unknown. The model detects generic balls
and cue only; it does not identify numbered balls.

## Next measurements

1. Capture a representative target-camera sequence and independently annotate
   visible, moving, colliding, settled, and occluded balls with physical table
   positions. Compare 15, 20, and 30 fps on ball recall, position error,
   identity continuity, cue quality, stale-state rate, and current-state age.
2. Investigate the 15 fps capture-helper shutdown failure before considering
   that setting for deployment.
3. Compare CPU ONNX Runtime with OpenVINO and test model input size or
   quantization only against the same reviewed accuracy set. Inference is the
   measured bottleneck; no model/runtime optimization is ready for production
   without its accuracy result.
4. Run trajectory-on and browser/video-viewer-on interference comparisons as
   separate arms. This cadence run used a headless state subscriber only.

The versioned aggregate is [cadence-qualification.json](cadence-qualification.json).
Individual run reports and bounded traces remain under each run ID:

- 15 fps: [report](15fps/20261023-085647-2548e800/report.md); producer shutdown is unclean.
- 20 fps: [run A](20-a/20261023-083918-3e4cdf51/report.md), [run B](20-b/20261023-084737-305faf88/report.md), [run C](20-c/20261023-085015-5bf002a4/report.md).
- 30 fps: [run A](30-a/20261023-084207-4f44a18b/report.md), [run B](30-b/20261023-084501-7dff273d/report.md), [run C](30-c/20261023-085251-aaaad580/report.md).
- 15 fps clean-source startup check: [report](15fps-startup-check/20261023-090442-4c521056/report.md); short window, also unclean at shutdown.
