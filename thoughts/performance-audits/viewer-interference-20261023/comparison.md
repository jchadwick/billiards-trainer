# Browser viewer interference comparison — 2026-10-23

## Result

This repeated comparison found no material backend slowdown while the `/live`
browser viewer consumed the direct MJPEG stream on the local Apple M4 Max. The
three-run results fluctuate in both directions: the viewer-on arm delivered
slightly more WebSocket state messages and had slightly lower median run-level
p95 state-receipt, publication-age, and inference times, while its unique
observed-ball update rate was 0.59% lower and its sampled detector CPU median
was 1.67% higher. These small mixed differences do not support a causal
performance recommendation. They also do not rule out small effects.

## Setup and controls

- Viewer-on: [`20261023-150130-b603254a`](viewer-open/20261023-150130-b603254a/report.md).
  The browser `/live` page was open and consumed the MJPEG `<img>` stream
  directly while the detector used the same local Rust MJPEG fan-out fixture.
- Viewer-off: [`20261023-150811-2c5bdd9f`](state-only/20261023-150811-2c5bdd9f/report.md).
  The browser viewer was closed while the detector consumed that same fixture.
- Each arm used a 30-second warm-up followed by three 120-second measured
  windows. The measured windows total 360 seconds per arm. Viewer-on ran first;
  the order was not counterbalanced.
- Both used the same `test_data/demo2_short.mp4` recording, SHA-256
  `4e09e30c49fed7e134f43f276c629394c209757e0440958a51ab148f12f6e388`,
  pre-encoded to JPEG at quality 5 with FFmpeg 8.1.1, then paced at 20 fps.
  They also used Rust runtime revision
  `fa8c5c1469d95c0b5009bd6f333ff42e9f79e357`, effective configuration
  SHA-256 `54f5fc0598cdb3b2a718e89b18bcf43f3981ff5c74e44c6d93b65004b441924c`,
  camera profile `camera-profile-v3:5f7a2f79c24fee59c08632588fd235f98f92a2cfd31417b057c4cd1b243ddae5`,
  CPU runtime, and detector model SHA-256
  `4ceec94901553ff8a22f71d8cca684ec314a9fec8b7f219c410a5df16be4c4fa`.
- The host was macOS aarch64, Apple M4 Max, 14 logical CPUs, 36 GiB memory.
  Receiver and live-pipeline timing reports were complete with zero reported
  malformed, oversized, queue-dropped, or retention-evicted audit records.

## Measurements

Rates divide the sum across three measured runs by 360 seconds. Latency and CPU
values are the median of the three run-level p95 or p50 values, respectively.
This summary keeps each measurement's original definition; observed-ball
updates count states whose ball component was newly observed, while state
messages also include cached or failed source updates.

| Measure | Viewer open | Viewer closed | Difference with viewer |
| --- | ---: | ---: | ---: |
| WebSocket state messages/s | 19.99 | 19.85 | +0.70% |
| Unique source observations/s | 19.81 | 19.66 | +0.76% |
| Unique observed-ball updates/s | 14.08 | 14.16 | −0.59% |
| State receipt interval p95 | 52.38 ms | 52.57 ms | −0.36% |
| Read-completion-to-publication age p95 | 47.20 ms | 48.32 ms | −2.32% |
| Model inference p95 | 45.97 ms | 46.82 ms | −1.82% |
| Detector process CPU p50 | 410.78%/core | 404.03%/core | +1.67% |

Per-run state counts, unique observed-ball updates, and receipt p95s were
viewer-on `2398 / 1686 / 52.38 ms`, `2397 / 1701 / 52.37 ms`, and
`2401 / 1681 / 52.78 ms`; viewer-off `2399 / 1700 / 52.17 ms`,
`2359 / 1689 / 52.57 ms`, and `2388 / 1709 / 55.31 ms`. The viewer-off third
run was slower in publication age and inference, showing why these small
between-arm differences should not be overinterpreted.

## Limits and next experiment

This only tests backend interference from one open browser viewer while
detector inference runs. Resource polling covers the detector child, not total
system CPU, browser CPU, browser memory, or rendered frame rate. It does not
measure physical exposure-to-state delay, a physical webcam, Linux target
behavior, Coral/OpenVINO placement, or localization/classification accuracy.
The live audit's WebSocket input record does not carry a video content hash;
the source recording hash is recorded above. The generated JPEG corpus is
temporary and was shared by the two runs through the same fixture. Accuracy
remains unavailable until temporal source frames are labeled.

For a stronger small-effect comparison, alternate viewer-on and viewer-off
windows, sample system-wide and browser process resources, and record rendered
frames. Run that on the intended target only after the physical camera path is
qualified. Do not change the production video fan-out or inference scheduling
based on this result.
