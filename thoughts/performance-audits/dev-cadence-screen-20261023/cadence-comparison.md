# Camera cadence screening comparison

This is a short, single-run-per-setting comparison to choose the next test. It is
not a production baseline, camera qualification, accuracy evaluation, or reason
to change the shipped default by itself.

## Result

**20 fps is the strongest next setting to qualify on the target webcam.** On this
workstation it retained almost all newly observed ball-update throughput from
the 30 fps cap, with lower read-completion-to-publication delay, lower detector
process CPU, and no repeated source observations or observed frame gaps. It
still showed some tail wait before inference, so this is a candidate, not a
claim that every frame is processed immediately.

At 15 fps, measured wait before inference was effectively zero and the
read-completion-to-publication p95 was lower still, but newly observed ball
updates fell by about one third versus 30 fps and 2 state messages were marked
stale. Ten fps had nearly the same read-to-publication p95 as 15 fps while
producing substantially fewer observed updates and 13 stale state messages, so
it is dominated for this workload unless lower CPU use matters more than motion
freshness.

| Requested capture cap | State messages/s | Newly observed ball-update frames/s | Cached/failed state updates/s | Stale state messages | Read completion → publication p95 (ms) | Read completion → inference start p95 (ms) | Detector CPU (median cores) | Repeated source messages | Missing source frame IDs |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 30 fps | 22.33 | 17.60 | 4.73 | 0 | 78.60 | 35.28 | 4.59 | 0 | 116 |
| 20 fps | 20.06 | 17.26 | 2.80 | 0 | 62.13 | 16.76 | 4.09 | 0 | 0 |
| 15 fps | 15.06 | 11.80 | 3.13 | 2 | 46.37 | 0.014 | 3.06 | 2 | 2 |
| 10 fps | 10.07 | 6.67 | 2.53 | 13 | 46.72 | 0.012 | 2.11 | 13 | 13 |

The CPU column is the median process CPU percentage divided by 100, so 4.59
means about 4.59 logical cores used by the sampled detector process. The capture
helper and whole-machine CPU were not sampled. State-message rate includes
cached publications; the observed-ball column counts source updates with a new
ball observation and is the more relevant detection-throughput comparison.

The p95 inference-start wait is the measured interval from parent completion of
a framed capture record to detector inference start. It is not camera exposure
latency. The p95 read-to-publication interval uses the host read-completion
clock and excludes unknown time between sensor exposure and host read. The local
state receive interval grows as the configured cadence falls, as expected; it
is a delivery interval, not processing latency.

## Method and provenance

The same release Rust gateway, Rust live detector, CPU ONNX Runtime provider,
model, confidence/IoU thresholds, resolution, state subscriber, timing audit,
and local video source were used in all four runs. Only `vision.camera.fps`
changed. Each setting had a 5-second warm-up and one 15-second measured window.
The app used GStreamer's HTTP-MJPEG path with `videorate drop-only` and a
one-buffer dropping appsink. The source was the looping local fixture serving
`test_data/demo2_short.mp4`, an H.264 1280×720 recording with a 30 fps nominal
rate and SHA-256
`4e09e30c49fed7e134f43f276c629394c209757e0440958a51ab148f12f6e388`.

The developer machine was an Apple M4 Max running macOS arm64. The measurement
used the managed Rust live server and its audit receiver. The detector process
was sampled once per second. Detailed per-setting JSON, JSONL timing traces,
model/runtime hashes, and environment provenance are preserved under
`fps-10/`, `fps-15/`, `fps-20/`, and `fps-30/`; the four launch-configuration
snapshots are in `configs/`. Run identities are:

- 30 fps: `20261023-074531-2bc458d4`
- 20 fps: `20261023-074615-b442cd41`
- 15 fps: `20261023-074637-53330348`
- 10 fps: `20261023-074700-2d4fe289`

All four runs completed with complete state messages, no invalid JSON, no
inference failures, and no audit-record loss. The 30- and 20-fps runs had no
stale state messages; the 15- and 10-fps runs had 2 and 13 respectively. Each
setting had one short window, so environmental and clip variation are not
averaged out.

## Limits and next steps

- The MJPEG fixture re-encodes a saved video. It is not the physical webcam or
the target machine, and its source timestamps are not camera exposure times.
- On this HTTP-MJPEG path, cadence limiting occurs after JPEG decode. The
  experiment measures less downstream frame work; it does not prove reduced
  network traffic or JPEG-decoder cost. A physical camera may honor a requested
  lower sensor mode directly, but the target webcam's supported modes remain to
  be checked.
- No reviewed temporal labels were available. Ball-position accuracy, identity
  continuity, and the risk of missing fast motion at 20/15/10 fps are unknown.
  The observed update rate is not an accuracy score.
- 20 fps is the recommended first target-machine check, with 15 fps as a
  lower-latency fallback if webcam mode negotiation and moving-ball accuracy
  support it. Keep the current latest-frame/drop-old-work behavior. Do not
  recommend 10 fps for motion tracking based on these results.
- The next evidence needed is a brief target-webcam mode/functional check, then
  labeled moving, collision, and settling segments at 20 versus 15 fps. For the
  current HTTP-MJPEG development source, evaluate dropping stale multipart JPEG
  payloads before JPEG decode if capture-helper CPU becomes material.
