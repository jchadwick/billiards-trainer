# Target physical-camera qualification

The target `hp-prodesk` now enumerates the attached UC40M webcam through
GStreamer. Camera discovery resolved its transient `/dev/video0` path through
the operating system's stable by-id link and produced selector
`gstreamer-device-v1:133d9f4857ec09809e1ce59152e48b282a4c53696dd04485fce7e629b7c48db1`.
The inventory is retained in this directory.

The camera advertises MJPEG at 1920×1080 up to 30 FPS. At the requested
1920×1080 MJPEG / 10 FPS profile, V4L2 reported 10 FPS and a direct GStreamer
read produced frames, including a 50-frame stream. No camera controls were
written. A sampled frame after several seconds remained nearly black, with no
table or balls visible.

The first bounded Rust live qualification ran while the camera view was dark
and obscured. It failed before measurement and recorded zero state events; its
failure record is preserved under `20261022-032505-82766a38/`. A later
illuminated sample clearly showed the green table and balls, and the same
3-second warm-up / 15-second measured configuration was rerun with the real
two-class cue/ball model. The run still failed before readiness, but schema-v3
timings now explain the rejection; see
[`20261022-040555-efa22995`](20261022-040555-efa22995/report.md). Neither failed
startup is a throughput, accuracy, or full measurement-window result.

The available model distinguishes only `cue` and `ball`; it cannot identify
numbered balls such as the 6, 7, or 8. The target host also did not expose an
OpenVINO shared library to the live process, so no provider comparison was run.

The next experiment should first make CPU inference fit the freshness budget
or qualify a faster runtime on the same input. Lowering capture FPS can reduce
discarded work, but it cannot by itself make a 267 ms median inference meet the
current 100 ms limit. Compare model input-size and available-runtime options
with identical camera profile and settings, then score accuracy against
labeled frames before recommending a production profile. The installed
OpenVINO runtime was not available to the container used for this CPU run.

Follow-up lower-rate source tests completed after the 10 FPS diagnostic. The
[3 FPS attempt](20261022-041120-212da569/failure.json) timed out before its
first physical sample; no detector comparison is possible. The
[5 FPS attempt](20261022-041134-4a04a28e/failure.json) completed 62 inferences,
all stale after inference, and rejected 24 additional frames before inference.
Its completion-age p95 was 372.5 ms, versus 363.4 ms in the single 10 FPS
startup diagnostic. Both published zero states. These short unrepeated runs
do not establish small performance differences; lowering FPS did not restore
fresh state delivery. See the [current-state review](../../current-state-review-2026-10-22.md)
for the recommended next sequence.

## API-hosted Rust live startup check (2026-10-22)

The retained Python service was launched with `perception.provider=rust` and
`perception.mode=live` using the staged target model and the target UC40M stable
selector. The first container-side GStreamer inventory returned zero cameras.
Adding read-only `/run/udev` and `/sys` mounts made container discovery see one
camera with the same stable by-id selector as the host. The API then passed
selector resolution but the Rust capture helper timed out waiting for its first
sample. The API exited during startup before inference or state publication.
This was a bounded startup diagnosis, not an inference or performance run; no
camera frames were exported. The stopped container was removed.

The Rust live provider has no working camera sample in this container
configuration yet. Investigate GStreamer source/pipeline negotiation and
container device/runtime packaging before client accuracy testing. Preserve
fail-closed startup; this attempt does not qualify physical live operation.
