# Local Rust live-path smoke

This is a short end-to-end wiring check, not a performance baseline or accuracy
qualification. It ran the release Rust gateway and live detector on the
workstation, fed a prerecorded clip through the maintained HTTP-MJPEG fixture,
and received public state over the gateway WebSocket.

The input was `test_data/demo2_short.mp4` (SHA-256
`4e09e30c49fed7e134f43f276c629394c209757e0440958a51ab148f12f6e388`), an H.264
1280×720, 30-fps recording served by `scripts/serve-test-video-mjpeg.sh`. The
launch configuration used the local Rust ONNX export and its source checkpoint;
`launch-input-config.json` preserves the supplied configuration. Its raw hash is
`c6577ab552526cf790dc4b316a057c62210d8a317e802fbb6c0719ff5ba48b3f`. The
runtime's normalized effective-configuration hash is recorded separately in
`summary.json`.

The measured window was one 8-second run with no warm-up. The report records
183 complete state messages, 158 newly observed ball updates, 25 cached ball
updates, no incomplete or uncorrelated messages, and 59 skipped source-frame
IDs. State receipt intervals were p50/p95/p99 44.07/46.65/48.08 ms. These are
state arrival intervals, not inference latency. The measured rates are only
roughly 22.9 state messages/s and 19.8 new ball updates/s against the clip's
30-fps input, too short for a throughput recommendation. The run verifies that
latest-frame skipping and state publication work; it does not show whether the
positions are accurate.

The source is network-style MJPEG, so its timestamps do not represent camera
exposure. No reviewed temporal labels or table calibration were supplied, and
the report correctly leaves detector accuracy and cross-process source-to-state
latency unavailable. Browser rendering was not measured. Use this report only
as local integration evidence.
