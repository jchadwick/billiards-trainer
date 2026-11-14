# Configuration

The Rust server is the runtime configuration authority. Its versioned schema
and defaults are in
[`perception/contracts/configuration/v1`](../perception/contracts/configuration/v1/).
Create a current configuration with `make config-example`, then set model paths
and machine-specific capture settings before starting live inference. The
example clears the recorded-video input path because the live capture runtime
accepts camera and network sources only.

A compatible four-corner table calibration supplies the physical aiming map.
Optional optics and ball-center assets improve localization but are not required
for approximate markers or direct-pot advice. Without a usable table mapping,
compatible image-space markers remain available while physical advice pauses.
A source or geometry change requires compatible calibration again.

Select another file for local runs with `RUST_CONFIG`:

```sh
RUST_CONFIG=/path/to/config.json make run
```

Validate a file without changing it:

```sh
RUST_CONFIG=config.json make config-check
```

Production live inference is fixed to the OpenVINO GPU provider. It uses
`perception.rust_live.onnx_path` and
`perception.rust_live.source_checkpoint_path` to name a matching ONNX model and
source checkpoint. The ONNX file's sibling `.manifest.json` must match both
artifacts. Model and checkpoint paths may be absolute or relative to
`--project-root`. The live schema, command-line interface, and packaged runtime
reject CPU and automatic provider selection.

The separate `billiards-perception-video-replay` audit tool accepts CPU,
automatic, and OpenVINO provider selection for offline analysis. Those replay
choices do not change the production live provider.

Coral Edge TPU inference was retired from the supported application path. The
saved target-device comparison found that its compiled graph assigned many
operators back to the CPU, missed one ball in the saved 15-ball scene, and had
much higher inference latency than OpenVINO GPU inference. These measurements
are recorded in [ADR-0104](architecture/decisions/0104-run-target-inference-on-the-verified-intel-gpu.md);
they are a backend selection result, not a claim of physical X/Y accuracy.
Coral model settings are no longer accepted by the live configuration schema.

Set `api.stream.viewer_url` to the browser-reachable MJPEG URL of the camera
service. The live view opens that URL directly. Use a camera service
origin separate from the application API. The camera image and perception
state are independent in this view; the viewer keeps the newest video and
normal center markers do not require frame IDs or exact-frame pairing. The
producer's active `observation_status.perception.table_projection` is the source-pixel to
center-origin table-millimeter mapping used for world positions and projected
aim paths. It must match the current camera/source geometry and calibration
context. Calibration must use that same camera view.

Configuration changes are validated before they are applied. Successful
updates preserve the configuration revision and restart-required semantics.
The configuration API has no authentication and the current scene producer
listener is loopback-only. Keep the API local; remote service authentication
and encrypted role-to-role transport remain planned work. Do not expose the
current API to a LAN or the public Internet.

See the [coordinate contract](COORDINATE_SYSTEMS.md) and
[API reference](API_REFERENCE.md) for related wire behavior. Per-phase
analysis rates and evidence-age limits are under
`perception.rust_live.phase_schedule`. The current cue evidence expiry default
is 250 ms. The legacy layout-scan expiry setting defaults to 1500 ms but does
not expire layout trust; current whole-table motion evidence governs trust.
These settings do not define physical accuracy or latency qualification. Older
consensus, tolerance, and layout-scan-expiry settings remain accepted for
configuration compatibility but do not gate the current trusted/stale layout
policy. See the [phase-aware perception guide](PHASE_AWARE_PERCEPTION.md) for
runtime behavior and supported tuning.

The live coach ranks feasible direct-pot suggestions using its default
policy: cut angle, then travel, then stable target and pocket IDs. The pure
geometry helper accepts ranking options; runtime configuration does not expose
a ranking setting. Geometry is
checked in physical table coordinates for both cue-to-contact and
object-to-pocket paths, accounting for ball radii. This is direct-pot aiming
guidance only; the application does not implement legal-game rules, spin,
banks, combinations, or full post-contact physics. Diagnostics and precision
calibration metrics help inspect the system but are not gates for these
approximate estimates.
