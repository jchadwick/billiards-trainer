# Configuration

The Rust server is the runtime configuration authority. Its versioned schema
and defaults are in
[`perception/contracts/configuration/v1`](../perception/contracts/configuration/v1/).
New installs start with built-in defaults. Open the app, select a discovered
camera or enter a supported HTTP(S) MJPEG URL, and calibrate the table. The
gateway starts before a camera is selected and keeps serving setup if capture
cannot start. The default local configuration path is `data/config.json`.
The first source selection creates a sparse file. Existing files keep their
saved camera and overrides.

The calibration screen freezes an identified camera frame. Mark the four
playing-area corners and confirm the playing-surface dimensions using a preset
or measurements. The runtime solves and persists the transform and binds it to
the capture geometry. A different source or incompatible geometry requires
calibration again. Optional optics and ball-center calibration assets refine
physical ball localization and are checked separately in the Setup calibration
checklist. They are not prerequisites for approximate center markers or live
direct-pot advice: compatible detector centers and the active table projection
are enough for those features. Without a usable table mapping, compatible
image-space markers can remain visible, while advice that needs physical
distances pauses. Use **Configure
calibration files** to save paths on the server. A path change requires a
restart and may require recalibrating the table for the new configuration
revision. Fit ball centers against the active table calibration revision.

For browser fitting, choose **Calibrate camera optics**. Download the four
US Letter tiles from the wizard, print at 100% scale, and assemble them flat.
Verify that a square measures 2 inches (50.8 mm). The assembled board has
9 by 6 inner corners. Follow the eight position and tilt prompts, replacing
any individual capture as needed. The gateway host needs
Python 3 with NumPy and OpenCV. Set `BILLIARDS_CALIBRATION_PYTHON` to select an
interpreter when the dependencies are installed in a virtual environment.
Then choose **Calibrate ball centers** and collect at least twelve distinct,
independently referenced ball placements. Choose any of the displayed targets,
with extra targets near the left and right edges. The wizard overlays the
selected target on the live camera preview and gives two distances from the
cushion noses in millimeters and inches.
Measure to a flat mark on the cloth and center the ball over that mark.
The green check confirms the camera found a ball in the target area. After
capture, confirm the suggested ball outline in the zoomed view or mark it manually.
Both wizards reload the compatible saved photos when reopened, and the page
reports held-out errors and saves successful fits automatically. Browser
assets are stored under `backend/calibration_data/browser-precision` in the
project root, take precedence for the matching camera, and are revalidated
against active table geometry. Refit ball centers after changing the optics or
table calibration.

Open Advanced for configuration overrides. Select a different file or validate
one locally with:

```sh
RUST_CONFIG=/path/to/config.json make run
RUST_CONFIG=/path/to/config.json make config-check
```

Production live inference uses OpenVINO on the supported Intel GPU. Packages
include the verified ONNX export, its sibling manifest, and the matching
checkpoint. The source checkout resolves the default export under
`models/exports/` when the packaged layout is absent. Advanced users can
override `perception.rust_live.onnx_path` and
`perception.rust_live.source_checkpoint_path` using paths relative to
`--project-root` or absolute paths.

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
service. Live Diagnostics opens that URL directly. Use a camera service
origin separate from the application API. The camera image and perception
state are independent in this view; the viewer keeps the newest video and
normal center markers do not require frame IDs or exact-frame pairing. The
producer's active `observation_status.perception.table_projection` is the source-pixel to
center-origin table-millimeter mapping used for world positions and projected
aim paths. It must match the current camera/source geometry and calibration
context. Setup can request a single still for table calibration.

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
