# Configuration

The Rust server is the runtime configuration authority. Its versioned schema
and defaults are in
[`perception/contracts/configuration/v1`](../perception/contracts/configuration/v1/).
Create a current configuration with `make config-example`, then set model paths
and machine-specific capture settings before starting live inference. The
example clears the recorded-video input path because the live capture runtime
accepts camera and network sources only.

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

`vision.camera` selects a local camera or a supported network stream. Configure
`api.stream.viewer_url` separately when the browser should display a video
stream. For exact spatial overlays, point it to
`http://<device>:<http-port>/api/v1/video/identified.mjpg`. The Rust gateway
bridges the capture preview bound to `perception.rust_live.frame_preview_port`
(default `49322`), which must be a nonzero loopback port for this route. Keep the
capture source and viewer source aligned. The browser never pairs by timestamp
or nearest frame. Legacy MJPEG sources remain available as video, while spatial
overlays are withheld until the displayed frame has matching capture identity.

### Identified video and exact-frame overlays

`api.stream.frame_pairing` bounds browser-side parsing and retention for
identified MJPEG streams:

| Setting | Default | Purpose |
| --- | ---: | --- |
| `maximum_frame_bytes` | 16 MiB | Maximum JPEG payload accepted for one part |
| `maximum_part_header_bytes` | 16 KiB | Maximum multipart header block |
| `buffered_frame_count` | 64 | Maximum number of decoded frames retained for pairing |

The stream must use `multipart/x-mixed-replace`; every part must provide a
`Content-Length` and `Content-Type: image/jpeg`, plus `X-Frame-Id`,
`X-Engine-Instance-Id`, `X-Source-Session-Id`, `X-Configuration-Revision`,
`X-Calibration-Revision`, and `X-Rectification-Revision`. Cross-origin viewer
URLs must allow browser `fetch` with the stream's CORS policy. Invalid or
oversized parts are rejected and do not receive a spatial overlay. These
limits bound memory and parsing work; they do not repair incorrect identity
metadata. The gateway obtains its JPEG bytes from the detector's exact captured
RGB frame, and emits all identity headers from that frame descriptor.

Configuration changes are validated before they are applied. Successful
updates preserve the configuration revision and restart-required semantics.
The configuration API has no authentication and the current scene producer
listener is loopback-only. Keep the API local; remote service authentication
and encrypted role-to-role transport remain planned work. Do not expose the
current API to a LAN or the public Internet.

See the [coordinate contract](COORDINATE_SYSTEMS.md) and
[API reference](API_REFERENCE.md) for related wire behavior. The initial
per-phase analysis rates and evidence-age limits are under
`perception.rust_live.phase_schedule`; see the
[phase-aware perception guide](PHASE_AWARE_PERCEPTION.md) before tuning them.
