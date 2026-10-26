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

CPU, auto, and OpenVINO live inference require `perception.rust_live.onnx_path`
and `perception.rust_live.source_checkpoint_path` to name a matching ONNX model
and source checkpoint. The ONNX file's sibling `.manifest.json` must match both
artifacts. Model and checkpoint paths may be absolute or relative to
`--project-root`.

To use a Coral Edge TPU with the Rust live inference runner, set
`perception.rust_live.provider` to `coral` and configure
`perception.rust_live.coral_model_path` to the compiled Edge TPU `.tflite`
model. The model's sibling directory must contain its `manifest.json`. The
Coral runner also uses `coral_python_executable` (default `python3`) and
`coral_delegate_library` (default `libedgetpu.so.1`); use these settings when
the target device installs either dependency at a nonstandard path.
The Coral manifest must declare
`model_contract.output.box_coordinate_space`. The current compiled artifact
uses `normalized_0_1`; the runtime scales its center and size values into model
pixels before letterbox unpadding and NMS. Use `model_pixels` only when an
artifact already reports box coordinates in model pixels.

Keep `perception.rust_live.source_checkpoint_path` pointed at the original
`.pt` checkpoint on the target as well. Startup verifies its digest against
the Coral manifest before loading the model, so deploy the checkpoint together
with the compiled model, manifest, and configuration.

The target Linux device must have a TensorFlow Lite API package
(`tflite_runtime`, `ai-edge-litert`, or TensorFlow), the Coral `libedgetpu`
runtime, and operating-system access to the Edge TPU device. The worker prefers
`tflite_runtime`, then LiteRT, and uses TensorFlow as a fallback. For example,
install LiteRT into the configured Python environment with
`python3 -m pip install ai-edge-litert`. If the model,
delegate, runtime, or device cannot be initialized, startup fails instead of
switching to the ONNX CPU backend. The compiled artifact may still run
compiler-unmapped operators on the CPU; check its compiler report for the split.
Hardware is not available on the development host, so validate model loading
and inference on the target device before relying on Coral deployment.
`perception.rust_live.coral_runtime` configures Coral's longer inference
windows independently of CPU settings. Its defaults allow frames up to 3000 ms
old, observations up to 9 seconds old, cached localization to expire after
18000 ms, motion evidence to remain valid for 3000 ms, and layout evidence to
remain valid for 10000 ms. Coral uses a 10-pixel layout-consensus tolerance and
an NMS IoU threshold of 0.2 to account for the compiled model's measured
6.3-pixel coordinate quantization and duplicate detections on a stationary
ball. `shot_ball_hz` keeps periodic ball observations running during detected
motion; those observations remain untrusted for guidance until quiet dwell and
layout consensus succeed. These Coral-specific settings preserve CPU and other
providers' generic capture and phase-schedule settings.
After a Coral inference error, the worker releases and recreates its TFLite
interpreter after `worker_restart_backoff_ms`, then retries the failed frame
up to `maximum_worker_restarts_per_frame` times. This lets the live process
recover from a transient USB transfer reset without repeatedly using a failed
delegate instance.

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
