# Local Rust distribution bundle

The distribution scripts build and run the Rust application server and live
detector with the retained static browser client. Python is not installed or
launched by this bundle. The browser application remains HTML/CSS/JavaScript.

## Build the bundle

The build host needs Rust/Cargo, Node.js/npm, `pkg-config`, `jq`, a SHA-256
utility, and GStreamer 1.x development libraries discoverable through
`pkg-config`. The native build is compatible with the build host by default.
Provide a verified model pair and a live-capable configuration:

```bash
BILLIARDS_ONNX_MODEL=/path/to/billiards-yolov8n.onnx \
BILLIARDS_SOURCE_CHECKPOINT=/path/to/billiards-yolov8n.pt \
make deploy
```

The ONNX model must have a sibling manifest named
`billiards-yolov8n.onnx.manifest.json`. The build validates the manifest schema
and the SHA-256 digests of both the ONNX export and source checkpoint against
that manifest before building. By default, `config.json` is the configuration
source. `BILLIARDS_CONFIG_SOURCE` selects another file, and `BILLIARDS_DIST_DIR`
selects the output directory (default: `dist/`).

Before building, the configuration must clear
`vision.camera.video_file_path` and select either a stable GStreamer camera
selector (`perception.capture.device_selector` beginning with
`gstreamer-device-v1:`) or an HTTP(S) MJPEG source in
`vision.camera.device_id`. Integer camera indices are not supported by this
live capture path yet. The builder copies the model pair and manifest into
`models/`, writes package-relative model and executable paths to `config.json`,
and includes `build-info.txt` with the host/target and source revision.

`BILLIARDS_RUST_TARGET` selects a Cargo target triple and
`BILLIARDS_RUST_FEATURES` selects the Rust feature set. A cross-build still
needs a working target toolchain and compatible target GStreamer and inference
runtime libraries. The package includes ONNX Runtime shared libraries found
beside the release binaries; GStreamer and OpenVINO remain native system
dependencies. A macOS bundle cannot run on Linux, and vice versa.

The bundle contains:

- `bin/`: Rust gateway, live detector, GStreamer capture helper, and any
  colocated ONNX Runtime libraries
- `frontend/`: built static browser client
- `models/`: verified ONNX model, source checkpoint, and manifest
- `config.json`: package-ready live configuration
- `data/`, `logs/`, `run/`, and `backend/calibration_data/`: runtime directories
- `run.sh` and `update.sh`: Rust-only start and restart entry points
- `build-info.txt`: build and source-revision metadata

The builder stages the complete bundle beside the destination and replaces the
previous `dist/` only after all builds and packaging steps succeed.

## Run and update

Copy the complete `dist/` directory to the matching macOS or Linux runtime
machine. Install the matching GStreamer runtime and source plugins on that
machine, then start the server:

```bash
cd /path/to/dist
./run.sh
```

`run.sh` starts live capture by default and requires the Rust server, live
detector, capture helper, nonempty configuration, and built `frontend/index.html`.
For the bundled default config it also checks that the three model files are
present. To start the gateway without live capture, set
`BILLIARDS_START_LIVE=false`; a config file and built browser client are still
required. The default HTTP listener is `0.0.0.0:8000`; the scene-ingress
listener defaults to loopback at `127.0.0.1:49321`.

After replacing the distribution files, run:

```bash
./update.sh
```

When `billiards-trainer.service` is installed, `update.sh` asks systemd to
restart it. Otherwise it stops only the Rust server recorded in this bundle's
PID file, waits for shutdown, and starts `run.sh`; it refuses to signal a PID
that does not match the configured server executable.

## Runtime overrides

`run.sh` accepts these environment variables:

| Variable | Default | Purpose |
|---|---|---|
| `BILLIARDS_CONFIG` | `<bundle>/config.json` | Runtime configuration path |
| `BILLIARDS_WEB_ROOT` | `<bundle>/frontend` | Built static client directory |
| `BILLIARDS_PROJECT_ROOT` | `<bundle>` | Root for relative data/config paths |
| `BILLIARDS_BIN_DIR` | `<bundle>/bin` | Directory containing the Rust binaries |
| `BILLIARDS_SERVER_EXECUTABLE` | `<bin>/billiards-server` | Rust gateway executable |
| `BILLIARDS_LIVE_EXECUTABLE` | `<bin>/billiards-perception-live` | Rust live detector executable |
| `BILLIARDS_CAPTURE_HELPER_EXECUTABLE` | `<bin>/billiards-perception-gstreamer-probe` | GStreamer helper executable |
| `BILLIARDS_HTTP_BIND` | `0.0.0.0` | HTTP bind address |
| `BILLIARDS_HTTP_PORT` | `8000` | HTTP port |
| `BILLIARDS_SCENE_BIND` | `127.0.0.1` | Scene-ingress bind address; Rust requires loopback |
| `BILLIARDS_SCENE_PORT` | `49321` | Scene-ingress port |
| `BILLIARDS_START_LIVE` | `true` | Start the supervised live detector |
| `BILLIARDS_RUN_DIR` | `<bundle>/run` | Runtime PID-file directory |
| `BILLIARDS_PID_FILE` | `<run-dir>/billiards-server.pid` | Server PID-file path |

`update.sh` also accepts `BILLIARDS_STOP_TIMEOUT_SECONDS` (default `10`).

## Qualification status

Packaging and process startup do not by themselves qualify the selected camera,
detector accuracy, calibrated table coordinates, sustained freshness, or
target-machine behavior. Record and verify those separately on the intended
runtime hardware. Numbered-ball classification and trajectory prediction are
not prerequisites for the primary live ball-localization path.
