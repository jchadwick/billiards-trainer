# Local Rust distribution bundle

The distribution scripts build and run the Rust application server and live
detector with the retained static browser client. The bundle does not install
Python; inference uses the Rust ONNX backends. Coral Edge TPU inference has
been retired from the supported application path. The browser application
remains HTML/CSS/JavaScript.

## Build the bundle

The build host needs Rust/Cargo, Node.js/npm, `pkg-config`, `jq`, a SHA-256
utility, and GStreamer 1.x development libraries discoverable through
`pkg-config`. The native build is compatible with the build host by default.
Build with the standard model assets:

```bash
make deploy
```

The builder looks for `models/exports/billiards-yolov8n.onnx`, its sibling
`.onnx.manifest.json`, and `models/billiards-yolov8n.pt`. If your checkout does
not include the artifacts, obtain a verified export and its matching checkpoint,
then pass their paths with `BILLIARDS_ONNX_MODEL` and
`BILLIARDS_SOURCE_CHECKPOINT`. The build checks the manifest schema and SHA-256
digests before compiling. Production live inference uses the Intel OpenVINO GPU.

The build host needs no camera or runtime configuration. For a preconfigured
deployment, set `BILLIARDS_CONFIG_SOURCE` to an existing config file. The builder
packages model and executable paths relative to the output directory and checks
any explicit capture source without opening it. Set `BILLIARDS_DIST_DIR` to
choose an output directory other than `dist/`.

`BILLIARDS_RUST_TARGET` selects a Cargo target triple and
`BILLIARDS_RUST_FEATURES` selects the Rust feature set. A cross-build still
needs a working target toolchain and compatible target GStreamer and inference
runtime libraries. The package includes ONNX Runtime shared libraries found
beside the release binaries; GStreamer and OpenVINO remain native system
dependencies. A macOS bundle cannot run on Linux, and vice versa.

The bundle contains:

- `bin/`: Rust gateway, live detector, GStreamer capture helper, camera
  discovery command, and any colocated ONNX Runtime libraries
- `frontend/`: built static browser client
- `models/`: verified ONNX model, source checkpoint, and export manifest
- `config.json`: runtime overrides, created by setup or included when an
  explicit source config is supplied
- `data/`, `logs/`, `run/`, and `backend/calibration_data/`: runtime directories
- `run.sh` and `update.sh`: Rust-only start and restart entry points
- `build-info.txt`: build and source-revision metadata

The builder stages the complete bundle beside the destination and replaces the
previous distribution only after every build and packaging step succeeds. It
builds the browser client directly inside that temporary staging directory, so
a failed or interrupted package build cannot partially overwrite `backend/static`
that another local server may currently be serving. It
marks bundles it owns, refuses to replace an unmarked directory or symlink,
and retains the previous marked bundle in a unique sibling backup directory
after successful replacement. This preserves the previous configuration and
runtime data for recovery; inspect and remove those backups manually after
confirming they are no longer needed. `make deploy-clean` removes only a
distribution carrying the recognized ownership marker.
Run `make test-deploy` to exercise package creation, replacement, rollback,
frontend-failure isolation, cleanup, and updater shutdown behavior with isolated
temporary bundles and processes; these tests do not need a camera or compiled
Rust binaries.

## Run and update

Copy the complete `dist/` directory to the matching macOS or Linux runtime
machine. Install the matching GStreamer runtime and source plugins on that
machine. Install the matching OpenVINO Runtime and GPU plugin on the target, configure
the Intel GPU device access, then start the server:

```bash
cd /path/to/dist
./run.sh
```

Open `http://127.0.0.1:8000` to choose a camera and calibrate the table. On later
starts the gateway opens the saved source and reuses compatible calibration.
Keep `backend/calibration_data/browser-precision/` when upgrading or replacing
the runtime directory. It contains the fitted optics and ball-center files
and the saved wizard photos and measurements. The old wizard's photos
cannot be recovered from a fitted file.
To run without live setup and capture, set `BILLIARDS_START_LIVE=false`.
The scene-ingress listener defaults to loopback at `127.0.0.1:49321`.

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
| `BILLIARDS_HTTP_BIND` | `127.0.0.1` | HTTP bind address |
| `BILLIARDS_HTTP_PORT` | `8000` | HTTP port |
| `BILLIARDS_SCENE_BIND` | `127.0.0.1` | Scene-ingress bind address; Rust requires loopback |
| `BILLIARDS_SCENE_PORT` | `49321` | Scene-ingress port |
| `BILLIARDS_START_LIVE` | `true` | Enable live setup and supervised capture |
| `BILLIARDS_RUN_DIR` | `<bundle>/run` | Runtime PID-file directory |
| `BILLIARDS_PID_FILE` | `<run-dir>/billiards-server.pid` | Server PID-file path |

`update.sh` also accepts `BILLIARDS_STOP_TIMEOUT_SECONDS` (default `10`) and
`BILLIARDS_FORCE_STOP_TIMEOUT_SECONDS` (default `2`). It starts the replacement
only after the old process has exited, including after forced termination.

## Qualification status

Packaging and process startup do not by themselves qualify the selected camera,
detector accuracy, calibrated table coordinates, sustained freshness, or
target-machine behavior. Record and verify those separately on the intended
runtime hardware. Numbered-ball classification and trajectory prediction are
not prerequisites for the primary live ball-localization path.
