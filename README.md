# Billiards Trainer

Billiards Trainer detects table state from an overhead camera and publishes it
to browser clients. The first priority is accurate, fresh ball locations. Cue
quality and trajectory prediction are additional capabilities; predictions are
not on the critical path for basic localization. Numbered-ball classification
(for example, distinguishing the 6, 7, and 8 balls) is not part of the current
Rust migration.

## Runtime architecture

- **Server:** Rust owns the supported application server, live capture and
  localization process, HTTP/WebSocket state delivery, and static asset
  serving. The gateway supervises the Rust live detector when started with
  `--start-live`.
- **Browser:** The web client remains a standard HTML/CSS/JavaScript application
  built from the React/TypeScript source in `frontend/web`. It is delivered as
  static browser assets; it is not WebAssembly.
- **Models:** The Rust live detector consumes a verified ONNX export and its
  source checkpoint. Number classification is intentionally excluded.
- **Python:** Python remains for offline compatibility tests and the optional
  ONNX model-export utility. The old dataset-preparation and standalone camera
  tools were retired. Install the model-export stack only with
  `make install-model-tools`. Python is not the supported application-server
  runtime.

The Rust live path and deployment packaging are still being qualified. A
successful build or replay is not proof of camera accuracy, calibrated table
coordinates, sustained freshness, or target-device behavior. The checked-in
runtime configuration must provide a qualified model pair and appropriate
capture/calibration settings before live operation can start. macOS and Linux
qualification are tracked separately.

## Run locally

Install the Rust toolchain, Node.js, and the GStreamer development/runtime
components needed by the selected camera or stream source. Run `make dev` to
fetch Rust dependencies and install the locked browser dependencies. Prepare
`config.json` with the Rust live model paths and capture settings, then run:

```bash
make run
```

The local Rust gateway defaults to `http://127.0.0.1:8001`; the frontend is
built and served by that gateway. The development-only Vite client can also be
run separately with `make run-frontend`. To run the gateway without starting
live capture, use `make run-api`.

The Rust configuration requires `perception.rust_live.onnx_path` and
`perception.rust_live.source_checkpoint_path`. The ONNX export must have its
matching `.manifest.json` alongside it. Paths may be absolute or relative to
the configured project root. Missing or unverified model inputs are reported
as startup errors rather than falling back to Python.

## Repository map

```text
perception/       Rust server, capture, inference, tracking, and contracts
frontend/web/     Browser client source and tests (HTML/CSS/JavaScript output)
backend/          Retained Python tests, model/off-device tools, and migration
                  reference code; not the supported server runtime
models/           Model artifacts and manifests, where present
docs/             Runtime, deployment, and architecture decision records
thoughts/          Performance audit reports and investigation notes
```

See [Rust perception/runtime notes](perception/README.md),
[architecture decisions](docs/architecture/README.md), and the
[Docker deployment guide](docs/DOCKER.md) for implementation and qualification
details. Historical Python module documentation is in
[`backend/README.md`](backend/README.md).
