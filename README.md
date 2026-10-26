# Billiards Trainer

Billiards Trainer detects pool balls from an overhead camera and publishes the
latest table state to the `/live` browser view. Current priorities are correct
ball locations and fresh updates. Numbered-ball recognition and trajectory
prediction are not implemented in the running detector.

## Current application

- The server, capture supervisor, detector, tracking, state conversion, HTTP,
  WebSocket delivery, and static-file serving are Rust.
- The client is a standard HTML/CSS/JavaScript application built from
  `frontend/web`; it is not WebAssembly. `/` redirects to `/live`.
- The `/live` page displays the configured camera stream directly in the
  browser and overlays the latest detected state received over WebSocket.
- Live inference uses the Rust detector with an ONNX model and its matching
  source checkpoint and export manifest. Model class numbers are not treated
  as pool-ball identities.
- Python is only used by the optional ONNX export utility. It is not part of
  the server, deployment image, or normal Rust and browser test workflows.

## Run locally

Install Rust, Node.js, and GStreamer development/runtime libraries for the
selected camera or network stream. Set the model paths and capture settings in
the current Rust configuration:

```sh
# Edit config.json: perception.rust_live.onnx_path and
# perception.rust_live.source_checkpoint_path must identify a matching model pair.
make run
```

The local gateway listens on `http://127.0.0.1:8001` by default. The Rust
server serves the production browser bundle from `backend/static`. Set
`RUST_CONFIG=/path/to/config.json` to select another configuration. `make
run-api` starts the gateway without capture; `make run-frontend` starts Vite
for browser development.

For [Docker deployment](docs/DOCKER.md), the [local distribution bundle](docs/LOCAL_DEPLOYMENT.md), or [model export](tools/README_MODEL_EXPORT.md), see the linked guides.

## Verify changes

```sh
make test
make test-rust-camera
make test-model-export
make test-deploy
make test-tooling
make lint
make format-check
make config-check
```

The runtime contracts and supported routes are documented in the
[architecture](docs/ARCHITECTURE.md), [API reference](docs/API_REFERENCE.md),
[configuration guide](docs/CONFIG.md), [coordinate contract](docs/COORDINATE_SYSTEMS.md),
and [performance audit guide](docs/PERFORMANCE_AUDIT.md).
