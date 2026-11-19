# Billiards Trainer

Billiards Trainer is a pool practice and live aiming coach. Use `/practice` to
edit a simulated table without a camera, or `/live` to work from the latest
camera video and producer-reported ball positions. The live view offers
estimated direct-pot suggestions when current table geometry supports them;
ball-number recognition and complete shot physics are not provided.

## Current application

- The server, capture supervisor, detector, tracking, state conversion, HTTP,
  WebSocket delivery, and static-file serving are Rust.
- The client is a standard HTML/CSS/JavaScript application built from
  `frontend/web`. The home page opens setup until the camera and table are ready;
  `/practice` is an editable sandbox whose simulated balls never stand in for
  live detections.
- The `/live` page displays the newest configured camera video while receiving
  producer state independently over WebSocket. Compatible ball centers remain
  visible without exact video-frame pairing.
- In `/practice`, place the cue ball and object balls on the table, confirm the
  cue ball, and optionally choose a target and pocket. The coach checks both
  path segments for blockers and shows the suggested contact point, ghost-ball
  center, desired cue path, and estimated initial object-ball path. It models
  direct pots only; it does not model legal-game rules, spin, banks,
  combinations, or full post-contact physics.
- In `/live`, select the cue ball and optionally a target and pocket using the
  marker labels. These labels identify producer tracks for manual selection;
  they are not recognized ball numbers. Automatic direct-pot suggestions are
  available when the layout is trusted and compatible physical table mapping
  is present. Solid markers represent trusted positions; dashed, dim markers
  are stale last-known positions and do not establish clear paths. Motion or
  lost evidence pauses advice until a clean settled scan replaces the layout.
  The newest video keeps playing during this process. The current detected cue
  line is ephemeral and shown separately from the desired aim suggestion; the
  desired suggestion can be used when no cue line is available.
- Live inference uses the Rust detector with an ONNX model and its matching
  source checkpoint and export manifest. Model class numbers are not treated
  as pool-ball identities.
- Python supports the optional ONNX export utility and browser checkerboard
  fitting. Checkerboard fitting uses OpenCV on the gateway host.

## Run locally

Install Rust, Node.js, and the GStreamer libraries and plugins for your camera.
The live runtime also requires the supported OpenVINO Intel GPU environment and
verified model assets. Start the application:

```sh
make run
```

Open the app, choose a camera or enter its HTTP(S) MJPEG URL, then follow the
table calibration steps. The app saves your choices and uses its own camera
preview automatically.

The local gateway listens on `http://127.0.0.1:8001` by default. The Rust
server serves the production browser bundle from `backend/static`. Set
`RUST_CONFIG=/path/to/config.json` to select another configuration. `make
run-api` starts the gateway without capture; `make run-frontend` starts Vite
for browser development and proxies API calls to the local gateway.

For [Docker deployment](docs/DOCKER.md), the [local distribution bundle](docs/LOCAL_DEPLOYMENT.md), or [model export](tools/README_MODEL_EXPORT.md), see the linked guides.

## Verify changes

Software behavior is covered by automated frontend and Rust checks. The
implemented coach has not yet been tried with the actual table camera in a
shooting session. Automated checks do not establish physical aiming accuracy
or how responsive the guidance feels during play; those need an ordinary
on-table trial, not a new qualification program.

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
