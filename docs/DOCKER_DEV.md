# Local development

The supported app runtime is Rust. Python remains available for compatibility
and reference tests, plus optional ONNX model export; it is not needed to
launch the server. The browser client remains HTML/CSS/JavaScript built from
the React/TypeScript source.

## Run the Rust application from source

Install Rust, Node.js, and GStreamer development files available to
`pkg-config`. Configure `config.json` with a verified Rust ONNX/checkpoint pair,
capture source, and any available calibration, then run:

```bash
make run
```

The gateway builds and serves the browser client at
`http://127.0.0.1:8001` by default. `RUST_HTTP_PORT` overrides the local HTTP
port, while `RUST_SERVER_ARGS` can pass additional server CLI options. The
Rust live path requires `perception.rust_live.onnx_path`,
`perception.rust_live.source_checkpoint_path`, and the ONNX export manifest
beside the model. An invalid or missing model pair prevents live startup.

For a gateway-only run that does not open a capture source:

```bash
make run-api
```

To work on the browser client while the Rust gateway is running, start Vite in
a separate terminal:

```bash
cd frontend/web
npm ci
npm run dev
```

The dev client and packaged static client are browser applications; there is no
WASM runtime.

## Run the packaged container locally

The Compose image compiles the Rust server and capture executables, builds the
static browser client, and installs native GStreamer runtime components. It
does not install Python. Supply the same model and capture configuration used
for a native run, then:

```bash
docker compose up --build backend
```

For gateway-only checks set `START_LIVE=false` in the environment. Container
configuration and model mounts are described in [DOCKER.md](DOCKER.md).

## Optional Python tools

Python remains for compatibility/reference tests and optional ONNX model
export. Dataset preparation and standalone calibration-authoring workflows
are retired; Rust owns temporal evaluation and live calibration.
Install the lighter workflow with `make install-python-tools`; use
`make install-model-tools` only when exporting the ONNX model used by the Rust
runtime. Both install targets use the checked-in `uv.lock`; install `uv` first.
Rust owns supported temporal evaluation and live performance auditing.
Python tools do not launch the supported application server, and
their throughput is not a substitute for real-time live-camera
qualification.
