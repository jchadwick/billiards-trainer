# System Integration

The supported application runtime is Rust. The browser remains a standard
HTML/JavaScript/CSS client. Python modules remain compatibility/reference code
and an optional ONNX model-export tool; they are not launched by the
application.

## Runtime components

- `billiards-server` owns the HTTP and WebSocket gateway, static browser
  delivery, runtime configuration, and latest-state publication.
- The Rust live perception process and its Rust capture helper are supervised
  by the gateway when live mode is selected. State delivery is best-effort and
  freshness-first: superseded frames and stale results may be dropped.
- The frontend is built with the existing Node/Vite toolchain. The production
  bundle is static content served by `billiards-server`; Node is not required
  by the deployed application.
- Python API modules remain for compatibility tests and legacy references;
  the optional exporter creates ONNX inputs consumed by Rust.
  `python -m backend.main` and `python -m backend.dev_server` are retired
  commands and do not start a service.

## Run locally

From the repository root:

```bash
make run
```

This builds the browser bundle and runs the Rust server with live capture. To
run the gateway without capture, use `make run-api`. To use a different
configuration file, set `RUST_CONFIG=/path/to/config.json` before either
command. The Rust server's bind, port, scene transport, and optional live
settings can be passed through the Makefile's `RUST_SERVER_ARGS` variable.

The server provides liveness and readiness at `/api/v1/health/live` and
`/api/v1/health/ready`. The browser state stream uses the existing anonymous
WebSocket route at `/api/v1/websocket/ws`. There is no authentication or
credential setup.

For a web-only development workflow, `make run-frontend` starts Vite and
connects the browser to the separately running Rust gateway. Vite is a local
development tool; packaged and deployed runs use the Rust server for static
delivery.

## Package and verify

Use `docker compose build` and `docker compose up -d backend` for the local
container image. It builds the Rust binaries and browser assets from the
repository root, then launches the Rust binary as the container entry point.
Use `make deploy` to create a local distribution containing Rust executables,
configuration, and static browser files.

The primary development checks are:

```bash
make test
make lint
make format-check
make config-check
```

`make test-python` runs compatibility and offline authoring-tool tests; it does
not launch the Python API service. Physical-camera accuracy and freshness still
require qualification on the intended hardware; replay and the remote MJPEG
development fixture are integration evidence, not camera-ground-truth evidence.
