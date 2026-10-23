# Getting Started

Billiards Trainer runs its supported server, camera capture, and live-state
delivery in Rust. The browser client remains a standard HTML, CSS, and
JavaScript application built from `frontend/web`.

## Prerequisites

- Rust toolchain with Cargo
- Node.js and npm
- GNU Make
- GStreamer development and runtime packages for the default camera-enabled
  Rust build

Install the prerequisites with your operating system's package manager. The
setup script does not install system packages or create a Python environment.

## Setup and run

```bash
./setup.sh
```

The script fetches Rust dependencies, installs the locked browser dependencies,
and checks the development toolchain. Prepare `config.json` with the verified
model paths and capture settings, then run:

```bash
make test
make run
```

The local Rust gateway serves the browser application at
`http://127.0.0.1:8001`. `make run-api` starts the gateway without live capture;
`make run-frontend` starts the separate Vite development server.

## Retained Python tooling

Python is not part of the supported server runtime. The repository keeps Python
for compatibility tests, dataset/calibration authoring, and optional model
export. Those tools are opt-in:

```bash
make install-python-tools
make test-python
make lint-python
make format-python
```

Install the larger Ultralytics/ONNX/CoreML stack only when exporting a model:

```bash
make install-model-tools
```

The retained Python test suite is also run by its dedicated CI workflow. Rust
and browser tests remain runnable without a Python environment.

## Useful commands

```bash
make help              # Show available targets
make dev               # Fetch Rust and browser dependencies
make build             # Build the browser client and Rust application
make test              # Run Rust and browser tests
make test-rust-camera  # Test the production GStreamer camera feature
make test-all          # Also run camera-feature and retained Python tests
make lint              # Check Rust and browser code
make format            # Format Rust code
make run               # Run the Rust server with live capture
make run-api           # Run the Rust gateway without live capture
```

See [the runtime README](../README.md), [Rust perception notes](../perception/README.md),
and [architecture decisions](architecture/README.md) for model, camera, and
qualification details.
