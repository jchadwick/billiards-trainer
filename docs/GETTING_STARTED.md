# Getting Started

Billiards Trainer runs its supported server, camera capture, and live-state
delivery in Rust. The browser client remains a standard HTML, CSS, and
JavaScript application built from `frontend/web`.

## Prerequisites

- Rust toolchain with Cargo
- Node.js and npm
- GNU Make
- GStreamer 1.x development files discoverable through `pkg-config` to build
  camera-enabled Rust binaries, plus the matching runtime and camera provider
  plugins to use a physical camera

Install system prerequisites with your operating system's package manager.
`make dev` does not install system packages or create a Python environment. It
prints a warning when GStreamer development files are unavailable; the
default non-camera development setup can still complete, but camera-enabled
builds and physical-camera capture require the GStreamer components above.

## Setup and run

```bash
make dev
```

This fetches locked Rust dependencies, installs the locked browser dependencies,
creates the local data/log/temp directories, and verifies the required Rust and
browser toolchain. Prepare `config.json` with the verified model paths and
capture settings, then run:

```bash
make test
make run
```

The local Rust gateway serves the browser application at
`http://127.0.0.1:8001`. `make run-api` starts the gateway without live capture;
`make run-frontend` starts the separate Vite development server.

## Retained Python tooling

Python is not part of the supported server runtime. The repository keeps Python
for compatibility/reference tests and optional ONNX model export. The removed
dataset and legacy calibration experiments are not supported tooling. The
remaining Python workflows are opt-in:

```bash
make install-python-tools
make test-python
make lint-python
make format-python
```

These commands use the checked-in `uv.lock`; install `uv` before setting up
the optional Python environment. The larger model-export dependencies remain
separate from the compatibility and test environment.

Install the larger Ultralytics/ONNX stack only when exporting a model for the
Rust runtime:

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
