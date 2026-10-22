# Retained Python video compatibility code

The supported application does not run this Python video process. Camera
ownership and live frame delivery belong to the Rust server and its supervised
Rust capture helper. The old command `python -m backend.video` now exits with a
message directing users to `make run`.

The Python `VideoProcess`, shared-memory structures, and associated helpers
remain in the repository only where offline compatibility tests and migration
comparisons still use them. They are not copied into the Docker image or local
Rust distribution. The older distortion-correction document in this directory
describes a Python-only implementation; its settings are not active Rust
configuration and its examples do not describe current camera behavior.

## Supported application commands

```bash
make run
make run-api
```

`make run` builds the browser bundle and starts the Rust gateway with live
capture. `make run-api` starts the Rust gateway without live capture. See the
[system integration guide](../../docs/SYSTEM_INTEGRATION.md) for configuration
and packaging.

## Retained compatibility tests

Run the offline Python compatibility suite from the repository root:

```bash
make test-python
```

These tests check historical buffer and message contracts; they do not start a
Python camera service or qualify the Rust capture implementation.
