# Python API compatibility reference

The supported application server is the Rust `billiards-server` binary. This
FastAPI package is retained for offline compatibility and differential tests;
it is not part of the installed application and must not be started with
Uvicorn. The old `python -m backend.main` command exits with guidance to use
the Rust launcher.

## Start the supported application

From the repository root:

```bash
make run
```

Use `make run-api` to start the Rust HTTP/WebSocket gateway without live
capture. The Rust gateway serves the built HTML/JavaScript/CSS client and
retains the live-view state WebSocket contract. See the [API reference](../../docs/API_REFERENCE.md)
for documented HTTP and state-stream behavior.

## Compatibility tests

Run the retained Python API tests without launching an application server:

```bash
make test-python
```

Python app factories remain importable only where a test needs a reference
implementation. The supported application API is implemented and verified in
the Rust gateway; Python-only routes are not part of the active product
contract.
