# ADR-0080: Use Rust as the only supported application server

- Status: accepted
- Date: 2026-10-22
- Deciders: Project maintainers
- Supersedes: None
- Superseded by: None

## Context

The local development, Docker Compose, systemd, and distribution paths now
build or launch `billiards-server`, which serves the browser bundle and owns
the application runtime. The packaged target does not contain Python. However,
`backend.main`, `backend.dev_server`, and `backend.video` still launch Python
services from source, and several guides still describe those commands as
supported application startup. That makes it unclear whether the Rust
migration is complete and permits a second server or camera owner to be
mistaken for the product runtime.

Python API modules remain useful as differential references and compatibility
fixtures in the offline test suite. Removing those modules or their Python
testing dependencies now would be a separate migration with broader impact on
the acceptance harness. The browser remains HTML, JavaScript, and CSS as
required by ADR-0066.

## Decision

The Rust `billiards-server` binary is the only supported application server
entry point for development, installation, and deployment. `make run` starts
the Rust server with live capture; `make run-api` starts the Rust gateway
without live capture. Python application modules remain importable only as
offline compatibility references and test fixtures. Their command-line
launchers must fail clearly and direct users to the Rust commands; they must
not start Uvicorn, FastAPI, camera capture, or any other application service.

Keep Python tools that prepare or evaluate data, train or inspect models, and
compare runtime behavior off the deployed machine. Keep the browser as the
existing HTML/JavaScript/CSS client. Do not add numbered-ball classification
or authentication as part of this cutover.

## Consequences

### Positive

- A documented or accidental Python command cannot start a competing server
  with different routes, configuration, or lifecycle behavior.
- Development and deployment use the same Rust gateway and capture owner.
- Python compatibility tests and offline model/evaluation tools remain
  available without being included in the application package.

### Negative

- Python API compatibility tests continue to carry dependencies for reference
  modules, even though those modules are not supported application services.
- Older scripts and instructions must be updated, and users invoking a retired
  Python command receive a migration message instead of a server.
- Python-only API behavior not implemented in the Rust gateway remains
  unavailable to the supported application until migrated and qualified.

## Alternatives considered

### Keep Python launchers as development conveniences

Rejected because a second executable server obscures which implementation is
supported and can give misleading results when its API or lifecycle differs
from Rust.

### Delete all Python API modules and tests in one change

Deferred because differential and compatibility suites still provide useful
evidence for remaining route and data-contract gaps. The test reference can be
removed after each dependent contract has migrated and no longer relies on it.

### Convert the browser or development tools to Rust

Rejected because the browser must remain a standard web application, and
Python data/model/acceptance tooling is explicitly outside the deployed server
runtime.

## Verification

- Unit tests assert `python -m backend.main`,
  `python -m backend.dev_server`, and `python -m backend.video` exit with status
  2 and direct users to the Rust Make targets without starting a listener or
  camera process.
- Rust and browser suites pass; Python compatibility tests still import the
  reference app factory without exposing a Python server command.
- Docker CI builds the repository-root Rust image at the actual `runtime`
  stage, and its image contents and entrypoint contain no Python service.
- Local distribution and systemd artifacts continue to execute
  `billiards-server`.

## Related records

- [ADR-0066](0066-run-server-services-in-rust-and-retain-the-standard-web-client.md)
  defines the final Rust-server and standard-browser boundary.
- [ADR-0078](0078-serve-the-prebuilt-browser-client-from-the-rust-gateway.md)
  defines Rust static delivery for the retained web client.
- [ADR-0079](0079-reduce-the-first-rust-hosted-client-to-live-view.md)
  limits the first Rust-hosted web application to the live view.
