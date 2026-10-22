# ADR-0081: Use direct Rust HTTP delivery without Nginx

- Status: accepted
- Date: 2026-10-22
- Deciders: Project maintainers
- Supersedes: ADR-0005 for enforcement in the supported Rust runtime only
- Superseded by: None

## Context

The supported application runs as one Rust server process that serves the
browser client, HTTP routes, and WebSocket state stream. The repository also
contains a separate Python deployment command that installs Nginx, plus two
unused Nginx configurations and a standalone Nginx frontend image. The Python
command generates a different configuration from the checked-in template; its
static alias points to a directory the bundle does not populate, and its
health route is stale. None of these Nginx paths is part of the supported
Compose, systemd, local-run, or distribution workflows.

The supported Rust gateway already bounds WebSocket concurrency, message
size, and slow-client writes, and serves the built browser bundle. It does not
implement Nginx's optional per-IP WebSocket attempt rate or general response
headers/compression. The user requires no authentication, and the app is
intended for a trusted local network. Reverse-proxy limits must not be
mistaken for Rust runtime guarantees.

## Decision

Use the Rust gateway directly for HTTP, WebSocket, and static browser delivery.
Build the local distribution through the existing shell-based `make deploy`
workflow, then run its `run.sh` or the Rust systemd unit. Do not install or
launch Nginx as part of application deployment. Retire the obsolete Python
deployment command, unused Nginx configuration, and standalone Nginx frontend
image. Keep the browser development server and Python offline compatibility,
model, and evaluation tools outside the deployed application.

For the supported runtime, use Rust's existing global WebSocket concurrency
cap, bounded message size, write deadline, and slow-client disconnect. There
is no per-IP connection-attempt rate limit in the Rust gateway. Reconsider
that bound before exposing the application outside its intended trusted local
network; do not add authentication as a substitute.

## Consequences

### Positive

- One application process owns capture, state, API, WebSocket delivery, and
  static browser files.
- Local development, Docker, systemd, and distribution no longer imply an
  optional Nginx service or a second static-file implementation.
- Deployment tooling uses the existing shell packaging path instead of a
  parallel Python installer that configures a mismatched service topology.

### Negative

- The Rust runtime does not currently provide per-IP WebSocket attempt
  limiting, gzip, or Nginx response headers. The gateway retains its global
  connection/message/write bounds; these do not provide per-client fairness.
- Direct serving uses conservative cache revalidation and may consume more
  bandwidth than immutable caching or compression for repeated browser loads.
- Operators who previously invoked the undocumented Python installer must
  build/copy the local distribution and use its Rust run entry point.

## Alternatives considered

### Keep Nginx as a thin reverse proxy

Rejected because it leaves a second server process on the target, duplicates
static/API routing, and the current generated configuration does not match the
checked-in template or the Rust bundle layout. Its per-IP policy can be
reconsidered as a Rust gateway feature if deployment moves beyond the trusted
local-network assumption.

### Port the Nginx policy into Rust before retiring the alternate path

Deferred because the supported runtime path did not use the generated Nginx
configuration, the live service already applies global bounded WebSocket
admission and slow-client isolation, and there is no measured requirement for
per-IP admission in the current single-table deployment. The missing per-IP
bound remains an explicit revisit condition.

## Verification

- Repository tests assert that the supported package and systemd unit start
  the Rust gateway directly and that `make deploy` invokes the shell bundle
  builder.
- Rust, browser, and Python compatibility suites pass; Python deployment code
  is no longer importable or runnable as a deployment path.
- Shell syntax checks pass for the local distribution scripts.
- A clean local Rust Docker image smoke serves `/live` and `/api/v1/health/live`
  without Python or Nginx in the image.

## Related records

- [ADR-0005](0005-distributed-rate-limiter.md) remains historical for the
  offline Python API reference; its per-IP runtime enforcement is superseded
  by the direct Rust deployment contract above.
- [ADR-0066](0066-run-server-services-in-rust-and-retain-the-standard-web-client.md)
  defines the Rust-server and standard-browser boundary.
- [ADR-0078](0078-serve-the-prebuilt-browser-client-from-the-rust-gateway.md)
  assigns static browser delivery to Rust.
- [ADR-0080](0080-use-rust-as-the-only-supported-application-server.md)
  makes Rust the only supported application server entry point.
