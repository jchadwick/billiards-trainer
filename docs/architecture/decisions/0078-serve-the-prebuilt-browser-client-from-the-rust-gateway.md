# ADR-0078: Serve the prebuilt browser client from the Rust gateway

- Status: accepted
- Date: 2026-10-22
- Deciders: Project maintainers
- Supersedes: None
- Superseded by: None

## Context

ADR-0066 keeps the existing browser application in HTML, JavaScript, and CSS
while moving every deployed server process to Rust. The React/Vite project
already creates a production bundle, but the Rust gateway currently exposes
only API and WebSocket routes. Starting the browser against the gateway
therefore returns 404 at `/`, leaving a split runtime or separate development
server necessary even after the bundle is built.

The bundle's asset paths are same-origin and its client-side routes need an
index-document fallback. API route misses and missing assets must keep their
404 behavior, and configured files must remain confined to the selected asset
directory.

## Decision

The Rust gateway may serve a prebuilt client from an explicitly configured
absolute `--web-root`. The web toolchain remains responsible for building the
static bundle; it is not a deployed application service. When the option is
omitted, the gateway remains API-only.

For a configured root, GET and HEAD requests serve files contained under that
root with an appropriate content type. Extensionless non-API paths fall back
to `index.html`; `/api` misses, missing extension-bearing assets, unsafe paths,
and paths escaping the root through symlinks return 404. Other methods return
405. The server does not introduce authentication or change API/WebSocket
contracts.

## Consequences

### Positive

- A single Rust process can serve the built browser client and its API on the
  same origin, removing a separate static-server process from deployment.
- Existing HTML/JavaScript/CSS, frontend dependencies, and production build
  remain unchanged; no Rust/WASM client is introduced.
- API misses remain distinguishable from client routes, and file lookup is
  bounded to the configured root.

### Negative

- A bundle must be built and deployed alongside the Rust gateway or supplied
  through the configured asset directory.
- The Rust service adds static-file reads and a small content-type policy; it
  does not replace the web toolchain used during development.
- Full browser qualification still depends on Rust implementations for the
  camera video and calibration API routes consumed by the existing client.

## Alternatives considered

### Keep a Python or Node static server in production

Rejected because the application server process tree must be Rust-only.
Node/Vite remains a build-time tool and can still be used for frontend tests.

### Convert the browser to Rust/WASM

Rejected because the browser must remain the existing standard web
application. That rewrite would expand scope without improving the server
runtime boundary.

### Serve every unknown request with `index.html`

Rejected because missing API routes and missing assets need to remain visible
as 404 responses instead of silently returning the application shell.

## Verification

- Unit tests verify root and asset serving, extensionless route fallback,
  missing API and asset 404s, encoded traversal rejection, and symlink escape
  rejection.
- Rust server CLI tests verify that the asset directory is optional and that
  configured paths must be absolute existing directories.
- The Vite production build creates `backend/static`; start the Rust gateway
  with that directory and verify that `/`, a generated JS asset, and an
  extensionless browser route return 200 while an unknown API route returns
  404.
- Route and browser compatibility beyond page delivery remain open until the
  Rust gateway implements the client's video and calibration endpoints.

## Related records

- [ADR-0066](0066-run-server-services-in-rust-and-retain-the-standard-web-client.md)
  defines the Rust server and standard browser boundary.
- [ADR-0068](0068-introduce-a-rust-latest-state-http-and-websocket-gateway.md)
  defines the Rust gateway foundation.
