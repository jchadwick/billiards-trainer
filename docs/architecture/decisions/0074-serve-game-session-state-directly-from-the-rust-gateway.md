# ADR-0074: Serve game-session state directly from the Rust gateway

- Status: accepted
- Date: 2026-10-22
- Deciders: Project maintainers
- Supersedes: None
- Superseded by: None

## Context

The Rust gateway already owns the current ordered scene and serves
`GET /api/v1/game/state`, but the response currently omits game-session
metadata. The transitional Python route reads and resets that metadata by
starting a one-shot Rust authority process. ADR-0067 already assigns the
revisioned session record and reset semantics to Rust, and says the future Rust
service should call the same store directly.

Ball locations remain the primary state. Session metadata must not delay or
erase localization, synthesize a rack, or modify calibration. Reset must return
the newest actual scene after the storage operation, preserve the current
anonymous HTTP contract, and reject `preserve_table=false`. History, events,
exports, physics, and ball-number classification are outside this slice.
The browser stays HTML/JavaScript/CSS.

## Decision

Add direct Rust gateway routes for `GET /api/v1/game/state` session overlay and
`POST /api/v1/game/reset`, using the existing `GameSessionStore` in-process.
Resolve its file path, size bound, and lock timeout from the effective
configuration document. Relative store paths are resolved against the explicit
deployment project root, or the service working directory when no root is
provided. Both relative and absolute paths must resolve under that root;
the gateway opens the canonical root from `/` component by component without
following symlinks and performs each store operation with descriptor-relative,
no-follow file access, so replacing a checked parent with a symlink cannot
redirect a later read or reset. The deployment owns the project-root directory
hierarchy; untrusted processes must not relocate it while a request is running.
A missing or invalid
optional session configuration must not
prevent the gateway from serving ball localization; state returns the scene
with unavailable session metadata, while reset fails closed with HTTP 503.

Run bounded filesystem operations on Tokio blocking workers. Move the bounded
capacity permit into the blocking worker so cancellation of an HTTP request
cannot release capacity while its file operation continues. State reads are
best effort: if the store is unavailable or its bounded request capacity is
full, return the scene with null game-session metadata. Reset acquires the
bounded session capacity, rejects a request that asks to discard the table,
loads the current revision, commits through the store's compare-and-swap reset,
then reads the current scene again. A stale concurrent reset maps to HTTP 409;
store or capacity failures map to sanitized HTTP 503. Reset keeps its legacy
reset-specific unavailable-scene response. The response reports
`backup_created: false`; no history or positions are manufactured or cleared.

Keep session metadata in its separate durable store. Preserve the current
state/reset route paths and response envelopes. Do not change the browser or
remove Python routes in this side-by-side qualification stage.

## Consequences

### Positive

- The Rust service owns the complete read/reset boundary for its own scene and
  session state without starting a helper process per request.
- Session read errors remain isolated from localization freshness and delivery.
- Reset semantics remain revision-safe and preserve the newest measured scene.
- The browser's established HTML/JavaScript/CSS client can continue to call the
  same HTTP endpoints after eventual gateway cutover.

### Negative

- A bounded blocking worker is still required for filesystem locking and
  durable writes; gateway capacity needs explicit limits.
- Session metadata remains a separate persistence domain and is not yet a full
  game-state engine. Scores/player changes, event history, export and backup
  remain unimplemented in Rust.
- The transitional Python API and the Rust side-by-side gateway both exist
  until route parity and operating gates pass.

## Alternatives considered

### Retain the one-shot helper process in the Rust gateway

Rejected because the gateway and domain code are now in the same Rust service,
so the process hop adds latency and another startup/packaging failure surface.

### Merge session metadata into the scene lifecycle store

Rejected because game commands and captured observations have distinct
ownership, update rates, and persistence requirements. Reset must not advance
scene order or alter location freshness.

### Make session-store failure fail the state route

Rejected because ball localization is primary and metadata is optional. Reads
degrade only the metadata fields; reset remains fail-closed because it mutates
durable state.

### Return a synthetic rack or clear localization on reset

Rejected because reset cannot manufacture measured positions or discard a
still-current observation.

## Verification

- Axum route tests cover unavailable scene, configured/unavailable/corrupt
  session store, metadata overlay without location changes, reset game types,
  invalid query values, `preserve_table=false`, compare-and-swap conflicts,
  current-scene reread, bounded request capacity under request cancellation,
  root-confined store paths including post-start symlink swaps, and anonymous
  access.
- Rust all-target tests, strict Clippy, formatting, and the repository ADR
  validator pass.
- Python route contract tests remain as the compatibility reference; no web
  build or WASM toolchain change is introduced.

## Related records

- [ADR-0066](0066-run-server-services-in-rust-and-retain-the-standard-web-client.md)
  keeps the browser standard HTML/JavaScript/CSS.
- [ADR-0067](0067-own-game-session-metadata-in-rust.md) defines the session
  record, reset semantics, and scene-preserving behavior.
- [Rust domain migration plan](../../../thoughts/rust-domain-migration-plan.md)
  tracks the staged server migration.
