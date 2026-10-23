# Migrate all deployed server services to Rust and retain the standard web client

Destination: every deployed server component runs in Rust, including the final
HTTP/WebSocket service, static-asset delivery, process/device supervision,
camera capture, image processing and inference, localization/tracking,
configuration and calibration authority, game state, physics and prediction.
The current Python backend is a temporary gateway during staged migration and
will be removed from the deployed server after Rust service parity and operating
gates pass. Keep the browser application as HTML/JavaScript/CSS; do not convert
it to Rust/WASM. Python remains an allowed off-device tool for model
training/export, annotation, and independent reference inference. Thin
development glue uses portable shell scripts; complex, stateful acceptance and
audit tools migrate to Rust, with Python retained only as a differential
reference until parity gates pass.
Ball-number classification is explicitly excluded because it currently does
not work; nullable number fields may remain in shared contracts. Preserve the
existing browser API while server ownership migrates; defer any necessary web
integration work until the Rust server is qualified.
The accepted
[ADR-0066](../docs/architecture/decisions/0066-run-server-services-in-rust-and-retain-the-standard-web-client.md)
records this corrected boundary. The earlier
[migration history](rust-perception-migration-plan.md) remains useful context,
but its permanent Python service/domain ownership and Rust/WASM client
destination are superseded. [ADR-0067](../docs/architecture/decisions/0067-own-game-session-metadata-in-rust.md)
defines the first Rust game-session metadata and reset slice.

## Verification before cutover

Preserve HTTP/WebSocket contracts and current user-visible behavior as
executable tests. Differentially compare Rust runtime-domain results with the
Python reference on deterministic fixtures. Test exclusive camera ownership,
configuration/calibration revisions, state expiry, overload, slow clients,
process shutdown, restart, persistence failures and recovery. Runtime cutover
requires Rust ownership for every deployed server responsibility, real-camera
and real-model evidence, physical/temporal accuracy and freshness checks, and
macOS/Linux builds. The final deployment must contain no Python server process;
Python training and acceptance tools remain off-device. Keep the browser
HTML/JavaScript/CSS client and its contracts; make any necessary client
integration changes only after Rust server gates pass. No Rust/WASM client is
planned.

## Development-device-first order

The current execution order is to finish and qualify the first migration pass
on this development Mac before deploying the application to the target:

1. Run the Rust detector, state gateway, and existing browser against frames
   from the user's remote MJPEG stream at `192.168.1.31:8001`. Treat that
   stream as an external test fixture; keep its user-started process running
   and do not open the target camera directly during this phase.
2. Migrate and exercise the application server responsibilities on the Mac,
   including the current browser routes, configuration/calibration behavior,
   startup/shutdown, and static assets. Confirm that no Python process is part
   of the local Billiards application runtime. Python model training/export
   and offline reference tools remain allowed. The user-provided stream server
   is outside the Billiards application runtime and remains available only as
   the requested input fixture.
3. Only after the complete local Rust server and browser path passes its
   functional checks, rebuild for the target and perform target-specific
   camera/runtime qualification. Do not use the target as the migration
   debugging environment.

For the first Mac integration gate, record the source as a remote MJPEG
publication with no verified camera-exposure timestamp. Do not claim
exposure-to-browser latency from this stream. Compare fresh localized states
and positions first; evaluate accurate physical coordinates separately with
verified calibration and labels.

## Sequential module tasks

1. Native public scene state: port complete source-to-client conversion,
   including image scale, calibrated units, unknown numbers, cue and observation
   provenance. Add receipt-local freshness, epoch ordering and latest-state
   lifecycle. Reuse perception and public contract fixtures. The transitional
   Python gateway may translate the native read model to current HTTP/WebSocket
   envelopes until the Rust service assumes those contracts.
2. Native configuration and calibration authority. Calibration solving,
   revisioned storage, stable capture-context validation, next-session
   activation, and a bounded Rust control protocol are implemented. Its
   FastAPI read/write adapter now relays opaque requests and responses to Rust.
   The unified-configuration sequence is: (a) implement one Rust-owned,
   revisioned store for the current document, including embedded defaults and
   schema, merged validation, revisioning, and atomic persistence; this store
   and its explicit-path one-shot command are implemented and pass Rust
   formatting, strict Clippy, 160 library tests, and 16 command integration
   tests; (b) make the transitional Python service a fail-closed read-through/
   command adapter, preserving route contracts and initializing its snapshot
   before configuration-dependent imports; then (c) move production capture-
   settings resolution into Rust. The production v3 capture-settings resolver
   now runs through the Rust authority and validates its generated request with
   the Rust wire parser; the Python implementation remains only in offline
   differential and audit tooling. The live supervisor requires Rust authority
   mode and fails closed rather than using Python defaults. The Python HTTP
   adapter remains active only until its compatibility and failure tests pass,
   after which the Rust service will call the same authority directly. See
   accepted ADR-0065. Preserve the calibration route contracts and defer
   browser authoring integration until Rust server qualification.
3. Native live capture and perception: compose sole camera ownership, frame
   sampling/latest-frame replacement, bounded inference, tracking/localization,
   and supervised recovery. Keep preview as a cached view of the Rust-owned
   current frame, never a competing Python camera.
4. Native game-state domain: migrate state ownership, validation/correction,
   events, reset/history/export operations and existing game routes. Reuse
   differential fixtures, and do not duplicate Rust's existing location-state
   ordering/freshness lifecycle.
5. Native physics and prediction: migrate only after current locations and game
   state are stable. Keep trajectory work behind independent bounded budgets
   and preserve existing invalidation, coalescing, freshness and shutdown rules.
   Ball-number classification is excluded from this migration.
6. Build the Rust HTTP/WebSocket gateway incrementally, beginning with a
   side-by-side current-scene state path. Then add static frontend delivery,
   request/response translation, process/device supervision and operating
   health. Remove the Python server process only after every route, lifecycle,
   failure-handling and packaging gate passes. Version local interfaces; no
   silent Python domain fallback. The early state-path prototype does not change
   the staged cutover requirement.
7. Complete compatibility and operations: preserve all current client routes,
   streaming/diagnostics/health/logs, training exports and projector workflows.
   Verify source ordering, health, current state and bounded slow-client behavior
   end to end through the retained gateway.
8. Package and qualify the complete Rust server for macOS/Linux, including the
   API gateway, native inference/capture runtimes, static assets and process
   supervision. Verify installed processes and dependencies, run API/real-camera
   acceptance, accuracy, freshness and bounded-memory gates, and exercise
   rollback before removing the Python service. Retain external Python
   tools/tests.
9. Browser integration, last: keep the current HTML/JavaScript/CSS application.
   After Rust server gates pass, make only the client changes required to
   consume the Rust routes and WebSocket messages. Preserve current workflows,
   freshness/order semantics, calibration interaction and supported browsers.
   Continue using the existing web build and browser test toolchain; no
   Rust/WASM framework is selected or planned.

## Current implementation status

- **2026-10-22 Rust audit-tooling migration:** ADR-0082 sets portable shell as
  thin development glue and Rust as the owner for stateful or complex tools.
  Commits `77147f5` and `341bc05` add the bounded WebSocket state receiver and
  opt-in process-resource collection. Commits `19aab7a`, `2bd9046`, and
  `2747095` move live qualification resource sampling and timing reduction to
  Rust. Commits `6feb7e7` and `8e74dff` add the native server-event reducer and
  route the Python collector through it. Commits `f445a07` and `b0be61a` add
  the native delivery-latency reducer and route the collector through it too.
  Commit `4eacd09` adds the native instrumentation-overhead comparison and a
  shell entry point for direct use. Shared golden fixtures check the full
  server-event and delivery-latency summaries and the overhead-comparison
  outcome, including clock comparability, state and trajectory correlation,
  deadline misses, component age, unique observations, resource trends,
  workload matching, evidence thresholds, and pass/over-budget/inconclusive
  status. The Rust all-target suite passes 253 library tests plus the
  integration targets; strict Clippy and formatting pass. The focused
  collector/configuration suite passes 148 tests, plus Ruff, Black, and mypy.
  The 5,074-line Python collector still owns event-file collection and
  clock-window selection, warm-up/repetition and replay orchestration,
  trajectory subscriptions, provenance and baseline selection, accuracy
  evaluation, and Markdown output. Keep it until these remaining contracts have
  Rust implementations and differential coverage. This tooling slice did not
  run a new full-app audit or establish a sustained performance baseline.

- **2026-10-22 Mac recorded-video live smoke:** On the development M4 Max, the
  Rust gateway and supervised live detector used the real two-class ONNX
  export and source checkpoint with the paced shell MJPEG fixture
  (`test_data/demo2_short.mp4`, 1280×720 at 30 FPS). At revision
  `621f80d50554517b2fff282db98b2e133467ba8d`, the gateway became ready and a
  short WebSocket sample delivered 10 state messages covering 7 distinct
  transport frames, with 11–12 reported balls and `fresh` observation status.
  Sampled read-completion-to-publication ages were about 98 ms. The input,
  ONNX, and checkpoint SHA-256 values are respectively
  `4e09e30c49fed7e134f43f276c629394c209757e0440958a51ab148f12f6e388`,
  `7beb3af7923290a5524fe1744de13bd66f4e43ea8a5048347bc8a12c51a1e4b5`, and
  `f04470378b10f02d7467dad046ad26fb48eaeb1002e7866cfc37f501b478dc0e`.
  This is a short functional smoke, not accuracy, camera-exposure latency, or
  sustained performance evidence. The remote target MJPEG endpoint initially
  returned data and the isolated helper decoded three frames, but later
  connection attempts timed out before the live detector received a first
  frame; do not treat the remote source as qualified by this run.

- **2026-10-22 Rust gateway foundation (accepted):** ADR-0068 adds a separately
  launchable Axum/Tokio service on an alternate port. It accepts the existing
  replay-v2 scene stream, uses the Rust scene lifecycle, and serves current
  state over anonymous HTTP and latest-value WebSocket delivery. The complete
  Rust target suite, strict Clippy/fmt, repeated real slow-reader tests, and a
  standalone replay-v2-to-HTTP process smoke pass. Python remains the active
  gateway; this is not a live-camera or deployment cutover. The browser remains
  HTML/JS/CSS and received no integration change in this stage. Versioned
  game-session state, configuration reads, calibration routes, static delivery,
  full process/device supervision and the rest of the API remained later gateway
  work at that initial foundation point; subsequent entries record the routes
  already moved.

- **2026-10-22 live-v3 gateway ingress:** ADR-0069 adds request-bound readiness
  validation to the Rust listener using the current Rust configuration
  authority. The gateway checks the v3 acknowledgement against the exact
  revision/digest and compatible source report before returning `ready`, then
  checks source identity and emitted frame geometry for each delivery. Ball
  state is rebuilt from the validated raw scene; table/game metadata and
  elapsed freshness ages are carried from the producer, with all server
  freshness decisions rebased to receiver-local monotonic time. Replay-v2
  remains available without configuration. The browser remains
  HTML/JavaScript/CSS and is untouched. Strict acknowledgement, invalid-ready,
  live delivery, lifecycle, HTTP/WebSocket, and slow-consumer cases pass the
  Rust all-target suite; live hardware cutover and the rest of the gateway API
  remain open.

- **2026-10-22 Rust configuration read routes:** ADR-0070 moves
  `GET /api/v1/config` and `GET /api/v1/config/schema` onto the Rust gateway,
  calling the existing Rust-owned authority directly. It preserves the current
  API envelopes, maps request errors into the deployed error format, and runs
  filesystem-backed authority operations on bounded blocking workers. It did
  not modify the HTML/JavaScript/CSS client.

- **2026-10-22 Rust configuration update route:** ADR-0071 adds both
  `PUT /api/v1/config` and its trailing-slash form to the Rust gateway. It
  preserves section normalization, validation-only/warning behavior, request
  envelopes, forced updates and restart reporting. A gateway-local mutation
  gate plus the authority's original-revision compare-and-swap protects
  deployment executable settings across validation and write. Validation-only
  does not persist legacy revision migration. Focused Axum tests pass; the full
  Rust all-target suite and strict all-target Clippy/format checks now pass.
  Import/export, remaining Python routes, deployment cutover and browser
  integration remain subsequent work; the browser stays HTML/JavaScript/CSS.

- **2026-10-22 Rust configuration reset route:** ADR-0072 moves
  `POST /api/v1/config/reset` to the same Rust gateway mutation path. The route
  remains anonymous and query-only; it preserves `all`, historical full-reset
  `user`, and ordered repeated `section` behavior, validates confirmation
  before configuration reads, ignores request bodies and unknown query values,
  and commits against the exported authority revision without retry. It uses
  the existing bounded configuration permit and gateway mutation gate. Tests
  cover aggregated query errors, duplicate scalar last-value behavior, warning
  precedence, section preservation, trusted executable defaults, corrupt
  stores, no-op revision stability, bounded shared capacity and response
  compatibility. All 23 configuration HTTP tests and the
  Rust all-target suite pass. Import/export and remaining route parity remain
  open; the browser stays HTML/JavaScript/CSS.

- **2026-10-22 Rust configuration import/export routes:** ADR-0073 moves
  multipart JSON/YAML import, JSON/YAML export, and attachment download onto
  the Rust gateway. Import preserves merge/replace, validation-only, warning
  override and CAS behavior, strips the transport-only `_metadata` field, and
  rejects HTTP changes to deployment-managed executable paths. The request
  body, blocking work, configuration concurrency and serialized output are
  bounded. Downloads return a bounded in-memory serialized payload and do not
  create shared temporary files. The browser remains HTML/JavaScript/CSS;
  Rust/WASM is outside the migration. All 31 configuration HTTP tests and the
  Rust all-target suite pass. The Python reference suite reports 3,161 passed,
  two model-dependent skips and 80.04% coverage. Rust formatting, strict Clippy
  and validation of all 73 ADRs pass. Python route parity, remaining server
  routes, static-asset packaging, browser integration and deployment cutover
  remain open.

- **2026-10-22 Rust game-session HTTP boundary:** ADR-0074 moves
  `GET /api/v1/game/state` session metadata and `POST /api/v1/game/reset`
  directly into the Rust gateway. The gateway reads and mutates the existing
  revisioned store in-process on bounded blocking workers, keeping capacity
  reserved until worker completion even if an HTTP request is cancelled. The
  store path is constrained to the deployment root and checked against
  traversal/symlink escapes. Session read or capacity failures leave the
  current location response available, and reset rereads the scene after
  commit. It rejects table mutation, returns no fabricated backup, and keeps
  localization/calibration independent. Each store operation uses an anchored
  securely opened root descriptor and no-follow path traversal, so post-start
  symlink swaps cannot redirect access outside the deployment root. The
  deployment owns that directory hierarchy and does not relocate it during a
  request. The browser remains
  HTML/JavaScript/CSS and is unchanged. History, events, session score updates,
  export, physics, remaining route parity and deployment cutover remain open.

- Rust inference, tracking, geometry, bounded capture, runtime OpenVINO and
  audit contracts exist; reuse and qualify them.
- Rust scene-to-public-state conversion matches 35 Python-oracle fixtures and
  its six semantic contract tests pass.
- Rust latest-state lifecycle now enforces explicit producer epochs, strictly
  increasing transport/snapshot order, source observation deduplication,
  receipt-local freshness, configured stale/expiry transitions, and bounded
  ball retention. Seven deterministic lifecycle tests pass.
- The live capture pipeline now runs that lifecycle and sends a versioned
  public-state delivery containing both the Rust-owned current state and the
  unchanged raw scene snapshot. The Python live gateway validates, caches, and
  relays this state without running its Python state adapter. Production live
  configuration requires this event and fails closed if it is missing.
- Explicit paced Rust replay now applies the same public-state lifecycle as
  Rust live capture and sends the versioned native delivery envelope through
  the gateway. Source arrival is recorded before inference, stale localization
  is expired by the shared lifecycle, and raw `SceneSnapshotV1` JSONL remains
  unchanged. The Python provider's separate replay path remains available as
  the default/reference implementation until broader parity and model-backed
  acceptance evidence is complete.
- The Python service now supports explicit Rust live selection behind the API.
  That path starts only the Rust live supervisor and publisher; Python Core,
  Vision, and integration services remain stopped. The bundled Python provider
  remains a temporary rollback/default during qualification, not the accepted
  destination architecture.
- Legacy Python camera preview, frame capture, and direct camera diagnostics
  return 503 when Rust owns live capture. The Rust-owned current-frame preview
  is now available through the bounded loopback listener and Python byte-proxy
  route, without a second camera owner.
- A target-host API startup attempt now reaches Rust source selection when the
  container receives read-only `/run/udev` and `/sys` metadata mounts. The
  helper then times out before its first sample, so live inference and client
  state remain unqualified on the target.
- The Rust configuration authority and first game-session metadata/reset slice
  exist, but the complete Rust server gateway and full game-state, physics, and
  prediction domains remain incomplete. The side-by-side Rust gateway now also
  serves configuration read/schema/update/reset routes, while the active
  Python service still hosts the rest of the public API. The current browser
  calibration wizard can display the
  Rust preview and map clicks to source pixels. Browser integration remains
  frozen until the complete Rust server is qualified; the client remains
  HTML/JavaScript/CSS and will not become Rust/WASM. Browser integration remains
  deferred; remaining route parity is being built directly in the Rust gateway
  while the Python service stays available for side-by-side qualification.
- **2026-10-22 game-session progress:** Rust now owns the bounded revisioned
  session record for game type, active player, scores and break state. The Rust
  gateway reads and resets the store in-process, exposes metadata separately
  from Rust scene localization, and uses compare-and-swap for resets. Reset
  returns the actual scene read model and leaves measured ball positions and
  calibration untouched. The session store path is confined beneath the
  deployment root, including symlink resolution; cancellation cannot release
  request capacity before a blocking operation ends. The browser files remain
  unchanged. History, event archive, export and backup are not
  implemented in Rust mode; broader game-state ownership and the in-process
  Rust HTTP service remain later work. See ADR-0067.
- **2026-10-22 earlier session bridge:** While FastAPI remains the active
  compatibility service, its session adapter invokes the packaged Rust
  authority companion and not an API-editable executable path; HTTP config
  updates/imports cannot replace installed server executables. The Rust gateway
  now bypasses that subprocess bridge and uses the same store in-process for
  its own state/reset routes. The browser remains HTML/JavaScript/CSS and is
  deferred to the final integration stage.
- **2026-10-22 calibration progress:** Rust now solves four ordered table
  corners into the existing provenance-bound, center-origin millimeter
  contract. Its revisioned store validates each bundle, uses compare-and-swap
  writes and monotonic clear tombstones, and replaces files atomically under a
  cross-process lock. Solver parity tests use the Python exporter fixture; the
  store tests cover conflicts, concurrent writers, malformed records, clear,
  and write failures. See ADR-0059 and ADR-0060.
- The Rust store now connects to the live capture process and the bounded
  calibration-control protocol. It validates the descriptor's active stable
  capture context, solves four ordered source-pixel corners with explicitly
  measured dimensions, writes through compare-and-swap, and activates a
  compatible calibration in the next process. Tombstone revisions remain
  visible, while malformed or mismatched records leave ball detection running
  with physical coordinates unavailable. FastAPI exposes bounded snapshot and
  submit routes as a byte relay and maps explicit Rust statuses to HTTP. The
  focused gateway suite passes 150 tests, all Rust targets pass, and the ADR
  validator accepts 65 decisions. The browser wizard is not connected to the
  store routes; frontend integration remains deferred to stage 9. macOS/Linux
  filesystem qualification, target-camera operation, and physical accuracy
  remain open.
- **Configuration progress (2026-10-22):** the Rust authority foundation now
  embeds the v1 defaults/schema and supports bounded snapshot, schema, defaults,
  validation, revision-checked update/replace/reset, and export operations over
  an explicit `--config` path. It serializes cooperating processes, uses
  compare-and-swap revisions and atomic replacement, and fails closed when a
  requested result cannot be emitted within the response limit. Adversarial
  review led to regression fixes for output-size-before-commit ordering,
  trusted-proxy CIDR validation, and URI control characters. Rust formatting,
  strict Clippy, all targets, and isolated command tests pass. The bounded
  Python client, read-through `Config` facade, early startup bootstrap, and
  configuration-route delegation are implemented for explicit Rust authority
  mode. Rust mode routes validation, defaults/schema, export, reset and writes
  through the Rust authority; they fail closed and map conflicts, warnings,
  validation errors and authority outages to HTTP. Ten facade tests, 31 route
  tests, 84 validation tests, and strict type/lint/format checks pass. A real
  one-shot Rust CLI smoke test passed snapshot, validation, compare-and-swap
  update, and persistence against an isolated temporary file. This is not the
  production default cutover: Python remains selectable/default during
  qualification, production capture-settings resolution now runs through the
  Rust authority, and the Python resolver remains an offline differential
  oracle. Rust live startup requires Rust authority mode; the active request is
  checked against the opened source before its first read. Desired-versus-
  applied reporting and final Rust service-gateway integration still need
  completion. The Python adapter will be replaced after domain migration and
  compatibility qualification.
- **2026-10-22 paced replay progress:** Rust replay TCP now publishes the
  versioned Rust-owned public-state envelope; the Python gateway validates,
  caches, and relays it without re-adapting the snapshot. Replay uses the
  identity handshake without a live capture acknowledgement, while live still
  requires the request-bound capture-settings acknowledgement. JSONL remains
  raw scene v1. Rust and Python tests cover envelope identity, stale expiry,
  raw export compatibility, and HTTP/WebSocket gateway delivery. See
  [ADR-0061](../docs/architecture/decisions/0061-use-rust-public-state-lifecycle-for-paced-replay.md).
- The separate Python provider/replay implementation is still present and
  remains the configured default/reference path. Rust replay is an explicit
  selection; this slice does not claim default cutover or real-model freshness
  qualification.
- Python remains the configured default provider and still owns the deployed
  Core/Vision/Integration path, general configuration, game state, physics,
  and prediction. No default cutover is claimed. The accepted target is a Rust
  HTTP/WebSocket/process gateway and Rust runtime domains, while the browser
  remains HTML/JavaScript/CSS and ball-number classification remains excluded.
  The Linux CPU run exceeded the configured 100 ms freshness limit at 1280
  input and the physical camera did not produce a first frame, so target
  localization accuracy and freshness remain unqualified.
- Numbered-ball classification is explicitly outside this migration because
  the feature is currently nonfunctional. Rust public-state contracts may carry
  null number fields, but no classifier will be ported or recommended under this
  plan.
