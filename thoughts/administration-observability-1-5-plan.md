# Implementation plan: administration observability priorities 1–5

Status: implemented and verified for Tasks 1–5. Date: 2026-11-02.

## Outcome and verification strategy

Deliver real cross-tab video status, browser-session rates/charts, browser-observed change events, actual loaded-model/build identity, and pushed independent stage health. Retain the current prototype theme and existing scene/overlay semantics. No HTTP polling, fabricated telemetry, persistent incident history, resource metrics, processing-duration dashboards, or backend event/audit service is included.

Plan-start baseline: all 54 frontend tests and TypeScript checks passed. Before each implementation task, add or identify the meaningful behavioral checks below. Work one top-level runtime module at a time, assigning a separate worker per task; integrate and verify each task before proceeding. Preserve the current uncommitted application/configuration changes.

Required final checks:

- Frontend: `npm test`, `npm run typecheck`, and `npm run build -- --outDir /tmp/billiards-observability-build` from `frontend/web`.
- Rust: diagnostics contract/transport/health tests, existing server/configuration/live/capture/scene tests, then `cargo test --all-targets`, `cargo fmt --all -- --check`, and `cargo clippy --all-targets -- -D warnings` from `perception`. Run the supported native-feature checks in the existing target environment; report feature/environment limitations explicitly.
- Validate configuration schema/default consistency, public JSON fixtures, and proposed/accepted ADR records with the repository tools.
- Browser: navigate all four tabs; disrupt video, state delivery, producer and diagnostics separately; exercise successful/failed configuration saves; inspect network traffic for no interval-triggered HTTP requests and no duplicate receivers.

## Decisions and invariants

1. One app-owned video receiver, one existing scene WebSocket, and one new diagnostics WebSocket. The diagnostics connection is separate because the scene socket currently closes without a scene and on producer restart; it must remain unchanged.
2. Keep browser and backend evidence distinct. Browser timestamps mean observed/received time; backend health describes explicitly reported work/progress, not inferred camera FPS or an authoritative event history.
3. Use monotonic clocks for elapsed time and expiry. Producer reports relative ages; gateway records receipt time and adds elapsed time. Never subtract independent machines' monotonic clocks. ISO timestamps are for display/export.
4. Preserve ADR-0100 exact-frame identity matching, bounded image ownership and stale-overlay rejection. Preserve ADR-0104/0105 provider verification and no silent inference fallback.
5. Add proposed ADRs for the app-owned video receiver and the independent producer/browser diagnostics transport before runtime implementation; accept them with verified implementation. Use the existing `docs/architecture/decisions/` convention; allocate IDs at implementation time.

## Task 1 — Frontend: share actual video delivery across tabs

### Ownership and data flow

- Extract camera configuration resolution, streaming connection, bounded frame ring and object-URL ownership from route-local `LiveView`/`useFrameAwareMjpeg` into a root-owned `VideoReceiver`. Start once after initial configuration load, continue across navigation, and dispose through application shutdown.
- Expose observable status, source generation, stream mode, connection/last-frame receipt times, latest frame identity, dimensions and sanitized failure reason. States are `unconfigured`, `connecting`, `receiving`, `stalled`, `error`, and `stopped`; source URL is not included in events/exports where it could contain credentials.
- `LiveView` consumes receiver-owned frames and computes exact matched selection against its current expected acquisition/geometry identity. Health and the header consume status only. Consumers cannot independently start streams or revoke shared URLs.
- Pause inspection freezes a consumer snapshot, never the receiver. Camera configuration replacement aborts the old generation, clears matching state and revokes its URLs; late callbacks from that generation are ignored.

### Compatibility and defaults

- Identified multipart frames use the existing parser and safe frame ring. Count delivery only after a validated frame arrives; distinguish image decode failure from network/parser failure. Preserve current exact-identity overlay tests.
- Preserve ordinary MJPEG compatibility using one root-owned native image element parked outside the visible route and attached to the live preview host when needed. Never run native image loading and a streaming fetch concurrently for the same source. Native load proves an image loaded; it does not prove ongoing frame freshness. Display `Loaded; frame freshness unavailable` in that mode and withhold unidentified overlays.
- Use the identified receiver for the existing gateway preview URL. For other URLs, attempt parser mode first; fall back once to native mode if browser policy/format prevents parsing, aborting the fetch first. No repeated automatic mode switching.
- Centralize validated frontend options: identified-video inactivity timeout 5,000 ms, reconnect initial delay 1,000 ms, cap 30,000 ms. Preserve existing frame/header/ring bounds. Reconnect after a confirmed failure/timeout is recovery, not periodic health polling. Keep policy overrides in named `VITE_*` settings with documented defaults.
- On confirmed configuration save, re-resolve receiver settings from the canonical response. If canonical reload fails, preserve the old receiver and mark its configuration synchronization unknown; an explicit later reload reconciles it. Draft edits never alter reception.

### Acceptance

- Live → Health → Events → Configuration → Live uses one receiver and retains truthful status. Initial failure, EOF, silence, restoration, URL replacement, shutdown and StrictMode remounts leak no listeners, requests or URLs.
- Native MJPEG never becomes falsely `receiving`/`stalled` based only on initial load. A receiver callback cannot claim source-camera or perception health.

## Task 2 — Frontend: measured session rates and charts

- Add a root-owned bounded observation store, independent of route mounts. Count well-formed received `state` envelopes before ordering rejection as `Received state messages/s`; count distinct accepted engine/session/frame identities separately as `Distinct observed source frames/s`. Phase-only updates and cached repeated frames do not add source-frame counts. Missing identity means unknown frame rate.
- Use a 10-second rolling window, dividing by actual observed duration during startup. Before enough elapsed time exists, show collecting rather than divide by zero. Counts come from receipts; a configured 1-second local sampler aggregates them, without network requests.
- Retain 900 one-second buckets (15 minutes) and implement the existing 1m/5m/15m selectors. Use a validated centralized frontend policy for window, bucket interval and retention; enforce finite positive bounds and cap retained records.
- Plot ball, cue and motion ages independently using shared freshness functions extracted from the current viewer. Repeated publications must not rejuvenate the same evidence. Expiry, missing evidence and producer/session replacement are handled consistently across Live, Health and Events.
- Connected silence can produce a measured zero message rate. Disconnected, suspended/unobserved and pre-start intervals are chart gaps; do not backfill synthetic samples after suspension. A sampler wakeup delayed by more than twice its interval creates a gap. Do not interpolate across process/session changes.
- Use the shadcn Chart wrapper with Recharts v3, matching the existing Radix/Tailwind theme, accessible legend/tooltips and reduced-motion settings. Avoid custom chart primitives. See the [official shadcn Chart documentation](https://ui.shadcn.com/docs/components/radix/chart).
- Export the retained, source-labeled browser samples in reviewed evidence collection; no historical backfill endpoint or persistent storage is added.

### Acceptance

Fake-clock tests cover startup denominator, rolling-window expiry, duplicate/out-of-order states, phase-only messages, missing identities, engine/session restart, disconnected zero-versus-gap behavior, background suspension and bounded storage. Browser charts show real samples and remove `.highlight-issue` only for the newly supported metrics.

## Task 3 — Frontend: browser-observed change events

- Extend the existing bounded history, search/filter/detail/export flow rather than introduce another event store. Keep the existing default 200-record limit and its validated override.
- Observe accepted state changes for phase, capture-session identity and configuration revision. First values establish baselines, not fictional start events. Ignore rejected old snapshots. Across reconnects or missing evidence, label observed changes as occurring across a gap, without claiming a complete transition history.
- Schedule local freshness deadlines from the same ball/cue/motion evaluator used by the viewer. Emit one stale/recovered record per observed crossing. Missing evidence becomes unknown, not recovered. Include evidence identity, previous/current status and reported-versus-browser-estimated age; never mutate server guidance validity.
- Consume video receiver transitions for initial failure, stalled delivery, recovery and URL replacement. Ordinary native-image mode cannot emit ongoing-frame timeout/recovery events it cannot observe.
- Add `Interface action` origin. After a successful configuration PUT, record its confirmation once, before canonical reload. Include changed field paths, restart requirement and known revision IDs, never setting values. A confirmed save with failed canonical reload is still a confirmed apply, with new revision unknown. Validation/import/format/review and rejected or unconfirmed saves are not applied events.
- Deduplicate events by source generation, evidence identity and transition, not arbitrary time suppression. Track HTTP save confirmation and WebSocket observed revision change as distinct evidence types; link them by revision when known rather than pretending they are the same source.

### Acceptance

Tests cover initial baselines, repeated messages, stale/recovery/unknown transitions without new state, reconnect gaps, camera transitions, successful PUT plus failed GET, rejected PUT, event eviction, filters, redaction and fixture exclusion. No raw configuration values enter event payloads. Keep the exact prototype examples only for genuinely unsupported authoritative/backend features.

## Task 4 — Perception: loaded model and build metadata

### Authoritative sources

- Add a shared build-info helper for gateway and producer: package name/version plus optional injected Git revision and dirty flag. Do not execute Git at runtime. Missing build provenance is null/unknown, and each binary reports its own build rather than inheriting gateway identity.
- Read active model identity from the successfully initialized `FrameInferenceEngine::model_identity()` and validated runtime session: artifact SHA-256, optional checkpoint SHA-256, classes, input dimensions, requested provider and verified execution provider/device information already available. Do not derive loaded identity from the configured filename, and do not hash large artifacts on repeated reports.
- Publish model as null while loading or after failure. Clear active producer/model metadata on process-generation replacement. Configuration application does not claim that a new model is loaded; the producer must confirm actual activation.

### Transport prerequisite

- Introduce versioned producer diagnostics v1 on a separate bounded, length-prefixed loopback TCP channel. Start the listener in the gateway independently of scene availability; start the child reporter before model initialization/capture preflight.
- Do not modify strict scene-snapshot v1 or live-v3 capture-ack schemas. The current live handshake cannot start before capture preflight, so it cannot carry startup failures safely.
- Default bind address is `127.0.0.1:0` (ephemeral); the supervisor passes the actual address and a fresh launch-generation ID to its child. Reject non-loopback binds, retired generations, non-increasing report sequences, malformed/nonfinite ages and oversized messages. An unsupervised/legacy producer with no registered reporter remains explicitly unsupported/unknown; do not open remote ingress as part of this work.
- Keep one newest report pending, separate from scene traffic. Socket operations and serialization stay off capture/inference hot paths. Start with enabled=true, maximum message bytes 65,536, producer heartbeat interval 1,000 ms and report-expiry 3,000 ms; validate expiry exceeds interval. Reuse existing bounded write/connect/backoff policies.
- Report version, generation/process ID, sequence, optional engine/session identity, relative evidence ages, runtime metadata and stage observations. Send on meaningful changes, coalescing ordinary progress updates to the configured interval. Cached immutable metadata may accompany full reports; no configuration documents or raw filesystem paths are included.
- Add the configuration fields to Rust defaults/validation and JSON schema first, then synchronize the root deployment config in its separate configuration task. Preserve all existing user overrides.

### Acceptance

Contract fixtures cover active model, loading/failure, missing build Git info, wrong generation/sequence/version and payload bounds. Test gateway and child built from different revisions, model reload, startup failure before any scene, and old identity invalidation. Existing provider qualification behavior and deployment tests must remain intact.

## Task 5 — Perception then frontend: pushed stage health

### Backend contracts and assessment

- Add `/api/v1/diagnostics/ws` with a versioned complete-snapshot envelope. Send an initial snapshot even with no scene or producer; stay connected across producer restart. Keep current scene WebSocket and HTTP liveness/readiness predicates unchanged.
- Snapshot fields: schema version, gateway instance ID, increasing diagnostics revision, assessment timestamp, producer generation/report age, gateway/producer runtime metadata and named stage assessments. Each stage carries status, stable reason code, evidence origin, optional sanitized detail, last-progress age and applicable deadline. Status vocabulary: `starting`, `healthy`, `idle`, `degraded`, `stalled`, `failed`, `stopped`, `unknown`.
- Separate stages: gateway service; camera/source lifecycle; capture/acquisition progress; model initialization; scheduled perception work; scene delivery/localization evidence. Browser preview is a separate browser assessment combined in the UI, never claimed by the backend.
- Hook actual capture source open/error/lifecycle and decoded-frame progress. `Healthy` requires positive evidence; source compatibility acknowledgements are not proof of hardware-applied settings. No evidence means unknown, not failed.
- Track actual scheduled work start/completion/failure and whether work is expected. Phase-gated idle inference is `idle`, not stalled. Apply a configurable default 30,000 ms deadline only while a job is in flight; existing capture timeouts and phase/freshness limits remain authoritative for their own evidence.
- Expired reporter heartbeat invalidates positive producer-stage assessments even if the reporter last said healthy. Explicit process exit/failure remains a recorded fact, while its last model/progress is last-known, not active. Never renew ball/cue/motion observation age merely because health telemetry arrives.
- Publish on reporter/supervisor changes, relevant scene transitions and monotonic expiry deadlines. Use latest-value watch delivery, bounded clients and send deadlines; slow clients do not accumulate an unbounded queue or block processing. Reuse the current gateway client/send limits unless diagnostics-specific overrides are necessary.
- Emit full snapshots on change and a configurable 1,000 ms gateway diagnostics heartbeat, with browser stream-expiry 3,000 ms. These are pushed liveness signals, not HTTP polling or proof that every stage is healthy. Do not republish scene messages in response to diagnostics heartbeats.

### Frontend integration

- Own one reconnecting diagnostics socket at application scope. Extend message typing/dispatch without adding another state subscription. Accept only valid versioned snapshots with newer revisions within a gateway instance; a new gateway ID resets diagnostic ordering and active metadata.
- Health consumes pushed assessments and combines them with shared browser video/state reception. Header, Events detail and evidence export consume the same stores. Runtime copy/export uses actual metadata and reports absent fields as unavailable.
- On diagnostics silence/disconnection, mark assessments last-known/stale and current health unknown; do not infer gateway death from a browser connection error. Scene and video streams continue independently.
- Connecting to an older gateway or unsupported endpoint must not cause HTTP polling. Keep explicit HTTP Health Refresh as fallback and show that pushed diagnostics are unsupported/disconnected.
- Remove `.highlight-issue` only from fields actually supplied by these changes. Keep unsupported resource metrics, historical events/images and processing-performance dashboards marked. Update all four page mappings and the consolidated gap report.

### Acceptance

Rust fake-clock tests cover no-scene startup, model/capture failure, normal idle, overdue active work, producer-report loss, expiry without more traffic, generation restart, slow subscribers, and no observation-age rejuvenation. Browser tests independently disrupt scene, diagnostics and video channels; verify stale metadata clears, old revisions cannot roll back new evidence, and no polling or extra state/video connections appear.

## Execution order and delivery gates

1. Record the two proposed ADRs and baseline verification evidence in `thoughts/`/existing ADR conventions.
2. Frontend task 1; test and integrate. Frontend task 2; test and integrate. Frontend task 3; test and integrate. These three ship independently using existing APIs.
3. Perception task 4 builds metadata plus bounded diagnostics IPC/contracts. Perception task 5 adds evidence assessment and diagnostics WebSocket. Verify both before changing frontend consumers.
4. Separate root configuration synchronization task, preserving local edits; run authority/default/schema checks.
5. Frontend task 5 integrates the new push contract and removes supported fixtures; verify older-gateway fallback.
6. Complete full checks, target-device fault-injection smoke tests, ADR validation and documentation/gap-inventory updates. Deploy gateway/producer before the consuming frontend; old state clients remain compatible throughout.

Completion means items 1–5 work from actual evidence, maintain bounded resources under navigation/reconnect/silence, and pass the listed checks. No automatic deployment, persistence, CPU/GPU metrics, authoritative backend event stream, or historical image retention is part of this plan.

## Implementation status and verification record

Tasks 1–5 are implemented. The frontend owns one identified video receiver,
browser-session chart samples, browser-observed events, and one independently
reconnecting diagnostics socket. Perception publishes verified producer
build/model metadata to the gateway; the gateway publishes versioned full
diagnostics snapshots with seven stage assessments. Configuration schema and
defaults include the diagnostics transport settings, and the root deployment
configuration was synchronized without replacing existing local edits.

Verification completed:

- Frontend: `npm test` passed (82 tests across 24 files), `npm run typecheck`
  passed, and production build passed. The build has a large Monaco chunk
  advisory.
- Perception: `cargo test --all-targets` passed (407 library tests plus binary
  and integration targets); `cargo clippy --all-targets -- -D warnings`,
  `cargo check --all-targets`, `cargo check --all-targets --features
  camera-gstreamer-source`, formatting, schema/fixture checks, and whitespace
  checks passed.
- Root configuration loaded successfully through the configuration authority
  after adding the diagnostics settings.
- Browser route smoke covered Live, Configuration, Health, and Events without
  a running backend. Configuration showed a retryable unavailable state,
  diagnostics remained unknown, and routes rendered without a crash. A live
  gateway/device smoke was unavailable; physical camera and GPU model runtime
  were not exercised.
- ADR-0106 and ADR-0107 are accepted after the implementation and checks above.

Remaining prototype/API gaps and their ease/value ranking are recorded in
[`prototype-api-gap-report.md`](prototype-api-gap-report.md). Durable history,
resource telemetry, completed-job duration distributions, frame drop counters,
backend event/audit history, and synchronized camera-to-display latency remain
outside Tasks 1–5.
