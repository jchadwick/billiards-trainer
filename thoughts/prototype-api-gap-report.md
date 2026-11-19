# Prototype versus current API: prioritized gap report

Reference: `/Users/jchadwick/Downloads/index.html`. Scope: administration
shell, Live Diagnostics, Configuration, System Health, and Events & Debugging.
This report describes the current Rust gateway and frontend, including the
approved diagnostics additions. Estimates are planning judgments: effort runs
from XS (small) to XL (large); value reflects operator impact and prototype
coverage. Rows are ordered from easier delivery toward larger projects, with
value used to order comparable effort.

## Implemented behavior

- The web app now uses a Vite/React/TypeScript shell with a reusable Tailwind,
  Radix, and shadcn component system and the prototype's graphite visual
  direction. Live Diagnostics, Configuration, System Health, and Events are
  route-based pages.
- One application-owned video receiver is shared across route changes. It
  preserves identified-frame overlay rules and reports native MJPEG as loaded
  without claiming ongoing freshness.
- Browser message rates, distinct accepted source-frame rates, and ball/cue/
  motion evidence-age charts use locally sampled WebSocket evidence and bounded
  browser-session memory. No telemetry HTTP polling is used.
- Events & Debugging retains bounded browser-observed connection, freshness,
  state, video, and confirmed configuration-action records. Reviewed evidence
  can include the current state, explicit health/config reads, and the current
  pushed diagnostics snapshot. It remains browser evidence, not a backend
  event timeline or audit log.
- The Rust gateway and supervised producer publish verified build/model
  metadata on a bounded, versioned diagnostics transport. The gateway exposes
  complete snapshots at `/api/v1/diagnostics/ws`, including startup snapshots
  without a scene and seven backend stages. It distinguishes fresh active data
  from inactive last-known data. The existing scene WebSocket and HTTP
  liveness/readiness semantics remain separate.
- Configuration uses the existing get/save/schema APIs and Monaco backed by
  the Rust-owned JSON schema. Diagnostics transport settings were added to
  that schema/defaults and synchronized into the root `config.json` while
  preserving its existing local edits.

## Remaining API/application work, ordered by implementation ease

| Rank | Gap | Effort | Value | Current limitation and likely implementation |
| --- | --- | --- | --- | --- |
| 1 | Per-ball observation age on the live push contract | S | High | `GET /api/v1/game/state` has `balls[].last_update`, but the current UI does not poll it and the state WebSocket does not carry an equivalent per-track age. Add the producer-reported age/evidence timestamp to the versioned public-state push contract, with fixtures and expiry semantics. |
| 2 | WebSocket close code/reason in browser evidence | S | Medium | The state client records disconnected/error transitions but does not expose the browser `CloseEvent` code/reason to Events. Preserve sanitized close metadata in the transport callback and test abnormal close/reconnect handling. This only improves browser-observed evidence; it does not create a backend event. |
| 3 | Completed perception processing durations | M | High | Diagnostics reports actual in-flight job kind/start age/deadline and completion/failure ages and counts, but not elapsed duration for completed jobs, percentiles, or duration trends. Capture monotonic start/completion pairs per scheduled job and publish bounded aggregates with an explicit window. |
| 4 | Frame drop/replace/skip counters | M | High | Scheduler expected/completed/failed job counters are present, but these are not frame delivery counters. Instrument each relevant capture/producer/state boundary and define whether a count means dropped, replaced, coalesced, or skipped; expose cumulative totals and reset/generation identity. A sequence gap alone is not a drop count. |
| 5 | CPU and memory metrics; GPU metrics by runtime | M–L | Medium | No resource telemetry or collection window exists. Add platform-aware sampling with measurement source, units, window, and unavailable states; GPU support depends on provider/platform and should be a separate capability. Never treat missing metrics as zero. |
| 6 | Authoritative backend event/audit stream | L | High | There is no backend event feed or durable config-change audit. Define event IDs, occurrence time, severity, component, payload redaction, ordering/cursor, retention, and confirmed config authorship. Keep browser observations and server events explicitly distinct. |
| 7 | Cross-session diagnostics history and paging | L | High | The diagnostics WebSocket is latest-value; frontend event/sample history is bounded in memory and lost on reload. Add persistent storage, retention policy, paging/cursors, and lifecycle/session boundaries before presenting historical trends or older events. |
| 8 | Synchronized camera-to-display latency | L | Medium | Current observation ages and browser receipt times do not provide a synchronized capture-to-render measurement. Add correlated frame identities and trustworthy timestamps at capture, gateway, and browser display, then define clock error and latency aggregation. |
| 9 | Historical state and matched camera images per event | XL | Medium | The video receiver is a bounded live buffer, not an archive, and events do not retain historical state/frame pairs. This requires explicit image/state retention, storage sizing, privacy controls, expiry, and lookup by event/frame identity. |

The order is a practical starting point, not a dependency graph. Items 1–4
improve high-value live diagnosis with bounded contract/instrumentation changes.
Items 6–9 require broader persistence, security, or clock-correlation design.

## Prototype fields that remain intentionally unsupported

- CPU, memory, GPU, and historical metric windows remain marked unavailable.
- Completed processing-duration charts, frame drop counts, per-ball live ages,
  and synchronized end-to-end latency are not synthesized from nearby values.
- Backend logs/events, persistent event history, durable audit authorship,
  historical state, and stored camera images remain unavailable.
- Exact illustrative timestamps, camera identifiers/FPS, prototype setting
  values, and other fixture-only content remain excluded from live data and
  evidence exports. Unsupported page data retain `.highlight-issue` so they
  can be inventoried.
- Coordinates are presented without physical measurement units, following the
  user's instruction. No API field was added solely because it appeared in the
  prototype; the only new API data are the approved current-configuration
  schema/defaults and supervised diagnostics contracts.

## Verification and limitations

- Frontend: `npm test` passed (82 tests across 24 files), `npm run typecheck`
  passed, and the production build passed. The build reports a large Monaco
  chunk advisory.
- Perception: `cargo test --all-targets` passed (407 library tests plus binary
  and integration targets); `cargo clippy --all-targets -- -D warnings`,
  `cargo check --all-targets`, the GStreamer feature compile check, and
  formatting checks passed.
- Configuration/schema and contract fixtures were validated, and the root
  configuration was loaded through the configuration authority after the
  additive diagnostics sync.
- Browser route smoke covered Live, Configuration, Health, and Events with the
  development frontend and no backend. The unavailable-API paths rendered
  without crashing; diagnostics stayed unknown and Events retained browser
  connection records. No live gateway snapshot, physical camera, or GPU model
  runtime was available for an end-to-end device smoke test.

Page-level mappings:

- `thoughts/frontend-prototype-mapping.md`
- `thoughts/configuration-prototype-mapping.md`
- `thoughts/system-health-prototype-mapping.md`
- `thoughts/events-prototype-mapping.md`
