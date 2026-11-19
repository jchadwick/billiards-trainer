# Billiards Trainer administration workspace

React, TypeScript and Vite serve Practice at `/practice`, Live Diagnostics at `/live`,
Configuration at `/configuration`, System Health at `/health`, and Events & Debugging
at `/events`. Tailwind CSS and shadcn-style Radix primitives share semantic
color tokens in `src/styles.css`; the graphite palette follows the prototype.
Operational tabs share the same shell, state WebSocket, and app-owned video receiver.

The standalone `/practice` route is a simulated table-space sandbox. Players can
place balls with the pointer or physical-coordinate inputs, explicitly confirm a
cue ball, and ask for a direct-pot suggestion or select a target and pocket. Its
pure geometry uses millimeter table coordinates and configurable dimensions / ball
radius; offline defaults mirror the physical localization defaults. Suggestions
check both travel segments against ball radii and rank by cut angle, then total
travel, then stable target/pocket identity. `suggestDirectPots` accepts alternate
ranking priorities and a maximum cut angle. This estimates a direct pot only; it
does not model spin, banks, combinations, or full post-contact physics. The route
does not consume live camera or producer state, so simulated balls are never
presented as detections. Live coaching reuses the same pure geometry with the
producer's current trusted table-plane layout.

## Development and verification

```sh
npm ci
npm run dev
npm test
npm run typecheck
npm run build -- --outDir /tmp/billiards-web-build
```

Without an output override, production assets are written to `../../backend/static`
for the Rust gateway. Set `VITE_API_BASE_URL` for a separately hosted API; otherwise
HTTP and WebSocket requests use the serving origin. The Vite dev server proxies
`/api` to the local gateway on port 8001 by default. Set `VITE_DEV_PROXY_TARGET`
to use a different gateway.

On first launch, Configuration lets you choose a discovered local camera or enter
a network camera's MJPEG URL. Freeze a frame of the empty table, mark the four
playing-surface corners, and confirm its dimensions to calibrate the table.
Ordinary corner calibration supplies the table-space aiming map. Optional
precision assets can improve center refinement; they are not required to start.
See [Configuration](../../docs/CONFIG.md).
The JSON editor is available under Advanced configuration.

The root shell opens one native MJPEG image directly from
`api.stream.viewer_url` and shares it across routes. The header and Health
report whether the browser loaded the image. State arrives separately over
WebSocket. `/live` always displays the newest video; compatible producer ball
centers and fresh positive cue evidence do not require frame IDs. Trusted balls
draw solid markers, while last-known producer balls draw dim dashed markers.
The browser does not retain or reconstruct a second ball layout.

Live coaching is available at `/coach` through **Practice → Live Coach**, and
inside `/live` diagnostics. See [the live coach guide](COACH.md) for the physical
table workflow and configurable alignment tolerances. It reuses the pure
`suggestDirectPots` geometry from `/practice`. Choose the cue ball, target, and
pocket, or leave target and pocket on automatic suggestion. Advice requires a
current trusted whole-table layout
and compatible producer table projection. Positions use the producer's
center-origin millimeters and are shifted to the sandbox's positive-origin
coordinates for coaching; paths are projected back through the inverse of the
same source-pixel-to-table homography. The desired green ghost-ball cue path
and orange object path are shown separately from the yellow detected shaft.
A signed cue that approaches the selected ball through its center supports
cyan current cue-ball travel and a pink initial struck-ball path. The coach
reports angular alignment and lateral shaft offset. An unknown tip direction
requires explicit player confirmation before predicting current aim. Measured
paths end at their next contact or finite-radius cushion boundary; post-contact
cue-ball roll, speed, spin, pocket capture, and rebounds are outside this model.
Stale ball centers remain visible
as possible blockers and never support an advice corridor. Missing compatible
projection disables live aiming but leaves usable image-coordinate centers
visible. A verified clean empty scan and an unscanned table have separate
messages.

Reconnect backoff is configured by `VITE_VIDEO_RECONNECT_INITIAL_MS` and
`VITE_VIDEO_RECONNECT_MAX_MS`.

The state WebSocket remains unchanged.
State updates arrive through the `state` subscription on `/api/v1/websocket/ws`;
there is no HTTP polling fallback. Camera frames arrive over one continuous MJPEG
response. Local display timers advance evidence ages even when the producer is
silent; they never request new data. WebSocket heartbeats and reconnect backoff
only maintain the pushed connection.

The app root also owns one reconnecting raw-snapshot WebSocket at
`/api/v1/diagnostics/ws`. Every message is runtime-validated against the v1
snapshot shape, including its seven named stages. The browser accepts increasing
revisions per gateway instance and starts revision ordering over when the
instance ID changes. Silence beyond the snapshot's `stale_after_ms`, an invalid
message, or a disconnected stream marks evidence stale or unknown; it does not
claim the gateway has stopped. Diagnostics do not open another state or video
connection and do not poll HTTP. Reconnect delays use
`VITE_DIAGNOSTICS_RECONNECT_INITIAL_MS` (default 1000, accepted range
100–30000) and `VITE_DIAGNOSTICS_RECONNECT_MAX_MS` (default 30000, accepted
range 1000–120000); invalid values use those defaults.

Health and the header combine backend stage assessments with the shared browser
state and video evidence. Active model/build details are shown only from a fresh
producer report. Expired report data is labeled inactive and last known. Events
detail and evidence export use the same latest diagnostics snapshot; they do not
pretend it was captured at an earlier event time. Health's liveness/readiness
checks remain initial and explicit-refresh HTTP snapshots.
Coordinates in operational diagnostics are displayed as reported, without
assuming physical units. Pausing inspection captures the displayed image and
diagnostic snapshot locally while the streams continue; a camera that disallows
browser capture reports a clear error and leaves inspection live. Optional
developer contours retain their source-frame provenance checks; production
center markers use compatible geometry and the newest video frame.

Configuration loads the existing `/api/v1/config` and `/api/v1/config/schema`
contracts once on initial load and on explicit reload/apply workflows, without
background polling. Monaco and its JSON validation workers are bundled locally. The editor
supports draft validation, difference review, applying, reload, and local import /
export. Drafts survive route navigation in this browser tab; reload discards them.
Apply uses the existing merge contract and reloads canonical values after success.
Read-only fields are excluded from update payloads. Server validation remains the
authority, and restart requirements are shown after saving.

Unsupported prototype data and actions carry `.highlight-issue` for inventory.
The mapping and exceptions are documented in `thoughts/frontend-prototype-mapping.md`
and `thoughts/configuration-prototype-mapping.md`. Prototype values do not masquerade
as observations or cause new API endpoints to be added.


System Health combines pushed state evidence with initial/explicit liveness and
readiness HTTP snapshots. It never polls. Events & Debugging retains real browser
connection transitions, accepted state changes, freshness/video transitions, and
confirmed configuration actions in memory. `VITE_EVENT_HISTORY_LIMIT` defaults to
200 and accepts values from 1 to 10000; invalid values use the default. Routine state
frames are not server logs. Search/filter/detail/selection and reviewed evidence downloads operate locally;
health and optional active-configuration reads happen only on explicit collection.
Prototype-only event rows remain separate and highlighted, excluded from downloads.
See `thoughts/system-health-prototype-mapping.md` and
`thoughts/events-prototype-mapping.md` for the supported fields and remaining gaps.

System Health rate charts count well-formed state messages received by the existing
state WebSocket and distinct accepted engine/session/frame identities separately.
They are browser-session observations, not producer or camera FPS. Ball, cue and
motion evidence ages are sampled independently from the API's reported ages; repeated
publications do not reset unchanged evidence. Disconnected, suspended and missing
identity periods remain gaps. Samples are held in memory only, with no network polling.

The browser metrics policy is validated centrally. `VITE_HEALTH_RATE_WINDOW_MS`
defaults to 10000 (accepted range 1000–60000), `VITE_HEALTH_SAMPLE_INTERVAL_MS`
defaults to 1000 (accepted range 250–5000 and capped at the rate window),
`VITE_HEALTH_SAMPLE_RETENTION_BUCKETS` defaults to 900 (accepted range 1–3600), and
`VITE_HEALTH_MAX_RATE_EVENTS` defaults to 50000 (accepted range 100–100000). Invalid
values use the documented defaults. The sampler aggregates existing socket receipts
locally and performs no requests.
