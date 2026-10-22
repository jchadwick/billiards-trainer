# Latest-state freshness contract

## User requirement

The live localization path must discard past frames and obsolete pending
states under load. A client should see the newest useful complete observation,
not a replay of the backlog. Intermediate updates may be lost; sequence gaps
and loss reasons remain measurable. Position accuracy is more important than
throughput.

## Application semantics

“UDP style” means best-effort, latest-value application semantics. It does not
require raw UDP. The browser uses WebSocket over TCP, and the local Rust/Python
bridge uses loopback TCP. Keep those interfaces while using bounded latest-wins
slots, explicit source identity, and freshness/loss signals. TCP bytes already
in the kernel or a remote client cannot be recalled, so pending work has age
limits and clients reject older source state when ordering evidence is
comparable.

Do not discard an in-flight detector call solely because a newer frame was
captured: if capture outpaces inference, that rule can suppress every result.
Instead, replace pending frames, keep a bounded detector operation in flight,
accept the newest completed usable result, and reject it when a newer result
already won or its source age exceeds the configured freshness limit.

## Implemented behavior

- Python live inference uses a single replaceable pending-frame slot. The old
  `latest_frame_queue_enabled=false` setting can no longer enable FIFO replay.
- Rust capture and inference use bounded latest-work mailboxes; Rust scene
  writing and Python scene ingress also retain at most one pending complete
  snapshot. Live Rust scene writes expire at the absolute deadline derived
  from capture read-completion plus the configured maximum frame age; expired
  snapshots are counted and never retried after reconnect.
- Integration rejects a completed Python observation before updating Core if
  its publication age has reached
  `perception.freshness.maximum_observation_age_seconds`. Repeated stale source
  identities are counted as duplicates instead of repeatedly incrementing the
  stale-drop count. While holding the serialized Core-update lock, integration
  also rejects a source frame number, explicit source timestamp, or comparable
  same-host publication tick that regresses within the same vision instance,
  producer process, and timestamp kind. It advances this ordering watermark
  only after Core accepts the new state; a producer restart begins a new epoch.
- Core state callbacks coalesce before asyncio task creation. Shutdown drops
  queued state, and ephemeral state publication makes one attempt without
  retry/backoff. Ordered controls retain their separate semantics.
- Each WebSocket connection replaces queued state with the newest snapshot,
  expires old replaceable messages before writing, recalculates same-host
  Python source age at the final send boundary, and uses a shorter configurable
  write deadline. Expiry and replacement losses are counted. Rust source age is
  advanced across local ingress and socket-queue residence only where its
  producer clock provides comparable timestamps; unknown age never renders a
  ball/cue overlay.
- HTTP MJPEG viewers read the latest shared-memory frame on each iteration and
  keep no application frame backlog. Their response-writer wait is bounded by
  `api.stream.performance.maximum_send_backpressure_seconds` (default 0.5 s); a
  slow viewer is disconnected after its in-flight chunk, so reconnecting starts
  from the current frame. The legacy raw-camera viewer follows the same rule.
- The frontend advances source age using its monotonic clock. It rejects an
  older transport sequence within an explicit producer epoch, resets that
  watermark when the producer restarts, and clears ball/cue overlays when
  state is stale or age is unavailable. Last-known locations remain labeled
  stale or freshness-unknown. The disabled WebSocket video callback now keeps
  only small frame metadata for cadence telemetry; it does not JPEG-encode,
  compress, base64-encode, or retain image payloads on the state event loop.

The shared state-age limit defaults to 0.5 seconds as an operational starting
point, not a qualified end-to-end latency target. WebSocket queue residence
defaults to 0.25 seconds and replaceable writes to 0.5 seconds. These settings
are validated and may be tuned from workload evidence. HTTP MJPEG viewer
backpressure defaults to 0.5 seconds. That server-side limit bounds a blocked
response writer; it does not recall a JPEG already accepted by TCP or claim
browser paint latency.

## Timing and transport limits

Shared-memory publication timestamps remain publication times, not camera
exposure. In the supervised physical-camera helper, age currently begins when
the parent finishes reading a complete frame record; pipeline PTS is not
verified as a comparable acquisition clock. A receiver can coalesce snapshots
that arrive while its consumer is busy, but TCP bytes already accepted by the
kernel or rendered by a remote client cannot be withdrawn. No camera-to-display
deadline or physical display latency is claimed. The WebSocket and local scene
bridge retain TCP framing for browser and cross-platform compatibility; the
best-effort guarantee is implemented with latest-wins queues, source-age
deadlines, sequence watermarks, and client-side stale-state hiding rather than
reliable delivery of every update.

## Verification and qualification boundary

Deterministic tests cover one-slot frame replacement, stale-before-Core
rejection, duplicate stale observations, state-event flood coalescing and
shutdown, one-shot state publication, WebSocket queue expiry and slow clients,
final-send monotonic age refresh and expiry, Rust scene clock-domain age
validation, expired Rust reconnect retries, frontend stale display, source
ordering, and producer restart epochs.

Target-camera freshness and localization accuracy remain unqualified until the
camera is connected and representative motion/occlusion recordings are
available. An earlier repeated-still sweep suggested a capacity balance near 3.5 FPS. A
subsequent 60-second unique-content replay found no pending-frame replacement
at 3.0 or 3.333 FPS, one replacement at 3.75 FPS, and increasing drops above
that. The highest no-replacement rate is only a single-run software replay
result: it uses synthetic post-decode timestamps, has roughly 324 ms p95
source-to-inference completion, and includes no temporal accuracy labels. It
is neither a physical-camera setting nor an accuracy recommendation. See the
[software sampling comparison](performance-audits/software-sampling-rate-20261021/sampling-rate-comparison.md).
