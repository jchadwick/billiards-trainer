# ADR-0052: Prefer the latest state throughout the perception pipeline

- Status: accepted
- Date: 2026-10-21
- Deciders: Project maintainers
- Supersedes: None
- Superseded by: None

## Context

Ball localization is useful only when it represents a recent observation.
Processing every captured frame in order can make a correct detector publish a
sequence of increasingly old positions. Backlogs can form at capture, inference,
Core event handling, process transport, and client delivery. A bounded queue by
itself does not prove freshness: an item can already be in flight, and a TCP
socket can retain bytes after the application queue has moved on.

The application uses a browser-facing WebSocket and a same-machine Rust/Python
scene boundary. Standard browser WebSockets run over TCP; browsers cannot consume
ordinary UDP sockets. Replacing those protocols with raw UDP would add a
separate browser transport and platform-specific lifecycle without removing the
need for identity, completeness, source-age, reconnect, and loss semantics.

## Decision

Treat live perception snapshots and other replaceable telemetry as **best-effort
latest values**, not as an event log. At each asynchronous boundary, retain at
most bounded work and replace pending obsolete work with the newest complete
value. Do not retry an old state publication after a newer state supersedes it.
Never make state production wait for a viewer or audit exporter.

Apply the contract as follows:

1. Capture and inference input queues use latest-wins admission. When the
   detector is busy, discard older *pending* frames and process the newest
   available frame next. A setting must not silently switch live localization
   back to FIFO behavior.
2. An in-flight detector call may finish and publish its result if it remains
   the newest completed observation. Do not reject every result merely because
   a newer frame was captured during inference; that would starve output when
   capture is faster than inference. Record source age and dropped-frame gaps,
   and reject a result when a newer result or a configured freshness deadline
   makes it obsolete.
3. Core-to-gateway state notifications and each consumer's pending scene slot
   are bounded and latest-wins. State publication is not retried with
   exponential backoff. The Core admission lock also compares source frame,
   source timestamp, and same-host monotonic publication ticks within a
   producer identity, so a still-fresh but out-of-order result cannot roll
   authoritative ball state backward. A producer restart starts a new ordering
   epoch. Command acknowledgements and configuration/control messages retain
   their existing ordered semantics.
4. The Rust/Python scene stream and per-client WebSocket retain bounded
   replaceable queues, sequence/gap evidence, send deadlines, and slow-client
   isolation. Pending state is coalesced to its newest snapshot. Receivers and
   clients reject older identities/sequences where they have a comparable
   watermark and use explicit source-age/freshness metadata; they must never
   interpret receipt time as observation time.
5. Report each kind of loss separately: capture replacement, inference
   supersession, stale-result rejection, transport sequence gaps, queue
   replacement, stale-before-send expiry, and slow-consumer disconnect. Loss is
   expected under overload and must remain visible in diagnostics.

Keep TCP/WebSocket as the initial transport for cross-platform and browser
compatibility, but do not promise delivery of every perceptual update. TCP
preserves byte ordering for an open connection and cannot retract a message
already accepted by its kernel buffers. Bound that exposure with small
application queues, a configured send deadline, explicit source identity and
age, and a client policy that does not replace a newer observation with an
older one. A future transport change requires a matched end-to-end benchmark
with identical complete snapshots and freshness semantics; raw UDP is an
experiment, not a requirement implied by this decision.

The public location state remains primary. Trajectory, video, identity,
diagnostic, and audit work remain independently bounded consumers and may skip
updates. Preserve prediction lifecycle protections, complete-snapshot
recovery, source-time semantics, and anonymous operation under ADR-0010,
ADR-0002, ADR-0003, ADR-0028, and ADR-0048.

## Consequences

### Positive

- Overload raises visible sequence gaps instead of unbounded state age.
- A slow consumer cannot make the detector replay historical frames.
- State production no longer depends on reliable retry of obsolete snapshots.
- Existing browser and local-process protocols remain usable while carrying
  best-effort application semantics.

### Negative

- Consumers can miss intermediate states and must use complete snapshots or
  explicitly recover from gaps.
- Source-age guarantees cannot be stronger than the timestamp and clock-domain
  evidence available at each boundary.
- TCP bytes already buffered by the operating system cannot be purged. A
  configured deadline can disconnect a slow client but cannot guarantee what a
  remote display has already rendered.
- Freshness thresholds and capture sampling rates require workload and accuracy
  evidence; they must be configured and qualified rather than inferred from
  queue capacities.

## Alternatives considered

### Process every frame and enlarge queues

Rejected because additional buffering increases displayed-state age and memory
under overload. It optimizes completeness of obsolete work over present-state
localization.

### Drop every in-flight result as soon as a newer frame is captured

Rejected because a detector slower than capture could have every result
discarded and never publish. Latest-wins applies to pending work; result
freshness and newer completed observations govern in-flight acceptance.

### Replace all transports with raw UDP

Rejected as the default because standard browsers cannot receive raw UDP, and
packet loss alone does not define source identity, complete-state recovery,
session restart, or freshness. UDP-like loss semantics can be implemented above
the existing transport and compared against alternatives with measured costs.

## Verification

- Queue flood tests prove pending work remains bounded and the newest complete
  snapshot is selected.
- Slow detector tests prove old pending frames are replaced, in-flight results
  do not starve publication, and results superseded by a newer completed state
  are not published.
- Slow Core/event-loop and socket tests prove pending state publications remain
  bounded, slow consumers do not delay production, and control ordering remains
  intact.
- Receiver/client tests cover sequence gaps, duplicate/out-of-order values,
  restart epochs, explicit freshness, and slow-client disconnect behavior.
- Core-boundary tests cover source frame or timestamp regression after a newer
  observation, concurrent update serialization, and producer identity changes.
- Sustained paced runs report age, gaps, drops, memory, and accuracy alongside
  throughput; a location-accuracy regression blocks any production
  recommendation.

## Revisit conditions and related records

Compare an alternative transport only after the same payload and freshness
contract is measured end to end, including serialization, queue age, copies,
reconnect, and client receipt. Change the state/control distinction only with a
versioned API decision. ADR-0003 remains the control-message and slow-client
isolation record; this ADR tightens its replaceable perception-stream policy.

## Evidence and implementation tracking

The Python live inference path now has one replaceable pending-frame slot; the
legacy FIFO setting is ignored. Stale completed observations are rejected
before Core using the shared, validated
`perception.freshness.maximum_observation_age_seconds` limit, with age
recomputed from the same-host monotonic shared-memory publication tick at Core
admission. Inside the serialized Core admission boundary, a source-ordering
watermark rejects decreasing source frame numbers, explicit source timestamps,
or comparable publication ticks for the same vision instance, producer
process, and timestamp kind. A producer restart resets the comparison epoch,
and the watermark advances only after Core accepts the snapshot. Core state events
are coalesced before asyncio task creation, shutdown discards pending state,
and state broadcasts do not use retry/backoff. The per-client WebSocket queue
coalesces replaceable messages, expires queue-old state, recalculates Python
source age immediately before the socket write, expires states at the shared
age limit, and disconnects a client whose replaceable write exceeds its
shorter configured deadline. Rust live scene snapshots carry an absolute
deadline anchored at source read completion and expire before reconnect or
write instead of being retried stale. Rust source-to-publication age is exposed
only when both monotonic timestamps declare the same clock domain; Python adds
local ingress delay before broadcast and outbound-queue delay before send. The
frontend advances age with its monotonic clock, drops out-of-order state when
explicit producer epoch/sequence evidence is comparable, and clears ball/cue
overlays when state is stale or its age is unavailable. Rust capture and scene
queues remain bounded latest-wins paths; the Python scene ingress has a
one-snapshot slot. The disabled WebSocket video callback records only bounded
frame metadata for cadence telemetry; it does not JPEG-encode, compress,
base64-encode, or retain image payloads on the state event loop. Received and
actually sent frame counts remain distinct. HTTP MJPEG viewers read the latest
shared-memory frame on each iteration without an application FIFO. A response
writer that stays blocked past the validated
`api.stream.performance.maximum_send_backpressure_seconds` limit (default 0.5
seconds) is disconnected after its current chunk; reconnecting starts from the
current frame. The stop endpoint removes active viewer IDs, and both MJPEG
generators terminate and release their reader when their ID is removed.

Verification covers bounded queue replacement, stale-before-Core rejection,
state-event floods and shutdown, no-retry state publication, WebSocket final-
send age refresh/expiry and slow-client isolation, Rust expired-reconnect
retries, comparable-clock validation, producer restarts and ordering, and
client display expiry. Stream-route tests cover latest shared-frame delivery,
stop-driven generator cleanup, and configured viewer backpressure disconnects.
`test_websocket_frame_best_effort.py` verifies that the disabled video callback
never invokes JPEG encoding and retains no image payload. These checks establish
bounded best-effort application semantics, not reliable delivery or a hard
camera-to-display deadline: one in-flight JPEG may already be accepted into
the TCP stack when a slow viewer is disconnected. The
configured 0.5-second state age is an operational starting point, not a
qualified latency target. The Rust gateway uses this validated setting as its
default stale threshold when `--stale-after-ms` is omitted; the CLI value is an
explicit development or deployment override. This keeps gateway publication
freshness aligned with the browser and live pipeline. For the supervised
physical-camera helper, freshness age starts when the parent finishes reading a
complete frame record; pipeline PTS
is not verified as a comparable acquisition clock and is discarded. TCP bytes
already accepted by a kernel or remote client cannot be recalled. Physical-
camera freshness, camera-to-display latency, and location accuracy remain to
be qualified on the target. See the [latest-state implementation
note](../../../thoughts/latest-state-freshness-contract.md), the [camera-rate
sweep](../../../thoughts/performance-audits/camera-rate-sweep-20261021/sampling-rate-sweep.md),
the [unique-video sampling comparison](../../../thoughts/performance-audits/software-sampling-rate-20261021/sampling-rate-comparison.md),
and the [performance findings](../../../thoughts/performance-audits/findings.md).
