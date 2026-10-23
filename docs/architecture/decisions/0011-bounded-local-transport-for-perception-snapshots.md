# ADR-0011: Bounded local transport for perception snapshots

- Status: superseded
- Date: 2026-10-19
- Deciders: Project maintainers
- Supersedes: None
- Superseded by: ADR-0025

> Historical proposal: the per-launch bearer-token decision below was rejected
> by ADR-0025 at the user's direction. The bounded loopback framing, one-slot
> delivery, deadlines, reconnect, schema, and freshness requirements remain in
> force. Authentication is not part of the application or local stream.

## Context

Rust perception and the retained Python gateway need to exchange small scene
snapshots and control traffic on the same machine. The previous audit measured
small JSON/socket costs relative to the detector call. Cross-platform operation
and inspectable testing matter more initially than speculative serialization
savings. Raw 4K images have very different size/lifetime requirements.

## Decision

Start with versioned, length-framed JSON over loopback TCP for complete scene
snapshots. Keep control acknowledgements independent from the replaceable
scene stream and preview/reference image data separate from both.

The initial frame prefix is an unsigned 32-bit big-endian payload length; the
configured maximum counts payload bytes and excludes that prefix. Reject zero
length and over-limit frames before allocating the body. The framing codec is
independent of any async runtime; this prototype does not select Tokio or commit
to socket lifecycle ownership.

Configure maximum frame/message sizes, accepted schema versions, bounded queue
capacities, write/connect deadlines and reconnection behavior. Producers retain
only the newest pending replaceable snapshot. A stalled receiver cannot back up
camera acquisition; coalesce or disconnect within policy and count losses.
Ingress rejects stale, out-of-order or invalid-session data. Reconnect delivers
the latest complete snapshot rather than historical backlog.

Kernel socket buffers can contain stale bytes too: application queue bounds
are not a proof of freshness. Test socket deadlines, recovery and age rejection
under a receiver that stops reading.

Bind internally to loopback and use per-launch peer authentication/identity so
the bridge does not become an alternative unauthenticated control API. Keep
credentials outside trace exports. Do not transport raw 4K frames as JSON.
Frame ownership stays within Rust under ADR-0015.

For the scene stream, each TCP connection begins with a length-framed JSON
`hello` containing protocol version 1 and a cryptographically random,
launch-scoped bearer token. The Python supervisor generates the token and
passes it to that Rust child through its environment, never as a command-line
argument. The receiver compares it in constant time and returns a framed
`ready` response containing its last accepted transport sequence before
accepting snapshots. The Rust sender serializes on the inference thread, then
submits through one pending slot to a dedicated network worker; one snapshot
may also be in flight. New submissions replace stale pending work. Reconnect
retries use configurable capped exponential backoff. After reconnect, the
sender transmits the latest snapshot only when its sequence is newer than the
receiver's handshake watermark. The watermark is a reconnect deduplication
aid, not a per-message acknowledgement: a successful local socket write does
not prove receiver acceptance, so sender-write and receiver-accept counters
remain separate. A launched producer may reconnect with the same token and
sequence identity; a process restart creates a new receiver/token pair. This
authenticates possession of the current launch credential on loopback; TCP
does not encrypt the payload or protect against a compromised same-user
process.

## Consequences

### Positive

- Same protocol works on macOS/Linux and remains easy to inspect from Python.
- Reuses JSON fixtures and existing receiver/reporting concepts.
- Bounded latest-state semantics align with the existing client-delivery policy.
- Separates control readiness from slow scene or preview consumption.

### Negative

- Serialization and copies remain and must be included in measured latency.
- Stream framing, partial reads, reconnects and peer identity require tests.
- A single scene feed needs a supervised consumer and explicit failure status.
- Loopback authentication is additional lifecycle state to manage correctly.

## Alternatives considered

### New raw-frame shared memory between Rust and Python

The new critical path does not need cross-process images. The existing protocol
also lacks a proven coherent snapshot guarantee. Avoid inheriting that risk;
a future bulk-frame bridge requires a separate ownership/synchronization ADR.

### Binary RPC or custom binary scene format immediately

Possible future alternatives, but small-payload measurements do not support
their cost now. They also add schema/tooling work during model integration.

### Unix-only sockets or a distributed broker

Unix sockets are viable on the target systems, but TCP directly matches the
existing harness and broader portability. A broker adds deployment and
delivery semantics unnecessary for one local table owner.

## Verification

- Framing tests cover partial/truncated/oversized messages and zero-length frames;
  the shared contract tests reject unsupported schema versions.
- The Python ingress binds to loopback, accepts one authenticated Rust
  producer, enforces a caller-supplied maximum frame length, and measures
  receipt/order/gap data with the receiver's local monotonic clock.
- The Rust replay sender uses explicit connect/write deadlines and a bounded
  asynchronous latest-only writer; inference does not wait for socket writes.
- Slow-handshake and non-reading-peer tests exercise queue replacement,
  producer-side nonblocking submission, write deadlines, and bounded shutdown.
- Reconnect tests use the receiver's last accepted sequence to avoid replaying
  an already accepted snapshot and send a newer complete snapshot.
- Reconnect tests deliver a new complete snapshot and reject old session data.
- A blocked scene channel does not block stop/reconfigure acknowledgement.
- Unauthorized local peers cannot inject control or authoritative observations.
- Matched transport measurements include serialization, copies and queue age.

## Revisit conditions and related records

Revisit transport only when full-path measurements identify it as a meaningful
budget consumer. Preserve ADR-0003 slow-client isolation and ADR-0010 freshness
semantics in any replacement. This proposal does not change the public
WebSocket endpoint or require a new browser protocol.


## Evidence and implementation tracking

Status is proposed; the recorded-replay transport is an experiment and does not
accept the production transport decision. This record governs migration stages
1 and 4.

- [Migration plan](../../../thoughts/rust-perception-migration-plan.md).
- [Architectural review and evidence limitations](../../../thoughts/architectural-review-2026-10-19.md).
- [Existing backend/api/websocket/handler.py](../../../backend/api/websocket/handler.py).
- [Existing backend/video/ipc/shared_memory.py](../../../backend/video/ipc/shared_memory.py).
- [Length-prefixed framing codec](../../../perception/src/framing.rs) and [fragmentation, truncation, and bounds tests](../../../perception/tests/framing.rs).

The Rust prototype validates four-byte big-endian framing, short reads and
writes, consecutive messages, truncated headers/bodies, zero-length rejection,
and oversize rejection before body allocation. Its recorded-image executable
can synchronously send length-framed JSON to a loopback receiver with explicit
message, connect, write, and session-time limits. The initial Python receiver
and its temporary Rust replacement belonged to a separate recorded
detection-batch replay experiment. ADR-0085 retired that synchronous
sender/receiver after the equivalence evaluator was removed; the historical
ten-image result delivered 10 batches and 73 detections, all matched against
the Python PT reference with 0.0325 px maximum center difference. The latest
recorded inter-arrival p50/p95/p99 was 92.128/94.518/94.518 ms; it is not a
freshness result, and producer/receiver monotonic epochs were not proven
comparable. The current live scene sender is an independent bounded
latest-only transport documented by ADR-0028 and ADR-0031.

That batch experiment did not meet the live-transport acceptance criteria: the
send path was synchronous, a slow receiver could stall replay, and the tool did
not implement live freshness. It was removed rather than promoted. Current
scene delivery and live provider boundaries are documented and tested under
ADR-0028 and ADR-0031. See the [historical batch replay report](../../../thoughts/performance-audits/detection-batch-tcp-smoke-20261020-r2/report.md)
for the retired experiment's results.

### Retired authenticated prototype evidence (historical, removed by ADR-0025)

> This section records an intermediate prototype only. Its token handshake and
> authenticated replay path were removed when ADR-0025 rejected authentication.
> None of the implementation or measurements below describe the current code.

The intermediate backend prototype contained an opt-in `SceneSnapshotIngress`
for the scene-snapshot v1 boundary. It bound to loopback and authenticated a
per-launch token before accepting bounded framed payloads. Eighteen focused
tests covered loopback delivery, authentication, limits, reordering,
producer-session pinning, reconnection within one launch, consumer lag, and
configuration validation, truncated/zero-length frames, and idle-producer
timeout accounting. Scoped Ruff, Black, isort, and mypy checks passed at that
time.

The intermediate Rust recorded-image executable also supported authenticated
scene-snapshot TCP output during paced replay. A local end-to-end smoke used
the CPU ONNX model and that prototype ingress. It is historical transport
interoperability evidence only; it does not describe current authentication
behavior or qualify real-time performance. The old run report and receipts
remain available in [the replay smoke](../../../thoughts/performance-audits/scene-tcp-replay-20261020T163659Z-853bb3/report.md).

The prototype sender used a dedicated writer thread, one pending slot,
configured connect/write deadlines, capped reconnect backoff, and receiver
sequence watermarks. Its tests and asynchronous smoke are retained as
historical evidence for transport behavior, not for authentication. The replay
processed 2.29 fps and missed all ten synthetic 30 fps deadlines. Mean decode
and preprocessing were 364 ms and 358 ms, while mean inference was 40.8 ms; this
short run points to decode and preprocessing
as larger measured costs than scene serialization/enqueue (0.45 ms), but is
not a controlled transport comparison or recommendation to optimize any
stage. Cross-process one-way latency remains unavailable. See the
[bounded asynchronous replay smoke](../../../thoughts/performance-audits/scene-async-tcp-replay-20261020T164934Z-df11e6/report.md).

The ingress is not started by FastAPI, supervised with a Rust provider
lifecycle, mapped into the existing API/Core adapters, or exercised through
public HTTP/WebSocket delivery. The replay also still requires synchronous
detection-batch artifact output, so it is not representative of the future
live path. Keep ADR-0011 proposed; this replay feature does not qualify live
transport. The implementation and tests are in
[scene ingress](../../../backend/perception/scene_ingress.py) and
[scene-ingress tests](../../../backend/tests/unit/test_scene_snapshot_ingress.py),
[Rust scene sender](../../../perception/src/inference/runtime.rs), and
[historical synchronous replay report](../../../thoughts/performance-audits/scene-tcp-replay-20261020T163659Z-853bb3/report.md) and [latest bounded asynchronous replay report](../../../thoughts/performance-audits/scene-async-tcp-replay-20261020T164934Z-df11e6/report.md).

The Rust verification after this slice passed `cargo fmt --all -- --check`,
`cargo clippy --locked --all-targets -- -D warnings`, and all 97 unit/contract
tests. The 1,347-test backend suite passes its tests but exits on the existing
49.80% coverage result against the unchanged 80% gate.

The complete backend suite passed 1,347 tests with one model-dependent replay
skipped. Coverage remains 49.80% against the configured 80% gate; no threshold
was changed. The focused ingress suite also passes without collecting the
repository-wide coverage report.

The recorded-replay experiment is allowed under this proposal and is not
acceptance for the live transport. Before accepting this decision for live or
production use, resolve the remaining prototype and compatibility questions
and record the evidence. Acceptance records commitment; it does not claim all
verification has passed. Link implementing changes and actual verification
artifacts as they exist. Follow the [decision register](../README.md) for status
changes and supersession.
