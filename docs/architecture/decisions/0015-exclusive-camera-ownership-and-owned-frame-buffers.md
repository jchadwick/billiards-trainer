# ADR-0015: Exclusive camera ownership and owned frame buffers

- Status: proposed
- Date: 2026-10-19
- Deciders: Project maintainers
- Supersedes: None
- Superseded by: None

## Context

The current video process, frame queues and shared-memory reader separate
acquisition from perception. The reviewed shared-memory header/copy protocol
does not prove coherent pixels-plus-metadata snapshots under concurrent
publication. Rust acquisition and perception can initially share frame ownership
inside one process, avoiding that protocol on the new critical path.

Camera ownership also matters during provider changes and shadow comparisons:
two engines opening the same device do not provide equivalent inputs.

## Decision

Give the selected perception provider exclusive camera ownership. In Rust mode,
Rust acquires and processes frames through bounded owned buffers within the
same executable. Buffer lifetime and immutable observation identity must remain
valid until consumers finish; reuse cannot overwrite a frame still being read.

Use sensor-driven acquisition for physical cameras and monotonic presentation
deadlines for replay. Record requested and negotiated modes, exposure/timing
capabilities and source session. Distinguish publication time from exposure.
Latest-work overload policies may drop frames but must count and expose gaps.

Provider handover stops and confirms release of the old acquisition owner before
opening the new owner. A failed handover leaves explicit degraded health, not
simultaneous owners or silent fallback. Camera restart creates a new source
session and invalidates incompatible retained spatial state.

The capture adapter contract is a single worker that owns the opened source for
its full read/release lifecycle. An adapter must provide an interrupt handle
that can unblock a pending read. Shutdown requests interruption and waits for
the worker to return from read and release the source before a replacement
owner may open the device. If a configured stop wait expires, the old worker
and source remain owned and restart remains prohibited until release is
confirmed. A new worker receives a new source-session identity. This is an
adapter contract, not a selection of a camera library or stop-timeout default.

Reuse a portable capture adapter first if it meets requirements. Native capture
adapters are replaceable behind the input interface when measured age, mode
support or timestamps justify them.

## Consequences

### Positive

- Removes cross-process raw-frame synchronization from the new primary path.
- Makes ownership and source identity explicit and testable.
- Prevents misleading live comparisons using different camera acquisitions.
- Supports safe buffer reuse without assuming zero-copy everywhere.

### Negative

- A perception-process restart also interrupts acquisition.
- Capture blocking behavior must be isolated from other engine workers.
- Platform camera quirks and native library lifetimes still require testing.
- Python capture-based tools need reference-image/control adapters.
- The frame-size setting bounds each application-owned RGB8 vector by its
  length and capacity. The runtime grants one receiver per session and invokes
  its consumer synchronously with a borrowed frame, preventing API-managed
  retention across receives. A required process-shared aggregate budget now
  accounts for all budgeted application RGB8 capacities. Native driver buffers
  and temporary source-side conversions remain outside that guarantee; live
  operation still needs an adapter, explicit configuration wiring, and
  sole-process ownership.
- `CaptureRuntimeCoordinator` instances share a private process-wide gate for
  the current single-source runtime. Multi-camera support requires a
  configured, source-keyed owner registry and is not enabled by this design.

## Alternatives considered

### Preserve the old shared-memory protocol unchanged

This carries an unproven coherence guarantee into the new engine. Use old
fixtures/reference behavior where valuable, not its synchronization assumptions.

### Build a new shared-memory frame bus first

It adds memory-order/slot ownership and cross-platform work not required for
one Rust acquisition/perception owner. A future multi-process need deserves a
separate measured protocol decision.

### Open the camera independently in both shadow engines

This risks exclusive-device failures and unequal exposures/frames. Compare
recorded identical frames, or introduce one proven fanout owner if live shadow
is necessary. Concurrent inference is also not a fair speed benchmark.

## Verification

- Tests delay consumers during buffer reuse and verify pixels/source metadata.
- Deterministic fake-source tests cover interruptible open/read, release
  ordering, stop timeout, failure reporting, and restart only after prior
  release. Interrupt handles must remain safe through source teardown.
- Fake-source ownership tests cover start/stop, interrupted reads, timeout,
  release failure, panic reporting, and restart after release confirmation.
- Hardware tests must cover actual device release/reopen and provider handover
  before Rust live mode is exposed.
- `CaptureRuntimeCoordinator` tests verify starts are serialized and a timed-out
  worker remains owned until it exits. The provider/runtime must route every
  start through the process-wide source gate for this single-source product.
- The configured RGB8 byte limit is supplied to each source read for preflight
  allocation via `CaptureFrameLimit::allocate_rgb8_buffer`; it is rechecked by
  the envelope and mailbox against both vector length and capacity. This bounds
  each application-owned RGB vector; driver-native memory remains outside it.
- The synchronous interrupt operation must return promptly; `stop(timeout)`
  applies to the wait after interrupt dispatch. Dropping a timed-out coordinator
  still joins its worker and can block, so shutdown must retain it until release
  is confirmed. The process-wide gate cannot coordinate independent processes.
- Hardware runs confirm camera release before reopening and negotiated settings.
- Overload tests retain newest work, bound memory and disclose observation gaps.
- Raw/rectified calibration stills remain correct under concurrent preview.

## Revisit conditions and related records

Revisit process isolation when capture reliability or compute contention
requires it. Any new raw-frame IPC requires explicit synchronization and
lifetime guarantees. Small scene IPC remains governed by ADR-0011.


## Evidence and implementation tracking

Status is proposed; Rust migration implementation and qualification are planned,
not completed. This record governs migration stages 5.

- [Migration plan](../../../thoughts/rust-perception-migration-plan.md).
- [Architectural review and evidence limitations](../../../thoughts/architectural-review-2026-10-19.md).
- [Existing backend/video/process.py](../../../backend/video/process.py).
- [Existing backend/vision/capture.py](../../../backend/vision/capture.py).
- [Rust camera adapter spike requirements](../../../thoughts/rust-camera-adapter-spike.md).

The Rust inference library accepts an immutable RGB frame, but no camera
dependency has been selected and no physical device has been qualified. A
candidate must pass the documented physical-device spike on macOS and Linux
before this decision is accepted or the Rust provider is exposed. The synthetic
frame-engine test validates inference ownership only. See the [corrected
frame-engine smoke
report](../../../thoughts/performance-audits/frame-engine-api-smoke-20261020-r2/report.md).

At the original worker-prototype checkpoint, callers constructed the gate and
could bypass it by creating independent instances. The runtime-owned follow-up
below closes that API path. At that checkpoint, the mailbox bounded queued
items only; later follow-ups add a per-frame byte bound and a single receiver
claim. An aggregate memory budget and camera/process ownership remain required
before live capture.

**Device-independent capture-boundary prototype (2026-10-20):**
[`perception/src/capture.rs`](../../../perception/src/capture.rs) now defines an
owned RGB8 frame envelope and a thread-safe one-slot, session-scoped mailbox.
The envelope preserves source timestamps and labels separately measured host
read-completion time; the mailbox reports replacements, rejects cross-session
frames, drains on graceful close, and discards pending work on cancellation.
Thirteen focused tests cover envelope validation, bounded replacement, session
identity, close/cancel/failure behavior, timeout, and consumer wakeups. The full
Rust suite passed 118 tests with formatting and strict Clippy. This seam is not
connected to inference or a camera adapter. ADR-0015 remains proposed: no
physical camera, platform adapter, negotiated mode, release/restart, preview,
or macOS/Linux hardware behavior has been qualified. The prototype adds no
authentication; session IDs remain provenance only.

**Fake-source capture worker prototype (2026-10-20):** `CaptureWorker` now
acquires a shared runtime-owned `CaptureOwnerGate` before the unopened source
factory can open. The worker owns open/read/release/drop on one thread and
publishes to the capacity-one session mailbox. A caller-supplied stop timeout
retains the worker and gate; a timed-out caller must keep the worker and wait
again before attempting a replacement. Stop before open skips hardware startup;
an in-progress open or read is cancelled through a latched interrupt handle
that remains safe during source teardown. Confirmed release clears the gate,
including after a caught read failure or panic. Release failure, source
destructor panic, or unexpected worker panic quarantines it. Deterministic
fake-source tests cover interrupted open/read, stop during release, timeout and
late-frame rejection, completion after release, fresh session identity on
restart, read/open failures, release failures, destructor panic, and
join-on-drop. Formatting, strict Clippy, and `cargo test --locked --all-targets`
pass (71 library, 7 binary, 51 integration tests). These tests verify software
lifecycle sequencing only: no real device, physical exclusivity,
capture-buffer reuse, preview, provider handover, or macOS/Linux camera behavior
has been tested. `CaptureOwnerGate` prevents concurrent source ownership only
when all starts for a physical source share the same instance; it does not
authenticate or authorize clients. The capacity-one mailbox does not cap frame
bytes or consumer-held frames. Runtime wiring, a configured byte limit, and a
single-consumer policy remain required before live capture. No authentication
is introduced, and source/session IDs remain provenance. ADR-0015 remains
proposed pending target-camera and hardware evidence.

**Runtime-owned capture and frame-size contract (2026-10-20):** the public
`CaptureRuntimeCoordinator` now retains the active worker and uses a private
process-wide source gate; callers can start only an unopened
`CaptureSourceFactory`, not supply an independent gate. Separate coordinator
instances are verified to share that gate. It retains a worker after stop
timeout and blocks replacement starts until the source exits. The coordinator receives a
non-zero `CaptureFrameLimit`; `CaptureSource::read` receives that limit so an
adapter can call `RgbFrameEnvelope::required_rgb8_len` before allocating its
owned RGB8 buffer, and both envelope construction and mailbox publication
recheck the limit. Python configuration now defines a 24 MiB default, enough
for one 3840×2160 RGB8 frame, with a validated 1 GiB ceiling. The source can
request a checked destination allocation before copying, and tests verify
capacity as well as length limits, exact/over-limit behavior, arithmetic
overflow, defensive publication rejection, cross-coordinator exclusivity, and
timeout retention. Rust formatting, strict Clippy, and all 133 Rust tests pass;
the configuration validator suite passes 53 tests. This remains a device-
independent prototype: the setting is not yet wired to a live camera process,
the gate cannot coordinate other processes, native driver buffers and
consumer-retained frames are outside the per-frame bound, and no camera
hardware or macOS/Linux capture adapter is qualified. No authentication is
added; source/session IDs remain provenance only.

**Single-consumer callback follow-up (2026-10-20):** the coordinator now grants
exactly one non-cloneable `CaptureFrameReceiver` for each active session. Its
receive API invokes a synchronous callback with a borrowed frame, so the caller
cannot retain that frame through the capture API while requesting the next
one. A slow callback leaves acquisition independent; the capacity-one pending
slot continues to replace stale frames and records the loss. Duplicate claims,
callback ordering, timeout and terminal outcomes, final-frame draining, and
slow-consumer coalescing are covered by deterministic tests. The focused Rust
capture suite passes 33 tests and the all-target suite passes 138 tests with
formatting and strict Clippy. This bounds the capture API's consumer side, not
the source adapter's in-progress buffers or native driver memory. The aggregate
budget is implemented in the follow-up below but still requires configuration
wiring before live capture. No authentication is added, and session IDs remain
provenance only.

**Aggregate application RGB8 memory budget (2026-10-20):**
`CaptureRuntimeCoordinator::new` requires a non-zero aggregate limit, and all
coordinators in one process share the same latched budget; conflicting limits
are rejected. A source allocates pixels through `CaptureFrameBudget`, and each
`BudgetedRgb8Buffer` retains an RAII reservation for its actual `Vec` capacity
through publication, mailbox replacement, and synchronous consumer access.
Foreign-budget frames cannot enter the runtime mailbox. Budget exhaustion is a
typed, nonfatal frame drop counted in the worker report. Focused capture tests
cover exact-limit allocations, capacity accounting, release on allocation
failure/drop/replacement/cancel, cross-runtime sharing, incompatible limits,
and worker continuation after a drop. The capture suite passes 39 tests; the
all-target Rust suite passes 144 tests with formatting and strict Clippy. The
bound covers only budgeted application RGB8 capacity: native camera buffers
and adapter-side temporary conversions remain outside it. No camera adapter,
runtime configuration bridge, or hardware path exists, so this does not enable
live capture or establish a total process-memory ceiling.

The Python configuration schema now defines
`perception.capture.maximum_aggregate_rgb_frame_bytes` as 75,497,472 bytes
(72 MiB) and validates that it is at least the configured per-frame limit,
with a 1 GiB upper bound. No provider launch consumes this setting yet; the
Rust coordinator still requires its caller to pass the aggregate limit
explicitly. The schema default is not a live runtime memory guarantee.

**Python reference lifecycle repair (2026-10-20):** the retained
`CameraCapture` no longer performs a separate startup probe read. The capture
worker owns acquisition reads; if shutdown exceeds `stop_timeout`, the caller
returns while the worker retains the source, restart is rejected, and the worker
releases the device only after the native read returns. A blocking fake verifies
once-only release, no release during read, no post-stop frame, and no overlapping
restart in
[`test_camera_capture.py`](../../../backend/tests/unit/test_camera_capture.py).
These tests verify the reference Python owner only. Rust capture, provider
handover, preview, hardware release behavior, and macOS/Linux qualification
remain unimplemented and unverified.

**Requested/negotiated source metadata prototype (2026-10-20):**
`CaptureSourceMetadata` now reports optional human-readable device name, device
and backend provenance, requested and negotiated video profiles, declared
capabilities, and timestamp semantics. Construction rejects a negotiated pixel
format absent from a known capability list and rejects a mismatch between the
source-timestamp capability and whether source timestamps are actually
reported. Unknown or unsupported timestamps remain valid when only host
read-completion time is available. Before publication, the worker also checks
every frame against the session declaration: source-time frames must match the
declared clock, domain, unit, and stage; read-completion-only frames must omit
source time. A mismatch fails the session instead of publishing ambiguous
timing. The ready report is published only while the source session is open,
and it clears on stop, failure, and restart. Tests cover invalid names,
inconsistent capability claims and frame timestamps, serialization, successful
open, failed open, and fresh metadata for a restarted session. The focused
capture module passes 46 tests; the full Rust suite passes 151 tests, and strict
Clippy and formatting pass. The complete backend suite passes 1,460 tests with
2 model-dependent skips, but reports 51.82% repository coverage against the
unchanged 80% gate; the broader coverage shortfall remains open. This remains a
fake-source contract: no adapter, camera hardware, mode negotiation, source
clock, release/reopen behavior, or macOS/Linux camera has been qualified.
Device/session IDs remain provenance; this work adds no authentication or
client-identity mechanism.

Before accepting this decision for implementation, resolve its relevant prototype
and compatibility questions and record the evidence. Acceptance records commitment;
it does not claim all verification has passed. Link implementing changes and actual
verification artifacts as they exist. Follow the [decision register](../README.md)
for status changes and supersession.

### Borrowed RGB8 inference entry point (2026-10-20)

`FrameInferenceEngine::infer_rgb8` now accepts validated packed RGB8 bytes, and
`prepare_rgb8` consumes those bytes without first constructing an `RgbImage`.
When source dimensions already match model dimensions, preprocessing reads the
borrowed source directly instead of cloning the full frame. Resized frames and
the model tensor still require bounded temporary allocations; those allocations
are not covered by `CaptureFrameBudget`. `CaptureScenePipeline::receive_one`
consumes frames inside the receiver's synchronous callback and composes
inference with complete scene tracking and newest-only publication. The capture
coordinator and a concrete adapter are not yet started by the same provider
loop, so this does not establish a live hardware path or a total process memory
bound.

Validation rejects empty/overflowing dimensions and byte lengths that disagree
with RGB geometry. Pipeline tests cover supplied and host-only timestamps,
staleness before and after inference, duplicate/out-of-order rejection, model
failure versus successful-empty output, bounded publication, and tracker
candidate-state staging when a payload exceeds the configured transport limit.
The oversized-payload regression verifies rejection before the pipeline's next
batch sequence advances. Accepted results also report host read-completion to
inference-start and scene-queue latency using the same process-local monotonic
clock, separate from source timestamp and socket receipt. Preprocessing API
parity and real-model output parity use the existing tiny ONNX fixture.
`cargo fmt --check`, strict Clippy, and all 160 Rust tests pass; 22 focused
Python no-auth, credential-free workflow, and local-only deployment tests also
pass. This does not qualify a camera adapter or hardware. No authentication,
credential, or client-identity mechanism is introduced.

### Revisioned capture preflight (2026-10-20)

At this initial checkpoint, the `live_config` module validated caller-supplied
device and requested/negotiated profile expectations against one opened session
report, alongside declared timestamps and required control capabilities. It
validated freshness and per-frame/aggregate RGB8 limits and calibration
consistency, but had not yet gated the capture worker or pipeline. This status
is superseded by the integration update below. The report still lacks confirmed
profile IDs, image-space and calibration revisions, and actual control values;
compatibility cannot prove settings that the adapter does not report.

### Preflight-gated capture and scene pipeline (2026-10-20)

`CaptureRuntimeCoordinator::start_preflighted(expected, factory)` checks its
per-frame and process-shared aggregate RGB8 limits before starting. The owner
thread opens the source and validates the resulting `CaptureSessionReport`
before its first read. A bounded readiness wait returns pending, compatible,
rejected, failed, or stopped status scoped to that source session. Timing out
the wait does not release the worker or source-owner gate. Rejection fails the
mailbox and wakes waiters before the owner releases the source, and performs no
reads. Source release and quarantine remain on the owner thread.

The report is checked for device, requested and negotiated profiles, required
reported capabilities, timestamp semantics, and actual RGB8 dimensions against
the negotiated profile before publication. It cannot confirm a camera-profile
ID, image space, calibration, or exposure/gain values that the adapter does not
report; this is compatibility evidence, not an applied-settings acknowledgement.

The caller constructs `CaptureScenePipeline` before capture starts, then binds
the compatible preflight result once and before consuming frames. Binding
requires matching explicit capture-configuration revision, image space, maximum
frame age, tracker camera profile and rectification revision, and a complete
validated calibration bundle. Capture-configuration revision is distinct from
the tracker's local revision. A bound pipeline rejects wrong-session frames,
dimensions, or timestamp semantics before inference. Without applied
calibration, table positions remain unavailable. The existing unbound replay
path remains compatible.

`cargo fmt --manifest-path perception/Cargo.toml --check`, strict Clippy, and
all 182 Rust tests pass. The anonymous API/workflow/local-deployment Python
selection passes 44 tests under `--no-cov`; that scoped run does not measure
coverage. No live provider or camera adapter is wired or enabled. The Python
effective-configuration/revision/ack bridge, actual applied-setting reports,
macOS/Linux camera qualification, and physical/temporal accuracy evidence
remain open. This capture lifecycle adds no authentication or authorization;
session identifiers are provenance only.
