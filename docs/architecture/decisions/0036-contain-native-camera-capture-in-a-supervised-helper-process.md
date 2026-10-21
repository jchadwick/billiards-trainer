# ADR-0036: Contain native camera capture in a supervised helper process

- Status: accepted
- Date: 2026-10-21
- Deciders: Project maintainers
- Supersedes: None
- Superseded by: None

## Context

The live Rust runtime has a hardware-neutral `CaptureSourceFactory` seam and
requires cancellation of native open, read, and release work. Its capture
worker owns the source on one thread, joins that thread on shutdown, and
quarantines camera ownership if release cannot be confirmed. A native API call
that blocks indefinitely therefore cannot be detached safely or made
interruptible merely by adding a Rust cancellation flag. The current standalone
default live build intentionally selects no camera source. A feature-gated
physical helper prototype is now available for explicit qualification runs.

Feasibility reviews found that Nokhwa's native API is blocking and has
unresolved AVFoundation buffering/release and crash concerns; OpenCV's generic
read-timeout support does not cover AVFoundation or V4L2; and GStreamer's
`appsink` can bound its own queue and yield frames through a pull API, but this
does not bound upstream driver buffering. The development Mac has GStreamer
1.28.3 and its AVFoundation source plugin. Linux V4L2 runtime behavior has not
been checked on a target device. GStreamer source elements also do not provide
verified applied exposure/gain values on both platforms. These findings are
recorded in the [camera-adapter feasibility note](../../../thoughts/rust-camera-adapter-spike.md),
including the GStreamer review. The Mac's read-only inventory reports a
built-in MacBook camera and an iPhone Continuity camera; the built-in camera
was selected for a short smoke using its discovery-generated stable selector.
This is candidate evidence, not production qualification.

The governing contracts remain ADR-0015 (one camera owner and bounded owned
frames), ADR-0029 (source-reported control readbacks), ADR-0031 (Rust capture
runtime lifecycle), ADR-0033 (separate persistent live executable), and
ADR-0034 (raw camera mode distinct from emitted RGB8 geometry). No physical
Linux target is currently identified. The intended deployment is macOS and
Linux, and the user has explicitly rejected authentication anywhere.

## Decision

For camera-adapter evaluation, prototype any backend with potentially blocking
native calls behind a separately supervised helper process. The Rust live
runtime remains the sole logical capture owner: it starts one helper for the
active session, passes the exact validated capture request, validates the
helper's source metadata and frames, and owns stop, kill, reap, and restart
ordering. The helper alone opens the physical device. Parent-to-helper control
and helper-to-parent data use bounded inherited local pipes; this adds no
authentication, network listener, or credential contract.

Use GStreamer for the first helper-based qualification candidate, both with
`videotestsrc` and an explicitly selected physical source. Physical mode uses
GStreamer's read-only inventory to resolve exactly one stable selector, applies
the requested input caps, converts/scales to bounded RGB8 output, and reports
the actual negotiated geometry. Capture-settings v3 `source_defaults` is
mandatory and the helper writes no camera controls. The physical factory is
compiled only by `camera-gstreamer-source`; the live executable selects it only
with an explicit `gstreamer-physical` argument and otherwise keeps the
unavailable source. This accepts GStreamer for opt-in qualification, not as a
generally available production provider. Preserve source timestamp clock/stage
as unknown unless the source supplies a verified external timestamp; record
host read completion separately. A helper must report actual mode and
control-readback status, not echo requested values.

The helper uses a versioned, size-bounded frame/metadata envelope and a
request-bound ready record. The live executable and Python supervisor now
exercise the physical helper end to end, but public API provider selection
remains closed until interruption, release/reopen, bounded-memory/freshness,
frame-geometry, source-readback, localization-accuracy, and macOS/Linux
qualification gates pass. If measured pipe cost is material, compare it with a
bounded shared-memory design using identical payloads and freshness semantics
before changing the transport.

The synthetic and physical helpers are connected to the capture worker through
an opt-in, feature-gated `CaptureSourceFactory`. The worker supplies the source
session ID. Each helper returns bounded, request-bound readiness metadata after
validating its first sample; the parent verifies that preface before publishing
source metadata or reading frames. Each frame is decoded through the shared
bounded protocol reader and allocated from the worker's `CaptureFrameBudget`.
Synthetic mode reports physical controls as unsupported. Physical
`source_defaults` mode makes no control writes and preserves unknown or
unsupported readbacks rather than treating requested values as applied. A
preflight integration test confirms that unsupported required exposure/gain
capabilities reject a strict synthetic-source request before its first frame.

## Consequences

### Positive

- A hung or crashed native capture call can be contained by terminating and
  reaping its helper instead of leaving an unjoinable camera-owner thread.
- Camera dependencies and platform-specific plugins can stay outside the
  inference process and remain opt-in during feasibility work.
- Synthetic capture can exercise negotiated caps, RGB conversion, queue
  freshness, restart, crash, and shutdown without claiming physical-camera
  behavior.
- The same bounded helper protocol can be exercised with a selected physical
  source while keeping blocking camera operations outside the inference
  process.
- The logical capture lifecycle, request digest, scene identity, owned RGB8
  frames, and downstream inference contracts remain in the existing Rust live
  runtime.

### Negative

- A process boundary adds serialization/framing, copies, scheduling jitter,
  another executable, and cross-platform packaging work.
- Parent and helper must agree on protocol version, request identity, source
  session, timestamps, geometry, errors, and shutdown state; inconsistent
  version handling can prevent startup or produce stale data.
- Killing the helper bounds its lifetime but does not prove the driver released
  the camera. Device handover and release/reopen still require physical tests.
- GStreamer appsink queue bounds alone do not prove bounded driver or upstream
  buffering, fresh frames, applied exposure/gain, or exposure-time timestamps.
- The tested Mac camera path currently uses explicit requested mode caps and
  driver-default controls; it does not discover the best mode automatically or
  establish table geometry and ball accuracy.
- This decision can be revisited if a direct native adapter demonstrates
  prompt, reliable cancellation of open/read/release on every supported backend
  without process isolation.

## Alternatives considered

### Direct native capture in the live process

Rejected for the current prototype because the source interface requires prompt
interruption of open/read/release, while reviewed APIs use blocking native
operations and do not establish that contract. A stuck native call would hold
the capture worker and owner gate. Reconsider only with executable cancellation
and release evidence on both target platforms.

### Nokhwa as the first live adapter

Deferred. Its optional native feature compiles on the current Mac and a Linux
container, but compilation does not resolve AVFoundation callback buffering,
process-abort reports, prompt interruption, strict source readbacks, or device
release/reopen. The current dependency remains build-only and opt-in.

### OpenCV Rust as the first live adapter

Deferred. It resembles the retained Python capture interface, but introduces
native OpenCV and libclang build dependencies and its generic timeouts do not
establish bounded AVFoundation/V4L2 reads or exact applied-control readbacks.

### Python capture remains the Rust source

Rejected as the target boundary because Python would retain physical device
ownership and would not exercise the Rust-owned capture lifecycle. Python may
remain the rollback provider until Rust qualifies, but it is not the adapter
being evaluated here.

## Verification

- A feature-gated synthetic GStreamer helper test runs
  `videotestsrc ! capsfilter ! videoconvert ! appsink` with an active bounded
  pull consumer, checks negotiated RGB geometry, validates conversion and
  stride against `CaptureFrameBudget`, and confirms that slow-reader drops do
  not grow an unbounded application queue.
- Process integration tests verify exact request forwarding, protocol-version
  and size-limit rejection, monotonically unique frame identity, explicit
  unknown timestamp semantics, helper crash, blocked-read termination, reap,
  restart identity, and no frame after stop.
- Run the synthetic helper and contract tests on macOS and Linux CI with the
  required development/runtime plugin packages. Check AVFoundation and V4L2
  plugin availability without opening a physical camera.
- Compare parent/helper transport against an identical in-process synthetic
  payload for p50/p95/p99 frame age, unique frame throughput, CPU, copy cost,
  memory, and slow-consumer behavior. Do not select shared memory without this
  comparison.
- Physical qualification on the selected devices must verify exact negotiated
  mode, supported/applied/unknown controls, source timestamp meaning, backlog
  age, interruption, release/reopen, permissions, and sustained freshness.
  Synthetic results cannot satisfy these checks.
- A MacBook built-in-camera smoke requested NV12 1920x1080 at 30 fps, verified
  negotiated RGB8 output, and emitted two frames before a clean helper exit.
  The continuous supervised live path then completed the v3 acknowledgement
  and published 31 scene updates to the local Python publisher in about 3.1
  seconds, then stopped with child exit 0. The live run reopened the camera
  after the preceding helper exited. Images were not saved. The detector was
  the checked-in test fixture and no table calibration was supplied; this is
  lifecycle/wiring evidence only, not production accuracy, source age,
  exposure-time, forced-interruption, sustained-load, or Linux hardware
  evidence.
- Ubuntu 24.04 x86_64 container verification with GStreamer 1.24 passes the
  feature-enabled Rust all-target suite and builds optimized live/helper
  binaries. This checks Linux compilation and synthetic contracts only; no
  V4L2 device was available in that container.
- Keep Rust live unavailable through normal API provider selection until all
  existing capture, request-bound readiness, accuracy, freshness, and
  cross-platform qualification gates pass. The explicit feature-gated path is
  for qualification only. Run `cargo fmt --check`, strict Clippy, Rust tests,
  Python supervisor tests, and ADR validation for implementation changes.

## Implementation progress (2026-10-21)

The opt-in `camera-gstreamer-probe` feature now builds a separate synthetic
helper executable using `videotestsrc`, fixed RGB caps, `videoconvert`, and an
appsink configured for one-buffer leaky delivery. The helper copies
stride-aware pixels into `CaptureFrameBudget` storage and writes a bounded,
versioned JSON-header/RGB-payload record to stdout. It reads exact Python v1/v2
capture-settings bytes from bounded stdin, rejects unsupported backends,
derives synthetic RGB geometry, rate, and frame budgets from the validated
request, and includes the request digest, configuration revision, and
camera-profile identity in each frame header. Five macOS subprocess tests cover
request-size rejection, request/frame correlation, frame geometry,
delayed-reader freshness gaps, and force-termination/reaping; unit tests cover
padded rows, short strides, truncated planes, and output lengths. The feature
is absent from the default dependency graph and does not register a
`CaptureSourceFactory`.

The full Rust target suite, formatting, and strict Clippy pass locally with the
feature on Apple Silicon macOS. A Debian bookworm container compiled the
GStreamer Rust bindings and probe but failed linking the repository's existing
ONNX Runtime static library because that image lacks required glibc C23 and
libstdc++ symbols. A separate Ubuntu 24.04 x86_64 container, with the GStreamer
and OpenSSL development packages installed, passed all five helper subprocess
tests, including v1/v2 settings validation and request/frame correlation. The
full feature-enabled Rust suite and strict Clippy pass on the development Mac.
The CI workflow installs the required development/base
plugins on Ubuntu 24.04 and GStreamer through Homebrew on macOS; hosted CI has
not yet reported.

The helper's bounded frame decoder and stateful session-binding validator now
live in `perception/src/capture_helper_protocol.rs`. They check record and
payload bounds before allocation, reject malformed or truncated records, and
validate session, contiguous frame identity, exact request digest, revision,
camera profile, RGB8 geometry, stride, and packed payload length. The probe
subprocess tests use this same decoder and validator; thirteen focused protocol
tests cover malformed input and boundary behavior, and all feature-enabled Rust
targets plus strict Clippy pass on macOS. The protocol module is reusable by a
future source adapter. At this point the protocol is still awaiting an initial
metadata-ready handshake and `CaptureSource` integration.

The following implementation update closes those two gaps. The stream now
begins with a bounded schema-v1 ready record, followed by the existing schema-v2
frame records. The ready record is emitted only after a real first sample has
negotiated the requested RGB dimensions and rational frame rate; that same
sample becomes frame zero. The parent binds ready metadata to the worker's
session ID and exact request digest, then validates every frame's identity,
sequence, format, stride, geometry, and payload length before handing it to the
worker. Pipeline PTS remains unverified, source timestamp remains absent, and
host read-completion time is created by the parent after the pipe read.

The feature-gated `GstreamerCaptureSourceFactory` now runs this helper through
the real capture worker, with explicit caller-supplied executable path,
request/header limits, frame count, and timeouts. It owns helper interruption,
bounded readiness wait, kill, and reap. Synthetic camera controls and source
timestamps are reported as unsupported, so preflight correctly rejects the
source instead of treating requested physical settings as applied readbacks.
The default live executable still uses its unavailable factory; no physical
camera or production provider is selected.

Verification on the development Mac: `cargo test --locked --all-targets
--features camera-gstreamer-probe` passes the entire Rust suite, including 114
library tests, 13 protocol boundary tests, five helper-process tests, and four
capture-worker adapter tests. The new adapter cases cover request-bound
readiness, exact worker-session correlation, budgeted frame delivery,
unsupported-controls rejection before reads, natural child exit/reap, stop
interruption/reap, and cleanup after a truncated ready record. Strict
feature-enabled Clippy and `cargo fmt --all -- --check` pass. Linux hardware,
physical camera behavior, device release/reopen, and IPC performance remain
unqualified.

The process-boundary tests use synthetic input. The Rust live executable and
Python supervisor now forward the same exact request and validate the source
acknowledgement, but the tests do not open a physical camera, exercise a blocked
physical native call, or prove hardware release/reopen. They do not qualify
exposure/gain readbacks, driver-level freshness, timestamps, camera accuracy,
physical permissions, or the pipe-copy latency budget. No production backend
is selected or enabled through the API.

## Transport microbenchmark evidence (2026-10-21)

The opt-in release benchmark in
`thoughts/performance-audits/capture-transport-20261021-run5/` compares a
capacity-one in-process `Vec` ownership handoff with a supervised helper stdout
pipe using the same setup-generated 3840×2160 RGB8 payload template. Across
three repetitions of 3,000 measured records, the pipe median was 324.4 frame
records/s with 3.36 ms sampled parent-receipt p95; the in-process arm handled
about 225,000 ownership handoffs/s with 0.005 ms sampled p95. The in-process
rate is not a payload-copy rate: application code transfers vector ownership
without copying the 24.9 MB payload. At 60 fps, the measured pipe rate is about
5.4× the target on this Apple M4 Max.

This narrows the question but does not qualify the decision. The payload is
synthetic and repeated; the setup SHA-256 identifies its template but timed
frames are not individually hashed. No camera read, detector, capture mailbox,
CPU time, resident memory, OS pipe capacity, or production freshness was
measured, and startup-to-ready setup differs between arms. The result suggests
that stdout-pipe transfer alone may fit a 60 fps budget on this machine; it does
not establish end-to-end latency, Linux behavior, or whether a shared-memory
design is worth its complexity. Keep the helper decision provisional until a
real-source run measures frame freshness, accuracy, CPU/RSS, copies, and
slow-consumer behavior on the target macOS and Linux hardware.

## Continuous physical helper mode (2026-10-21)

The helper now distinguishes finite test runs from supervised physical
sessions. Synthetic mode requires a positive finite frame count. The physical
mode can instead be launched with `--until-stopped`; its validated Rust source
configuration uses `frame_count: None` only for the feature-gated physical
factory. The capture worker remains responsible for interrupting the helper,
then killing and reaping it through the existing release path. Appsink and the
Rust capture mailbox continue to retain at most one pending frame each.

Unit and integration tests verify CLI mode conflicts, positive finite synthetic
counts, physical-only unbounded configuration, continuous physical v3 selector
validation, and prompt kill/reap of the feature-gated synthetic helper. No
physical webcam was opened to make these changes. A physical device may still
stall, disconnect, or fail to release; driver-level buffering, automatic
recovery, and macOS/Linux device behavior remain unqualified. The live binary
and Python supervisor can now select this source only when the caller supplies
the explicit feature-enabled helper and a valid v3 stable selector. The default
remains unavailable, and API provider selection remains closed pending
platform validation.

## Physical-source prototype progress (2026-10-21)

An additional opt-in `camera-gstreamer-source` feature now exercises physical
device selection inside the supervised helper. It accepts only capture-settings
v3 `source_defaults` with a stable selector, re-enumerates the current provider
inventory, requires exactly one match, requests the configured input profile,
and validates actual negotiated input and RGB8 output profiles before writing
request-bound readiness. It performs no camera-control writes. The live
executable registers this source only in feature-enabled builds and requires
an explicit command-line choice; its default source remains unavailable.

The source-enabled Rust all-target suite passes with 128 library tests and all
applicable integration tests, strict Clippy passes, and formatting passes. The
feature-disabled suite also passes with 120 library tests and all applicable
targets, confirming the source remains opt-in. Tests cover deterministic
selector resolution, provider-identity binding, no-match/duplicate rejection,
and per-buffer mapped stride. The live executable and Python supervisor can
select the physical factory only when the feature is compiled and an exact v3
selector is supplied; default source selection and API provider selection
remain closed.

**Mac short live qualification (2026-10-21):** the built-in webcam ran through
the opt-in GStreamer helper using v3 `source_defaults` (no camera-control
writes), requesting 1920x1080 at 30 fps with NV12 input and RGB8 1920x1080
output. After two seconds of warmup, the five-second measured interval
delivered 150 unique source observations and state snapshots (29.99/s), with
zero repeated IDs, gaps, out-of-order or uncorrelated snapshots, state-receipt-
interval p95 of 37.228 ms, zero audit export loss, and no ingress replacements.
Startup-to-first-localization was 1061.9 ms. The child exited cleanly with code
0. The deterministic checked-in tiny ONNX test model was used; no table
calibration or ground truth was available, so accuracy and numbered-ball
identity were not measured. The timestamp stage is `camera_read_complete`;
exposure time and source-to-state latency remain unavailable. See the [short-run
qualification report](../../../thoughts/performance-audits/20261021-103042-aeb9f8d8/report.md).
This is short smoke/throughput evidence only, not sustained-performance,
production-accuracy, forced-interruption/release, or Linux physical-hardware
evidence. The API provider remains closed.
