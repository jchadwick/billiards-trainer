# ADR-0033: Separate Rust live capture process from replay evaluator

- Status: accepted
- Date: 2026-10-21
- Deciders: Project maintainers
- Supersedes: None
- Superseded by: None

## Context

ADR-0031 defines the Rust-owned capture lifecycle and ADR-0032 defines the
Python supervisor's exact-request, readiness, and child-process contract. The
only Rust executable is still a recorded-image evaluator, while the Python
supervisor emits live-only arguments and a one-shot capture-settings request.
Making replay and live acquisition share one command would blur their input,
timing, and lifecycle guarantees.

The Rust crate has a hardware-neutral `CaptureSourceFactory` seam and a
tested `LiveCaptureRuntime`, but no qualified macOS or Linux camera adapter.
The new process must therefore compose the production inference and transport
path without fabricating camera observations or making an unqualified adapter
available to provider selection.

## Decision

Add a dedicated `billiards-perception-live` executable and a reusable live
application runner. The executable accepts the Python supervisor's explicit
inference, tracking, transport, identity, lifecycle, and bounded-stdin settings;
it reads the exact capture request once and passes those unchanged bytes to
`LiveCaptureRuntime::start_with_session_id`. Detector warm-up happens before a
source factory can open hardware. The runner retains the runtime through
request-bound protocol-v3 readiness and frame processing, and explicitly
stops it when a termination signal arrives.

The runner accepts an injected `CaptureSourceFactory`, which allows end-to-end
software tests to use deterministic frames and the checked-in ONNX model. The
standalone executable uses a factory that reports a clear unsupported-adapter
error until a concrete platform adapter is implemented and qualified. It must
not synthesize frames, choose a camera library, or become selectable by the
production provider while that adapter is absent. The replay executable and
its interface remain unchanged.

## Consequences

### Positive

- Replay and live process contracts remain explicit and independently
  testable.
- The future camera adapter plugs into the same process runner that owns the
  real detector, tracking, request-bound acknowledgement, scene transport, and
  bounded release lifecycle.
- A missing adapter fails visibly; software-only integration evidence cannot
  be mistaken for physical camera qualification.

### Negative

- The standalone live executable cannot produce observations until a concrete
  Rust camera adapter is added.
- Live process configuration now has a larger explicit CLI and Python
  supervisor contract, including independent request and scene-payload limits.
- Graceful termination depends on the source adapter honoring interruption
  and returning from open/read/release within its configured budget.

## Alternatives considered

### Extend the recorded-image evaluator

Rejected because a replay command has file enumeration, replay pacing, and
measurement output semantics that do not belong in a persistent camera
process. Combining them would make live startup and shutdown behavior harder
to test and easier to invoke incorrectly.

### Select a camera library as part of the process boundary

Rejected because this host and the available evidence do not qualify a device
or physical Linux target, and current adapter options have unresolved
packaging or stability risks. Adapter selection remains a separate decision
based on actual macOS and Linux hardware tests.

### Use the existing Python capture implementation as the Rust source

Rejected because that preserves Python as camera owner and fails to test the
target ownership boundary. The application runner instead accepts the Rust
source factory seam directly.

## Verification

- Rust CLI tests parse the supervisor argument contract and reject invalid or
  missing live settings.
- Bounded stdin tests prove exact-byte preservation, UTF-8 validation, EOF
  handling, and rejection above the configured request limit.
- Live application tests use a deterministic `CaptureSourceFactory`, the
  checked-in two-class ONNX fixture, and the production v3 receiver to verify
  model warm-up, source identity, readiness, scene delivery, and explicit
  source release.
- Live-application shutdown tests set the shutdown flag and verify that the
  application requests source release, waits for completion, and reports an
  unconfirmed release as failure. OS signal delivery is registered by the
  executable but is not independently exercised by these tests.
- `cargo fmt --check`, strict Clippy, all Rust targets, ADR validation, and
  Python supervisor tests verify interface consistency. Physical camera and
  platform qualification remain prerequisites to provider selection.

## Implementation progress (2026-10-21)

`perception::live_application` composes the verified frame engine, explicit
warm-up, request-derived calibration and revisions, tracking, live runtime,
request-bound v3 readiness, and bounded shutdown. The separate live binary
parses the matching process contract, caps stdin before parsing, and handles
SIGTERM/SIGINT by retaining the runtime until source release completes. Its
current source factory fails explicitly because no qualified Rust camera
adapter is registered. A separate integration-test process uses the checked-in
two-class ONNX model, a deterministic source and the production TCP receiver;
it confirms the source identity in the v3 acknowledgement and scene, one
ball-location output, and release after shutdown. All 213 Rust tests pass with
formatting and strict Clippy. Ten focused Python supervisor tests pass, and
Ruff and mypy pass. This verifies the software process composition only; it
does not claim camera capture, macOS/Linux hardware qualification, live provider
selection, or physical accuracy.

**Camera-adapter feasibility review (2026-10-21):** the Rust crate has no
OpenCV dependency, and the development host has Python OpenCV but no OpenCV
development package metadata or headers; a `pkg-config opencv4` probe fails.
The Rust OpenCV binding therefore adds a native build prerequisite that the
current Linux/macOS CI matrix does not install. The generic OpenCV capture
properties also depend on the selected backend, driver, and device, while the
capture contract requires source-reported negotiated settings and exact
readbacks for required exposure and gain. A successful setter or a copied
request value is insufficient under ADR-0029. The `CaptureInterrupt` contract
also requires interruption of open/read, which a generic blocking `read()`
does not establish. These constraints match OpenCV's documented
[backend/device-dependent property behavior](https://docs.rs/opencv/latest/opencv/videoio/enum.VideoCaptureProperties.html)
and [native build requirements](https://github.com/twistedfall/opencv-rust/blob/master/INSTALL.md).

The available `nokhwa` native backend is a candidate for a build-only spike:
its documentation lists AVFoundation on macOS and Video4Linux on Linux, and
exposes blocking frame acquisition and stream stop operations
([backend list](https://docs.rs/nokhwa/latest/nokhwa/utils/enum.ApiBackend.html),
[camera lifecycle](https://docs.rs/nokhwa/latest/nokhwa/struct.Camera.html)).
The dependency is pinned behind the opt-in Cargo feature
`camera-nokhwa-build-spike` and is not selected by either executable factory.
The command `cargo check --locked --all-targets --features camera-nokhwa-build-spike`
passes on Apple Silicon macOS and Linux x86_64 in a Debian container with the
V4L2 build headers installed. The repository CI now runs that feature check on
its Ubuntu and macOS matrix; hosted CI has not yet provided a result. This
proves native compilation only. It does not qualify camera, control-readback,
interruption or release behavior, and does not resolve the unbounded buffering
and lifecycle concerns in the feasibility review. First exercise synthetic
frame conversion and strict metadata, then decide whether a process-isolated
or different adapter can satisfy bounded freshness and shutdown before
physical-device qualification on both operating systems.

The retained Python video process can select `vision.camera.video_file_path`
instead of the numeric device index, while the Rust capture-settings resolver
uses only the numeric index. The default configuration names a demo video.

**Camera spike retirement (2026-10-23):** the compile-only `nokhwa` dependency
and `camera-nokhwa-build-spike` CI target were removed after the application
selected the supervised GStreamer capture path. The spike never opened a
camera, produced frames, or participated in live source selection. Its build
results and API research remain historical evidence in
`thoughts/rust-camera-adapter-spike.md`; do not treat them as a supported
backend or camera qualification.

**Source-resolution follow-up (2026-10-21):**
`backend.perception.capture_configuration.resolve_capture_settings()` now
checks the effective `vision.camera.video_file_path`, including its bundled
default, before building a live camera request. A nonempty file path is
rejected with guidance to clear the path explicitly or select recorded replay;
the Python request therefore cannot describe a numeric camera while Python's
camera process would open the demo video. Empty string and null both mean that
no video source is selected, after which `device_id` must still be a valid
nonnegative camera index. Tests cover the bundled-default rejection, explicit
clear plus numeric device, configured video rejection, and malformed path type.
This closes the software configuration mismatch only; it does not qualify a
camera adapter, enable Rust live provider selection, or change the Python
default configuration. The complete backend suite passes 1,633 tests with two
skips when coverage collection is disabled; the repository's independent
coverage gate remains open.
