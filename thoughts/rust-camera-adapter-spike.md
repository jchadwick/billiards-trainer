# Rust camera adapter spike before production selection

> Historical dependency review (2026-10-21). The opening boundary and next
> steps below predate the Rust GStreamer capture helper and are superseded by
> [ADR-0033](../docs/architecture/decisions/0033-separate-rust-live-capture-process-from-replay-evaluator.md).
> Nokhwa remains unselected; its compile-only feature and CI check were removed
> after the tool audit. Use this note only as candidate-risk history.

**Status:** dependency decision deferred. This note records candidates and the
hardware evidence required before Stage 5 commits to one.

## Current boundary

`perception/` currently consumes recorded PNG/JPEG files in its executable.
The new `FrameInferenceEngine` accepts an immutable in-memory `RgbImage`, but
no Rust component opens a camera, records source frame identity, or owns the
capture lifecycle. The Python reference uses `cv2.VideoCapture`; its tests
replace the device with fakes and do not qualify physical camera modes or
release/reopen behavior on macOS and Linux.

The application has no verified capture device model or negotiated capture
profile recorded in the repository. It would be premature to install a
production capture dependency based only on its platform support table.

## Candidate evidence

- **Nokhwa** exposes native AVFoundation on macOS and V4L2 on Linux, camera
  formats and controls, and raw/decoded frame APIs. Its documentation permits
  a backend timestamp only when a backend supplies one; that value must not be
  called exposure time without validating timestamp point and clock domain.
  There is also a July 2026 upstream report of process abort on a macOS camera
  path with multiple formats in v0.10.11. Treat it as a spike candidate and
  test the actual USB camera/format combination before pinning a version.
  [Crate API](https://docs.rs/nokhwa/latest/nokhwa/struct.Camera.html),
  [buffer API](https://docs.rs/nokhwa/latest/nokhwa/struct.Buffer.html),
  [reported macOS issue](https://github.com/l1npengtul/nokhwa/issues/247).
- **Rust OpenCV bindings** provide the closest match to the retained Python
  `VideoCapture` adapter and expose AVFoundation/V4L2 backend selectors. They
  require installed OpenCV headers/libraries, build discovery and libclang;
  the current development host has no native OpenCV development install.
  This has packaging and build cost and still requires validating mode
  negotiation, buffering and timestamp semantics.
  [Rust bindings and installation requirements](https://docs.rs/crate/opencv/latest/source/README.md),
  [video I/O backend API](https://docs.rs/opencv/latest/opencv/videoio/enum.VideoCaptureAPIs.html).
- **GStreamer** remains useful for non-capturing discovery, pixel conversion,
  and synthetic helper tests. The installed macOS `avfvideosrc` does not expose
  exposure or gain controls, so it cannot by itself satisfy the live source
  contract. Its native runtime and plugins still need target-platform
  qualification if used for conversion.

**Review update (2026-10-20):** the current Nokhwa documentation reports
version 0.10.11 and lists AVFoundation and V4L2 under its `input-native`
feature. Its optional callback-thread API moves frame reads to a library-owned
thread, but the public API documentation alone does not establish that camera
release or interruption is prompt. More importantly, upstream issue
[#247](https://github.com/l1npengtul/nokhwa/issues/247) reports an uncatchable
process abort on macOS 15/26 with Nokhwa 0.10.11 when a device has multiple
formats at one resolution. The proposed fix in
[#248](https://github.com/l1npengtul/nokhwa/pull/248) remains open; a second
fix proposal, [#249](https://github.com/l1npengtul/nokhwa/pull/249), is also
open. Issue [#224](https://github.com/l1npengtul/nokhwa/issues/224) reports
that AVFoundation capture may remain reserved from other applications after
`stop_stream()` and drop on 0.10.10. These are reports, not proof that every
device fails, but they rule out treating the current release as production
qualified. Do not pin or activate Nokhwa in the application until the selected
device passes the exact-format open, interruption, release, and reopen tests
and an upstream fix is present in a reviewed release or a locally audited patch.

The current development host has GStreamer 1.28.3, but this is only host
availability evidence; it does not establish a distributable macOS/Linux
dependency set or a tested AVFoundation/V4L2 pipeline. The pool camera model
and the physical Linux target remain unidentified. Therefore no production
camera library is selected by this update. The hardware-independent
requested/negotiated-mode and capability report is now part of the capture
boundary, and the worker checks each frame's timestamp against that session
report before publishing it. Next use an explicitly invoked, disposable probe
against the identified device and verify it on both platforms. The probe and
its identifiers remain provenance data; they do not gate API, stream, or
camera access.

These documents establish candidate capabilities, not performance, camera
compatibility or production readiness.

## Source-level feasibility findings (2026-10-21)

The pinned Nokhwa 0.10.11 crate was inspected at source level. It can build on
the development Mac with `default-features = false` and `input-native`, without
OpenCV headers. The Linux V4L2 feature needs the CI runner's Clang/libclang and
kernel headers. That makes a feature-build spike possible, but does not make
the current adapter contract safe to implement with Nokhwa:

- `CaptureVideoProfile` is the raw camera mode (including rational FPS and
  input pixel format), while `RgbFrameEnvelope` always contains RGB8. Nokhwa
  requests a raw FourCC and decodes it to RGB8. The settings currently lack
  separate expected output dimensions, so the pipeline assumes that RGB output
  dimensions equal the raw camera dimensions. Rectification or conversion that
  changes dimensions cannot be reported accurately.
- AVFoundation mode selection and FPS readback do not establish an exact
  negotiated profile. The backend truncates advertised FPS, may select a
  same-resolution format with another FourCC, and can hit the upstream
  [same-resolution format-selection crash report](https://github.com/l1npengtul/nokhwa/issues/247).
- Nokhwa's AVFoundation callback copies camera buffers into an unbounded
  channel. A slow Rust reader can therefore accumulate stale frames and grow
  memory without bound, contrary to the latest-frame capture contract.
- Control identifiers and units differ by backend; the macOS `Gain` mapping
  is not a cross-platform sensor-gain value. The existing strict preflight
  must not treat those readbacks as equivalent.
- Linux device indices follow `/dev/videoN` enumeration and can change after
  reconnect. AVFoundation has a stable unique device ID, so identity handling
  also differs by platform.
- Both native reads can block without a cross-thread cancel operation. The
  current source trait requires prompt interruption; relying on process exit
  would require an explicit successor decision and subprocess shutdown tests.

The smallest prerequisite is ADR-0034: keep camera input mode separate from
the RGB8 output geometry, introduce capture-settings/acknowledgement v2 for the
new geometry, and preserve the strict v1 contracts unchanged. After that
contract is implemented in Rust and Python, evaluate the OpenCV Rust binding
as the next candidate because it matches the retained Python backend more
closely. It still needs exact source readback, bounded buffering, process-safe
shutdown, and both-platform qualification; no adapter is selected by this
review.

## Build-only native feature probe (2026-10-21)

The optional `camera-nokhwa-build-spike` Cargo feature now compiles Nokhwa
0.10.11's native backend set without registering a capture source or changing
the live executable's factory. The exact command
`cargo check --locked --all-targets --features camera-nokhwa-build-spike`
passes on the development Mac (Apple Silicon) and on a Linux x86_64 Debian
container after installing `clang`, `libclang-dev`, `libudev-dev`,
`pkg-config`, and `linux-libc-dev`. The repository's Ubuntu/macOS CI workflow
now repeats this opt-in check, installing the native packages on Linux only;
hosted CI has not yet run for this change.

This resolves native compilation feasibility for the probe only. It does not
resolve Nokhwa's unbounded callback buffering, strict mode/control readback,
blocking-open/read interruption, release/reopen, device identity, or physical
camera issues documented above. The feature stays opt-in and must not be
enabled in the production provider. Continue evaluating a source design that
can meet bounded freshness and shutdown before registering a factory.

## Supervised GStreamer synthetic helper (2026-10-21)

ADR-0036 now puts potentially blocking native camera APIs behind a separately
killable helper during adapter evaluation. This is an evaluation boundary, not
a production backend selection. The first prototype uses GStreamer's
`videotestsrc`, `videoconvert`, fixed RGB caps, and a one-buffer appsink; it
does not open a physical camera or enter the live provider path. The prior
source-level recommendation to evaluate OpenCV next is deferred until helper
lifecycle and IPC cost are measured.

The opt-in Cargo feature `camera-gstreamer-probe` builds a distinct probe
executable. Each synthetic record has a versioned JSON header and a
length-prefixed RGB8 payload. Width, height, frame rate, frame count, session
identity, and both per-frame and aggregate application-memory limits are CLI
inputs. Conversion copies only visible pixels from the negotiated video
stride into `CaptureFrameBudget` storage. GStreamer PTS is retained as
unverified pipeline presentation time; the probe reports no source timestamp
and records host monotonic read-completion separately.

On the development Mac (GStreamer 1.28.3), the full Rust target suite passes
with this feature, including three subprocess tests: odd-width RGB framing and
identity, PTS gaps under a delayed reader, and force-termination/reaping. Unit
tests reject short strides, truncated planes, and mismatched destination
lengths. Strict Clippy and formatting pass. The Linux Debian bookworm container
compiled the GStreamer Rust dependencies and probe but failed at linking the
repository's existing ONNX Runtime static library: required glibc C23 and
libstdc++ symbols are absent from that older image. A separate Ubuntu 24.04
x86_64 container, with GStreamer and OpenSSL development packages installed,
passed all five helper subprocess tests, including the request-bound v1/v2
path. The checked-in CI job installs the required development and base-plugin packages on Ubuntu 24.04; hosted results
are pending.

This prototype proves synthetic caps/conversion, bounded appsink configuration,
and the operating-system kill/reap mechanism. It now consumes exact
capture-settings bytes directly from bounded stdin, validates the existing v1/v2
contract, derives geometry/rate/memory bounds from that request, and attaches
the exact digest and configuration revision to each output frame. This has not
yet been exercised through a live supervisor or checked as a
`CaptureSessionReport`. The helper has not tested an interrupted native camera
open/read, verified release/reopen, measured IPC latency/copy overhead, or
established physical device timestamps, controls, driver buffering, or camera
accuracy. The live factory remains unavailable. Next, connect the request-bound
protocol to a supervised `CaptureSource`, then compare helper pipes with
identical in-process synthetic payloads before selecting any physical adapter.

The bounded record decoder and session-binding validator have now been moved
into reusable Rust library code. The probe subprocess tests use the shared
decoder and reject malformed/truncated records and mismatched session, sequence,
request digest, revision, profile, geometry, format, stride, or payload length.
Thirteen focused protocol tests and the full feature-enabled Rust suite pass on
the development Mac; strict Clippy passes as well. This closes the protocol
parser test gap only. There is still no initial metadata-ready record, no
helper-backed `CaptureSource`, and no real camera qualification.

## Spike sequence

1. Use the actual target USB camera on physical macOS and Linux machines. Keep
   the candidate adapter optional and outside the API provider path.
2. Enumerate device modes and controls. Request the project camera profile and
   record requested versus negotiated resolution, frame rate, pixel format,
   exposure controls and all timestamp fields.
3. Capture a ten-minute run through a camera-owner thread and a capacity-one
   newest-frame handoff into `FrameInferenceEngine`. Record read-complete time,
   optional source timestamp and its clock semantics, drops, frame age, CPU,
   allocations and conversion/copy cost. Compare decoded RGB samples with the
   Python reference from the same device and settings.
4. Repeatedly start, stop, unplug and reconnect. Verify one owner, deterministic
   stream release, a new source-session identity after restart, and recovery
   without opening a second camera or falling back silently.
5. Build and package from clean macOS and Linux environments. On macOS, verify
   the app identity and camera usage-description/permission flow for the
   intended distribution.
6. Compare freshness, correctness and operational behavior against the Python
   capture path using the same camera and detector settings. Reject any
   candidate that cannot report the raw input mode and RGB8 output geometry,
   verify source controls, bound stale buffering, or shut down under the
   configured process deadline. First exercise the supervised GStreamer helper
   with synthetic input and compare its pipe-copy cost with in-process
   payloads. Evaluate generic camera crates only against those same device
   tests; if their control or lifecycle surface is insufficient, use thin
   platform-native camera helpers and retain the shared Rust parent protocol.

Keep the camera handle confined to its owner thread. Transfer immutable owned
frames into the bounded latest-work slot; do not share a mutable capture buffer
with inference. A library timestamp is recorded with its documented source and
clock domain; otherwise software read-complete time remains explicitly a
publication/receipt timestamp.

Do not accept a camera dependency until both platforms can build and package it,
the target device negotiates the requested profile without silent fallback,
capture release/restart is repeatable, and measured frame age/drop behavior is
acceptable for the detector. This spike does not itself authorize changing the
Python default or activating the Rust provider.

## Bounded helper-to-worker integration update (2026-10-21)

The previously noted missing ready handshake and helper-backed source have now
been implemented for synthetic evaluation. The feature-gated GStreamer helper
waits for its first `videotestsrc` sample, verifies actual negotiated RGB
dimensions and rational frame rate, emits a bounded ready preface, then reuses
that same sample as frame zero. The test harness and adapter use the same
library decoder and stateful frame-binding validator.

`GstreamerCaptureSourceFactory` now opens the helper through the existing
capture worker. A backward-compatible session-aware factory hook passes the
worker's session identity into the helper. The adapter binds readiness and
every frame to the exact request bytes, allocates payloads from the worker's
shared frame budget, and kills/reaps the process during interruption or
release. It declares synthetic camera controls and source timestamps
unsupported; the host records read completion separately, and pipeline PTS is
never promoted to exposure time. A preflight test confirms that the source is
rejected before reads when the requested physical exposure/gain controls are
unavailable.

The full feature-enabled Rust target suite passes on the development Mac,
including the four worker-adapter lifecycle tests, five helper-process tests,
and thirteen bounded-protocol tests. Strict Clippy and formatting pass. This
verifies protocol and worker composition only. The helper still does not open a
physical camera, provide real device-control readbacks, establish sensor-time
semantics, or qualify Linux hardware, driver buffering, or release/reopen. Keep
the live provider unavailable while those device-specific gates remain open.

## GStreamer source-control feasibility review (2026-10-21)

`gst-inspect-1.0 avfvideosrc` on the development Mac shows that installed
GStreamer 1.28.3 can enumerate the built-in camera and advertise source caps,
but its source element exposes no exposure-mode, exposure-value, or gain
properties. The available device selector is an enumeration index. The
official [`avfvideosrc` property reference](https://gstreamer.freedesktop.org/documentation/applemedia/avfvideosrc.html)
documents `unique-id` only as a GStreamer 1.30 addition, so upgrading could
improve device selection but would not add the missing control contract.

Apple's [`AVCaptureDevice`](https://developer.apple.com/documentation/avfoundation/avcapturedevice)
API identifies devices by unique ID and requires an explicit configuration
lock for hardware controls. Linux V4L2 exposes camera control IDs through its
[camera-control API](https://docs.kernel.org/userspace-api/media/v4l/ext-ctrls-camera.html),
but the target driver's supported controls and values are unknown. These
platform APIs are candidates for native helpers, not proof that the intended
table camera supports the requested controls. ADR-0038 proposes keeping
physical identity and readback verification in platform-native helpers behind
ADR-0036's supervised process boundary; GStreamer remains for inventory,
conversion where appropriate, and synthetic tests.

No physical source was opened during this review. The installed Mac plugin
inspection and official API docs are feasibility evidence only. The table
camera model, Linux target, normalized exposure/gain units, and physical
qualification remain outstanding, so the live provider remains unavailable.

## Isolated pipe-transfer measurement (2026-10-21)

The opt-in Rust transport microbenchmark now compares bounded in-process
`Vec` ownership handoff with a supervised helper writing the same deterministic
3840×2160 RGB8 payload template over stdout. Payload generation and template
SHA-256 verification are outside the timed loops. On the development M4 Max,
three 3,000-frame repetitions measured a median 324.4 pipe frame records/s and
3.36 ms sampled parent-receipt p95. The in-process result was about 225,000
ownership handoffs/s, but it moves no frame bytes through application code and
must not be read as memory bandwidth. At 60 fps the measured pipe rate is about
5.4× the target under this synthetic workload.

This makes a loopback pipe a credible candidate for an initial 60 fps helper
prototype on this Mac, not a production choice. The run did not open a camera,
run the detector, measure frame age end-to-end, measure CPU or RSS, query pipe
capacity, or verify every timed payload with a checksum. Confirm the trade-off
using actual source frames and detection, then repeat on the target Linux
device before deciding whether shared memory or direct in-process capture
provides a necessary benefit. The raw result and machine/worktree details are
in [run 5](performance-audits/capture-transport-20261021-run5/report.md).
