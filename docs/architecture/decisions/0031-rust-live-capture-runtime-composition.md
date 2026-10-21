# ADR-0031: Rust live capture runtime composition

- Status: accepted
- Date: 2026-10-20
- Deciders: Project maintainers
- Supersedes: None
- Superseded by: None

## Context

The Rust migration has independently tested a strict capture-settings parser,
source ownership coordinator, preflight comparison, RGB8 inference pipeline,
and request-bound protocol-v3 readiness handshake. The Rust executable still
only runs recorded inputs, so no production caller composes those pieces. The
Python gateway likewise has no live supervisor that can provide one exact
capture-settings request to both the Rust process and its scene ingress.

An opened camera must not produce a usable scene until Rust compares the
source-reported profile, geometry, capabilities, timestamp semantics and
required control readbacks to the request and Python accepts the matching
source report. The same request bytes must be hashed and validated across both
processes. A failed or timed-out start must retain source ownership until
release is confirmed; otherwise a retry could open a second owner while the
first device handle is still active.

There is no selected or qualified camera adapter. The current development host
has a built-in camera but not the identified pool camera, and the physical Linux
target is unknown. The runtime lifecycle can still be implemented and tested
with injected sources without selecting a device API or enabling live mode.

## Decision

Add a Rust-owned live runtime that composes the existing source coordinator,
capture-settings wire parser, `CaptureScenePipeline`, and `LatestSceneWriter`
for one source session. The runtime receives the exact Python-serialized
capture-settings bytes and an already configured detector, tracking policy,
scene transport and source factory. It parses those bytes once, derives the
validated frame budgets and source expectations from that result, opens the
injected source under the process-wide owner gate, waits for source preflight,
binds the complete report through the v3 readiness handshake, and only then
allows the sole frame receiver to process observations.

The runtime owns the coordinator and pipeline through startup, processing and
shutdown. A preflight rejection or acknowledgement failure is an observable
failed state; the caller must retain the runtime until stop/release completes.
A wait timeout reports pending state and never releases ownership. The runtime
does not select a camera backend, invent applied settings, resolve model policy,
or enable the `rust` + `live` application provider. Concrete adapters and the
Python supervisor are added as separate dependent work after this library
boundary is covered.

## Consequences

### Positive

- A future provider can reuse one tested lifecycle rather than hand-compose
  preflight, receiver readiness, pipeline binding and cleanup.
- The exact request digest used by Rust remains the same one Python gives to
  ingress for independent validation.
- Injected fake sources can test the actual runtime state machine and scene
  output without camera access or model assets.
- Replay APIs and their version-2 handshake remain unchanged.

### Negative

- The runtime is not independently deployable until a concrete source factory
  and the application launch path are implemented.
- Callers must keep and stop a failed runtime before retrying; dropping an
  active coordinator can wait indefinitely for a misbehaving adapter.
- Detector and scene settings still need a separately validated effective
  runtime configuration; capture settings alone do not describe the model.

## Alternatives considered

### Let each application caller compose the existing public pieces

Rejected because it duplicates lifecycle sequencing and makes it easier for a
caller to process a frame before request-bound readiness or discard a timed-out
source owner.

### Couple live runtime construction to one camera library

Deferred because the pool camera model and physical Linux target are unknown,
candidate libraries have unresolved release/build concerns, and no target
device has passed format, control, interruption, release/reopen or packaging
qualification. The injected factory keeps the runtime testable without turning
that uncertainty into a production dependency choice.

### Reuse the recorded-image audit command for live input

Rejected because `run_audit` has finite file scheduling, replay timestamps and
report semantics. Camera capture needs sensor/read-completion provenance,
exclusive resource ownership, preflight and interruptible shutdown.

## Verification

- Parse the exact capture-settings bytes and prove the resulting digest is
  echoed by Python's v3 readiness peer before frame inference can occur.
- Verify preflight mismatch and wrong acknowledgement fail closed and retain
  the source owner until explicit stop/release.
- Verify a timeout leaves the runtime pending and does not permit another
  source owner to start.
- Exercise one complete source frame through the real runtime and scene
  transport with deterministic detector/source fixtures.
- Preserve all recorded replay CLI, v2 transport and gateway lifecycle tests.
- Keep application live-provider selection disabled until a real camera
  adapter and macOS/Linux qualification meet ADR-0015 and ADR-0018 gates.

## Implementation progress (2026-10-20)

`perception::capture_runtime::LiveCaptureRuntime` now implements the library
composition in this decision. It parses the caller's exact JSON bytes once,
derives the coordinator's per-frame and aggregate buffer budgets from the
resolved request, rejects mismatched pipeline/tracker configuration before
source open, and creates the capture-to-scene pipeline before starting the
injected factory. A bounded readiness wait retains the coordinator. The
runtime exposes its sole frame receiver only after compatible source
preflight and a matching protocol-v3 `ready` response. Preflight rejection,
handshake mismatch and stop timeout remain observable; source ownership is
retained until explicit stop/release completes.

Deterministic integration tests exercise complete v3 scene delivery, source
preflight rejection without inference, mismatched readiness without scene
delivery, ownership retention through a timed-out stop, and rejection of
duplicated pipeline settings before the factory opens. `cargo fmt --check`,
strict Clippy and all-target Rust tests pass (208 tests). These checks verify
software composition and ownership behavior only. No camera adapter is
selected, no Python live supervisor/provider is wired, and macOS/Linux
physical-device qualification remains outstanding.

## Revisit conditions and related records

ADR-0015 governs exclusive camera ownership and owned frame buffers; ADR-0018
governs platform readiness and reversible rollout; ADR-0029 defines source
readbacks; ADR-0030 defines the request-bound acknowledgement; ADR-0028 defines
the bounded epoch-scoped local transport. Revisit the runtime ownership model
if multi-camera operation, multiple concurrent detectors, or measured pipeline
deadlines require an independent worker/process boundary.
