# ADR-0032: Python supervisor validates Rust live startup

- Status: superseded
- Date: 2026-10-20
- Deciders: Project maintainers
- Supersedes: None
- Superseded by: ADR-0057 (provider-selection gate only; process contract retained)

## Context

ADR-0031 composes Rust's capture ownership, source preflight, inference, scene
pipeline, request-bound protocol-v3 acknowledgement, and bounded stop lifecycle.
The Rust runtime now has a separate live executable and an opt-in, feature-gated
GStreamer physical helper prototype. Neither has been qualified with physical
hardware. The FastAPI application owns the existing API, configuration, scene
ingress, public state publication, and provider lifecycle. Its Rust provider
currently supports recorded replay only; Python remains the default live
provider.

Python can serialize an immutable `CaptureSettingsSnapshotV1` into exact,
deterministic UTF-8 JSON bytes, and the ingress can independently validate the
source report in the Rust v3 hello against that same snapshot. The supervisor
must preserve those bytes across the process boundary and must not claim
readiness merely because the process starts or the socket binds. The physical
helper can be selected explicitly for qualification runs, but normal API
provider selection remains unavailable until target-device and accuracy
qualification are complete.

## Decision

The Python gateway will own a `RustLiveSupervisor` for the lifetime of a live
Rust provider. It resolves one immutable capture-settings snapshot, passes its
exact `to_json()` UTF-8 bytes to the Rust child over the child's standard input
and closes that input after the one-shot request, and configures the local
scene ingress with the same snapshot. Rust must parse those exact bytes and
return the compatible opened-source report through the request-bound v3
handshake before Python ingress accepts any scene. Python reports the provider
ready only after a complete usable localization reaches its retained scene
state publisher, consistent with ADR-0027.

The supervisor passes the non-secret process contract explicitly as arguments:
`--capture-settings-stdin`, `--capture-settings-max-bytes`,
`--engine-instance-id`, `--source-session-id`, `--max-detections`, optional
`--intra-op-threads`, `--startup-timeout-ms`, `--poll-interval-ms`,
`--graceful-stop-timeout-ms`, `--scene-snapshots-tcp`,
`--scene-snapshots-max-message-bytes`, `--scene-snapshots-connect-timeout-ms`,
`--scene-snapshots-write-timeout-ms`,
`--scene-snapshots-reconnect-initial-delay-ms`, and
`--scene-snapshots-reconnect-maximum-delay-ms`. When the exact v3 request
contains a validated stable GStreamer selector, the caller must also provide
the helper executable, ready/frame-record bounds, sample-pull timeout,
readiness timeout, and child-exit timeout/poll interval. The supervisor passes
these as explicit helper arguments and selects the physical source only in the
feature-enabled build. An absent selector continues to choose the unavailable
source. The stdin request bound and outbound scene-payload bound are
independent. The child graceful-stop deadline must leave time for the
supervisor's later process-reap deadline. Capture settings are not duplicated
in arguments or environment variables.

The supervisor owns the child process, bounded output capture, ingress,
consumer worker, startup deadline and stop/reap sequence. It does not forward
the parent environment wholesale and does not fall back to Python if the Rust
child fails. Provider selection remains restricted to existing qualified
paths until physical source behavior and ball accuracy are qualified and the
API lifecycle is deliberately wired. Replay remains on its existing command
and v2 handshake.

The supervisor also requires an explicit inference/tracking settings object.
It validates detector values, model paths, ONNX/checkpoint hashes and export
manifest lineage before process creation, then passes those values as explicit
arguments. If omitted, the intra-op thread count preserves ONNX Runtime's own
thread policy. Camera profile, image space, configuration and rectification
revisions, and the complete calibration bundle remain sourced from the same
capture request used by ingress; they are not duplicated in another settings
channel.

## Consequences

### Positive

- One exact capture request is independently checked by the Rust source owner
  and the Python scene ingress; the acknowledgement cannot silently describe
  a different configuration revision.
- Python keeps API state publication and process cleanup in its existing
  lifecycle, while Rust remains the sole future camera owner.
- Startup, child failure, source silence, and cleanup can be tested with a
  deterministic child fixture without claiming physical camera readiness.

### Negative

- The physical helper remains a prototype. It does not qualify camera mode,
  driver freshness, device release, or ball localization.
- This supervisor is not constructed by the production API. Preview and
  calibration handover and target-device qualification remain open.
- Python configuration changes that affect capture require a new supervisor
  session so the request snapshot and acknowledged source report remain
  immutable.
- Both process and ingress readiness must be cleaned up on every failed start,
  and the gateway must report degraded state instead of silently switching
  engines.

## Alternatives considered

### Command-line arguments or a temporary file

Put the request in command-line arguments or a temporary file. Arguments are a
poor fit for a potentially calibration-bearing JSON document and risk platform
argument-size limits; a temporary file adds an unnecessary persistent
cross-process artifact and cleanup path. A one-shot stdin payload preserves the
exact bytes without storing a second copy on disk.

### Let Rust own the public API and provider lifecycle

Rejected because FastAPI, its current clients, Core integration, and state
read model are intentionally retained. Moving those boundaries would broaden
the migration without improving the primary camera-to-location path.

### Select Rust live as soon as the supervisor exists

Rejected because the supervisor only proves process and protocol behavior. It
cannot establish camera mode negotiation, physical release/reopen, image-space
correctness, platform packaging, or ball-position accuracy. Those gates remain
separate prerequisites.

## Verification

- Python child-process tests prove the exact settings bytes are delivered once,
  the ingress rejects mismatched request-bound acknowledgements, and a usable
  scene—not process start alone—establishes readiness.
- Child-process tests inspect the explicit transport/session arguments and
  verify the request body is absent from the environment and command line.
- Child-process tests inspect request/model bounds, optional thread policy,
  startup/poll budgets, and the child-versus-parent shutdown deadlines.
- Failure tests prove startup timeout, child exit, stream failure and shutdown
  reap the child and close ingress/tasks without Python fallback.
- Keep `perception.provider=rust` with `mode=live` rejected until physical
  source and localization qualification on macOS and Linux is complete.
- Preserve recorded-replay v2 tests and validate the full ADR register.

## Implementation progress (2026-10-20)

`backend.perception.rust_live_supervisor.RustLiveSupervisor` implements the
process and ingress lifecycle contract with explicit required settings. It
writes the capture snapshot's exact UTF-8 bytes once to child stdin, closes the
pipe, and passes the bound scene endpoint, transport limits, and fresh
engine/session identities as the documented arguments. The same snapshot
configures the Python v3 ingress. Startup requires an accepted matching source
report and a usable localization published while the child is still running
and state remains fresh. Child output retention is bounded, and startup error,
timeout, and close paths terminate/reap the process and stop the worker/ingress.

The required inference/tracking settings object validates numeric bounds,
model artifact hashes, manifest lineage, input size and class mapping before
launch. Its validated values and request-derived camera/tracking revisions
are passed explicitly to the child; calibration remains in the exact capture
request. Tests cover these checks along with exact request bytes, explicit
arguments, matching and mismatched acknowledgements, usable-scene readiness,
early exit, timeout and shutdown cleanup. The focused supervisor and replay
selection passes 17 tests; Ruff, formatting and strict mypy pass. The latest
full backend result and configured coverage result are tracked in the current
migration-plan progress entry. The production API does not construct this
supervisor yet, and no live executable, camera adapter, or target-platform
qualification exists. Rust live remains unavailable through provider
selection.

The Rust runtime now accepts the supervisor's `--source-session-id` as the
capture-worker identity. The same validated value is used by source frames,
the v3 source acknowledgement and emitted scene snapshots; invalid identities
are rejected before the source factory opens. A loopback integration test
asserts equality across the hello, full acknowledgement and first scene. This
resolves the identity ownership needed by the live executable but does not
select a camera adapter.

**Implementation progress (2026-10-21):** ADR-0033 adds the separate
`billiards-perception-live` process contract and reusable Rust runner. The
Python launch now includes independent capture-request and scene-payload
limits, explicit detection count, optional ONNX intra-op threads, startup and
poll intervals, and a child release deadline shorter than the parent reap
deadline. Ten focused supervisor tests pass; Ruff formatting/lint and mypy
pass. Production provider selection remains closed because this binary build
has no qualified camera adapter.

**Physical helper launch progress (2026-10-21):** `RustLiveSettings` now binds
helper launch configuration to the exact capture snapshot. A valid v3
`gstreamer-device-v1:<lowercase sha256>` selector, raw image space, revision
zero, and `source_defaults` policy are required; the helper path must be an
absolute executable file, and all byte/time limits are validated before the
scene listener or child starts. The supervisor passes these settings
explicitly and continues to write the exact request only to stdin. The live
binary keeps `UnavailableCaptureSourceFactory` as its default and requires an
explicit feature-enabled `gstreamer-physical` source choice. Tests verify both
default behavior and feature-disabled failure, plus feature-enabled factory
construction without opening a device. Python supervisor tests verify explicit
argv values and reject invalid selectors, policies, paths, and timeouts. The
API does not construct this supervisor, and no physical webcam has been
opened; startup, accuracy, freshness, interruption, release/reopen, and
macOS/Linux behavior remain unqualified.

## Scope supersession (2026-10-22)

ADR-0057 supersedes this record only where it kept `perception.provider=rust`
with `perception.mode=live` unavailable at API preflight. The Python service
now permits that explicit mode and constructs this supervisor, while keeping
Python as the bundled default and requiring the exact stable selector, helper,
first-localization readiness, and fail-closed behavior. This record's request,
acknowledgement, timeout, and shutdown contracts remain active.
