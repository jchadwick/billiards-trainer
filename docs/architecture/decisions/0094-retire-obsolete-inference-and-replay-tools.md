# ADR-0094: Retire obsolete inference and replay tools

- Status: accepted
- Date: 2026-10-23
- Deciders: Project maintainers
- Supersedes: ADR-0051, ADR-0061, and ADR-0076
- Superseded by: None

## Context

This decision also retires only the replay-v2 ingress portion of ADR-0069;
its live-v3 readiness and capture-acknowledgement decision remains active.

The Rust `billiards-perception-infer` executable replayed still images from a
file or directory. It was not deployed and had no caller outside its own
source, tests and documentation. Its image replay, synthetic cadence, raw
tensor dumps and retained provider profiles did not validate the maintained
live or saved-video workflows.

The Python compatibility API also retained a `recorded_replay` mode backed by
`RustReplaySupervisor`. That supervisor launched the removed still-image
binary with arguments unsupported by the current video-replay executable. Its
tests used fake children, so they did not detect the broken executable
contract. The current saved-video command runs the shared production inference
path and writes bounded JSONL/report artifacts; it does not publish scenes to
the application gateway.

The Rust gateway's replay-v2 TCP ingress and the recorded-replay configuration
surface had no remaining producer after removal of the image CLI. Keeping the
protocol would preserve extra framing, parsing, identity and lifecycle code
without a supported workflow. The live-v3 ingress remains the only scene
producer protocol for the application gateway.

## Decision

Retire the still-image inference executable and its image-only runtime path,
flags, report fields, tests and public APIs. Remove the Python API
`recorded_replay` mode, its supervisor and tests, along with its configuration
fields and executable-path protection. Remove replay-v2 scene ingress and its
payload compatibility tests from the Rust gateway. Keep live-v3 validation and
its configuration-bound capture acknowledgement unchanged.

Do not move retired diagnostics or replay delivery into another command without
a distinct current caller and a test that validates the end-to-end workflow.
Keep the shared frame-inference engine, live capture pipeline, provider
profiling used by maintained evidence paths, saved-video JSONL and scene
artifacts, calibration, and reviewed temporal-evaluation APIs. The supported
evidence paths are:

- `billiards-perception-video-replay` for PTS-paced freshness or unpaced
  recorded-video capacity, with optional reviewed-label scoring.
- `billiards-performance-audit-run` for live state freshness, delivery, gaps,
  and optional detector/trajectory interference evidence.
- `billiards-temporal-evaluate` for localization and tracking accuracy against
  reviewed temporal annotations.

The video evaluator's outputs remain offline artifacts. It does not imply a
gateway replay mode or provide live camera or browser-display latency.

## Consequences

### Positive

- Removes a broken Python replay launch path and a Rust gateway protocol with
  no current producer.
- Removes a second, weaker still-image inference entry point and narrows the
  supported evaluation surface to tools that correspond to maintained flows.
- Keeps shared inference, live capture, saved-video evaluation, JSONL, and
  temporal accuracy workflows intact.

### Negative

- There is no standalone command for inspecting inference on one isolated
  image, and the application API no longer accepts recorded replay as a state
  producer. Isolated model cases can use a short video fixture or shared-engine
  tests; end-to-end delivery checks use the live-v3 producer contract.
- Historical CLI and replay-v2 records remain useful only as archived design
  and performance evidence, not as runnable workflows.

## Alternatives considered

### Keep the replay-v2 listener for future use

Rejected because it had no supported producer after the image-only CLI and
Python replay supervisor were removed. The saved-video evaluator has a distinct
offline purpose and does not require API state publication. If a concrete
gateway replay use case returns, add a new versioned contract with an active
producer and end-to-end tests.

### Retarget the Python supervisor to the video-replay executable

Rejected because video replay is an offline evidence command with artifact
outputs, not a service process or state publisher. Turning it into one would
recreate the removed gateway replay architecture instead of fixing the dead
path.

## Verification

- Remove the `billiards-perception-infer` binary and its image-only runtime
  API; confirm maintained source, CI and current documentation contain no
  runnable references.
- Remove Python API replay supervision and replay-specific configuration;
  verify stale config is rejected instead of silently selecting another
  runtime.
- Remove replay-v2 gateway handling while retaining live-v3 handshake,
  identity, freshness and state-delivery tests.
- Run Rust formatting, all-target tests and strict Clippy; run the full Python
  compatibility suite, configuration checks, and ADR validation.
- Exercise the retained video-replay CLI help and its shared inference tests.
