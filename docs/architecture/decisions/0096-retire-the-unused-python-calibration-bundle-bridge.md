# ADR-0096: Retire the unused Python calibration bundle bridge

- Status: accepted
- Date: 2026-10-23
- Deciders: Project maintainers
- Supersedes: ADR-0013's Python calibration exporter and direct-test tooling;
  ADR-0059's allowance to retain that exporter as an optional reference tool
- Superseded by: None

## Context

ADR-0013 introduced a Python adapter that converted the legacy
`CameraCalibrator` meter homography into the versioned Rust millimeter bundle.
The adapter has no production caller: the Python compatibility service accepts
caller-supplied calibration bundles, while the supported Rust runtime solves,
validates, persists, and applies calibration itself. Its only callers were
direct unit tests and one capture-settings test. Keeping this converter made a
test-only bridge look like a supported runtime tool.

The shared skewed-perspective bundle remains useful evidence. Rust tests use it
to validate the perspective geometry against the Rust solver and projector;
its historical Python origin is provenance, not an ongoing dependency. The
Python service still validates calibration-bearing capture requests, so those
request-contract tests remain and use the shared fixture.

## Decision

Retire `backend/vision/calibration/perception_export.py` and its direct tests.
Keep the capture-settings boundary and its validation tests, using the shared
perspective fixture rather than building a calibration through the obsolete
Python converter. Keep the versioned calibration schema, historical
Python-generated perspective fixture, Rust solver/store/projector, and their
geometry tests. Do not port the adapter to Rust: Rust already owns the
production capability. The pixel and physical-coordinate contract in ADR-0013
remains active and unchanged. ADR-0059's optional allowance for keeping the
Python exporter as a differential/reference tool is superseded; Rust solver
and projection tests plus independent physical measurements are the retained
validation path.

## Consequences

### Positive

- Removes a test-only converter and its 200-line validation implementation
  without removing the supported Rust calibration path or capture-request
  validation.
- Keeps a fixed historical perspective bundle for Rust solver/projection
  regression checks.

### Negative

- The deleted Python adapter no longer serves as a live differential
  implementation. Future calibration changes rely on the Rust solver,
  contract, geometry tests, and independent physical measurements.
- The fixture filename records its historical Python origin even though no
  supported workflow now executes that exporter.

## Alternatives considered

### Port the converter into Rust as a separate CLI

Rejected because the production Rust server already solves and applies
calibration bundles, and no current caller needs a second conversion command.
Recreating the test-only API would preserve the old bridge rather than the
supported calibration workflow.

## Verification

- `backend/tests/unit/test_capture_configuration.py` validates a supplied
  shared calibration bundle against the requested profile and revisions.
- `perception/tests/calibration_v1.rs` verifies the retained perspective
  fixture against the Rust solver and projection behavior.
- Search maintained backend source/tests for imports or calls to
  `perception_export` or `build_table_calibration_v1`; none should remain.
- Run `make test-all`, Rust formatting/strict Clippy, and validate all ADRs.
