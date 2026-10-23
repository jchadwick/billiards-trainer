# ADR-0059: Solve table calibration in Rust with capture provenance

- Status: superseded
- Date: 2026-10-22
- Deciders: Project maintainers
- Supersedes: None
- Superseded by: ADR-0096 for the optional Python exporter reference-tool allowance

> The decision that Rust owns production table-calibration solving remains
> active. ADR-0096 supersedes only the optional allowance to retain the Python
> exporter as a differential/reference tool.

## Context

Rust live perception now owns public-state construction, but the capture
request currently carries no approved table calibration. Rust can validate an
existing `TableCalibrationV1` and project measured pixel centers into table
millimeters, but it has no production solver from the four ordered table
corners. The Python calibration exporter can make such a bundle from a legacy
transform, but it has no production caller and the legacy transform does not
retain the camera profile, source image space, or rectification revision.

Table coordinates are useful only when their axes, units, calibration
revision, and exact source-image provenance are explicit. A solver must not
infer those fields from corner coordinates or reuse a calibration across a
different camera mode.

## Decision

Make Rust the production owner of table-plane calibration solving. Its typed
solver accepts exactly four finite, ordered image-space corners, positive
table dimensions in millimeters, a calibration revision, camera profile ID,
rectification revision, image space, and source image dimensions. Corner 0 is
the image-space top-left; corners continue clockwise. The resulting v1
homography maps those corners to the center-origin table frame in millimeters
and must pass the existing structural, numerical, and exact-frame-context
validators before use.

Missing or incompatible provenance leaves table coordinates unavailable.
At the time of this decision, Python could retain the old exporter as a
differential/reference test tool, but production capture and localization must
not depend on Python calibration math. ADR-0096 later retired that optional
reference-tool allowance. This decision covers solving only; persistence,
revision allocation, route delegation, and live calibration update semantics
are separate implementation steps and are not claimed complete by a solver
API.

## Consequences

### Positive

- Table geometry is computed by the same Rust domain that applies it to live
  observations.
- Required camera and image provenance travels with every solved bundle, so
  mismatched captures fail closed instead of producing plausible but wrong
  physical locations.
- The v1 coordinate contract and millimeter semantics stay stable for the
  existing API and training consumers.

### Negative

- The solver alone does not make legacy calibration routes or persisted
  corner settings Rust-owned; a later gateway-delegation step is required.
- Four-corner projective calibration still depends on users supplying accurate
  corners and table dimensions; solver tests do not establish physical
  measurement accuracy.

## Alternatives considered

### Keep the solver in Python

Continue using the Python calibration exporter in production and pass its
result to Rust. Rejected because it would leave table-solving behavior in the
domain runtime that ADR-0056 assigns to Rust, and because legacy calibration
data does not carry enough capture provenance to be trusted directly.

### Infer capture provenance from saved corners

Derive the camera mode, source dimensions, or rectification revision from
legacy corner settings. Rejected because those values are not reliably
represented there, and a mismatched homography can still produce plausible
coordinates.

## Verification

- Rust unit/integration tests cover exact and skewed perspective mappings,
  the v1 corner ordering and center-origin axes, millimeter units, invalid and
  degenerate corners, numerical rejection, and source-context mismatch.
- Rust calibration and public-state differential fixtures continue to validate
  the serialized v1 contract consumed by Python.
- `cargo fmt --all -- --check`, `cargo test --all-targets`, and
  `cargo clippy --all-targets -- -D warnings`.
- Physical table accuracy, persistence, route delegation, and use by the live
  supervisor require separate evidence and are not established by these tests.
