# ADR-0099: Require a confirmed table appearance reference for guidance

- Status: superseded
- Date: 2026-10-24
- Deciders: Project maintainers
- Supersedes: None
- Superseded by: ADR-0114 (mandatory appearance-reference qualification only)

## Context

The configured homography describes geometry but does not prove that the table
is currently visible or that the camera still matches the calibration. Empty
detections can be consistent while the playing surface is covered or displaced.

## Decision

Capture a bounded visual reference from the exact preview confirmed during table
calibration and persist it atomically with the calibration. Validate live table
visibility against this immutable reference independently of motion settlement.
Missing or uncertain visibility evidence invalidates guidance. Existing
calibration records without a reference remain readable but require operator
confirmation before guidance is enabled.

## Consequences

### Positive

- A quiet detector output cannot by itself establish that the calibrated table
  remains visible.
- Reference provenance follows the calibration revision and source geometry.

### Negative

- Calibration confirmation stores additional bounded reference data.
- Existing users must confirm a new reference before receiving guidance.
- Appearance matching requires qualification across lighting and table fabrics.

## Alternatives considered

### Treat static calibration and motion quietness as table visibility

This cannot distinguish a stationary obstruction or shifted camera from an
empty table.

### Add a trained table detector

This requires a representative labeled dataset and model lifecycle. It is not
required for the first fail-closed reference comparison.

## Verification

- Store migration tests cover legacy records, atomic reference persistence,
  calibration changes, and missing-reference behavior.
- Visibility tests cover empty, covered, dark, displaced, changed-lighting, and
  low-quality reference cases.
- Reviewed recordings confirm no false guidance in unavailable-table cases.
