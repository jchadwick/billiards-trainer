# ADR-0103: Resolve deterministic native ball geometry through exact frame observations

- Status: accepted
- Date: 2026-10-29
- Deciders: Project maintainers
- Supersedes: None
- Superseded by: None

## Context

ADR-0102 selects native pixels, Intel JPEG decoding and Coral proposals. Its
historical follow-ups record zero accepted measured centers and a twelve-point
calibration restriction. Saved-image work now identifies concrete search,
convergence, crop and deadline defects. Those findings change the implementation,
not the fixed 2D table model or the need to report uncertainty honestly.

## Decision

Keep ADR-0102's hardware and frame ownership architecture. Replace random boundary
search with bounded deterministic cyclic contour paths. Resolve converged solutions
in common source coordinates; allow inlier membership to settle without declaring
geometrically convergent fits ambiguous. Include the complete background sampling
footprint inside a bounded source ROI. Keep actual pixel-support, residual and
ambiguity gates; a common physical diameter alone does not prove a center.

Carry the accepted source contour and center together through tracking and public
state. Draw that contour only on its exact observed session/frame. Clear it when
an observation becomes stale; never draw a retained polygon on a newer image.

Permit six or more unique, distributed ball-center correspondences, with validated
full and every leave-one-out fit, conditioning, pole checks and explicit coverage
and extrapolation reports. Nine existing references can therefore build a
revision-bound diagnostic mapping without invented points. This does not certify
physical accuracy: their measurement uncertainty is unknown. Recompute every
asset after estimator revisions; do not relabel old observations.

Use configurable 150 ms batch and 30 ms proposal budgets as the measured local
starting point. The old 25/10 ms limits discarded otherwise qualified observations.
These bounds do not choose a production FPS. Begin target comparison at 5 FPS and
prefer reliable availability over nominal speed. No additional production mode is
introduced.

## Consequences

### Positive

- Real saved detector proposals now yield reproducible measured centers and a
  complete detector-to-tracking-to-X/Y replay, rather than report-only candidates.
- Exact fitted outlines reach the existing display without violating frame pairing.
- Calibration exposes held-out consistency and extrapolation instead of treating
  a point count or reported training residual as proof of accuracy.

### Negative

- Some pocket observations still lack sufficient visible boundary support.
- Local timing is not target Coral/GPU throughput; target qualification remains.
- Existing reference consistency does not establish independent 2 mm accuracy.
- Old calibration assets are intentionally incompatible with changed refiners.

## Alternatives considered

### Increase random search or relax geometric gates

Earlier saved-image sweeps did not fix boundary selection. Relaxing support to
force pocket acceptance would conceal ambiguity instead of fixing it.

### Infer centers from a fixed pixel radius

The physical diameter is fixed, but projected contour and center bias remain.
The saved size-prior audit finds sensitivity to radius error; unsupported arcs
cannot become precise merely because a circle can be fitted.

### Require additional user measurements before implementing repairs

Existing pixels expose reproducible software defects and suffice to verify these
repairs. No new image, positioning jig or tape measurement is required.

## Verification

- Rust ball-localization, audit, profile, calibration, tracking and public-state
  tests cover deterministic search, ROI bounds, rejected geometry, revisions,
  stale outlines and exact source ownership; all-target tests and strict Clippy.
- Frontend ball-outline tests cover source identity, scaling, stale observations,
  invalid geometry and off-frame clipping; typecheck and production build.
- `thoughts/calibration/actual-detector-refiner-roi-20261029/summary.json`
  records revision 6: 13/15 accepted, including 9/9 measured references.
- `thoughts/calibration/runtime-localization-20261029/` contains reproducible
  revision-bound mappings. Full saved replay and development timing evidence
  must be reported with hardware, detector and physical-reference limitations.
