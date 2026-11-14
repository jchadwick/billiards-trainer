# ADR-0103: Resolve deterministic native ball geometry through exact frame observations

- Status: accepted
- Date: 2026-10-29
- Deciders: Project maintainers
- Supersedes: None
- Superseded by: None

## Context

When ADR-0103 was proposed, ADR-0102 assigned proposals to Coral and selected
native pixels and Intel JPEG decoding. ADR-0104 later replaced only that
inference assignment with explicit Intel GPU inference. ADR-0102's historical
follow-ups record earlier revisions with zero accepted measured centers and a
twelve-point calibration restriction. Saved-image work now identifies concrete
search, convergence, crop and deadline defects. Those findings change the
implementation, not the fixed 2D table model or the need to report uncertainty
honestly.

## Decision

Keep ADR-0102's native decoding and frame ownership architecture; use the
inference provider selected by the later ADR-0104. Replace random boundary
search with bounded deterministic cyclic contour paths. Resolve converged solutions
in common source coordinates; allow inlier membership to settle without declaring
geometrically convergent fits ambiguous. Include the complete background sampling
footprint inside a bounded source ROI. Keep actual pixel-support, residual and
ambiguity gates; a common physical diameter alone does not prove a center.

Carry the accepted source contour and center together through tracking and public
state for provenance. As narrowed by ADR-0114, contours are optional
developer-only diagnostics behind an explicit flag, drawn only on their exact
observed session/frame and cleared when stale. Production uses the single
table-plane center/radius marker; it does not require contour display or exact
video-frame pairing for centers. Never draw a retained polygon on a newer image.

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
- Local timing is not simultaneous target decode/GPU inference throughput;
  target qualification remains.
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

## Follow-up — 2026-10-29: schedule once per source frame and retain the corner observation

The live scheduler now consumes work cadence once for each captured source
frame. Motion and asynchronous cue observations refresh phase/evidence without
calling a second scheduling tick, and a receive timeout advances evidence
expiry without consuming work due for the next frame. Regression tests cover
the timeout and asynchronous cue-completion boundary. This fixes work being
skipped even when a source frame was available; it does not increase source
cadence or change the preference for fresh measurements over nominal FPS.

The R10 localization default now permits a maximum seed-center spread of 1.6
reference pixels. In the exact reconstructed far-corner box
`[250.875, 270.75, 367.875, 371.25]`, that value accepts seven agreeing seed
fits, where 1.5 rejected the proposal as `unstable_center`. The proposal's
confidence was 0.8677 against a 0.3 detector threshold, so detector confidence
was not changed and was not the cause of the missing coordinate. The paired
saved-corpus audit at
`thoughts/calibration/refiner-r10-validation-20261029/paired-center-spread-audit-r10.json`
found no changed outcomes on its 15 actual GPU proposals, 18 proposal-seed
variants, or 297 synthetic known-center cases. These checks support this
availability setting for the observed case only; the 1.524-pixel seed spread
is near the new threshold and is not a center-error guarantee. The accepted
center has no independent physical reference.
