# Native crop refiner check on saved nine-position scene

This is a diagnostic run, not calibration or accuracy evidence.

## Input

- Source: `table-grid-20261026-114241.jpg`, 1280 × 720.
- Intrinsics: exact zero-radial full-fit matrix from
  `checkerboard-eight-view-baseline.json`, used only for this offline check.
- Proposals: manually seeded 36 × 36 pixel boxes around the nine centers from
  `table-grid-feasibility-results.json`; these are not current YOLO boxes.
- Refiner: production Rust implementation and default thresholds. Its source
  pixel gates scale from the 3840-pixel reference width, so this 1280-pixel
  image is tested at one-third the native 4K pixel thresholds.

## Result and ambiguity-gate correction

With the initial ambiguity rule, the 720p frame accepted 1 of 9 proposals.
Inspection found that one high-support RANSAC sample at a distant center was
enough to reject a fit. The rule now requires a spatial cluster of at least
three plausible competing samples. It still rejects a repeatable alternate
center; it ignores an isolated random alternative. The subsequent review found
that these remain raw three-point hypotheses rather than distinct converged
contour fits, so the correction is provisional.

After that correction, the 720p frame accepted 4 of 9 proposals at default
pixel scaling. A separate 4K framing-verification image with three balls
accepted 3 of 3 manually seeded crops. Their final fit residuals were
0.33–0.42 px and quadrant-removal spreads were 0.10–0.24 px. One crop had
partial contour coverage (78% of rays) near a rail and passed the current
coverage gate.

## Interpretation

The 2026-10-27 review found that quality fields can retain values from an
earlier aiming iteration when a later iteration rejects. Therefore a low
reported residual/spread beside an ambiguity rejection does not establish a
stable final rejected fit. The all-ball check with different proposals accepts
5/9 measured balls; those four measured rejections are all ambiguity failures.
Holding each crop fixed and varying only detection index changes acceptance
for seven of the nine measured balls. These experiments are recorded in
`review-and-next-step-20261027.md` and `offline-refiner-identity-check.json`.
The 4/9 and 5/9 runs differ in proposals and indices and are not a clean
before/after comparison.

The 720p and 4K checks use manual proposals, not current YOLO boxes. The 4K
check uses a 3× scaled checkerboard intrinsics matrix, not a qualified native
4K optics profile, and its three balls are not the nine measured positions.
Neither run validates physical centers or X/Y accuracy. They show that the
one-sample ambiguity rule affected acceptance, and that
the revised rule accepts the saved 4K crops without bypassing final
residual, coverage, or quadrant-stability checks. Continue qualification on
matched native frames and do not tune gates solely to increase acceptance.
