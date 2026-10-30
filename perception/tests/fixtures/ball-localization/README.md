# Frozen pocket/rail RGB fixtures

These are lossless 128 × 128 crops of the saved measured-table photograph
`thoughts/calibration/table-grid-20261026-114241.jpg` (SHA-256
`06ef40f52e0f93db17601585c4ed1d014895aaadcaf6b04c6abbd6e986d57e7b`).
They were decoded once with the production Rust `image` dependency, then cropped
and saved as PNG without resizing, smoothing, recoloring, or annotation.

- `top-left-pocket.png`: original pixel origin (39, 43).
- `bottom-right-pocket.png`: original pixel origin (1126, 564).

The regression uses the actual target Intel GPU proposal boxes, translated into
crop coordinates. Intrinsics are translated by the same origins. Configuration
reference width is 384 so that the 128-pixel crops retain exactly the original
1280/3840 pixel-threshold scaling. All coordinates and crop origins are integral
multiples of the configured source lattice spacing.

These fixtures establish full-gate source-pixel repeatability and agreement
across proposal perturbations. They establish no true physical ball center or
independent millimeter accuracy. Known-center synthetic tests cover estimator
bias, highlights, shadows, partial contours, and cloth/rail backgrounds.
