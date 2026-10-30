# Saved-image robustness audit — 2026-10-27

This is a controlled sensitivity check of the current production refiner on
saved images. The detailed findings, exact hashes, candidate traces, and limits
are in [interpretation.md](interpretation.md) and [summary.json](summary.json).

| Experiment | Runs | Outcome |
|---|---:|---|
| Proposal center offsets, X then Y | 150 | 1 accepted result; all other variants rejected. |
| Proposal width/height factors | 150 | 0 accepted results. |
| RANSAC seeds on the 15 measured-scene boxes | 240 | 5 accepted observations across 4 of 16 seeds; production seed 716 accepted 0/15. |
| Alternate 720p scene | 15 | 0 accepted; 13 ambiguous, 1 inadequate coverage, 1 insufficient boundary. |
| Native 4K three-ball scene | 3 | 0 accepted; all 3 ambiguous. |
| Batch/proposal budget settings | 7 batches | The 25/10 ms defaults timed out all 15 proposals in the development-host run; 1000/1000 and 10000/10000 ms completed them in 287.6 and 289.4 ms. |

The two reserved-scene JPEGs now have manually reviewed approximate crop
regions in
[`reserved-scene-coarse-proposals-20261027.json`](../reserved-scene-coarse-proposals-20261027.json).
Those boxes are not detector outputs or center labels. Full-source and
per-proposal overlays, plus complete per-proposal candidate traces, are in
`reserved-overlays/` and `reserved-cases/`.

The six sensitivity variants that happened to pass the estimator's gates are
also preserved with full traces and crop overlays in
`accepted-variant-traces/` and `accepted-variant-overlays/`. All six crops were
visually reviewed. Their outlines overlap visible ball boundaries, but the
accepted fits depend on fitting seed or proposal shift and have no independent
image-center truth; they are not selected as calibration observations.

All rates and timings here are diagnostic outputs from saved images on the
development host. This audit does not establish millimeter accuracy, qualify
the target camera pipeline, or justify changing the production fitting seed.
