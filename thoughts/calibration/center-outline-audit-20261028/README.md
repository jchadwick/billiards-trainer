# Center versus outline audit — 2026-10-28

This report reads frozen traces only. It changes no refiner, threshold, calibration, camera or deployment.

Across 33 baseline cases, all observations remain rejected. Of 30 aiming iterations with at least two individually qualified fits, 0 have center spread within the existing source-space threshold while their sampled outlines disagree beyond it.

The threshold is the existing 1.5 px at width 3840 (0.5 px at width 1280); this is a diagnostic classification, not a newly validated uncertainty or acceptance bound.

| Scene / ball | Iteration | Fits | Center max px | Outline max px | Radius median range px | Stable centers / differing outlines |
|---|---:|---:|---:|---:|---:|---|
| measured-grid-720 / bottom-black-second-left-ball | 1 | 3 | 1.781 | 2.589 | 0.862 | False |
| measured-grid-720 / bottom-left-pocket-ball | 1 | 4 | 1.744 | 2.034 | 0.954 | False |
| measured-grid-720 / bottom-middle-pocket-ball | 1 | 6 | 1.458 | 1.569 | 0.638 | False |
| measured-grid-720 / bottom-right-pocket-ball | — | 0 | — | — | — | None |
| measured-grid-720 / bottom-yellow-second-right-ball | 1 | 4 | 1.569 | 2.004 | 0.450 | False |
| measured-grid-720 / far-left-yellow-middle-row-ball | 1 | 6 | 1.507 | 2.208 | 1.358 | False |
| measured-grid-720 / far-right-purple-middle-row-ball | 1 | 2 | 0.641 | 1.240 | 0.636 | False |
| measured-grid-720 / interior-green-second-right-middle-row-ball | 1 | 1 | — | — | 0.000 | None |
| measured-grid-720 / interior-green-second-right-middle-row-ball | 2 | 2 | 0.579 | 0.666 | 0.087 | False |
| measured-grid-720 / interior-purple-second-left-middle-row-ball | 1 | 4 | 0.979 | 1.381 | 0.750 | False |
| measured-grid-720 / middle-row-center-ball | 1 | 4 | 0.778 | 1.129 | 0.663 | False |
| measured-grid-720 / top-left-pocket-ball | — | 0 | — | — | — | None |
| measured-grid-720 / top-middle-pocket-ball | 1 | 2 | 0.591 | 0.958 | 0.349 | False |
| measured-grid-720 / top-orange-solid-second-left-ball | 1 | 2 | 1.658 | 1.665 | 0.011 | False |
| measured-grid-720 / top-red-striped-second-right-ball | 1 | 8 | 1.709 | 2.072 | 1.319 | False |
| measured-grid-720 / top-right-cue-pocket-ball | 1 | 2 | 0.628 | 0.868 | 0.227 | False |
| framing-after-4k / 4k-top-center-striped | 1 | 6 | 4.286 | 5.642 | 3.924 | False |
| framing-after-4k / 4k-top-left-orange | 1 | 2 | 3.578 | 4.542 | 0.936 | False |
| framing-after-4k / 4k-top-right-red-stripe | 1 | 4 | 4.304 | 5.602 | 3.111 | False |
| alternate-grid-720 / alt-bottom-black | 1 | 2 | 0.530 | 0.976 | 0.437 | False |
| alternate-grid-720 / alt-bottom-center-red-stripe | — | 0 | — | — | — | None |
| alternate-grid-720 / alt-bottom-left-red | 1 | 4 | 3.104 | 4.298 | 1.018 | False |
| alternate-grid-720 / alt-bottom-right-blue-stripe | 1 | 1 | — | — | 0.000 | None |
| alternate-grid-720 / alt-bottom-right-blue-stripe | 2 | 2 | 0.894 | 1.689 | 0.734 | False |
| alternate-grid-720 / alt-bottom-yellow | 1 | 3 | 1.279 | 1.469 | 0.378 | False |
| alternate-grid-720 / alt-middle-green-stripe | 1 | 2 | 0.343 | 0.498 | 0.154 | False |
| alternate-grid-720 / alt-middle-green-stripe | 2 | 1 | — | — | 0.000 | None |
| alternate-grid-720 / alt-middle-green-stripe | 3 | 2 | 0.696 | 0.893 | 0.197 | False |
| alternate-grid-720 / alt-middle-left-purple-stripe | 1 | 4 | 1.831 | 1.937 | 0.956 | False |
| alternate-grid-720 / alt-middle-left-yellow | 1 | 2 | 0.695 | 1.326 | 0.566 | False |
| alternate-grid-720 / alt-middle-red-stripe | 1 | 3 | 0.576 | 1.268 | 0.692 | False |
| alternate-grid-720 / alt-middle-right-purple | 1 | 2 | 0.558 | 0.596 | 0.031 | False |
| alternate-grid-720 / alt-top-left-pocket | — | 0 | — | — | — | None |
| alternate-grid-720 / alt-top-orange | 1 | 3 | 1.286 | 1.437 | 0.330 | False |
| alternate-grid-720 / alt-top-red-stripe | 1 | 3 | 1.083 | 1.219 | 0.167 | False |
| alternate-grid-720 / alt-top-right-cue | 1 | 2 | 1.350 | 2.348 | 0.922 | False |
| alternate-grid-720 / alt-top-yellow-stripe | 1 | 7 | 1.739 | 1.846 | 0.898 | False |

## Same-nine-reference comparison

| Method | Eligible centers | Diagnostic leave-one-out RMS mm | p95 mm | max mm |
|---|---:|---:|---:|---:|
| existing_segmentation_ellipse_centers | 9 diagnostic observations | 8.706 | 13.299 | 13.982 |
| manual_proposal_midpoints | 9 diagnostic observations | 7.888 | 10.228 | 10.760 |
| current refiner | 0 accepted | no fit | no fit | no fit |

The two diagnostic methods use exactly the same nine physical references and source JPEG. There is no common accepted subset with the current refiner. Manual proposal centers are not preserved raw detector outputs. The older segmentation LOO result is reproduced; none of these residuals establishes independent accuracy.

## Interpretation

The saved cases do not support recovering useful measurements by removing unique-outline rejection alone: zero multi-candidate iterations have stable centers while only outlines disagree. Every iteration whose outlines exceed the existing tolerance also has a center spread exceeding that tolerance. One intermediate iteration has both center and outline spread below the threshold, but a later aiming iteration rejects with materially different centers. The evidence favors addressing search/measurement stability rather than relaxing outline qualification.

The saved perturbation audit still contains material fitting-start/proposal dependence. Its accepted outputs and rejection counts are reproduced per ball in summary.json. Only accepted perturbation traces preserve full fit geometry, so this report does not claim a bound over the rejected runs or all possible contours.

Candidate-center sensitivity is also expressed through a nine-point segmentation-derived diagnostic map for the measured cases. That map was fit on the same physical references; its millimeter spreads are conditional sensitivity, not an error bound. All current refiner outcomes remain unavailable and no calibration asset is emitted.

## Limits

- Manual proposal midpoints are not raw YOLO outputs; no saved detector batch establishes that comparison.
- Same image, repeated trials and candidate fits are not independent ball placements.
- Source center is projected local-view circle center, not a surveyed physical sphere-center label.
- All plausible contours share image bias; pairwise agreement does not bound true error.
- Candidate radius summary is distance from projected local center; off-axis projection produces varying radius and is not a physical ball diameter.
- Sampled-outline Hausdorff can include angular discretization; it is not exact continuous-curve distance.
- Current refiner has no accepted measured points; hypothetical candidates are not substituted into a production map.
- Nine references have unknown uncertainty; fitted-homography errors mix mapping, reference and center bias.
- Only three native-4K saved ball crops exist; no full-table independent native4K accuracy data.
- Optics are frozen exploratory profiles here; new optics comparison belongs to separate audit.
- Rejected perturbation runs lack full candidate traces; only outcomes and accepted-center sensitivity are available.

## Reproduction

```sh
/tmp/billiards-checkerboard-env/bin/python thoughts/calibration/offline-center-outline-audit-20261028.py --self-test
/tmp/billiards-checkerboard-env/bin/python thoughts/calibration/offline-center-outline-audit-20261028.py --inputs thoughts/calibration/center-outline-audit-20261028/inputs.json
```

The compact inputs.json bundle preserves the qualified per-iteration numerical evidence, perturbation outcomes, segmentation observations and physical references, with full source trace hashes and extraction revision. Analysis reads this explicit bundle and does not require full traces or images; image hashes were checked during extraction. Re-extract only on the recorded experiment revision using --extract-inputs /tmp/center-outline-inputs.json. The JSON preserves every input hash, all per-iteration qualified fits and pairwise distances, perturbation outcome counts, accepted variant centers, each homography holdout prediction and residual, and the exact source image identity. Synthetic checks cover concentric radius ambiguity, known center separation, shared bias despite perfect agreement, exact projective recovery/LOO, and degenerate mapping rejection.
