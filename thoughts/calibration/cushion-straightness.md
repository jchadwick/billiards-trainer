# Cushion-associated edge straightness

Offline diagnostic only. Frozen eight-view lens candidates are compared by transforming points; source images, runtime settings and calibration stay unchanged.

ROIs were inspected on original images and enlarged rail strips. The bottom inner transition and stronger outer seam are reported separately. Pockets, balls and the left-middle checkerboard are excluded. Colored pixels in the PNG show the nominal extracted samples on unwarped images.

## view-009.jpg

Ranges below span three smoothing scales and search bands shifted ±2px. They quantify extraction sensitivity, not confidence bounds.

| Feature | Raw RMS px | Raw smooth bow px | k1 RMS px | k1 smooth bow px | k1+k2 RMS px |
|---|---:|---:|---:|---:|---:|
| top-left | 0.28–0.30 | 0.72–0.77 | 0.73–0.74 | 2.46–2.50 | 0.72–0.74 |
| top-right | 0.24–0.26 | 0.46–0.54 | 0.66–0.68 | 2.27–2.35 | 0.66–0.68 |
| bottom-left-nose | 0.19–1.88 | 0.24–0.99 | 0.19–1.90 | 0.30–2.56 | 0.19–1.90 |
| bottom-right-nose | 0.26–0.37 | 0.49–0.72 | 0.49–0.65 | 1.62–2.15 | 0.48–0.65 |
| left-nose | 0.12–0.14 | 0.23–0.26 | 0.87–0.89 | 3.54–3.57 | 0.87–0.88 |
| right-nose | 1.46–2.65 | 0.75–4.74 | 1.73–2.58 | 2.54–5.99 | 1.73–2.58 |
| bottom-left-seam | 0.14–0.20 | 0.38–0.52 | 0.61–0.65 | 2.11–2.23 | 0.60–0.65 |
| bottom-right-seam | 0.14–0.15 | 0.40–0.43 | 0.61–0.62 | 2.15–2.18 | 0.61–0.62 |

- bottom-left-nose: Weak edge: fewer than65% of planned samples survive in some variants.

- bottom-left-nose: Ambiguous/local edge jumps: do not use this feature alone to choose a lens model.

- right-nose: Weak edge: fewer than65% of planned samples survive in some variants.

- right-nose: Ambiguous/local edge jumps: do not use this feature alone to choose a lens model.

## table-grid-20261026-114241.jpg

Ranges below span three smoothing scales and search bands shifted ±2px. They quantify extraction sensitivity, not confidence bounds.

| Feature | Raw RMS px | Raw smooth bow px | k1 RMS px | k1 smooth bow px | k1+k2 RMS px |
|---|---:|---:|---:|---:|---:|
| top-left | 0.23–0.91 | 0.51–0.56 | 0.68–0.98 | 2.26–2.31 | 0.68–0.98 |
| top-right | 0.26–0.29 | 0.57–0.60 | 0.70–0.71 | 2.38–2.40 | 0.70–0.71 |
| bottom-left-nose | 0.17–0.25 | 0.28–0.40 | 0.20–0.60 | 0.48–2.01 | 0.20–0.59 |
| bottom-right-nose | 0.22–0.23 | 0.13–0.42 | 0.46–0.58 | 1.50–1.94 | 0.45–0.58 |
| left-nose | 0.14–0.14 | 0.33–0.35 | 0.89–0.90 | 3.63–3.65 | 0.89–0.90 |
| right-nose | 1.67–2.58 | 2.76–3.29 | 1.86–2.59 | 4.26–6.18 | 1.86–2.59 |
| bottom-left-seam | 0.14–0.20 | 0.36–0.53 | 0.60–0.65 | 2.09–2.25 | 0.60–0.65 |
| bottom-right-seam | 0.14–0.15 | 0.40–0.45 | 0.61–0.62 | 2.15–2.20 | 0.61–0.62 |

- top-left: Ambiguous/local edge jumps: do not use this feature alone to choose a lens model.

- bottom-left-nose: Weak edge: fewer than65% of planned samples survive in some variants.

- right-nose: Weak edge: fewer than65% of planned samples survive in some variants.

- right-nose: Ambiguous/local edge jumps: do not use this feature alone to choose a lens model.

## Interpretation and limits

The raw-versus-candidate differences are a model-adequacy diagnostic. If a candidate consistently increases smooth bow on independently observed rail transitions, that argues against applying that candidate; it does not identify the cause of ball elongation. Nose and seam disagreement or unstable extraction calls for reviewing the annotated evidence, not selecting whichever edge favors a lens model.

- Observed image edges may trace shadow, cloth texture or a seam instead of the exact physical nose; their straightness is not a physical survey.
- Search-band and smoothing variants measure extraction sensitivity, not physical confidence intervals; these variants are correlated.
- Long segments split around pockets and balls; separate rail sections are not assumed collinear. The left-board region is excluded.
- Point-undistortion comparison reuses exactly the same extracted raw samples for every candidate; no refitting lens coefficients or image warping.
- Camera mode matching is assumed for these two1280x720 images; sources are hashed. No millimeter ball-center accuracy follows.
- Small residuals do not prove zero lens distortion; larger smooth bow can reflect extraction bias or physical edge geometry.

In these two images, both top rail sections, the left inner transition and the independent bottom outer seams become more bowed under the frozen positive-k1 candidate; k1+k2 behaves similarly. This is complementary evidence against applying these particular frozen radial corrections. The right-side extraction is ambiguous and the view-009 bottom-left nose transition has weak support, so neither is treated as decisive. The zero-distortion candidate leaves points unchanged; its metrics match raw coordinates to numerical precision.

Keep production geometry unchanged. Coordinate calibration and actual ball-outline refinement remain separate unresolved work; these edge diagnostics do not prove zero residual lens distortion, physical cushion straightness or accurate ball X/Y.

## Reproduce

```sh
/tmp/billiards-checkerboard-env/bin/python thoughts/calibration/analyze_cushion_straightness.py --self-test
/tmp/billiards-checkerboard-env/bin/python thoughts/calibration/analyze_cushion_straightness.py
```

Synthetic tests cover a true straight line, known radial distortion, successful point correction, worsened straightness under wrong correction, and noisy sloped-edge extraction.
