# Fisheye optics model audit — 2026-10-27

Offline evaluation only. No profile was activated, and no camera or new measurement was used.

## Held-out checkerboard comparison

The same whole-view holdout and alternating checkerboard parity pose split was used for the existing zero/Brown models and the OpenCV fisheye model. Training views were 001–008. View 009 was kept entirely out of its fisheye fit and scored prospectively by parity after fitting on 001–008.

| Model | LOO held-out RMS (px) | Successful folds | Fit / fold status |
|---|---:|---:|---|
| Brown zero | 0.2148 | 8/8 | existing solver |
| Brown k1 | 0.2190 | 8/8 | existing solver |
| Brown k1k2 | 0.2209 | 8/8 | existing solver |
| Fisheye theta polynomial (4 coefficients) | unavailable | 0/8 | 8 failed/implausible folds |

The fisheye model fit is **blocked/unstable and inadmissible as a correction candidate**. All eight matched leave-one-view-out fits have enormous training residuals, and the independent prospective score is worse than the frozen Brown scores. No further coefficient tuning was attempted. Exact coefficients, per-view errors, and parameter ranges are in `summary.json`.

| Held-out training view | Fisheye training RMS (px) | Result |
|---|---:|---|
| view-001 | 1002.8574 | fit failed: training_residual_exceeds_diagnostic_plausibility_bound |
| view-002 | 437.3182 | fit failed: training_residual_exceeds_diagnostic_plausibility_bound |
| view-003 | 458.1287 | fit failed: training_residual_exceeds_diagnostic_plausibility_bound |
| view-004 | 468.8359 | fit failed: training_residual_exceeds_diagnostic_plausibility_bound |
| view-005 | 473.1968 | fit failed: training_residual_exceeds_diagnostic_plausibility_bound |
| view-006 | 548.3904 | fit failed: training_residual_exceeds_diagnostic_plausibility_bound |
| view-007 | 532.4913 | fit failed: training_residual_exceeds_diagnostic_plausibility_bound |
| view-008 | 460.1670 | fit failed: training_residual_exceeds_diagnostic_plausibility_bound |

The OpenCV 5.0.0 Python binding exposes no `cv2.fisheye.CALIB_*` constants, so the documented numeric C++ flag values were used. The returned camera matrix nevertheless has non-zero skew despite `FIX_SKEW`; the flag/API contract is therefore unresolved in this environment. Together with the failed residuals, this blocks interpretation of the model fit. The synthetic projection/undistortion self-test passes but does not validate calibration flags.

## Prospective held-out view 009

Fisheye model fit on views 001–008 only; `view-009` was not included in the fit. Held-out parity RMS: 1.3266 px. Fit issues: training_residual_exceeds_diagnostic_plausibility_bound.

Existing frozen Brown-model validation scores on view-009: zero=0.2279px, k1=0.2307px, k1k2=0.2299px

## Same-point cushion diagnostics

All existing cushion point samples were reused without re-extraction. The following table compares ranges across the source report's smoothing and band-shift variants. These visible transitions are not surveyed physical rails and extraction variants are correlated.

| Image | Feature | Raw bow range (px) | Frozen zero bow range (px) | Fisheye bow range (px) | Fisheye RMS range (px) | Same point counts |
|---|---|---:|---:|---:|---:|---|
| view-009.jpg | top-left | 0.72–0.77 | 0.72–0.77 | 9.78–9.88 | 3.24–3.25 | True |
| view-009.jpg | top-right | 0.46–0.54 | 0.46–0.54 | 9.58–9.69 | 3.16–3.19 | True |
| view-009.jpg | bottom-left-nose | 0.24–0.99 | 0.24–0.99 | 1.52–12.48 | 0.51–3.32 | False |
| view-009.jpg | bottom-right-nose | 0.49–0.72 | 0.49–0.72 | 7.23–8.85 | 2.53–2.58 | False |
| view-009.jpg | left-nose | 0.23–0.26 | 0.23–0.26 | 11.00–11.03 | 2.70–2.71 | True |
| view-009.jpg | right-nose | 0.75–4.74 | 0.75–4.74 | 3.16–13.36 | 1.57–4.07 | False |
| view-009.jpg | bottom-left-seam | 0.38–0.52 | 0.38–0.52 | 8.35–8.51 | 2.76–2.82 | True |
| view-009.jpg | bottom-right-seam | 0.40–0.43 | 0.40–0.43 | 8.36–8.43 | 2.69–2.70 | True |
| table-grid-20261026-114241.jpg | top-left | 0.51–0.56 | 0.51–0.56 | 9.06–9.59 | 3.13–3.18 | False |
| table-grid-20261026-114241.jpg | top-right | 0.57–0.60 | 0.57–0.60 | 9.73–9.75 | 3.20–3.22 | True |
| table-grid-20261026-114241.jpg | bottom-left-nose | 0.28–0.40 | 0.28–0.40 | 3.27–7.62 | 1.03–2.56 | False |
| table-grid-20261026-114241.jpg | bottom-right-nose | 0.13–0.42 | 0.13–0.42 | 6.77–8.86 | 2.44–2.50 | False |
| table-grid-20261026-114241.jpg | left-nose | 0.33–0.35 | 0.33–0.35 | 11.10–11.12 | 2.72–2.73 | True |
| table-grid-20261026-114241.jpg | right-nose | 2.76–3.29 | 2.76–3.29 | 8.37–14.80 | 2.83–3.98 | False |
| table-grid-20261026-114241.jpg | bottom-left-seam | 0.36–0.53 | 0.36–0.53 | 8.34–8.51 | 2.75–2.82 | True |
| table-grid-20261026-114241.jpg | bottom-right-seam | 0.40–0.45 | 0.40–0.45 | 8.37–8.45 | 2.69–2.71 | True |

## Interpretation and limits

- Do not activate this fisheye fit or its point transform. The candidate is blocked/unstable and the cushion transform is shown only to document consequences of applying the failed fit.
- The checkerboard and cushion comparisons assess projection consistency and visible-edge geometry only; neither establishes table X/Y accuracy or the true physical rail nose.
- View-009 is prospective: it is scored only after coefficients are frozen from views 001–008. It is not used in any fit that reports its score.
- The board views are small with limited tilt diversity; fisheye fold instability is evidence the current saved data may not identify the added coefficients, not proof the lens has no fisheye distortion.
- The test is offline and leaves the production zero-radial optics profile unchanged. The ball-center refiner still rejects all nine measured positions; no accepted centers or map are produced.
- No ball height, depth, or Z output is introduced.

## Reproduction

```sh
/tmp/billiards-checkerboard-env/bin/python thoughts/calibration/offline-fisheye-optics-audit-20261027.py --self-test
/tmp/billiards-checkerboard-env/bin/python thoughts/calibration/offline-fisheye-optics-audit-20261027.py
```

The raw JSON records exact input and source-code hashes, all fold residuals, parameter ranges, point-transform metrics, and software version.
