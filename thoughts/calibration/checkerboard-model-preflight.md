# Checkerboard lens-model preflight

Views: 8. Offline only; no calibration activated.

Hold out entire view from intrinsics/lens fit; estimate its pose on checkerboard parity 0, score parity 1; swap and concatenate.

Other corners of the held-out image fit pose, so this tests projection consistency with unseen poses, not independently measured table XY or ball centers. Neighboring corner errors may be correlated.

| Model | Training RMS px | Whole-view holdout RMS px | Scored folds | Fold instability |
|---|---:|---:|---:|---|
| zero | 0.2015 | 0.2148 | 8/8 | False |
| k1 | 0.1916 | 0.2190 | 8/8 | False |
| k1k2 | 0.1915 | 0.2209 | 8/8 | False |

At eight views, neither radial model improves whole-view held-out error over the zero-distortion baseline. Added radial correction is not supported by this experiment. Full eight-view parameters are frozen separately in checkerboard-eight-view-baseline.json without additional tuning. A prospective left-middle tilted ninth view will be scored with those coefficients fixed; only its board pose will be estimated. No prospective result is available until that image is captured. A single new view cannot establish table or ball-center accuracy.

## zero

Full-fit plausibility checks: no configured bound violations (not proof of accuracy).

Fold parameter ranges (fx, fy, cx, cy), pixels:

- Minimum: [692.238377835272, 691.4411861050147, 640.0182477901004, 348.22257775012287]
- Maximum: [699.3142585656093, 698.2480290418505, 650.0736584007687, 357.2482950595146]

| Held-out view | Validation RMS px | Training-fit warnings |
|---|---:|---|
| view-001 | 0.1355 | none |
| view-002 | 0.1886 | none |
| view-003 | 0.2274 | none |
| view-004 | 0.2234 | none |
| view-005 | 0.2780 | none |
| view-006 | 0.2285 | none |
| view-007 | 0.2630 | none |
| view-008 | 0.1217 | none |

## k1

Full-fit plausibility checks: no configured bound violations (not proof of accuracy).

Fold parameter ranges (fx, fy, cx, cy), pixels:

- Minimum: [687.2299498879312, 683.9004467745726, 645.5657897667838, 347.0376526866442]
- Maximum: [763.3756444817618, 760.3631682438593, 698.855019674651, 353.25713989518937]

| Held-out view | Validation RMS px | Training-fit warnings |
|---|---:|---|
| view-001 | 0.1447 | none |
| view-002 | 0.1616 | none |
| view-003 | 0.2057 | none |
| view-004 | 0.2007 | none |
| view-005 | 0.2624 | none |
| view-006 | 0.2837 | none |
| view-007 | 0.2632 | none |
| view-008 | 0.1881 | none |

## k1k2

Full-fit plausibility checks: no configured bound violations (not proof of accuracy).

Fold parameter ranges (fx, fy, cx, cy), pixels:

- Minimum: [689.9003120629757, 686.5932050844692, 646.9924871592774, 346.19125772826357]
- Maximum: [772.2752966233728, 768.7728399247889, 701.3513929870986, 352.38636074462863]

| Held-out view | Validation RMS px | Training-fit warnings |
|---|---:|---|
| view-001 | 0.1510 | none |
| view-002 | 0.1614 | none |
| view-003 | 0.2075 | none |
| view-004 | 0.2009 | none |
| view-005 | 0.2626 | none |
| view-006 | 0.2895 | none |
| view-007 | 0.2634 | none |
| view-008 | 0.1899 | none |

## Limits and next capture

- Small target and limited tilt diversity can make intrinsics weakly identified despite low pixel residuals.
- Physical pitch/flatness uncertainty, JPEG and corner detection systematic errors are not estimated.
- No table millimeter or ball-center accuracy is established; no Z ball output is introduced.
- No corrected-image preview is emitted: parameter plausibility alone does not justify a visually persuasive warp.
- Model comparison is preflight evidence, not permission to select the lowest residual or activate a calibration.

Most useful next capture: an additional clear tilt about the other board axis if not already present, then a tilted board toward an image edge. Keep full pattern and white border visible. Prefer larger native corner spacing where practical without altering camera settings. Repeat a pose to probe corner repeatability. Orthogonal tilted views must remain represented in training folds; losing the only tilted view can invalidate that fold.

Reproduce with the isolated OpenCV/NumPy environment:

```sh
/tmp/billiards-checkerboard-env/bin/python thoughts/calibration/analyze_checkerboard_session.py --self-test
/tmp/billiards-checkerboard-env/bin/python thoughts/calibration/analyze_checkerboard_session.py
```

Zero, k1 and k1+k2 models fix all tangential and higher-order terms. No extra tangential/fisheye model is searched on this small session. Local linear standard deviations are diagnostic approximations, not confidence guarantees. Raw JSON preserves coefficients, per-fold diagnostics and thresholds.
