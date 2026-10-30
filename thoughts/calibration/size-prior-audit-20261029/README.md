# Fixed-diameter partial-boundary feasibility audit

This is a saved-data research comparison, **not a replacement estimator or an acceptance result**. Production revision 5 remains unchanged. The report uses the fifteen actual-detector trace files and provisional nine-reference 2 D calibration. The calibration has unknown physical-reference uncertainty and does not establish independently validated millimeter accuracy.

## Reproduce

From the repository root:

```sh
/tmp/billiards-checkerboard-env/bin/python thoughts/calibration/size-prior-audit-20261029/audit.py
```

`report.json` includes dependency versions, calibration hash, each input trace hash, all eight path results, and synthetic checks. Two consecutive executions produced byte-identical reports. OpenCV recovers the local-to-source projective transform from the retained observed point correspondences; each reconstruction is checked within 0.0001 source pixels. No camera, inference model, source asset, deployed configuration, or service is changed.

## Method

The fixed 57.15 mm diameter and provisional image-to-table mapping imply a local apparent radius through the calibrated projection. This calculation estimates no individual ball height and emits no Z output. Normalize the common first-two-column norm of `Q = inverse(K) * inverse(image_to_table_mm)`. For a table point, apply Q to its homogeneous XY coordinate. The local focal length and fixed physical radius then supply the projected radius.

For each first-aim coherent candidate path, bounded two-parameter Cauchy IRLS fits only the center while holding radius fixed. Repeat at nominal radius and ±4%. The four-percent sensitivity range is an exploratory perturbation motivated by the approximate agreement of predicted and fitted radii for the nine accepted measured balls; it is **not** a calibrated confidence interval. The fitter uses observed points only and creates no synthetic hidden arc. The comparison does not repeat runtime foreground/background, coverage, independent-quadrant, alternative-boundary, or final source-support qualification.

## Results and next decision

| Case | Existing runtime stage | Predicted local radius | Maximum source-center spread under ±4% radius |
|---|---|---:|---:|
| Bottom-left pocket | coverage |10.332 px |1.941 px |
| Top-left pocket | coverage |10.577 px |1.907 px |
| Bottom-right pocket | background sampling | unavailable | no recorded candidates |
| Nine accepted measured balls | complete | position-dependent |0.087–0.239 px |

The two rejected pocket fits are much more sensitive to the assumed radius than the accepted measured cases. Their center variation exceeds the existing 0.5 source-pixel center-spread threshold at 1280 px image width. Therefore the size prior alone does **not** justify accepting them. Both remain rejected. The bottom-right case is rejected before boundary generation, so fitting a known-radius circle cannot repair its missing candidate evidence.

Known-center synthetic checks cover 60 °,120 °,180 °,240 °,300 ° arcs with deterministic 0.08 px Gaussian noise. Correct-radius center errors are below 0.021 px. For the 60 ° arc, a four-percent radius error moves the inferred center by 0.409–0.430 px while residual remains only 0.084–0.088 px. A low residual is insufficient evidence of an accurate center under a radius prior.

A credible pocket improvement must first preserve background evidence at a pocket/rail boundary and establish which observed transitions are actually the ball silhouette. Replacing the current global felt background model with a local, occluder-aware boundary likelihood is an implicated architectural change; this audit does not implement or validate it. Any such change needs synthetic occlusion/rail cases, unchanged accepted-case comparisons, conservative observed-support reporting, and a source-center uncertainty check. Merely replacing free radius with the known physical diameter would manufacture confidence here.

## Follow-up: the bottom-right rejection contains an ROI geometry defect

`annulus_audit.py` independently reproduces the runtime View transform, source crop, bilinear RGB sampling, annulus median, and exact color tests. Run it with the same Python environment; output is `annulus-report.json`. Its original bottom-right result is exactly the runtime result: 115 of 192 supported samples (0.5989583), immediately below the configured 0.6 gate. Of 192 positions, 27 fall outside the artificial source ROI; **none fall outside the actual image**. The median proposal-derived local radius is 12.4838 px, versus the physical-size diagnostic prediction of 10.3432 px.

| Sampling experiment | Bottom-right valid / supported | Bottom-left valid / supported | Top-left valid / supported |
|---|---:|---:|---:|
| Original proposal radius and ROI | 165 / 115 | 178 / 130 | 177 / 162 |
| Same radius; include projected annulus outside artificial ROI | 192 / 135 | 192 / 142 | 192 / 182 |
| Physical-size radius for sampling only | 192 / 135 | 192 / 137 | 192 / 173 |

The simpler correction is therefore to construct a bounded source ROI that includes the actual projected local sampling footprint before sampling. The off-axis rotation invalidates the assumption that a bbox-padded source rectangle encloses a locally circular annulus. This repair needs neither a size prior nor a background mixture, and changes no color threshold. Its implementation must still enforce configured source ROI and patch limits and reject actual frame-outside samples. The research's unrestricted-ROI comparison isolates the cause; unrestricted sampling is not proposed as production logic.

Continuing radial tests diagnostically after the old failing background gate still finds transitions on 82 of 96 bottom-right rays, versus 90 of 96 with the physical-size sampling seed. These counts are raw supported-ray feasibility, not qualified circles. Proper crop geometry must be tested through the complete estimator before concluding the pocket can be accepted. RGB mixture changes are premature until that correction is evaluated.
