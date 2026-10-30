# Source-only checkerboard observations

All views require independent pixel review. Machine agreement is not corner accuracy. Window disagreement is retained as a flag; source-bound review can select a visibly supported existing refinement. Missing intersection support or invalid geometry cannot be bypassed. Pending review is ineligible for scoring. Sources and raw coordinates are preserved.

| View | Machine-supported corners | Maximum window spread px | Scoring |
|---|---:|---:|---|
| view-001 | 49/54 | 1.291 | independent_pixel_review_pending, machine_validation_rejected |
| view-002 | 25/54 | 2.447 | independent_pixel_review_pending, machine_validation_rejected |
| view-003 | 52/54 | 0.647 | independent_pixel_review_pending, machine_validation_rejected |
| view-004 | 50/54 | 0.967 | independent_pixel_review_pending, machine_validation_rejected |
| view-005 | 46/54 | 20.031 | independent_pixel_review_pending, machine_validation_rejected |
| view-006 | 51/54 | 1.330 | independent_pixel_review_pending, machine_validation_rejected |
| view-007 | 45/54 | 1.554 | independent_pixel_review_pending, machine_validation_rejected |
| view-008 | 45/54 | 0.846 | independent_pixel_review_pending, machine_validation_rejected |

Raw and largest-window contact sheets show every source corner. Tiny source crops are enlarged only for display with nearest-neighbor interpolation (minimum 64-pixel cells); all measurements retain native source coordinates. Red crosses are original observations, green crosses show the largest declared refinement window. `FLAG` indicates at least one image-support, identity or window-agreement rejection; `OK?` is only machine support pending review.

Quadrant contrast and consistency are appearance heuristics. Paper seams, blur, shadows and shared localization bias can invalidate them. No image-derived number here establishes physical target scale, flatness or a true location uncertainty.

Original images, capture manifests, original corners and camera models remain unchanged. JSON stores every neighborhood result, quadrant sample, per-corner rejection and review state. A frozen-model evaluator requires reviewed source-bound observations and at least eight supported corners in each parity with spatial coverage. Every excluded corner remains in the denominator and record. Reviewer selections cannot depend on camera residuals. Review records bind source hash, revision and configuration.
