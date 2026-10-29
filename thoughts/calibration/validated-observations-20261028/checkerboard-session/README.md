# Source-only checkerboard observations

All views require independent pixel review. Machine agreement is not corner accuracy. Window disagreement is retained as a flag; source-bound review can select a visibly supported existing refinement. Missing intersection support or invalid geometry cannot be bypassed. Pending review is ineligible for scoring. Sources and raw coordinates are preserved.

| View | Machine-supported corners | Maximum window spread px | Scoring |
|---|---:|---:|---|
| view-001 | 10/54 | 1.074 | independent_pixel_review_pending, machine_validation_rejected |
| view-002 | 4/54 | 1.317 | independent_pixel_review_pending, machine_validation_rejected |
| view-003 | 8/54 | 1.335 | independent_pixel_review_pending, machine_validation_rejected |
| view-004 | 4/54 | 1.378 | independent_pixel_review_pending, machine_validation_rejected |
| view-005 | 0/54 | 1.717 | independent_pixel_review_pending, machine_validation_rejected |
| view-006 | 2/54 | 1.095 | independent_pixel_review_pending, machine_validation_rejected |
| view-007 | 2/54 | 1.071 | independent_pixel_review_pending, machine_validation_rejected |
| view-008 | 12/54 | 1.027 | independent_pixel_review_pending, machine_validation_rejected |
| view-009 | 5/54 | 1.632 | independent_pixel_review_pending, machine_validation_rejected |

Raw and largest-window contact sheets show every source corner. Tiny source crops are enlarged only for display with nearest-neighbor interpolation (minimum 64-pixel cells); all measurements retain native source coordinates. Red crosses are original observations, green crosses show the largest declared refinement window. `FLAG` indicates at least one image-support, identity or window-agreement rejection; `OK?` is only machine support pending review.

Quadrant contrast and consistency are appearance heuristics. Paper seams, blur, shadows and shared localization bias can invalidate them. No image-derived number here establishes physical target scale, flatness or a true location uncertainty.

Original images, capture manifests, original corners and camera models remain unchanged. JSON stores every neighborhood result, quadrant sample, per-corner rejection and review state. A frozen-model evaluator requires reviewed source-bound observations and at least eight supported corners in each parity with spatial coverage. Every excluded corner remains in the denominator and record. Reviewer selections cannot depend on camera residuals. Review records bind source hash, revision and configuration.
