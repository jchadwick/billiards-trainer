# Candidate representative cap audit — 2026-10-27

This saved-image audit varies only `maximum_refined_candidates`: diagnostic
caps 1, 2, 4, 6, and 8, plus the production default cap 16. Every run uses
the exact frozen measured 720p image, the same exploratory zero-radial camera
matrix, the same manual annotation-derived proposal for each ball, seed 716,
128 RANSAC iterations, default three aiming iterations, all other default
settings, and 10-second batch/per-proposal diagnostic budgets.

All 18 runs rejected as `ambiguous_boundary`. The cap did not turn any of
these three saved observations into a precise center. At the production cap
16, candidate search was not truncated for any of the three proposals; their
ambiguity therefore comes from multiple distinct individually qualified
fits, not the representative cap.

| Proposal | Cap 1 | Cap 2 | Cap 4 | Cap 6 | Cap 8 | Production cap 16 |
|---|---:|---:|---:|---:|---:|---:|
| Interior purple, second-left middle row | 1 qualified / 1 evaluated; truncated | 2 / 2; truncated | 4 / 4; truncated | 4 / 5; complete | 4 / 5; complete | 4 / 5; complete |
| Top orange, second-left | 0 / 1; truncated | 0 / 2; truncated | 1 / 4; truncated | 2 / 6; truncated | 2 / 8; complete | 2 / 8; complete |
| Top red stripe, second-right | 1 / 1; truncated | 2 / 2; truncated | 3 / 4; truncated | 4 / 6; truncated | 5 / 8; truncated | 8 / 16; complete |

The pairs show individually qualified candidates over evaluated representative
fits. “Truncated” is the estimator's own `candidate_search_truncated` quality
flag. Representative evaluation order/rank, raw-hypothesis triplets and rays,
candidate fit metrics, and within-iteration pairwise source-outline Hausdorff
distances are in `summary.json`; full schema-3 event traces are in `cases/`.
All 18 traces were checked: no dropped events and no event-cap truncation.

At cap 16, the purple proposal has four fully qualified fits, the orange
proposal has two, and the red-striped proposal has eight. Their accepted
outlines form distinct source-image boundaries under the current 0.5 px
same-boundary threshold scaled to 1280 px. For example, the purple fits span
0.758–1.381 px pairwise Hausdorff distance; the red-striped fits span
0.943–2.072 px. The fits may lie on visible ball edges, but this image and its
manual proposals do not establish the true center. The physical reference
uncertainty is unknown. Do not switch caps to choose a preferred center.

These proposals are manual annotation-derived, not detector outputs. The
frozen optics model is exploratory, not a qualified runtime profile. This
audit does not establish physical accuracy, detector quality, target-device
latency, or live availability. It does show that the production cap fully
examines its bounded representative set on these three cases and preserves
the ambiguous/unavailable result.

## Reproduction

From the repository root:

```text
cp thoughts/calibration/offline-refiner-candidate-cap-audit-20261027.rs perception/examples/offline_refiner_candidate_cap_audit.rs
cargo run --locked --manifest-path perception/Cargo.toml --example offline_refiner_candidate_cap_audit
rm perception/examples/offline_refiner_candidate_cap_audit.rs
```
