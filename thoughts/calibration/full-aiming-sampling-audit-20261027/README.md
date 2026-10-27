# Full-aiming RANSAC sampling audit — 2026-10-27

This is the second phase of the purple-ball sampling audit. It uses the same
saved 1280 × 720 image, manual proposal, exploratory intrinsics, production
seed 716, and RANSAC counts as a first-aiming-iteration audit, but restores the
normal default maximum of three aiming iterations. Only RANSAC iteration
count varies. The per-proposal and batch diagnostic budgets are 10 seconds;
production settings and quality gates are unchanged.

| RANSAC iterations | Raw hypotheses across aiming iterations | Qualified fit evaluations | Aiming iterations reached | Final outcome |
|---:|---:|---:|---:|---|
| 128 | 88 | 5 | 1 | Rejected: `ambiguous_boundary` |
| 256 | 186 | 6 | 1 | Rejected: `ambiguous_boundary` |
| 512 | 740 | 19 | 2 | Rejected: `ambiguous_boundary` |
| 1024 | 738 | 10 | 1 | Rejected: `ambiguous_boundary` |

The 512-iteration result that passed the isolated first aiming iteration does
**not** survive normal aiming. After its first fit updated the aiming axis, a
second iteration generated four individually qualified fits with distinct
outlines; their pairwise source-outline Hausdorff distances ranged from 0.686
to 1.500 px. The observation then rejected as ambiguous with no selected
center. The 128-, 256-, and 1024-iteration runs reject during their first
iteration. Increasing the RANSAC count therefore does not produce a stable
precise result in the normal iterative flow.

The complete schema-3 trace for each run is in `cases/`. `summary.json`
retains raw-hypothesis triplets and rays, candidate evaluation metrics,
within-iteration pairwise source-outline distances, outcomes, exact hashes,
and trace completeness. All four traces have zero dropped events and remain
below the event cap. The sample triplets cover valid raw hypotheses only;
draws rejected before a raw hypothesis are not emitted by the audit schema.

The image, proposal, and optics provenance is the same as in the
[first-iteration audit](../sampling-audit-20261027/README.md). This is
diagnostic behavior on one saved image, not physical-center truth, a target
performance result, or a reason to change production settings. The evidence
continues to favor rejecting uncertain observations over promoting whichever
seed/count happens to return a center.

## Reproduction

From the repository root:

```text
cp thoughts/calibration/offline-refiner-full-aiming-sampling-audit-20261027.rs perception/examples/offline_refiner_full_aiming_sampling_audit.rs
cargo run --locked --manifest-path perception/Cargo.toml --example offline_refiner_full_aiming_sampling_audit
rm perception/examples/offline_refiner_full_aiming_sampling_audit.rs
```
