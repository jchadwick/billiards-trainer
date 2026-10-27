# First-iteration RANSAC sampling audit — 2026-10-27

This audit isolates the first local-view aiming iteration for the saved
interior purple ball. It holds image pixels, manual proposal, exploratory
intrinsics, crop axis, all non-search settings, and broad fit-quality gates
fixed. Only the deterministic RANSAC seed and its iteration count vary. The
production settings remain seed 716 and 128 iterations; this audit does not
change them.

| Seed | RANSAC iterations | Raw hypotheses | Fully qualified candidates | Selected outcome |
|---:|---:|---:|---:|---|
| 716 | 128 | 88 | 4 | Rejected: `ambiguous_boundary` |
| 2 | 128 | 89 | 1 | Accepted at (357.984, 363.656) px |
| 10 | 128 | 93 | 1 | Accepted at (358.014, 363.796) px |
| 11 | 128 | 99 | 2 | Accepted at (358.067, 363.903) px |
| 716 | 256 | 186 | 4 | Rejected: `ambiguous_boundary` |
| 716 | 512 | 373 | 4 | Accepted at (358.049, 363.909) px |
| 716 | 1024 | 738 | 4 | Rejected: `ambiguous_boundary` |

The selected outcome is non-monotonic in iteration count: with seed 716, the
fit rejects at 128 and 256, passes at 512, then rejects again at 1024. At 512,
one newly found alternative lies 0.497 source pixels from the best outline,
inside the scaled 0.5-pixel same-boundary threshold. At 1024, another
plausible-support alternative is 0.767 pixels away and triggers ambiguity.
The additional candidate families at the larger search counts therefore do
not resolve which visible transition is the true ball silhouette.

The seed 11 run has two individually qualified fits. The second has 90/95
support (94.74%), below the 95% alternative-support threshold, so it is not
counted as a plausible competing fit by the final selection logic. Its
presence still shows that passing an individual fit's quality checks does not
alone determine whether it blocks selection.

The production seed's four qualified fits at 128 iterations also select a
different mix of the three retained transition-strength ranks. The rank is
the order returned by per-ray gradient-strength selection; it is not a
semantic label such as “outer edge.”

| Fit ID | Support | Rank 0 | Rank 1 | Rank 2 |
|---:|---:|---:|---:|---:|
| 5 | 95 | 57 | 23 | 15 |
| 31 | 92 | 18 | 38 | 36 |
| 69 | 92 | 44 | 34 | 14 |
| 87 | 90 | 38 | 30 | 22 |

This is consistent with the fits assembling nearby transitions across the
same rays, but it does not reveal which transition is the true silhouette.
The ranks are reconstructed from the complete `boundary_candidate` and
`selected_inlier` events in the trace; their candidate indices and rays are
recorded directly in each valid raw hypothesis.

Every complete case trace is in `cases/`; `summary.json` includes the sampled
triplets and candidate rays for valid raw hypotheses, candidate quality,
pairwise source-outline Hausdorff distances, exact input/source hashes, and
the trace file hashes. All seven traces use audit schema 3, have zero dropped
events, and remain below the event cap. Only valid raw hypotheses emit sampled
triplet records; draws rejected before a raw hypothesis are not represented.

The saved image is 1280 × 720 and uses manually reviewed annotation-derived
proposals plus an exploratory zero-radial intrinsics model. There is no
independent pixel-center truth or qualified optics profile for this audit.
Accepted results are not promoted to calibration observations. This work
measures RANSAC sensitivity only; it does not establish physical accuracy,
camera FPS, or target-device latency.

## Reproduction

From the repository root, copy the exact runner into a temporary Cargo
example and run it:

```text
cp thoughts/calibration/offline-refiner-sampling-audit-20261027.rs perception/examples/offline_refiner_sampling_audit.rs
cargo run --locked --manifest-path perception/Cargo.toml --example offline_refiner_sampling_audit
rm perception/examples/offline_refiner_sampling_audit.rs
```
