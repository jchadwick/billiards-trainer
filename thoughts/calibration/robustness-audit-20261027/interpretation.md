# Saved-image robustness audit interpretation — 2026-10-27

## Evidence boundary

This audit invokes the production refiner in [`ball_localization.rs`](../../../perception/src/ball_localization.rs)
through the exact Rust runner
[`offline-refiner-robustness-audit-20261027.rs`](../offline-refiner-robustness-audit-20261027.rs).
It uses the final Task D refiner revision 2 and its fixed production seed of
716 for the center/size and baseline checks. The 15-ball measured-scene boxes
are saved manual annotation-derived proposals, not detector output. The
alternate-grid and 4K proposal boxes were visually reviewed as coarse crop
seeds for this audit; they are also not detector output or center labels.

The inputs are frozen saved images. No new camera frames or physical
measurements were used. Repeated fits and box perturbations on the same still
are not independent ball placements. The nine measured physical references
are attached to the measured-grid arrangement only, and their uncertainty is
not independently established. Neither reserved scene has matching physical
X/Y references.

For geometry sensitivity, batch and per-proposal budgets were both 10,000 ms
to avoid treating ordinary deadline expiry as fit instability. The 16 fitting
seeds are `716, 0..14`; only one seed is used in production. The work-budget
comparison is a separate same-frame workload. Timing was collected on a
Darwin arm64 development host and is not a target-device or end-to-end latency
measurement.

## Proposal center and size sensitivity

Each of the 15 measured-scene proposals was tested with an X-only offset and a
Y-only offset of −2, −1, 0, +1, and +2 source pixels. Width and height were
then scaled separately by 0.90, 0.95, 1.00, 1.05, and 1.10. All axes were
changed independently; there are 150 center runs and 150 size runs. The
original proposal region is used as the silhouette-containment reference, so
shrinking a dimension is conservatively labelled `may_clip_original_silhouette`.

| Experiment | Result counts |
|---|---|
| Center, X and Y separately (150) | 1 accepted; 132 `ambiguous_boundary`; 14 `insufficient_boundary`; 2 `inadequate_coverage`; 1 `unstable_center`. |
| Size, width and height separately (150) | 0 accepted; 137 `ambiguous_boundary`; 11 `insufficient_boundary`; 2 `inadequate_coverage`. |

The only accepted center perturbation was for
`bottom-black-second-left-ball` with its proposal moved 2 px upward. It
returned `(362.574, 626.567)` source pixels. The unperturbed production-seed
result rejects. This single acceptance shows proposal-center sensitivity; it
does not identify which center is physically correct. No size perturbation
accepted a center. None of the 300 runs reached its diagnostic deadline.

## Fitting-start sensitivity

The 15 frozen proposals were each run under all 16 listed seeds, for 240
results. The selected production seed 716 rejected all 15 proposals. Five
seed/proposal combinations accepted a source-image center, distributed across
four seeds:

| Seed | Accepted proposal | Center in source pixels |
|---:|---|---:|
| 2 | interior-purple-second-left-middle-row-ball | (357.888, 363.790) |
| 3 | top-right-cue-pocket-ball | (1195.395, 97.233) |
| 8 | far-right-purple-middle-row-ball | (1197.250, 362.188) |
| 10 | interior-purple-second-left-middle-row-ball | (358.177, 363.861) |
| 11 | interior-purple-second-left-middle-row-ball | (358.154, 364.418) |

Across seeds 2, 10, and 11, the three accepted centers for the same purple
ball span up to 0.682 source px. Two of those three balls have physical
references for this arrangement; the cue-pocket ball does not. The other
eleven seeds accepted no observations. Across all runs, the statuses were 5
accepted, 209 `ambiguous_boundary`, 18 `insufficient_boundary`, and 8
`inadequate_coverage`.

This is material sensitivity to the configured RANSAC seed. Do not switch to
seeds 2, 3, 8, 10, or 11 to increase accepted counts: no independent evidence
selects their centers as correct, and accepted outputs vary by seed. The fixed
seed keeps production repeatable for a given build, but the seed audit shows
that repeatability at one seed is not evidence that the search has found the
right boundary.

All six variants that passed the estimator gates during the separate center
and seed sensitivity runs were visually reviewed using their preserved crop
overlays. The fitted circles overlap visible ball boundaries, including the
seed-sensitive purple and pocket-adjacent examples. That visual plausibility
does not distinguish the alternatives or establish a true center; none of the
six is promoted into the production measurement set.

## Work-budget sensitivity

The same 15 proposals were run as a batch with these budget pairs. All
completed outputs were geometrically rejected; the table reports deadline
rejections only.

| Batch budget | Per-proposal budget | Deadline rejections | Elapsed on Darwin arm64 |
|---:|---:|---:|---:|
| 5 ms | 5 ms | 15 | 5.0 ms |
| 10 ms | 10 ms | 15 | 10.0 ms |
| 25 ms | 10 ms | 15 | 25.0 ms |
| 50 ms | 25 ms | 12 | 50.0 ms |
| 100 ms | 50 ms | 10 | 100.0 ms |
| 1000 ms | 1000 ms | 0 | 293.4 ms |
| 10000 ms | 10000 ms | 0 | 289.5 ms |

At the current initial 25/10 ms configuration, this 15-proposal development
host batch finished no observations before the batch limit. A complete batch
with relaxed diagnostic budgets took about 0.29 s. This is a concrete reason
to optimize or explicitly qualify the work schedule before release. It is
not evidence of the Intel target's timing, Coral cost, decoder cost, desired
5 FPS behavior, or application availability. Do not reuse stale centers to
hide deadline rejections.

## Reserved saved-scene baselines

The proposal boxes are recorded separately in
[`reserved-scene-coarse-proposals-20261027.json`](../reserved-scene-coarse-proposals-20261027.json).
The full saved frames and enlarged candidate traces are in
[`reserved-overlays/`](reserved-overlays/) and [`reserved-cases/`](reserved-cases/).
Every case retained all trace events. Results by manually reviewed proposal:

| Saved scene | Proposal | Visibility note | Result | Qualified candidates |
|---|---|---|---|---:|
| Alternate 720p | alt-top-left-pocket | Partial pocket-jaw overlap | `inadequate_coverage` | 0 |
| Alternate 720p | alt-top-orange | Fully visible | `ambiguous_boundary` | 3 |
| Alternate 720p | alt-top-yellow-stripe | Near top-center pocket | `ambiguous_boundary` | 7 |
| Alternate 720p | alt-top-red-stripe | Probable top-rail contact | `ambiguous_boundary` | 3 |
| Alternate 720p | alt-top-right-cue | Partial pocket-jaw overlap | `ambiguous_boundary` | 2 |
| Alternate 720p | alt-middle-left-yellow | Left-rail contact | `ambiguous_boundary` | 2 |
| Alternate 720p | alt-middle-left-purple-stripe | Fully visible | `ambiguous_boundary` | 4 |
| Alternate 720p | alt-middle-red-stripe | Fully visible | `ambiguous_boundary` | 3 |
| Alternate 720p | alt-middle-green-stripe | Fully visible | `ambiguous_boundary` | 2 |
| Alternate 720p | alt-middle-right-purple | Right-rail contact | `ambiguous_boundary` | 2 |
| Alternate 720p | alt-bottom-left-red | Partial pocket-jaw overlap | `ambiguous_boundary` | 4 |
| Alternate 720p | alt-bottom-black | Bottom-rail contact | `ambiguous_boundary` | 2 |
| Alternate 720p | alt-bottom-center-red-stripe | Near bottom-center pocket | `insufficient_boundary` | 0 |
| Alternate 720p | alt-bottom-yellow | Bottom-rail contact | `ambiguous_boundary` | 3 |
| Alternate 720p | alt-bottom-right-blue-stripe | Partial pocket-jaw overlap | `ambiguous_boundary` | 2 |
| Native 4K | 4k-top-left-orange | Fully visible, near top rail | `ambiguous_boundary` | 2 |
| Native 4K | 4k-top-center-striped | Fully visible, near top rail | `ambiguous_boundary` | 6 |
| Native 4K | 4k-top-right-red-stripe | Fully visible, near top rail | `ambiguous_boundary` | 4 |

The alternate image hash is
`8371e508f79c5556cfd2fe6632b9095ba87d8f7052741b2f6b01563b448c385b` and its
1280 × 720 intrinsics are the exploratory frozen zero-radial model. The 4K
image hash is
`5d88a4d6037e7718723916b3af6c930f8020ddcb03e008122e9c8fadc5530206`; its
3840 × 2160 intrinsics are a diagnostic 3× scale of that model. Neither
provenance establishes a qualified runtime camera profile. All 18 reserved
proposals reject, and none has matching physical X/Y truth.

## Conclusion and next step

The current result depends on proposal centering and RANSAC seed, and all
reserved-scene proposals reject under the frozen settings. Increasing the
radial search to the planned 96 samples exposed multiple alternatives but did
not establish accurate centers. Choosing a convenient seed or treating an
individual qualified fit as the final center would turn uncertainty into an
unsupported measurement.

Do not produce a nine-reference mapping or a calibration asset from this
result. Continue by investigating why qualified candidate families vary with
seed, while preserving the current fixed-seed baseline and fail-closed
behavior. Target responsiveness must be qualified on the actual device later;
the development-host timings are a warning about the initial deadline, not a
camera FPS selection.

## Reproduction

Copy the exact runner into a temporary Cargo example, then run it from the
repository root:

```text
cp thoughts/calibration/offline-refiner-robustness-audit-20261027.rs perception/examples/offline_refiner_robustness_audit.rs
cargo run --locked --manifest-path perception/Cargo.toml --example offline_refiner_robustness_audit
rm perception/examples/offline_refiner_robustness_audit.rs
```

The harness validates saved image hashes and dimensions before processing.
`summary.json` contains the complete compact outcomes, quality values, budget
results, per-scene refiner traces, source hashes, and paths/hashes for all
reserved overlays and traces.
