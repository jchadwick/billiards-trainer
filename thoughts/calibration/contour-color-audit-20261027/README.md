# Saved-image contour and color audit — 2026-10-27

This audit records the RGB evidence around the revision-2 refiner's qualified
competing fits. It asks whether candidate rank or the sampled inside/outside
colors provide a repeatable cue for choosing the physical ball silhouette.
It does not change production behavior or use a candidate to provide a center.

## Inputs and fixed settings

The runner processes saved pixels only:

| Saved proposal set | Proposals | Image / optics profile | Reference status |
|---|---:|---|---|
| Measured grid | 9 balls with user-confirmed physical references | 1280 × 720, exploratory zero-radial profile | Matching positions exist; their uncertainty is unknown. |
| Alternate grid | 15 manually reviewed crop regions | 1280 × 720, same exploratory profile | Previously seen; no physical XY references. |
| 4K framing scene | 3 manually reviewed crop regions | 3840 × 2160, row-scaled diagnostic profile | No physical XY references; framing profile is not qualified. |

All crop proposals are manual, not detector output. The refiner ran unchanged
at seed 716, 128 RANSAC iterations, candidate cap 16, default three aiming
iterations, and default quality gates. Only the diagnostic batch/proposal
budgets were extended to 10 seconds so the saved crops could finish. No
camera access, new measurement, physical fixture, or calibration fitting was
involved.

For each individually qualified fit, the audit matches every selected inlier
ray to its retained boundary candidate, records the candidate's gradient
strength rank, then samples RGB at the candidate radius plus and minus the
refiner's scaled `background_probe_px`. Sampling uses the same `image` crate
JPEG decoding, local-view rotation, source projection, bilinear RGB lookup,
ROI bounds, and annulus-median background calculation as the production
refiner. Four local/source/local round trips are asserted for every evaluated
fit. Colors and RMS distances are normalized to `[0, 1]`.

## Results

All 27 final observations reject:

| Proposal set | Ambiguous boundary | Inadequate coverage | Insufficient boundary | Qualified competing-fit profiles |
|---|---:|---:|---:|---:|
| Measured grid, 9 physical references | 9 | 0 | 0 | 36 |
| Alternate 720p, 15 crops | 13 | 1 | 1 | 43 |
| 4K framing scene, 3 crops | 3 | 0 | 0 | 12 |
| **Total** | **25** | **1** | **1** | **91** |

The 91 qualified fit profiles contain 7,891 selected inlier-ray RGB samples.
The per-ray rank counts across those inliers are 3,569 strongest retained
transitions, 2,627 second-ranked transitions, and 1,695 third-ranked
transitions. Eighty-three of the 91 competing fits use all three ranks. The
same ball often has alternatives with different rank mixes, so rank is not a
semantic label for the outer silhouette.

Across the individually qualified fits, the mean normalized RGB RMS distance
from the estimated annulus background ranges from 0.225 to 0.504 for the
inside samples, 0.057 to 0.183 for the outside samples, and 0.059 to 0.222 for
the sampled transition gradient. These ranges overlap across proposals and
competing fits. Several alternatives for the same ball pass the current
inside-contrast and outside-background checks while describing distinct
outlines. The measured profiles therefore explain why the existing local
contrast checks can retain alternatives, but they do not provide a grounded
rule for choosing one.

The three crop montages visually show the qualified outlines over the saved
JPEGs: [measured 720p crops](overlays/measured-grid-720-crops.svg),
[alternate 720p crops](overlays/alternate-grid-720-crops.svg), and
[4K crops](overlays/framing-after-4k-crops.svg). Colors distinguish
representative-fit order within each proposal; crosses mark fitted centers
and dashed boxes are the manual crop proposals. Across the reviewed crops,
the competing outlines generally follow the same apparent ball perimeter
within a narrow rim. No single family is consistently separated as an
internal stripe edge, shadow edge, or uniquely identifiable physical
silhouette. That visual observation is not an independent center annotation.

**Conclusion:** this audit found no supported per-ray rank or simple RGB
summary cue that identifies the true outer contour among the qualified fits.
The saved pixels show multiple nearby transitions that satisfy the current
quality checks. Keep ambiguous results unavailable; do not select a fit using
the profile means, ranks, alternate seeds, or this same image as its own
ground truth.

## Evidence and limits

`summary.json` preserves all 27 outcomes, exact harness/refiner/corpus/image
hashes, configuration, fit-level profile summaries, and trace-file hashes.
`cases/` contains each complete schema-3 audit trace plus every measured
per-ray sample. The overlay SVGs embed the corresponding unchanged JPEG and
all qualified outlines. Source pixels, proposal identities, and generated
case hashes are recorded in the summary.

The measured positions' uncertainty is unknown. The alternate and 4K scenes
were previously seen, their manual crop centers are approximate, and they do
not have independent physical XY truth. Thus the audit cannot determine
which candidate is the actual mathematical silhouette, establish
millimeter-level XY accuracy, or qualify the runtime optics profile. It does
show that increasing the candidate cap was not needed to expose ambiguity on
these cases and that average inside/outside RGB contrast does not resolve it.

## Reproduction

From the repository root:

```text
cp thoughts/calibration/offline-refiner-contour-color-audit-20261027.rs perception/examples/offline_refiner_contour_color_audit.rs
cargo run --locked --manifest-path perception/Cargo.toml --example offline_refiner_contour_color_audit
rm perception/examples/offline_refiner_contour_color_audit.rs
rustfmt --check --edition 2021 thoughts/calibration/offline-refiner-contour-color-audit-20261027.rs
```

This runner writes `summary.json`, case traces, and overlay montages under
this directory. The source JPEGs and production implementation are read-only
inputs to the audit.
