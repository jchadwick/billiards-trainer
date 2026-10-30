# Converged-candidate audit interpretation — 2026-10-27

## Purpose and evidence boundary

This is a diagnostic run of refiner revision 2 on the frozen 1280 × 720 image
`table-grid-20261026-114241.jpg`. The 15 square proposals are saved
manual/annotation-derived boxes. They are not outputs from the detector and
are not ground-truth image centers. Nine proposals correspond to the saved
physical references for that same arrangement; those references have
uncertainty that is not independently established. The other six balls are
near pockets and have no corresponding physical reference.

The run used the exact frozen corpus intrinsics and production refiner logic.
The batch and per-proposal diagnostic budgets were both 10,000 ms to avoid
confusing a fitting rejection with ordinary live deadlines. Traces use audit
schema 2, retain per-fit outlines and quality, and report no dropped events.
The configured 1.5 px center-spread and same-boundary-outline limits scale to
0.5 source px at this 1280 px width. These are algorithm settings, not measured
accuracy claims.

## Result

**No precise center was accepted: 0/15 overall and 0/9 for the proposals with
physical references.** Fourteen observations reject as `ambiguous_boundary`;
the top-left pocket observation rejects as `insufficient_boundary`.

| Proposal | Matching physical reference | Final result | Qualified candidate fits |
|---|---:|---|---:|
| top-left-pocket-ball | No | `insufficient_boundary` | 0 |
| top-orange-solid-second-left-ball | Yes | `ambiguous_boundary` | 2 |
| top-middle-pocket-ball | No | `ambiguous_boundary` | 2 |
| top-red-striped-second-right-ball | Yes | `ambiguous_boundary` | 8 |
| top-right-cue-pocket-ball | No | `ambiguous_boundary` | 2 |
| far-left-yellow-middle-row-ball | Yes | `ambiguous_boundary` | 6 |
| interior-purple-second-left-middle-row-ball | Yes | `ambiguous_boundary` | 4 |
| middle-row-center-ball | Yes | `ambiguous_boundary` | 4 |
| interior-green-second-right-middle-row-ball | Yes | `ambiguous_boundary` | 2 |
| far-right-purple-middle-row-ball | Yes | `ambiguous_boundary` | 2 |
| bottom-left-pocket-ball | No | `ambiguous_boundary` | 4 |
| bottom-black-second-left-ball | Yes | `ambiguous_boundary` | 3 |
| bottom-middle-pocket-ball | No | `ambiguous_boundary` | 6 |
| bottom-yellow-second-right-ball | Yes | `ambiguous_boundary` | 4 |
| bottom-right-pocket-ball | No | `ambiguous_boundary` | 0 |

Thirteen observations produced at least two individually qualified candidate
fits in the final aiming iteration. Their maximum pairwise center spans range
from 0.579 px to 1.781 px in the source image. In every case, at least one plausible alternative is
distinct from the top-ranked fit under the effective 0.5 px center and
outline-comparison limits, so the resolver withholds a center. A qualified
outline means that fit passed its own boundary checks; it does not mean the
overall observation was accepted. The bottom-right pocket proposal produced
no qualified fit: its single representative failed bounded inlier reselection
at `fit_refinement`. The top-left pocket ball had no usable boundary
candidate.

No search hit its candidate cap, and no case dropped trace events. Thus the
observed rejections were not caused by an incomplete diagnostic trace or
candidate-search truncation.

## Interpretation and next action

This result differs from the earlier identity audit's 3/9 measured acceptance
under the pre-revision-2 selection behavior. That earlier acceptance was
repeatability evidence only, and the frozen identity audit showed it depended
on detection identity for seven of nine measured proposals. Revision 2
removes that identity dependence and compares converged source-image
boundaries. The zero-acceptance result is not evidence that the centers are
wrong, but it does show that current saved proposals do not yield a resolved
precision center under the conservative ambiguity checks. Conversely, the
earlier 3/9 accepted count is not evidence that those centers were right.

Do not loosen the spread, outline, or residual gates to recover an acceptance
count. The manual boxes and nine physical references do not identify which
of the competing image outlines is the correct ball boundary independently.
Do not pass rejected candidates to tracking as precise points. There are no
eligible centers for a production ball-center homography, so the nine-point
mapping task is report-only at this checkpoint: no mapping asset or physical
accuracy score is produced.

Continue with controlled proposal-center/size and fitting-start sensitivity
audits on the saved images. If those tests show stable, repeatable candidate
families, retain that as diagnosis only. To establish the right contour among
competing fits, obtain independent image-center evidence or use a verifiable
physical placement already represented in the saved image; do not infer truth
from the mapping being evaluated. Physical reference collection remains
unnecessary for this diagnostic step.

## Reproduction and artifacts

- Per-case trace, complete fit outlines, quality, and rejection stage: `cases/`.
- Standalone full-source overlay and one enlarged crop per proposal:
  `overlays/`.
- Machine-readable summary and source/configuration hashes: `summary.json`.
- Exact offline runner: `../offline-refiner-converged-candidate-audit-20261027.rs`.
- Frozen image SHA-256: `06ef40f52e0f93db17601585c4ed1d014895aaadcaf6b04c6abbd6e986d57e7b`.
- Frozen corpus SHA-256: `4389155bf13fac6dd49d45a6488a53f37f4864b586646c0ced543f151c41a1de`.
- Refiner source SHA-256 for this run: `602266310962d8a4703a264515c272dd024c731d1b45edc56ef04a5d2bfb1b82`.
- Refiner revision: 2; trace schema version: 2.
- Effective source sampling: 96 radial samples × 96 angular samples.

The recorded diagnostic wall times include trace construction and run on a
development host. They are not capture, target-device, or real-time performance
measurements.
