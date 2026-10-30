# Converged-candidate audit — 2026-10-29

This audit uses the frozen saved 1280 × 720 image and actual saved detector boxes. It diagnoses estimator behavior; it does not establish physical X/Y accuracy. Each crop has a bounded candidate trace. The current build accepts **13 of 15** precise image centers, including **9 of 9** proposals with matching physical references. Do not use candidate outlines as resolved measurements.

| Proposal | Physical reference | Result | Qualified candidates | Search truncated | Trace dropped |
|---|---:|---|---:|---:|---:|
| far-left-yellow-middle-row-ball | Yes | accepted / none | 1 | false | 0 |
| bottom-yellow-second-right-ball | Yes | accepted / none | 1 | false | 0 |
| far-right-purple-middle-row-ball | Yes | accepted / none | 1 | false | 0 |
| top-orange-solid-second-left-ball | Yes | accepted / none | 1 | false | 0 |
| bottom-left-pocket-ball | No | accepted / none | 1 | false | 0 |
| top-red-striped-second-right-ball | Yes | accepted / none | 1 | false | 0 |
| top-right-cue-pocket-ball | No | accepted / none | 1 | false | 0 |
| middle-row-center-ball | Yes | accepted / none | 1 | false | 0 |
| top-middle-pocket-ball | No | accepted / none | 1 | false | 0 |
| bottom-black-second-left-ball | Yes | accepted / none | 1 | false | 0 |
| interior-purple-second-left-middle-row-ball | Yes | accepted / none | 1 | false | 0 |
| bottom-middle-pocket-ball | No | accepted / none | 1 | false | 0 |
| top-left-pocket-ball | No | rejected / inadequate_coverage | 0 | false | 0 |
| bottom-right-pocket-ball | No | accepted / none | 1 | false | 0 |
| interior-green-second-right-middle-row-ball | Yes | accepted / none | 1 | false | 0 |

The full-source overlay is [overlays/full-source.svg](overlays/full-source.svg). Enlarged per-ball overlays are in `overlays/`; complete per-crop traces are in `cases/`. Green outlines passed an individual candidate's gates, but the resolver can still reject the observation when qualified solutions disagree. Magenta outlines show rejected ambiguous fits; orange outlines failed another gate. Cyan dots are sampled boundary candidates, yellow dots are selected inliers, and blue crosses are raw hypotheses. A green center marker appears only for an accepted final result. Crop candidate labels are placed in a side column to keep them off the fitted outlines.
