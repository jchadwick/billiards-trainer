# Detection identity invariance audit — 2026-10-27

## Result

The refiner now returns the same semantic result for every tested detection
index from 0 through 15 on all 15 frozen manual proposals. This includes the
accepted center, rejection reason, quality measurements, and projected outline.
The Rust regression also checks reversed proposal order with sufficient
budgets. Detection identity no longer changes the RANSAC sample sequence.

The saved still contains nine user-confirmed physical references and six
pocket-ball proposals without physical X/Y truth. Under the corrected fixed
seed, the estimator accepted three of the nine measured proposals and rejected
six. The six pocket proposals were also rejected. These results are not a
physical accuracy or availability qualification.

| Saved proposal | Physical reference | Result across IDs 0–15 |
|---|---:|---|
| top-left-pocket-ball | No | 0 accepted; insufficient boundary |
| top-orange-solid-second-left-ball | Yes | 0 accepted; ambiguous boundary |
| top-middle-pocket-ball | No | 0 accepted; insufficient boundary |
| top-red-striped-second-right-ball | Yes | 16 accepted; same center and outline |
| top-right-cue-pocket-ball | No | 0 accepted; excessive center shift |
| far-left-yellow-middle-row-ball | Yes | 0 accepted; ambiguous boundary |
| interior-purple-second-left-middle-row-ball | Yes | 16 accepted; same center and outline |
| middle-row-center-ball | Yes | 0 accepted; ambiguous boundary |
| interior-green-second-right-middle-row-ball | Yes | 0 accepted; ambiguous boundary |
| far-right-purple-middle-row-ball | Yes | 0 accepted; ambiguous boundary |
| bottom-left-pocket-ball | No | 0 accepted; excessive center shift |
| bottom-black-second-left-ball | Yes | 0 accepted; ambiguous boundary |
| bottom-middle-pocket-ball | No | 0 accepted; excessive center shift |
| bottom-yellow-second-right-ball | Yes | 16 accepted; same center and outline |
| bottom-right-pocket-ball | No | 0 accepted; inadequate coverage |

Compared with the frozen pre-fix audit, the acceptance-index set changed on
7 of the 9 measured positions. This is evidence that identity-dependent
sampling had been affecting acceptance; it is not evidence that the new
acceptances are more accurate. The six rejections remain visible outcomes and
are carried into the next boundary-selection investigation.

## Inputs and method

The audit uses the saved 1280 × 720 `measured-grid-720` image, exact exploratory
intrinsics recorded in the corpus manifest, and the 15 manifest boxes derived
from saved manual segmentation/gradient annotations. These are not YOLO
proposals. For each box, the production Rust estimator ran separately with
detection indices 0 through 15, the same configured seed, a 5,000 ms batch
budget, and a 400 ms per-proposal budget. No camera access or new measurements
were used.

Full per-index results, quality values, outlines, input hashes, and source
hashes are in
[`offline-refiner-identity-audit-20261027.json`](offline-refiner-identity-audit-20261027.json).
The executable research harness is
[`offline-refiner-identity-audit-20261027.rs`](offline-refiner-identity-audit-20261027.rs).
The frozen pre-fix reference is
[`offline-refiner-identity-check.json`](offline-refiner-identity-check.json).

## Limits

- Repeated runs on one still are not independent placements.
- The labels and boxes do not independently establish ball-center pixels.
- The physical reference uncertainty is unknown; no millimeter-accuracy claim
  follows from this identity check.
- The corrected code is an experimental branch result. It does not select the
  correct outer contour in every crop; the six measured rejections still need
  the converged-boundary analysis.
