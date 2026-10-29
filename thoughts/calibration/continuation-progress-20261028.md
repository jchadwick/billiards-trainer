# Continued implementation — 2026-10-28

The offline continuation implements source-bound checkerboard observations,
explicit review masks and frozen parity scoring; meaningful synthetic tests
pass. Pixel-review records are not yet completed, so these exports remain
pending and are not eligible calibration observations.

Actual source-checkpoint inference finds 15/15 corresponding coarse ball regions
in each saved 720p scene and 3/3 in the native4K scene. It is a development CPU
reference, not Coral equivalence. The bounded saved-video replay exposes count
and association changes around the rack; it does not qualify temporal tracking.

A deterministic cyclic-path comparator is implemented and its exact top-eight
objective verified against exhaustive small graphs. Known shadow/concentric
counterexamples prevent interpreting optimization as physical truth. Across
33 old manually proposed cases, 15 best paths pass geometric-only diagnostics.
The current full refiner still rejects all cases.

Real detector boxes were then passed through the unchanged Rust refiner on the
15-ball measured image: 0/15 accept, including 0/9 referenced balls. The
same deterministic comparator on those actual-box traces yields seven geometric
passes, but hypothetical all-nine mapping RMS is 8.7223 mm, versus 8.7056 mm
for the old segmentation baseline. Neither is independently measured accuracy;
this does not demonstrate physical improvement or qualify a production change.
Pixel qualification and complete aiming remain untested in the comparator.

All new files remain research on the experiment branch. No runtime settings,
source images, camera, deployment or main implementation changed. A request
for the minimal independent small-target image has been sent; no camera access
is allowed until the user confirms the table is uncovered and ready. Further
useful software work is full source-pixel qualification and actual-proposal
perturbation testing, not more parameter sweeps or acceptance-gate relaxation.
