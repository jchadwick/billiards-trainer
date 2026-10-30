# Deterministic cyclic boundary-path comparison — 2026-10-28

Research only. Exact top-eight paths optimize a declared continuity/gradient objective on frozen source candidates. All original runtime rejections remain unchanged. No point is accepted for production.

{'case_count': 15, 'baseline_accepted': 0, 'iterations': 16, 'best_path_geometric_diagnostic_pass': 7, 'production_accepted': 0}

| Scene / ball | Iteration | Best local RMS px | Best source center | Geometric failures | Top8 center spread source px |
|---|---:|---:|---|---|---:|
| measured-grid-720 / bottom-black-second-left-ball | 1 | 0.8485721240823373 | [361.9762140220905, 626.6646570471574] | ['excessive_residual'] | 0.0405222581342702 |
| measured-grid-720 / bottom-left-pocket-ball | 1 | 2.3474088605826915 | [105.405729990331, 617.3757647122011] | ['excessive_residual', 'unstable_center'] | 0.08235338830317362 |
| measured-grid-720 / bottom-middle-pocket-ball | 1 | 0.3313269821834365 | [641.1669004894611, 629.117564071418] | [] | 0.08583510467126393 |
| measured-grid-720 / bottom-right-pocket-ball | — | — | — | no retained boundary candidates | — |
| measured-grid-720 / bottom-yellow-second-right-ball | 1 | 0.36697145779249274 | [925.2327329825147, 630.0953960883917] | [] | 0.04885010506002904 |
| measured-grid-720 / far-left-yellow-middle-row-ball | 1 | 0.5383117266744818 | [95.93669557695978, 362.00723138342613] | ['excessive_residual'] | 0.041491903218315036 |
| measured-grid-720 / far-right-purple-middle-row-ball | 1 | 1.0284133243378348 | [1197.17033009996, 361.93926799082345] | ['excessive_residual'] | 0.05152120326302125 |
| measured-grid-720 / interior-green-second-right-middle-row-ball | 1 | 0.43319405132894917 | [924.4260466992148, 361.77678805571395] | [] | 0.027972946585554215 |
| measured-grid-720 / interior-purple-second-left-middle-row-ball | 1 | 0.6593395942454 | [358.62483324104323, 363.6040859657735] | ['excessive_residual'] | 0.09314025104780621 |
| measured-grid-720 / middle-row-center-ball | 1 | 0.3661840178909467 | [643.0066117272285, 361.2169158253932] | [] | 0.04998195067838499 |
| measured-grid-720 / top-left-pocket-ball | 1 | 2.1753985758502012 | [104.43680031540727, 107.0035304955991] | ['excessive_residual', 'unstable_center'] | 0.11026007533237968 |
| measured-grid-720 / top-left-pocket-ball | 2 | 2.062529606121664 | [104.56512215209611, 107.13122484628016] | ['excessive_residual', 'unstable_center'] | 0.15178567141154464 |
| measured-grid-720 / top-left-pocket-ball | 3 | 2.25709771409995 | [104.70657595155983, 107.16075967126679] | ['excessive_residual', 'unstable_center'] | 0.27324635043741613 |
| measured-grid-720 / top-middle-pocket-ball | 1 | 0.23527718363543212 | [641.3994798002778, 97.33185627304726] | [] | 0.048202289358357665 |
| measured-grid-720 / top-orange-solid-second-left-ball | 1 | 0.2949239994636496 | [352.47965304590934, 96.81213235809211] | [] | 0.035661432099809584 |
| measured-grid-720 / top-red-striped-second-right-ball | 1 | 0.31347987615913686 | [927.5972126387674, 93.65631700920221] | [] | 0.06171164118585271 |
| measured-grid-720 / top-right-cue-pocket-ball | 1 | 0.9201103564261585 | [1195.3439000149501, 96.89097393941361] | ['excessive_residual'] | 0.044473638668550905 |

## Interpretation

The synthetic concentric-circle case has two equally optimal coherent contours. A coherent stronger shadow ring outranks the true silhouette in another counterexample. Thus this search supplies deterministic hypotheses and a test of continuity, not independent silhouette truth. Candidate families beyond the top eight may remain important. Small top-eight center spread is not a calibrated error bound.

The full outputs preserve all eight paths, their sampled points, source projection, circle quality, and per-iteration center spread. Pixel foreground/background support and full iterative aiming are intentionally untested, so a geometric diagnostic pass is not comparable to a fully accepted runtime measurement.

Current refiner accepts zero referenced measurements. This experiment does not fit a physical homography from rejected or diagnostic centers; the existing nine-reference comparison remains a no-fit production result.

Reproduce: `python thoughts/calibration/offline-cyclic-boundary-20261028.py --inputs thoughts/calibration/cyclic-boundary-audit-20261028/inputs.json --output /tmp/cyclic-replay`; run `--self-test` for exact/exhaustive search, ambiguity counterexamples, known-center recovery and support checks. Re-extract compact inputs only from the recorded experiment checkout using `--extract-inputs PATH`. Source hashes are retained in inputs.json.

## Conditional nine-reference comparison

The all-nine best-path homography is hypothetical and includes geometrically rejected paths. It is not a production map. All nine physical reference uncertainties are unknown.

| Diagnostic method | Same-nine LOO RMS mm |
|---|---:|
| Cyclic best path, including failed geometric diagnostics | 8.722304093691195 |
| Earlier segmentation, same nine points | 8.705609317078913 |

Geometrically rejected IDs in the hypothetical fit: ['far-right-purple-middle-row-ball', 'far-left-yellow-middle-row-ball', 'interior-purple-second-left-middle-row-ball', 'bottom-black-second-left-ball']. Full per-point residuals and common passing-subset comparisons are in JSON; neither reduced residual nor subset selection proves accuracy.
