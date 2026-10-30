# Deterministic cyclic boundary-path comparison — 2026-10-28

Research only. Exact top-eight paths optimize a declared continuity/gradient objective on frozen source candidates. All original runtime rejections remain unchanged. No point is accepted for production.

{'case_count': 33, 'baseline_accepted': 0, 'iterations': 36, 'best_path_geometric_diagnostic_pass': 15, 'production_accepted': 0}

| Scene / ball | Iteration | Best local RMS px | Best source center | Geometric failures | Top8 center spread source px |
|---|---:|---:|---|---|---:|
| measured-grid-720 / bottom-black-second-left-ball | 1 | 0.7525257486367226 | [361.99871846863095, 626.7023459874683] | ['excessive_residual'] | 0.01137644167379883 |
| measured-grid-720 / bottom-left-pocket-ball | 1 | 2.3193710194293966 | [102.7593441713373, 619.8337554689402] | ['excessive_residual', 'unstable_center'] | 0.27617612719429196 |
| measured-grid-720 / bottom-middle-pocket-ball | 1 | — | — | insufficient_angular_support | None |
| measured-grid-720 / bottom-right-pocket-ball | 1 | 2.026875066203781 | [1193.7142064401155, 630.8681497314316] | ['excessive_residual', 'unstable_center'] | 0.14425137412607994 |
| measured-grid-720 / bottom-yellow-second-right-ball | 1 | 0.37374699843579623 | [925.2094194922395, 630.1012427543379] | [] | 0.042777791934317286 |
| measured-grid-720 / far-left-yellow-middle-row-ball | 1 | 0.48035318121105763 | [95.8340025553896, 362.0334472609622] | [] | 0.03109784509106739 |
| measured-grid-720 / far-right-purple-middle-row-ball | 1 | 0.9360037064648905 | [1197.212713463293, 361.9080628230785] | ['excessive_residual'] | 0.02714370482444891 |
| measured-grid-720 / interior-green-second-right-middle-row-ball | 1 | 0.4407306053703002 | [924.3806979447522, 361.7855054831107] | [] | 0.04833075605087836 |
| measured-grid-720 / interior-green-second-right-middle-row-ball | 2 | 0.47189280786202065 | [924.4087202769233, 361.76965932284867] | [] | 0.012634265654410252 |
| measured-grid-720 / interior-purple-second-left-middle-row-ball | 1 | 0.6197174934704957 | [358.645697820281, 363.60362207610467] | ['excessive_residual'] | 0.19656345833961122 |
| measured-grid-720 / middle-row-center-ball | 1 | 0.36196195571350237 | [642.9267213841388, 361.2351630779627] | [] | 0.04789769540913452 |
| measured-grid-720 / top-left-pocket-ball | 1 | — | — | insufficient_angular_support | None |
| measured-grid-720 / top-middle-pocket-ball | 1 | 0.23301766846429695 | [641.3652471979291, 97.28381316239984] | [] | 0.060228443086951536 |
| measured-grid-720 / top-orange-solid-second-left-ball | 1 | 0.46181828792150126 | [352.4749139676844, 96.80007473181772] | [] | 0.03365932553203915 |
| measured-grid-720 / top-red-striped-second-right-ball | 1 | 0.2950488514264093 | [927.6014376811485, 93.66009036064325] | [] | 0.05092100216462409 |
| measured-grid-720 / top-right-cue-pocket-ball | 1 | 3.285188733246024 | [1195.625383502721, 96.67633492475971] | ['excessive_residual', 'unstable_center'] | 0.04473078016455612 |
| framing-after-4k / 4k-top-center-striped | 1 | 0.46024250244134596 | [1924.9346349253944, 292.80814680630664] | [] | 0.09926121727756612 |
| framing-after-4k / 4k-top-left-orange | 1 | 1.3730818348794722 | [1056.2254899228135, 290.80274595077447] | [] | 0.10709867556633425 |
| framing-after-4k / 4k-top-right-red-stripe | 1 | 0.5180145630279865 | [2782.883774139365, 282.08105806152525] | [] | 0.05339183753448381 |
| alternate-grid-720 / alt-bottom-black | 1 | 0.5162379195276552 | [369.6301212673688, 626.6873210922055] | ['excessive_residual'] | 0.040867499014524994 |
| alternate-grid-720 / alt-bottom-center-red-stripe | — | — | — | no retained boundary candidates | — |
| alternate-grid-720 / alt-bottom-left-red | 1 | 2.3953033472293863 | [105.05441326303816, 618.1398507788642] | ['excessive_residual', 'unstable_center'] | 0.2234187508230925 |
| alternate-grid-720 / alt-bottom-right-blue-stripe | 1 | 3.3558996356832855 | [1189.543093155832, 626.171414079357] | ['excessive_residual'] | 0.08136571233156784 |
| alternate-grid-720 / alt-bottom-right-blue-stripe | 2 | 0.5262241716702452 | [1188.971050051412, 625.6110125869126] | ['excessive_residual'] | 0.06985078566939076 |
| alternate-grid-720 / alt-bottom-yellow | 1 | 0.5254391977992172 | [925.1950170013482, 630.118132987998] | ['excessive_residual'] | 0.02383563585805175 |
| alternate-grid-720 / alt-middle-green-stripe | 1 | 0.6883056356456978 | [925.1777752478056, 360.6328138205082] | ['excessive_residual', 'unstable_center'] | 0.1112673999644362 |
| alternate-grid-720 / alt-middle-green-stripe | 2 | 0.7718648534505126 | [924.9275331499877, 360.04832096893324] | ['excessive_residual', 'unstable_center'] | 0.09667282796008501 |
| alternate-grid-720 / alt-middle-green-stripe | 3 | 0.7253491378019009 | [924.9690514604551, 360.0749805532863] | ['excessive_residual', 'unstable_center'] | 0.05411881535401023 |
| alternate-grid-720 / alt-middle-left-purple-stripe | 1 | 0.562858262432239 | [361.70090950582954, 362.5895819129916] | ['excessive_residual'] | 0.1940357838879572 |
| alternate-grid-720 / alt-middle-left-yellow | 1 | 0.6535622384433536 | [95.95406865794033, 357.9578298978802] | ['excessive_residual'] | 0.02636567399938083 |
| alternate-grid-720 / alt-middle-red-stripe | 1 | 0.3406994106389527 | [630.9766177872867, 361.93287654069326] | [] | 0.04935636867633082 |
| alternate-grid-720 / alt-middle-right-purple | 1 | 1.0711254029513846 | [1197.3131859677032, 363.306551607179] | ['excessive_residual'] | 0.038186833882696315 |
| alternate-grid-720 / alt-top-left-pocket | 1 | 1.9089570377412481 | [102.53776831266484, 105.93685274653077] | ['excessive_residual', 'unstable_center'] | 0.22605378919107894 |
| alternate-grid-720 / alt-top-orange | 1 | 0.42955883126744604 | [360.56236821435624, 97.06080181715743] | [] | 0.03369051956878606 |
| alternate-grid-720 / alt-top-red-stripe | 1 | 0.29155288677202046 | [930.2514266433761, 93.51131720826646] | [] | 0.07128882456706873 |
| alternate-grid-720 / alt-top-right-cue | 1 | 0.27054672808387537 | [1195.3297420096435, 96.78910314454235] | [] | 0.05644750472883992 |
| alternate-grid-720 / alt-top-yellow-stripe | 1 | 0.27485997967220377 | [633.2390616617326, 94.61667283477932] | [] | 0.09833930661622146 |

## Interpretation

The synthetic concentric-circle case has two equally optimal coherent contours. A coherent stronger shadow ring outranks the true silhouette in another counterexample. Thus this search supplies deterministic hypotheses and a test of continuity, not independent silhouette truth. Candidate families beyond the top eight may remain important. Small top-eight center spread is not a calibrated error bound.

The full outputs preserve all eight paths, their sampled points, source projection, circle quality, and per-iteration center spread. Pixel foreground/background support and full iterative aiming are intentionally untested, so a geometric diagnostic pass is not comparable to a fully accepted runtime measurement.

Current refiner accepts zero referenced measurements. This experiment does not fit a physical homography from rejected or diagnostic centers; the existing nine-reference comparison remains a no-fit production result.

Reproduce: `python thoughts/calibration/offline-cyclic-boundary-20261028.py --inputs thoughts/calibration/cyclic-boundary-audit-20261028/inputs.json --output /tmp/cyclic-replay`; run `--self-test` for exact/exhaustive search, ambiguity counterexamples, known-center recovery and support checks. Re-extract compact inputs only from the recorded experiment checkout using `--extract-inputs PATH`. Source hashes are retained in inputs.json.

## Conditional nine-reference comparison

The all-nine best-path homography is hypothetical and includes geometrically rejected paths. It is not a production map. All nine physical reference uncertainties are unknown.

| Diagnostic method | Same-nine LOO RMS mm |
|---|---:|
| Cyclic best path, including failed geometric diagnostics | 8.792404207094663 |
| Earlier segmentation, same nine points | 8.705609317078913 |

Geometrically rejected IDs in the hypothetical fit: ['far-right-purple-middle-row-ball', 'interior-purple-second-left-middle-row-ball', 'bottom-black-second-left-ball']. Full per-point residuals and common passing-subset comparisons are in JSON; neither reduced residual nor subset selection proves accuracy.
