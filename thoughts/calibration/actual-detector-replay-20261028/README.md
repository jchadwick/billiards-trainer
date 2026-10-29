# Actual saved-frame detector replay — 2026-10-28

The verified source checkpoint runs on the development CPU with the configuration defaults: 1280-square input, confidence .3, IoU .45, maximum 16 detections. These are real model proposals rather than the previous manually supplied crops. This does not establish compiled Coral parity, precise ball centers, physical X/Y or target performance.

| Saved image | Ball detections | Other detections | Manual crop matches | Unmatched manual crops |
|---|---:|---:|---:|---|
| measured-grid-720 | 15 | 0 | 15 | none |
| alternate-grid-720 | 15 | 0 | 15 | none |
| framing-after-4k | 3 | 2 | 3 | none |

Manual matching is deterministic greedy IoU ≥ .10 against previously reviewed coarse crops; its matches/misses are not independent detection accuracy. All boxes, classes, confidence values, source dimensions, encoded/decoded hashes and exact midpoints are saved. Identical-pixel repeated inference produced identical proposal lists on all three stills.

## Bounded saved native4K video replay

The first 4 seconds of demo3 are sampled at 5 Hz (20 source frames). First-frame montage confirms the same home table, with touching rack balls; no clean segment was selected after scoring. Ball counts are `[14, 14, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 14, 15, 15, 15, 15, 15, 14, 15]`. The simple image-space association records 2 missing-previous and 17 new-track events (including initial births). IDs may swap in the rack; no semantic identity or temporal accuracy is claimed.

The selected clip and the other saved clips were used during INT8 representative calibration, so these frames are not independent heldout model data. Sampling at 5 Hz is a replay workload, not proof that the deployed system can sustain 5 FPS. The input video is preserved locally and its SHA-256 and sampled decoded-pixel hashes are recorded.

## Reproduce

Use `/tmp/billiards-detector-replay-env/bin/python thoughts/calibration/offline-detector-replay-20261028.py --self-test`, then the same script with `--output /tmp/actual-detector-replay-check`. The checkpoint/video dependencies are intentionally not copied into research commits. The exact checkpoint hash must match the local conversion manifest. The compact `results.json` preserves actual proposals for later refiner/contour replay. `timings.json` records variable development-host telemetry separately. No camera, model training, runtime configuration or deployment was used.
