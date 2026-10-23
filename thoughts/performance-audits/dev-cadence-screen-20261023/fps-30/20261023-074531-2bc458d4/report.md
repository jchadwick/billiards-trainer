# State delivery audit `20261023-074531-2bc458d4`

Mode: **repeated**. Started: 2026-10-23T07:45:31.090Z. Finished: 2026-10-23T07:45:52.370Z.

Warm-up: 5.00s requested, 5.00s observed, 115 state messages, not collected trajectory messages (excluded from measurements).

| Run | State messages | Observed source frames | Incomplete states | Receive p50 / p95 / p99 (ms) |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 335 | 264 | 0 | 44.39 / 45.97 / 50.26 |

## Live pipeline timing

Receiver status: **complete**. 904 records were received and 904 remain in the bounded detail tail; receiver losses: malformed 0, oversized 0, queue-dropped 0, retention-evicted 0. Clock domain: `managed-live-audit-649d92a0-db44-4900-aecf-2520c91ae390`.

| Run | Timing status | Matched frames | Inference p95 (ms) | Timing details |
| ---: | :--- | ---: | ---: | :--- |
| 1 | complete | 264 | 44.75 | `runs/run-001-live-pipeline-timing.jsonl` |

## Provenance

- Input: `ws://127.0.0.1:18081/api/v1/websocket/ws`
- Revision: `475dbaa55922f144e13be413ab318084bfb80cfc`
- Working tree dirty: `true`
- Platform: `macos aarch64`
- Logical CPUs: `14`
- Total memory: `38654705664` bytes
- Maximum retained samples per run: `10000`
- Managed server: `/Users/jchadwick/repos/jchadwick/billiards-trainer/perception/target/release/billiards-server` (PID `55798`), first observed frame `3`, ready after 1.02s across 5 probes; shutdown `terminated_gracefully`
- Live camera profile: `camera-profile-v3:8fb9bc6ae4831044ae8189849ef01b508e6f06cf6b97a192a52a0eb097b0e4f1`; provider selection: `cpu`; model size: `1280`; confidence / IoU: `0.4000` / `0.4500`; ball/cue classes: `1` / `0`
- Effective configuration SHA-256: `b4e79877e0e6b83a4eada66138caa43f008f23bc10482bfba1a989ed61338aa2`
- Capture settings SHA-256: `2c473b75646dd0c607e79f76928d5d035abbf92390976d99d042a093f85724c6`
- Detector model SHA-256: `4ceec94901553ff8a22f71d8cca684ec314a9fec8b7f219c410a5df16be4c4fa`
- Model manifest SHA-256: `8a6408809b7221f6e509c193b0388102d80f599a0c358a92ba5c1d98301c8a64`
- Source checkpoint SHA-256: `f04470378b10f02d7467dad046ad26fb48eaeb1002e7866cfc37f501b478dc0e`
- Detector executable SHA-256: `9b7a0ae255a170b6ce75fb5f1a60ab563f8c6747919749ac8ca8560bd63c3797`
- Capture helper SHA-256: `6d1a5e3c5aa0cbd2e4f38f43a4ef6170e7e9913d65f7546b08a88d2ae90f2078`

## Limitations

- This receiver measures state publication and network delivery; it does not measure camera exposure-to-state latency.
- Detector accuracy is unavailable without temporally aligned ground-truth annotations.
- A network state stream has no input-video content hash.
- State receipt intervals and source-reported ages use their documented clock domains; unrelated clocks are not subtracted.
- The recorded provider is the configured selection; actual per-node accelerator placement and dynamic runtime version are unavailable.
- Managed live process sampling follows the latest detector PID in timing records for each measured window; detector restarts during a window are reported as incomplete process-resource evidence.
- Live-pipeline timing is diagnostic and opt-in; it does not provide temporal ground-truth accuracy evidence or camera-exposure timing.
