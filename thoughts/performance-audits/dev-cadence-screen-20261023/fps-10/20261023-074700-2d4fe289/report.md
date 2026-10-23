# State delivery audit `20261023-074700-2d4fe289`

Mode: **repeated**. Started: 2026-10-23T07:47:00.786Z. Finished: 2026-10-23T07:47:22.067Z.

Warm-up: 5.00s requested, 5.00s observed, 51 state messages, not collected trajectory messages (excluded from measurements).

| Run | State messages | Observed source frames | Incomplete states | Receive p50 / p95 / p99 (ms) |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 151 | 100 | 0 | 99.96 / 107.06 / 110.20 |

## Live pipeline timing

Receiver status: **complete**. 419 records were received and 419 remain in the bounded detail tail; receiver losses: malformed 0, oversized 0, queue-dropped 0, retention-evicted 0. Clock domain: `managed-live-audit-34af636c-1524-450b-b7a8-d689302aa40a`.

| Run | Timing status | Matched frames | Inference p95 (ms) | Timing details |
| ---: | :--- | ---: | ---: | :--- |
| 1 | complete | 100 | 45.55 | `runs/run-001-live-pipeline-timing.jsonl` |

## Provenance

- Input: `ws://127.0.0.1:18081/api/v1/websocket/ws`
- Revision: `475dbaa55922f144e13be413ab318084bfb80cfc`
- Working tree dirty: `true`
- Platform: `macos aarch64`
- Logical CPUs: `14`
- Total memory: `38654705664` bytes
- Maximum retained samples per run: `10000`
- Managed server: `/Users/jchadwick/repos/jchadwick/billiards-trainer/perception/target/release/billiards-server` (PID `56738`), first observed frame `1`, ready after 1.02s across 5 probes; shutdown `terminated_gracefully`
- Live camera profile: `camera-profile-v3:fc59ce84b528132afc57a33f1f193e9aecc74100e97c144b7eed95d0e1feaaa0`; provider selection: `cpu`; model size: `1280`; confidence / IoU: `0.4000` / `0.4500`; ball/cue classes: `1` / `0`
- Effective configuration SHA-256: `2351ce354166620d16471d4b9a62602a5cd1294bba93b7a51b5785a896397fd4`
- Capture settings SHA-256: `2864f5d43df1ba9a72096cf8be101368e5e9000c583b7e2e948b124810be1294`
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
