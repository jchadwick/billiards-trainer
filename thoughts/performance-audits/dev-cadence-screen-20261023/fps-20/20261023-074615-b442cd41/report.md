# State delivery audit `20261023-074615-b442cd41`

Mode: **repeated**. Started: 2026-10-23T07:46:15.121Z. Finished: 2026-10-23T07:46:36.468Z.

Warm-up: 5.00s requested, 5.00s observed, 101 state messages, not collected trajectory messages (excluded from measurements).

| Run | State messages | Observed source frames | Incomplete states | Receive p50 / p95 / p99 (ms) |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 301 | 259 | 0 | 46.93 / 61.69 / 64.64 |

## Live pipeline timing

Receiver status: **complete**. 807 records were received and 807 remain in the bounded detail tail; receiver losses: malformed 0, oversized 0, queue-dropped 0, retention-evicted 0. Clock domain: `managed-live-audit-e8783070-e470-49f3-9964-77493b410b8a`.

| Run | Timing status | Matched frames | Inference p95 (ms) | Timing details |
| ---: | :--- | ---: | ---: | :--- |
| 1 | complete | 259 | 45.38 | `runs/run-001-live-pipeline-timing.jsonl` |

## Provenance

- Input: `ws://127.0.0.1:18081/api/v1/websocket/ws`
- Revision: `475dbaa55922f144e13be413ab318084bfb80cfc`
- Working tree dirty: `true`
- Platform: `macos aarch64`
- Logical CPUs: `14`
- Total memory: `38654705664` bytes
- Maximum retained samples per run: `10000`
- Managed server: `/Users/jchadwick/repos/jchadwick/billiards-trainer/perception/target/release/billiards-server` (PID `56248`), first observed frame `0`, ready after 1.02s across 5 probes; shutdown `terminated_gracefully`
- Live camera profile: `camera-profile-v3:5f7a2f79c24fee59c08632588fd235f98f92a2cfd31417b057c4cd1b243ddae5`; provider selection: `cpu`; model size: `1280`; confidence / IoU: `0.4000` / `0.4500`; ball/cue classes: `1` / `0`
- Effective configuration SHA-256: `71acd5cb4a2858d6f075cdbf1aea0cc3689f7923ebf2c90a124a7bbd42992812`
- Capture settings SHA-256: `061cf6059155fcb759d8046a3538909b9c8478f63f8ad8e9508849b381ba0793`
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
