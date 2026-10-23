# State delivery audit `20261023-150130-b603254a`

Mode: **repeated**. Started: 2026-10-23T15:01:30.448Z. Finished: 2026-10-23T15:08:01.759Z.

Warm-up: 30.00s requested, 30.00s observed, 602 state messages (excluded from measurements).

| Run | State messages | Observed source frames | Incomplete states | Receive p50 / p95 / p99 (ms) |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 2398 | 1686 | 0 | 49.99 / 52.38 / 53.69 |
| 2 | 2397 | 1701 | 0 | 49.99 / 52.37 / 53.57 |
| 3 | 2401 | 1681 | 0 | 49.99 / 52.78 / 53.95 |

## Live pipeline timing

Receiver status: **complete**. 15670 records were received and 15670 remain in the bounded detail tail; receiver losses: malformed 0, oversized 0, queue-dropped 0, retention-evicted 0. Clock domain: `managed-live-audit-7aa5dc3a-544f-4d37-8053-e737e7cc69cc`.

| Run | Timing status | Matched frames | Inference p95 (ms) | Frame elapsed p95 (ms) | Model exclusive p95 (ms) | Timing details |
| ---: | :--- | ---: | ---: | ---: | ---: | :--- |
| 1 | complete | 1685 | 45.97 | 47.36 | 45.97 | `runs/run-001-live-pipeline-timing.jsonl` |
| 2 | complete | 1701 | 46.02 | 47.34 | 46.02 | `runs/run-002-live-pipeline-timing.jsonl` |
| 3 | complete | 1680 | 45.81 | 47.06 | 45.81 | `runs/run-003-live-pipeline-timing.jsonl` |

## Provenance

- Input: `ws://127.0.0.1:18080/api/v1/websocket/ws`
- Revision: `fa8c5c1469d95c0b5009bd6f333ff42e9f79e357`
- Working tree dirty: `true`
- Platform: `macos aarch64`
- Logical CPUs: `14`
- Total memory: `38654705664` bytes
- Maximum retained samples per run: `10000`
- Managed server: `/Users/jchadwick/repos/jchadwick/billiards-trainer/perception/target/release/billiards-server` (PID `93667`), first observed frame `1`, ready after 1.02s across 5 probes; shutdown `terminated_gracefully`
- Live camera profile: `camera-profile-v3:5f7a2f79c24fee59c08632588fd235f98f92a2cfd31417b057c4cd1b243ddae5`; provider selection: `cpu`; model size: `1280`; confidence / IoU: `0.4000` / `0.4500`; ball/cue classes: `1` / `0`
- Effective configuration SHA-256: `54f5fc0598cdb3b2a718e89b18bcf43f3981ff5c74e44c6d93b65004b441924c`
- Capture settings SHA-256: `061cf6059155fcb759d8046a3538909b9c8478f63f8ad8e9508849b381ba0793`
- Detector model SHA-256: `4ceec94901553ff8a22f71d8cca684ec314a9fec8b7f219c410a5df16be4c4fa`
- Model manifest SHA-256: `8a6408809b7221f6e509c193b0388102d80f599a0c358a92ba5c1d98301c8a64`
- Source checkpoint SHA-256: `f04470378b10f02d7467dad046ad26fb48eaeb1002e7866cfc37f501b478dc0e`
- Detector executable SHA-256: `8b49eef7f6324b75fc2c6cba475008f95f0cf3e1381c135febc3385281f09013`
- Capture helper SHA-256: `34cdcbb5c02a4fa94bdf918e6ab81887e5efcfff1d548c222ff7f73a681fc0e4`

## Limitations

- This receiver measures state publication and network delivery; it does not measure camera exposure-to-state latency.
- Detector accuracy is unavailable without temporally aligned ground-truth annotations.
- A network state stream has no input-video content hash.
- State receipt intervals and source-reported ages use their documented clock domains; unrelated clocks are not subtracted.
- The recorded provider is the configured selection; actual per-node accelerator placement and dynamic runtime version are unavailable.
- Managed live process sampling follows the latest detector PID in timing records for each measured window; detector restarts during a window are reported as incomplete process-resource evidence.
- Live-pipeline timing is diagnostic and opt-in; it does not provide temporal ground-truth accuracy evidence or camera-exposure timing.
