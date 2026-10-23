# State delivery audit `20261023-105629-2e773c25`

Mode: **repeated**. Started: 2026-10-23T10:56:29.645Z. Finished: 2026-10-23T11:03:01.464Z.

Warm-up: 30.00s requested, 30.00s observed, 601 state messages, not collected trajectory messages (excluded from measurements).

| Run | State messages | Observed source frames | Incomplete states | Receive p50 / p95 / p99 (ms) |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 2377 | 1604 | 0 | 45.39 / 67.33 / 73.93 |
| 2 | 2377 | 1588 | 0 | 45.75 / 67.03 / 71.81 |
| 3 | 2379 | 1620 | 0 | 45.41 / 67.10 / 72.91 |

## Live pipeline timing

Receiver status: **complete**. 15529 records were received and 15529 remain in the bounded detail tail; receiver losses: malformed 0, oversized 0, queue-dropped 0, retention-evicted 0. Clock domain: `managed-live-audit-5798d0c9-5eb3-43e0-bfde-5a89ae3379ea`.

| Run | Timing status | Matched frames | Inference p95 (ms) | Frame elapsed p95 (ms) | Model exclusive p95 (ms) | Timing details |
| ---: | :--- | ---: | ---: | ---: | ---: | :--- |
| 1 | complete | 1603 | 45.36 | 67.57 | 45.36 | `runs/run-001-live-pipeline-timing.jsonl` |
| 2 | complete | 1587 | 45.67 | 67.48 | 45.67 | `runs/run-002-live-pipeline-timing.jsonl` |
| 3 | complete | 1620 | 45.38 | 67.66 | 45.38 | `runs/run-003-live-pipeline-timing.jsonl` |

## Provenance

- Input: `ws://127.0.0.1:18101/api/v1/websocket/ws`
- Revision: `c2f43fb5b052efd9fd20b9a1d1c1a92a6bb23ff7`
- Working tree dirty: `false`
- Platform: `macos aarch64`
- Logical CPUs: `14`
- Total memory: `38654705664` bytes
- Maximum retained samples per run: `10000`
- Managed server: `/Users/jchadwick/repos/jchadwick/billiards-trainer/perception/target/release/billiards-server` (PID `93975`), first observed frame `10`, ready after 1.53s across 7 probes; shutdown `terminated_gracefully`
- Live camera profile: `camera-profile-v3:5f7a2f79c24fee59c08632588fd235f98f92a2cfd31417b057c4cd1b243ddae5`; provider selection: `cpu`; model size: `1280`; confidence / IoU: `0.3000` / `0.4500`; ball/cue classes: `1` / `0`
- Effective configuration SHA-256: `baeb24f555101f3f91e15b411ef3b14fc103370bea143f0aa803081e9c8ba417`
- Capture settings SHA-256: `061cf6059155fcb759d8046a3538909b9c8478f63f8ad8e9508849b381ba0793`
- Detector model SHA-256: `4ceec94901553ff8a22f71d8cca684ec314a9fec8b7f219c410a5df16be4c4fa`
- Model manifest SHA-256: `8a6408809b7221f6e509c193b0388102d80f599a0c358a92ba5c1d98301c8a64`
- Source checkpoint SHA-256: `f04470378b10f02d7467dad046ad26fb48eaeb1002e7866cfc37f501b478dc0e`
- Detector executable SHA-256: `61e71692c29aa8a34e965f6ece05b207e602578506341de7aec5fbe71082081d`
- Capture helper SHA-256: `2b6d394ed6365ccf5527dc54513faf29262c2869d76a2524490c959458f67235`

## Limitations

- This receiver measures state publication and network delivery; it does not measure camera exposure-to-state latency.
- Detector accuracy is unavailable without temporally aligned ground-truth annotations.
- A network state stream has no input-video content hash.
- State receipt intervals and source-reported ages use their documented clock domains; unrelated clocks are not subtracted.
- The recorded provider is the configured selection; actual per-node accelerator placement and dynamic runtime version are unavailable.
- Managed-server mode qualifies state delivery and process lifecycle; managed detector CPU/RSS sampling is disabled unless --sample-managed-live-process-resources is selected.
- Live-pipeline timing is diagnostic and opt-in; it does not provide temporal ground-truth accuracy evidence or camera-exposure timing.
