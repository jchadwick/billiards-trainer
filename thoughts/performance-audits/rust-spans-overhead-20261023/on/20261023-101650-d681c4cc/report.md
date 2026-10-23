# State delivery audit `20261023-101650-d681c4cc`

Mode: **repeated**. Started: 2026-10-23T10:16:50.886Z. Finished: 2026-10-23T10:23:22.459Z.

Warm-up: 30.00s requested, 30.00s observed, 675 state messages, not collected trajectory messages (excluded from measurements).

| Run | State messages | Observed source frames | Incomplete states | Receive p50 / p95 / p99 (ms) |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 2676 | 1855 | 0 | 44.23 / 46.15 / 47.79 |
| 2 | 2676 | 1850 | 0 | 44.31 / 46.10 / 47.25 |
| 3 | 2673 | 1819 | 0 | 44.28 / 46.23 / 47.67 |

## Live pipeline timing

Receiver status: **complete**. 17463 records were received and 17463 remain in the bounded detail tail; receiver losses: malformed 0, oversized 0, queue-dropped 0, retention-evicted 0. Clock domain: `managed-live-audit-d8d09ead-b6c8-4aee-a59e-7df6bda828a6`.

| Run | Timing status | Matched frames | Inference p95 (ms) | Frame elapsed p95 (ms) | Model exclusive p95 (ms) | Timing details |
| ---: | :--- | ---: | ---: | ---: | ---: | :--- |
| 1 | complete | 1855 | 44.98 | 77.79 | 44.98 | `runs/run-001-live-pipeline-timing.jsonl` |
| 2 | complete | 1849 | 44.89 | 77.93 | 44.89 | `runs/run-002-live-pipeline-timing.jsonl` |
| 3 | complete | 1819 | 45.03 | 77.47 | 45.03 | `runs/run-003-live-pipeline-timing.jsonl` |

## Provenance

- Input: `ws://127.0.0.1:18101/api/v1/websocket/ws`
- Revision: `8b9ec3c62883e36569e68c5d96a961c74105541f`
- Working tree dirty: `true`
- Platform: `macos aarch64`
- Logical CPUs: `14`
- Total memory: `38654705664` bytes
- Maximum retained samples per run: `10000`
- Managed server: `/Users/jchadwick/repos/jchadwick/billiards-trainer/perception/target/release/billiards-server` (PID `69477`), first observed frame `6`, ready after 1.28s across 6 probes; shutdown `terminated_gracefully`
- Live camera profile: `camera-profile-v3:8fb9bc6ae4831044ae8189849ef01b508e6f06cf6b97a192a52a0eb097b0e4f1`; provider selection: `cpu`; model size: `1280`; confidence / IoU: `0.3000` / `0.4500`; ball/cue classes: `1` / `0`
- Effective configuration SHA-256: `457414171eb3ce792a0230f40f05306d7b489f1f338ac3c00aa49feb66963b0b`
- Capture settings SHA-256: `2c473b75646dd0c607e79f76928d5d035abbf92390976d99d042a093f85724c6`
- Detector model SHA-256: `4ceec94901553ff8a22f71d8cca684ec314a9fec8b7f219c410a5df16be4c4fa`
- Model manifest SHA-256: `8a6408809b7221f6e509c193b0388102d80f599a0c358a92ba5c1d98301c8a64`
- Source checkpoint SHA-256: `f04470378b10f02d7467dad046ad26fb48eaeb1002e7866cfc37f501b478dc0e`
- Detector executable SHA-256: `2663be366a7a66c18af71b3b0ccee2a72fd16e76c5b68a5f48ecc1a6d84a1b60`
- Capture helper SHA-256: `fca26c7a3ea78e9b8560d57f55305c1d34ae12a4437c86a5083e5eaf6f732959`

## Limitations

- This receiver measures state publication and network delivery; it does not measure camera exposure-to-state latency.
- Detector accuracy is unavailable without temporally aligned ground-truth annotations.
- A network state stream has no input-video content hash.
- State receipt intervals and source-reported ages use their documented clock domains; unrelated clocks are not subtracted.
- The recorded provider is the configured selection; actual per-node accelerator placement and dynamic runtime version are unavailable.
- Managed-server mode qualifies state delivery and process lifecycle; managed detector CPU/RSS sampling is disabled unless --sample-managed-live-process-resources is selected.
- Live-pipeline timing is diagnostic and opt-in; it does not provide temporal ground-truth accuracy evidence or camera-exposure timing.
