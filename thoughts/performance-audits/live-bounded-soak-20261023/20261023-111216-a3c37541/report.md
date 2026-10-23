# State delivery audit `20261023-111216-a3c37541`

Mode: **soak**. Started: 2026-10-23T11:12:16.849Z. Finished: 2026-10-23T11:32:48.550Z.

Warm-up: 30.00s requested, 30.00s observed, 665 state messages, not collected trajectory messages (excluded from measurements).

| Run | State messages | Observed source frames | Incomplete states | Receive p50 / p95 / p99 (ms) |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 26374 | 18081 | 0 | 44.70 / 47.68 / 49.18 |

## Live pipeline timing

Receiver status: **partial**. 54270 records were received and 40000 remain in the bounded detail tail; receiver losses: malformed 0, oversized 0, queue-dropped 0, retention-evicted 14270. Clock domain: `managed-live-audit-b534a811-1de9-4b0e-bf76-ae9628d5b233`.

| Run | Timing status | Matched frames | Inference p95 (ms) | Frame elapsed p95 (ms) | Model exclusive p95 (ms) | Timing details |
| ---: | :--- | ---: | ---: | ---: | ---: | :--- |
| 1 | partial | 13660 | 46.39 | 77.75 | 46.39 | [`runs/run-001-live-pipeline-timing.jsonl.gz`](runs/run-001-live-pipeline-timing.jsonl.gz) |

## Provenance

- Input: `ws://127.0.0.1:8001/api/v1/websocket/ws`
- Revision: `2f1e5af0896fd4c0f4858dd13f416f4e6a2d2b20`
- Working tree dirty: `false`
- Platform: `macos aarch64`
- Logical CPUs: `14`
- Total memory: `38654705664` bytes
- Maximum retained samples per run: `40000`
- Managed server: `/Users/jchadwick/repos/jchadwick/billiards-trainer/perception/target/release/billiards-server` (PID `4289`), first observed frame `6`, ready after 1.27s across 6 probes; shutdown `terminated_gracefully`
- Live camera profile: `camera-profile-v3:8fb9bc6ae4831044ae8189849ef01b508e6f06cf6b97a192a52a0eb097b0e4f1`; provider selection: `cpu`; model size: `1280`; confidence / IoU: `0.3000` / `0.4500`; ball/cue classes: `1` / `0`
- Effective configuration SHA-256: `457414171eb3ce792a0230f40f05306d7b489f1f338ac3c00aa49feb66963b0b`
- Capture settings SHA-256: `2c473b75646dd0c607e79f76928d5d035abbf92390976d99d042a093f85724c6`
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
- Managed live process sampling follows the latest detector PID in timing records for each measured window; detector restarts during a window are reported as incomplete process-resource evidence.
- Live-pipeline timing is diagnostic and opt-in; it does not provide temporal ground-truth accuracy evidence or camera-exposure timing.

## Retained detail traces

The state and timing JSONL detail traces are gzip-compressed to keep the
committed audit evidence compact. Decompress `runs/run-001.jsonl.gz` and
`runs/run-001-live-pipeline-timing.jsonl.gz` to recover the original one-object-
per-line files. The timing trace is the retained 40,000-record tail described
above; it does not include the 14,270 older records evicted at the configured
retention limit.
