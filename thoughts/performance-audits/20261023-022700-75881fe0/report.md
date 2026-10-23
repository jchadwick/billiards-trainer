# State delivery audit `20261023-022700-75881fe0`

Mode: **repeated**. Started: 2026-10-23T02:27:00.718Z. Finished: 2026-10-23T02:27:09.683Z.

| Run | State messages | Observed source frames | Incomplete states | Receive p50 / p95 / p99 (ms) |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 43 | 42 | 0 | 101.08 / 201.14 / 385.67 |

## Provenance

- Input: `ws://127.0.0.1:18081/api/v1/websocket/ws`
- Revision: `cb1cacc639798733b238cdd8d93f1806e76d4da3`
- Working tree dirty: `true`
- Platform: `macos aarch64`
- Logical CPUs: `14`
- Total memory: `38654705664` bytes
- Maximum retained samples per run: `10000`
- Managed server: `/Users/jchadwick/repos/jchadwick/billiards-trainer/perception/target/debug/billiards-server` (PID `98499`), first observed frame `9`, ready after 3.85s across 15 probes; shutdown `exited_during_shutdown`

## Limitations

- This receiver measures state publication and network delivery; it does not measure camera exposure-to-state latency.
- Detector accuracy is unavailable without temporally aligned ground-truth annotations.
- A network stream has no input-content hash, and the receiver cannot identify the detector model or runtime.
- State receipt intervals and source-reported ages use their documented clock domains; unrelated clocks are not subtracted.
- Managed-server mode qualifies state delivery and process lifecycle; inference-process resource sampling is not yet included.
