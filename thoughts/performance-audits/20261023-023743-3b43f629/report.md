# State delivery audit `20261023-023743-3b43f629`

Mode: **repeated**. Started: 2026-10-23T02:37:43.239Z. Finished: 2026-10-23T02:37:52.199Z.

| Run | State messages | Observed source frames | Incomplete states | Receive p50 / p95 / p99 (ms) |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 24 | 19 | 0 | 103.24 / 499.28 / 658.43 |

## Provenance

- Input: `ws://127.0.0.1:18081/api/v1/websocket/ws`
- Revision: `b16ee8cca9d1351b000be861104b9f55bf466b1b`
- Working tree dirty: `true`
- Platform: `macos aarch64`
- Logical CPUs: `14`
- Total memory: `38654705664` bytes
- Maximum retained samples per run: `10000`
- Managed server: `/Users/jchadwick/repos/jchadwick/billiards-trainer/perception/target/debug/billiards-server` (PID `10121`), first observed frame `8`, ready after 3.84s across 16 probes; shutdown `terminated_gracefully`

## Limitations

- This receiver measures state publication and network delivery; it does not measure camera exposure-to-state latency.
- Detector accuracy is unavailable without temporally aligned ground-truth annotations.
- A network stream has no input-content hash, and the receiver cannot identify the detector model or runtime.
- State receipt intervals and source-reported ages use their documented clock domains; unrelated clocks are not subtracted.
- Managed-server mode qualifies state delivery and process lifecycle; inference-process resource sampling is not yet included.
