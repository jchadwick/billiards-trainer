# State delivery audit `20261023-103759-73f3e28a`

Mode: **repeated**. Started: 2026-10-23T10:37:59.091Z. Finished: 2026-10-23T10:44:31.905Z.

Warm-up: 30.00s requested, 30.00s observed, 681 state messages, not collected trajectory messages (excluded from measurements).

| Run | State messages | Observed source frames | Incomplete states | Receive p50 / p95 / p99 (ms) |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 2664 | 1900 | 0 | 44.38 / 46.31 / 48.14 |
| 2 | 2653 | 1850 | 0 | 44.54 / 46.93 / 48.86 |
| 3 | 2676 | 1866 | 0 | 44.30 / 46.15 / 47.55 |

## Provenance

- Input: `ws://127.0.0.1:18101/api/v1/websocket/ws`
- Revision: `8e2c049f969c40c8a68bef3b5d16f9073a4cb867`
- Working tree dirty: `false`
- Platform: `macos aarch64`
- Logical CPUs: `14`
- Total memory: `38654705664` bytes
- Maximum retained samples per run: `10000`
- Managed server: `/Users/jchadwick/repos/jchadwick/billiards-trainer/perception/target/release/billiards-server` (PID `83704`), first observed frame `2`, ready after 2.55s across 11 probes; shutdown `terminated_gracefully`
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
- Managed-server mode qualifies state delivery and process lifecycle; managed detector CPU/RSS sampling is disabled unless --sample-managed-live-process-resources is selected.
