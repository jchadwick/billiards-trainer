# State delivery audit `20261023-101002-47d4603f`

Mode: **repeated**. Started: 2026-10-23T10:10:02.092Z. Finished: 2026-10-23T10:16:33.369Z.

Warm-up: 30.00s requested, 30.00s observed, 681 state messages, not collected trajectory messages (excluded from measurements).

| Run | State messages | Observed source frames | Incomplete states | Receive p50 / p95 / p99 (ms) |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 2669 | 1858 | 0 | 44.28 / 46.35 / 47.92 |
| 2 | 2661 | 1812 | 0 | 44.45 / 46.43 / 48.10 |
| 3 | 2681 | 1852 | 0 | 44.20 / 46.15 / 47.55 |

## Provenance

- Input: `ws://127.0.0.1:18101/api/v1/websocket/ws`
- Revision: `8b9ec3c62883e36569e68c5d96a961c74105541f`
- Working tree dirty: `true`
- Platform: `macos aarch64`
- Logical CPUs: `14`
- Total memory: `38654705664` bytes
- Maximum retained samples per run: `10000`
- Managed server: `/Users/jchadwick/repos/jchadwick/billiards-trainer/perception/target/release/billiards-server` (PID `65678`), first observed frame `2`, ready after 1.02s across 5 probes; shutdown `terminated_gracefully`
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
