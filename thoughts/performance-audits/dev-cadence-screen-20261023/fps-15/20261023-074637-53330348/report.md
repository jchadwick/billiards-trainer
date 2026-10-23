# State delivery audit `20261023-074637-53330348`

Mode: **repeated**. Started: 2026-10-23T07:46:37.208Z. Finished: 2026-10-23T07:47:00.059Z.

Warm-up: 5.00s requested, 5.00s observed, 76 state messages, not collected trajectory messages (excluded from measurements).

| Run | State messages | Observed source frames | Incomplete states | Receive p50 / p95 / p99 (ms) |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 226 | 177 | 0 | 66.65 / 73.85 / 75.38 |

## Live pipeline timing

Receiver status: **complete**. 645 records were received and 645 remain in the bounded detail tail; receiver losses: malformed 0, oversized 0, queue-dropped 0, retention-evicted 0. Clock domain: `managed-live-audit-5759f5ba-9a34-4c9d-a463-51e16de6a182`.

| Run | Timing status | Matched frames | Inference p95 (ms) | Timing details |
| ---: | :--- | ---: | ---: | :--- |
| 1 | complete | 176 | 45.31 | `runs/run-001-live-pipeline-timing.jsonl` |

## Provenance

- Input: `ws://127.0.0.1:18081/api/v1/websocket/ws`
- Revision: `475dbaa55922f144e13be413ab318084bfb80cfc`
- Working tree dirty: `true`
- Platform: `macos aarch64`
- Logical CPUs: `14`
- Total memory: `38654705664` bytes
- Maximum retained samples per run: `10000`
- Managed server: `/Users/jchadwick/repos/jchadwick/billiards-trainer/perception/target/release/billiards-server` (PID `56488`), first observed frame `25`, ready after 2.55s across 9 probes; shutdown `terminated_gracefully`
- Live camera profile: `camera-profile-v3:24dcdb0dae71735e3a4123b9bf331aeb8c2b734ce13df283e0faa90644742a55`; provider selection: `cpu`; model size: `1280`; confidence / IoU: `0.4000` / `0.4500`; ball/cue classes: `1` / `0`
- Effective configuration SHA-256: `f66cf11143282aabc260617b8c0b54641b11818dd818dbdbb845a50a522e4c2a`
- Capture settings SHA-256: `3415a51d296f4ab56deb6388cd4ba042efcb2e78e0a360c589c3b23f002deee7`
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
