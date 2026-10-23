# Recorded-video performance audit

Run `smoke-20261023-real-recording` measured **PTS-paced latest-frame replay** using `test_data/demo2_short.mp4`.

## Input and runtime

- Video SHA-256: `4e09e30c49fed7e134f43f276c629394c209757e0440958a51ab148f12f6e388`
- Selected video stream: ordinal 0, global index 0 (1280×720, 1177 source frames)
- Source presentation duration: 39.27 seconds; ffprobe: `ffprobe version 8.1.1 Copyright (c) 2007-2026 the FFmpeg developers`; ffmpeg: `ffmpeg version 8.1.1 Copyright (c) 2000-2026 the FFmpeg developers`
- Model SHA-256: `7beb3af7923290a5524fe1744de13bd66f4e43ea8a5048347bc8a12c51a1e4b5`; checkpoint SHA-256: `f04470378b10f02d7467dad046ad26fb48eaeb1002e7866cfc37f501b478dc0e`; manifest SHA-256: `74c229ff058ec138b6194593921e5768dbeada092440ffd2c2a9f84e6b0dc707`
- Provider requested: `CPUExecutionProvider`; observed providers: `CPUExecutionProvider`
- Settings: image size 1280, confidence 0.3, IoU 0.45, maximum detections 16, inference threads backend default
- Resource caps: video 8589934592 bytes, probe JSON 536870912 bytes, 250000 frames, 67108864 pixels/frame, decoded RGB 8589934592 bytes, subprocess 600000 ms, trace records 0
- Revision: `a9972a8b62e58f608e856fc21d4fdea30a200c49`; dirty worktree: true; dirty status SHA-256: `fc059fad3b1438156fb16b343d0f1609359369a4a106671484e9403ad639169f`
- Host: `macos aarch64`; processor `Apple M4 Max` at 4512 MHz; logical CPUs: 14; total memory: 38654705664 bytes

## Results

- Processed 863/1177 source observations; latest-frame replacements: 314
- Processed rate: 861 observations/s; source rate: 21.95 observations/s
- Processed-frame source deadlines missed: 29.97
- Source-to-inference-start latency (863 samples): p50 18.33, p95 33.68, p99 35.26 ms
- Source-to-inference-complete latency (863 samples): p50 62.22, p95 77.46, p99 79.33 ms
- Mean preprocessing: 0.98 ms; inference: 43.59 ms; decode/NMS: 0.09 ms
- Measured wall time: 39309.61 ms; offline decode 7226.91 ms was excluded
- Detailed trace records: 0 written, 0 omitted by the configured limit

## Temporal accuracy

Status: **unavailable**.

no reviewed temporal annotation file was supplied

## Limits

Video is decoded into a bounded temporary RGB spool before measurement, so codec startup and offline decoding are excluded. PTS pacing measures detector-side replay freshness; it does not qualify physical camera exposure timing, live decoder contention, viewer/render interference, process CPU/memory trends, or display latency. Use a reviewed annotation set for accuracy claims.

- JSON report: `/Users/jchadwick/repos/jchadwick/billiards-trainer/thoughts/performance-audits/video-replay-smoke-20261023/real-recording/video-replay-report.json`
- Markdown report: `/Users/jchadwick/repos/jchadwick/billiards-trainer/thoughts/performance-audits/video-replay-smoke-20261023/real-recording/video-replay-report.md`
