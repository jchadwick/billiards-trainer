# Recorded-video performance audit

Run `temporal-localization-prep-20261023` measured **PTS-paced latest-frame replay** using `test_data/demo2_short.mp4`.

## Input and runtime

- Video SHA-256: `4e09e30c49fed7e134f43f276c629394c209757e0440958a51ab148f12f6e388`
- Selected video stream: ordinal 0, global index 0 (1280×720, 1177 source frames)
- Source presentation duration: 39.27 seconds; ffprobe: `ffprobe version 8.1.1 Copyright (c) 2007-2026 the FFmpeg developers`; ffmpeg: `ffmpeg version 8.1.1 Copyright (c) 2000-2026 the FFmpeg developers`
- Model SHA-256: `7beb3af7923290a5524fe1744de13bd66f4e43ea8a5048347bc8a12c51a1e4b5`; checkpoint SHA-256: `f04470378b10f02d7467dad046ad26fb48eaeb1002e7866cfc37f501b478dc0e`; manifest SHA-256: `74c229ff058ec138b6194593921e5768dbeada092440ffd2c2a9f84e6b0dc707`
- Provider requested: `CPUExecutionProvider`; observed providers: `CPUExecutionProvider`
- Settings: image size 1280, confidence 0.3, IoU 0.45, maximum detections 16, inference threads backend default
- Resource caps: video 8589934592 bytes, probe JSON 536870912 bytes, 250000 frames, 67108864 pixels/frame, decoded RGB 8589934592 bytes, subprocess 600000 ms, trace records 0
- Revision: `6f1503c39a22293f163542feb838eee8b3ef1dc6`; dirty worktree: false; dirty status SHA-256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- Host: `macos aarch64`; processor `Apple M4 Max` at 4512 MHz; logical CPUs: 14; total memory: 38654705664 bytes

## Results

- Processed 810/1177 source observations; latest-frame replacements: 367
- Processed rate: 20.58 observations/s; source rate: 29.97 observations/s
- Processed-frame source deadlines missed: 809
- Source-to-inference-start latency (810 samples): p50 20.93, p95 37.07, p99 40.25 ms
- Source-to-inference-complete latency (810 samples): p50 64.84, p95 80.97, p99 83.95 ms
- Mean preprocessing: 0.99 ms; inference: 43.58 ms; decode/NMS: 0.09 ms
- Measured wall time: 39315.62 ms; offline decode 22733.20 ms was excluded
- Detailed trace records: 0 written, 0 omitted by the configured limit

## Temporal accuracy

Status: **unavailable**.

no reviewed temporal annotation file was supplied

## Limits

Video is decoded into a bounded temporary RGB spool before measurement, so codec startup and offline decoding are excluded. PTS pacing measures detector-side replay freshness; it does not qualify physical camera exposure timing, live decoder contention, viewer/render interference, process CPU/memory trends, or display latency. Use a reviewed annotation set for accuracy claims.

- JSON report: `/Users/jchadwick/repos/jchadwick/billiards-trainer/thoughts/performance-audits/temporal-localization-prep-20261023/video-replay-report.json`
- Markdown report: `/Users/jchadwick/repos/jchadwick/billiards-trainer/thoughts/performance-audits/temporal-localization-prep-20261023/video-replay-report.md`
