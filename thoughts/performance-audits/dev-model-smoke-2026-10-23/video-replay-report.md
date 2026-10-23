# Recorded-video performance audit

Run `31b8d084-0f15-45e5-89f9-34ea9217e6b1` measured **PTS-paced latest-frame replay** using `test_data/demo2_short.mp4`.

## Input and runtime

- Video SHA-256: `4e09e30c49fed7e134f43f276c629394c209757e0440958a51ab148f12f6e388`
- Selected video stream: ordinal 0, global index 0 (1280×720, 1177 source frames)
- Source presentation duration: 39.27 seconds; ffprobe: `ffprobe version 8.1.1 Copyright (c) 2007-2026 the FFmpeg developers`; ffmpeg: `ffmpeg version 8.1.1 Copyright (c) 2000-2026 the FFmpeg developers`
- Model SHA-256: `4ceec94901553ff8a22f71d8cca684ec314a9fec8b7f219c410a5df16be4c4fa`; checkpoint SHA-256: `f04470378b10f02d7467dad046ad26fb48eaeb1002e7866cfc37f501b478dc0e`; manifest SHA-256: `8a6408809b7221f6e509c193b0388102d80f599a0c358a92ba5c1d98301c8a64`
- Provider requested: `CPUExecutionProvider`; observed providers: `CPUExecutionProvider`
- Settings: image size 1280, confidence 0.3, IoU 0.45, maximum detections 300, inference threads backend default
- Resource caps: video 8589934592 bytes, probe JSON 536870912 bytes, 250000 frames, 67108864 pixels/frame, decoded RGB 8589934592 bytes, subprocess 600000 ms, trace records 0
- Revision: `18aff22fd64f00d74a2f0bc97adad860d1c891ed`; dirty worktree: false; dirty status SHA-256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- Host: `macos aarch64`; processor `Apple M4 Max` at 4512 MHz; logical CPUs: 14; total memory: 38654705664 bytes

## Results

- Processed 853/1177 source observations; latest-frame replacements: 324
- Processed rate: 852 observations/s; source rate: 21.70 observations/s
- Processed-frame source deadlines missed: 29.97
- Source-to-inference-start latency (853 samples): p50 19.46, p95 34.50, p99 37.41 ms
- Source-to-inference-complete latency (853 samples): p50 63.22, p95 78.35, p99 82.25 ms
- Mean preprocessing: 0.99 ms; inference: 43.68 ms; decode/NMS: 0.09 ms
- Measured wall time: 39295.63 ms; offline decode 9170.42 ms was excluded
- Detailed trace records: 0 written, 0 omitted by the configured limit

## Temporal accuracy

Status: **unavailable**.

no reviewed temporal annotation file was supplied

## Limits

Video is decoded into a bounded temporary RGB spool before measurement, so codec startup and offline decoding are excluded. PTS pacing measures detector-side replay freshness; it does not qualify physical camera exposure timing, live decoder contention, viewer/render interference, process CPU/memory trends, or display latency. Use a reviewed annotation set for accuracy claims.

- JSON report: `/Users/jchadwick/repos/jchadwick/billiards-trainer/thoughts/performance-audits/dev-model-smoke-2026-10-23/video-replay-report.json`
- Markdown report: `/Users/jchadwick/repos/jchadwick/billiards-trainer/thoughts/performance-audits/dev-model-smoke-2026-10-23/video-replay-report.md`
