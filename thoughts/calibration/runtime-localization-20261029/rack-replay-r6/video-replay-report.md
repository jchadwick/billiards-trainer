# Recorded-video performance audit

Run `3547819b-bc73-4439-856b-714ef9e209ce` measured **unpaced processing capacity** using `/tmp/billiards-home-5fps-20261029.mkv`.

## Input and runtime

- Video SHA-256: `1b3999b4233e552b4fcd2141aabcfafaceeed5a599dc3a30e45396189e943d4c`
- Selected video stream: ordinal 0, global index 0 (3840×2160, 20 source frames)
- Source presentation duration: 4.00 seconds; ffprobe: `ffprobe version 8.1.1 Copyright (c) 2007-2026 the FFmpeg developers`; ffmpeg: `ffmpeg version 8.1.1 Copyright (c) 2000-2026 the FFmpeg developers`
- Model SHA-256: `63acdbd2bf8952150d4178a3a095ff1914545d858553ce42ef2d7cda231a032b`; checkpoint SHA-256: `f04470378b10f02d7467dad046ad26fb48eaeb1002e7866cfc37f501b478dc0e`; manifest SHA-256: `8470127850eac6527d01e88653dde3da3471ed9ae9620c501a8561142d4c3a0c`
- Provider requested: `CPUExecutionProvider`; observed providers: `CPUExecutionProvider`
- Settings: image size 1280, confidence 0.3, IoU 0.45, maximum detections 16, inference threads 1
- Resource caps: video 8589934592 bytes, probe JSON 536870912 bytes, 20 frames, 67108864 pixels/frame, decoded RGB 8589934592 bytes, subprocess 600000 ms, trace records 0
- Revision: `8129f333a337db1c8c6e7f32869f9dabf914b317`; dirty worktree: true; dirty status SHA-256: `5861edf4bc7329d1faf47d4fb73412ec88caba8cf135a6cbdfddb99a1a68afca`
- Host: `macos aarch64`; processor `Apple M4 Max` at 4512 MHz; logical CPUs: 14; total memory: 38654705664 bytes

## Results

- Processed 5/20 source observations; latest-frame replacements: 0
- Processed rate: 7.10 observations/s; source rate: unavailable observations/s
- Processed-frame source deadlines missed: unavailable

Phase schedule used source PTS (`normalized_source_presentation_timestamps`): 20 selected frames, 20 motion checks (5.00/s), 5 cue observations (1.25/s), 5 full detector calls (1.25/s), and 15 intentional detector skips. Trusted-layout time: 0 ms; recovery delays: 0 samples.

Effective phase schedule:

```json
{
  "aiming": {
    "ball_hz": 1.0,
    "cue_hz": 5.0,
    "motion_hz": 15.0
  },
  "cue_evidence_expiry_ms": 250.0,
  "layout_consensus_count": 3,
  "layout_evidence_expiry_ms": 1500.0,
  "layout_position_tolerance_px": 4.0,
  "motion_evidence_expiry_ms": 200.0,
  "pre_shot": {
    "ball_hz": 1.0,
    "cue_hz": 5.0,
    "motion_hz": 15.0
  },
  "quiet_dwell_ms": 800.0,
  "recovering": {
    "ball_hz": 5.0,
    "cue_hz": 5.0,
    "motion_hz": 15.0
  },
  "shot": {
    "ball_hz": 0.0,
    "cue_hz": 0.0,
    "motion_hz": 15.0
  }
}
```

- Source-to-inference-start latency (0 samples): p50 unavailable, p95 unavailable, p99 unavailable ms
- Source-to-inference-complete latency (0 samples): p50 unavailable, p95 unavailable, p99 unavailable ms
- Mean preprocessing: 10.41 ms; inference: 82.47 ms; decode/NMS: 0.12 ms
- Measured wall time: 704.04 ms; offline decode 465.90 ms was excluded
- Detailed trace records: 0 written, 0 omitted by the configured limit

## Temporal accuracy

Status: **unavailable**.

no reviewed temporal annotation file was supplied

## Limits

Video is decoded into a bounded temporary RGB spool before measurement, so codec startup and offline decoding are excluded. PTS pacing measures detector-side replay freshness; it does not qualify physical camera exposure timing, live decoder contention, viewer/render interference, process CPU/memory trends, or display latency. Use a reviewed annotation set for accuracy claims.

- JSON report: `/tmp/billiards-full-replay-r6/video-replay-report.json`
- Markdown report: `/tmp/billiards-full-replay-r6/video-replay-report.md`
