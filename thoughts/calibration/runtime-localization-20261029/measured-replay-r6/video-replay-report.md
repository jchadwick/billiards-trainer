# Recorded-video performance audit

Run `8bfbe692-cc46-417b-a5dd-3e0a14cb8051` measured **PTS-paced latest-frame replay** using `/tmp/billiards-measured-still-5fps.mkv`.

## Input and runtime

- Video SHA-256: `7050f2151aa4615c46d9f3d12e5607a1dcf403a7dcdf562a77771a47eef63082`
- Selected video stream: ordinal 0, global index 0 (1280×720, 20 source frames)
- Source presentation duration: 4.00 seconds; ffprobe: `ffprobe version 8.1.1 Copyright (c) 2007-2026 the FFmpeg developers`; ffmpeg: `ffmpeg version 8.1.1 Copyright (c) 2000-2026 the FFmpeg developers`
- Model SHA-256: `63acdbd2bf8952150d4178a3a095ff1914545d858553ce42ef2d7cda231a032b`; checkpoint SHA-256: `f04470378b10f02d7467dad046ad26fb48eaeb1002e7866cfc37f501b478dc0e`; manifest SHA-256: `8470127850eac6527d01e88653dde3da3471ed9ae9620c501a8561142d4c3a0c`
- Provider requested: `CPUExecutionProvider`; observed providers: `CPUExecutionProvider`
- Settings: image size 1280, confidence 0.3, IoU 0.45, maximum detections 16, inference threads 1
- Resource caps: video 8589934592 bytes, probe JSON 536870912 bytes, 20 frames, 67108864 pixels/frame, decoded RGB 8589934592 bytes, subprocess 600000 ms, trace records 0
- Revision: `af1f9055eeabde4e37f3f6bafb2164eeb632eaa5`; dirty worktree: true; dirty status SHA-256: `1abd9a11188a92954b15a700606a1067f180e4c872a8a1a98f776ad4baf78e06`
- Host: `macos aarch64`; processor `Apple M4 Max` at 4512 MHz; logical CPUs: 14; total memory: 38654705664 bytes

## Results

- Processed 11/20 source observations; latest-frame replacements: 0
- Processed rate: 2.75 observations/s; source rate: 5.00 observations/s
- Processed-frame source deadlines missed: 0

Phase schedule used source PTS (`normalized_source_presentation_timestamps`): 20 selected frames, 20 motion checks (5.00/s), 20 cue observations (5.00/s), 11 full detector calls (2.75/s), and 9 intentional detector skips. Trusted-layout time: 2600 ms; recovery delays: 0 samples.

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

- Source-to-inference-start latency (11 samples): p50 8.96, p95 17.11, p99 17.11 ms
- Source-to-inference-complete latency (11 samples): p50 92.62, p95 111.45, p99 111.45 ms
- Mean preprocessing: 1.16 ms; inference: 84.59 ms; decode/NMS: 0.11 ms
- Measured wall time: 3805.02 ms; offline decode 96.29 ms was excluded
- Detailed trace records: 0 written, 0 omitted by the configured limit

## Temporal accuracy

Status: **unavailable**.

no reviewed temporal annotation file was supplied

## Limits

Video is decoded into a bounded temporary RGB spool before measurement, so codec startup and offline decoding are excluded. PTS pacing measures detector-side replay freshness; it does not qualify physical camera exposure timing, live decoder contention, viewer/render interference, process CPU/memory trends, or display latency. Use a reviewed annotation set for accuracy claims.

- JSON report: `/tmp/billiards-measured-full-r6/video-replay-report.json`
- Markdown report: `/tmp/billiards-measured-full-r6/video-replay-report.md`
