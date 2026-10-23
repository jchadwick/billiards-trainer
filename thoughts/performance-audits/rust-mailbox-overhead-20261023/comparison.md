# Current Rust mailbox-timing A/B

## Result

The mailbox timing change passed the instrumentation overhead screen on this
development Mac: unique source-frame updates were 21.980/s with timing off and
22.108/s with timing on (+0.58%). Mean per-run p95 read-completion-to-publication
age was 78.33 ms off and 77.05 ms on (-1.62%); local state-receive interval p95
was 46.46 ms off and 46.16 ms on. These are relational results, not evidence
that instrumentation improves performance.

The new measurement identifies the pre-inference delay: p95 capacity-one
receiver mailbox residence was 32.57 ms, while p95 read-completion-to-inference
start was 32.57 ms. Pairing values by frame across all 8,027 measured frames,
the residual after dequeue was 1.7 microseconds at p50, 2.4 microseconds at
p95, and 2.7 microseconds at p99 (maximum 21.1 microseconds). On this workload,
nearly all of the measured pre-inference interval is mailbox residence. The
latest-frame queue is staying bounded; it is replacing old frames while the
roughly 45 ms p95 model call runs.

Unique updates whose ball component was marked `observed` were 15.600/s off
and 15.266/s on (-2.14%). The audit does not pair the exact source frames across
the separate runs or score those detections against ground truth, so this
difference cannot be called an accuracy regression or an instrumentation
throughput cost. The model's state was marked cached, failed, skipped, or
unavailable on roughly 29–31% of distinct updates. This is a useful follow-up
accuracy and coverage question, not a production recommendation.

All three measured windows in both arms had complete state messages, zero
invalid JSON, zero WebSocket sequence gaps, and zero audit record/export loss.
The on arm retained 5,496 unique ball-observed updates and 5,496 timing-matched
frame records. The video source is configured to play at 30 fps; the runtime
delivered about 22 unique source-frame updates/s, with source-sequence gaps
consistent with latest-frame replacement. Camera exposure-to-state latency was
not measured.

## Workload and provenance

- Host: Apple M4 Max, macOS, arm64, 14 logical CPUs, 36 GiB memory.
- Revision: `8e2c049f969c40c8a68bef3b5d16f9073a4cb867` in both arms; both worktrees
  were clean, with matching empty git-status SHA-256
  `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.
- Input: looped `test_data/demo2_short.mp4` through the local one-client MJPEG
  fixture at its configured 30 fps. File SHA-256:
  `4e09e30c49fed7e134f43f276c629394c209757e0440958a51ab148f12f6e388`.
- Runtime/provider: Rust live detector, ONNX Runtime CPU, 1280 model input,
  1280×720 RGB capture output. The camera profile, source checkpoint, capture
  helper, detector executable, model, effective configuration, and capture
  settings hashes matched between arms.
- Effective configuration SHA-256:
  `457414171eb3ce792a0230f40f05306d7b489f1f338ac3c00aa49feb66963b0b`.
- Capture settings SHA-256:
  `2c473b75646dd0c607e79f76928d5d035abbf92390976d99d042a093f85724c6`.
- Detector model SHA-256:
  `4ceec94901553ff8a22f71d8cca684ec314a9fec8b7f219c410a5df16be4c4fa`.
- Detector executable SHA-256:
  `61e71692c29aa8a34e965f6ece05b207e602578506341de7aec5fbe71082081d`.
- Both arms used a 30-second warm-up followed by three 120-second measured
  windows. The on arm differed only by `--live-pipeline-timing-audit`.

| Measure | Audit off | Audit on | Change |
| --- | ---: | ---: | ---: |
| Unique source-frame updates/s | 21.980 | 22.108 | +0.58% |
| Ball-observed updates/s | 15.600 | 15.266 | -2.14% |
| Read-completion-to-publication p95 | 78.33 ms | 77.05 ms | -1.62% |
| State receipt interval p95 | 46.46 ms | 46.16 ms | -0.65% |
| Mailbox residence p95, audited arm | — | 32.57 ms | measured |
| Model execution p95, audited arm | — | 44.96 ms | measured |

## Interpretation and limits

This supports retaining the current audit instrumentation: it stayed within the
2% throughput-loss and p95-latency-increase budgets for this matched local
workload. The dominant observed pipeline costs are model execution and a
capacity-one mailbox wait while processing a 30 fps source. The wait is not an
unbounded backlog, and the result supports the existing best-effort latest
frame contract.

The next experiment should compare 30 fps with 20 fps using this same source,
model, resolution, calibration, and state semantics. Measure unique source
updates, receiver mailbox residence, state age, source-frame replacements, and
ball-observed update coverage in paired runs. Add reviewed temporal labels
before deciding whether any coverage change is an accuracy regression. This
Mac run does not qualify the Linux target, physical camera, Coral/OpenVINO,
numbered-ball recognition, trajectory computation, viewer interference, or
temporal accuracy. The off arm logged one helper-readiness timeout before the
managed detector restarted and passed readiness; it occurred before warm-up
and measured windows. The on arm passed readiness on its first attempt.

## Raw artifacts

- Audit off: [report](off/20261023-103759-73f3e28a/report.md),
  [summary](off/20261023-103759-73f3e28a/summary.json), and bounded detailed
  state traces in `runs/`.
- Audit on: [report](on/20261023-104439-c05107d2/report.md),
  [summary](on/20261023-104439-c05107d2/summary.json), and bounded detailed
  state and pipeline-timing traces in `runs/`.
- Shared validated configuration: [live-config.json](../rust-spans-overhead-20261023/live-config.json).
