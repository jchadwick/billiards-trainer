# Capture transport microbenchmark — run 5

## Result

On this Apple M4 Max (14 logical CPUs, 36 GiB RAM, macOS 26.6.2), the supervised stdout-pipe arm moved a repeated 3840×2160 RGB8 frame template at a median **324.37 frame records/s** across three 3,000-frame repetitions. Its median sampled frame-receipt p95 was **3.361 ms**. The in-process arm completed a median **227,490 frame ownership handoffs/s**, with a sampled receipt p95 around **0.005 ms**.

The in-process result is not a payload-copy or memory-bandwidth measurement: it transfers `Vec` ownership through a bounded channel, while the pipe arm writes and reads the 24,883,200-byte payload. The reported logical payload MiB/s is only configured frame volume divided by elapsed time.

| Repetition | In-process frames/s | Handoff p95 (ms) | Pipe frames/s | Pipe logical MiB/s | Pipe receipt p95 (ms) |
|---:|---:|---:|---:|---:|---:|
| 1 | 225013.35 | 0.005 | 326.52 | 7,748.36 | 3.316 |
| 2 | 225207.61 | 0.005 | 324.37 | 7,697.54 | 3.361 |
| 3 | 206881.70 | 0.007 | 324.37 | 7,697.41 | 3.382 |

At a 60 fps target, the measured pipe throughput provides about **5.4×** capacity in this synthetic workload. The pipe p95 is about **6.7%** of the 50 ms reference budget. This suggests pipe transfer by itself may fit a 60 fps budget on this machine, but it is not enough evidence to choose the production capture architecture.

## What this tells us

- Process-boundary transfer is measurable: the 4K pipe arm took roughly 3 ms per parent-side receipt, compared with microseconds for in-process ownership handoff.
- The isolated pipe throughput is above 60 fps with headroom. Actual camera reads, format conversion, detector execution, calibration, and state delivery will consume the same frame budget and were not part of this test.
- This only supports a follow-up experiment: compare the real supervised capture helper with the same camera or recorded frame source, and measure source-to-parent latency, end-to-end state freshness, CPU/RSS, and detection accuracy. Include a same-payload in-process or shared-memory path if the source supports it.
- These results do not establish which transport is best for Linux or other hardware. Repeat there before making a cross-platform architecture decision.

## Measurement limits

- The payload was deterministic and repeated; there was no camera or detector, no real pixel variation, and no per-frame payload checksum in the timed path. The setup checksum identifies the repeated template but does not prove every timed payload survived unchanged.
- Browser rendering, client delivery, capture freshness, CPU time, resident memory, and operating-system pipe capacity were not measured.
- Startup-to-ready values include different arm-specific preparation work and are not comparable as an architecture startup result.
- 200 of 3,000 measured frame receipts were retained per arm per repetition; the benchmark reported no sampling loss. This is a focused microbenchmark, not the planned 120-second production audit.

## Reproducibility

Revision `e748c0f99e8e8c089633aa282154a05697adb243`; worktree was dirty (225 paths at capture). See `run-metadata.json` for the worktree fingerprint, toolchain, hardware, invocation, and source hashes. The JSON result is `capture-transport.json`.

Verification: the Rust all-target suite passed (119 library tests and integration targets), the transport integration suite passed, Clippy passed with warnings denied, and the Python collector’s 26 focused tests, Ruff, Black, and mypy passed.
