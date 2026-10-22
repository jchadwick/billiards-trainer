# Rust detection-batch loopback replay smoke

This is a ten-still, unpaced development replay through the Rust CPU inference binary, length-framed loopback TCP, the Python headless receiver, and the existing Python equivalence scorer.

- Input images and expected batches: 10
- Received batches: 10 (complete=True, ordered=True)
- Received detections: 73
- Receiver inter-arrival p50/p95/p99: 94.847/103.472/103.472 ms
- Rust connection setup: 0.164 ms
- Rust synchronous batch build/serialization/socket-write time: 0.442 ms total
- Rust image-processing window: 938.969 ms total; summed image pipeline 938.480 ms
- Python PT cross-runtime comparison: 73 matched; 0 candidate and 0 reference detections unmatched
- Center difference mean/p95/max: 0.002744/0.015878/0.032512 px

Receipt times use Python's local monotonic clock. Rust and Python clock epochs were not compared, so one-way observation-to-receipt latency is unavailable. These static stills establish transport and output equivalence only; they do not establish physical accuracy, temporal tracking, paced freshness, production-model preprocessing parity, live camera behavior, Linux qualification, slow-consumer isolation, or production safety.

Machine-readable evidence: [Rust report](rust-replay-report.json), [receiver report](receiver-report.json), [received batches](received-detection-batches.jsonl), and [equivalence report](equivalence-20261020T145952.415142Z.json).
