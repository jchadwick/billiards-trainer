# Bounded asynchronous scene TCP replay smoke

Historical run from before ADR-0025: this replay used the temporary launch-token handshake, which has been removed. The recorded measurements remain transport and scheduling evidence, not live-camera qualification; the token-related counters describe only that old protocol.

- Run: `scene-async-tcp-replay-20261020T164512Z-10dffc`
- Input: 10 recorded still images paced at 30 fps
- Rust scenes queued: 10; Python accepted: 10
- Rust transport: {"connection_attempts": 1, "connection_failures": 0, "dropped_snapshots": 0, "last_written_sequence": 10, "queued_snapshots": 10, "replaced_pending_snapshots": 0, "successful_connections": 1, "write_failures": 0, "written_snapshots": 10}
- Python transport: {"accepted_snapshots": 10, "authentication_failures": 0, "connection_active_at_capture": false, "connection_count": 1, "latest_source_frame_id": "frame-000010", "latest_transport_sequence": 10, "queue_replacements": 9, "rejected_messages": 0, "run_id": "scene-async-tcp-replay-20261020T164512Z-10dffc", "schema_version": 1, "sequence_gap_total": 0}
- Elapsed replay process time: 11.709 seconds
- One-way latency: unavailable; cross-process monotonic clock epochs are not qualified.

No live camera, temporal labels, physical table calibration, Core/API mapping, WebSocket publication, or physical-position ground truth was exercised. Accuracy and freshness remain unqualified.

Artifacts: [Rust report](rust-report.json), [receiver report](receiver-report.json), and [raw detection batches](detection-batches.jsonl). The launch token is not retained.
