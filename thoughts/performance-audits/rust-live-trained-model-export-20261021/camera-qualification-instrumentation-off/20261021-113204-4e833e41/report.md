# Rust live camera qualification

Run: `20261021-113204-4e833e41`

**Run failed before a complete measurement was produced.**

Error type: `RuntimeError`

Error: Rust live supervisor lost readiness during warm-up: {"failure": null, "last_observation_age_seconds": 0.156185459, "process_return_code": 1, "process_status": "failed", "source_health": "process_failed", "stderr_tail": "Error: live capture frame processing failed\n\nCaused by:\n    0: failed to build complete captured scene\n    1: source-time tracking error: ActiveTrackCapacityExceeded\n"}

State-event exporter status: `{'run_id': '6800a4df-2345-4503-a49f-411e160237f6', 'process_id': 88720, 'host_id': 'mbp-m4max-14.local', 'accepted_events': 0, 'written_events': 0, 'dropped_events': 0, 'sampled_out_events': 0, 'export_limit_dropped_events': 0, 'export_errors': 0, 'queue_depth': 0, 'queue_capacity': 2048, 'sample_rate': 1.0, 'max_exported_events': 200000, 'worker_alive': False, 'closed': True}`

No camera image/frame bytes are recorded by this runner. This failure is not an accuracy or performance result.
