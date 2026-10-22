# Rust live camera qualification

Run: `20261022-041120-212da569`

**Run failed before a complete measurement was produced.**

Error type: `RustLiveStartupError`

Error: Rust live process exited with status 1 before a usable localization: GStreamer capture probe failed: timed out waiting for first physical camera sample
Error: live capture failed before readiness: capture source preflight failed: source open failed: helper did not send a valid ready record: unexpected EOF before helper-ready record

State-event exporter status: `{'run_id': '000970fa-9416-4ebe-8e00-80c56765b303', 'process_id': 1, 'host_id': 'hp-prodesk', 'clock_domain_id': 'Linux:6aecfd3a-7cb5-41e9-89c7-85636f8db1de:clock_gettime(CLOCK_MONOTONIC)', 'clock_implementation': 'clock_gettime(CLOCK_MONOTONIC)', 'accepted_events': 0, 'written_events': 0, 'dropped_events': 0, 'sampled_out_events': 0, 'detailed_sampling': {}, 'export_limit_dropped_events': 0, 'export_errors': 0, 'queue_depth': 0, 'queue_capacity': 2048, 'sample_rate': 1.0, 'max_exported_events': 200000, 'worker_alive': False, 'closed': True}`

No camera image/frame bytes are recorded by this runner. This failure is not an accuracy or performance result.
