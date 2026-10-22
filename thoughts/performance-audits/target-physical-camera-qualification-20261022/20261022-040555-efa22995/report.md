# Rust live camera qualification

Run: `20261022-040555-efa22995`

**Run failed before a complete measurement was produced.**

Error type: `RustLiveStartupError`

Error: Rust live process did not publish a usable localization before the startup deadline

State-event exporter status: `{'run_id': '6c78febc-9c4b-4dd8-b2de-27cfe97a03f5', 'process_id': 1, 'host_id': 'hp-prodesk', 'clock_domain_id': 'Linux:6aecfd3a-7cb5-41e9-89c7-85636f8db1de:clock_gettime(CLOCK_MONOTONIC)', 'clock_implementation': 'clock_gettime(CLOCK_MONOTONIC)', 'accepted_events': 0, 'written_events': 0, 'dropped_events': 0, 'sampled_out_events': 0, 'detailed_sampling': {}, 'export_limit_dropped_events': 0, 'export_errors': 0, 'queue_depth': 0, 'queue_capacity': 2048, 'sample_rate': 1.0, 'max_exported_events': 200000, 'worker_alive': False, 'closed': True}`

No camera image/frame bytes are recorded by this runner. This failure is not an accuracy or performance result.

## Startup timing diagnostic

The camera was pointed at the illuminated table with balls present. This run
used the real two-class model at 1280-pixel input on CPU, with a 100 ms
freshness limit. The audit received 62 stale-after-inference outcomes and a
clean footer, with no audit-record loss; no state or scene was published. This
is diagnostic startup evidence, not a measured performance window or an
accuracy result.

Across those startup observations, parent-read-complete-to-inference-complete
age was 317.3 ms at p50 and 363.4 ms at p95, so all 62 attempts exceeded the
100 ms limit. Model inference took 267.1 ms at p50 and 292.3 ms at p95;
preprocessing took 37.6 ms at p50 and 38.3 ms at p95. Inference is the dominant
measured cost, with preprocessing second. The camera helper's appsink wait is
inside the parent framed-record-read span; those durations must not be added
together.

The helper frame IDs advanced from 0 to 187 while 62 frames reached inference,
consistent with best-effort frame dropping instead of a growing backlog. The
source supplied no verified timestamp, so this does not measure sensor
exposure-to-state latency. The model only detects cue and ball classes; no
numbered-ball or localization accuracy was evaluated. See
[`timing-diagnostic.json`](timing-diagnostic.json) for the bounded per-stage
distribution and provenance.
