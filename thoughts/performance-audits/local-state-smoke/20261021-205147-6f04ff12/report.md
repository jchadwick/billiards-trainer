# Real-time state performance audit

Run: `20261021-205147-6f04ff12`
Scenario: `state-only-live-freshness-verified`
WebSocket: `ws://127.0.0.1:8000/api/v1/websocket/ws`

| Repetition | Streams | State messages/s | Unique observations/s | Source publication-to-receipt p95 (ms) | Source gaps / restarts / out-of-order |
|---:|---|---:|---:|---:|---:|
| 1 | state | 26.40 | 26.40 | 66.139292 | 6 / 0 / 0 |

## Measured vision-stage work

Vision-stage timing is unavailable in the retained server trace. The run cannot rank detector, preprocessing, tracking, or state-assembly work.

Accuracy against labels: unavailable for this live receiver run. Use the recorded-video benchmark and annotated still set for accuracy evidence.
Source timestamps are labeled by the application and indicate shared-memory publication time where available; they do not represent camera exposure time.
Same-host publication-to-receipt timing requires retained server events and a shared perf-counter clock domain.
State publication-to-receipt p95 (ms): unavailable.
Source publication-to-state p95 (ms): unavailable.
Deadline misses above 50.0 ms: {'source_publication_to_state': {'sample_count': 0, 'missed_deadlines': 0}, 'source_publication_to_client_receipt': {'sample_count': 264, 'missed_deadlines': 259}, 'state_publication_to_receipt': {'sample_count': 0, 'missed_deadlines': 0}}.
Client trace losses: dropped=0, sampled=0, export-limit-dropped=0, errors=0.
Vision profiler detail sampling (post-window process totals, includes warm-up): unavailable.
Detailed event sampling (process totals from final or last resource snapshot, includes warm-up): unavailable.
Final server/video exporter status by process: unavailable.
Instrumentation overhead comparison: not requested.

Next experiments (not production recommendations): profile allocations and copies inside the highest-ranked measured stages; compare image conversion and detector preprocessing paths at the same accuracy; measure coordinate conversion, JSON serialization, integration lock waits, and executor waits; quantify repeated work on unchanged observations; compare polling with observation-triggered scheduling; compare shared workers, dedicated workers, and process isolation; compare in-process handoff with bounded loopback TCP using identical payloads; then repeat state-only versus trajectory and video workloads. For every candidate, report expected benefit as a measured bound, accuracy effects, implementation complexity, and a validation experiment.
Accuracy gate: no production candidate is recommendable from this run alone. Checked-in still labels cover ball precision/recall and position error; temporal identity continuity, cue quality and labeled freshness need temporal annotations.
