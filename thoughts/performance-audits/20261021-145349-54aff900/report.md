# Real-time state performance audit

Run: `20261021-145349-54aff900`
Scenario: `state-audit-only-sampled-detail`
WebSocket: `ws://127.0.0.1:8000/api/v1/websocket/ws`

| Repetition | Streams | State messages/s | Unique observations/s | Source publication-to-receipt p95 (ms) | Source gaps / restarts / out-of-order |
|---:|---|---:|---:|---:|---:|
| 1 | state | 24.00 | 24.00 | 61.101459 | 23 / 0 / 0 |
| 2 | state | 23.77 | 23.77 | 58.6375 | 25 / 0 / 0 |
| 3 | state | 23.90 | 23.90 | 58.853542 | 18 / 0 / 0 |

## Measured vision-stage work

Stages are ranked by cumulative exclusive time in the retained frame records. This is workload attribution, not a claim that removing a stage yields the same end-to-end speedup.

| Rank | Stage | Count | Exclusive mean (ms) | Exclusive p95 (ms) | Share of measured exclusive stage time |
|---:|---|---:|---:|---:|---:|
| 1 | model_execute | 231 | 12.031 | 16.651 | 52.2% |
| 2 | detector_backend | 231 | 6.846 | 9.653 | 29.7% |
| 3 | tracking | 231 | 1.744 | 2.940 | 7.6% |
| 4 | cue_detection | 231 | 1.014 | 3.177 | 4.4% |
| 5 | masking | 231 | 0.581 | 0.994 | 2.5% |
| 6 | postprocessing | 231 | 0.482 | 0.591 | 2.1% |
| 7 | queue_wait | 231 | 0.157 | 0.046 | 0.7% |
| 8 | classification_and_adaptation | 231 | 0.088 | 0.130 | 0.4% |
| 9 | detector_inference | 231 | 0.058 | 0.075 | 0.3% |
| 10 | result_building | 231 | 0.023 | 0.027 | 0.1% |
| 11 | ball_detection | 231 | 0.018 | 0.025 | 0.1% |
| 12 | detector_cache_lock_wait | 231 | 0.003 | 0.003 | 0.0% |
| 13 | detector_model_lock_wait | 231 | 0.001 | 0.001 | 0.0% |
| 14 | preprocessing | 231 | 0.001 | 0.001 | 0.0% |

Accuracy against labels: unavailable for this live receiver run. Use the recorded-video benchmark and annotated still set for accuracy evidence.
Source timestamps are labeled by the application and indicate shared-memory publication time where available; they do not represent camera exposure time.
Same-host publication-to-receipt timing requires retained server events and a shared perf-counter clock domain.
State publication-to-receipt p95 (ms): 2.399667.
Source publication-to-state p95 (ms): 58.129625.
Deadline misses above 50.0 ms: {'source_publication_to_state': {'sample_count': 2149, 'missed_deadlines': 478}, 'source_publication_to_client_receipt': {'sample_count': 2150, 'missed_deadlines': 547}, 'state_publication_to_receipt': {'sample_count': 2149, 'missed_deadlines': 1}}.
Client trace losses: dropped=0, sampled=0, export-limit-dropped=0, errors=0.
Vision profiler detail sampling (post-window process totals, includes warm-up): run 1: configured detail rate=0.1, process frames=1248, detailed=127, sampled-out=1121, failed=0, retained in history=127/10000; run 2: configured detail rate=0.1, process frames=2232, detailed=250, sampled-out=1982, failed=0, retained in history=250/10000; run 3: configured detail rate=0.1, process frames=3213, detailed=349, sampled-out=2864, failed=0, retained in history=349/10000.
Final server/video exporter status by process: mbp-m4max-14.local:32971: dropped=0, sampled=0, export-limit-dropped=0, errors=0.
Instrumentation overhead comparison: over_budget.

Next experiments (not production recommendations): profile allocations and copies inside the highest-ranked measured stages; compare image conversion and detector preprocessing paths at the same accuracy; measure coordinate conversion, JSON serialization, integration lock waits, and executor waits; quantify repeated work on unchanged observations; compare polling with observation-triggered scheduling; compare shared workers, dedicated workers, and process isolation; compare in-process handoff with bounded loopback TCP using identical payloads; then repeat state-only versus trajectory and video workloads. For every candidate, report expected benefit as a measured bound, accuracy effects, implementation complexity, and a validation experiment.
Accuracy gate: no production candidate is recommendable from this run alone. Checked-in still labels cover ball precision/recall and position error; temporal identity continuity, cue quality and labeled freshness need temporal annotations.
