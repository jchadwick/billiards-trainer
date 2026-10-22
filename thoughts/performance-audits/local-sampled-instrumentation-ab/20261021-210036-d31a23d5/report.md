# Real-time state performance audit

Run: `20261021-210036-d31a23d5`
Scenario: `state-audit-only-sampled-default`
WebSocket: `ws://127.0.0.1:8000/api/v1/websocket/ws`

| Repetition | Streams | State messages/s | Unique observations/s | Source publication-to-receipt p95 (ms) | Source gaps / restarts / out-of-order |
|---:|---|---:|---:|---:|---:|
| 1 | state | 24.72 | 24.72 | 71.185 | 67 / 0 / 0 |
| 2 | state | 24.73 | 24.73 | 71.05175 | 61 / 0 / 0 |
| 3 | state | 24.55 | 24.55 | 72.905833 | 78 / 0 / 0 |

## Measured vision-stage work

Stages are ranked by cumulative exclusive time in the retained frame records. This is workload attribution, not a claim that removing a stage yields the same end-to-end speedup.

| Rank | Stage | Count | Exclusive mean (ms) | Exclusive p95 (ms) | Share of measured exclusive stage time |
|---:|---|---:|---:|---:|---:|
| 1 | model_execute | 909 | 25.708 | 28.994 | 88.6% |
| 2 | tracking | 909 | 1.288 | 2.658 | 4.4% |
| 3 | cue_detection | 909 | 0.894 | 2.954 | 3.1% |
| 4 | masking | 909 | 0.450 | 0.483 | 1.5% |
| 5 | postprocessing | 909 | 0.366 | 0.521 | 1.3% |
| 6 | detector_backend | 909 | 0.114 | 0.145 | 0.4% |
| 7 | queue_wait | 909 | 0.067 | 0.026 | 0.2% |
| 8 | classification_and_adaptation | 909 | 0.066 | 0.085 | 0.2% |
| 9 | detector_inference | 909 | 0.042 | 0.049 | 0.1% |
| 10 | result_building | 909 | 0.020 | 0.024 | 0.1% |
| 11 | ball_detection | 909 | 0.014 | 0.018 | 0.0% |
| 12 | detector_cache_lock_wait | 909 | 0.003 | 0.003 | 0.0% |
| 13 | preprocessing | 909 | 0.001 | 0.001 | 0.0% |
| 14 | detector_model_lock_wait | 909 | 0.001 | 0.001 | 0.0% |

Accuracy against labels: unavailable for this live receiver run. Use the recorded-video benchmark and annotated still set for accuracy evidence.
Source timestamps are labeled by the application and indicate shared-memory publication time where available; they do not represent camera exposure time.
Same-host publication-to-receipt timing requires retained server events and a shared perf-counter clock domain.
State publication-to-receipt p95 (ms): 7.994416.
Source publication-to-state p95 (ms): 66.406833.
Deadline misses above 50.0 ms: {'source_publication_to_state': {'sample_count': 888, 'missed_deadlines': 869}, 'source_publication_to_client_receipt': {'sample_count': 8880, 'missed_deadlines': 8665}, 'state_publication_to_receipt': {'sample_count': 888, 'missed_deadlines': 0}}.
Client trace losses: dropped=0, sampled=0, export-limit-dropped=0, errors=0.
Vision profiler detail sampling (post-window process totals, includes warm-up): run 1: configured detail rate=0.1, process frames=4049, detailed=432, sampled-out=3617, failed=0, retained in history=432/10000; run 2: configured detail rate=0.1, process frames=7834, detailed=802, sampled-out=7032, failed=0, retained in history=802/10000; run 3: configured detail rate=0.1, process frames=11598, detailed=1169, sampled-out=10429, failed=0, retained in history=1169/10000.
Detailed event sampling (process totals from final or last resource snapshot, includes warm-up): mbp-m4max-14.local:44220 state.video_frame_published: selected=1192/11824, sampled-out=10632; mbp-m4max-14.local:44290 state.integration_vision_poll: selected=1157/11479, sampled-out=10322; mbp-m4max-14.local:44290 state.integration_observation_processed: selected=1145/11341, sampled-out=10196; mbp-m4max-14.local:44290 state.state_publication_enqueued: selected=1145/11341, sampled-out=10196; mbp-m4max-14.local:44290 state.state_published: selected=1145/11341, sampled-out=10196; mbp-m4max-14.local:44290 state.integration_observation_duplicate: selected=13/134, sampled-out=121; mbp-m4max-14.local:44290 state.websocket_message: selected=1117/11075, sampled-out=9958.
Final server/video exporter status by process: mbp-m4max-14.local:44220: dropped=0, sampled=0, export-limit-dropped=0, errors=0; mbp-m4max-14.local:44290: dropped=0, sampled=0, export-limit-dropped=0, errors=0.
Instrumentation overhead comparison: over_budget.

Next experiments (not production recommendations): profile allocations and copies inside the highest-ranked measured stages; compare image conversion and detector preprocessing paths at the same accuracy; measure coordinate conversion, JSON serialization, integration lock waits, and executor waits; quantify repeated work on unchanged observations; compare polling with observation-triggered scheduling; compare shared workers, dedicated workers, and process isolation; compare in-process handoff with bounded loopback TCP using identical payloads; then repeat state-only versus trajectory and video workloads. For every candidate, report expected benefit as a measured bound, accuracy effects, implementation complexity, and a validation experiment.
Accuracy gate: no production candidate is recommendable from this run alone. Checked-in still labels cover ball precision/recall and position error; temporal identity continuity, cue quality and labeled freshness need temporal annotations.
