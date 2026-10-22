# Real-time state performance audit

Run: `20261021-212922-98f45b17`
Scenario: `fixed-segment-state-audit-5pct-config-refresh`
WebSocket: `ws://127.0.0.1:8000/api/v1/websocket/ws`

| Repetition | Streams | State messages/s | Unique observations/s | Source publication-to-receipt p95 (ms) | Source gaps / restarts / out-of-order |
|---:|---|---:|---:|---:|---:|
| 1 | state | 25.03 | 25.03 | 70.730667 | 71 / 0 / 0 |
| 2 | state | 25.12 | 25.12 | 70.811833 | 69 / 0 / 0 |
| 3 | state | 25.03 | 25.03 | 70.925083 | 78 / 0 / 0 |

## Measured vision-stage work

Stages are ranked by cumulative exclusive time in the retained frame records. This is workload attribution, not a claim that removing a stage yields the same end-to-end speedup.

| Rank | Stage | Count | Exclusive mean (ms) | Exclusive p95 (ms) | Share of measured exclusive stage time |
|---:|---|---:|---:|---:|---:|
| 1 | model_execute | 449 | 25.073 | 27.156 | 90.4% |
| 2 | tracking | 449 | 0.953 | 1.887 | 3.4% |
| 3 | cue_detection | 449 | 0.706 | 3.001 | 2.5% |
| 4 | masking | 449 | 0.432 | 0.459 | 1.6% |
| 5 | postprocessing | 449 | 0.319 | 0.371 | 1.2% |
| 6 | detector_backend | 449 | 0.102 | 0.126 | 0.4% |
| 7 | classification_and_adaptation | 449 | 0.061 | 0.069 | 0.2% |
| 8 | detector_inference | 449 | 0.040 | 0.046 | 0.1% |
| 9 | queue_wait | 449 | 0.021 | 0.023 | 0.1% |
| 10 | result_building | 449 | 0.020 | 0.023 | 0.1% |
| 11 | ball_detection | 449 | 0.013 | 0.015 | 0.0% |
| 12 | detector_cache_lock_wait | 449 | 0.002 | 0.003 | 0.0% |
| 13 | preprocessing | 449 | 0.001 | 0.001 | 0.0% |
| 14 | detector_model_lock_wait | 449 | 0.001 | 0.001 | 0.0% |

Accuracy against labels: unavailable for this live receiver run. Use the recorded-video benchmark and annotated still set for accuracy evidence.
Source timestamps are labeled by the application and indicate shared-memory publication time where available; they do not represent camera exposure time.
Same-host publication-to-receipt timing requires retained server events and a shared perf-counter clock domain.
State publication-to-receipt p95 (ms): 7.70175.
Source publication-to-state p95 (ms): 63.753792.
Deadline misses above 50.0 ms: {'source_publication_to_state': {'sample_count': 440, 'missed_deadlines': 420}, 'source_publication_to_client_receipt': {'sample_count': 9022, 'missed_deadlines': 8830}, 'state_publication_to_receipt': {'sample_count': 440, 'missed_deadlines': 1}}.
Client trace losses: dropped=0, sampled=0, export-limit-dropped=0, errors=0.
Vision profiler detail sampling (post-window process totals, includes warm-up): run 1: configured detail rate=0.05, process frames=4223, detailed=233, sampled-out=3990, failed=0, retained in history=233/10000; run 2: configured detail rate=0.05, process frames=8070, detailed=410, sampled-out=7660, failed=0, retained in history=410/10000; run 3: configured detail rate=0.05, process frames=11912, detailed=585, sampled-out=11327, failed=0, retained in history=585/10000.
Detailed event sampling (process totals from final or last resource snapshot, includes warm-up): mbp-m4max-14.local:70970 state.integration_vision_poll: selected=573/11788, sampled-out=11215; mbp-m4max-14.local:70970 state.integration_observation_processed: selected=574/11635, sampled-out=11061; mbp-m4max-14.local:70970 state.state_publication_enqueued: selected=574/11635, sampled-out=11061; mbp-m4max-14.local:70970 state.state_published: selected=574/11635, sampled-out=11061; mbp-m4max-14.local:70970 state.integration_observation_duplicate: selected=6/149, sampled-out=143; mbp-m4max-14.local:70970 state.websocket_message: selected=555/11263, sampled-out=10708; mbp-m4max-14.local:70941 state.video_frame_published: selected=591/12075, sampled-out=11484.
Final server/video exporter status by process: mbp-m4max-14.local:70970: dropped=0, sampled=0, export-limit-dropped=0, errors=0; mbp-m4max-14.local:70941: dropped=0, sampled=0, export-limit-dropped=0, errors=0.
Instrumentation overhead comparison: pass.

Next experiments (not production recommendations): profile allocations and copies inside the highest-ranked measured stages; compare image conversion and detector preprocessing paths at the same accuracy; measure coordinate conversion, JSON serialization, integration lock waits, and executor waits; quantify repeated work on unchanged observations; compare polling with observation-triggered scheduling; compare shared workers, dedicated workers, and process isolation; compare in-process handoff with bounded loopback TCP using identical payloads; then repeat state-only versus trajectory and video workloads. For every candidate, report expected benefit as a measured bound, accuracy effects, implementation complexity, and a validation experiment.
Accuracy gate: no production candidate is recommendable from this run alone. Checked-in still labels cover ball precision/recall and position error; temporal identity continuity, cue quality and labeled freshness need temporal annotations.
