# Real-time state performance audit

Run: `20261021-151341-a0226100`
Scenario: `state-audit-only-sampled`
WebSocket: `ws://127.0.0.1:8000/api/v1/websocket/ws`

| Repetition | Streams | State messages/s | Unique observations/s | Source publication-to-receipt p95 (ms) | Source gaps / restarts / out-of-order |
|---:|---|---:|---:|---:|---:|
| 1 | state | 25.57 | 25.57 | 61.605208 | 28 / 0 / 0 |
| 2 | state | 23.93 | 23.93 | 60.492375 | 49 / 0 / 0 |
| 3 | state | 25.70 | 25.70 | 58.401958 | 22 / 0 / 0 |

## Measured vision-stage work

Stages are ranked by cumulative exclusive time in the retained frame records. This is workload attribution, not a claim that removing a stage yields the same end-to-end speedup.

| Rank | Stage | Count | Exclusive mean (ms) | Exclusive p95 (ms) | Share of measured exclusive stage time |
|---:|---|---:|---:|---:|---:|
| 1 | model_execute | 243 | 11.716 | 16.665 | 50.6% |
| 2 | detector_backend | 243 | 6.917 | 9.723 | 29.9% |
| 3 | tracking | 243 | 1.814 | 2.961 | 7.8% |
| 4 | cue_detection | 243 | 1.189 | 3.603 | 5.1% |
| 5 | masking | 243 | 0.542 | 0.928 | 2.3% |
| 6 | postprocessing | 243 | 0.506 | 0.665 | 2.2% |
| 7 | queue_wait | 243 | 0.262 | 0.041 | 1.1% |
| 8 | classification_and_adaptation | 243 | 0.091 | 0.141 | 0.4% |
| 9 | detector_inference | 243 | 0.063 | 0.087 | 0.3% |
| 10 | result_building | 243 | 0.023 | 0.028 | 0.1% |
| 11 | ball_detection | 243 | 0.021 | 0.028 | 0.1% |
| 12 | detector_cache_lock_wait | 243 | 0.003 | 0.004 | 0.0% |
| 13 | detector_model_lock_wait | 243 | 0.001 | 0.002 | 0.0% |
| 14 | preprocessing | 243 | 0.001 | 0.002 | 0.0% |

Accuracy against labels: unavailable for this live receiver run. Use the recorded-video benchmark and annotated still set for accuracy evidence.
Source timestamps are labeled by the application and indicate shared-memory publication time where available; they do not represent camera exposure time.
Same-host publication-to-receipt timing requires retained server events and a shared perf-counter clock domain.
State publication-to-receipt p95 (ms): 2.810958.
Source publication-to-state p95 (ms): 57.790875.
Deadline misses above 50.0 ms: {'source_publication_to_state': {'sample_count': 234, 'missed_deadlines': 59}, 'source_publication_to_client_receipt': {'sample_count': 2256, 'missed_deadlines': 617}, 'state_publication_to_receipt': {'sample_count': 234, 'missed_deadlines': 0}}.
Client trace losses: dropped=0, sampled=0, export-limit-dropped=0, errors=0.
Vision profiler detail sampling (post-window process totals, includes warm-up): run 1: configured detail rate=0.1, process frames=1205, detailed=125, sampled-out=1080, failed=0, retained in history=125/10000; run 2: configured detail rate=0.1, process frames=2199, detailed=250, sampled-out=1949, failed=0, retained in history=250/10000; run 3: configured detail rate=0.1, process frames=3253, detailed=353, sampled-out=2900, failed=0, retained in history=353/10000.
Detailed event sampling (process totals from final or last resource snapshot, includes warm-up): mbp-m4max-14.local:48667 state.integration_vision_poll: selected=395/3661, sampled-out=3266; mbp-m4max-14.local:48667 state.integration_observation_processed: selected=339/3154, sampled-out=2815; mbp-m4max-14.local:48667 state.state_publication_enqueued: selected=339/3154, sampled-out=2815; mbp-m4max-14.local:48667 state.state_published: selected=339/3154, sampled-out=2815; mbp-m4max-14.local:48667 state.integration_observation_duplicate: selected=45/465, sampled-out=420; mbp-m4max-14.local:48667 state.websocket_message: selected=321/3013, sampled-out=2692; mbp-m4max-14.local:48539 state.video_frame_published: selected=377/3518, sampled-out=3141.
Final server/video exporter status by process: mbp-m4max-14.local:48667: dropped=0, sampled=0, export-limit-dropped=0, errors=0; mbp-m4max-14.local:48539: dropped=0, sampled=0, export-limit-dropped=0, errors=0.
Instrumentation overhead comparison: pass.

Next experiments (not production recommendations): profile allocations and copies inside the highest-ranked measured stages; compare image conversion and detector preprocessing paths at the same accuracy; measure coordinate conversion, JSON serialization, integration lock waits, and executor waits; quantify repeated work on unchanged observations; compare polling with observation-triggered scheduling; compare shared workers, dedicated workers, and process isolation; compare in-process handoff with bounded loopback TCP using identical payloads; then repeat state-only versus trajectory and video workloads. For every candidate, report expected benefit as a measured bound, accuracy effects, implementation complexity, and a validation experiment.
Accuracy gate: no production candidate is recommendable from this run alone. Checked-in still labels cover ball precision/recall and position error; temporal identity continuity, cue quality and labeled freshness need temporal annotations.
