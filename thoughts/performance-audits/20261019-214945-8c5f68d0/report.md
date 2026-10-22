# Real-time state performance audit

Run: `20261019-214945-8c5f68d0`
Scenario: `state-audit-soak-heartbeat`
WebSocket: `ws://127.0.0.1:8000/api/v1/websocket/ws`

| Repetition | Streams | State messages/s | Unique observations/s | Source publication-to-receipt p95 (ms) | Source gaps |
|---:|---|---:|---:|---:|---:|
| 1 | state | 21.67 | 17.83 | 292.987041 | 9638 |

Accuracy against labels: unavailable for this live receiver run. Use the recorded-video benchmark and annotated still set for accuracy evidence.
Source timestamps are labeled by the application and indicate shared-memory publication time where available; they do not represent camera exposure time.
Same-host publication-to-receipt timing requires retained server events and a shared perf-counter clock domain.
State publication-to-receipt p95 (ms): 2.305333.
Source publication-to-state p95 (ms): 281.107875.
Deadline misses above 50.0 ms: {'source_publication_to_state': {'sample_count': 23763, 'missed_deadlines': 22916}, 'source_publication_to_client_receipt': {'sample_count': 26001, 'missed_deadlines': 25244}, 'state_publication_to_receipt': {'sample_count': 23763, 'missed_deadlines': 12}}.
Client trace losses: dropped=0, sampled=0, export-limit-dropped=0, errors=0.
Final server/video exporter status by process: mbp-m4max-14.local:12618: dropped=0, sampled=0, export-limit-dropped=0, errors=0; mbp-m4max-14.local:12740: dropped=0, sampled=0, export-limit-dropped=16309, errors=0.
Instrumentation overhead comparison: not requested.

Investigation backlog (hypotheses pending measurement): copies and allocations; image conversion and detector preprocessing; coordinate conversion; JSON serialization; lock and executor waits; unchanged-observation work; polling versus observation-triggered scheduling; shared versus dedicated workers; process isolation; in-process handoff versus bounded loopback TCP; trajectory/video interference; repeated per-client encoding and publishing fan-out.
Accuracy gate: no production candidate is recommendable from this run alone. Checked-in still labels cover ball precision/recall and position error; temporal identity continuity, cue quality and labeled freshness need temporal annotations.
