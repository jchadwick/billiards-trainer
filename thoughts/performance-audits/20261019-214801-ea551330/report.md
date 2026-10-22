# Real-time state performance audit

Run: `20261019-214801-ea551330`
Scenario: `state-audit-heartbeat-smoke`
WebSocket: `ws://127.0.0.1:8000/api/v1/websocket/ws`

| Repetition | Streams | State messages/s | Unique observations/s | Source publication-to-receipt p95 (ms) | Source gaps |
|---:|---|---:|---:|---:|---:|
| 1 | state | 28.18 | 22.72 | 117.731667 | 247 |

Accuracy against labels: unavailable for this live receiver run. Use the recorded-video benchmark and annotated still set for accuracy evidence.
Source timestamps are labeled by the application and indicate shared-memory publication time where available; they do not represent camera exposure time.
Same-host publication-to-receipt timing requires retained server events and a shared perf-counter clock domain.
State publication-to-receipt p95 (ms): 2.578459.
Source publication-to-state p95 (ms): 116.4275.
Deadline misses above 50.0 ms: {'source_publication_to_state': {'sample_count': 2536, 'missed_deadlines': 2490}, 'source_publication_to_client_receipt': {'sample_count': 2536, 'missed_deadlines': 2496}, 'state_publication_to_receipt': {'sample_count': 2536, 'missed_deadlines': 1}}.
Client trace losses: dropped=0, sampled=0, export-limit-dropped=0, errors=0.
Final server/video exporter status by process: mbp-m4max-14.local:12740: dropped=0, sampled=0, export-limit-dropped=0, errors=0; mbp-m4max-14.local:12618: dropped=0, sampled=0, export-limit-dropped=0, errors=0.
Instrumentation overhead comparison: not requested.

Investigation backlog (hypotheses pending measurement): copies and allocations; image conversion and detector preprocessing; coordinate conversion; JSON serialization; lock and executor waits; unchanged-observation work; polling versus observation-triggered scheduling; shared versus dedicated workers; process isolation; in-process handoff versus bounded loopback TCP; trajectory/video interference; repeated per-client encoding and publishing fan-out.
Accuracy gate: no production candidate is recommendable from this run alone. Checked-in still labels cover ball precision/recall and position error; temporal identity continuity, cue quality and labeled freshness need temporal annotations.
