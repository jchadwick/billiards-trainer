# Real-time state performance audit

Run: `20261019-211614-4cc71a62`
Scenario: `instrumentation-off-after-on`
WebSocket: `ws://127.0.0.1:8000/api/v1/websocket/ws`

| Repetition | Streams | State messages/s | Unique observations/s | Source publication-to-receipt p95 (ms) | Source gaps |
|---:|---|---:|---:|---:|---:|
| 1 | state | 28.36 | 24.23 | 107.60225 | 145 |

Accuracy against labels: unavailable for this live receiver run. Use the recorded-video benchmark and annotated still set for accuracy evidence.
Source timestamps are labeled by the application and indicate shared-memory publication time where available; they do not represent camera exposure time.
Same-host publication-to-receipt timing requires retained server events and a shared perf-counter clock domain.
State publication-to-receipt p95 (ms): unavailable.
Source publication-to-state p95 (ms): unavailable.
Deadline misses above 50.0 ms: {'source_publication_to_state': {'sample_count': 0, 'missed_deadlines': 0}, 'source_publication_to_client_receipt': {'sample_count': 3403, 'missed_deadlines': 3180}, 'state_publication_to_receipt': {'sample_count': 0, 'missed_deadlines': 0}}.
Client trace losses: dropped=0, sampled=0, export-limit-dropped=0, errors=0.
Final server/video exporter status by process: unavailable.
Instrumentation overhead comparison: not requested.

Investigation backlog (hypotheses pending measurement): copies and allocations; image conversion and detector preprocessing; coordinate conversion; JSON serialization; lock and executor waits; unchanged-observation work; polling versus observation-triggered scheduling; shared versus dedicated workers; process isolation; in-process handoff versus bounded loopback TCP; trajectory/video interference; repeated per-client encoding and publishing fan-out.
Accuracy gate: no production candidate is recommendable from this run alone. Checked-in still labels cover ball precision/recall and position error; temporal identity continuity, cue quality and labeled freshness need temporal annotations.
