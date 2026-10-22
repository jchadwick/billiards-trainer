# Real-time state performance audit

Run: `20261019-212225-55fb39be`
Scenario: `state-audit-soak`
WebSocket: `ws://127.0.0.1:8000/api/v1/websocket/ws`

| Repetition | Streams | State messages/s | Unique observations/s | Source publication-to-receipt p95 (ms) | Source gaps |
|---:|---|---:|---:|---:|---:|
| 1 | state | 7.95 | 6.78 | 163.301125 | 1405 |

**INVALID RUN:** The state subscriber stopped receiving after roughly seven minutes while the server kept publishing. Its aggregate rate includes the silent tail and is not a valid delivery-throughput measurement.


Accuracy against labels: unavailable for this live receiver run. Use the recorded-video benchmark and annotated still set for accuracy evidence.
Source timestamps are labeled by the application and indicate shared-memory publication time where available; they do not represent camera exposure time.
Same-host publication-to-receipt timing requires retained server events and a shared perf-counter clock domain.
State publication-to-receipt p95 (ms): 2.322833.
Source publication-to-state p95 (ms): 256.342333.
Deadline misses above 50.0 ms: {'source_publication_to_client_receipt': {'missed_deadlines': 8877, 'sample_count': 9541}, 'source_publication_to_state': {'missed_deadlines': 24802, 'sample_count': 26945}, 'state_publication_to_receipt': {'missed_deadlines': 5, 'sample_count': 9541}}.
Client trace losses: dropped=0, sampled=0, export-limit-dropped=0, errors=0.
Final server/video exporter status by process: mbp-m4max-14.local:88105: dropped=0, sampled=0, export-limit-dropped=0, errors=0; mbp-m4max-14.local:88225: dropped=0, sampled=0, export-limit-dropped=0, errors=0.
Instrumentation overhead comparison: not requested.

Investigation backlog (hypotheses pending measurement): copies and allocations; image conversion and detector preprocessing; coordinate conversion; JSON serialization; lock and executor waits; unchanged-observation work; polling versus observation-triggered scheduling; shared versus dedicated workers; process isolation; in-process handoff versus bounded loopback TCP; trajectory/video interference; repeated per-client encoding and publishing fan-out.
Accuracy gate: no production candidate is recommendable from this run alone. Checked-in still labels cover ball precision/recall and position error; temporal identity continuity, cue quality and labeled freshness need temporal annotations.
