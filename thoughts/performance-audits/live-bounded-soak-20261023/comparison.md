# Twenty-minute Rust live bounded-memory soak

## Result

The managed Rust gateway, live detector, and saved-video MJPEG input completed
one 1,200-second measured run after the requested 30-second warm-up. The server
shut down gracefully. The state receiver collected 26,374 complete messages:
26,119 were fresh, unique source-frame updates (21.77/s), with a p95/p99 local
receive interval of 47.68/49.18 ms. It observed no invalid JSON, incomplete
states, or uncorrelated state messages.

The ball component was marked `observed` in 18,081 states and `cached` in 8,293
(31.4% of all state messages); 255 messages were stale. The cue component was
observed in 12,016 states. These are freshness/status counts, not detection
precision or recall. Without temporally reviewed labels, they do not establish
whether cached positions or absent cue observations were correct.

The Rust detector child had 1,189 available one-second resource samples. Its
resident memory ranged from 253.4 to 299.5 MiB, starting at 295.7 MiB and ending
at 275.1 MiB. No sustained upward RSS trend appeared in this process over this
run. CPU use was high: p50 455.1% and p95 465.6% per core, or roughly 4.6
logical cores. This sampler follows only the detector child; it does not
represent total host, gateway, fixture, or browser memory/CPU.

The producer completed all 27,028 inference attempts with no inference
failures. Eleven frames were rejected as stale. The state trace retained all
26,374 messages. Timing export stayed within its configured 40,000-record and
64-MiB bounds, but received 54,270 records and evicted 14,270 old records from
retention. Producer drops, queue drops, malformed records, and oversized
records were zero. The timing section is therefore correctly marked partial;
its retained tail is diagnostic and must not be used as a complete 20-minute
stage distribution.

## Interpretation and limits

- This run qualifies sustained delivery and detector-child memory behavior for
  the development Mac, CPU provider, ONNX model, and saved-video MJPEG fixture.
  It does not qualify the Linux target, physical webcam buffering/exposure
  age, or OpenVINO/Coral behavior.
- The 21.77 fresh state updates/s and 18,081 states with an observed ball
  component are different measurements. Investigate ball-component cache age
  alongside reviewed moving-ball annotations before recommending a sampling
  rate.
- Detector CPU use is a strong candidate for a CPU/provider or inference-cadence
  experiment, but neither is production-qualified without temporal accuracy
  evidence.
- The audit input is a WebSocket stream and does not carry the source-video
  content hash. The prior matched configuration used the looped
  `test_data/demo2_short.mp4` fixture; the live report records model/config
  hashes but explicitly marks network-stream content identity unavailable.
- Browser/viewer load, allocations, event-loop lag, and physical-display timing
  were not measured. The 20-minute results must not be used to claim those
  properties.

## Artifacts

- [Generated state-delivery report](20261023-111216-a3c37541/report.md)
- [Generated versioned summary](20261023-111216-a3c37541/summary.json)
- [State trace, gzip-compressed JSONL](20261023-111216-a3c37541/runs/run-001.jsonl.gz)
- [Retained timing tail, gzip-compressed JSONL](20261023-111216-a3c37541/runs/run-001-live-pipeline-timing.jsonl.gz)

The trace files are byte-for-byte gzip-compressed copies of the runner's JSONL
output. Decompress them to recover one JSON object per line.
