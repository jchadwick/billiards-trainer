# Paced latest-frame replay smoke

**Run:** `paced-latest-frame-smoke-20261020-r2`
**Purpose:** verify that the recorded-image harness paces inputs, replaces stale
pending work under overload, preserves source-frame gaps, and reports freshness
separately from detector runtime.

This is a short implementation smoke, not a camera benchmark or a production
qualification. It used the real two-class checkpoint and the existing ten
still-image corpus, read from disk during a requested 30 fps schedule. The
worker ran the 1280-square ONNX export with the CPU provider, confidence 0.4,
IoU 0.45, a 300-detection cap, and one warm-up inference. The replay has no
temporal labels, camera, tracking, Python gateway, WebSocket, or receipt client.

## Results

The source scheduled and decoded all 10 files. The one-frame latest-work slot
replaced 2 pending frames before inference, so the worker processed 8. Source
indices `4` and `6` were skipped; batch publication sequences remained
contiguous while source frame IDs retained those gaps. The Python scorer joined
the JSONL batches to their aggregate results and compared all 46 detections on
the 8 processed images with the retained PT model: 46 matched, none unmatched,
maximum center difference 0.000245 px. This is model-output equivalence on the
processed stills, not evidence that either model is physically accurate.

| Measurement | Result |
|---|---:|
| Requested source pace | 30 fps |
| Source files scheduled / decoded / processed | 10 / 10 / 8 |
| Latest-slot replacements | 2 |
| Source-read schedule misses | 10 of 10 |
| Mean / maximum source-read lateness | 79.51 / 108.90 ms |
| Observed source-read rate | 24.51 fps |
| Processed rate, including replay elapsed time | 16.29 fps |
| Source arrival to inference start, p50 / p95 / p99 (8 samples) | 23.67 / 48.59 / 48.59 ms |
| Source arrival to inference completion, p50 / p95 / p99 (8 samples) | 71.71 / 94.08 / 94.08 ms |
| Mean image decode / preprocessing / inference / decode+NMS | 44.08 / 9.84 / 46.08 / 0.09 ms |
| Mean summed image-pipeline duration | 100.09 ms |
| PT-to-Rust output comparison | 46 matched; 0 unmatched; max center delta 0.000245 px |

The source timestamp is a synthetic monotonic `frame_arrival` recorded after
the image file has been decoded. It is not camera exposure time. The file
decoder took a mean 44.08 ms per input, so this source adapter itself could not
meet a 30 fps schedule. The CPU model call averaged 46.08 ms, which also
exceeds a 33.3 ms frame interval on this machine. These short samples identify
candidate constraints; they do not isolate live camera acquisition or establish
a steady-state rate. The earlier auto-provider run used a different unpaced
workload and is not a like-for-like comparison.

The retained still labels remain the strongest accuracy evidence: the earlier
review found only 18 of 52 annotated centers within 2 px, with incomplete
labels and no temporal or physical-coordinate ground truth. Output agreement
with the same checkpoint does not repair that accuracy shortfall. See the
[Rust inference prototype report](../rust-inference-prototype-20261020/report.md).

The sender wrote local JSONL synchronously. This smoke did not exercise TCP,
the Python headless receiver, observation-to-receipt latency, bounded-memory
soak, Linux, live capture, or cross-clock subtraction. The 8-sample tail values
are weak estimates and must not be treated as stable p95/p99 production
latencies.

## What to improve next

1. **Localization evidence first.** Add temporally annotated recordings with
   adjudicated ball centers, dense/occluded/moving cases, and calibrated table
   coordinates. Set physical and image-space tolerances before ranking model or
   runtime changes. Existing still data cannot yet support a production
   accuracy recommendation.
2. **Measure the intended source path.** Replay already-decoded frames or use
   one real capture owner so compressed-file decoding does not dominate the
   synthetic input schedule. Preserve separate source-read, preprocessing,
   inference, publication, and receiver timing.
3. **Compare providers at identical pace.** Run CPU and the observed auto
   provider on the same decoded frames, then repeat the planned three 120-second
   runs and 20-minute bounded-memory run. Keep the 30 fps target and a higher
   stress rate as separate cases; retain all drop and freshness counts.
4. **Complete the location delivery path.** Use the existing headless receiver
   on paced output, then verify source identity, ordering, completeness and
   gaps. One-way source-to-receipt latency remains unavailable until Rust and
   Python monotonic clock comparability is proven.
5. **Keep identity separate.** The current `cue`/`ball` classes cannot identify
   6-, 7- or 8-balls. Evaluate identity with separate labeled evidence after
   anonymous localization is accurate and fresh.

The machine-readable [Rust report](rust-replay-report.json),
[detection batches](detection-batches.jsonl), [run provenance and input
hashes](run-metadata.json), and [Python PT equivalence report](equivalence/equivalence-20261020T151758.327711Z.md)
preserve the exact configuration and evidence. This run leaves the migration
and the 80% backend coverage gate open.
