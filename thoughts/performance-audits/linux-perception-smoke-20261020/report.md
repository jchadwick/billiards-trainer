# Linux ARM64 ONNX inference smoke

**Run:** `linux-perception-smoke-20261020`
**Purpose:** verify that the existing Rust ONNX inference slice builds and runs
with the real billiards model in a Linux ARM64 environment, then compare its
outputs to the retained Python PT model.

This ran in a Debian 13 ARM64 container (`rust:1.92-trixie`, rustc 1.92.0)
through the Linux container engine on the Apple M4 Max development machine.
It provides Linux userspace/build/runtime evidence; the container is not a
separate physical Linux target and its timings are not target-hardware
performance evidence.

The run used the same hash-verified ONNX export and source checkpoint as the
macOS replay, all ten repository still images, the CPU execution provider, one
warm-up inference, 1280-square input, confidence 0.4, IoU 0.45 and a 300
detection limit. It was an unpaced recorded-image run with synchronous JSONL
batch output. It did not acquire a camera or publish to a TCP receiver.

## Results

The binary completed on Linux ARM64, observed `CPUExecutionProvider` for all
214 profiled nodes, and emitted ten complete detection batches covering 73
detections. The retained Python equivalence tool processed the same ten images
and matched all 73 detections with no unmatched outputs. Mean/p95/maximum
center difference was 0.0027/0.0159/0.0325 px. This demonstrates cross-runtime
output equivalence on this static corpus, not physical accuracy: the existing
still-label review found only 18 of 52 centers within 2 px, with incomplete
labels and no temporal or calibrated-coordinate ground truth.

| Linux container measurement | Result |
|---|---:|
| Image decode, mean | 48.52 ms |
| Preprocessing, mean | 10.24 ms |
| ONNX Runtime inference, mean | 77.61 ms |
| Decode and NMS, mean | 0.09 ms |
| Summed per-image pipeline, mean | 136.46 ms |
| Total recorded-image pipeline time | 1364.59 ms |
| Synchronous batch export | 2.22 ms |

These are ten-image diagnostic values in a virtualized container. Do not use
them to compare Linux with macOS, claim a sustained frame rate, infer physical
Linux hardware performance, or set a production freshness budget. Image
decoding and inference dominate this short run; optimize them only after the
accuracy and real capture paths are measured independently.

The raw Rust report retains `/repo/...` container input paths. The scorer
matched those paths using source-frame index plus encoded-image hash. That
case exposed a mapping gap for duplicate, byte-identical input images when a
run's absolute path root differs from the comparison machine; the scorer now
uses the source index only after validating the indexed input hash and any
resolvable reported path. Reports without a valid index still fail on an
ambiguous content match.

## Remaining evidence gaps

- No physical Linux host or accelerator was tested; only ARM64 Linux userspace
  execution under virtualization was covered.
- This is unpaced still-image processing, not source-to-client real-time
  freshness. There is no camera, receiver, or comparable cross-process clock.
- The model remains a generic `cue`/`ball` detector and cannot distinguish
  numbered balls.
- The known still-label accuracy result is insufficient to recommend the model
  for production localization.
- No repeated 120-second runs or 20-minute bounded-memory soak were performed.

Raw results and provenance are preserved in the [Rust JSON report](rust-replay-report.json),
[detection-batch JSONL](detection-batches.jsonl), [run metadata](run-metadata.json),
and [Python equivalence report](equivalence/equivalence-20261020T152704.641143Z.md).
