# Detection batch JSONL export smoke

**Run ID:** `detection-batch-jsonl-smoke-20261020-fixed`

**Purpose:** Verify that the Rust recorded-image executable can turn the real
two-class model's complete outputs into ordered `detection-batch v1` records.
This is a contract/export smoke, not a latency baseline or live pipeline test.

## Result

The run emitted 10 ordered, schema-valid batches with 73 total detections. The
source frame IDs were scoped by the explicit engine and input-session IDs.
Every still-image source timestamp remained null. Rust's contract decoder
round-tripped records in unit tests; Python's Draft 2020-12 JSON Schema
validator accepted every emitted line. The aggregate report records the same
10 batches and input/model hashes.

The retained Python equivalence tool also consumed the JSONL together with the
aggregate report, checked per-frame detections and provenance, and compared the
results against the source PT checkpoint. It paired all 73 detections with no
unmatched outputs; center-error mean/p95/max was 0.00274/0.01588/0.03251 px.
The versioned [comparison JSON](equivalence-final/equivalence-20261020T133321.599979Z.json)
and [Markdown report](equivalence-final/equivalence-20261020T133321.599979Z.md)
preserve this evidence. This is output equivalence under the fixed 1280-square
still-image harness, not ground-truth accuracy or a live receiver test.

The first real-model run exposed a numeric boundary issue: detector centers
were calculated in `f32`, then compared with the serialized `f64` box midpoint.
The exporter now derives the emitted center from the emitted bounding box in
`f64`, preserving the contract's geometry invariant without weakening its
validation tolerance. The second run passed on all 10 frames.

## Measurement limits

The executable ran in the Cargo development profile on unpaced 4K still images.
It processed each file once and synchronously built, serialized, wrote, and
flushed each record. The report measured mean batch export work at 1.53 ms for
this run, while the combined image decode, preprocessing, inference, and
decode/NMS pipeline averaged 761.32 ms per file. These values are retained as
smoke diagnostics only: ten samples, development-mode preprocessing, cold
filesystem effects, and synchronous file output make them unsuitable for a
performance comparison or real-time claim.

There was no camera, paced replay, Python headless receiver, socket transport,
cross-process clock comparison, frontend, trajectory worker, or temporal
accuracy measurement. A successful JSONL export proves the artifact contract
and serialization path only. The aggregate JSON and line count must agree
before treating an output as complete; failed runs can leave a valid JSONL
prefix.

## Reproduction inputs

- Model: temporary ONNX export at `/tmp/billiards-rust-stage2/billiards-yolov8n.onnx`, SHA-256 `77cbca76dfa71714dfba121193d164492ef9773a0ccc577bc7df16b52c2fb6ca`.
- Source checkpoint: `billiards-yolov8n.pt`, SHA-256 `f04470378b10f02d7467dad046ad26fb48eaeb1002e7866cfc37f501b478dc0e`.
- Input: the ten-image `backend/vision/test_data` corpus, previously hashed in the [Rust inference prototype report](../rust-inference-prototype-20261020/report.md).
- Configuration: CPU provider, 1280 square input, confidence 0.4, IoU 0.45, maximum 300 detections, one warm-up inference.
- Engine instance: `rust-audit-smoke-fixed-20261020`.
- Source session: `backend-test-data-20261020`.

The exact per-image hashes and recorded stage timings are in
[`audit-report.json`](audit-report.json). Emitted messages are in
[`detection-batches.jsonl`](detection-batches.jsonl). This machine's temporary
ONNX file is outside the repository; reproduce it using the retained exporter
and the matching checkpoint manifest before running again.
