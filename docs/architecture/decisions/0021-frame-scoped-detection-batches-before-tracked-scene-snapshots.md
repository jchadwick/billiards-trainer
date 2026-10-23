# ADR-0021: Frame-scoped detection batches before tracked scene snapshots

- Status: accepted
- Date: 2026-10-20
- Deciders: Project maintainers
- Supersedes: None
- Superseded by: None

## Tooling reference note (2026-10-23)

ADR-0085 retires the still-image Python equivalence utility and its focused
tests, which are referenced in this record's historical implementation notes.
The detection-batch contract remains accepted for live pipeline observations
and replay JSONL artifacts. The standalone TCP batch receiver and its
synchronous still-image sender were retired on 2026-10-23 because their only
remaining workflow was the retired equivalence evaluator. Current live
delivery evidence uses the Rust scene-state stream and performance-audit
subscriber; linked old receiver runs are historical evidence.

## Context

Stage 2 of the Rust migration runs the existing detector on recorded frames,
but it has no tracker. The current immutable scene-snapshot v1 contract requires
stable, unique track IDs and includes tracked, occluded, lost, estimate, and
identity semantics. Giving each frame-local detection a synthetic track ID
would falsely claim continuity across frames and undermine localization-first
reporting. Changing v1 would also violate its immutability rule. Stage 2 must
still publish complete, source-timed detections to a receiver so accuracy and
freshness can be measured before tracking is implemented in Stage 3.

## Decision

Add an independent, versioned `detection-batch` contract for the complete model
output from exactly one source frame. Preserve `scene-snapshot v1` unchanged as
the tracked-scene contract. A successful empty batch means inference completed
and found no detections; failed or skipped batches carry an explicit reason and
must not be interpreted as evidence that the table is empty.

Each detection carries its zero-based `detection_index`, model class ID/name,
confidence, and source-image bounding box and center in declared pixel
coordinates. The tuple of engine instance, source session, source frame, and
detection index identifies that detection observation. Its index is stable only
within the immutable batch and must not be interpreted as a cross-frame track
ID or numbered-ball identity. A repeated delivery of the same batch preserves
all source identity and detection indices. Batches include source publication
time and inference completion time with their clock domains and timestamp
stages; publication time never claims camera exposure time.

Stage 2 publishes these batches. Stage 3 introduces tracking and may then
construct scene-snapshot v1 values with actual stable track IDs. Python ingress
must keep the detection-batch and tracked-scene types distinct until that
association exists. The retained public API and frontend contracts remain
unchanged.

## Consequences

### Positive

- Allows immediate location and model-output measurement without fabricated
  track continuity.
- Keeps the existing tracked-scene contract immutable and reusable after the
  Rust tracker exists.
- Makes repeated delivery and empty-versus-failure semantics testable at the
  Stage 2 boundary.

### Negative

- The process boundary temporarily has two related message types and requires
  explicit ingress dispatch and eventual adaptation.
- Detection indices do not identify physical balls across frames; continuity,
  occlusion handling, and identity remain unavailable until later stages.
- A complete batch repeats small frame metadata with each published frame.

## Alternatives considered

### Reuse scene-snapshot v1 with frame-derived track IDs

Rejected because a detection index or source frame key cannot establish that a
physical ball is the same object in another frame. It would encode a false
guarantee in the required `track_id` field.

### Loosen scene-snapshot v1 or introduce scene-snapshot v2 now

Rejected because v1 is explicitly immutable and the Stage 2 runtime has no
tracking semantics to justify a second tracked-scene version. Revisit only if
Stage 3 evidence shows that the tracked-scene contract itself needs new
semantics.

### Wait for tracking before publishing locations

Rejected because it delays the primary localization and its evaluation for an
optional association capability. The complete frame-scoped batch is sufficient
for Stage 2 receiver and accuracy tests.

## Verification

- Shared JSON Schema fixtures validate non-empty success, empty success,
  failure, repeated delivery, and restarted source sessions.
- Rust decode/encode tests round-trip the same fixtures and reject unknown
  fields, unsupported versions, invalid geometry, invalid confidence, and
  duplicate detection indices.
- The opt-in Rust static replay exporter emits one ordered detection-batch v1
  JSONL record per still image. A standalone diagnostic TCP batch receiver and
  synchronous replay sender were later retired under ADR-0085; the JSONL
  artifact remains for inspection and accuracy workflows.
- The initial Stage 2 qualification used a Python headless receiver to validate
  the shared schema, record local monotonic receipts and sequence metrics, and
  preserve payloads for the scorer. A temporary Rust receiver later replaced
  it, then both diagnostic receiver paths were retired when their only
  maintained consumer disappeared.
- The retired Rust receiver tests covered shared fixtures, payload preservation,
  identity/schema failures, sequence gaps and duplicates, bounded details,
  slow-read timeouts, loopback enforcement, and command output. Those checks are
  historical; live-scene delivery continues to have separate transport tests.
- A historical real-model ten-image CPU replay delivered 10 ordered, complete batches and
  73 detections through Rust TCP to the Python receiver. The existing
  cross-runtime scorer matched all 73 against Python PT with no unmatched
  detections and 0.0325 px maximum center difference. Receiver inter-arrival
  p50/p95/p99 was 94.847/103.472/103.472 ms in this unpaced ten-image smoke;
  it is not a real-time latency measurement.
- Producer and receiver monotonic epochs were not proven comparable, so
  one-way observation-to-receipt latency remains unavailable. The initial
  unpaced run did not establish camera freshness, throughput, physical
  accuracy, or platform qualification.
- A follow-up opt-in `--replay-rate-fps` mode now presents decoded recorded
  images on a monotonic schedule through a capacity-one latest-frame mailbox.
  It gives each batch a synthetic `frame_arrival` timestamp after file decode,
  keeps source frame IDs independent from contiguous publication sequence,
  and reports decode, replacement, scheduling lateness, source-to-inference
  latency percentiles, and processed rate. It does not label this as exposure
  time or a camera timestamp.
- In one 30 fps CPU smoke over ten still files, the producer decoded all ten,
  the mailbox replaced two pending images, and eight batches were processed.
  Source file readiness averaged 24.51 fps and missed all ten synthetic
  deadlines; source-arrival-to-inference-complete p50/p95/p99 was
  71.71/94.08/94.08 ms across eight samples. The existing PT scorer matched
  all 46 detections for those eight processed images, with zero unmatched and
  0.000245 px maximum center difference. This verifies skip accounting and
  cross-runtime scoring for the processed subset; it does not show that the
  model is physically accurate, that camera frames can be read at 30 fps, or
  that the system publishes locations to a client at this rate. See the
  [paced replay report](../../../thoughts/performance-audits/paced-latest-frame-smoke-20261020-r2/report.md),
  [run metadata](../../../thoughts/performance-audits/paced-latest-frame-smoke-20261020-r2/run-metadata.json),
  and [PT equivalence result](../../../thoughts/performance-audits/paced-latest-frame-smoke-20261020-r2/equivalence/equivalence-20261020T151758.327711Z.md).
- Tests demonstrate that batch-local indices may recur in a later source frame
  without being treated as track identity.
- Historical Stage 2 replay and the headless receiver preserved session/frame
  identity, source timestamp semantics, ordering, completeness, and gaps for a
  static single-connection prototype. The live scene transport separately
  covers bounded asynchronous delivery, slow-client isolation, reconnect
  behavior, and live freshness.
- Stage 3 tracking tests prove that only genuinely associated observations
  become scene-snapshot v1 track IDs.

## Related decisions and implementation tracking

ADR-0010 defines immutable tracked scene snapshots; ADR-0019 separates ball
number evidence from location tracks; ADR-0020 defines source-observation
deduplication. This record clarifies the Stage 2 boundary without superseding
those decisions. See the [migration plan](../../../thoughts/rust-perception-migration-plan.md),
the [detection-batch contract](../../../perception/contracts/detection-batch/v1/README.md),
and its [shared schema](../../../perception/contracts/detection-batch/v1/detection-batch.schema.json).

The contract implementation remains in [Rust types and semantic validation](../../../perception/src/detection_batch.rs), [live capture pipeline](../../../perception/src/capture_pipeline.rs), and [recorded inference JSONL exporter](../../../perception/src/inference/runtime.rs). The former standalone TCP receiver, sender path, and receiver tests were removed in the 2026-10-23 tooling cleanup; their old smoke reports remain historical evidence only. Shared fixtures remain covered by the [Python schema tests](../../../backend/tests/unit/test_perception_detection_batch_contract.py). The separate Python equivalence utility was retired under ADR-0085; the earlier comparison result remains historical rather than a current workflow. Live capture, bounded asynchronous delivery, reconnect freshness, Linux qualification, temporal/physical accuracy, and production freshness remain open. Authentication is explicitly excluded by ADR-0025.
