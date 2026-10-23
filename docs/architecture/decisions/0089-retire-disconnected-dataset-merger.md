# ADR-0089: Retire disconnected dataset merger

- Status: accepted
- Date: 2026-10-23
- Deciders: Project maintainers
- Supersedes: ADR-0085's and ADR-0087's temporary retention of the generic dataset merger
- Superseded by: None

## Context

ADR-0087 retired an unowned video-frame sampler but retained the separate
`tools/merge_yolo_datasets.py` helper because it had focused tests and safe
output handling. A broader tool audit found that this helper has no caller
outside its own tests and guide. The tests show generic file manipulation, but
they do not qualify a model-training workflow or prove compatibility with the
Rust runtime.

The merger alphabetically sorts class names. That can assign `ball` class ID 0
and `cue` class ID 1, while the Rust live configuration defaults to ball ID 1
and cue ID 0. The helper also permits arbitrary classes, although the live
detector expects its configured two-class contract. A resulting dataset can
therefore train an artifact whose semantics do not match runtime IDs, without
the utility detecting the mismatch. No in-repository training or evaluation
workflow consumes the merged output.

## Decision

Retire the generic dataset merger, its isolated regression tests, and its usage
guide. Do not replace it with a Rust port: there is no owned training workflow
to support, and a language port would preserve the same unverified class
semantics. Keep the lineage-bound ONNX model exporter because it supplies the
artifact format and provenance required by Rust live inference. Keep reviewed
video annotation and temporal evaluation as the supported source of accuracy
evidence.

If a concrete training workflow is later adopted, its dataset composition step
must preserve a canonical class ontology and validate the generated model
manifest against the configured Rust cue and ball IDs before training or
export results can be used. That workflow must have an actual consuming task
and end-to-end tests; generic class-name merging alone is insufficient.

## Consequences

### Positive

- The repository no longer advertises an isolated workflow that can silently
  invert runtime class semantics or emit a dataset contract the live detector
  rejects.
- Tests focus on supported Rust inference, annotation, and temporal-evaluation
  behavior rather than a utility with no maintained consumer.
- The ONNX export path and reviewed temporal-label path remain available for
  model qualification.

### Negative

- Combining curated YOLO datasets now requires an external, reviewed workflow.
- A future in-repository training pipeline must define and test dataset class
  identity before it can reuse or replace this capability.

## Alternatives considered

### Repair the merger and retain it

Rejected because there is no active training caller or maintained model-data
workflow. Fixing class ordering would still leave the tool unconnected to
runtime manifest validation and would retain a generic feature the project
does not currently use.

### Port the merger to Rust

Rejected because a Rust implementation would not resolve missing ownership,
class-ontology validation, or end-to-end model qualification. Revisit only if
those requirements arrive with a real workflow.

## Verification

- Repository search finds no active references to the retired merger, guide,
  or regression tests; references in historical ADRs and audit records explain
  the superseded decision.
- Python, Rust, frontend, formatting, typing, and ADR checks pass after
  removing the isolated utility and its tests.
