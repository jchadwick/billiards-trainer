# ADR-0087: Retire the heuristic video frame dataset creator

- Status: accepted
- Date: 2026-10-23
- Deciders: Project maintainers
- Supersedes: ADR-0085's dataset-frame-extraction disposition only
- Superseded by: None

## Context

ADR-0085 retained `tools/dataset_creator.py` as an optional way to extract
candidate images for model annotation. A tool-by-tool review found no caller
beyond its own README and no focused extraction tests. Its quality, ball-count,
occlusion, and motion-blur fields are unvalidated heuristics and can be
mistaken for annotation evidence. Random sampling raises when the requested
count exceeds the video's frame count, while hybrid sampling divides by zero
for a target count of one. The strategies rely on OpenCV seek/FPS behavior and
can emit approximate timestamps. The current reviewed temporal-label workflow
operates on the source video directly, and no maintained training workflow
depends on the extractor.

The extractor duplicates nearly a thousand lines of one-off frame analysis
without establishing localization accuracy or improving the supported Rust
runtime. Keeping a broken, unowned path beside the reviewed CVAT workflow
creates misleading evidence and maintenance cost.

## Decision

Retire `tools/dataset_creator.py` and its usage guide. Keep the separately
tested `tools/merge_yolo_datasets.py` workflow for combining curated manifests,
and keep the lineage-bound model exporter under its own guide. Use the original
video and the Rust temporal annotation/evaluation tools for reviewed temporal
evidence. Do not replace the extractor with another utility unless a concrete
training workflow needs it and its outputs have focused tests and clear
evidence boundaries.

This narrowly supersedes ADR-0085's decision to retain the frame extractor;
ADR-0085 remains in force for the other tooling decisions.

## Consequences

### Positive

- No maintained workflow directs users to an untested sampler with invalid
  count edge cases or presents heuristic frame metadata beside reviewed labels.
- Dataset merge and model export remain supported as separate, tested
  authoring capabilities.
- Video-bound temporal annotations retain source identity and timing rather
  than inheriting approximate OpenCV timestamps.

### Negative

- Users who relied on this local frame sampler must select images with another
  reviewed workflow before creating a still-image training set.
- The removed utility's heuristic candidate ranking is no longer available;
  it had no validated quality or accuracy claim.

## Alternatives considered

### Repair and retain the extractor

Rejected because there is no active owner or caller, its heuristics are not
validated, and fixing its count edge cases would still leave approximate
timing and an untested output contract. A future concrete training workflow
can justify a smaller, tested replacement.

### Port the extractor to Rust

Rejected because the tool does not perform live capture, inference, or
reviewed-label scoring. A Rust rewrite would preserve an unsupported workflow
without improving the primary detection path.

## Verification

- Repository searches show no remaining source, script, test, or build target
  references to `dataset_creator.py` or `README_DATASET_CREATOR.md`.
- The focused dataset-merge suite and Python tooling checks pass after removal.
- ADR structure, links, supersession, and register consistency pass validation.
