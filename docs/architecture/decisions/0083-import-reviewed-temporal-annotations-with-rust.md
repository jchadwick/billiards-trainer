# ADR-0083: Import reviewed temporal annotations with Rust

- Status: accepted
- Date: 2026-10-23
- Deciders: Project maintainers
- Supersedes: None
- Superseded by: None

## Context

Temporal localization accuracy cannot be judged from speed measurements alone;
it needs labels bound to the exact recorded video and reviewed source frames.
The existing CVAT Video 1.1 importer is a substantial Python program that
converts reviewed CVAT keyframes, per-frame presentation timestamps, source
identity, and optional calibration/session epochs into the engine-neutral
annotation contract scored by the Rust evaluator.

The user has directed that the Python application service may remain during the
migration, while other server-side and significant development tooling moves to
Rust. ADR-0082 already assigns complex audit tooling to Rust and retains Python
references temporarily, but its annotation-tool exception does not fit this
reviewed temporal-label path. The importer is a critical part of accuracy
qualification and should use the same Rust schema and validation rules as the
evaluator. The existing Python importer and tests provide an independent
compatibility reference until parity is demonstrated.

CVAT interpolation is not evidence that a frame was reviewed. Source video
hash, selected stream geometry and frame count, strictly increasing
best-effort presentation timestamps, and explicit keyframes are required to
prevent labels from silently attaching to the wrong video or inventing
unreviewed positions. Annotation files and `ffprobe` output must be bounded,
and publishing the generated file must not overwrite an input or an existing
result.

## Decision

Implement the reviewed CVAT temporal annotation importer as a typed Rust
library module and `billiards-temporal-import-cvat` command. Keep the current
Python importer as a differential oracle until fixture and real-video output
parity has passed; then retire it from supported tooling. Parity was verified
on 2026-10-23 against a generated three-frame MP4 using real `ffprobe`, and the
Python importer and its Python-only tests have now been removed. The thin shell
wrapper is the supported development entry point; parsing, validation,
provenance, and serialization remain in Rust.

The importer must preserve the version-1 annotation contract and require the
operator-attested SHA-256 of the CVAT source video. It must use only explicit
reviewed frame ordinals with corresponding CVAT keyframes, preserve ball
track IDs and visible/occluded/removed lifecycle, preserve cue tip-then-butt
ordering, use selected-stream best-effort presentation timestamps normalized
to the first frame, and bind optional epoch events by full source-frame index.
It must validate the final typed document with the existing Rust temporal
evaluator, cap input and subprocess output, enforce a configurable `ffprobe`
timeout, and atomically create a new output without replacing inputs or an
existing destination.

This focused decision narrows ADR-0082's exception for Python annotation
tooling only for this temporal-label importer. ADR-0082 remains the ownership
decision for performance audits. Model training, dataset preparation, and
other annotation workflows are not moved by this record.

## Consequences

### Positive

- Temporal ground truth and temporal scoring share one typed Rust contract and
  validator, reducing the chance that importer and evaluator disagree.
- Explicit review status, source identity, timestamp origin, and lifecycle
  semantics remain inspectable in every produced annotation document.
- Bounded parsing, probing, and no-clobber output make large annotation jobs
  predictable and protect existing source artifacts.
- The Python implementation served as an independent comparison during
  qualification and was retired only after output parity passed.

### Negative

- The Rust importer adds code and tests to the perception crate and depends on
  an installed `ffprobe` executable for exact stream timestamps.
- During migration, the independent Python implementation and test suite had
  to be maintained alongside Rust until output parity was established.
- A default resource cap may require an explicit larger configured bound for
  unusually large source videos or CVAT exports.

## Alternatives considered

### Keep the importer in Python

Rejected for supported tooling because the operation is central to validating
the Rust localization path, has strict cross-module schema and timestamp
semantics, and is complex enough that a shell wrapper would not provide a real
migration. Python is retained only as a differential reference while the Rust
path is qualified.

### Replace the importer with shell commands

Rejected because shell does not safely implement bounded XML parsing,
timestamp validation, track lifecycle mapping, epoch joins, SHA-256
provenance, and schema validation. Shell remains suitable for passing CLI
arguments to the Rust command.

### Accept interpolated CVAT positions as labels

Rejected because interpolation creates positions not explicitly reviewed by a
human and would overstate temporal accuracy evidence.

## Verification

- Rust tests cover reviewed-keyframe-only behavior, CVAT Video 1.1 validation,
  source hash/geometry/frame count, PTS ordering and normalization, epoch
  ordering and safe revisions, lifecycle and cue ordering, empty-ball opt-in,
  resource bounds, timeout/error handling, input protection, and atomic
  no-clobber output.
- `uv run pytest --no-cov -q backend/tests/unit/test_temporal_annotation_import.py`
  passed all 21 Python reference tests before retirement; the Rust importer
  passed all 10 focused module tests.
- Python and Rust produced semantically identical documents for the same
  generated 100×80, three-frame H.264 input using real `ffprobe` (video SHA-256
  `f21ab49b65c0b2eedc677eb3128df5e38dc96b0ce77021c96613d769d8794a76`; two
  reviewed frames and two ball instances).
- The Rust command and its shell wrapper both completed the generated-video
  import; a second write to the output was rejected without replacing it.
- Run `cargo fmt --check`, strict `cargo clippy --all-targets -- -D warnings`,
  and `cargo test --all-targets`; the implementation commit hook passed each
  check. `bash -n` and shell-wrapper help checks also passed.
- The shared temporal evaluator contract remains unchanged. Held-out human
  label quality and detector accuracy are still unqualified and cannot be
  inferred from importer parity.
