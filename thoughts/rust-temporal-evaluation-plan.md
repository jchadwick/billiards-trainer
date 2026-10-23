# Rust temporal localization evaluation

## Verification first

The Rust evaluator must match the retained Python temporal evaluator on a
shared, versioned fixture before it can be used to recommend detector changes.
The fixture and focused tests must cover fresh versus cached results, missing
frames, gated global one-to-one ball matching, observation age, identity
switches, occlusion reacquisition, cue presence and geometry, calibration
revision changes, source-session restarts, and invalid or ambiguous input.
Rust command tests must also exercise raw scene JSONL normalization, annotation
and video identity binding, bounded input, and report metadata. Run the Rust
test, format, and strict Clippy gates plus the current Python evaluator tests
until the new CLI is established as the replacement.

## Scope

Port `backend/tools/temporal_evaluation.py` and
`backend/tools/evaluate_rust_scene_snapshots.py` into a Rust library module and
standalone CLI. Keep the Python evaluator as an offline differential oracle
until shared fixtures and file-based parity pass. Preserve the annotation and
scene schemas, freshness semantics, report fields, and existing defaults.
Ball-number classification, model training, annotation authoring, and browser
code are outside this slice.

The Rust CLI will bind annotations to the exact video by SHA-256 and geometry,
normalize validated `SceneSnapshotV1` JSONL using annotation presentation
times, and emit the existing versioned temporal-localization report. It must
never infer exposure time or use Rust monotonic ticks as video presentation
time. Video geometry is obtained through a configurable `ffprobe` executable;
all file, line, and input limits are configurable and reported.

## Implementation order

1. Add typed, strict annotation and normalized-frame parsing in one Rust
   evaluator module.
2. Implement deterministic maximum-cardinality, minimum-distance assignment,
   freshness and temporal identity metrics, and unavailable-metric reporting.
3. Add a bounded raw-scene JSONL adapter using the existing Rust scene
   validator and exact annotation frame identities.
4. Add the file identity/geometry CLI and a thin shell wrapper.
5. Add one shared golden fixture exercised by Rust and Python, then compare
   the Rust report with the Python output on the full focused corpus.
6. Update ADR-0082 implementation evidence after all checks pass.

The Python evaluator will remain available until parity has been demonstrated
and the Rust command has been exercised on a real recorded detector replay.

## Implementation checkpoint — 2026-10-22

The Rust module and `billiards-temporal-evaluate` CLI now implement the
annotation parser, localization/tracking/cue/freshness metrics, global
one-to-one matching, Rust scene-snapshot JSONL adapter, input identity binding,
and bounded `ffprobe` geometry lookup. The shell entry point is
`scripts/evaluate-temporal-localization.sh`; report and scoring logic remain in
Rust. A shared fixture is asserted by both Rust and Python, and the Rust
file-level test verifies video hash/geometry binding with the scene adapter.

Verification completed so far: all 12 focused Rust evaluator tests pass; all
43 focused Python evaluator/benchmark tests pass with `--no-cov`; the complete
Rust all-target suite passes (276 library tests plus all binary and integration
tests), and strict Clippy passes. The isolated Python test command without
`--no-cov` exits nonzero because the repository-wide 80% coverage gate is
intended for the full suite and a single file produces 8.90% total coverage.

This does not yet replace Python. Broader recorded temporal clips and a real
Rust detector replay still need report-by-report comparison. Once those are
qualified, remove the Python evaluator and its compatibility tests, then move
on to the next substantial Python audit tool.
