# ADR-0017: Retain Python tools as cross-runtime acceptance tests

- Status: proposed
- Date: 2026-10-19
- Deciders: Project maintainers
- Supersedes: None
- Superseded by: None

## Context

The project has Python benchmarks, still labels, annotation tools, a real
WebSocket headless collector, replay integration tests and report formats.
Some tests construct Python internals directly, so they cannot execute Rust
unchanged. The current eight stills/52 balls do not establish temporal identity
or physical precision. Excessive observed audit off/on differences also limit
fine performance attribution.

## Decision

Keep Python as the external evaluation and acceptance harness for both
implementations. Add selectable engine adapters to existing tools; preserve
scoring/report formats and real-socket receiver semantics. Keep Python unit
tests for retained code and reference inference. Write Rust tests for new
ownership/time/geometry/tracking/runtime logic instead of translating all
pytest tests mechanically.

Promote observable replay, ordering, freshness, restart and slow-client assertions
into language-neutral fixtures and black-box tests of selected executables.
Synthetic detectors validate contracts and instrumentation; real model assets
are required for inference accuracy and speed claims.

Extend the existing labeled corpus with independent physical placements and
source-timed moving/occlusion/collision sequences. Score missed balls and
unknown identity alongside matched-center error and latency. Set declared
acceptance tolerances before evaluating candidates; Python parity is not
ground truth when a known defect is intentionally corrected.

Distinguish synchronous detector/scheduler replay capacity from independently
paced production capture with latest-frame dropping. Keep default-off bounded
instrumentation, disclosed losses, actual backend/model/config provenance,
and interleaved off/on comparisons to separate overhead from sustained drift.

## Consequences

### Positive

- Retains testing/tooling investment and one scoring interpretation.
- Compares languages/backends through observable behavior.
- Separates model/export changes from runtime integration.
- Keeps known defects from becoming required parity.

### Negative

- Internal Python fixtures need extraction/adapters rather than zero changes.
- Temporal/physical labels require new data collection and annotation.
- Platform/hardware tests cannot all run in ordinary hardware-free CI.
- Fixing existing quality-gate failures remains work, not automatically solved
  by passing new Rust tests.

## Alternatives considered

### Rewrite all evaluation tooling in Rust

This discards a useful independent oracle and risks changing both the runtime
and its scoring at once.

### Keep only unit tests or compare headline FPS

Unit tests cannot establish physical/temporal accuracy; repeated messages can
inflate FPS while observations remain old.

### Require exact Python behavior everywhere

This preserves stale age, failure ambiguity and identity mistakes. Document
intentional corrected behavior through requirement fixtures.

## Verification

- Existing headless collector and reports process both providers.
- Equivalent fixtures exercise actual adapters, not test-side copies of logic.
- Real-model replay, physical/temporal labels and separated workload runs produce
  versioned results with actual backend/model hashes and disclosed losses.
- Instrumentation satisfies its configured overhead gate or remains unqualified.
- Retained Python/ frontend gates and Rust format/lint/tests pass; the existing
  Python coverage deficit is not hidden by lowering the configured threshold.

## Revisit conditions and related records

Split scoring implementations only if independently cross-validated and
required by scale or maintenance evidence. Preserve ADR-0008 opt-in audit and
accuracy-before-optimization requirements. Release/cutover criteria are ADR-0018.


## Evidence and implementation tracking

Status is proposed; Rust migration implementation and qualification are planned,
not completed. This record governs migration stages 0, 2 and 8.

- [Migration plan](../../../thoughts/rust-perception-migration-plan.md).
- [Architectural review and evidence limitations](../../../thoughts/architectural-review-2026-10-19.md).
- [Existing backend/tools/benchmark_vision_profiles.py](../../../backend/tools/benchmark_vision_profiles.py).
- [Existing backend/tools/collect_performance_audit.py](../../../backend/tools/collect_performance_audit.py).
- [Versioned retained HTTP game-state contract](../../../backend/tests/fixtures/contracts/v1/game_state_http.json), exercised by [test_http_game_state_contract.py](../../../backend/tests/integration/test_http_game_state_contract.py).
- [Versioned WebSocket state contract](../../../backend/tests/fixtures/contracts/v1/websocket_state.json), exercised through the real replay/socket path by [test_pipeline_replay.py](../../../backend/tests/integration/test_pipeline_replay.py).
- [Versioned WebSocket control contracts](../../../backend/tests/fixtures/contracts/v1/websocket_controls.json), exercised by [test_websocket_control_contract.py](../../../backend/tests/integration/test_websocket_control_contract.py).
- [Versioned WebSocket trajectory contract](../../../backend/tests/fixtures/contracts/v1/websocket_trajectory.json), exercised through the real socket by [test_pipeline_replay.py](../../../backend/tests/integration/test_pipeline_replay.py).
- [Explicit model-output comparison tool](../../../tools/test_backend_equivalence.py), its [usage guide](../../../tools/README_EQUIVALENCE_TEST.md), and regression tests at [test_model_equivalence_tool.py](../../../backend/tests/unit/test_model_equivalence_tool.py). This compares model outputs only; it is not an end-to-end performance or accuracy scorer.
- The comparison tool accepts the prototype's saved Rust static-image result, verifies its checkpoint/class/settings/source mapping, and records actual provider plus candidate provenance. The [ten-image CPU run](../../../thoughts/performance-audits/rust-inference-prototype-20261020/tool-run/equivalence-20261020T123923.419656Z.json) matched 73 detections with no unmatched outputs and maximum center difference of 0.0325 px. This validates one static replay adapter; it does not yet exercise the live Rust executable, headless subscriber, paced freshness, temporal or physical accuracy, or Linux runtime.
- The saved-result adapter has 27 focused tests; see the linked test module. Its acceptance scope is intentionally narrower than this ADR's full completion criteria.

Before accepting this decision for implementation, resolve its relevant prototype
and compatibility questions and record the evidence. Acceptance records commitment;
it does not claim all verification has passed. Link implementing changes and actual
verification artifacts as they exist. Follow the [decision register](../README.md)
for status changes and supersession.
