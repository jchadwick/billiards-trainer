# ADR-0107: Publish supervised producer diagnostics on a separate versioned channel

- Status: accepted
- Date: 2026-11-02
- Deciders: Project maintainers
- Supersedes: None
- Superseded by: None

## Context

The existing scene WebSocket requires localized state and closes during producer restart. The live-v3 scene handshake occurs only after capture preflight and both it and SceneSnapshotV1 are strict versioned contracts. They cannot report model-loading/capture-start failures without weakening existing contracts. Health and runtime identity must remain distinct from scene measurements.

## Decision

Use a bounded versioned loopback producer diagnostics channel and a separate `/api/v1/diagnostics/ws` browser channel. Start producer reporting before model initialization and capture preflight. The gateway serves an initial unknown/starting assessment without scenes, validates reporter process generation and report sequence, and pushes newest complete assessments on changes and configured heartbeats/expiry deadlines. Keep the existing state WebSocket, scene contract, and HTTP liveness/readiness behavior unchanged. Treat absent reporter evidence as unknown. Do not infer healthy idle/stalled status without scheduler expectations and progress evidence.

## Consequences

### Positive

- Startup, failure, restart, model, capture and scheduled-work evidence is available independently of scene delivery.
- Existing scene clients and measurement-age semantics retain their current contract.

### Negative

- Two bounded transport contracts and a producer supervisor integration must be maintained.
- Legacy/external producers without the reporter expose unknown stage details.

## Alternatives considered

### Add optional diagnostics fields to the scene WebSocket

It cannot deliver before capture preflight, is absent during no-scene startup, and risks coupling health heartbeat semantics to measurement cadence.

## Verification

- `cargo test --all-targets` passed after the final capture-closure and scheduler-evidence changes: 407 library tests and all binary/integration tests passed.
- `cargo clippy --all-targets -- -D warnings` passed.
- `cargo check --all-targets --features camera-gstreamer-source` passed; this verifies compilation, not physical camera/GPU operation.
- `cargo fmt --all -- --check` and `git diff --check` passed.
- AJV draft 2020-12 validation passed for browser `no-scene-startup.json` and `last-known-expired.json`, and producer `active.json`, `loading.json` and `failed.json` fixtures (`--validate-formats=false`).
- `python3 /Users/jchadwick/.agents/skills/manage-architecture-decisions/scripts/validate_adrs.py docs/architecture/decisions` passed for all nine ADRs.
- Focused regression tests cover immediate no-scene WebSocket snapshots/revision order, report expiry without new packets, preserved last-known evidence, active-work deadlines versus phase-gated idle, producer exit/spawn outcomes, atomic source/capture stop on closure, bounded diagnostics clients, and unchanged scene/health behavior.

The implementation has not been exercised with a physical camera or GPU model runtime. Browser route smoke ran without a live gateway; pushed-stream behavior is covered by contract, transport, store, and UI tests.
