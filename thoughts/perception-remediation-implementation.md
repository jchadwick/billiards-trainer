# Perception remediation implementation

## Goal

Make live aiming state trustworthy and frame-aligned, reduce coordinator work,
and allow capture, perception, gateway, and browser display to run on separate
LAN/VPN hosts. Maintain old observation ages when retaining positions and fail
closed when table or frame identity is uncertain.

## Ordered work

| Stage | Behavior | State |
| --- | --- | --- |
| 1 | Filter layout consensus to configured balls; consume ready detector results before due dispatch while preserving motion invalidation | Implemented and unit/integration tested |
| 2 | Persist an operator-confirmed table-appearance reference and gate trust on live visibility | Implemented through calibration preview/store/runtime; hardware qualification remains |
| 3 | Review sidecar, per-decision trace, and coaching-specific evaluation | Partial: hash-bound reviewed temporal annotations, localization evaluation, and cue-trace scoring exist; a live per-decision sidecar and coaching evaluation protocol remain |
| 4 | Bounded asynchronous cue geometry and reusable inference buffers | Partial: cue estimation now runs on a bounded single-flight worker using shared frame leases; reusable model/inference scratch buffers remain |
| 5 | Acquisition identity, production stream fanout, and authenticated remote service roles | Partially implemented: scene/source identities and fixture headers exist; production fanout and remote auth/TLS do not |
| 6 | Exact-frame browser video/overlay pairing with bounded queues and safe expiry | Implemented and browser-tested; requires production camera parts to emit the contract |
| 7 | Reviewed replay, target-device qualification, soak, and operational documentation | Partially implemented: contract and deployment limits documented; reviewed-corpus, target-hardware qualification, and soak remain |

## Verification record

- `cargo test --locked --manifest-path perception/Cargo.toml --lib`: passed (355 tests).
- `cargo test --locked --manifest-path perception/Cargo.toml --all-targets --quiet`: passed across all library, binary, and integration targets; reran the affected library diagnostic test and live runtime integration target after the final Rust cleanup.
- `cargo test --locked --manifest-path perception/Cargo.toml --test detector_worker`: passed (4 tests).
- `cargo test --locked --manifest-path perception/Cargo.toml --test live_capture_runtime -- --test-threads=1`: passed (10 tests), including single-flight cue dispatch/completion and motion-ordering behavior.
- `cargo test --locked --manifest-path perception/Cargo.toml --bin billiards-mjpeg-fanout`: passed (7 tests).
- `cargo test --locked --manifest-path perception/Cargo.toml --test calibration_store`: passed (12 tests).
- `cargo test --locked --manifest-path perception/Cargo.toml --test table_visibility`: passed (8 tests).
- `cargo test --locked --manifest-path perception/Cargo.toml --test live_application`: passed (1 test).
- `npm run typecheck` in `frontend/web`: passed.
- `npm test` in `frontend/web`: passed (35 tests across 9 files).
- `npm run build` in `frontend/web`: passed.
- `cargo fmt --manifest-path perception/Cargo.toml -- --check`: passed.
- `git diff --check`: clean for changed files after normalizing `LiveView.tsx` line endings.
- `cargo clippy --locked --manifest-path perception/Cargo.toml --lib -- -D warnings`: still reports two existing findings in `inference/runtime.rs` (`manual_slice_size_calculation` and `chunks_exact_to_as_chunks`). Clippy passes when those two baseline lint IDs are allowed; no new warnings remain in the changed paths.

Remote services, production per-part frame identity, reusable inference scratch
buffers, and hardware qualification are deliberately still marked incomplete.
The current HTTP/WebSocket API is unauthenticated and the scene listener is
loopback-only; the deployment docs warn against LAN exposure.

## Architecture records

- [ADR-0099](../docs/architecture/decisions/0099-require-confirmed-table-appearance-for-guidance.md): confirmed table reference.
- [ADR-0100](../docs/architecture/decisions/0100-pair-guidance-with-the-exact-displayed-frame.md): exact-frame overlay pairing.
- [ADR-0101](../docs/architecture/decisions/0101-support-authenticated-remote-perception-services.md): authenticated remote roles.
