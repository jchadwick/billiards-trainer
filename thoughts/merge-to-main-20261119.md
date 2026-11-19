# Merge-to-main preparation — 2026-11-19

## Scope and safety

This preparation covered the complete dirty project state on `codex/native-ball-localization`. The user authorized committing all modified and untracked project work, including the administration, diagnostics, setup, calibration, and shared video receiver files that had previously remained dirty. All 51 initially modified tracked files and 90 initially untracked project files were preserved and committed. The requested preparation note is also committed.

No reset, clean, stash, push, production deployment, or camera operation was performed. The existing `main` worktree at `/Users/jchadwick/.codex/worktrees/proven-localization-findings/billiards-trainer` was clean at the start and remains unchanged and clean. The other linked ball-identification worktrees were left untouched with their preexisting changes.

## Starting Git state and audit

- Preparation worktree: `/Users/jchadwick/repos/jchadwick/billiards-trainer`
- Branch and starting commit: `codex/native-ball-localization`, `3f5cdd211bbe807441c219e43d4c2904a46567dd`
- Fetched `origin/main`: `f88b418459edebfb220990df72b870121ca63543`; it is an ancestor of the prepared branch.
- Local `main`: `d57ca3afd3cc031ed83d8d639be63dfde0fada7b`, clean and 251 commits ahead of fetched `origin/main`.
- The local-main and feature histories diverged at `ba458767d33b79cc789293f0991f79d1f61f63d8`. The local-main history was merged into the candidate branch and is now an ancestor of it.
- The audit found no private-key, common access-token, or bearer-token patterns in changed or untracked text files. Credential-like JSON key matches were dependency names in `package-lock.json`; no credential-bearing configuration field was found.
- Changed and untracked files were source, tests, docs, schemas, configuration, and research notes/data. No generated binaries or temporary files were included. The frontend build output under ignored `backend/static/` remains untracked and is not part of the commits.

## Module-scoped commits

- `943f9bcd` — deployment/runtime configuration and packaging
- `e5e0bab9` — perception setup, calibration, browser diagnostics, producer diagnostics, and contracts
- `d211705d` — frontend setup, administration, diagnostics, events, health, and shared video reception
- `cc34cc44` — workflow/API/deployment documentation and ADRs 0106–0113
- `4b568320` — high-resolution model-export test
- `596b809e` — calibration and ball-identification research artifacts, results, and preparation note
- `4fb6fdaf` — merge local `main` history into the candidate

The candidate retains its 14 preexisting branch-only commits. The integration commit has candidate `596b809e` as its first parent and local `main` `d57ca3af` as its second parent. A local safety ref, `backup/pre-main-integration-2026-11-19`, points to `596b809e` before the history merge.

## Conflict resolution

The merge preview identified conflicts in `perception/src/live_supervisor.rs` and `perception/src/phase_scheduler.rs`.

- The supervisor keeps browser-selected calibration assets and packaged/source model-path fallbacks, along with the local-main OpenVINO GPU-provider error detail. The combined error also retains the existing unsupported-provider key and value context.
- The scheduler keeps separate regression tests for stale layout evidence under fresh motion evidence and for repeated layout updates during slow inference. Both test scenarios passed in the merged tree.

## Validation

The following checks passed against the fully merged candidate:

- `make format-check`
- `make test` — all Rust targets and 32 Vitest files / 152 browser tests passed
- `make test-rust-camera` — passed with `camera-gstreamer-source`
- `make lint` — Rust Clippy with warnings denied and frontend typecheck
- `make config-check`
- `make test-deploy`
- `make test-tooling`
- `make test-model-export` — 8 Python tests passed
- `npm --prefix frontend/web run build`
- `git diff --check` and `git diff --cached --check`

The first merge-commit hook attempt saw one failure in the existing `temporal_evaluation::tests::ffprobe_output_and_runtime_are_bounded` assertion. The focused test passed immediately when rerun alone, and the complete perception hook passed on the next merge-commit attempt. No source change was required for that transient test failure.

Vite emitted its large-chunk advisory for the configuration editor and health page; the production build succeeded. Check logs are under `/tmp/billiards-merge-to-main-20261119/`.

## Handoff state

- Integration commit: `4fb6fdafed6eb831803324bc62d0dbf2c77bed87`
- At integration, local `main` was an ancestor of the candidate; the candidate was 21 commits ahead of local `main` and 272 commits ahead of fetched `origin/main`.
- The preparation worktree is clean apart from ignored generated build output. The separate `main` worktree remains clean at `d57ca3af`.
- No merge to `main` and no push has been performed; this candidate is ready for independent review.
- After review, the existing `main` worktree can fast-forward with `git -C /Users/jchadwick/.codex/worktrees/proven-localization-findings/billiards-trainer merge --ff-only codex/native-ball-localization`.
