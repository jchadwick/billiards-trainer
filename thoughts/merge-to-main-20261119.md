# Merge-to-main preparation — 2026-11-19

## Scope and safety

This preparation covers the complete dirty project state on `codex/native-ball-localization`. The user explicitly authorized committing all modified and untracked project work, including the administration, diagnostics, setup, calibration, and shared video receiver files that had previously remained dirty. No other worktree files are being changed. The existing detached ball-identification worktrees and their changes remain preserved in place.

No reset, clean, stash, push, production deployment, or camera operation is part of this preparation. The existing `main` worktree at `/Users/jchadwick/.codex/worktrees/proven-localization-findings/billiards-trainer` was clean when inspected and will not be modified during preparation.

## Initial Git state

- Preparation worktree: `/Users/jchadwick/repos/jchadwick/billiards-trainer`
- Branch and starting commit: `codex/native-ball-localization`, `3f5cdd211bbe807441c219e43d4c2904a46567dd`
- Fetched remote main: `origin/main` at `f88b418459edebfb220990df72b870121ca63543`
- The preparation branch contains fetched `origin/main` as an ancestor.
- Existing local main worktree: clean at `d57ca3afd3cc031ed83d8d639be63dfde0fada7b`; its history has commits not present on the preparation branch and will be integrated before review.
- Initial dirty set: 51 modified tracked files and 90 untracked files (141 total).
- Review of changed and untracked text found no private-key, common access-token, or bearer-token patterns. JSON fields named like credentials were checked; matches were dependency names in the package lockfile, with no credential-bearing configuration field found.
- Changed/untracked files are source, tests, docs, schemas, configuration, and research notes/data; the scan found no generated binary files to commit.

## Validation

Validation commands and results will be recorded here after they run.

## Prepared commits and integration

Commits, ancestry reconciliation, final diff statistics, and remaining review notes will be recorded here before handoff. The candidate will stop before merge to `main` or push.

Validation before integration:

- `make format-check` — passed.
- `make test` — passed; all Rust targets passed and Vitest reported 32 files / 152 tests passed.
- `make test-rust-camera` — passed with `camera-gstreamer-source` enabled.
- `make lint` — passed (Rust Clippy with warnings denied and frontend TypeScript typecheck).
- `make config-check` — passed; Rust schema and runtime validation accepted the configuration.
- `make test-deploy` — passed (distribution packaging and update/restart smoke tests).
- `make test-tooling` — passed (security-check dispatch/failure smoke test).
- `make test-model-export` — passed; 8 Python tests passed. Included because this dirty set adds a model-export test.
- `cd frontend/web && npm run build` — passed; the production Vite bundle was generated. Vite reports the existing large chunk warning for the configuration editor and health page; it does not fail the build.
- `git diff --check` — passed.
- Logs are under `/tmp/billiards-merge-to-main-20261119/`.

The frontend build output is ignored under `backend/static/`; it is generated validation output and is not part of the source changes.

The pre-integration diff contains 51 modified tracked files and 91 untracked files after adding this preparation note, with 10,011 insertions and 1,903 deletions in tracked files. The exact commit inventory and post-merge verification will be appended before handoff.
