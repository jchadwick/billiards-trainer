## Calibration Workflow Bug Investigation

- **Primary Issue**: Runtime error `Failed to apply calibration: imgRef is not defined`.
- **Goal**: Ensure calibration applies successfully without runtime errors and add coverage (manual/automated) that exercises the fixed path.

### Subagent QA-Plan (Verification Strategy)
- Target command: `npm test --prefix frontend/web -- --run` to execute the Vitest suite covering the React calibration wizard.
- Manual check: Trigger calibration application in the UI smoke script (if available) after automated tests.
- Success criteria: No traceback in logs; calibration completes and caches persist as expected.

### Task Flow
1. **Subagent Investigate**: Trace the call stack to locate the undefined `imgRef`, determine ownership, and gather context.
2. **Subagent Fixer**: Implement a durable solution ensuring `imgRef` is initialized or refactor logic to avoid stale references. Add inline comments where logic is subtle.
3. **Subagent Tester**: Execute automated tests, review results, and document outcomes here.

### Notes
- Avoid hardcoded constants in production code; leverage configuration defaults.
- Update or add documentation/comments reflecting the new logic.

---

#### Subagent Investigate Findings (2024-04-05)
- `imgRef` exists only inside `VideoFeedCanvas` in `frontend/web/src/components/config/calibration/CalibrationWizard.tsx`.
- `applyCalibration` within `PlayingAreaCalibrationStep` references `imgRef.current`, but no ref is declared in that scope, yielding the runtime `ReferenceError`.
- Proposed direction: expose natural video resolution from `VideoFeedCanvas` via a callback/state bridge instead of reaching into its internal ref.

#### Subagent Fixer Plan
- Report the interactive calibration canvas dimensions (the UI width/height) directly to the backend so it can scale to the 4K space.
- Remove the invalid direct `imgRef` access and keep the video feed component self-contained for rendering purposes only.
- Centralize the canvas size constants to avoid duplicated literals and ensure future adjustments stay consistent.

#### Subagent Tester Log
- `npm test --prefix frontend/web -- --run` → fails because Playwright suite (`test-websocket-diag.spec.ts`) is incompatible with Vitest runner and a pre-existing integration test imports missing files. No changes from this task triggered these failures.
- `npm run typecheck` (executed from `frontend/web`) → fails due to long-standing type errors in unrelated accessibility and store modules; untouched in this change set.
- No module-specific automated tests exist for `CalibrationWizard`; manual verification required downstream.
