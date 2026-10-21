# Backend reliability: replay and quality gates

Implemented 2026-10-18. Scope: tests, quality-tool configuration, and CI.

## Verification approach

`backend/tests/integration/test_pipeline_replay.py` reads the checked-in
`backend/vision/test_data/cue_aiming.png` recording and replaces only inference
with deterministic ball/cue detections. Production vision masking, coordinate
normalization, result construction, core state updates, integration scheduling,
physics calculation, broadcaster, subscription manager, admission endpoint,
bounded writer, and real TCP/WebSocket transport remain in the path.

The fixture's two-ball shot is a controlled inference scenario over recorded
pixels; it does not measure neural-network detection accuracy on those pixels.
An explicit `model_assets` test runs the recording through the production vision
module with `BILLIARDS_REPLAY_MODEL_PATH` set to an installed model file. It skips
when that environment variable is absent and never downloads assets for CI.

The local server binds an ephemeral loopback port. A server-ready event gates
connection attempts. An ASGI send gate holds one real client's state send while
another receives every update. Assertions verify bounded pending storage,
coalescing, disconnect/reconnect sessions, and writer shutdown. No throughput or
FPS assertion depends on sleep duration.

## Representative measurements

One hardware-free run on this development machine (six replay publications):

| Measurement | Result |
| --- | ---: |
| Frame-to-delivery mean | 2.69 ms |
| Frame-to-delivery maximum | 8.60 ms |
| Maximum publication age at receipt | 8.60 ms |
| Healthy-client maximum queue depth | 1 |
| Healthy-client dropped updates | 0 |
| Geometrically checked prediction outputs | 6/6 |
| Healthy updates delivered while peer blocked | 12 |
| Blocked-client pending queue depth / capacity | 1 / 2 |
| Superseded blocked-client updates dropped | 10 |

The replay checks canonical coordinates, a horizontal cue launch, collision
presence, the incoming center's contact distance, and the outgoing target-ball
branch origin/direction within 0.01 canonical pixels. The trajectory calculator
now starts the outgoing branch at the target ball's own center (`x=2400`); the
incoming collision center remains one combined radius from it. Deterministic
head-on and oblique unit cases also verify that caller-owned ball geometry stays
unchanged.

These six frames use deterministic inference and do not establish real-camera
or installed-model capacity. **30–60 FPS remains a measurement target, not a
guarantee.** The harness records measurements in the test's JUnit properties and
prints them for runs with `-s`.

## Commands and results

At the start of this stage, `pytest backend/tests -m 'not slow and not hardware'
--no-cov -q` passed 933 tests. Existing production/test work from other stages
continued in the shared workspace during verification. A separate pre-replay
coverage run passed 934 tests but failed the unchanged 80% coverage gate at
36.78%.

Final stage verification, including the sequential cue and collision-origin
corrections:

- `.venv/bin/python -m pytest backend/tests/integration/test_pipeline_replay.py --no-cov -q -s`: 2 passed, 1 optional installed-model test skipped; latest replay mean 2.69 ms and max 8.60 ms.
- Focused physics plus replay: 33 passed, 1 optional installed-model test skipped.
- `.venv/bin/python -m pytest backend/tests -m 'not slow and not hardware and not model_assets' --no-cov -q`: 938 passed, 1 deselected.
- The same complete selection with coverage: 938 passed, 1 deselected; **coverage gate fails below its required 80%** (latest full run before the physics-only test additions reported 39.37%).
- `.venv/bin/python -m black --check backend`: 173 files pass.
- `.venv/bin/python -m isort --check-only backend`: passes after sorting imports in the two new rate-limit test files.
- `uvx --from ruff==0.17.1 ruff check backend/tests/integration/test_pipeline_replay.py`: passes.
- `uvx --from ruff==0.17.1 ruff check backend`: existing debt, initially 446 findings; later concurrent changes produced 447. No new ignore/exclude entries were introduced.
- `uvx --from mypy==1.18.2 mypy --strict --python-version 3.11 --python-executable .venv/bin/python backend`: 1,346 findings in 97 of 131 source files.
- `.venv/bin/python -m compileall -q backend`: passes; project TOML and workflow YAML parse successfully.

CI pins Ruff 0.17.1 and mypy 1.18.2, runs required full-backend Ruff and strict
typing checks, and preserves formatting, integration, system, and coverage
checks. The strict type command targets the tested Python 3.11 runtime because
current installed Pydantic sources use syntax newer than the project's legacy
Python 3.9 type target. Ruff's existing selection/ignore settings were moved to
the supported `tool.ruff.lint` tables without weakening their scope.

Remaining work: sequential production-module lint/type cleanup, broader test
coverage to meet the existing gate, and installed-model/camera replay
measurements.
