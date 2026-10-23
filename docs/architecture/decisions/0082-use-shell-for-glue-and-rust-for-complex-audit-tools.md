# ADR-0082: Use shell for glue and Rust for complex audit tools

- Status: superseded
- Date: 2026-10-22
- Deciders: Project maintainers
- Supersedes: ADR-0017 for long-term implementation ownership of complex audit tools
- Superseded by: ADR-0085

## Supersession note (2026-10-23)

ADR-0085 supersedes this record's temporary Python collector-retention and
parity requirements. The shell-for-glue and Rust-for-complex-tooling decision
remains in force. The retired Python collector/profile benchmark and their
collector-only Rust adapters are historical; the current state/trajectory audit
and temporal localization evaluation are Rust-owned. Viewer interference and
recorded-video replay remain explicit gaps rather than retained Python tools.

## Context

The supported application server and live perception path are moving to Rust.
The user has directed that simple development orchestration use shell scripts
where practical, while complicated or significant tooling should also migrate
to Rust. The existing performance-audit collector is a substantial Python
program: it consumes live WebSocket state, tracks observation identity and
freshness, bounds long-run samples, collects process resources for its generic
audit modes, starts a Rust live process, and writes versioned reports. Shell is
not a suitable replacement for those stateful protocols and measurements.

The Python collector also contains useful independent measurement semantics.
Changing its implementation and the Rust runtime together could silently change
fresh-versus-cached accounting, loss disclosure, clock comparability, or report
interpretation. The Python backend service remains a temporary migration bridge;
Python training and annotation workflows remain off-device. Ball-number
classification remains excluded.

## Decision

Use portable shell scripts for thin development glue such as build, launch,
readiness, and one-command test workflows. Implement complex or stateful
development and acceptance tools in Rust, including WebSocket collection,
bounded observation accounting, process/resource sampling, and versioned audit
reports. Add Rust functionality incrementally around the live path and its
existing contracts; do not create a competing harness or translate Python
internal fixtures mechanically.

For a managed live audit, derive the detector child PID from positive
`process_id` values in the live timing side channel and run the existing bounded
resource sampler for each measured state window. Keep this sampling separately
opt-in. Do not substitute the gateway PID for the inference process. If more
than one detector PID is observed in a window, or timing/resource evidence is
incomplete, report the PIDs and mark resource evidence partial or unavailable;
do not combine samples across process restarts or imply coverage of an
unsampled replacement process.

Retain the existing Python audit collector and its tests as an off-device,
independent differential reference during the Rust port. Retire the Python
collector only after Rust contract fixtures and real-socket tests cover the
same identity, ordering, unique-observation, freshness, bounded-retention,
loss-reporting, clock, and provenance semantics. Python model training and
annotation tools remain outside this tooling migration. No authentication or
credential mechanism is introduced.

**Scope update (2026-10-23, ADR-0083):** the reviewed CVAT temporal localization
importer is a separate, accuracy-critical tooling responsibility and is now
owned by Rust. Its Python reference and Python-only tests were removed after
the Rust output matched on a generated video using real `ffprobe`. This does
not move model training, dataset preparation, or other annotation workflows.

## Consequences

### Positive

- Simple developer workflows stay readable and portable without requiring
  another runtime.
- Stateful protocols and measurement logic use the same language and test
  toolchain as the Rust server.
- The existing Python implementation remains an independent oracle while
  Rust behavior is established, reducing the risk of silently changing audit
  conclusions.

### Negative

- Rust implementations must preserve explicit evidence semantics and may take
  more initial work than a thin shell wrapper.
- Python reference code and its dependencies remain in the repository until
  parity and retirement gates pass.
- The Rust audit tool and current Python reports may differ in presentation
  while the contracts are being migrated; each report must declare its schema
  and unavailable measurements.

## Alternatives considered

### Keep all performance audit tooling in Python indefinitely

Rejected as the long-term direction because the user has directed that
significant tooling migrate and the collector is a large, stateful part of the
development workflow. Keeping Python as a temporary independent reference
preserves its validation value without making it a permanent runtime
dependency.

### Replace the collector with shell scripts

Rejected because shell is not appropriate for bounded WebSocket parsing,
identity-window accounting, percentiles, clock semantics, and structured
reports. Shell remains appropriate for invoking and composing tools.

### Port the entire collector in one rewrite

Rejected because one wholesale rewrite would change the implementation and
measurement semantics at once, making regressions harder to distinguish from
intentional corrections. Port one contract-tested responsibility at a time.

## Implementation status

The first Rust slice is committed as `77147f5`. The
`billiards-performance-audit` command receives anonymous plain-WebSocket state
messages, tracks bounded source-identity and stream-order windows, freshness,
producer-reported read-completion-to-publication age, and receiver-local state
intervals. It can emit a versioned JSON report and bounded JSONL state details.
The command rejects URL userinfo, query strings, and fragments, and it does not
support authentication headers or credentials.

The receiver now negotiates the retained API's welcome and explicit state
subscription protocol, sends its JSON application heartbeat independently of
WebSocket control frames, and still accepts the Rust gateway's immediate
push-style state stream. Loopback real-socket tests exercise both contracts;
qualification against a running application has now completed against the
Rust gateway. The heartbeat defaults to 20 seconds and is configurable on the
Rust CLI.
The version-4 report distinguishes unique observed source frames from repeated
state publication, counts complete and uncorrelated messages, normalizes
component observation statuses, and detects bounded source gaps, restarts,
duplicates, and out-of-order delivery. Source ordering and identity windows
disclose their capacities and evictions.

The `billiards-performance-audit-run` command now provides the state-only
repeatable workflow: configurable warm-up, repeated measured windows or one
bounded soak, bounded JSON and JSONL artifacts, revision and dirty-worktree
provenance, environment metadata, and a Markdown report. Its thin
`scripts/performance-audit-run.sh` wrapper passes arguments to Cargo and keeps
measurement and report logic in Rust. Loopback integration tests exercise
multiple actual WebSocket sessions, report/detail output, and explicit failure
artifacts.

The same Rust runner can now manage a local application-server child through
the `--server-executable` and repeated `--server-argument` options. It accepts
only loopback state URLs in this mode, probes until it receives an individual
state containing both a fresh observation identity and observed ball
positions, and does not begin warm-up or measurements before that gate. Its
readiness receiver stops on that exact state instead of waiting out the whole
probe interval. A child exit during the audit fails the run. On Unix, the
runner launches the gateway in a private process group; on success, timeout,
or receive failure it sends `SIGTERM` to the group, waits for the gateway and
its detector/capture descendants, and sends `SIGKILL` to the group only after
the configured grace period expires. It then reaps the gateway. Schema
version 3 records executable path, PID, an argument digest, readiness timing
and probe count, the first observed frame identity, and shutdown outcome
without copying raw arguments into the report. For managed `--start-live`
runs, it also hashes the effective configuration, resolved capture settings,
ONNX model and manifest, source checkpoint, detector executable, and capture
helper. It records the camera-profile identity, configured provider, input
size, thresholds, and ball/cue class mapping without exposing the raw camera
URL or configuration contents. These are configured identities, not proof of
actual accelerator placement or dynamic runtime version. The new thin
`scripts/performance-audit-live.sh` builds the
runner, gateway, live detector, and GStreamer capture helper with the explicit
source feature, then forwards gateway arguments without shell evaluation.
Real loopback tests cover a failed observation before
readiness, successful repeated measurements, startup timeout, failure
artifacts, and child cleanup. A live run against the configured remote MJPEG
input verified the complete managed-server path on the development Mac. In a
five-second smoke window, the receiver recorded 24 complete state messages,
19 fresh unique observed frames, and five repeated states. There were no
WebSocket message-sequence gaps; the source frame counter had three gaps
totaling 21 skipped frame IDs. The state-receipt interval p50/p95/p99 was
103/499/658 ms; the producer-reported read-completion-to-publication age was
95.9/98.0/98.4 ms. Startup took 3.84 s and the gateway exited with code zero
without forced termination. This short run confirms integration only; it is
not a performance baseline or an accuracy qualification. The first attempt
exposed a stale supervisor CLI contract for frame preview, which is now fixed
and covered by configuration/argument tests.

The Rust run controller now supports an optional state-plus-trajectory
subscription on the same connection. Trajectory allocation is absent by
default; when enabled, its summary and receipt tail are bounded, its metadata
and lifecycle fields are tested on a real loopback WebSocket, and the run
writer emits versioned per-window JSON and JSONL artifacts. A configurable
subscriber delay exercises best-effort delivery and exposes resulting gaps.
The collector leaves cross-process trajectory latency unavailable because it
does not yet verify a shared monotonic clock domain. Its Python summary engine
remains an independent differential reference.

The Python collector is still not replaced. Its remaining migration work
includes viewer workload, server event-window and clock joins, baseline
selection, full camera-profile provenance, and end-to-end report orchestration
not yet covered by the Rust runner. The Rust runner owns managed full-server
lifecycle for state and optional trajectory audits and has both loopback
protocol coverage and a remote-input smoke check. Accuracy remains unavailable
without temporal labels.

This slice adds optional process CPU and resident-memory sampling to the
Rust receiver. It uses `sysinfo` on a dedicated thread, binds samples to the
observed process start-time value, respects the CPU sampler's minimum refresh
interval, keeps a bounded tail, reports total and retained counts, and marks
inaccessible or changed processes unavailable. The
sampler is absent unless `--resource-process-id` is supplied. Its optional
report section was schema version 3 before source/completeness fields were
added in schema version 4. A thin
`scripts/performance-audit-state.sh` wrapper locates the Rust manifest and
passes arguments through unchanged.

The Rust live-camera qualification now uses a second small Rust executable,
`billiards-performance-audit-resources`, to sample its supervised inference
child. The Python orchestration starts this helper after warm-up and stops it
before child shutdown. It announces readiness only after process identity and
its monotonic baseline are initialized. Its control stream accepts bounded JSON
Lines, its summary line is capped at 16 MiB, and it retains at most 50,000
sample details; the report records requested/effective retention and labels
statistics as all-sample or retained-tail. Sample elapsed times remain in the
helper's monotonic clock domain. `sysinfo` reports process start time at
one-second resolution, so same-second PID reuse remains ambiguous and is
reported explicitly. This removes the Python `psutil` sampling thread from
Rust live qualification only. The larger Python collector, its generic audit
modes, and report orchestration remain and must be migrated only after their
evidence contracts have Rust parity.

The native Rust managed audit runner now also offers
`--sample-managed-live-process-resources`, independently opt-in alongside
`--live-pipeline-timing-audit`. It learns the detector child PID from the
loopback timing records and starts the bounded Rust sampler for each measured
window. The window report lists sampled and observed detector PIDs. A restart,
resource-sampling failure, timing loss, or missing identity evidence marks the
resource result partial or unavailable; post-window footer data is excluded
from measured-window PID accounting. Runner report schema version 6 adds this
optional configuration and per-window status alongside optional trajectory
receipts. CLI validation, latest-PID
replacement/timeout behavior, complete-resource summaries, restart handling,
and footer exclusion have dedicated tests. The all-target Rust tests, strict
Clippy, formatting check, and ADR validator pass (2026-10-22). The existing
Python qualification path remains an independent reference and is not retired
by this managed-runner feature.

Commits `6feb7e7` and `8e74dff` move the server-event summary reducer into the
Rust library and `billiards-performance-audit-server-summary` command. It
reduces retained JSON Lines events for vision, integration, state delivery,
video, trajectory, resources and viewers. The Rust unit and command tests use a
shared fixture whose expected summary was produced by the retained Python
implementation. The Python collector now delegates this reducer after it has
selected the same-clock measurement window; collection, clock selection,
export-status merging and Markdown rendering remain owned by Python. The
command rejects boolean metric values and bounds each line, event count and
total input size. Its defaults are 1 MiB per line, 200,000 events and 128 MiB
total input. The Python bridge enforces the same event and total-input bounds
before launching the command and normalizes non-finite floats to JSON null.
Rejecting booleans corrects the Python `bool`-is-`int` edge case in the former
reducer so flags cannot appear as timing or resource measurements.

Commits `f445a07` and `b0be61a` move delivery-latency correlation and
statistics into the Rust library and the
`billiards-performance-audit-delivery-summary` command. The reducer joins state
and trajectory publication, socket completion, and client receipt by domain,
sequence, and client identity; it subtracts producer/receiver monotonic
timestamps only when both sides report the same single verified OS clock
domain. The Python collector still selects the event window and supplies the
read-only clock identity for baselines where instrumentation is off, then
delegates bounded JSON Lines input to Rust. A shared fixture checks the full
report, including overlapping state/trajectory sequence numbers, deadline
misses, component ages and clock labeling. Invalid boolean timestamps, metrics
and client identities are ignored. The command shares the 1 MiB per-line,
200,000-event and 128 MiB total-input defaults.

Commit `4eacd09` moves the instrumentation-overhead decision into the Rust
library and `billiards-performance-audit-overhead-summary` command. The Python
collector continues to load the selected baseline report and assemble bounded
comparison input. Rust checks repetition and delivery-sample minimums, machine
and workload identity, source/model/worktree hashes, relevant and audit
configuration, collector settings, throughput change, and p95 delivery change.
The minimum evidence requirements are validated settings with defaults of
three repetitions and 100 delivery samples per run. The reducer reports an
inconclusive interpretation whenever evidence or comparability is insufficient,
even if one metric happens to be within budget. A thin
`scripts/performance-audit-overhead-summary.sh` wrapper exposes the native
command for direct development use.

Commit `2001493` adds the bounded Rust trajectory receipt accumulator and
`billiards-performance-audit-trajectory-summary` JSON Lines command. The
collector now starts this child only for trajectory subscriptions and forwards
each receipt through a nonblocking queue on a separate writer task. Queue and
record-limit rejection counts are reported separately from the asynchronous
JSONL trace export's dropped/error counts; intentional sampling does not count
as loss. Rust output is checked against the retained Python accumulator's
stable fields, and Python remains the reported fallback on startup, process, or
parity failure. This is a migration bridge, not final collector parity.

The trajectory command defaults to 10,000 retained samples, 200,000 input
records, 1 MiB per line, 128 MiB total input, and 1 MiB output. The Python
collector passes its configured history limit and event limit plus the two
protocol control records; the bridge queue reuses the configured asynchronous
export queue capacity. Native hard caps are one million samples, five million
records, 16 MiB per line, 1 GiB total input and 16 MiB output. The thin
`scripts/performance-audit-trajectory-summary.sh` entry point exposes the
standalone JSON Lines command without duplicating its parsing or summary logic.

The new `billiards-temporal-evaluate` Rust library and CLI port temporal
localization metrics and the Rust scene-snapshot adapter. It binds annotations
to video SHA-256 and geometry, uses annotated presentation times, hashes the
exact scene bytes consumed, rejects cached or stale observations as detections,
and bounds annotation/JSONL reads plus the `ffprobe` timeout and output. The
thin `scripts/evaluate-temporal-localization.sh` entry point resolves the
repository root and forwards CLI arguments. A shared report fixture is checked
by both Rust and the retained Python evaluator, and a file-level test covers
video binding and scene normalization. This establishes a focused port, not
full temporal-corpus parity: keep the Python scoring code until the broader
temporal cases and a recorded detector replay have been compared, then retire
the Python evaluator and its compatibility tests as part of that follow-up.
The redundant standalone Python command adapter has been retired: developers
use the shell wrapper, which runs the Rust CLI, while the Rust file-level tests
cover video binding and scene normalization. This removes the duplicate command
implementation without discarding the broader Python differential reference.

## Verification

- Rust unit and integration tests cover malformed and bounded configuration,
  rejected credential-bearing URLs, fresh/stale/repeated source identities,
  camera-session restarts, stream order and gaps, bounded retention, JSON/JSONL
  command output, optional resource sampling of a real process, bounded
  resource retention, and unavailable cross-process clock comparisons.
- The integration test opens a real loopback WebSocket connection and exercises
  both the library API and the executable. It does not claim full gateway or
  live-model qualification.
- The trajectory integration test validates the explicit state-plus-trajectory
  subscription, interleaved receipts, sequence gaps, invalidation counts,
  bounded detail retention, and versioned run artifacts. State-only collection
  remains the default and does not allocate trajectory retention.
- The native live-qualification helper is exercised as a real child process,
  including its ready/start/stop protocol, bounded sample details, process
  identity, memory sample, and failure fallback in the Python coordinator.
- The server-event reducer matches a shared golden fixture across stage work,
  trajectory outcomes, unique source observations, resource trends, and viewer
  records. Its JSON Lines command also rejects malformed and over-limit input.
- The server-event, delivery-latency, and overhead reducers match shared golden
  fixtures. Their tests cover mismatched clocks, instrumentation-off clock
  identity, malformed booleans, comparison pass/over-budget/inconclusive
  results, and bounded command input. The focused collector/configuration
  suite passes 148 tests, and Ruff, Black, and mypy pass.
- The trajectory receipt reducer matches a shared golden fixture. Python bridge
  tests exercise a real child-process protocol round trip, record and queue
  bounds, nonblocking receipt enqueue, exporter/bridge loss separation, and
  Python fallback when the native tool is unavailable. Collector integration
  verifies the state-only path does not start this trajectory child.
- The Rust temporal evaluator matches the shared Python reference report.
  Rust tests cover strict annotation parsing, global assignment including
  unmatched truths, fresh-versus-cached results, unavailable samples,
  session/calibration transitions, label-time freshness, bounded inputs,
  bounded `ffprobe` output and timeout, and exact file identity. The focused
  Python reference/evaluator suite passes 42 tests; it requires the targeted
  no-coverage invocation because the repository-wide coverage gate is not met
  by running one test file in isolation.
- The overhead shell entry point resolves and runs from outside the repository
  root.
- Rust formatting, strict Clippy, and the all-target test suite pass.
- The recorded-video live smoke in the migration plan validates integration
  only; it does not establish detector accuracy or a sustained performance
  baseline.
- The shared reducer fixtures pass; differential coverage for remaining
  collector behavior, full collector parity, measured instrumentation overhead
  and final retirement remain required before removing the Python collector.

## Related records

- [ADR-0017](0017-retain-python-tools-as-cross-runtime-acceptance-tests.md)
  remains the historical rationale for retaining an independent Python
  reference during migration.
- [ADR-0047](0047-measure-rust-scene-transport-at-the-python-receiver.md),
  [ADR-0049](0049-explicit-performance-audit-count-semantics.md),
  [ADR-0050](0050-normalize-audit-evidence-across-python-and-rust-runtimes.md),
  and [ADR-0051](0051-repeat-recorded-input-for-bounded-paced-audits.md)
  define receiver, counting, clock, and replay evidence contracts.
- [ADR-0066](0066-run-server-services-in-rust-and-retain-the-standard-web-client.md)
  defines the server/browser language boundary.
- [Rust domain migration plan](../../../thoughts/rust-domain-migration-plan.md)
  records the staged implementation and qualification status.
