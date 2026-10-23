# ADR-0082: Use shell for glue and Rust for complex audit tools

- Status: accepted
- Date: 2026-10-22
- Deciders: Project maintainers
- Supersedes: ADR-0017 for long-term implementation ownership of complex audit tools
- Superseded by: None

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

Retain the existing Python audit collector and its tests as an off-device,
independent differential reference during the Rust port. Retire the Python
collector only after Rust contract fixtures and real-socket tests cover the
same identity, ordering, unique-observation, freshness, bounded-retention,
loss-reporting, clock, and provenance semantics. Python model training and
annotation tools remain outside this tooling migration. No authentication or
credential mechanism is introduced.

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

The collector is still not replaced. Trajectory collection, producer
timing/export analysis, configuration and hardware provenance, input replay
orchestration, cross-run comparison, accuracy evaluation, and the final
Markdown report remain Python-owned and are required parity work. The Python
collector must remain until those contracts have Rust implementations and
differential fixtures. The Rust receiver's real-socket tests use a loopback
test peer; the command has not yet been qualified against the full application
process or as a performance baseline.

This slice adds optional process CPU and resident-memory sampling to the
Rust receiver. It uses `sysinfo` on a dedicated thread, binds samples to the
observed process start-time value, respects the CPU sampler's minimum refresh
interval, keeps a bounded tail, reports total and retained counts, and marks
inaccessible or changed processes unavailable. The
sampler is absent unless `--resource-process-id` is supplied. Its optional
report section advances the receiver report to schema version 3. A thin
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

## Verification

- Rust unit and integration tests cover malformed and bounded configuration,
  rejected credential-bearing URLs, fresh/stale/repeated source identities,
  camera-session restarts, stream order and gaps, bounded retention, JSON/JSONL
  command output, optional resource sampling of a real process, bounded
  resource retention, and unavailable cross-process clock comparisons.
- The integration test opens a real loopback WebSocket connection and exercises
  both the library API and the executable. It does not claim full gateway or
  live-model qualification.
- The native live-qualification helper is exercised as a real child process,
  including its ready/start/stop protocol, bounded sample details, process
  identity, memory sample, and failure fallback in the Python coordinator.
- The server-event reducer matches a shared golden fixture across stage work,
  trajectory outcomes, unique source observations, resource trends, and viewer
  records. Its JSON Lines command also rejects malformed and over-limit input.
- Both Rust reducers match shared golden fixtures; the delivery reducer also
  covers mismatched clocks, instrumentation-off clock identity, malformed
  booleans and bounded command input. The backend performance collector's 60
  focused unit tests pass through the Rust bridges, and Ruff, Black, and mypy
  pass.
- Rust formatting, strict Clippy, and the all-target test suite pass.
- The recorded-video live smoke in the migration plan validates integration
  only; it does not establish detector accuracy or a sustained performance
  baseline.
- The shared server-event and delivery-latency reducer fixtures pass;
  differential coverage for remaining collector behavior, full collector
  parity, measured instrumentation overhead and final retirement remain
  required before removing the Python collector.

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
