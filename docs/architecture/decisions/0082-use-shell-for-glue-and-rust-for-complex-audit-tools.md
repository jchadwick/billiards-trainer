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
freshness, bounds long-run samples, collects process resources, starts a Rust
live process, and writes versioned reports. Shell is not a suitable replacement
for those stateful protocols and measurements.

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

## Verification

- Rust tests cover malformed and oversized messages, fresh and repeated source
  identities, stream order and gaps, bounded retention, and unavailable clock
  comparisons.
- A local real-WebSocket test exercises the subscriber against the Rust
  gateway, including startup readiness and connection termination.
- Differential fixtures compare Rust summaries with the retained Python
  reference on the same versioned state messages; inconclusive or unsupported
  measurements remain explicitly unavailable.
- Rust formatting, strict Clippy, all-target tests, and ADR validation pass.
- A recorded-video live smoke validates integration only; it does not establish
  detector accuracy or a performance baseline.

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
