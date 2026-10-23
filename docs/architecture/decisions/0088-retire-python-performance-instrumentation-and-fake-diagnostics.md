# ADR-0088: Retire Python performance instrumentation and fake diagnostics

- Status: accepted
- Date: 2026-10-23
- Deciders: Project maintainers
- Supersedes: ADR-0085's temporary retention of Python compatibility profiler and status endpoints
- Superseded by: None

## Context

The supported perception and application runtime now runs in Rust. The retained
Python FastAPI package is an offline compatibility reference; its old vision
pipeline is not the pipeline used for live capture or state delivery. ADR-0085
therefore retired the standalone Python profiler and collector, but temporarily
kept the compatibility service's profiler/status routes while the service
remained available as a reference.

That boundary proved misleading in practice. Python audit hooks and settings
still crossed camera capture, shared-memory publication, vision inference,
integration, MJPEG streaming, and WebSocket publication. They added dormant
branches and tests to a pipeline that does not run the supported application.
The Python routes could not measure Rust, and their reported latency could not
be used to qualify the actual runtime. Separately, the Python system-diagnostics
route reported fixed disk usage, fixed process metadata, and zero upload bytes;
its output looked like system evidence but did not inspect the machine.

The same compatibility API also retained unused health-diagnostics models and
a profiler-style WebSocket performance stream with no current publisher. Its
separate metrics-message schema did not match the transport counters emitted by
the REST metrics endpoint. Those abandoned types kept an unsupported wire
contract alive after their producers were removed.

The Rust live audit already measures current state delivery and optional
detector/resource interference. The Rust recorded-video replay measures
detector capacity and freshness, and the Rust temporal evaluator measures
accuracy against reviewed labels. These tools have bounded inputs, explicit
runtime boundaries, and tests for their protocols. Real health, readiness,
request, and WebSocket-delivery metrics remain available where they describe
actual current state and transport behavior.

## Decision

Retire the Python vision-profiler route, fake diagnostics route, Python-only
event collector, and every hook whose sole purpose was measuring the retired
Python capture and detection pipeline. Remove their configuration defaults and
validation, obsolete report adapters, diagnostics-only models, unproduced
WebSocket performance/metrics messages, and profiler-only tests. Do not keep
the instrumentation solely to service offline compatibility tests.

Use the Rust live audit for live state receipt and optional detector/resource
measurements; use Rust recorded-video replay for saved-input freshness and
capacity; and use the Rust temporal evaluator for accuracy evidence. Their run
settings are explicit command arguments, not a global application
`performance_audit` configuration section.

Keep operationally meaningful behavior and metrics: health/readiness results,
current host measurements, request timing, source freshness, per-client queue
depth and drops, latest-state replacement, stale-result rejection, and bounded
slow-client isolation. This decision removes measurement side channels only;
it does not change current state publication, scheduling, or delivery semantics.

## Consequences

### Positive

- There is one supported measurement owner for the Rust pipeline, so the
  compatibility package cannot produce results that look like Rust evidence.
- Removing dormant timing branches reduces maintenance and avoids validating
  instrumentation that has no supported application caller.
- The fake diagnostics route can no longer imply that placeholder values are
  measurements of the current machine.
- Tests now focus on retained source identity, freshness rejection, state
  publication, queue coalescing, and slow-client behavior rather than retired
  profiler events.

### Negative

- Historical Python profiler reports and routes are no longer reproducible as
  a current workflow; old reports remain historical only.
- Python compatibility tests no longer expose profiler or synthetic system
  diagnostics endpoints.
- Rust live audit does not include browser rendering or physical display
  latency, and accuracy still requires representative reviewed temporal labels.

## Alternatives considered

### Keep Python profiler and status endpoints only for compatibility tests

Rejected because no supported product or test caller needs those endpoints,
and their presence encouraged comparing results from a retired pipeline with
Rust measurements. Focused tests merely kept the stale route surface alive.

### Port the Python profiler/status APIs to Rust

Rejected because the Rust live, replay, temporal-evaluation, health, and
delivery-metrics surfaces already own the applicable measurements. Recreating
the retired API shape would preserve an unnecessary compatibility contract.

### Preserve the system diagnostics route with clearer labels

Rejected because fixed disk usage and process metadata are not useful
diagnostics. Real host and runtime information remains available from the
existing health/readiness paths.

## Verification

- The retired profiler, collector, route, hooks, and config keys have no code or
  test references.
- Freshness, unique-observation, latest-state, slow-client, and stream
  backpressure contract tests pass after removing instrumentation-only checks.
- The Python compatibility API does not register profiler or fake diagnostics
  routes.
- The compatibility WebSocket protocol has no profiler-only performance stream
  or unproduced metrics message; real HTTP health and transport counters remain.
- The full backend test and configured coverage gate pass; Rust audit behavior
  remains covered by its existing unit and integration suites.
- ADR validation succeeds with ADR-0085's scoped supersession recorded.
