# ADR-0086: Retire the incomplete native visualizer

- Status: superseded
- Date: 2026-10-23
- Deciders: Project maintainers
- Supersedes: None
- Superseded by: ADR-0091 for projector test and browser-packaging tooling

## Context

The supported browser application is the HTML/JavaScript/CSS live view, and the
user has limited its route set to `/live`. The parallel LÖVE visualizer is a
separate client whose README still lists WebSocket integration as future work
and describes ball, cue, table, and camera rendering as placeholders. Its
network module requires `lib.websocket`, which is absent from the repository;
the `make test-visualizer` smoke check therefore fails during module loading.
The client only draws trajectory overlays and a diagnostic HUD, which duplicates
a secondary feature while adding an unmaintained UI and a misleading Makefile
entry point.

The separate `frontend/projector` application serves a distinct projection use
case and is not covered by this decision.

## Decision

Retire `frontend/visualizer`, its initialization runner, and the
`run-visualizer`/`test-visualizer` Make targets. Keep the standard browser
application as the only general-purpose operator UI. Preserve
`frontend/projector` as a separate native consumer. Its retained test and
packaging tooling is reconsidered by ADR-0091.

## Consequences

### Positive

- Removes an unbuildable duplicate UI that has no ball, table, or camera
  rendering and cannot connect because its WebSocket transport dependency is
  missing.
- Makes `make help` describe executable workflows rather than an incomplete
  trajectory-only prototype.

### Negative

- The separate LÖVE trajectory window is no longer available. Trajectory
  predictions remain a secondary Rust/WebSocket feature; future display work
  belongs in the supported browser application or the projector client.

## Alternatives considered

### Restore the missing Lua WebSocket dependency and complete the LÖVE client

Rejected because it would revive a second general-purpose client with
placeholder ball/table/video modules after the supported browser surface was
deliberately reduced to `/live`. No current workflow depends on the prototype.

## Verification

- `make help` and repository searches contain no `run-visualizer`,
  `test-visualizer`, or `frontend/visualizer` entry point.
- `make test-rust`, `make test-web`, and the full compatibility test suite pass;
  the retired visualizer-specific smoke test is no longer part of the supported
  check set.
- The separate projector protocol tests and projector files remain present.
- The ADR validator accepts this record and its register entry.
