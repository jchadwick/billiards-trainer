# Pool coach reset execution record

Software implementation complete; physical-table play acceptance not performed.
The accepted scope is in [the target](product-reset-20261113.md).

## Delivered

- `c0d9c313`: editable, explicitly simulated `/practice` sandbox and reusable
  physical table-space direct-pot geometry.
- `d94bf963`: approved lean product target and execution requirements.
- `3b0c6d2a`: ADR-0114 and supersession/narrowing of conflicting requirements.
- `c4a90cd7`: perception-owned trusted/stale layout, clean-scan replacement,
  slow-result admission under uninterrupted quiet, stable IDs, capture-based
  ages, ephemeral cue results, ordinary table homography fallback, and published
  revision-bound projection metadata.
- `ad8cac7e`: `/live` labeled solid/ghost markers, player-confirmed cue/target
  selection, projected direct-pot suggestions, independent current cue output,
  image-center fallback, contained-video alignment, and compatible source guards.
- Final documentation aligns README, architecture, API, configuration, and
  phase-aware behavior with the implementation.

Executors used Luna with xhigh effort. Independent reviewers examined each slice.
Review findings were fixed and covered with focused regressions, including
source replacement, subsequent same-source revisions, absent precision assets,
late worker polling, stable track IDs, and image letterboxing.

## Verification

The final command suite ran against implementation head `ad8cac7e` and passed:

- `make test`
- `make test-rust-camera`
- `make lint`
- `make format-check`
- `make config-check`
- `make test-deploy`
- `make test-tooling`

Frontend typecheck and production build passed; the working checkout ran 116
frontend tests. An export containing only the selected core commit and its
prerequisites independently passed 65 frontend tests, typecheck, and build.
The perception commit was similarly verified as an isolated export, including
all-target Rust tests and deployment smoke coverage. Commit hooks also passed.
Documentation links, ADR structure, and changed-file whitespace were checked.

Existing administration/shared-receiver work was already modified or untracked
before this reset. It remains in the working tree rather than being swept into
core commits. Small compatibility adaptations remain with that existing work:
practice navigation, optional source-selection revision metadata in the shared
receiver, and correct precision-asset diagnostics when coarse fallback succeeds.
These are not hidden build dependencies of the independently verified core.

## Use and remaining acceptance

Build/start using the existing supported runtime (`make run`), then use
`/practice` without camera data or `/live` with the configured camera and ordinary
corner calibration. Marker labels identify selectable tracks, not recognized
pool-ball numbers. Green is desired aim, orange is the estimated object path,
and yellow is the current cue; signed current aim can also show its own travel
estimate. Stale layouts retain ghost markers but cannot supply current advice.

No physical billiards-table camera session, shooting trial, production deployment,
or physical accuracy/latency qualification was performed. The live runtime's
existing Intel GPU/model requirements remain. The next acceptance activity is
ordinary play: leave the table still through slow scans, aim, shoot, wait for
settlement/reacquisition, briefly occlude the view, and test feed loss. Tune
configuration from observed play problems instead of adding qualification gates.
