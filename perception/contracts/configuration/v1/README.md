# Configuration v1

`defaults.json` and `schema.json` define the runtime configuration consumed by
the Rust configuration authority and HTTP API. Rust performs both structural
and cross-field validation before a change is applied.

The `billiards-perception-config` command accepts one JSON request on stdin and
returns one JSON response on stdout. Operations include `snapshot`, `schema`,
`defaults`, `validate`, `update`, `replace`, `reset_section`, and `export`.
Mutations require the expected configuration revision. Rust validates the
complete resulting document and atomically persists successful changes.

Live defaults have no recorded-video path or selected camera. The gateway's
setup action saves the selected source using the same revision and validation
rules. Existing recorded-video settings are accepted by the configuration
contract and rejected by live capture.

The command bounds input/output size, stored-file size, and lock wait time.
Warnings do not change configuration; mutations with warnings require an
explicit `force_update`. The HTTP routes expose the same validation and
persistence rules.

## Whole-table perception continuity

`perception.rust_live.phase_schedule.quiet_dwell_ms` is the configured settle
debounce. A successful clean scan after this quiet interval replaces the
active table layout wholesale. The legacy `layout_consensus_count` and
`layout_position_tolerance_px` keys remain accepted for older configuration
documents, but no longer delay replacement. `layout_evidence_expiry_ms` is
also retained for wire compatibility and diagnostic age reporting; trusted
layout freshness follows `motion_evidence_expiry_ms` instead. This prevents
slow detector time and old-but-stationary scan timestamps from expiring a
layout while current whole-table quiet evidence continues.
