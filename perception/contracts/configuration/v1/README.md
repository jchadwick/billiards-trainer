# Configuration v1

`defaults.json` and `schema.json` define the runtime configuration consumed by
the Rust configuration authority and HTTP API. Rust performs both structural
and cross-field validation before a change is applied.

The `billiards-perception-config` command accepts one JSON request on stdin and
returns one JSON response on stdout. Operations include `snapshot`, `schema`,
`defaults`, `validate`, `update`, `replace`, `reset_section`, and `export`.
Mutations require the expected configuration revision. Rust validates the
complete resulting document and atomically persists successful changes.

New local examples clear `vision.camera.video_file_path` for the live camera
path. The versioned defaults retain a recorded-video setting for existing
configuration clients; the live capture runtime rejects a nonempty file path.

The command bounds input/output size, stored-file size, and lock wait time.
Warnings do not change configuration; mutations with warnings require an
explicit `force_update`. The HTTP routes expose the same validation and
persistence rules.
