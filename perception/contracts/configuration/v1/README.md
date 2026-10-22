# Unified configuration contract v1

`defaults.json` and `schema.json` define the unified application configuration
document consumed by the Rust configuration authority. The defaults preserve
the existing public JSON shape and use revision `1` as the initial store
revision. The schema includes the cross-field rules currently exposed by the
Python API. Rust adds the matching semantic checks for camera source URIs,
stable camera selectors, image-size alignment, calibration-control settings,
and the existing high-frame-rate warning.

The schema is JSON Schema-style with the existing `x-cross-field-rules`
extension. A string-valued camera source accepts either an anonymous HTTP(S)
MJPEG URL or a `gstreamer-device-v1:` selector; `schema.json` models both
alternatives explicitly. Credential-bearing URLs and URL fragments are
rejected.

The one-shot command is `billiards-perception-config`. It requires an explicit
`--config <path>` and accepts one JSON object on stdin. Every request includes
`protocol_version: 1` and an `operation`. Responses are a single JSON object on
stdout, followed by a newline. Standard input, standard output, stored-file
size, and cross-process lock wait are bounded by CLI options with defaults of
1 MiB, 2 MiB, 1 MiB, and 2 seconds respectively.

The command also rejects configured limits above 16 MiB of input, 32 MiB of
output, 16 MiB of stored configuration, or a 60-second lock wait, so changing
CLI options cannot disable the resource bounds.

Supported operations are:

- `snapshot`: read the current configuration and store revision.
- `schema` and `defaults`: return the versioned contract.
- `validate`: validate `data` without persisting it. `merge` defaults to true;
  false validates replacement/import semantics. Warnings are reported without
  applying them.
- `update`: recursively merge a non-empty `data` object into the current
  document.
- `replace`: replace the document with a non-empty `data` object. An imported
  `metadata.configuration_revision` is normalized to the current store value
  and cannot select the committed revision.
- `reset_section`: restore each listed top-level `sections` entry from the
  defaults.
- `export`: return the current document, optionally filtered by `sections`,
  with optional default expansion and metadata removal.

Mutations require `expected_revision`. A mismatch returns
`error.code = "revision_conflict"` and `current_revision`. Updates cannot set
`metadata.configuration_revision`; replace/import ignores the supplied value
and normalizes it to the current revision. Rust assigns the next revision only
after complete merged validation and a successful atomic write. A no-op keeps
the existing revision. Existing files without a revision are validated and
atomically migrated to the contract's initial revision on their first snapshot
read; validate-only remains read-only, and a changed mutation writes the
migrated document with its next store revision.

The temporary file is flushed before its atomic rename. If the rename succeeds
but the containing directory cannot be synchronized, the command reports the
committed revision with `durability: "uncertain"`; it does not report a failed
mutation after the new revision is already visible.

All errors use `{ "ok": false, "error": { "code", "message" } }`; validation
errors and warnings are included as separate arrays where relevant. Mutations
with warnings require `force_update: true`; validation-only requests always
report warnings without changing the file.
