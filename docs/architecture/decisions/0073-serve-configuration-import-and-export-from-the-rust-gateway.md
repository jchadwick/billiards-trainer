# ADR-0073: Serve configuration import and export from the Rust gateway

- Status: accepted
- Date: 2026-10-22
- Deciders: Project maintainers
- Supersedes: None
- Superseded by: None

## Context

Configuration reads, updates, and resets now run directly in the Rust gateway
under ADR-0070, ADR-0071, and ADR-0072. Import, export, and download still use
Python routes, leaving configuration management split across server runtimes.
The browser already sends multipart uploads and consumes the existing JSON
export/download response contracts. Configuration remains anonymous; installed
server executable paths are deployment-managed and must not be replaceable via
HTTP.

Imports accept JSON and YAML objects, `replace` or `merge`, validation-only
requests, and an explicit override for validation warnings. They must preserve
validation, revision compare-and-swap, and executable-path protection. Export
supports JSON or YAML, repeated section selection, optional defaults and an
optional transport metadata envelope. Download returns the serialized file as
an attachment. These endpoints need bounded input, output, concurrency, and
blocking work so file parsing or serialization cannot stall the gateway.

The browser boundary is fixed by ADR-0066: the client remains HTML, JavaScript,
and CSS. Static browser assets may be served by Rust, but no browser code is to
be converted to Rust/WASM.

## Decision

Implement anonymous configuration import, export, and download directly in
the Rust gateway, preserving the established route paths, query/form fields,
success and error envelopes, and serialized data semantics. Parse imports as
bounded JSON or safe YAML, require an object root, discard the transport-only
`_metadata` root field, and protect deployment-managed executable paths. Use
the Rust configuration authority for defaults, section validation, validation,
revision reads, and revision-checked merge/replace. Validation-only requests
must not write the configuration file. Warning-bearing imports require
`force_update=true`; validation errors remain failures even when forced.

Parse multipart uploads only after checking route parameters and acquiring the
existing bounded configuration-request permit. Run authority operations and
CPU-heavy serialization on bounded blocking workers. Serialize mutations
through the shared mutation lock and commit against the revision read for the
same import; do not retry conflicts. Enforce the authority's configured input
limit for the complete multipart body and output limit for generated exports.
When the request exceeds a configured bound or capacity is unavailable, return
a sanitized service error without mutating state.

JSON and YAML exports support repeated section filters, defaults, and the
transport `_metadata` envelope. The envelope's `sections` list describes the
configuration sections before `_metadata` is added. Report `size` and SHA-256
checksum for the exact UTF-8 serialized representation. Downloads use the same
serializer and bounded export data, return an attachment with a generated
UTC-second filename and `application/octet-stream`, and do not write temporary
files. Retain JSON as the default format, no defaults as the default, and
metadata inclusion as the default. Keep the browser HTML/JavaScript/CSS and
make no WASM conversion.

## Consequences

### Positive

- Rust becomes the direct owner of the full configuration HTTP surface and no
  Python route is needed for configuration file import/export.
- Existing browser workflows can continue using the same multipart and
  download contracts while static delivery and all server processes migrate to
  Rust.
- Bounded parsing, export, and in-memory download avoid unbounded buffering,
  temporary-file collisions, and event-loop blocking.

### Negative

- YAML output formatting follows the selected Rust YAML serializer and may not
  be byte-for-byte identical to PyYAML; semantic data and response checksum
  correctness remain the compatibility contract.
- Multipart framing consumes part of the configured request-size budget, so
  the largest accepted file is smaller than that total-body limit.
- Configuration import/export parity does not complete Python gateway
  replacement, static-asset packaging, or browser integration against the
  qualified Rust server.

## Alternatives considered

### Keep import/export in Python while moving only reads and mutations

Rejected because it leaves a routine configuration surface split between
server runtimes after the Rust authority and Rust HTTP gateway are available.

### Buffer uploads without limits or serialize downloads through temporary files

Rejected because client-controlled sizes can exhaust memory or block request
handling, while shared temporary names can collide and require cleanup. The
gateway can return bounded in-memory attachment responses.

### Replace the browser application during this server migration

Rejected by ADR-0066 and the explicit product boundary: the browser remains
HTML/JavaScript/CSS. The Rust service serves its assets without converting the
client to Rust/WASM.

## Verification

- Multipart contract tests cover file field handling, extensions, malformed
  input, JSON/YAML objects, `_metadata`, and bounded bodies.
- HTTP integration tests cover validate-only non-mutation, merge/replace,
  validation warnings and force behavior, executable-path rejection, and
  revision-safe commits.
- Export tests cover JSON/YAML, repeated and unknown sections, defaults,
  metadata, exact serialized size/checksum, configured output bounds, and
  download headers/content.
- Rust formatting, all-target tests, strict Clippy, and the repository ADR
  validator pass.
- Existing HTML/JavaScript/CSS build and browser tests remain applicable; no
  WASM target or toolchain is introduced.
