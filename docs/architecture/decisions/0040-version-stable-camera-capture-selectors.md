# ADR-0040: Version stable camera capture selectors separately from provider identifiers

- Status: superseded
- Date: 2026-10-21
- Deciders: Project maintainers
- Supersedes: ADR-0037, only for discovery-report identity and selector fields
- Superseded by: ADR-0054, only for Linux persistent-identity resolution and
  the source of the reported persistent identifier

ADR-0054 changes only how Linux discovery obtains a persistent identifier when
GStreamer reports a transient V4L node. This record remains in force for the
discovery schema, selector hashing, exact-match capture selection, and
capture-settings compatibility.

## Context

ADR-0037 added a non-capturing GStreamer DeviceMonitor inventory with a raw
provider identifier and a classification of whether that identifier is
persistent. macOS AVFoundation exposes `avf.unique_id`; Linux commonly exposes
a stable `/dev/v4l/by-id/...` or `/dev/v4l/by-path/...` path. The existing
capture-settings v1/v2/v3 `device_id` contract intentionally accepts only
bounded ASCII identifiers containing alphanumeric characters, underscore,
period, colon, and hyphen. It cannot carry Linux paths, and broadening a closed
request schema without versioning would break cross-runtime validation.

The opt-in supervised physical helper must bind one exact provider device and
must not fall back to a numeric index or an arbitrary first enumerated camera.
The helper needs a selector that can be carried by the unchanged capture
request contract while preserving the raw provider identity for diagnostics.
The selector is a stable reference derived from public provider metadata; it
is not a credential, secret, authenticator, or proof of physical-device
behavior.

## Decision

Version the camera-discovery report to schema v2 and add a nullable
`capture_selector` to each device. Populate it only when discovery recognizes
one of the documented persistent identity forms. Derive its bounded ASCII
value as `gstreamer-device-v1:<lowercase-sha256>`, hashing a
domain-separated, unambiguous encoding of platform name, provider-property
key, and exact persistent provider identifier. Keep `identifier_value`,
`identifier_source`, and `identifier_classification` as separate diagnostic
fields; do not replace or reinterpret their existing meanings.

The physical helper must re-enumerate devices and recompute the same selector
from current provider metadata. It opens only one exact selector match; no
match, duplicate matches, an unrecognized/transient identifier, or a failed
mode negotiation prevents readiness. Capture-settings v1/v2/v3 remain
unchanged because the selector fits their existing identifier grammar. Python
configuration may pass the discovered selector in a later integration slice;
numeric IDs remain valid for the retained Python camera path but do not select
a physical Rust source.

This decision does not enable the physical source in the live executable,
select a default webcam, guarantee that a provider identifier is truthful,
alter camera controls, or claim camera freshness or release behavior. The
helper remains opt-in until it is qualified on macOS and Linux hardware.

## Consequences

### Positive

- A single exact selector can identify persistent macOS and Linux devices
  without putting provider-specific path syntax into the capture-settings
  contract.
- The raw identifier and its provider property remain visible for diagnosis,
  while transient `/dev/videoN` identifiers cannot become selectors.
- Existing capture-settings request hashes and v1/v2/v3 parsers remain
  byte-for-byte compatible.

### Negative

- The discovery response gains a new schema version and an additional field;
  discovery clients must validate the version before consuming the response.
- Python configuration and any camera-selection UI must be updated before a
  user can select a discovered stable device for Rust capture.
- The hash hides provider text from the capture request but does not make the
  provider identifier more stable or verify the physical device behind it.

## Alternatives considered

### Put provider paths directly in capture-settings v3

Rejected because the closed v3 identifier grammar excludes `/`, and silently
changing that grammar would create cross-runtime incompatibility. A future
request-schema change is unnecessary for this selector.

### Continue using numeric device indexes

Rejected for Rust physical capture because enumeration order may change across
restart or hot-plug and may identify a different camera. Numeric indices remain
available to the Python compatibility path.

### Use the raw identifier as the selector

Rejected because Linux provider paths do not fit the current bounded request
identifier grammar and because provider-specific IDs have inconsistent syntax.

### Pick the first camera when a stable selector is absent

Rejected because a built-in or virtual camera can precede the table camera.
Selection must be explicit and resolved against provider metadata.

## Verification

- Discovery schema-v2 tests must verify persistent-ID selector derivation,
  deterministic output, platform/property-key domain separation, and absence
  for transient or unavailable identifiers.
- Physical selection tests must verify exactly one match, no match, duplicate
  matches, and rejection of raw numeric/transient IDs.
- The existing capture-settings v1/v2/v3 fixtures and Python/Rust parsers must
  continue to pass unchanged; schema-v3 must accept the bounded selector.
- The opt-in helper must report the resolved provider identifier and exact
  negotiated camera mode before publishing readiness, with RGB8 output geometry
  checked separately.
- Physical open, interruption, freshness, release, and reopen tests remain
  required on macOS and Linux before production source selection.

## Implementation progress (2026-10-21)

Discovery schema v2 now emits deterministic `gstreamer-device-v1:<sha256>`
selectors only for recognized persistent provider identities. The opt-in
`camera-gstreamer-source` feature adds a supervised physical helper path that
re-enumerates and matches exactly one selector, requests the configured camera
input mode, separately verifies negotiated input and RGB8 output profiles, and
publishes request-bound readiness only after the first valid frame. The
one-buffer appsink keeps its application queue bounded. The path accepts v3
`source_defaults` and writes no camera controls. Linux by-id/by-path identity
must be one safe symlink basename; traversal and nested paths are rejected.
Both synthetic and physical RGB copies use the mapped frame's plane stride,
including buffers with custom `GstVideoMeta`.

The source-enabled all-target Rust suite passes (128 library tests plus all
applicable binary and integration tests), as does the feature-disabled suite
(120 library tests plus all applicable targets). Strict Clippy passes with the
physical-source feature enabled, and formatting passes. Exact selector
resolution, identity binding, request validation, and padded mapped-stride
behavior are tested. Python configuration now accepts a nullable
`perception.capture.device_selector`; when set, the v3 resolver places it in the
existing `requested.device_id` field, which already participates in the camera
profile fingerprint. It rejects malformed selectors, non-GStreamer backends,
and selector use with v1/v2, while an unset selector preserves the numeric
Python camera ID and all existing fixtures. Config validation and its generated
JSON Schema enforce the same lowercase selector form. The feature-enabled Rust
live executable and Python supervisor now accept an explicit physical-helper
selection and validated helper launch settings; the default source remains
unavailable. No camera-discovery UI or API has been added, and the production
API still does not select Rust live. No physical camera has been opened, so
device mode availability, freshness, interruption, release/reopen, and image
quality remain unqualified on macOS and Linux.
