# ADR-0039: Represent driver-default camera controls explicitly in capture settings v3

- Status: accepted
- Date: 2026-10-21
- Deciders: Project maintainers
- Supersedes: None
- Superseded by: None

## Context

The intended device is a regular webcam with no per-camera control setup. The
current capture request nevertheless defaults to automatic exposure and gain
1.0, marks both as required, and demands exact capability/readback evidence.
The legacy Python camera path also writes those values. A generic camera may
operate correctly with its driver defaults while reporting those controls as
unknown or unsupported; requiring fabricated readbacks would make that source
unusable, while silently ignoring requested values would misstate the request.

The existing v1/v2 contracts are closed and have strict semantics. They must
remain unchanged for recorded fixtures and any existing consumers. The source
report already distinguishes supported, unsupported, unknown, and applied
values, so no new camera-control acknowledgment states are needed.

## Decision

Add capture-settings v3 with an explicit tagged camera-control policy:

- `source_defaults` tells the adapter to perform no exposure, gain, focus, or
  image-control writes. It carries no requested control values and imposes no
  camera-control capability or readback startup requirements. The adapter
  still reports every capability and readback it can actually observe.
- `require` carries explicit requested controls and required capabilities. It
  retains the existing strict behavior: readiness fails if a required
  capability is missing or unknown, or a required applied readback differs.

Keep v1/v2 parsing and behavior unchanged. Make `source_defaults` the effective
policy for the generic no-configuration webcam path; explicit controls remain
available through `require`. Preserve exact request hashing, mode/geometry and
device-identity validation, memory/freshness bounds, and source lifecycle
checks in both modes. Never invent an exposure or gain readback. This policy
does not promise that driver defaults are optimal for ball detection; measure
image quality and localization separately.

## Consequences

### Positive

- Allows a regular webcam to start without pretending its driver controls are
  configurable or known.
- Preserves a strict opt-in path for deployments that set camera controls.
- Leaves the existing v1/v2 interop behavior stable.

### Negative

- Driver-default exposure and gain can vary by device, OS, lighting, and
  firmware; detection quality must be evaluated on the actual table.
- Adds a third request version that Python and Rust must implement
  consistently. The existing acknowledgement v2 already carries the complete
  source report and remains compatible with v3 requests.

## Alternatives considered

### Encode no requirements as false flags in v1/v2

Rejected because v1/v2 require concrete `requested_controls`, and Rust
validates those fields against the capability flags. An all-false combination
would leave unclear whether the adapter should still write the values and
would change a closed contract's semantics.

### Continue requiring automatic exposure and gain

Rejected for the generic default path because the user has no per-camera
configuration and ordinary webcam drivers do not all expose these controls.
Explicit control requests keep the current strict checks in v3's `require`
mode.

## Verification

- Rust and Python v3 fixtures round-trip the `source_defaults` tagged variant;
  v1/v2 fixtures and behavior remain unchanged.
- A source reporting controls as unsupported or unknown can become ready under
  `source_defaults`, and the acknowledgment preserves those exact statuses.
- The same source fails under v3 `require` when required controls are missing,
  unknown, or mismatched.
- A camera test double confirms source-default mode performs no control writes.
- The generic camera helper still verifies requested/negotiated mode, identity,
  frame geometry, bounded freshness, and release/restart independently.
