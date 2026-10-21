# ADR-0034: Separate camera input mode from emitted RGB frame geometry

- Status: accepted
- Date: 2026-10-21
- Deciders: Project maintainers
- Supersedes: None
- Superseded by: None

## Context

The capture request already carries a source profile: raw camera width,
height, rational frame rate, and pixel format. The owned-frame boundary emits
tightly packed RGB8, which can be decoded from a compressed or packed camera
mode and can also have different dimensions after rectification. The current
preflight compares the raw negotiated profile's dimensions directly with each
emitted RGB8 frame. That conflates two independently observable facts and
prevents an adapter from truthfully reporting a compressed/native camera mode
and a transformed inference frame.

Capture settings and request-bound acknowledgements reject unknown fields.
Changing their v1 shapes would break the Python/Rust contract and existing
fixtures. No Rust camera adapter is qualified; adapter research exposed this
gap before device selection, and the target camera and Linux host remain
unidentified.

## Decision

Keep the requested and negotiated capture profiles as the raw camera input
mode. Add a distinct output-frame geometry for the RGB8 frames actually
emitted to inference, including width and height; RGB8 is fixed by
`RgbFrameEnvelope`. Keep frame rate attached to the camera input profile.
Treat calibration dimensions as belonging to the emitted image space, and
validate every frame against the acknowledged output geometry.

Introduce capture-settings and capture-session-acknowledgement v2 contracts
for the new required geometry field. Preserve the exact v1 request and
acknowledgement schemas and their existing parser behavior. V2 preflight must
fail closed if the source omits or mismatches the requested output geometry.
V1 remains a legacy path whose source-frame dimensions equal the negotiated
profile dimensions; it must not be used to qualify an adapter that changes
frame geometry.

This decision does not select a camera library, alter exposure/gain units,
or qualify a physical camera. The backend configuration and acknowledgement
consumer must migrate to v2 before any adapter using transformed dimensions
can be selected for live mode.

## Consequences

### Positive

- Camera mode, conversion output, and rectification dimensions remain separate
  facts, so source reports do not mislabel compressed or packed input as RGB8.
- The inference boundary can verify the exact dimensions it receives while
  retaining the existing raw-mode negotiation check.
- Existing v1 clients and fixtures remain unchanged during the versioned
  migration.

### Negative

- The source report, request parser, Python configuration producer, and Python
  acknowledgement validator need coordinated v2 support before the new field
  is usable end to end.
- V1 cannot describe dimension-changing capture transforms and remains limited
  to equal input/output dimensions.

## Alternatives considered

### Continue treating the negotiated camera profile as RGB8 output

Rejected because it hides the raw device mode and cannot distinguish camera
format negotiation from adapter decoding or rectification. That makes adapter
readbacks and calibration provenance ambiguous.

### Add the new field to v1

Rejected because both schemas reject unknown properties and the request digest
binds the exact serialized bytes. Mutating v1 would silently change a strict
cross-language contract.

## Verification

- Rust unit and integration tests must accept raw camera dimensions that differ
  from emitted RGB8 dimensions, then reject frames that differ from the v2
  output geometry.
- V2 request and acknowledgement fixtures must validate in Rust and Python;
  existing v1 fixtures must continue to validate byte-for-byte.
- The live capture preflight must reject a v2 source report with absent or
  mismatched output geometry before frame inference or scene publication.
- The ADR validator must pass for the full decision register.

## Evidence and implementation tracking

- [Rust source profile and RGB8 frame boundary](../../../perception/src/capture.rs).
- [Capture-to-scene frame preflight](../../../perception/src/capture_pipeline.rs).
- [Strict v1 request schema](../../../perception/contracts/capture-settings/v1/README.md).
- [Camera adapter feasibility review](../../../thoughts/rust-camera-adapter-spike.md).
- [Python v1/v2 request resolution and memory/calibration checks](../../../backend/perception/capture_configuration.py).
- [Python acknowledgement and scene-geometry validation](../../../backend/perception/capture_acknowledgement.py).
- [V2 native-default and transformed-output fixtures](../../../perception/contracts/capture-settings/v2/README.md).
- [Implementation status and verification evidence](../../../thoughts/rust-perception-migration-plan.md#v2-frame-geometry-bridge-and-adversarial-boundary-corrections-2026-10-21).
