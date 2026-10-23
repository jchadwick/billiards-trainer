# ADR-0013: Calibrated table positions with image-coordinate compatibility

- Status: superseded
- Date: 2026-10-19
- Deciders: Project maintainers
- Supersedes: None
- Superseded by: ADR-0096 for the Python exporter and direct-test tooling only

> The physical-coordinate contract and Rust calibration implementation remain
> active. ADR-0096 supersedes only the Python bundle exporter and its direct
> tests; the implementation notes below preserve the historical migration
> evidence.

## Context

Existing Core and frontend contracts use canonical 3840-by-2160 image coordinates
with explicit scale metadata under ADR-0007. The primary conversion performs
resolution scaling, which does not establish physical table precision.
Table-meter homography/reference calculations and calibration assets already
exist elsewhere and can be reused.

The new requirement is precise physical ball locations while retaining current
clients and physics behavior. Silently substituting meters into existing pixel
fields would break geometry and invalidate predictions.

## Decision

Make calibrated table-space ball centers the physical-location contract of the
new perception engine, alongside retained source image measurements. Use explicit
coordinate-space/unit/origin definitions and calibration/rectification revisions.
Retain image geometry even when physical geometry is unavailable.

For the first table frame, preserve the existing `CameraCalibrator` convention:
the origin is the center of the playing surface; X increases from ordered
corner 0 to corner 1; Y increases from corner 0 to corner 3. The calibration
corner order is top-left, top-right, bottom-right, bottom-left in the image
space used to author the calibration. Keep the frame fixed to that calibrated
corner order, not to a client viewport or camera orientation. Export lengths
and matrix output in millimeters, converting the retained Python table
dimensions from meters at the boundary. Do not silently change the retained
Python/Core pixel frame.

Rust accepts an immutable versioned table-calibration bundle containing the
calibration revision, camera-profile identity, source image dimensions and
space, rectification revision, measured table dimensions, and image-pixel to
table-millimeter homography. A frame can use it only when all source/profile,
space, dimensions, and revision fields match exactly. Apply the transform to
source-image coordinates after undoing detector resize/letterbox mapping. Do
not scale, crop-adjust, or reinterpret a calibration implicitly. A homography
does not correct lens distortion: raw-space output is usable only with a
calibration explicitly authored and validated for that raw profile; otherwise
physical coordinates are unavailable. Invalid matrices or projective
singularities also produce unavailable geometry, never a pixel-derived
approximation.

The transform establishes coordinates, not accuracy. Keep table-position
uncertainty unavailable until held-out physical measurements support a
calibrated uncertainty estimate. Do not clamp out-of-table measurements; retain
the measured coordinate and let an explicit quality policy classify it.

Keep existing canonical-pixel public/Core fields through one compatibility
adapter with unchanged scale semantics. Add distinctly typed table coordinates;
do not change old fields' units in place. Preserve the exact dimensions of the
source frame in Core camera-resolution metadata; when the current observation
explicitly reports dimensions as unavailable, do not substitute configured
camera dimensions that may describe a different capture mode. Retain the
configured-size fallback only for legacy payloads that omit source-resolution
metadata. The same immutable calibration bundle defines transformations for
all observations carrying that revision.

Validate sensor mode, distortion/rectification, crop, resize and calibration
compatibility. Missing or incompatible calibration produces unavailable/degraded
physical coordinates rather than fabricated precision. Existing calibration
solvers and UI remain under Python initially.

Box centers are the initial measurement reference, not proof of physical-center
accuracy. Evaluate local refinement or a center model separately, using held-out
physical measurements. Quantify ball-height/parallax assumptions; do not infer
millimeter precision from image round-trip tests or rescaled detector error.

## Consequences

### Positive

- Makes physical position accuracy meaningful across camera resolutions/poses.
- Preserves existing UI and physics investment through explicit adaptation.
- Keeps raw/rectified/table/projector geometry distinguishable.
- Calibration changes can invalidate stale spatial outputs deterministically.

### Negative

- Two deliberate geometry representations require careful schemas and tests.
- Physical calibration and held-out measurements are needed for precision claims.
- Refinement and sphere geometry may add complexity if measured error demands it.
- Legacy physics remains image-based until independently migrated.

## Alternatives considered

### Retain canonical pixels as the only authoritative geometry

This is sufficient for some overlays but cannot establish physical table error
or uniform physical distances under perspective.

### Use a top-left table origin

This is a reasonable new convention, but it conflicts with the existing
`CameraCalibrator` world coordinates, which center the origin. Selecting it
would require an additional conversion contract for existing calibration and
physics consumers without improving measurement accuracy.

### Convert all Core, frontend and projector geometry to meters at once

This enlarges the migration substantially and risks unit errors. Add physical
observations first and migrate individual consumers through tested boundaries.

### Apply a plane transform and declare precision solved

A transform must match the frame's rectification and camera mode, and apparent
sphere centers need physical validation. Calibration fit error alone is not
held-out ball-location accuracy.

## Verification

- Shared fixtures preserve source/crop/resize offsets and canonical scale.
- Detector-to-Core tests verify actual source dimensions at 1080p and 4K, and
  preserve unavailable camera-resolution metadata rather than substituting a
  configured capture mode ([integration contract](../../../backend/tests/integration/test_detector_to_core_localization.py),
  [legacy fallback test](../../../backend/tests/unit/test_core_coordinate_invariants.py)).
- Calibration bundle fixtures pin center-origin axes, millimeter units, exact
  image-space/resolution/profile/revision matching, and unavailable outcomes
  for incompatible or numerically invalid transforms.
- Legacy payloads and Core coordinate tests remain unchanged in meaning.
- Held-out table placements measure center/rail/corner error in physical units.
- Incompatible modes, raw-frame fallback and calibration replacement produce
  explicit unavailable geometry or correctly revised observations.
- In-flight results remain tied to the calibration actually used.

## Revisit conditions and related records

This accepted decision changes new perception authority, not existing
image-space API semantics. ADR-0007 remains applicable to compatibility fields.
If its scope changes later, record the precise superseded scope and retained
rules; do not erase the historical pixel decision. ADR-0014 governs calibration
ownership.


## Evidence and implementation tracking

The v1 coordinate frame, units, provenance fields, exact compatibility rules,
and unavailable-on-mismatch behavior are accepted. Rust runtime integration,
tracking, physical accuracy qualification, and gateway compatibility work are
not complete. This record governs migration stages 3 and 4.

- [Migration plan](../../../thoughts/rust-perception-migration-plan.md).
- [Architectural review and evidence limitations](../../../thoughts/architectural-review-2026-10-19.md).
- [Existing backend/integration_service_conversion_helpers.py](../../../backend/integration_service_conversion_helpers.py).
- [Existing backend/core/integration.py](../../../backend/core/integration.py).
- The legacy Python calibration bundle exporter converted the retained
  meter-based homography to millimeters and required explicit source/profile/
  revision provenance. It had no production caller and was retired under
  [ADR-0096](0096-retire-the-unused-python-calibration-bundle-bridge.md).
- [Rust calibration validator and projector](../../../perception/src/calibration.rs)
  rejects mismatched metadata and numerically undefined transforms.
- [Rust calibration audit options](../../../perception/src/inference/runtime.rs)
  load and validate the hashed bundle as one immutable value, binding reported
  provenance to the geometry used by replay.
- [Rust detection-batch geometry adapter](../../../perception/src/geometry.rs)
  preserves source identity, validates applied frame metadata for successful and
  empty batches, and returns no partial batch when any detection cannot project.
- [Shared calibration contracts and fixtures](../../../perception/contracts/calibration/v1/)
  retain the historically Python-generated perspective bundle as a golden
  geometry reference consumed by Rust tests.
- [Rust calibration tests](../../../perception/tests/calibration_v1.rs)
  verify the solver, projection, center-origin frame, and exact context
  matching. Capture-request validation remains covered by the Python
  compatibility test suite.

The former Python exporter and Rust projection test agreed on all four corners
of a non-affine perspective calibration. This is historical migration evidence;
the adapter and direct tests were later retired under ADR-0096 after Rust took
ownership of calibration. The fixture remains for Rust solver/projection
regressions. These results verify contract translation, not held-out physical
ball-center accuracy. The
recorded-image audit now has an opt-in path that applies the adapter, reports
per-frame calibrated measurements or unavailable status, and records geometry
time separately while retaining raw batches. The experimental source-time
tracker can reuse that per-frame geometry result to add calibrated table
coordinates to scene-snapshot v1 while preserving image positions on
calibration failure. This is replay-only and synchronously writes scene JSONL.
The live producer is not yet wired to the adapter, and live detection batches do
not yet carry the camera profile or applied calibration revisions needed to
construct the frame context. The repository currently contains camera
intrinsics but no saved table-plane transform or source-profile provenance.
Physical uncertainty remains unavailable until measured.

The gateway compatibility prototype now keeps pixel and physical coordinates
distinct in the existing state payload: raw image positions are scaled once to
canonical 4K pixels, while valid table positions carry millimeters and the
center-origin label in a separate field. Rectified points are rejected until a
matching rectified preview or inverse transform exists. Tests cover 4K and
1080p scaling, calibrated coordinates, source identity, and rejection of
incompatible geometry. Its radius is explicitly the retained standard-ball
rendering assumption, not a measured scene dimension. FastAPI provider
selection, live calibration application, and a browser-to-server real-socket
run remain open; no physical accuracy claim follows from this adapter.

The device-independent Rust capture pipeline now accepts an optional validated
calibration with an explicit applied profile and calibration/rectification
revisions. It rejects startup disagreement with tracking configuration or
image space, validates every frame's dimensions during projection, preserves
image measurements, and withholds table coordinates when projection is
unavailable or invalid. A fake-source real-socket integration test verifies
that both pixel and table-millimeter locations reach the loopback receiver;
the pipeline reports geometry time separately. This closes the library-level
capture/calibration seam but does not prove that Python configuration was
acknowledged by a real camera, that a physical table transform is accurate, or
that temporal ball-center annotations meet an error target. Rust live startup,
camera-profile/mode acknowledgement, and physical validation remain open.
