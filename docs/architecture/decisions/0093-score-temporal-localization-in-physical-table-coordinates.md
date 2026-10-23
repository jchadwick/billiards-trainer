# ADR-0093: Score temporal localization in physical table coordinates

- Status: accepted
- Date: 2026-10-23
- Deciders: Project maintainers
- Supersedes: None
- Superseded by: None

## Context

ADR-0013 makes calibrated millimeter coordinates the physical-location
contract, but explicitly says calibration fit and pixel round trips do not
establish physical ball-center accuracy. The Rust temporal evaluator scores
pixel-space detections only, and recorded-video replay currently refuses a
calibration bundle. No independently measured temporal physical-label corpus
exists, so the project cannot yet rank the physical accuracy of detector or
calibration changes.

The physical axes already belong to the immutable table-calibration v1
contract: the origin is the center of the playing surface; positive X follows
calibration corner 0 to corner 1; positive Y follows corner 0 to corner 3.
For the standard image ordering, these point right and down respectively. Any
physical labels must use the same frame and millimeter units. Labels derived
from the calibration under test would be circular and cannot qualify accuracy.

## Decision

Extend the existing temporal annotation/evaluation path with annotation schema
v2. Preserve v1 as pixel-only. V2 records the exact table coordinate frame and
dimensions, an explicit independent physical-measurement method, its evidence
description, and an estimated 95% radial position uncertainty. Each frame
declares physical ground truth either complete or unavailable. A complete frame
must supply a table-space center for every visible ball; an unavailable frame
must supply none. This prevents incomplete labels from contaminating physical
precision and recall.

The evaluator independently matches current-frame observed detections in pixel
space and table space. Physical matching uses a required, caller-supplied
millimeter gate; there is no default physical tolerance. Only detections with a
current calibrated table position can match physical truth. Report calibrated
and unlocalized detections separately, score physical precision/recall and
millimeter error on complete frames, and count out-of-table predictions without
clamping them. Cached, predicted, skipped, failed, and missing-frame results do
not count as detections, consistent with ADR-0035.

Allow recorded-video replay to apply its existing immutable calibration bundle
when producing scene output. Preserve the source presentation-time basis for
PTS-paced replay. Report source-to-calibration duration only for paced replay;
unpaced capacity runs report that latency as unavailable. Replay timing remains
synthetic and is never camera exposure age. The ONNX detector, inference model,
scene transport, and browser contracts do not change.

## Consequences

### Positive

- Physical localization claims can be evaluated in the same Rust-owned temporal
  evidence path as pixel localization and freshness.
- Calibration and detector error are measured against independently supplied
  truth instead of being inferred from homography fit or pixel error.
- Incomplete physical labels and missing calibrated output remain visible and
  cannot silently produce optimistic precision/recall.
- Existing v1 labels, importer output and pixel metrics remain compatible.

### Negative

- Physical accuracy still cannot be claimed until a held-out set of complete,
  independently measured frames is collected and reviewed.
- V2 adds annotation cost and requires the annotator to record a defensible
  measurement method and uncertainty.
- The caller must select a physical match gate appropriate to label uncertainty
  and the product requirement; results from different gates are not directly
  comparable unless the gate is recorded.
- Recorded-video calibration qualifies geometry plumbing, not camera-exposure
  timing, live capture, camera calibration quality, or numbered-ball identity.

## Alternatives considered

### Infer millimeter accuracy from image-space error or homography residual

Rejected because neither measurement is independent of the transform and neither
captures held-out ball-center, parallax, or distortion error.

### Reuse the pixel-radius gate after converting it to millimeters

Rejected because apparent pixel radius varies with perspective and would make
the physical acceptance tolerance position-dependent without a validated
uncertainty model.

### Keep physical labels out of the temporal evaluator

Rejected because the existing scene contract already carries calibrated table
positions. A separate scorer would split frame identity, freshness, and
matching semantics and invite incompatible accuracy reports.

## Verification

- Rust unit tests cover v1 compatibility, v2 completeness rules, explicit gate
  validation, missing calibration, out-of-table output, independent pixel/mm
  assignment, and unavailable/cached observations.
- Shared JSON Schema tests validate valid and invalid v1/v2 annotation samples.
- Scene-adapter tests verify calibrated table coordinates survive into the
  evaluator under the scene's valid calibration state. Runtime unit tests
  verify calibration timing is reported only for paced replay; an automated
  end-to-end real-video calibration fixture remains follow-up work.
- CLI tests verify the explicit physical gate and calibration options are
  validated and forwarded.
- ADR validation and perception/backend test suites must pass. Until real
  independently measured held-out labels exist, tests prove contract behavior
  only and the production physical-accuracy status remains open.
