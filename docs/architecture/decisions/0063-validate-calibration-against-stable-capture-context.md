# ADR-0063: Validate calibration against stable capture context

- Status: accepted
- Date: 2026-10-22
- Deciders: Project maintainers
- Supersedes: None
- Superseded by: None

## Context

ADR-0062 binds a browser calibration workflow to one Rust-owned still image.
The camera continues capturing while an operator selects four corners, so the
capacity-one preview slot normally contains a newer frame by submission time.
Requiring that the selected image remain the slot's latest frame would reject
ordinary calibration attempts even when the active camera session and every
calibration-relevant setting remain unchanged. Rust already owns the solver and
revisioned store; the retained Python service should only proxy the request.

The application intentionally has no authentication mechanism. Frame IDs and
descriptors provide provenance and consistency data, not a secret or proof of
caller identity. The preview response couples its descriptor to the exact JPEG
shown by the frontend, while the Rust control endpoint validates the stable
capture context at submission.

## Decision

When Rust receives a calibration submission, it validates the supplied
descriptor against the currently active engine and capture session, camera
profile, source dimensions, pixel format, image space, configuration revision,
calibration revision, rectification revision, and timestamp semantics. The
source-frame ID and timestamp remain attached as provenance, but the submitted
frame need not still occupy the capacity-one latest-frame slot. This supports
selection on the frozen JPEG while rejecting submissions from ended or
reconfigured capture sessions.

Rust validates ordered source-pixel corners and explicitly measured table
dimensions, solves the calibration, and writes it through `CalibrationStore`
using the client's expected store revision as a compare-and-swap condition. A
failed context check, solve, or store write leaves the active record unchanged.
Saving does not mutate the running capture pipeline; the new calibration is
applied when the next Rust live capture session starts. The Python service may
validate request size and relay results, but it must not decide applicability,
solve, or persist the calibration.

The endpoint does not claim cryptographic proof that a source-frame ID was
issued by Rust. Requiring that proof would need a bounded frame-lease or
descriptor registry, with explicit retention accounting beyond the current
capacity-one preview slot. The UI therefore submits only descriptors returned
with Rust preview images, and stable session context is the acceptance rule.

## Consequences

### Positive

- A frozen preview remains usable while live capture advances, without keeping
  old pixel buffers alive for the duration of operator input.
- Capture-session and revision changes invalidate stale calibration requests,
  while the store's compare-and-swap prevents lost updates.
- Rust remains the sole owner of calibration decisions and persistence.

### Negative

- Rust cannot independently prove that the caller's source-frame ID came from a
  prior preview response; this is provenance validation, not authentication.
- A calibration is for the camera pose visible in the frozen image. Physical
  camera movement that does not restart or reconfigure the session cannot be
  detected by descriptor checks alone.
- A successful save becomes active on the next live capture session, requiring
  a restart/reconfiguration boundary before coordinates use the new revision.

## Alternatives considered

### Require the submitted frame to remain the latest captured frame

Rejected because normal corner selection takes longer than one camera frame;
the frame slot is intentionally latest-only and will replace the selected
image before the operator finishes. This would make the normal frozen-image
workflow unusable.

### Retain a per-frame lease until calibration submission

Rejected for the first implementation because it adds a second frame-retention
budget, expiry policy, and abandoned-lease cleanup. Consider it only if exact
Rust-side proof of the issued source-frame identity becomes a measured
correctness requirement.

## Verification

- Add control-protocol tests beside the existing preview tests in
  `perception/src/frame_preview.rs` that accept an earlier frame in the active
  unchanged session and reject mismatched engine/session, profile, geometry,
  image-space, and revision fields.
- Extend `perception/tests/calibration_store.rs` and the control-protocol tests
  to prove stale expected revisions conflict and invalid submissions leave the
  stored record unchanged.
- Extend
  `frontend/web/src/tests/integration/CalibrationWizardRuntime.test.tsx` to
  prove the descriptor stays paired with its still and refresh clears points
  when frame identity changes.
- Extend `backend/tests/unit/test_rust_live_supervisor.py` and
  `backend/tests/unit/api/test_rust_frame_preview_route.py` to prove
  calibration operations are bounded proxies and never invoke Python
  calibration or configuration writers.
