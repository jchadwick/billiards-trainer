# ADR-0113: Guide ball calibration with live placement metadata

- Status: accepted
- Date: 2026-11-13
- Deciders: Project maintainers
- Supersedes: None
- Superseded by: None

## Context

The ball-center wizard must guide placement even when the normal game
pipeline has no localized balls. The preview already connects to the camera
service directly. The fit needs independent physical measurements.

## Decision

Show all available placements together with more coverage on the left and
right sides. Overlay the selected target on the direct camera preview and request placement
metadata while the placement step is open. The producer checks its latest
raw image near that target and returns the result as coordinates. Display
confirmation only while a recent result locates the ball in the target.
Prescribe each physical reference as two distances from the cushion noses.
The operator marks that point before placing the ball. The live check
confirms proximity while the prescribed mark provides the independent
physical coordinates used by the fit. The operator confirms the detected
outline on an explicitly captured still before saving each sample.

## Consequences

### Positive

- An operator sees where to put the ball and gets immediate placement feedback.
- The feedback does not depend on a calibrated game-state ball layout.
- The existing camera connection carries the video.

### Negative

- Placement feedback adds bounded work to the producer while in use.
- The camera preview and the placement result are independently delivered.
  Frame-exact outline review remains in the captured still.

## Alternatives considered

Game-state observations can be unavailable before a trusted layout is
established. Repeated JPEG capture would add image encoding and transfer
to the placement loop.

## Verification

Run placement tests for valid, missing, and stale evidence and for the
browser overlay and confirmation states. Check that the deployed placement
route returns metadata and the preview URL still belongs to the camera service.
