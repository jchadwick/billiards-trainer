# ADR-0111: Fit precision calibration from browser stills

- Status: accepted
- Date: 2026-11-03
- Deciders: Project maintainers
- Supersedes: None
- Superseded by: None

## Context

Setup showed missing precision calibration but relied on externally fitted
files. The runtime requires exact camera geometry and independent physical
references. Live video already goes directly from the camera to the browser.

## Decision

Setup collects authenticated single stills for each checkerboard pose and
measured ball placement. Fit the runtime pinhole optics with OpenCV, checking
held-out views and pose coverage. Refine selected ball outlines on the
captured still with the runtime algorithm, and fit against independently
measured table coordinates with the existing solver. Save assets atomically
under the camera profile and reload the producer without changing the
configuration revision. Validate generated and manually configured assets
with the same runtime compatibility rules.

## Consequences

### Positive

- Operators can collect, fit and apply precision calibration from Setup.
- Failed fits preserve the prior asset and return the reason.
- Camera and table bindings are checked before the fit starts.

### Negative

- Optics fitting requires Python with OpenCV on the gateway host.
- Physical references and varied board poses still require an operator.
- Ball outlines are refined from encoded calibration stills. The fit report
  measures consistency with the operator's references and does not establish
  independent physical accuracy.

## Alternatives considered

Applying the table homography directly to ball centers would introduce the
height-dependent error addressed by ADR-0102. Adding a live preview relay
would conflict with ADR-0109. Requiring configuration edits after every
fit would invalidate the existing table binding.

## Verification

Exercise the optics fitter with known projected checkerboards and repeated
poses. Run the browser wizard tests and the Rust ball-center regression tests.
Verify that the deployed gateway issues still tokens and rejects invalid fits.
