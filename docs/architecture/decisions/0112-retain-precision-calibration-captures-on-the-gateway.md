# ADR-0112: Retain precision calibration captures on the gateway

- Status: accepted
- Date: 2026-11-13
- Deciders: Project maintainers
- Supersedes: None
- Superseded by: None

## Context

Checkerboard poses and measured ball placements take time to reproduce.
The setup wizards lose their input photos when the browser reloads, even
though the fitted calibration files can survive a restart.

## Decision

Keep explicitly saved calibration stills in the gateway's persistent
calibration data directory. Bind them to the camera and applicable geometry.
The wizard loads compatible captures when reopened and supports replacing
individual checkerboard poses and removing or clearing ball measurements.
Use a bounded set for each calibration workflow. Preserve the independent
physical measurements alongside the ball images. A ball proposal only
suggests an image position and still requires operator confirmation.

## Consequences

### Positive

- Operators can resume collection or rerun fitting across restarts.
- The saved stills show what the operator used to fit the calibration.
- The browser continues to connect directly to the live camera service.

### Negative

- Calibration storage now contains operator-selected images and must be
  carried forward when replacing a release directory.
- Changes that invalidate capture geometry require compatible captures.

## Alternatives considered

Browser storage would bind the evidence to one browser and could lose it
when local data is cleared. Keeping only the generated parameters would
still require repeating the physical procedure whenever a fit needs review.

## Verification

Run the gateway archive tests and the precision wizard tests. Check that
saved captures reload under the same camera binding and that an incompatible
binding cannot use those captures. Verify that reload restores the wizard
and that archived photos are only served as discrete calibration stills.
