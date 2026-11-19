# ADR-0108: Own camera setup in the local gateway

- Status: accepted
- Date: 2026-11-02
- Deciders: Project maintainers
- Supersedes: None
- Superseded by: None

## Context

The configuration authority already provides versioned defaults and atomic
revisioned writes. Requiring a capture source to start the gateway made those
facilities inaccessible during first setup. Model deployment, browser preview,
and calibration each required additional operator knowledge.

## Decision

The gateway serves first-run setup before starting capture. It saves the
user-selected camera through the configuration authority and restarts its
supervised producer. Existing explicit sources start automatically. The
browser transport is specified in ADR-0109. Calibration uses the
producer's pinned-frame and revision checks and persists in the existing
capture-bound store. Packages provide verified model assets independently of
source selection. General overrides keep their existing configuration API.

## Consequences

### Positive

- New installations can start without an operator-authored file.
- Camera choice and table calibration use the same capture identity as guidance.

### Negative

- Applying a camera change or table calibration briefly interrupts capture.
- Source discovery and calibration remain local gateway operations.

## Alternatives considered

Automatically opening an arbitrary detected camera would reduce clicks but can
select a camera unrelated to the table. A separate setup executable would add
a second configuration and calibration workflow.

## Verification

Run the Rust gateway and configuration suites, browser setup tests, and
distribution smoke checks. Verify that a source-free install serves setup and
that calibration survives restart only for compatible capture geometry.
