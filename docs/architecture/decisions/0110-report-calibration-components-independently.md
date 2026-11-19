# ADR-0110: Report calibration components independently

- Status: accepted
- Date: 2026-11-03
- Deciders: Project maintainers
- Supersedes: None
- Superseded by: None

## Context

Table geometry, camera optics, and physical ball-center correction are
separate inputs. A successful table calibration previously looked like the
whole guidance setup was complete.

## Decision

Setup shows table geometry separately and checks configured precision assets
against the same capture and table context as the producer. The checklist
distinguishes missing files, validation failures, blocked checks, and
compatible files. Operators can configure asset paths from Setup using the
existing configuration API. The table wizard continues to solve table geometry.
ADR-0111 defines the browser fitting workflow for physical reference captures and measurements.

## Consequences

### Positive

- Operators can see which calibration needs attention in one place.
- File compatibility uses the producer's validation rules.

### Negative

- Checking configured files adds bounded file reads to setup status requests.
- Fitting precision assets requires physical measurements by the operator.

## Alternatives considered

A single calibration flag concealed missing precision inputs. Checking only
whether paths were set could report incompatible assets as ready.

## Verification

Check that an active table and absent precision assets report distinct
statuses, and that a malformed optics file is rejected by the setup check.
