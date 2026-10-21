# ADR-0019: Asynchronous ball identity separate from location tracks

- Status: proposed
- Date: 2026-10-19
- Deciders: Project maintainers
- Supersedes: None
- Superseded by: None

## Context

The audited model detects generic balls and cue; it does not identify individual
numbers. Optional color/stripe heuristics are unvalidated identity guesses.
The current adapter discards most numbered classes, and some clients infer
number from tracking ID. Introducing identity into association without
uncertainty can fragment tracks or delay the primary location output.

## Decision

Treat numbered-ball identity as an asynchronous evidence consumer separate from
localization. Publish stable track/session IDs and fresh positions immediately
with nullable number and independent identity confidence. Tracking ID never
implies ball number; detection confidence is not identity confidence.

Reuse Python training, annotation and model export workflows. Compare a
high-resolution crop classifier with joint numbered-ball detection through
the same acceptance harness. Accumulate clear-view color/stripe/numeral evidence
over time, allow unknown, and calibrate confidence rather than forcing an answer.
Table inventory constraints are soft evidence, not permission to invent labels.

Identity updates refer to the same live track/session and retained observation
provenance. Reject stale results after track expiry, restart or incompatible
revision changes. Updating identity advances annotation/snapshot revision, not
the time at which a position was measured. Optional identity work has its own
resource budget and must not block localization.

Retain existing generic-ball model as the localization baseline. Reliable
anonymous localization may qualify before numbered identity; expose feature
capabilities explicitly rather than advertising generic detection as recognition.

## Consequences

### Positive

- Primary locations remain available while recognition is uncertain.
- Reuses trained model and ML-tooling investment.
- Makes confident wrong labels and unknown rates measurable.
- Preserves track continuity independently of changing identity evidence.

### Negative

- Reliable numbers require new labeled evidence and likely a classifier or
  separately trained model; language migration alone does not add the feature.
- Temporal evidence and association need careful reset/reacquisition behavior.
- Some numerals are invisible from one camera, so unknown is unavoidable.
- UI/client adapters must preserve explicit semantic fields consistently.

## Alternatives considered

### Treat track IDs as numbers or force color-based labels

IDs reflect runtime association, not physical markings. Color/stripe appearance
varies with lighting, spin and ball sets; forced labels hide uncertainty.

### Wait for identity before publishing position

This directly violates the primary requirement and increases first-detection
latency even when a ball center is clear.

### Replace localization with a multiclass model immediately

A joint model is a valid experiment, but changes localization/identity together
and still requires adapters that preserve labels and uncertainty.

## Verification

- Shared fixtures include track 42 identified as 8, track 7 with unknown identity,
  arbitrary cue-ball IDs and reclassification without track changes.
- Held-out ball sets/lighting/spin/occlusion tests report class confusion,
  confident-error/unknown rates and time to identify.
- Crossing/collision/reacquisition tests report track identity switches separately
  from number recognition mistakes.
- Delayed identity results cannot annotate expired or restarted tracks.
- Identity-enabled workload runs preserve localization accuracy and freshness.

## Revisit conditions and related records

Adopt a joint model only after full-path localization and identity evidence
supports it. Revisit camera coverage if a specified recognition requirement
cannot be met with available views. ADR-0010 defines annotation provenance;
ADR-0016 defines optional resource isolation; ADR-0018 defines feature qualification.


## Evidence and implementation tracking

Status is proposed; Rust migration implementation and qualification are planned,
not completed. This record governs migration stages 7.

- [Migration plan](../../../thoughts/rust-perception-migration-plan.md).
- [Architectural review and evidence limitations](../../../thoughts/architectural-review-2026-10-19.md).
- [Existing backend/vision/detection/detector_adapter.py](../../../backend/vision/detection/detector_adapter.py).
- [Existing frontend/web/src/stores/VideoStore.ts](../../../frontend/web/src/stores/VideoStore.ts).

Before accepting this decision for implementation, resolve its relevant prototype
and compatibility questions and record the evidence. Acceptance records commitment;
it does not claim all verification has passed. Link implementing changes and actual
verification artifacts as they exist. Follow the [decision register](../README.md)
for status changes and supersession.
