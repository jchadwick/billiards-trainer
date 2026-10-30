# ADR-0105: Retire Coral as a supported production inference provider

- Status: accepted
- Date: 2026-10-29
- Deciders: Project maintainers
- Supersedes: None
- Superseded by: None

## Context

ADR-0102 initially kept detector inference on Coral because Intel GPU inference
was unqualified. ADR-0104 records a same-input target comparison that changes
that tradeoff: the OpenVINO Intel GPU path measured 69.68 ms inference p95,
compared with 1551.04 ms p95 for the existing Coral model. On the same saved
scene at the same confidence threshold, GPU inference produced proposals for
all 15 balls while Coral missed one. The compiled Coral graph assigned 127
operators to EdgeTPU and 132 to CPU. The evidence and scripts are preserved in
`thoughts/calibration/target-gpu-saved-20261029/` and
`thoughts/calibration/target-coral-saved-20261029/`.

This backend comparison establishes a substantial measured throughput and
proposal-availability advantage for the tested GPU path. It does not establish
physical X/Y accuracy, native-4K camera reliability, moving-scene performance,
or the production FPS. The separate 60-second static replay used an upscaled
720p saved image and is not accuracy or native-camera evidence.

## Decision

Support one production inference provider: the explicitly selected OpenVINO
Intel GPU path defined by ADR-0104. Retire Coral as an active production
provider: remove it from runtime, configuration, and packaging selection, and
do not retain an automatic or user-selectable Coral fallback. If the required
Intel GPU runtime or device is unavailable, startup must fail clearly rather
than silently selecting another inference backend.

Keep the prior deployed release/package and the Coral comparison artifacts
outside the active supported runtime as recoverable rollback and research
references. Restoring that whole prior release remains an operational rollback
action; it does not make Coral a current fallback or parallel supported
provider. Reconsider Coral or another backend only through a new measured
qualification against the selected GPU path, using the same model inputs and
saved detector corpus, reporting actual execution placement, end-to-end
availability, latency, and failure behavior. A new comparison must not claim
physical localization accuracy without independent X/Y validation.

## Consequences

### Positive

- The active runtime has one inference path and cannot silently move onto the
  substantially slower, partially CPU-executed Coral graph.
- Runtime configuration, packaging, and startup behavior have fewer backend
  combinations to qualify.

### Negative

- Intel GPU drivers, OpenVINO, and device access are required for the supported
  deployment; their absence is a startup failure.
- Coral-specific runtime code and installation/configuration support are no
  longer available for a quick in-process fallback. The saved package and
  comparison evidence remain separate for whole-release rollback and future
  evaluation.
- The measured comparison is a saved-input inference result, not proof of
  localization accuracy, continuous live availability, or a production camera
  rate.

## Alternatives considered

### Keep Coral selectable

Retain Coral as a selectable production provider. This preserves flexibility,
but requires maintaining, packaging, and qualifying a second runtime path
despite its measured 1551.04 ms p95 and one missed ball on the compared saved
scene. No evidence presently shows that the second path improves system
availability or accuracy.

### Automatically fall back when GPU inference is unavailable

This conceals missing GPU support and changes latency and proposal behavior
without an explicit operational decision. Report the GPU startup failure and
use the recoverable prior release if rollback is needed.

## Verification

- Configuration and CLI tests reject CPU, automatic, and Coral providers.
  The live composition API also rejects any provider except OpenVINO before it
  opens capture. The OpenVINO session explicitly requests the GPU device and
  propagates initialization errors without a fallback. The user-visible error
  for a missing GPU still requires target-device qualification.
- Configuration-schema and packaging checks prove Coral provider settings,
  launch arguments, worker/runtime dependencies, and Coral installation steps
  are absent from the supported build while the rollback package remains
  preserved outside it.
- `thoughts/calibration/target-gpu-saved-20261029/openvino-saved-report.json`
  and `thoughts/calibration/target-coral-saved-20261029/coral-saved-report.json`
  preserve the same-input timing and proposal comparison, model/runtime
  provenance, and raw-output references. Repeating that backend comparison is
  required before any future backend is considered for support.
- The target saved-image replay report in
  `thoughts/calibration/refiner-r10-validation-20261029/r10-scheduler-and-corner-availability.md`
  records its static upscaled-720p input and explicitly does not qualify
  physical accuracy or native-4K reliability.
