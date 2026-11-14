# Phase-aware perception

This guide documents the implemented behavior accepted in
[ADR-0114](architecture/decisions/0114-use-trusted-and-stale-whole-table-perception-state.md).
Perception owns the working ball layout; consumers render and passively age
that state. The live video remains the newest available video, independent of
perception timing.

## Runtime structure

The coordinator owns phase selection, motion assessment, cue evaluation,
tracking, calibration, and publication. A dedicated worker owns inference.
Keep detector work bounded: while a call is in flight, skip the scheduled
opportunity rather than queueing frames or catching up in bursts. Tag work and
results with source, configuration, calibration/geometry context, ordering, and
existing motion generation so incompatible or pre-motion results cannot restore
trust. Capture and analysis rates remain independently configurable; use the
versioned configuration defaults rather than introducing production rate or
freshness constants in this specification.

Use the existing whole-table motion signal to choose settled or moving. Motion
makes the whole layout stale. Occlusion, missing frames/evidence, or unavailable
whole-table status also makes it stale and cannot count as settled. Quietness
after motion does not restore trust: a new clean scan is required. Do not infer
regional certainty, moving-ball trajectories, pocket events, or numbered-ball
identity.

## Layout lifecycle

A successful clean settled scan evaluates the visible calibrated table with
valid geometry, compatible source/calibration context, and uninterrupted quiet
evidence since capture. It replaces the active layout wholesale. Balls absent
from that scan are removed; a clean scan with no balls is a valid empty layout.
Failed, occluded, incomplete, or context-invalid work is not an empty scan and
cannot clear the previous layout. Preserve IDs only for unambiguous
associations.

Completion time alone does not make a scan unusable. A slow result may be
accepted if its ordering and context still match and quiet evidence has been
continuous since capture. A motion generation change, evidence gap, or context
change rejects it from restoring pre-shot coordinates as trusted. Keep the last
known layout stale until a clean replacement arrives. Preserve measurement
capture time/`last_seen_at`; publication and repeated messages do not refresh it.
Finite configurable freshness limits downgrade trust when motion checks or the
feed stop, even if other state messages continue. Consumers do not keep another
layout cache or extend producer freshness.

Scan completion has no detection-only deadline for an otherwise current
result. The public envelope still carries `layout_verification_max_age_ms`, but
the legacy layout scan-expiry value does not gate trust. Trust expires from
whole-table motion evidence instead. Legacy configuration values for consensus
counts and tolerances are also accepted for compatibility without gating trust
or replacement. One clean settled scan is sufficient, and a clean empty scan
replaces the prior layout with an empty one. Failed, incomplete, or occluded
work cannot masquerade as an empty result.

The production marker is a table-plane center, fixed physical radius, stable
track ID, and trusted/stale flag. Project its center and radius through
compatible geometry. A usable detector center may be used as an explicitly
approximate fallback when precision refinement is unavailable; do not label it
as a measured contour. Missing precise calibration certification does not by
itself block a marker when a usable mapping exists. Missing usable mapping or
other dependent geometry disables the affected projection or live advice at
the point of use. Optional contours are developer-only provenance protected by
an explicit developer flag and exact source-frame ownership. Missing video
frame IDs do not block center markers, and the browser never freezes or rewinds
video to pair an overlay.

The active `observation_status.perception.table_projection` is the actual mapping used for
production world coordinates. Its `pixel_to_table_mm` homography maps source
image pixels to millimeters relative to the playing-surface center, with table
dimensions and camera/source/calibration context. The browser inverts that same
mapping for projected advice paths. Compatible image centers remain a fallback
for marker display when world mapping is unavailable; advice requiring physical
coordinates pauses. A newer configuration revision on the same selected source
remains compatible, while a producer revision older than the source-selection
revision floor cannot satisfy the replaced feed. A source-generation change
invalidates the previous display context. Known engine, session, configuration,
calibration, rectification, and geometry mismatches still suppress incompatible
spatial output; a frame-ID difference alone does not.

## Cue lifecycle and phase behavior

Each completed current cue evaluation supplies either a line or no line. A
negative result clears the current line immediately. Capture time determines
whether an asynchronous result is still current; context-obsolete or
over-age results cannot reintroduce it. Motion clears the current cue line; a
result from an earlier motion generation cannot restore it, even within the age
limit. Reuse one configured capture-age bound for silence/debounce rather than
adding separate lifecycle timers. Cue state is ephemeral: do not remember an old
line or require exact video-frame pairing. Desired practice aim is separate
from the live cue comparison. Ball layout freshness is independent of cue
updates. The current default cue evidence expiry is 250 ms. A positive cue line
without signed direction may remain visible as a line, but cannot drive a
directional cue path. Manual desired-aim guidance remains available independently
of a detected cue.

Phases may schedule lightweight motion checks, full-layout scans, and cue work
at configured rates. A shot stops old advice at the motion decision. Recovery
continues motion checks and schedules clean scans until current quiet evidence
and a successful clean scan establish a trusted replacement. Retain bounded
workers and skip missed slots; rates are operational settings, not universal
FPS qualification gates.

## Deterministic behavior and verification

Implementation must cover these outcomes with fake clocks and controlled
worker completion rather than relying on timing luck:

| Condition | Required behavior |
|---|---|
| Slow scan completes while compatible quiet evidence remained continuous | Admit it if order and context still match; preserve its original capture/last-seen time. |
| Motion or a continuity gap occurs while a scan is blocked | Mark the whole layout stale; reject the result from restoring pre-motion trust. Later quietness alone does not restore trust. |
| Clean settled scan omits balls or is validly empty | Replace the active layout exactly with its contents, including an empty layout. |
| Scan fails, is occluded, incomplete, or context-invalid | Do not clear the active layout; keep last-known positions stale as applicable. |
| Motion evidence expires or feed stops while other state messages continue | Downgrade trust at the configured evidence deadline; do not rejuvenate measurements. |
| Cue evaluation is positive, negative, silent, late, or context-obsolete | Draw only current positive output; clear immediately on negative and expire/reject silent or obsolete output by configured capture-age/debounce rules. |
| Video has no frame IDs but usable center geometry exists | Draw trusted or stale center markers while displaying newest video; do not require contour pairing. |
| Source, calibration, or geometry context changes | Do not project incompatible old coordinates onto new imagery; reacquire. |

Existing useful checks remain: valid geometry and context matching, bounded
worker behavior, honest capture and evidence timestamps, no trust rejuvenation,
no old cue/current-truth presentation, and clean shutdown. Precision contour
qualification, fixed detector FPS, and universal appearance-reference
qualification are not prerequisites for this target. Accuracy and timing should
be reported honestly when measured, but missing a perfection threshold is not
a product blocker for approximate markers or the first useful coaching loop.

## Implementation status

The trusted/stale whole-table lifecycle, clean replacement, slow-result
admission, ephemeral cue handling, compatible center rendering, and direct-pot
practice/live coaching are implemented. Automated frontend, Rust, configuration,
and tooling checks have been run. The actual billiards-table camera and play
acceptance has not been performed; no physical aiming-accuracy, camera-to-display
latency, or deployment-performance qualification is claimed. Precision metrics
and developer diagnostics remain useful inspection tools, not prerequisites for
the approximate markers or direct-pot estimates.
