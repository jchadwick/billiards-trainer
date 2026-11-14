# Product reset: a helpful, responsive pool coach

Status: user-approved target, updated 2026-11-13; implementation pending.

## Goal and authority

Build a fun, responsive pool coach that remembers the table, updates what changes,
shows useful aiming feedback, and helps the player choose and play shots. Improve
precision where it improves advice. Do not freeze video, hide useful ball markers
because frame IDs are unavailable, or erase stationary knowledge because full
inference is slow.

This target takes precedence over conflicting ADRs, specifications, defaults,
contracts, and tests. Updating or superseding them is authorized implementation
work without another approval step. Preserve historical rationale and useful
provenance protections. Documentation accompanies working changes; it is not an
additional qualification or permission gate.

## Lean behavior: two ball states and whole-table motion

Perception owns the working table model. Downstream code renders it and advances
freshness during silence; it must not maintain another ball-layout cache.

- **Trusted:** a usable settled scan plus uninterrupted compatible quiet evidence.
  Draw a solid marker; the position can support coaching.
- **Stale/unverified:** motion, occlusion, missing continuity, or expired evidence.
  Draw a dim/dashed last-known marker; never present its corridors as clear.

These are the only product ball states. Removal is absence from the latest clean
settled scan, not another state or a multi-scan removal policy. A clean scan is a
successfully completed detector evaluation of the visible calibrated table with
valid geometry, matching source/calibration context, and no motion/occlusion or
continuity gap since its capture. A failed, occluded, or incomplete evaluation must
not masquerade as a successful empty scan. A successful clean empty scan is valid.
Replace the active layout wholesale on each clean settled scan, retaining track
IDs only for unambiguous associations. Do not infer pocket events or numbered
identity from absence.

Use one whole-table **settled/moving** decision. Occlusion or unavailable evidence
makes the layout stale; it cannot establish settled trust. No regional motion or
regional invalidation is in scope. A shot invalidates the whole layout. Quietness
after motion cannot make old coordinates trusted again; a new clean scan must.

Keep the original measurement timestamp/last_seen_at and a trusted/stale flag.
Reuse existing table-level motion evidence and motion-generation tagging; do not
add per-ball continuity clocks or a historical reconstruction engine. Age or
republication alone never makes a measurement new. Keep finite configurable
continuity limits so stopped motion checks or a lost feed downgrade trust even
when WebSocket messages continue.

Allow slow completed scans to land when their context/order is valid and quiet
continuity has held since capture. Detection runtime alone must not reject them.
A scan spanning movement or a continuity gap cannot restore pre-shot positions.
Keep the last-known layout stale until a clean replacement arrives. A source,
calibration, or geometry change stops incompatible coordinates from overlaying
new imagery; reacquire instead of reusing them. Keep bounded in-flight work and
periodic full scans to discover changes cheap motion checks miss.

Audit tracker expiry, public-state lifecycle, gateway retention, and browser
caching so they implement this single policy. Extend the existing state contract
only as needed; no new cache, service, or transport. Keep configurable defaults
consistent with the intended low full-scan cadence.

## One production marker and an ephemeral cue

Use one ball-marker representation: table-plane center, fixed physical radius,
track ID, and trusted/stale flag. Project centers/radii to the display. Use
compatible detector-center fallback when precise refinement is unavailable, with
honest estimated status; never manufacture a measured contour. Missing precise
calibration certification must not block approximate camera markers when a usable
mapping exists. Explain missing geometry at the point of use.

Contours are optional developer diagnostics behind an explicit developer flag,
not a second production renderer or a prerequisite for markers. Exact-frame
ownership still applies when inspecting an actual source contour. Lack of video
frame IDs alone must not block production center markers. Always keep the newest
available live video; never rewind/freeze it for overlay pairing.

Cue output is ephemeral: each completed current cue evaluation supplies a line or
no line. A negative result clears the line immediately. Use only a small configured
maximum age/debounce with a default to clear silence/late results; do not store a
remembered cue, add cue lifecycle states, or replay an old line. Capture time, not
arrival time, determines whether an asynchronous cue result can still be shown.
A late result spanning movement/context change cannot reintroduce old aim. Missing
frame IDs do not imply exact-frame pairing. The desired aim can remain visible
without a detected cue; label it separately from the live aim comparison.

## Implement the useful coach first

Start in `frontend/web` with a complete practice sandbox using static editable
or click-to-place table-space balls. These are explicit simulation inputs, never
mock data silently substituted for live detections. This lets the user evaluate
pockets, cut angles, blocker corridors, ghost-ball/contact aiming, and usefulness
before Rust continuity work. Reuse the same pure geometry and view for live input.

1. Player confirms cue ball, selects target/pocket, or requests a suggestion.
   Automated number/stripe/solid recognition is not required.
2. Generate direct-pot candidates for known/configured pockets. Check clearance
   for cue-to-contact and object-to-pocket travel, including ball radii. Stale balls
   remain possible blockers and cannot justify a clear path. If the live table is
   stale, suspend live recommendations until its next clean scan.
3. Rank feasible candidates by configurable policy with defaults: smaller cut
   angle, then shorter travel, then stable target/pocket identity. Show selected
   target/pocket, contact/ghost-ball point, desired cue-ball path, and estimated
   initial object-ball path. Explain when no supported direct pot exists.
4. Compute in calibrated table-plane coordinates, then project into video. Reuse
   usable existing mapping and table/ball dimensions; never apply physical-circle
   geometry to perspective-distorted image distances. Missing live mapping/pocket
   data disables dependent live advice, not the sandbox or usable camera markers.
5. The suggestion and live cue comparison are distinct. A shot immediately stops
   old advice at the next motion decision. After settled reacquisition, recompute
   without repeating setup; keep selection only when association is unambiguous.

This is practice aiming guidance, not a claim to implement legal-game rules,
spin, banks, combinations, or complete post-contact physics. Those capabilities
must not block the first useful loop.

## Execution order and ownership

1. **Frontend practice coaching (`frontend/web/`):** extend the existing pure
   `shot-prediction.ts` geometry and add the editable practice view. Verify useful
   direct-pot guidance with static layouts before live integration.
2. **Perception continuity (`perception/`):** implement trusted/stale whole-table
   knowledge, clean-scan replacement, useful slow-result admission, ephemeral cue,
   and consistent lifecycle/gateway publication. Reuse existing motion generation
   and state path. Test before exposing the changed contract to the frontend.
3. **Live wiring (`frontend/web/`):** consume the producer state directly, use one
   production center renderer with solid/ghost styles, and connect the same
   coaching geometry to live table coordinates and manual identity selection.
   Remove redundant browser retention and exact-frame production marker gates.
4. **Integration and documentation:** verify the complete loop and update ADRs,
   specifications, API/configuration docs, and README to match actual behavior.
   Modify one top-level module at a time, in separate executor tasks. Document
   consequential decisions before/alongside code, without delaying the sandbox.

ADR-0100's blanket exact-frame overlay requirement must be superseded. Narrow
ADR-0103 to developer contour provenance while preserving unrelated localization
work. Update ADR-0099 and phase-aware expiry requirements wherever they force
precision/reference qualification before useful approximate markers or coaching.
Keep ordinary compatible geometry and source-change protection.

## Verification: behavioral assertions, not perfection gates

Use fake clocks, controlled worker completion, and configured deadlines.

| Scenario | Required result |
|---|---|
| Editable practice layout | Player can select/place balls and obtain target/pocket/contact guidance without camera or server; the view is explicitly simulated. |
| Geometry | Reject blocked cue/object paths, account for ball radii, use stable ranking, and report no direct pot when none is supported. Equivalent table layouts under different image perspective yield the same recommendation. |
| Slow scan with uninterrupted quiet | Results reach the working layout despite former detection-only deadlines; solid markers survive multiple skipped scan opportunities while quiet evidence remains current; last_seen_at stays honest. |
| Motion during blocked scan | All balls become stale; old advice disappears; late completion cannot restore pre-shot trust. Later quietness alone cannot restore trust. |
| Clean settled replacement | Active balls exactly match the new clean scan, including disappearance or an empty table. A failed/occluded scan cannot clear the layout. |
| Motion checks stop but publications continue | Trust expires at the configured table-evidence deadline; repeated messages do not extend it or clear remembered markers. |
| Cue positive, negative, late, or silent | Current positive draws, negative immediately clears, over-age/context-obsolete output cannot draw, silence clears at configured debounce; balls persist. |
| Direct video without frame IDs | Solid/ghost center markers draw with compatible geometry, video stays live, and normal UI does not depend on contour pairing. |
| Source/calibration changes | Incompatible old positions never overlay new imagery. |
| Live practice loop | Confirm cue/target, see guidance, shoot, see stale ghosts/no old advice, reacquire, then receive a fresh suggestion without setup reset. |

Start with existing detector-worker, public-state lifecycle, tracking,
LiveViewFreshness, and shot-prediction tests. Revise old assertions that encode
unconditional expiry/exact-frame suppression, preserving no-rejuvenation and
context-reset coverage. Run relevant suites during each slice, then `make test`,
`make lint`, `make format-check`, `make config-check`, camera-feature tests for
capture changes, and the frontend production build. Resolve failures before
marking each slice complete. Review every executor result and commit its task
changes while preserving unrelated pre-existing work.

Finally demonstrate on the actual camera: static table through slow scans, moving
cue, shot/settlement/reacquisition, brief occlusion, and feed loss. Record observed
responsiveness and configuration; tune only where actual play exposes a problem.
No new qualification program, regional motion, automatic identity, precision
calibration project, or diagnostic infrastructure is required to finish this loop.
Implementation/replay/hardware results must be reported honestly; this target
alone is not evidence that any of these checks have passed.
