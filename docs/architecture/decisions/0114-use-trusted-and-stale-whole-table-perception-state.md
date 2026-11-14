# ADR-0114: Use trusted and stale whole-table perception state

- Status: accepted
- Date: 2026-11-13
- Deciders: Project maintainers
- Supersedes: ADR-0099 (mandatory appearance-reference qualification only), ADR-0100
- Superseded by: None

## Context

The current product rules can discard useful stationary knowledge when full
detection is slow, require an exact video-frame join before drawing any marker,
and retain cue output after a negative or obsolete evaluation. They also require
multiple successful scans before replacing a layout, despite the product target
being a responsive coach. The project needs a single owner and a small, testable
state model that tolerates slow work without resurrecting pre-shot positions.

ADR-0099's concerns about table geometry, motion, occlusion, and source or
calibration changes remain valid. A mandatory stored appearance reference is
not a universal prerequisite for useful markers or guidance when calibrated
geometry and current compatible evidence are available. ADR-0103's native
localization, calibration, and measurement-uncertainty decisions remain in
force; only production display of fitted contours is narrowed here.

## Decision

Perception is the sole owner of the working ball layout. The product exposes
only trusted and stale/unverified ball knowledge. Trusted means a usable scan
of the calibrated visible table and uninterrupted compatible quiet evidence.
Motion, occlusion, unavailable evidence, incompatible context, or expired
table-level evidence makes the whole layout stale. Unknown or occluded is never
settled. There is no regional invalidation or per-ball continuity state engine.
Use existing table-motion evidence and motion-generation tagging with finite,
configurable freshness limits; silence downgrades trust even if state messages
continue. Preserve original `last_seen_at`/capture time. Consumers passively
age or downgrade the producer state and do not cache a second layout.

Every successful clean settled scan replaces the active layout wholesale,
including removing absent balls or representing a valid empty table. A failed,
occluded, or incomplete evaluation cannot clear it. Track IDs survive only
unambiguous associations; absence does not imply pocket events or numbered
identity. A completed slow scan is admissible when its context and ordering
match and uninterrupted quiet evidence has held since capture. Any motion
generation change or continuity gap prevents old coordinates from restoring
trust. Keep inference work bounded and retain periodic full scans to discover
changes that cheap motion checks miss.

Production uses one marker representation: table-plane center, fixed physical
radius, stable track ID, and trusted/stale flag. Project center and radius to
the display. A compatible coarse detector-center fallback may draw an honestly
approximate marker when precise refinement or certification is unavailable;
never manufacture a measured contour. Contours are optional developer-only
provenance behind an explicit developer flag and remain tied to their exact
source frame. Missing video frame IDs alone do not block center markers, and
the newest available video is always shown without rewinding or freezing.

Cue output is ephemeral: each current evaluation returns a line or no line, and
a negative result clears it immediately. Motion clears the current cue line;
a result from an earlier motion generation cannot restore it, even within the
age limit. Reuse one configured capture-age bound for silence/debounce rather
than adding separate lifecycle timers. Reject late results using capture time
plus matching context. Do not retain a remembered cue or require exact video-frame pairing.
Desired practice aim remains distinct from live cue comparison.

## Consequences

### Positive

- Slow scans can preserve stationary knowledge without changing its measurement
  time, while motion and missing continuity reliably make it stale.
- Clean replacements, valid empty tables, and immediate cue clearing have
  unambiguous behavior.
- Approximate center markers remain useful with independent video and state
  streams, while exact-frame contour provenance remains available for debugging.
- One perception-owned layout avoids conflicting browser or gateway caches.

### Negative

- Whole-table invalidation sacrifices potentially useful regional certainty.
- Approximate centers can be less precise than certified contour refinements
  and must be presented honestly.
- A stale layout remains visible as last-known information until a clean scan
  replaces it, so the UI must distinguish it from trusted coordinates.

## Alternatives considered

### Require exact-frame pairing and certified contours for all markers

This suppresses useful markers when the video transport has no frame IDs or
precision refinement is unavailable, even when compatible table geometry and
center detections exist. Exact source ownership remains required for optional
contour diagnostics.

### Keep the previous multi-scan consensus and clear on scan failure

Consensus delays replacement and failure-as-empty can erase valid knowledge.
The accepted rule instead distinguishes a successful clean empty scan from an
unsuccessful or incomplete evaluation.

### Keep browser-side layout and cue memories

Additional caches can extend stale coordinates or cue lines past producer
invalidation. Consumers instead age the single producer state and cue results
expire by capture time.

## Verification

- Perception tests with controlled clocks and worker completion cover slow
  completion during uninterrupted quiet, motion-generation changes, gaps,
  clean wholesale replacement, valid empty scans, and failed/occluded scans.
- State lifecycle tests cover finite evidence expiry, preserved measurement
  timestamps, no rejuvenation, and whole-table stale behavior on motion or
  occlusion.
- Frontend tests cover solid/stale center markers without frame IDs, live-video
  continuity, and no second layout cache.
- Cue tests cover positive, immediate negative clearing, silence expiry, and
  capture-time rejection of late or context-obsolete results.
- Contract and configuration checks verify compatible geometry/context
  protection and configurable freshness limits. These tests are implementation
  requirements; this accepted ADR does not claim they have passed.
