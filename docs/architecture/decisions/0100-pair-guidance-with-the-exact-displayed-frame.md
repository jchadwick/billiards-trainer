# ADR-0100: Pair guidance with the exact displayed frame

- Status: accepted
- Date: 2026-10-24
- Deciders: Project maintainers
- Supersedes: None
- Superseded by: None

## Context

The browser receives MJPEG video and state independently, so the pixels on
screen cannot be tied to the source frame evaluated by the perception runtime.
Buffering differences can misplace otherwise accurate overlays during motion.

## Decision

Carry acquisition session, frame, and geometry identity from the camera
streamer through perception state and gateway delivery. The browser may draw a
spatial overlay only when the decision references the exact displayed frame
and compatible geometry. Bound buffering and wait only for a configured
alignment interval; an unmatched frame is shown without an overlay. Never
choose a nearest-frame substitute.

## Consequences

### Positive

- Spatial guidance is tied to known pixels, including when video and state
  arrive at different rates.
- Missing or late state has an explicit safe display behavior.

### Negative

- Frame identity and bounded buffering become public protocol concerns.
- Legacy MJPEG without identities can be shown only without spatial overlays.

## Alternatives considered

### Continue independent MJPEG and WebSocket rendering

No reliable frame join can be made from those streams.

### Composite the overlay into video at the gateway

This couples display rendering to the gateway and prevents the selected browser
viewer from independently composing display choices.

## Verification

- Contract tests reject malformed, stale, cross-session, and incompatible-frame
  pairs.
- Browser tests cover delay, drops, freezes, geometry changes, and expiry.
- Target-device traces report decode and render times without comparing
  unrelated monotonic clock domains.
