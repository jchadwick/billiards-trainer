# Billiards Trainer Live View

The browser application is intentionally limited to the live table at `/live`.
It remains a React, HTML, JavaScript, and CSS client. The Rust gateway serves
the built assets and the state WebSocket; the camera image is loaded directly
from the configured browser-reachable MJPEG URL.

## Development

```sh
npm ci
npm run dev
```

Vite serves the client on port 3000. Set `VITE_API_BASE_URL` when the API is on
a different origin. When unset, HTTP and WebSocket traffic use the origin that
serves the page. The live page requests `/api/v1/config` to read
`api.stream.viewer_url` and `perception.freshness.maximum_observation_age_seconds`.
The camera URL must be reachable by the browser. If it is absent or invalid,
the page reports that the direct stream is unavailable; it does not proxy video
through the API.

## Checks

```sh
npm test
npm run typecheck
npm run build
```

The production build writes assets to `../../backend/static`, as expected by the
Rust gateway. To build into a temporary directory instead, pass Vite an output
directory, for example `npm run build -- --outDir /tmp/billiards-web-build`.

The live client subscribes only to the latest `state` WebSocket stream. It
ignores out-of-order snapshots when the producer session provides ordering
identity, shows the most recent ball state in a fixed 16-slot layout, and marks
locations stale when their source observation ages past the configured limit.
Identity classification, trajectory prediction, calibration controls,
monitoring dashboards, and diagnostics are outside this browser surface.

## Practice coaching sandbox

The standalone `/practice` route is a simulated table-space sandbox. Players can
place balls with the pointer or physical-coordinate inputs, explicitly confirm a
cue ball, and ask for a direct-pot suggestion or select a target and pocket. Its
pure geometry uses millimeter table coordinates and configurable dimensions / ball
radius; offline defaults mirror the physical localization defaults. Suggestions
check both travel segments against ball radii and rank by cut angle, then total
travel, then stable target/pocket identity. `suggestDirectPots` accepts alternate
ranking priorities and a maximum cut angle. This estimates a direct pot only; it
does not model spin, banks, combinations, or full post-contact physics. The route
does not consume live camera or producer state, so simulated balls are never
presented as detections. Live coaching reuses the same pure geometry with the
producer's current trusted table-plane layout.

## Live coaching

Live coaching reuses the pure `suggestDirectPots` geometry from `/practice`.
Choose the cue ball, target, and pocket on `/live`, or leave target and pocket
on automatic suggestion. Advice requires a current trusted whole-table layout
and compatible producer table projection. Positions use the producer's
center-origin millimeters and are shifted to the sandbox's positive-origin
coordinates for coaching; paths are projected back through the inverse of the
same source-pixel-to-table homography. The desired ghost-ball cue path and
object path are shown separately from an ephemeral current cue line. If cue
direction is signed, a cyan current-aim travel estimate is shown; otherwise the
interface reports that direction as unknown. Stale ball centers remain visible
as possible blockers and never support an advice corridor. Missing compatible
projection disables live aiming but leaves usable image-coordinate centers
visible. A verified clean empty scan and an unscanned table have separate
messages.
