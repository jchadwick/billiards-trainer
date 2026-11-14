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
presented as detections. The same pure API is intended for later use with trusted
live table-plane inputs.
