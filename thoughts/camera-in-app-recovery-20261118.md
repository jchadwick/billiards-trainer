# Camera in-app recovery — 2026-11-18

## Finding

The deployed frontend is requesting the camera URL supplied by the target's
runtime configuration. The target returns
`api.stream.viewer_url = http://127.0.0.1:18001/stream`. That URL resolves on
the browser device, not on the target gateway. No service listens on port
18001, so a user's browser gets `net::ERR_CONNECTION_REFUSED` and the visible
camera image has `naturalWidth = 0`. The app correctly changes its header to
“Video error”; the preview remains a black image area.

The camera itself is healthy and directly reachable on the LAN at
`http://192.168.1.31:8001/stream`. Target process 1651 (`ustreamer`) listens on
`0.0.0.0:8001`; a client request returns HTTP 200 with
`multipart/x-mixed-replace`. The application server (PID 98364) serves HTTP on
port 80. The target has no listener on port 18001 and its own request to
`127.0.0.1:18001/stream` is refused. The app's camera URL is not proxied by its
API; the browser uses `api.stream.viewer_url` directly, as documented in
`docs/API_REFERENCE.md`.

## Narrow target configuration correction

For clients on the verified LAN, change only
`api.stream.viewer_url` in
`/home/jchadwick/apps/billiards-trainer/config.json` from
`http://127.0.0.1:18001/stream` to
`http://192.168.1.31:8001/stream`. If users open the app through another
reachable target hostname, configure that browser-reachable hostname instead.
This worker did not write target configuration or deploy anything.

## Browser evidence

All checks used the actual app origin `http://192.168.1.31/live`, with no SSH
port forwarding. The configuration API returned the loopback URL. Chromium
made two attempts to `http://127.0.0.1:18001/stream`; both failed with
`ERR_CONNECTION_REFUSED`. The app rendered `Video error`, a 1030×579 preview,
and a zero-sized image. There were no uncaught page exceptions.

A Playwright-only response override changed the returned config URL (without
changing the target) to `http://192.168.1.31:8001/stream`. The same deployed
page displayed a 3840×2160 frame in the same 1030×579 diagnostics preview. The
browser received HTTP 200 `multipart/x-mixed-replace`; CDP recorded 11.1 MB and
then 24.5 MB during adjacent sample windows, and screenshots of the camera
image differed. `/live → /practice → /live` retained one image and one stream
response, with no failed requests, page errors, or console errors. Diagnostics
still had no coaching controls.

Screenshots:

- Broken deployed configuration: `/tmp/billiards-camera-app-live-before-config-fix.png`
- Browser-only corrected configuration: `/tmp/billiards-camera-app-live-config-override.png`

The image visibly contains pool balls, while the current producer state still
reports phase `recovering` and “Ball observations 0.” Camera transport is
verified; ball detection remains the independent perception task.

## Frontend and verification

No frontend source file was changed. Relevant pre-edit file copies and hashes
are in `/tmp/billiards-camera-recovery-baseline/`; all copied frontend files
still match those baselines. The frontend accepts the configured HTTP(S) URL
and connects directly, so rewriting loopback values in production frontend
logic would mask this deployment setting instead of correcting it.

- `npm run typecheck`: passed.
- `npm test`: 30 files and 138 tests passed.
- `npm run build -- --outDir /tmp/billiards-camera-recovery-build`: passed.
- The fresh build's app, LiveView, and CSS asset hashes match the assets served
  by the deployed app.
- No configuration write, deployment, service restart, perception edit, or
  commit was made.
