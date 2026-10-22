# ADR-0077: Use a bounded remote MJPEG source for local Rust qualification

- Status: accepted
- Date: 2026-10-22
- Deciders: Project maintainers
- Supersedes: None
- Superseded by: None

## Context

The first migration qualification must run on the development Mac before any
target-machine changes. The target webcam is not a Mac-local device, but the
target currently exposes its camera feed as an HTTP multipart MJPEG stream.
The live Rust capture path only accepted a physical GStreamer camera. Treating
the URL as a camera identifier would incorrectly claim physical-device
identity, camera controls, or exposure timestamps. Buffering the HTTP stream
without a latest-frame policy would also let the detector drift behind the
current table state.

The existing capture-settings authority binds backend, geometry, cadence,
image space, revisions, and source identity into a request digest. The capture
runtime already isolates capture in a supervised process and its parent keeps
only the newest frame. The remote input must reuse those contracts without
changing the physical-camera path.

## Decision

Add an explicitly selected `gstreamer-http-mjpeg` source for local
qualification. It uses the existing Rust live detector and a supervised Rust
GStreamer helper. The source pipeline decodes HTTP multipart JPEG, applies the
configured output geometry and maximum frame rate, and keeps only one pending
appsink sample. The capture runtime's existing capacity-one newest-frame
handoff remains the only frame queue beyond GStreamer.

The remote source receives the stable source identity
`remote-http-mjpeg-stream`; it does not claim a physical camera selector or
camera-control support. The source URL is validated as credential-free HTTP(S)
and contributes to the camera-profile digest, but is not copied into the
capture-settings request. The helper uses local GStreamer arrival timing to
pace `videorate`; it reports no source timestamp, and never labels that timing
as camera exposure. Host read-completion time and unverified GStreamer PTS stay
separately identified.

The source is a qualification and development input, not a replacement for the
normal physical-camera source. It must be explicitly selected and supplied an
URL. It is not enabled by default.

## Consequences

### Positive

- The real Rust capture, inference, tracking, and publication path can be
  qualified on the Mac with the target's live camera feed.
- Both GStreamer appsink and the capture-runtime handoff are bounded to the
  newest work, so a slow detector does not accumulate a historical queue.
- Source identity, configuration revision, and profile digest remain
  correlated end to end without persisting a URL in the capture request.
- Remote arrival timing is explicit and cannot be mistaken for exposure time.

### Negative

- HTTP-MJPEG is a test/development source with network and server behavior that
  differs from a local webcam; it does not qualify physical camera controls,
  driver timing, USB behavior, or target-platform performance.
- GStreamer needs the Soup, multipart, JPEG, app, video-conversion, and scaling
  plugins available on each host.
- Source read-completion latency includes network delivery and must not be
  compared with sensor exposure latency as though the clocks shared the same
  semantics.

## Alternatives considered

### Pretend the URL is a physical camera

Rejected because it conflates a network source with device discovery, control
readback, stable physical selectors, and exposure-time provenance.

### Download the stream and replay a saved file

Rejected for the first integration check because a file does not exercise
live-source backpressure, network timing, reconnect behavior, or the newest
frame policy. Recorded replay remains useful for repeatable detector
comparisons.

### Let every consumer independently connect to the source

Rejected as the capture contract. The live detector uses one supervised
producer and bounded latest-frame handoffs; additional viewer connections
would create load and could observe a different frame than the detector.

## Verification

- Capture-settings tests verify URL-derived profile identity, URL omission from
  the settings request, and rejection of physical selector/control mixes.
- Live CLI tests verify explicit URL selection and reject credential-bearing
  URLs.
- The helper accepts exactly five decoded 1280x720 RGB8 samples from the live
  target MJPEG stream; each emitted sample has the configured identity and no
  source exposure timestamp.
- A real Mac smoke run using the Rust GStreamer helper, Rust detector, and Rust
  gateway published observed ball state with the freshness status `fresh` and
  sampled publication age around 100–180 ms. This does not establish ball
  precision/recall or table-relative position error; table calibration and
  temporal ground truth remain separate acceptance gates.
- Run the qualification again with the physical camera attached before
  target deployment. Compare the same model and capture settings, and record
  physical-source and calibrated-position accuracy independently from this
  network-source smoke.

## Related records

- [ADR-0036](0036-contain-native-camera-capture-in-a-supervised-helper-process.md)
  defines the supervised capture boundary reused by this source.
- [ADR-0040](0040-version-stable-camera-capture-selectors.md) defines identity
  requirements for physical cameras; this decision deliberately avoids
  claiming that identity for an HTTP stream.
- [ADR-0052](0052-latest-state-best-effort-freshness.md) defines the
  latest-state freshness contract.
- [ADR-0066](0066-run-server-services-in-rust-and-retain-the-standard-web-client.md)
  keeps the browser client standard web technology and requires server-side
  runtime ownership in Rust.
