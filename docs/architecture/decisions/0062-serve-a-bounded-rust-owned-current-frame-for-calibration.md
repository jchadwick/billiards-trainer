# ADR-0062: Serve a bounded Rust-owned current frame for calibration

- Status: superseded
- Date: 2026-10-22
- Deciders: Project maintainers
- Supersedes: None
- Superseded by: ADR-0063

## Context

Rust live capture is the sole camera owner when the Rust provider is selected.
The capture worker places RGB8 frames in a capacity-one latest-frame mailbox and
the synchronous perception callback drops each image after processing. The
retained Python preview and calibration wizard use the legacy Python camera and
shared-memory path, which is unavailable in Rust mode. The wizard also stretches
video into a fixed 640x360 canvas and stores canvas-relative clicks, so its
points do not identify source pixels or a particular Rust capture frame.

ADR-0059 and ADR-0060 establish Rust-owned, provenance-bound calibration
solving and storage. They deliberately do not provide the wizard with the exact
frame and active capture identity needed to create a calibration safely.
Calibration must use the same source image and geometry as perception, without
opening a second camera, retaining a history of full frames, or allowing slow
preview clients to hold camera buffers indefinitely.

## Decision

Add a Rust-owned, capacity-one latest-captured-frame slot. The capture producer
updates it when a valid frame is published, before inference, so a slow detector
cannot make the calibration preview trail the camera. The slot retains a shared
reference to the existing budget-accounted RGB8 pixels rather than copying the
full image. Existing aggregate frame-memory limits continue to account for the
underlying allocation; any frame still borrowed by inference remains charged
until that reader releases it.

Expose a one-shot preview request over a bounded loopback-only local channel
owned by the Rust live process. The Python service may proxy the request through
its existing HTTP route, but it does not read the camera, decode or transform the
image, infer state, or decide calibration validity. The Rust process encodes the
latest frame as JPEG on demand, limits concurrent encodes and response size from
validated configuration, and returns the JPEG together with a descriptor for
the same retained frame. It keeps no per-client frame queues. Empty, stale, or
closing capture state returns an explicit unavailable outcome and does not
reuse a previous session's image.

The descriptor is built from the active request and Rust's accepted capture
acknowledgement, never from caller-supplied values. It includes engine and
capture-session identities, frame identity, source dimensions and pixel format,
image space, camera-profile identity, configuration and rectification
revisions, and source timestamp semantics. Timestamp stage and clock domain
remain explicit; a read-completion timestamp must not be labeled as exposure.
The preview response couples the descriptor to the exact JPEG bytes so the
browser cannot accidentally attach clicks to a later frame.

The wizard maps pointer positions through the actual rendered image rectangle
back to source-image pixels and submits the returned descriptor with its
ordered corners and explicitly measured table dimensions. Rust validates the
descriptor against the currently active capture session and latest retained
frame, then delegates solving and persistence to the existing calibration
solver/store. A stale frame, changed profile or revision, out-of-bounds point,
or unmeasured table size is rejected without changing canonical calibration.
Legacy corner settings remain available only for legacy-provider behavior and
are not imported into the Rust store.

The local channel binds only to loopback. Producer/session/frame identifiers
are provenance and ordering data, not authentication; this application adds no
authentication mechanism.

## Consequences

### Positive

- Calibration pixels and their provenance come from the single Rust camera
  owner and are tied together in one response.
- Preview uses the newest captured frame even when inference replaces pending
  frames under load.
- A capacity-one shared reference, bounded on-demand JPEG encoding, and
  configured output limits preserve memory and slow-client isolation.
- The frontend can submit source-pixel points that Rust can verify against the
  active camera profile and revision before saving physical calibration.

### Negative

- The capture path must keep one shared frame reference available while
  inference may concurrently hold another; the existing aggregate frame budget
  must count those actual retained allocations.
- Rust and Python need a small local preview protocol and explicit process
  readiness/shutdown handling in addition to scene delivery.
- The browser must replace the legacy MJPEG calibration interaction with a
  descriptor-bound still-image workflow and correct source-pixel mapping.
- This provides a calibration snapshot only. It does not restore continuous
  Rust-mode viewer video or qualify physical coordinate accuracy.

## Alternatives considered

### Reopen the camera from Python for calibration

Rejected because it creates a second camera owner, can renegotiate different
geometry, and does not prove that the selected pixels match Rust inference.

### Use the legacy MJPEG stream and post canvas coordinates

Rejected because the browser cannot associate each multipart frame with a
source-frame descriptor, and the existing stretched canvas coordinates are not
source pixels.

### Return the last processed image

Rejected because detector overload intentionally skips pending frames; a
processed-frame preview would show the past instead of the newest capture.

### Keep a queue of preview frames per client

Rejected because a slow viewer could retain old frame allocations, consume the
aggregate memory budget, and display past images. Requests read only the latest
slot and never enqueue preview work per client.

## Verification

- Rust tests prove the preview slot returns the newest captured frame while
  inference holds an older one, and that its descriptor and RGB bytes come from
  the same frame/session/profile.
- Capacity and shutdown tests prove preview references remain within the
  configured frame-memory budget and are released when capture stops or fails.
- IPC tests cover unavailable, stale, malformed, oversized, slow, and
  concurrent requests without blocking capture or inference.
- Gateway tests prove Rust-mode requests never invoke Python camera/shared
  memory and that image bytes and descriptor are relayed together.
- Frontend tests cover letterboxing/cropping, CSS and device-pixel scaling,
  canvas borders, inverse source-pixel mapping, and descriptor retention.
- Rust calibration tests reject stale frame/session, profile, geometry,
  image-space, configuration or rectification revisions and ensure failed
  submissions leave the stored calibration untouched.
- Real-camera calibration and table-coordinate accuracy remain unqualified
  until the target device passes live capture and measured-table validation.
