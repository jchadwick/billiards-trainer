# Rust runtime and perception

This crate contains the complete supported server-side application runtime.
`billiards-server` serves the browser client and owns live-process lifecycle;
`billiards-perception-live` captures frames, runs the configured ONNX detector,
tracks balls, and sends bounded scene updates to the gateway. The GStreamer
helper owns platform capture interaction when that source is selected.

The current detector reports generic ball locations and cue status. It does
not identify numbered balls or produce trajectory predictions. State delivery
is freshness-first: capture and publication buffers are bounded, pending frames
can be replaced by newer frames, and clients may miss intermediate states.

Important implementation areas:

- `capture*` and `gstreamer_capture_source.rs`: source selection, ownership,
  frame limits, and latest-frame handoff.
- `inference/`: verified model loading, preprocessing, inference, decoding, and
  recorded-video replay.
- `tracking.rs`, `public_state.rs`, and `server.rs`: observation identity,
  public-state conversion, and best-effort delivery.
- `configuration_authority.rs` and `contracts/`: validated runtime settings and
  versioned wire contracts.
- `live_audit*`, `performance_audit*`, and `temporal_evaluation.rs`: bounded
  timing, delivery, replay, and reviewed-label evaluation tools.

Build and verify through the root Makefile. The current workflows are described
in the [architecture](../docs/ARCHITECTURE.md) and
[performance audit](../docs/PERFORMANCE_AUDIT.md) guides. Rust unit, contract,
fixture, camera-feature, and browser tests are the supported regression suite.

Ball refiner revision 6 computes its native sampling ROI from the projected
local annulus and boundary-probe footprint on every aiming iteration. A padded
proposal box alone can exclude valid off-axis background pixels. The resulting
access region is clipped to the source image and checked against the configured
ROI limit; the bilinear sampler still rejects missing pixels. This geometry
correction changes no color, support, coverage, or center-stability threshold.
Optics and ball-center assets must explicitly match the current refiner revision.

The angular background field uses only observed annulus pixels on each viewing
ray, so visible cloth/rail boundaries can support the same ball contour. Missing
background observations are unavailable; they are not interpolated. Pixel color,
coverage, alternative-boundary, and center-stability gates remain authoritative.
`aiming_convergence_px` is a source-pixel geometric stopping budget, separate from
the numerical IRLS `convergence_px`; it cannot exceed the center-stability budget
and is checked only after the complete candidate qualification passes.

Refiner revision 9 uses a bounded source-fixed seed lattice when the first view
cannot qualify a measurement. Every configured lattice position must be visited
before its existing deadline; identical initial views reuse their result. At
least `minimum_seed_agreement` fully pixel-qualified observations must agree,
and the pairwise spread of **all** qualified centers must fit the existing
center-stability budget. A conflicting qualified seed invalidates consensus.
The selected center and outline come from one real qualified observation;
averaging does not manufacture a contour. Each seed also retains the original
proposal's center-displacement bound. `seed_spacing_px` uses the declared
reference-width pixel scale; the configured grid and consensus count are bounded
before allocation. The default grid contains nine positions. Audit version 6
records seed visits, reuse, and consensus separately from circle candidates.
