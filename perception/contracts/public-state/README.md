# Public-state contract

`public-state-delivery-v1.schema.json` defines the state sent by the Rust
server over HTTP and WebSocket. The JSON fixtures in this directory exercise
conversion of valid Rust scene snapshots into the public contract, including
observed and cached positions, cue status, calibration, source-clock semantics,
and invalid or unavailable observations. The Rust contract tests are the
source of truth for expected output.

`legacy-policy-v1.json` supplies the versioned canonical image dimensions and
assumed ball radius used by the current public-state adapter. These settings
define the shared rendering plane; they are not camera resolution or physical
table dimensions. When a scene measurement includes detector box half-extents,
the adapter publishes them as `radius_extent` in canonical pixels, and the
viewer uses that per-ball size. The configured radius remains the fallback for
older scenes without box geometry. `lifecycle-policy-v1.json` supplies defaults for bounded
cached state: positions become stale after 100 ms and are removed after one
second, with a maximum of 64 cached balls. Runtime configuration can override
validated values.

The adapter preserves a source producer identity, session, frame identity, and
timestamp for every observation. It rejects duplicate or out-of-order
observations, accepts forward gaps from latest-only delivery, and clears cached
state when a new producer epoch starts. Health records do not refresh ball
locations. Stale state does not claim detector confidence; expired state does
not retain old coordinates.

`public-state-delivery-v2.schema.json` adds the phase-aware live envelope. A
delivery may carry a raw `source_snapshot` for a new localization, or omit it
for a cue/phase-only update. Both forms carry the complete retained
`current_state` when available and a `perception` object with independent ball,
motion, and cue evidence ages. The gateway rebases those ages to its local
monotonic clock. Phase-only updates never renew localization or per-ball
measurement timestamps; expired motion/layout evidence removes eligibility,
while cue expiry removes aiming eligibility without invalidating the layout.
An emitted cue estimate also carries its `source_frame_id`; a viewer can draw
cue guidance only over that exact displayed frame, even while retaining ball
positions measured from an earlier frame with their original age.

The v2 perception status also carries `empty_layout_verified`. A raw successful
empty detector result remains in `source_snapshot` for evaluation, but this flag
stays false until quiet dwell and the configured number of consistent successful
empty full-table observations establish an empty layout. Motion, context changes,
and expired layout evidence clear the flag.

Run the contract tests with:

```sh
make test-rust
```
