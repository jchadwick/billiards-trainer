# Capture settings request v3

V3 retains v2's explicit `output_frame_geometry` for emitted, tightly packed
RGB8 frames. It replaces the unconditional control requirement and requested
values with a closed tagged policy at `requested.camera_controls`.

`{"mode":"source_defaults"}` means the capture source owns its existing
camera defaults. Rust submits no control request and requires no control
capabilities or readbacks. The source report still records supported,
unsupported, or unknown capability/readback statuses; no requested or applied
values are synthesized.

`{"mode":"require","requirements":...,"requested":...}` requests concrete
control values and requires the corresponding capabilities plus exact
readbacks before a frame may be read. It carries the same control fields and
validation rules as v1/v2. Missing fields, extra fields, unknown modes, and
inconsistent requirements fail closed.

The default fixture uses `source_defaults`; the require fixture demonstrates
the strict path. V1 and v2 retain their original schemas and strict semantics.
The existing capture-session acknowledgement v2 already returns the complete
source report and needs no version change for v3 settings.

The v3 default `maximum_frame_age_ms` is 500 ms. The earlier 100 ms default
rejected captured frames before a low-cadence full scan could be scheduled and
processed. The wider bounded allowance is an acquisition freshness window, not
a promise that the resulting coordinates are current: phase-aware admission
still requires uninterrupted quiet evidence from capture through decision, and
motion-generation/context changes reject obsolete results. Keep the frame-age
default aligned with configured capture/scan cadence rather than treating
inference completion time as the measurement time.
