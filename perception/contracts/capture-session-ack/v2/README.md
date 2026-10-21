# Capture session acknowledgement v2

V2 retains the request revision, exact request-byte SHA-256, and source-session
report while requiring `source_report.output_frame_geometry`. Rust emits this
version only from a compatible v2 preflight, where the request geometry and
source report geometry already matched. The emitted pixels are RGB8 by the
frame-envelope contract; the negotiated profile continues to describe the raw
camera input mode.

The acknowledgement is configuration correlation, not authentication or proof
of physical camera behavior. A compatible report only establishes that Rust's
preflight compared the source-reported metadata with the request. The checked-in
fixture uses the v2 capture-settings example and its canonical request digest.

The v1 acknowledgement schema and serialization remain unchanged. V1 has no
output geometry field and is limited to source frames whose dimensions equal
the negotiated camera profile.
