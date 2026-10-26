# Capture session acknowledgement v1

The v1 acknowledgement binds one source report to the exact capture request
revision and request-byte SHA-256. It distinguishes source-reported applied,
unsupported, and unknown values from requested values. A compatible outcome
means the Rust preflight checks passed; it does not prove physical camera
behavior.

The source-session identity must match the producer session. Readiness is
established only after the acknowledgement matches the request and the producer
handshake completes. Timestamps retain their source event meaning, and
read-completion is not camera exposure time.

V1 does not include separate emitted RGB8 dimensions. The current live path uses
the v2 acknowledgement, which includes those dimensions.
