# Producer diagnostics v1

This report is separate from scene snapshot v1 and the live-v3 capture handshake.
It can describe model loading or failure before any capture/scene exists. It is
not an HTTP health assessment or a processing-performance metrics service.

The gateway binds a loopback-only ephemeral address by default and registers a
fresh supervised launch generation. The supervisor supplies the actual endpoint
and generation to its child. Frames have a four-byte big-endian payload length;
the payload is nonempty JSON, limited to at most 65,536 bytes before allocation.
The receiver rejects unregistered/retired generations, process identity changes
within a generation, nonincreasing report sequences, unknown versions, malformed
fields and oversized messages. Reporter transport retains one newest full state;
heartbeats do not queue a backlog. Sequence numbers are positive safe JSON integers.

Build provenance is compiled into each executable. `BILLIARDS_BUILD_GIT_REVISION`
may inject a 7–64 hexadecimal source revision; `BILLIARDS_BUILD_DIRTY` may inject
`true` or `false`. Missing/invalid injected provenance stays null. No Git subprocess
runs at runtime. Producer and gateway can legitimately report different builds.

`metadata.model` is null while loading, failed or stopped. Active model metadata
must come from the successfully initialized inference engine's verified artifact
identity, never a configured filename. Provider request is separate from effective
execution proof, which remains null without verified evidence. Reports include no
raw model paths, configuration document, secret values or absolute clock ticks.

Evidence ages are nonnegative safe-integer milliseconds relative to report time.
The gateway timestamps receipt and adds its own elapsed monotonic time. An empty
`stages` list means no independent stage observations have been provided. Later
health assessment must not infer a healthy camera or processing stage from it.
The supervised reporter provides source lifecycle, capture progress, model
initialization, and scheduled-perception stages. Scheduled work carries explicit
state, job kind, in-flight status, start age, expected/completed/failed counts,
last completion age, and stable last failure code/age. A 30,000 ms deadline is
present only while an actual job is in flight; a completed result, phase-gated
work, or a low-rate idle interval has no active deadline. Stage identifiers must
be unique. Active model class IDs/names must be unique; these cross-field
constraints are enforced by Rust in addition to JSON schema.

Configuration is `perception.rust_live.diagnostics`. Old documents without it use
these defaults: enabled, `127.0.0.1:0`, 65,536 maximum bytes, 1,000 ms producer
heartbeat, 3,000 ms report expiry, 1,000 ms browser heartbeat, 3,000 ms browser
stale threshold, and 30,000 ms active-work timeout. Durations must be positive
and at most 300,000 ms; each expiry/stale threshold must exceed its heartbeat.
Configuration schema documents structural limits; Rust additionally validates
loopback and the duration relationships.

Fixtures cover an active loaded model, pre-scene loading, and failure before capture.
