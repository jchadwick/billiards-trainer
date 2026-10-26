# Capture settings request v1

V1 describes a requested camera profile and its associated capture settings.
It is a versioned request contract parsed and validated by Rust. Requested
values and source-reported readbacks are distinct: an adapter must report what
it applies or supports, and the runtime does not infer applied values from the
request.

The request binds source selection, input mode, image space, calibration
revision, and camera profile identity. The profile identifier is a
configuration fingerprint, not a device serial number. Source timestamps retain
their event label; read-completion time is not camera exposure time.

V1 does not carry emitted RGB8 dimensions independently. Its frame geometry
must match the negotiated input profile. The current live application emits
capture-settings v3; v1 remains available to callers that use its exact schema.
