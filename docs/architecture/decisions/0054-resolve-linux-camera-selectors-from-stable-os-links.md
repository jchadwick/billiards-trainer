# ADR-0054: Resolve Linux camera selectors from stable OS links

- Status: superseded
- Date: 2026-10-21
- Deciders: Project maintainers
- Supersedes: ADR-0040 for Linux persistent-identity resolution only
- Superseded by: ADR-0075 for Linux persistent-identity resolution when stable V4L symlinks are not mounted

ADR-0075 preserves this record's direct symlink lookup and selector contract,
and adds a fallback through the matching sysfs and udev records when a
container exposes those read-only records without mounting `/dev/v4l`.

## Context

During target qualification, GStreamer's Linux `DeviceMonitor` exposed the
camera's transient `/dev/video0` path through `device.path`, even though Linux
provided a stable `/dev/v4l/by-id/...` symlink to the same node. ADR-0040
required a stable provider identity and correctly rejected a numeric node for
capture selection, but its initial implementation could not use the stable
identity already maintained by the operating system. Numeric node order can
change after restart or hot-plug and must not become the selector.

Discovery and the supervised camera helper share the same metadata classifier;
they must derive the same identity and continue to select exactly one current
device. The discovery schema and capture-settings request contracts should
remain unchanged.

## Decision

On Linux, when the provider reports exactly `device.path=/dev/videoN`, resolve
that node through `/dev/v4l/by-id` first and `/dev/v4l/by-path` second. Accept
only an OS directory entry that is a symlink, has one safe basename, and
canonicalizes to the same node. Sort aliases lexically and select the first
match in the preferred directory. Report the resolved `/dev/v4l/...` path as
the persistent `identifier_value`, retain `device.path` as its source, and
preserve all raw provider properties unchanged for diagnostics.

If lookup fails, keep `/dev/videoN` classified as transient and issue no
capture selector. Do not scan or select arbitrary devices, infer a node from
display names, or fall back to a numeric camera index. ADR-0040's schema,
selector hash encoding, exact-match selection, request compatibility, and
non-capturing inventory decisions remain in force.

## Consequences

### Positive

- GStreamer can use Linux's stable device identity even when its provider
  metadata exposes only a transient V4L node.
- The same deterministic selector is derived by inventory and capture without
  changing wire formats or weakening exact-device matching.

### Negative

- Resolving identity now depends on OS-maintained V4L symlinks. Containers or
  restricted environments that do not expose those links remain unavailable
  for stable physical capture and fail closed as before.

## Alternatives considered

### Continue rejecting Linux devices when GStreamer reports only `/dev/videoN`

Rejected because the target's OS already exposed a stable alias for that exact
node. Rejecting it would make the physical camera unusable even though stable
identity is available without numeric fallback.

### Select by numeric `/dev/videoN` path

Rejected because Linux enumeration order can change and the path is explicitly
transient under ADR-0040.

## Verification

- Rust camera-discovery tests verify by-id preference, lexical determinism,
  by-path fallback, exact canonical-node matching, invalid-link fallback, and
  continued transient classification when no stable alias exists.
- The target camera discovery report must resolve the UC40M webcam to its
  `/dev/v4l/by-id` link, and the supervised physical helper must open that exact
  selector with a negotiated mode before reporting readiness.
- Capture-settings v1/v2/v3 fixtures, exact-one selection tests, formatting,
  and the source-enabled Rust test suite remain unchanged and must pass.

## Implementation evidence (2026-10-22)

- The source-enabled Linux camera-discovery tests pass: 9 passed. The
  feature-enabled release build of the live executable, capture helper, and
  discovery executable also succeeds; `cargo fmt --check` and the ADR
  validator pass.
- On target `hp-prodesk`, GStreamer discovery resolves the UC40M's
  `/dev/video0` to `/dev/v4l/by-id/usb-DSJ-211216-ZW_UC40M_Audio_01.00.00-video-index0`
  and produces selector
  `gstreamer-device-v1:133d9f4857ec09809e1ce59152e48b282a4c53696dd04485fce7e629b7c48db1`.
- Direct GStreamer capture delivered frames at the requested 1920×1080 MJPEG /
  10 FPS mode. The sampled scene was nearly black. The Rust live qualification
  consequently produced no usable localization and no performance or accuracy
  result. See the [target qualification notes](../../../thoughts/performance-audits/target-physical-camera-qualification-20261022/qualification-notes.md)
  and the retained failure record.
