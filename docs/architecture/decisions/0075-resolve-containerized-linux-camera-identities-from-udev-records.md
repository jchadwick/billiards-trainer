# ADR-0075: Resolve containerized Linux camera identities from udev records

- Status: accepted
- Date: 2026-10-22
- Deciders: Project maintainers
- Supersedes: ADR-0054 for Linux persistent-identity resolution only
- Superseded by: None

## Context

ADR-0054 resolves a provider's transient `/dev/videoN` value through actual
`/dev/v4l/by-id` and `/dev/v4l/by-path` symlinks. This is deterministic and
works on the host, but a container may receive the camera device node plus
read-only `/sys` and `/run/udev` metadata without mounting `/dev/v4l`. In that
configuration GStreamer can enumerate and capture from the camera, while the
Rust helper cannot bind the configured stable selector and fails before its
first frame. The target host's udev record maps the exact character-device
number to the same stable V4L aliases, so the selector contract need not be
weakened or changed.

## Decision

When a Linux GStreamer provider reports exactly a transient `/dev/videoN`
path, camera identity resolution will first retain ADR-0054's direct stable
symlink lookup. If those symlinks are not visible, resolution may use the
matching `/sys/class/video4linux/videoN/dev` device number to read that exact
character device's `/run/udev/data/c<major>:<minor>` record. Only `S:` aliases
in the `v4l/by-id/` or `v4l/by-path/` namespaces are eligible. Require one safe
basename, prefer `by-id`, sort aliases lexically, and report the same
`/dev/v4l/<namespace>/<basename>` identifier and `device.path` source that
direct symlink resolution would have produced.

The fallback must bind the sysfs and udev records to the enumerated node; it
must not infer identity from display names, serial-number fields, unrelated
udev records, or `/dev/videoN` alone. If sysfs or udev metadata is missing,
malformed, or does not contain a supported alias, keep the identifier
transient and issue no capture selector. Discovery schema, selector hashing,
exact-match source selection, capture-settings requests, and anonymous access
remain unchanged.

## Consequences

### Positive

- Linux containers can select the same camera as the host using read-only
  kernel and udev metadata without mounting a wider `/dev` tree.
- The exact-device selector and fail-closed behavior remain intact; the
  fallback reports the same stable alias as the existing symlink path.
- Deployments without udev metadata retain the existing direct-link path and
  otherwise remain explicitly unavailable.

### Negative

- Linux persistent discovery has another OS metadata path that requires
  fixture coverage for malformed and mismatched records.
- A container that exposes neither stable V4L symlinks nor the matching sysfs
  and udev metadata still cannot use the Rust physical camera source.

## Alternatives considered

### Mount `/dev/v4l` into every capture container

Rejected as the only solution because each deployment and diagnostic runner
would need a broader device-tree mount even when it already exposes the
read-only sysfs and udev records required to identify the exact node.

### Select from udev serial or bus-path strings directly

Rejected because those fields would introduce a second selector identity
format and could mismatch the per-node V4L function. The udev device-number
record already names the OS-managed stable V4L aliases for the exact node.

### Accept the transient `/dev/videoN` selector

Rejected because numeric enumeration can change after reboot or hot-plug and
could silently bind a different camera.

## Verification

- Rust tests resolve the exact device-number udev record when V4L symlinks are
  absent, preserve by-id preference and lexical ordering, reject unsafe alias
  names and mismatched or absent records, and leave unsupported nodes
  transient.
- The source-enabled capture helper on Linux must recompute the same selector
  as host discovery and report readiness only after opening the exact camera
  and receiving a valid frame.
- Run the source-enabled Rust suite, formatting, strict Clippy, and ADR
  validation; verify the selector against the target camera inventory.
