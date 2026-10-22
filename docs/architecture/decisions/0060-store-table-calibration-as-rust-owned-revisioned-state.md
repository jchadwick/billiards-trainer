# ADR-0060: Store table calibration as Rust-owned revisioned state

- Status: accepted
- Date: 2026-10-22
- Deciders: Project maintainers
- Supersedes: None
- Superseded by: None

## Context

ADR-0059 moves four-corner homography solving into Rust, but the solved
`TableCalibrationV1` is currently only an in-memory value. The live Rust
capture request already carries an optional, provenance-checked calibration,
while the retained Python configuration service persists legacy editor corners
without camera profile, image space, rectification revision, or a complete
physical table definition. Those legacy corners cannot safely be promoted to a
live physical-coordinate calibration by inference.

The Python HTTP/WebSocket service may remain as a gateway, but table-plane
calibration is application-domain state and must have one durable Rust owner.
Updates may race across requests or processes, interrupted writes must not
destroy the last usable calibration, and clearing a calibration must not reset
its revision history.

## Decision

Add a Rust-owned, versioned table-calibration store around the existing solver
and `TableCalibrationV1` validator. The store envelope contains a storage schema
version, a monotonically increasing store revision, and either the calibration
at that revision or a tombstone. A missing store has no revision; the first
successful save receives revision 1. Each save or clear must supply the
expected current revision (including an explicit expectation that no record
exists), and a mismatch fails without changing the stored state. The solver
assigns the next store revision to the resulting calibration. A clear writes a
tombstone and advances the revision rather than deleting history.

Every read validates the envelope, the calibration bundle, and their revision
agreement. Read-size and lock-wait limits are supplied by the caller. Every
write validates the solved bundle, serializes to a temporary file in the same
directory, flushes and synchronizes it, atomically replaces the store, and
synchronizes the parent directory where supported. A stable sibling lock file
provides cross-process exclusive access on supported macOS and Linux
filesystems. Timeout, stale revision, malformed/unsupported data, pre-replace
I/O failure, and post-replace durability uncertainty remain distinguishable
outcomes. A solve or any failure before replacement leaves the previous record
intact. If the replacement succeeds but parent-directory synchronization fails,
the new revision may already be visible; the caller must reread the store before
retrying.

The existing Python configuration fields remain a legacy editor/gateway
contract until a separate integration slice supplies exact capture provenance
and delegates reads and writes through a Rust interface. They are not an
approved source for physical coordinates. Old corner data is not automatically
imported into the new store. Without an approved calibration matching the
active capture profile, Rust continues to mark table coordinates unavailable.

This decision establishes the store ownership and consistency contract. It
does not choose or claim completion of the Python-to-Rust invocation mechanism,
the API payload changes needed to provide provenance, live calibration update
semantics, or physical accuracy qualification.

## Consequences

### Positive

- Rust becomes the only writer of canonical, provenance-bound table
  calibration state.
- Compare-and-swap updates, cross-process locking, tombstones, and atomic
  replacement prevent silent lost updates and partial/corrupt state.
- Legacy corners can remain available to the editor without being mistaken
  for a calibrated live coordinate system.

### Negative

- The API gateway cannot make the canonical store useful until a Rust
  invocation path and complete provenance-bearing input are implemented.
- The file store adds a local lock-file and filesystem durability contract
  that must be tested on both macOS and Linux; directory-sync failure can leave
  a committed but not durably confirmed revision.
- Revisions are now durable domain history; callers must handle conflicts and
  cannot reset them by clearing calibration.

## Alternatives considered

### Keep canonical calibration in Python configuration

Rejected because the Python configuration currently stores only editor corners
and cannot validate the exact camera/image context required by Rust. It would
also leave domain calibration ownership in Python contrary to ADR-0056.

### Store only the corner list and solve during each live startup

Rejected because current corners lack complete provenance, startup would
recompute the calibration from mutable editor settings, and readers would not
have a stable immutable revision to acknowledge.

### Delete the file to clear calibration

Rejected because deletion resets revision history and makes a later save
indistinguishable from the first calibration. A tombstone preserves ordering
and lets readers observe an explicit cleared state.

### Use a database for this single local calibration record

Rejected for the current single-table deployment because it would add a
service/runtime dependency without improving the required single-record atomic
replacement contract. Revisit if the product gains multiple tables, users, or
concurrent remote writers.

## Verification

- Rust tests cover missing state, bounded reads, round trip, revision
  assignment, tombstone advancement, stale revision conflicts, simultaneous
  writers, malformed and unsupported envelopes, preservation of the previous
  record after pre-replace failure, and explicit post-replace durability
  uncertainty.
- `cargo fmt --all -- --check`, `cargo test --all-targets`, and strict Clippy
  validate the store implementation.
- macOS and Linux filesystem behavior, API gateway delegation, live request
  injection, and physical table accuracy remain separate verification gates.
- No calibration accuracy or live coordinate availability is claimed until a
  complete matching capture provenance record is supplied and exercised.

## Implementation tracking (2026-10-22)

The bounded control protocol is now multiplexed onto the Rust current-frame
preview listener. `BTCR0001` reads an opaque versioned snapshot and `BTCA0001`
submits a bounded JSON request. Rust checks the stable active capture context,
solves the ordered source-pixel corners, enforces the expected store revision,
and persists through the store defined here. A compatible saved calibration is
loaded by a later capture process; an incompatible or malformed record leaves
ball detection running while physical coordinates remain unavailable.

The Python service now exposes versioned snapshot and submit routes as a
bounded byte relay. It forwards request bytes unchanged and maps explicit Rust
statuses to HTTP responses; it does not solve, validate calibration geometry,
or write the store. Preview and calibration-control configuration is passed as
a complete, validated group to the Rust child. The focused backend protocol,
route, supervisor, and config suite passes 150 tests, and
`cargo test --all-targets --quiet` passes on the development Mac.

The existing browser wizard has not been wired to these store routes. The
browser client is frozen until the final Rust/WASM client stage under
ADR-0064. macOS/Linux filesystem qualification, target-camera operation, and
physical accuracy remain open; these protocol tests do not claim them.
