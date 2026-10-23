# ADR-0097: Protect deployment bundle replacement and restart

- Status: accepted
- Date: 2026-10-23
- Deciders: Project maintainers
- Supersedes: None
- Superseded by: None

## Context

The Rust distribution builder replaces a path selected by
`BILLIARDS_DIST_DIR`. It previously treated any existing directory there as an
old application bundle, then recursively deleted its backup after installing
the new one. That could delete unrelated user data. Second-resolution backup
names could collide and make the deletion broader. The restart script also
started a new server immediately after SIGKILL without confirming the old
server had stopped, risking two server processes.

The package has runtime configuration and writable data directories, so a
successful replacement must leave the prior bundle available for recovery.
The scripts must remain usable on macOS and Linux without requiring a camera,
systemd, or a Rust compiler during their behavioral tests.

## Decision

Every completed distribution will contain an exact ownership marker. The
builder will refuse to replace an existing path that is a symlink, is not a
directory, or lacks the recognized marker. It will reserve a unique sibling
backup directory, move the old bundle there, install the completed staging
directory, restore the previous bundle if installation fails, and retain the
previous bundle after success. The cleanup command will delete only a marked
distribution. Users can inspect and remove retained backups manually.

The updater will wait for graceful shutdown, escalate to SIGKILL after its
configured timeout, then wait a second bounded interval. If the old process is
still alive, it will fail without starting `run.sh`. A recognized process
zombie counts as exited. A detected systemd service remains authoritative; a
failed systemd restart will not fall back to an unmanaged process.

Exercise package creation/replacement/failure/rollback and updater process
identity/shutdown/systemd branches through isolated shell tests using fake
build tools and disposable processes. Keep the existing real Rust `run.sh`
health/static/shutdown integration test as a separate layer.

## Consequences

### Positive

- Unmarked user data is never recursively removed by deployment or cleanup,
  failed installs can restore the old distribution, and successful installs
  retain a recoverable prior bundle.
- Restart cannot intentionally overlap the old and new server after escalation
  without first observing process exit.

### Negative

- Older distribution folders without the new marker cannot be automatically
  replaced. Operators must identify and move or remove those folders
  themselves; this avoids treating user data as application-owned content.
- Retained previous bundles consume disk space until an operator inspects and
  removes them.

## Alternatives considered

### Continue replacing any directory named `dist`

Rejected because the output path is configurable and may contain unrelated
data. A name is not proof of ownership, and recursive deletion after a
same-second backup collision can erase more than the intended package.

### Delete the previous bundle after a successful replacement

Rejected because it may contain the only copy of runtime configuration or
calibration data. Retaining a uniquely named prior bundle makes recovery
possible without guessing which files are disposable.

## Verification

- `scripts/deploy/tests/build-dist-smoke.sh` verifies validation, package
  layout, relative configuration, permissions, safe replacement, preservation
  of prior bundles, and rollback when the final rename fails.
- `scripts/deploy/tests/update-smoke.sh` verifies graceful and forced shutdown,
  no restart after a live process survives escalation, unrelated PID
  protection, stale PID cleanup, and systemd behavior.
- `perception/tests/deployment_bundle.rs` continues to exercise a real Rust
  server through staged `run.sh` startup, health/static delivery, and shutdown.
- Run `make test-deploy`, `make test-all`, and the ADR validator.
