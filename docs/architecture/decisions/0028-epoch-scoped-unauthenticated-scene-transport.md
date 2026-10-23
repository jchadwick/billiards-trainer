# ADR-0028: Epoch-scoped unauthenticated scene transport

- Status: accepted
- Date: 2026-10-20
- Deciders: Project maintainers
- Supersedes: ADR-0025 for handshake and producer-epoch behavior only
- Superseded by: None

> ADR-0030 adds a request-bound capture acknowledgement for live startup.
> ADR-0094 later retired replay-v2 ingress. This record's no-authentication
> rule and producer-epoch semantics apply to the active live-v3 scene path.

## Context

The scene receiver returns its last accepted transport sequence so a sender can
avoid replaying a scene after reconnect. The v1 handshake returned one
receiver-global watermark, while the Rust tracker starts its transport sequence
at one for a new engine process. The Python ingress also pinned one engine and
source session for its full lifetime. As a result, a restarted producer could
skip every new scene up to the old watermark and then still be rejected for
changing identity. A camera-session change within a long-lived process had the
same identity-pinning problem.

The project has no authentication boundary, and the user explicitly requires
that there be no authentication anywhere. This includes application APIs,
frontend clients, Rust/Python IPC, and repository-controlled build/deployment
automation. Engine and source-session IDs already appear in every scene
snapshot and establish data provenance; they are not secret and do not prove
who connected. Loopback binding and schema/sequence validation limit the
intended topology and accidental malformed data, not same-user access.

The existing container workflow previously logged in to GHCR with a GitHub
workflow token to publish images. That is incompatible with the project-wide
no-authentication requirement and is replaced by local image builds from
anonymous public-source fetches. Build jobs do not publish packages, upload
artifacts, or use authenticated cache services.

## Decision

Advance the internal scene-stream handshake to protocol version 2. The hello
contains the protocol version plus the engine-instance ID and source-session
ID that will appear in scene payloads. Constrain both IDs using the existing
bounded identifier contract. Do not add a credential, login, permission check,
token, signed identity, or authentication claim.

Do not add credentials to other project-controlled surfaces or workflows.
Hosted build/test workflows fetch the public source anonymously, declare an
empty permission set, and build locally; they do not publish images, upload
artifacts, or deploy to remote hosts.

Optional shared WebSocket rate limiting remains an anonymous availability
control. Its Redis connection URL must not contain username/password userinfo
or authentication query parameters; a deployment must expose Redis over its
trusted local network without service credentials. IP limits may reject excess
anonymous connections, but they do not identify or authorize a client.

The receiver returns a transport-sequence watermark only when the hello's
engine/source-session pair matches the current producer epoch. A different
pair starts a new epoch: reset transport and snapshot-revision watermarks,
source-observation ordering baselines, and source freshness; discard any
queued snapshot from the prior epoch and count that loss. The first accepted
scene must match the pair declared in the handshake. The sender reconnects when
engine or source-session identity changes and scopes its watermark comparison
to that handshake. A snapshot superseded before the first write remains
replaceable under the latest-only policy.

## Consequences

### Positive

- A new engine process or capture session can publish sequence one without
  waiting for the previous epoch's counter to be reached.
- Reconnect deduplication remains available within one engine/source-session
  epoch.
- Cached scenes, source timestamps, and sequence state from the previous epoch
  cannot be mistaken for current observations after a restart.
- The stream remains anonymous and credential-free; protocol validation is
  data-integrity behavior only.
- CI can run without project secrets because source is public and build/test
  jobs do not publish or upload outputs. Container images are available only
  in the local runner for that job; the project no longer provides automated
  registry distribution or remote deployment.

### Negative

- Protocol v1 and v2 peers cannot communicate; sender and receiver must be
  upgraded together.
- Engine and session IDs are producer claims. Any local process that can reach
  the loopback listener can claim another ID or submit structurally valid data.
- Public HTTP and WebSocket routes remain anonymous, including mutating
  configuration and lifecycle operations. The current API defaults to
  `0.0.0.0`; reachable clients are therefore governed by host/network policy,
  not by a credential check.
- All application and internal service surfaces remain anonymous. Operators
  must run the service only on a network they intend to expose; there is no
  account, role or secret-based access boundary.
- Hosted workflows assume the repository remains public. They cannot publish
  images, persist CI artifacts or deploy remotely without violating the
  no-authentication constraint.
- A producer that reuses both IDs after a restart cannot be distinguished from
  a reconnect of its prior epoch. Runtime owners must create a fresh source
  session when capture restarts and a fresh engine instance when the producer
  process restarts.
- Resetting the latest queue on an epoch change intentionally drops the old
  cached state; downstream health/state consumers must not carry it forward as
  fresh.

## Alternatives considered

### Keep a receiver-global watermark and pin identity for the receiver lifetime

Rejected because process/camera restarts can permanently block fresh scenes or
force a coordinated receiver restart even when the listener itself is healthy.

### Add a random per-launch token to distinguish producer epochs

Rejected because the product has no authentication contract and the user
requires no authentication anywhere. A token would add secret lifecycle without
making the loopback channel private from processes running as the same user.

### Keep authenticated GHCR publishing or add deployment credentials

Rejected because the user requires no authentication anywhere in the
project-controlled system. The image-building workflow now builds locally and
discards its outputs when the job ends. Distribution and remote deployment
remain manual local-file operations.

### Infer the epoch from the first snapshot after connecting

Rejected because the sender needs the correct watermark before deciding whether
to transmit that snapshot. The handshake must state the producer provenance
before the receiver reports its epoch-scoped sequence.

## Verification

- Rust transport tests verify that engine/source-session IDs are sent in the
  hello, that an identity change reconnects, and that the new epoch receives a
  fresh watermark while same-epoch reconnects retain deduplication behavior.
- Rust transport tests also verify over-limit rejection, complete-operation
  deadlines under trickling/slow peers, safe-integer watermarks, and newest-only
  delivery after a slow handshake.
- Python ingress tests verify same-epoch watermark retention, new-epoch reset,
  cached-queue discard accounting, restarted sequence one, scene/hello identity
  matching, malformed identifiers, and loopback/size limits.
- The fake-producer API lifecycle path verifies that the supervisor's configured
  engine/session IDs match the handshake and scene payload.
- Workflow review verifies there are no registry logins, package-write scopes,
  credential-bearing source checkout steps, authenticated cache/artifact
  uploads, or remote deploy commands; public source fetches and empty Actions
  permission sets are checked in all workflow files.
- Local-access tests verify that API OpenAPI schemas define no security schemes,
  routes ignore credential-looking headers, and CORS does not enable browser
  credentials. Deployment regression tests verify project-owned scripts and
  current deployment guides do not invoke SSH/SCP or remote-host rsync.
- WebSocket rate-limit tests verify Redis URLs carrying userinfo or
  authentication query parameters are rejected before a Redis client is
  initialized.
- `docs/docker-deployment.md` and the local distribution script describe only
  local builds and anonymous application access.
- Full Rust and backend suites, strict Clippy, formatters, and ADR validation
  must pass. The opt-in real-model cross-runtime path remains separately
  required for model-backed transport evidence.

## Revisit conditions and related records

This record changes internal scene transport continuity and restates the
project-wide no-authentication boundary. ADR-0025 remains the historical record
of removing application and stream credentials. ADR-0010 governs source/session
observation semantics; ADR-0021 governs detection and scene records. Revisit the
handshake only with a versioned protocol change and tests for receiver restart,
producer restart, capture-session restart, same-epoch reconnect, and freshness
of the first scene after each transition. This protocol remains an experimental
loopback path and does not qualify live camera operation.

## Evidence and implementation tracking

The Rust sender implementation is in
[`perception/src/scene_transport.rs`](../../../perception/src/scene_transport.rs);
the Python receiver is in
[`backend/perception/scene_ingress.py`](../../../backend/perception/scene_ingress.py).
Their unit and lifecycle tests are linked above. The migration plan records
remaining live-camera, physical-accuracy, temporal-accuracy, and backend
coverage evidence gaps. This ADR does not select or add any authentication
mechanism.

### Capture-to-scene pipeline publication (2026-10-20)

`CaptureScenePipeline` serializes each complete scene with the v1 encoder and
submits it through the same `LatestSceneWriter` and protocol-v2 epoch handshake
as recorded replay. The stream carries engine and source-session identifiers as
provenance; it has no token, credential, or client-authorization field. The
pipeline's bounded queue and lifecycle tests pass with the full Rust suite
(160 tests). The focused API, workflow, and deployment no-auth selection passes
22 tests. This library path has not yet been launched by a camera provider or
qualified against hardware, and it adds no authentication anywhere.

### Preflight-bound capture publication (2026-10-20)

The capture coordinator now validates the opened source session before its
first read, and `CaptureScenePipeline` can be bound once to that compatible,
session-scoped result before frame consumption. The binding checks capture
revision, image space, freshness, tracker profile/rectification and calibration
consistency; wrong-session frames and mismatched geometry/timestamp semantics
are rejected before inference. The scene sender still uses the existing
epoch-scoped protocol-v2 handshake and latest-only writer. Engine and
source-session identifiers remain provenance only. The bounded readiness
status, rejection, timeout, and owner-thread release behavior are exercised by
the fake-source Rust tests; all 182 Rust tests pass with formatting and strict
Clippy. The provider remains replay-only: no camera adapter is enabled, and the
Python effective-configuration/acknowledgement bridge and physical/temporal
accuracy qualification remain open. The application, clients, IPC, and
project-controlled workflows continue to have no authentication anywhere.

### Rust child environment containment (2026-10-20)

The replay supervisor no longer copies the parent process environment into the
Rust child. It passes only `TMPDIR` when configured; the executable, arguments,
and working directory are explicit, and this build does not require a dynamic
runtime-library override. A regression test injects API keys, auth tokens,
passwords, database URLs, and cloud/CI secrets into the parent and verifies
that none reach the child. The supervised path remains anonymous; this change
removes unnecessary secret propagation rather than adding an authentication
mechanism. The supervisor/API lifecycle selection passes 10 tests; the opt-in
real-model lifecycle check is skipped when its external model paths are absent.

### Supervised child restart epoch (2026-10-22)

The Rust live supervisor now generates a fresh engine-instance ID and
source-session ID at each child spawn. An unexpected child exit therefore
starts a new producer epoch, so the gateway resets the sequence watermark and
drops the prior child's cached state. IDs remain stable for the life of one
child, preserving same-epoch transport reconnect behavior. The supervisor
regression test launches two fake children and verifies both IDs change and
remain valid UUIDs; gateway epoch tests verify the receiver resets old state
for changed IDs and retains the watermark for same-epoch reconnects. This
corrects the earlier implementation gap described by the negative consequence
above without changing the protocol or adding authentication.
