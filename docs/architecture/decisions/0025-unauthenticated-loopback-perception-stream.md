# ADR-0025: Unauthenticated loopback perception stream

- Status: superseded
- Date: 2026-10-20
- Deciders: Project maintainers
- Supersedes: ADR-0011
- Superseded by: ADR-0028 for handshake and producer-epoch behavior only

> Current status: the no-authentication decision remains active. Python replay
> supervision and replay-v2 ingress cited in this historical evidence were
> removed by ADR-0094; the Rust live-v3 producer is the current scene path.

> ADR-0028 supersedes this record's handshake and producer-epoch details. Its
> prohibition on authentication remains active and is reaffirmed by ADR-0028.
> ADR-0028 extends the prohibition to repository-controlled build and
> deployment workflows; this historical record's third-party-service boundary
> describes only the application scope at the time it was written.

## Context

The application is a single-user local training system. Its HTTP and WebSocket
APIs have no login, credential, or authenticated-client identity requirement.
The Rust
perception child and Python gateway also run under the same user on the same
machine; adding a second credential system only to their internal stream would
contradict that product boundary and add secret lifecycle, environment, and
failure-handling code without changing the deployment trust boundary.

The scene stream still needs protocol compatibility, bounded resource use,
producer-session continuity, and stale/duplicate rejection. Those are data
integrity and flow-control properties, not authentication. Binding the listener
to a loopback address and accepting only one active producer confines this
experimental local IPC endpoint to its intended topology. It does not protect
against another process running as the same OS user.

## Decision

Do not implement authentication anywhere in the application or Rust/Python
perception boundary. The scene handshake contains `type: hello`, the protocol
version, and the producer's existing engine-instance and source-session IDs.
Those IDs are provenance copied from every snapshot, not credentials or proof
of permission. The ready response reports the last accepted sequence only for
the matching producer epoch. Remove launch tokens, auth headers, login/session
APIs, JWT/API-key configuration, and authentication-only UI, diagnostics, and
data models.

Keep the perception listener loopback-only and single-producer, and retain
schema/version validation, size limits, deadlines, identity/revision/order
checks, bounded latest-only buffering, and observable rejection/drop counters.
These checks validate protocol and data integrity; they must not be described
or implemented as credentials, permissions, or identity proof. The existing
application APIs remain directly accessible to local clients without login.

Redacting credential-like headers from logs remains a defensive logging rule;
it does not enable, require, or imply an authentication feature. Credentials
required by separately operated third-party services (for example dataset
providers) are outside the application's client-authentication boundary.

## Consequences

### Positive

- No credentials or login flow are required to launch, use, or test the local
  application, and the internal process boundary has no token synchronization
  failure mode.
- Protocol versioning, loopback binding, schema validation, bounded queues, and
  source/session/revision checks continue to prevent malformed, stale, or
  oversized data from being accepted accidentally.

### Negative

- Another local process can connect to the loopback listener, and another
  process running as the same OS user can inject structurally valid data. This
  is an explicit consequence of the single-user local deployment model.
- The application is unsuitable for direct exposure to an untrusted network or
  multi-user hosting without a separately reviewed product/security decision.

## Alternatives considered

### Per-launch bearer token on loopback

Rejected because the user explicitly requires no authentication anywhere and
the current application has no authenticated-client boundary. A token would
only distinguish cooperating same-machine processes, while adding generation,
environment propagation, rotation, redaction, tests, and startup failure modes.

### Bind the internal stream beyond loopback and rely only on schema validation

Rejected because it expands the endpoint beyond its intended local process
topology. The listener stays loopback-only; loopback is a topology constraint,
not an authentication substitute.

## Verification

- `backend/tests/unit/test_scene_snapshot_ingress.py` verifies the
  identity-scoped handshake, rejects unsupported protocol versions, and
  validates loopback, frame limits, identity, sequence, and freshness behavior.
- Rust `scene_transport` tests verify the matching identity-scoped handshake,
  bounded latest-only delivery, reconnect watermarks, deadlines, and shutdown.
- The Rust writer rejects over-limit scenes before queueing and counts the
  rejection; one absolute deadline bounds each complete handshake and scene
  frame, including peers that trickle partial bytes. After a slow handshake,
  the newest waiting scene replaces the not-yet-written older scene.
- Historical Python replay-supervisor tests verified the retired replay child
  used no credential environment variables or command-line secrets.
- Provider/API lifecycle tests verify local API startup has no auth dependency.
- An opt-in black-box integration test runs the real Rust inference executable
  with the unauthenticated handshake and checks that model-backed observations
  reach HTTP and WebSocket clients without credentials.
- Repository searches and docs checks verify application login/token/API-key
  paths and auth-only setup have been removed. Header redaction is retained as
  logging hygiene, not an auth feature.

Historical implementation evidence (2026-10-20): the Python ingress and Rust sender
exchange an identity-scoped hello/ready handshake, and the supervisor no longer
creates or passes a launch credential. The API OpenAPI/CORS contract is covered
by `backend/tests/unit/api/test_local_access_contract.py`; ingress and child
lifecycle behavior were covered by the then-current scene-ingress and replay-
supervisor suites. The full backend suite
passed 1,394 tests with one model-dependent replay skipped; Rust formatting,
strict Clippy, and 99 tests passed. Frontend Vitest passed 42 tests and the Vite
production bundle built. Project-wide TypeScript typecheck remains red on
existing unrelated diagnostics and store/component errors.

The 2026-10-20 gateway follow-up also removes unused username/password
validators, auth-only error catalog entries, and unused HTTP 401/403 enum
values. The environment template no longer carries retired JWT/API-key
placeholders or unused database/dashboard credentials. The regression suite
checks the API schemas, CORS behavior, retired validators/statuses, and those
environment entries. The opt-in real-model test
passes through the rebuilt Rust binary, loopback scene receiver, anonymous
HTTP route, and anonymous WebSocket stream. Full verification for that run is
recorded in `thoughts/performance-audits/rust-model-api-gateway-20261020/`;
the existing 80% backend coverage gate remains unmet.

**Retained frontend follow-up (2026-10-20):** the web API and WebSocket clients
have no token setters, credential query parameters, or authenticated-socket
factory, and HTTP requests explicitly omit browser credentials. Regression
tests assert this anonymous transport contract. The strict TypeScript check,
all 48 frontend tests, and the production build now pass; the default build no
longer masks either typecheck or bundler failures. A real-model Rust replay
again reached anonymous HTTP and WebSocket clients in the opt-in lifecycle
test. Header-value redaction remains log hygiene only and adds no auth path.

**Transport hardening follow-up (2026-10-20):** adversarial review found that
oversized direct submissions could be retried indefinitely, socket timeouts
could be extended by trickling partial progress, and a snapshot held during a
slow handshake could delay fresher state. The Rust scene writer now rejects
and counts oversized submissions before they enter its one-slot queue, bounds
the complete handshake and frame operations by monotonic deadlines, validates
the receiver watermark against the JSON-safe integer limit, and sends the
newest waiting scene after handshake. Tests cover oversized rejection, a
trickling handshake, a slowly draining peer during shutdown, unsafe watermark
rejection, and supersession during handshake. The protocol remains
unauthenticated: its hello contains producer/session provenance only, with no
credential. Cross-process restart/session epoch scoping is now supported by
the version-2 handshake and receiver epoch reset; replay sessions remain
subject to the same source-identity checks as snapshot payloads.

## Related decisions and implementation tracking

This record supersedes ADR-0011's proposed per-launch token while retaining its
framing, loopback, queue, timeout, reconnection, and stale-data requirements.
ADR-0028 refines the handshake and epoch behavior without changing this
record's no-authentication decision.
The Rust stream remains experimental replay infrastructure; this decision does
not qualify live camera operation, deployment to untrusted networks, or
multi-user hosting.
