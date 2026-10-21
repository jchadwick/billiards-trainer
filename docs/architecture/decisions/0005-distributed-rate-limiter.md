# ADR-0005: Fail-closed WebSocket connection rate limiting

- Status: accepted
- Date: 2026-10-18
- Deciders: Project maintainers
- Supersedes: None
- Superseded by: None

## Context

The WebSocket endpoint limits connection attempts by client IP. The original
implementation kept an unbounded, process-local dictionary, trusted forwarded
address headers from any peer, and performed its check synchronously. Deploying
multiple workers would also give each worker an independent quota and would
create multiple owners for the same table camera and game state.

## Decision

Use an asynchronous rate-limit store interface. The default in-memory store
keeps a bounded, expiring sliding window and refuses new keys when full so it
does not evict active quota history. Redis is an optional shared store selected
with `api.rate_limiting.store = "redis"` and configured through
`BILLIARDS_RATE_LIMIT_REDIS_URL`. Its Lua admission operation uses Redis server
time and updates each client's sorted set atomically across API processes.
Both stores perform anonymous IP-based admission only; neither authenticates a
client. The Redis URL must be credential-free: URL userinfo and authentication
query parameters are rejected before the Redis client is initialized. A Redis
deployment must therefore allow this application to connect without service
credentials and rely on its trusted local network boundary.

Use the configured per-IP attempt limit and window, defaulting to 10 attempts
per 60 seconds. The memory store retains at most
`api.rate_limiting.max_tracked_ips` keys (default 10,000) and admits no new key
while that capacity is full. When Redis is selected, startup readiness probes Redis and
reports the API unready while it is unavailable. Admission fails closed on
missing configuration and runtime Redis errors. The API stays available for
health diagnostics during an outage.

Redis client connection and response timeouts, plus application-level deadlines
for initialization, readiness probes, admission, and shutdown, use
`api.rate_limiting.store_timeout_seconds` (default 2 seconds). Uvicorn's proxy
header rewriting is disabled so `scope.client` remains the direct transport
peer; the application then accepts forwarding headers only from configured
trusted proxy networks.

Only accept `X-Forwarded-For` or `X-Real-IP` when the direct peer address falls
within `api.rate_limiting.trusted_proxies`. For a forwarded chain, walk from the
direct peer toward the client and stop at the first untrusted hop.

Deploy one backend process per table. Redis shares only the connection rate
limit; it does not distribute camera ownership, game state, or WebSocket
broadcasts. Multi-worker table-state coordination needs a separate design.

## Consequences

### Positive

- Connection attempts share one quota across processes when Redis is selected.
- Store failures cannot silently disable connection enforcement.
- Forwarded address spoofing is ignored unless it arrives through a trusted
  proxy.
- The default deployment preserves one owner for each table's camera and state.

### Negative

- Redis deployments require the optional `redis` Python package and a reachable
  credential-free URL in the service environment; Redis requiring username or
  password authentication is unsupported.
- Redis outages make the service unready and block new WebSocket connections.
- The bounded in-memory store rejects new client keys when full until entries
  expire, favoring enforcement over admission.

## Alternatives considered

### Keep independent process-local quotas

This permits clients to multiply their quota by reconnecting across workers and
does not satisfy shared enforcement.

### Trust all proxy headers

Any direct client could spoof a forwarded address and bypass per-IP controls.

## Verification

- Deterministic store tests cover expiry, capacity, and atomic admission across
  two store instances sharing a Redis test backend.
- Failure tests cover missing Redis configuration, Redis outages, and readiness.
- URL tests reject username/password userinfo and authentication query
  parameters before importing the optional Redis client.
- Address resolution tests cover trusted proxy chains and spoofed headers from
  untrusted peers.
- Deployment tests verify the generated config uses one worker and the service
  starts the real `backend.main` launcher with proxy-header rewriting disabled.
- Gated Redis fakes verify initialization, readiness, and admission deadlines
  fail closed; Nginx tests verify the canonical WebSocket route.
