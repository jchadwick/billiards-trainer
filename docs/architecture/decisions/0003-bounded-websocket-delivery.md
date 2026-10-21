# ADR-0003: Bounded WebSocket delivery per client

- Status: accepted
- Date: 2026-10-18
- Deciders: Project maintainers
- Supersedes: None
- Superseded by: None

## Context

The WebSocket handler previously awaited socket writes in the task that
requested a send. A slow client could therefore hold up broadcast work. A
client that stopped reading could also accumulate work faster than its socket
could transmit it. State and trajectory updates become stale quickly, while
control messages such as acknowledgements, alerts, and errors need ordered
delivery.

## Decision

Give each active connection one supervised writer task. All application
messages enter that connection's bounded outbound buffer, and only its writer
calls `send_text`. Bound replaceable updates and ordered control messages
separately. Coalesce state, trajectory, metrics, and performance messages by
stream; when the replaceable capacity is full, evict the oldest pending update.
Keep control messages in enqueue order, and disconnect a client if its control
capacity is exhausted or a socket write exceeds its configured deadline.

Read capacities and the write deadline from the shared configuration at
`api.websocket.delivery.*`, with defaults of 32 replaceable updates, 32 control
messages, and five seconds. Keep subscription aggregation buffers bounded at 32
entries by default and retain their newest data. Keep WebSocket video delivery
disabled.

## Consequences

### Positive

- A slow client no longer blocks healthy clients' broadcast paths.
- Per-client memory use is bounded, and replaceable updates favor current state.
- Control queue saturation and socket timeouts are visible and terminate the
  affected connection instead of silently losing ordered messages.

### Negative

- Slow clients may miss intermediate state, trajectory, metrics, or performance
  updates by design.
- Control queue saturation closes the connection, requiring the client to
  reconnect and resubscribe.
- Each active connection consumes one writer task and bounded queue storage.

## Alternatives considered

### Await each socket send from the broadcast caller

This allows one blocked client to delay unrelated delivery and does not bound
pending work.

### Drop arbitrary messages whenever a single FIFO queue is full

This could lose acknowledgements, errors, alerts, or configuration changes.

## Verification

- Deterministic gated-socket tests show that healthy delivery continues while
  another client's writer is blocked.
- Queue tests verify latest-update coalescing, oldest-update eviction, control
  ordering, and disconnect on control overflow.
- Timeout tests verify stalled sockets are closed and delivery metrics update.
- Subscription tests verify aggregation buffers retain only their newest
  configured number of entries.
