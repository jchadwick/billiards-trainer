# ADR-0006: API model and error contracts

- Status: accepted
- Date: 2026-10-18
- Deciders: Project maintainers
- Supersedes: None
- Superseded by: None

## Context

The application uses distinct core, HTTP API, and WebSocket models. Their
transport-specific fields differ, but shared ball properties must serialize
consistently. HTTP exception handling also needs to preserve request
correlation and protocol headers while keeping internal failures out of 5xx
responses.

## Decision

Keep domain and transport models separate. Centralize conversion of shared
core ball properties so HTTP and WebSocket representations preserve IDs,
numbers, position and velocity values with scale metadata, cue-ball and pocket
state, and confidence. Each transport continues to own its existing additional
fields and schema.

Use registered error codes for domain failures and status-code fallbacks for
framework HTTP errors. Map conflicts, validation failures, bad gateways, and
gateway timeouts to their registered codes. Preserve exception headers and
emit the same request ID in the response body and `X-Request-ID` header.
Return registered public messages for server errors and keep implementation
details in logs.

## Consequences

### Positive

- HTTP and WebSocket clients receive matching shared ball properties without
  coupling their independent schemas.
- Clients can correlate failures consistently and retain retry guidance
  supplied through HTTP headers.
- Internal exception messages are not exposed in server error responses.

### Negative

- New shared ball properties must be deliberately added to both transport
  schemas and the common converter mapping.
- Callers that relied on internal exception strings in 5xx responses must use
  server logs and request IDs for diagnosis.

## Alternatives considered

### Use one transport model everywhere

This would couple HTTP and WebSocket evolution and would require each transport
to carry fields that are not meaningful in its contract.

### Return framework exception details for every status

This is convenient for debugging but can expose filesystem paths, model
internals, and device diagnostics to clients.

## Verification

- Unit tests compare shared ball properties across API and WebSocket
  serialization while asserting the transport-specific schemas remain intact.
- Error-handler tests cover registered status mappings, request IDs, header
  preservation, domain error messages, and 5xx sanitization.
