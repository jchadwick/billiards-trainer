# ADR-0027: Report perception source health after readiness

- Status: accepted
- Date: 2026-10-20
- Deciders: Project maintainers
- Supersedes: None
- Superseded by: None

> Current status: live source-health reporting is owned by the Rust gateway
> and live supervisor. The Python replay-supervisor and API lifecycle evidence
> from the original migration is historical; ADR-0094 removed that path.

## Context

The Rust replay process can produce a valid localization and then stop or fail.
The scene worker already marks retained ball positions stale after the
configured source-silence interval, but `/health/perception` previously exposed
only process start/running/return-code fields. It did not report observation
age or distinguish a normal finite replay completion from a failed process.

The same stale transition intentionally removes the top-level transport
sequence. That sequence describes a published transport record, not a new
localization, and retaining it without a clear stale marker risks consumers
treating cached locations as current. The HTTP scene response nevertheless
required a non-null frame number, turning the intended stale response into a
server error.

## Decision

Expose bounded source freshness and process outcome in the existing anonymous
`/api/v1/health/perception` response. The scene worker reports its source
health (`awaiting_localization`, `healthy`, or `source_silent`), age of the last
successful localization using the gateway monotonic clock, and configured
silence threshold. The replay supervisor reports process status
(`not_started`, `running`, `completed`, or `failed`) and reports
`process_failed` as source health when the child exits nonzero. A zero exit is
normal completion for finite recorded replay; it is not by itself a process
failure. The source can still become silent as its last localization ages.

Keep API readiness separate from perception-source health for finite replay.
Do not infer failure from `process_running == false`, and do not add automatic
restart or a second publisher in this slice. Preserve the existing source
silence behavior: after the configured threshold, publish cached ball
locations with `is_observed=false`, unavailable current confidence, and
`freshness_status=source_silent` without advancing their observation times.

Make the scene HTTP `frame_number` nullable only for gateway-generated
`rust_scene_health` states whose transport sequence is absent. Keep a required
integer sequence for observed scene states. The status marker and absent
sequence together prevent a stale cached position from being represented as a
new scene frame.

No application, API, WebSocket, or process-stream authentication is added.
Health reporting, source-session correlation, loopback binding, and rate
limiting do not introduce credentials or identity checks.

## Alternatives considered

### Treat every exited replay process as a failure and restart it

Rejected because the current mode consumes finite recorded input and a clean
exit is expected. Automatic restart would replay old frames, risk duplicate
observations, and obscure the distinction between a completed file and a
failed live source. A future persistent live mode needs its own restart
decision and source-session policy.

### Mark the API unready whenever perception stops

Deferred because readiness currently describes the local API's ability to
serve requests, while recorded replay may complete normally and stale-state
diagnostics remain useful. A future live provider may define a separate
fresh-localization readiness policy.

### Keep returning the last integer transport sequence

Rejected for gateway-generated silence updates because it can be mistaken for
the sequence of the current health publication. The nested observation
identity already retains the last actual source frame for provenance.

## Consequences

### Positive

- Clients can distinguish a running source, a failed process, a completed
  replay, and a source whose last location has become stale.
- The health endpoint exposes freshness age and its configured threshold
  without retaining per-frame history.
- HTTP continues returning stale state with cached locations explicitly marked
  unobserved instead of converting the loss-of-source transition into a 500.
- The finite replay lifecycle remains useful after normal process completion.
- Existing anonymous access is unchanged.

### Negative

- Consumers must inspect source health and per-ball observation flags rather
  than assuming that a successful HTTP response contains fresh locations.
- A future persistent camera provider needs a separate policy for readiness,
  process restart, and recovery.

## Verification

- Worker tests verify monotonic observation age and source-silence status.
- API lifecycle integration starts an anonymous client, verifies live scene
  state, requests a controlled child failure, verifies process/source health,
  and confirms stale HTTP state has a null frame number and unobserved balls.
- The local-access contract test verifies that OpenAPI has no authentication
  routes or security schemes and that CORS does not enable credentialed
  requests.

## Related decisions and implementation tracking

This decision extends ADR-0010's stale/failure semantics and ADR-0024's shared
scene read model. ADR-0025 continues to prohibit authentication. Current live
health behavior is in `perception/src/server.rs` and
`perception/src/live_supervisor.rs`; the former Python replay supervisor and
API lifecycle suite were retired by ADR-0094.
