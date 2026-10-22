# ADR-0066: Run server services in Rust and retain the standard web client

- Status: accepted
- Date: 2026-10-22
- Deciders: Project maintainers
- Supersedes: ADR-0064
- Superseded by: None

## Context

The earlier migration plan kept the Python backend service as the permanent
HTTP/WebSocket gateway and replaced the browser client with Rust/WASM. This
decision supersedes that plan after the user clarified both boundaries: every
deployed server-side component must ultimately run in Rust, while the browser
application remains a conventional HTML/JavaScript/CSS web application. The
existing React client, routes, WebSocket messages, models, and Python
training/development tools remain useful.

Rust capture, inference, public-state, calibration, and configuration
components are being qualified behind the existing Python service. Replacing
that service before its responsibilities have clear Rust owners would put API
compatibility, camera lifecycle, and domain migration into one high-risk cutover.
The working Rust configuration authority and local runtime protocols must be
preserved while the final server gateway moves languages.

Ball-number classification is reported nonfunctional and remains explicitly
outside the migration. The product priority is timely and accurate ball
locations. Game-state and prediction features remain secondary. Authentication
is not part of this application and must not be introduced by the migration.

## Decision

The final deployed server runtime will contain no Python service process. Rust
will own the HTTP and WebSocket gateway, static-asset serving, local process and
device supervision, configuration/calibration authority, capture, inference,
localization/tracking, public state, game state, and any retained physics or
prediction domains. The current Python API/service is a transitional adapter
only; it will be replaced after its routes, lifecycle, and operating duties
have Rust implementations and pass parity and platform qualification.

Keep the browser application as HTML, JavaScript, and CSS. Do not convert it to
Rust/WASM. Preserve its existing API and WebSocket contracts through server
migration; defer any necessary browser integration changes until Rust server
ownership is qualified. Static web assets may be built with the existing web
toolchain and served by the Rust service.

Keep Python only for off-device or development work such as model training,
annotation, data preparation, differential oracles, and acceptance tooling.
Such tools are not part of the deployed server runtime. Do not migrate or add
ball-number classification under this program. Shared contracts may retain
nullable number fields, but unknown identity remains unknown. Do not add
authentication, authorization, or credential flows. Because the HTTP
configuration surface remains unauthenticated, installed server executable
paths are deployment-managed and cannot be changed through HTTP updates or
imports.

Migrate sequentially: complete Rust ownership of configuration and capture
settings; perception and public-state lifecycle; game-state behavior; optional
physics/prediction; replace the Python server gateway; package and qualify the
complete Rust server on macOS and Linux; then make only the browser integration
changes required to consume the qualified Rust service. Keep each cutover
reversible until its compatibility, freshness, accuracy, resource, and shutdown
gates pass.

## Consequences

### Positive

- One native runtime owns every deployed server responsibility, avoiding a
  permanent Python/Rust service split.
- The existing browser application, trained models, API contracts, and
  development/training tools can be retained.
- The server migration can be qualified without also changing the browser
  language or introducing a WASM toolchain.
- Excluding the nonworking classifier keeps effort focused on reliable ball
  locations and freshness.

### Negative

- Rust must eventually implement the HTTP/WebSocket service, static delivery,
  supervision, and operational duties currently provided by Python.
- Python and Rust temporarily coexist during staged migration, so differential
  tests and explicit ownership boundaries are required until cutover.
- The web client remains a separate JavaScript application and must continue to
  follow versioned API and state-freshness contracts.
- Ball identity remains unavailable until a separate decision is supported by
  an effective, accuracy-qualified classifier.

## Alternatives considered

### Keep the Python backend service as the permanent gateway

Rejected because the user explicitly requires all deployed server components
to run in Rust. Retaining Python is only a migration bridge.

### Convert the browser application to Rust/WASM

Rejected because the user explicitly requires the browser application to stay
HTML/JavaScript/CSS. A framework rewrite would add risk without satisfying a
server-side requirement.

### Replace the Python gateway and browser application together

Rejected because changing both boundaries at once would complicate diagnosis of
route, WebSocket, state-ordering, and deployment regressions. The existing web
client remains while the server is replaced and qualified.

### Migrate numbered-ball classification with the other perception code

Rejected because the feature currently does not work and is not required to
establish primary location accuracy or freshness.

## Verification

- Rust API and WebSocket contract tests cover the existing browser routes,
  response shapes, errors, latest-state ordering, and slow-client behavior.
- Runtime and packaging checks show that the deployed process tree contains
  only Rust server-side processes and required native libraries; Python training
  and acceptance tools remain outside the deployment package.
- macOS and Linux builds, startup/shutdown/restart tests, configuration and
  calibration persistence tests, and real-camera/model acceptance pass before
  removing the Python service.
- Existing HTML/JavaScript/CSS build, type-check, browser tests, and end-to-end
  flows pass against the Rust service; no WASM build is introduced.
- State-quality and freshness gates prioritize ball-location precision, recall,
  position error, identity continuity where supported, and observation age.
  Numbered-ball classification remains absent from production scope.
- Repository search and deployment tests confirm no authentication or
  authorization system is introduced.

## Related records

- [ADR-0056](0056-retain-the-python-api-gateway-while-migrating-domain-logic-to-rust.md)
  defined the temporary Python gateway and is superseded in the final server
  language and ownership boundary.
- [ADR-0064](0064-defer-rust-web-client-and-exclude-ball-number-classification.md)
  is superseded because it selected a Rust/WASM browser client and retained the
  Python gateway as the destination.
- [ADR-0065](0065-move-unified-runtime-configuration-authority-to-rust.md)
  remains active for configuration schema, validation, revisions, persistence,
  and desired-versus-applied semantics. Its Python adapter is transitional.
- [Native domain migration plan](../../../thoughts/rust-domain-migration-plan.md)
  records ordered migration and qualification work.
