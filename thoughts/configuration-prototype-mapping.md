# Configuration prototype → API map

Reference: `/Users/jchadwick/Downloads/index.html`, decoded `OH` configuration component. Configuration remains an API-owned JSON document edited in Monaco against the Rust-owned schema. The configuration contract also includes the producer/diagnostics transport settings described below; these are infrastructure settings, not a commitment to a setting-specific UI.

| Prototype point / action | Integration |
| --- | --- |
| Editor configuration document | GET `/api/v1/config`, `.values`; full current settings, no prototype sample keys |
| Schema source / inline diagnostics | GET `/api/v1/config/schema`, raw backend schema, local Monaco JSON worker |
| Active revision | `.values.metadata.configuration_revision`, displayed read-only; stripped before validate/save |
| Last loaded | Browser timestamp after successful GET |
| Last applied | Browser timestamp after confirmed successful PUT; unknown before this tab applies |
| Validate / problems / warnings | PUT `/api/v1/config?validate_only=true`, `success`, `validation_errors`, `warnings`; Monaco diagnostics separately |
| Review changes | Local Monaco diff between loaded baseline and draft, after backend validation |
| Apply configuration | PUT `/api/v1/config`, raw object body; warnings require explicit `force_update=true` opt-in |
| Restart result | PUT response `.restart_required`; no invented restart endpoint |
| Reload active | GET current configuration + schema; confirmation before discarding draft |
| Import / Export | Browser file import/export of the draft; no application until explicit review/apply |
| Find / replace / Format | Monaco built-in actions |
| Reset defaults | Prototype text `Reset defaults` copied verbatim with `.highlight-issue`, disabled. Existing POST reset mutates active configuration immediately and cannot implement the prototype's draft-only reset; no defaults-read endpoint is exposed. |
| Revision conflict | GET latest immediately before apply and reject changed baseline locally. Existing HTTP PUT has no browser-supplied atomic revision precondition, so no atomic conflict guarantee is claimed. |
| Draft retention | In-memory retention across route changes; browser unload warning for unapplied edits |

PUT is recursive merge, not replacement. Removed object keys are rejected locally with their paths rather than silently retaining them server-side. Schema read-only settings are stripped from payloads. Store revision edits are explicitly rejected. Save failures preserve the draft; successful save reloads the canonical merged document and assigned revision. Network failures indicate unconfirmed outcome and recommend reloading before retry.

## Current diagnostics configuration

`GET /api/v1/config/schema` exposes the current versioned schema. The additive
`perception.rust_live.diagnostics` object configures the loopback producer
reporter and browser diagnostics stream: enablement, bind/ephemeral port,
message bound, producer heartbeat/report expiry, browser snapshot heartbeat/
staleness, and the in-flight processing deadline. The root deployment
`config.json` includes these defaults while preserving its existing local
overrides. As configuration settings evolve, this schema and the defaults must
stay synchronized; no prototype-only setting has been added.
