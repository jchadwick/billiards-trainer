# Prototype → API mapping: administration shell and Live Diagnostics

Source: `/Users/jchadwick/Downloads/index.html` (self-contained prototype), decoded import-map main bundle. This inventory precedes/records implementation decisions; Configuration has its separate `configuration-prototype-mapping.md`.

## Shared shell

| Prototype point | Existing source / treatment |
| --- | --- |
| Billiards Trainer brand, graphite theme, four-dot mark, navigation | Presentation only; Tailwind semantic tokens and shadcn/Radix primitives |
| Backend `trainer.local:8000` | Actual configured API/WebSocket host; no literal fixture host |
| State delivery Receiving/Connecting/Disconnected | Existing RootStore state-WebSocket connection; browser connection does not establish server health |
| Backend diagnostic stream status and overall stage summary | One root-owned `/api/v1/diagnostics/ws` connection and versioned full snapshots; stale/disconnected stream is shown as unknown/last-known rather than inferred gateway failure |
| Latest state received `84 ms ago` | Browser monotonic elapsed time since accepted state delivery |
| Phase `aiming` | `state.observation_status.perception.phase` |
| Camera Available / Connecting | Existing LiveView camera delivery and unavailable state; header retains camera label, detailed preview owns status |
| Inspection live/paused | Local image capture + cloned diagnostic state; streams continue; capture failures leave inspection live |
| Operational workspace `/ engine-01` | Actual engine instance, unavailable fallback |
| Configuration unsaved-dot/header status | Configuration page owns dirty indicator; drafts survive route navigation |
| System Health, Events & Debugging and event count | Implemented at `/health` and `/events`; health uses pushed backend stage snapshots, while the event count remains bounded browser-session evidence rather than a backend event total |

## Live Diagnostics

| Prototype point | Existing source / treatment |
| --- | --- |
| Latest received age vs observation age | Browser receipt clock is distinct from reported evidence ages |
| Ball observation min–max age | Live WebSocket provides aggregate `last_ball_observation_age_ms`, not independent track ages; show aggregate only. HTTP `/api/v1/game/state` has `last_update`, but is not polled |
| Cue observation age, stale limit | `cue_evidence_age_ms`, `cue_evidence_max_age_ms` plus browser elapsed time |
| Active limits `cfg-0042` | `observation_status.configuration_revision`; link Configuration |
| Camera preview | Config `api.stream.viewer_url`; existing direct MJPEG/frame-aware viewer |
| Source `camera-0`, image timestamp, FPS | Not supplied with equivalent contracts; omitted from compact camera toolbar; no fabricated live metrics |
| Exact frame matching/overlays | Existing frame-aware reader matches producer/session/frame/configuration/calibration/rectification identities; unsafe overlays withheld |
| Balls/Cue/Table/Labels overlay controls | Current iteration exposes combined detection-overlay visibility; individual toggles/table outlines omitted |
| Expand camera | Browser-only grid expansion |
| Pause inspection | Captured current image plus overlay and state; frozen evidence ages include elapsed time at capture |
| View events | Working link to `/events`; no event endpoint added |
| Snapshot/server diagnostic bundle | Only actual inspected-state JSON download provided; server history/bundle workflows omitted |
| Perception phase | `perception.phase` |
| Layout usable, trust reason | `perception.layout_trusted`, `trust_reason` |
| Guidance validity | `perception.guidance_valid` |
| Table visibility and reason | `perception.table_visibility.status`, `.reason` |
| Motion available/No motion reported | No equivalent boolean/value asserted; show `latest_motion_check_age_ms` and `motion_evidence_max_age_ms` instead |
| Ball/Cue/Motion freshness limits | `layout_verification_max_age_ms`, `cue_evidence_max_age_ms`, `motion_evidence_max_age_ms` |
| Ball track IDs | `balls[].id` |
| Ball identity | `number`, `type`, `is_cue_ball`; unavailable identity remains Unknown |
| Ball coordinates | `balls[].position.x/y/scale`; shown without physical-unit assumptions as requested in subsequent user steering |
| Prototype physical coordinates | Removed per explicit user steering to use existing coordinates and remove mm/specific measurement labels |
| Ball observation kind | `position_kind` / `is_observed` |
| Ball confidence | `confidence`, `confidence_kind`; unavailable confidence is not shown as zero certainty |
| Per-track age 84/420/980 ms, 1.84 s | No independent per-track ages in the live WebSocket contract (HTTP `last_update` is not polled). Verbatim age samples retained under collapsed unsupported inventory with `.highlight-issue` |
| Track source frame | `balls[].source_observation.frame_id` |
| Ball table filters | Existing observation/retention flags; no fixture filtering logic |
| Selected track detail dialog | Deferred; current table exposes available coordinates, identity, kind, confidence and source directly |
| Cue status, age, frame, confidence | `perception.cue`, `cue_evidence_age_ms`, cue `source_frame_id`, `confidence` |
| Cue shaft / axis / signed direction | `shaft_start_px`, `shaft_end_px`, `axis_unit`, `signed_direction`; labels omit measurement units |
| Cue tip | Existing `state.cue.tip_position`, or identified signed shaft endpoint; Unknown when direction unavailable |
| Cue direction unit `0.96, −0.28` | Directed live axis only when signed direction exists; otherwise exact fixture text `.highlight-issue` |
| Cue prototype tip in mm | Removed per subsequent coordinate-only user instruction |
| Producer / capture session / source frame | `observation_identity` with existing legacy observation-status aliases |
| Configuration/layout/calibration/rectification revisions | Existing observation status and perception revision fields |
| Snapshot revision / transport sequence | Existing observation identity/status fields |
| Technical raw JSON / copy | Actual inspected-state JSON shown; JSON download available |
| Illustrative events, retained history, synthetic fixture switching | Prototype demonstration tooling excluded. The separately approved backend additions are the supervised producer metadata contract and pushed diagnostics snapshots; they do not turn fixture events into backend history. |

## Verification and boundaries

Existing viewer regression tests remain intact. Added Diagnostics coverage verifies unsupported age markers remain separate from live rows, unavailable preview cannot enter paused state, live identity/confidence rendering, and retained-row filtering. Configuration tests cover update payload/read-only and merge handling plus draft failure behavior. Full frontend suite: 82 tests across 24 files. Typecheck and production build pass; route smoke covers all four tabs with unavailable-API paths and locally bundled JSON-only Monaco workers.

The application uses one state WebSocket, one separate diagnostics WebSocket, and one shared video receiver; HTTP requests are explicit user actions or initial configuration loads, not telemetry polling. `.highlight-issue` remains on prototype-only values/actions without API evidence, including per-ball live ages and illustrative cue values. Missing runtime values use Not available; current diagnostics stage/model/build data are shown only when supported by a valid fresh snapshot, with inactive last-known metadata labeled separately. Browser chart sampling defaults to 250 ms and can be configured with `VITE_DIAGNOSTICS_REFRESH_MS`; this timer samples already received browser state and does not make network requests.
