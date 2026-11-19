# Events & Debugging: prototype → API mapping and gaps

Reference: `/Users/jchadwick/Downloads/index.html`, decoded `kH` component and its eight `RA` fixture records. Browser-observed event history uses state, video receiver, configuration apply, and diagnostics evidence. The diagnostics snapshot arrives on the separate `/api/v1/diagnostics/ws` push stream; this is a health snapshot, not a backend event feed or audit log.

## Supported integration

| Prototype point or action | Current implementation and source |
| --- | --- |
| Events & debugging tab; Live diagnostics / System health detail links | `/events`, `/live`, `/health` using existing client routing |
| Current browser session start / event count | Actual RootStore creation time and bounded, in-memory retained record count; not the fixture timestamp or server totals |
| State delivery connection lifecycle | Existing shared WebSocket client's `onConnectionState`; deduplicated connection-state transitions are browser observations |
| Accepted phase / capture-session / configuration-revision changes | Diff accepted ordered state snapshots; first values establish baselines, repeated and rejected old snapshots do not produce records, and changes after reconnect or missing evidence are labeled across a gap |
| Cue / ball / motion freshness crossings | Existing reported ages and limits, advanced with the viewer's monotonic freshness helpers; stale/recovered/unknown crossings are browser estimates and recovery requires a new evidence identity |
| Camera preview failure / stall / recovery / source replacement | Shared app-owned receiver transitions; an identified received frame confirms recovery, native image load does not, and source URLs are omitted |
| Configuration apply confirmation | Successful existing config PUT response; changed field paths, restart requirement and known revision IDs are recorded before canonical reload, without setting values |
| WebSocket response details | Existing `status`, `error`, `subscribed`, `unsubscribed` control responses; credential values are redacted before retention and timestamps describe browser receipt |
| Current pushed backend diagnostic snapshot | The root-owned diagnostics WebSocket store supplies current stream status, snapshot revision, gateway identity, stage assessments, and active/last-known producer metadata; detail/export labels its freshness and source |
| Severity/component/type | Browser classification of transport, state, freshness, video and interface-action records (Info/Warning/Error), explicitly separate from backend event severity |
| Search, severity/component/type/time filters | Local filtering of retained real browser records; session/last1minute/last5minutes; no network requests |
| Follow live / new-event count / return latest | Local selection follows new pushed records; pause stops automatic detail selection, not backend delivery |
| Select individual/all filtered events | Local selection for reviewed export; no server mutations |
| Detail ID, timestamp, source, component, type, payload | Browser-generated event ID and observation time, explicit Browser observed / WebSocket control response / Interface action origin; payload reflects only evidence for that source |
| Copy detail | Browser clipboard with secret-key and URL-credential redaction; failure message if clipboard unavailable |
| Collect evidence / review / download | Local reviewed JSON containing only selected supported sections. Highlighted sample rows are never exported |
| Current health evidence | Explicit one-shot GET `/api/v1/health/live` and `/api/v1/health/ready` when Prepare review is clicked; includes checked time and HTTP status, including503, or request failure |
| Current state | Latest accepted state from the existing single `/api/v1/websocket/ws` state subscription, frozen at collection start; not state at historical event time |
| Runtime and revisions | Existing latest-state observation identity, engine/session and configuration/calibration/rectification revision fields when supplied; absent values not invented |
| Optional active configuration | Explicit GET `/api/v1/config`, `.values`; unsaved editor drafts excluded; known secret keys and credentials in URLs redacted recursively before review/download |
| Session retention | Frontend memory only, bounded by `VITE_EVENT_HISTORY_LIMIT` with default200; oldest records evicted; no localStorage or persistent backend history |

## Gaps: what the prototype asks for and what is available

| Prototype expectation | API / application limitation | Treatment |
| --- | --- | --- |
| Backend perception freshness event, authoritative onset, reason and frame | Browser estimates freshness crossings from accepted state ages/limits; no backend event feed or authoritative onset exists | Real browser stale/recovered/unknown records are source-labeled. Verbatim `evt-0208` remains highlighted because its backend time, payload and frame are unsupported. |
| Camera preview delivery restored and preview timeout events | Shared receiver reports identified-stream failure, stall, recovery on a received frame, and source replacement. Native image mode cannot prove ongoing freshness | Receiver transitions are recorded without source URLs. Verbatim `evt-0207` and `evt-0206` remain highlighted; their exact times, timeout, message and payload are not observed. |
| Configuration applied audit record and prior/current revisions | Successful PUT confirms an interface action and updated field paths/restart requirement; new revision is unknown until canonical reload and no durable audit API exists | A local Interface action record is emitted before reload and contains no setting values. Verbatim `evt-0205` remains highlighted because its IDs/timestamps and audit history are unsupported. |
| Reconnected record with message ID | Browser connection callbacks prove socket reconnection, not the prototype's `message_id:"state-18410"` or first subsequent valid state receipt | Real lifecycle records are supported; exact `evt-0204` fixture and its unsupported message ID remain highlighted |
| Close event with close code1006 | Current WebSocket abstraction reports disconnected/error state and optional Error, but does not expose CloseEvent code/reason to RootStore | Real disconnect state captured without invented code; exact `evt-0203` fixture highlighted |
| Backend phase transition event | Accepted snapshots reveal observed phase changes; latest-state delivery may coalesce intermediate transitions and timestamps describe browser observation | Browser phase transitions are recorded without claiming authoritative onset/history. Verbatim `evt-0202` remains highlighted. |
| Capture-session-start event | Accepted snapshots reveal changed session identity, not its start event/time or complete history | Browser session identity changes are recorded without claiming a start. Verbatim `evt-0201` remains highlighted. |
| Backend-reported log/event origin | Rust gateway exposes state/control messages and a separate pushed diagnostics snapshot, but no event feed or logs endpoint. The projected `events` list remains empty. | Real records are labeled Browser observed / WebSocket control response; diagnostics is included only as current evidence, not converted into historical backend events. All eight original backend/sample event records remain separately marked with `.highlight-issue`. |
| Event frame, capture session and configuration provenance | State/freshness records carry only their own observation identity; video events use a source generation; transport events are not enriched with unrelated latest-state fields | No historical snapshot is attached to an event; each record remains scoped to its own evidence source |
| Related state snapshot retained with historical event | No event-to-state snapshot store exists; old state cannot be reconstructed from a frame or event ID | Historical retention not offered; current snapshot only available as separately labeled collection-time evidence |
| Historical camera image | MJPEG is live delivery with a bounded viewer buffer, not an archive | Explicitly unavailable; never invent an image from a frame identifier |
| Persistent history and server logs | No API or persistence contract | Prototype `Not available` label highlighted; bundle lists both as missing |
| Resource monitoring metrics | Not present in state or health endpoints | Bundle lists missing, not inferred from browser connection/latency |
| Server-generated diagnostic bundle | No endpoint | Download is explicitly a browser-session evidence file, with missing sections listed |
| Exact illustrative session time/timezone/day | Real browser events occur now and use ISO receipt timestamps plus browser locale display | Real values replace fixtures; original sample dates/times remain verbatim inside separate fixture inventory |

## Behavior boundaries and verification

- One existing state WebSocket, one diagnostics WebSocket, and the shared video receiver feed bounded browser history/evidence. The diagnostics socket contributes current snapshots, not event history. No extra event subscription and no HTTP polling are introduced. Repeated state frames are omitted; accepted changes, freshness crossings, receiver transitions and confirmed interface actions become records.
- Health and configuration reads occur only on explicit Prepare review; partial health failures are evidence, while configuration read failure prevents claiming an active-configuration snapshot.
- Event IDs are browser-local sequence IDs. Timestamps describe browser receipt/connection observation and must not be read as backend event onset.
- The evidence collector snapshots state and selected records at collection start, separately timestamps subsequent health/configuration reads, and lists missing sources.
- Secret redaction covers sensitive property names recursively and username/password/query credentials in standalone URL strings. Review remains available before download; arbitrary unstructured prose cannot be guaranteed secret-free.
- Tests cover accepted-state baselines/order, reconnect gaps, local freshness expiry/recovery/unknown transitions, receiver recovery across `connecting`, confirmed versus rejected applies, secret-free field paths, no HTTP traffic while idle, bounded retention/filtering, prototype isolation, readiness 503 evidence, and nested/URL redaction.
