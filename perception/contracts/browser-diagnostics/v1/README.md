# Browser diagnostics full snapshots v1

`/api/v1/diagnostics/ws` sends complete JSON snapshots. A client receives one
immediately after connecting, then receives the newest snapshot on evidence
changes and on the configured heartbeat. Intermediate snapshots may be coalesced.
The browser must not poll HTTP for this channel.

`gateway.instance_id` is stable for one gateway process and `gateway.revision`
increases for each published snapshot. `assessed_at_utc` is presentation time;
all ages and expiry decisions use the gateway's monotonic clock. `producer.state`
describes report freshness, not process health. Producer metadata is present only
while the supervised report is fresh. A retired process remains explicitly
`stopped`; its `exit_reason` distinguishes graceful stop, successful exit,
nonzero exit, launch failure and forced termination without exposing raw OS
errors or executable paths. An expired report retains its last-known
`report_age_ms`, while active metadata and `expires_in_ms` are cleared. An absent
or expired report cannot keep producer-positive stages green. The separate
`producer.last_known.report` contains the most recent accepted full report,
including model identity and producer stage evidence, and remains labeled as
historical across expiry, process retirement and generation replacement.

Stages always include the gateway, source lifecycle, capture progress, model
initialization, scheduled perception work, scene delivery and localization.
`source` names the evidence boundary. `unknown` means the gateway lacks evidence;
it must not be converted to healthy, idle or stalled. Scheduled work can be
stalled only while an actual job is explicitly in flight and past its declared
deadline. Phase-gated and low-rate work is idle. Localization continues to use
the scene lifecycle's measurement timestamps; diagnostics reports never refresh
ball, cue or motion evidence.

The stream heartbeat and browser stale threshold are supplied by validated
configuration. `no-scene-startup.json` is the contract example for an immediate
snapshot before producer or scene evidence exists.
