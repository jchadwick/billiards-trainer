# ADR-0106: Share identified video reception across application routes

- Status: superseded
- Date: 2026-11-02
- Deciders: Project maintainers
- Supersedes: None
- Superseded by: ADR-0109

## Context

Live Diagnostics currently owns the MJPEG fetch and identified frame buffer. Route navigation stops reception, so System Health cannot report whether the browser is receiving video. Starting another viewer per route would duplicate camera traffic and image memory. ADR-0100 requires overlays to use the exact identified frame and matching geometry.

## Decision

Move camera configuration resolution, one stream connection, identified frame buffering, object URL ownership, and delivery status into an application-scoped receiver. Keep it active during route navigation and dispose it with the application. Route components subscribe to the same receiver. Preserve exact identity matching and bounded buffering. A native MJPEG fallback may report that an image loaded, but cannot claim continuing per-frame freshness.

## Consequences

### Positive

- Health and Events can read the real browser receiver state without opening another video connection.
- Frame ownership and teardown remain centralized while navigation changes.

### Negative

- The receiver lifecycle spans the application and needs explicit configuration-change and shutdown handling.
- Ordinary unidentifiable MJPEG cannot expose ongoing frame freshness or support identified overlays.

## Alternatives considered

### Create one receiver per route

It duplicates bandwidth and memory, and route transitions can report false interruption/recovery events.

## Verification

- Frontend suite passed: 82 tests across 24 files, including an app-scoped receiver lifecycle test, timeout/recovery, source replacement, stale callbacks, object URL cleanup, and ordinary MJPEG fallback labeling.
- Existing exact-frame overlay tests continue to pass. Live, Configuration, Health, and Events route smoke rendered with the development backend unavailable; a live video request-count check was not possible in that smoke run.
- TypeScript checks and production build passed. The build reports the Monaco large-chunk advisory.
