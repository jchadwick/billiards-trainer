# Projector Application Specification

## Purpose and boundaries

The projector is a local LÖVE application for displaying billiards graphics
and exercising its message receiver. It binds a custom UDP receiver to
`127.0.0.1:9999`. It has no application accounts or credential settings, does
not connect to the backend API, and has no remote deployment or remote-control
mechanism.

The native app is the only maintained projector runtime. The product browser
application is the separate HTML/JavaScript/CSS client in `frontend/web`.
Projector receiver behavior and local workflows are documented in
[README.md](README.md), [INSTALL.md](INSTALL.md), and [TESTING.md](TESTING.md).

## Implemented behavior

- Initialize the renderer, calibration state, local UDP receiver, and drawing
  modules from `main.lua`.
- Run fullscreen using `conf.lua` or windowed using `conf_windowed.lua`.
- Validate incoming coordinate payloads before passing messages to modules.
- Render locally received trajectories and calibration overlays.
- Run hardware-free receiver and coordinate contract checks.

## Explicitly not implemented

- Backend-to-projector publishing or a schema/coordinate adapter.
- A projector WebSocket client or browser-specific live transport.
- Remote installation, file synchronization, process control, or log retrieval.
- A system service, automatic restart manager, or centralized system logging.
- Browser rendering qualification or physical display latency measurement.

Backend integration requires a separately designed and tested adapter. The
receiver contract checks validate geometry rules only; they do not demonstrate
live backend integration or rendered output.

## Current source structure

```text
projector/
├── main.lua                    # Native app entry point
├── conf.lua                    # Fullscreen-capable LÖVE configuration
├── conf_windowed.lua           # Local windowed test configuration
├── core/                       # Renderer, calibration, UDP receiver, contracts
└── modules/                    # Trajectory, calibration, and debug drawing
```

## Local development and verification

Install LÖVE through the current machine's package manager. Then use:

```bash
cd frontend/projector
love . conf_windowed.lua
lua test_coordinate_contract.lua
```

The Lua check validates coordinate payload contracts without hardware. It does
not verify production backend integration, rendering behavior, long-duration
stability, or physical projection accuracy.
