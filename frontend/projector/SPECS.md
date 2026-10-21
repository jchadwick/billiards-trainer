# Projector Application Specification

## Purpose and boundaries

The projector is a local LÖVE application for displaying billiards graphics
and exercising its message receiver. It binds a custom UDP receiver to
`127.0.0.1:9999`. It has no application accounts or credential settings, does
not connect to the backend API, and has no remote deployment or remote-control
mechanism.

The native app is the active runtime. An optional love.js build packages it for
browser preview, but the browser build does not provide a WebSocket client or
backend transport. The receiver contract and local workflows are documented in
[README.md](README.md), [INSTALL.md](INSTALL.md), [TESTING.md](TESTING.md), and
[WEB_DEPLOYMENT.md](WEB_DEPLOYMENT.md).

## Implemented behavior

- Initialize the renderer, calibration state, local UDP receiver, and drawing
  modules from `main.lua`.
- Run fullscreen using `conf.lua` or windowed using `conf_windowed.lua`.
- Validate incoming coordinate payloads before passing messages to modules.
- Render locally received trajectories and calibration overlays.
- Build a static browser package with `build_html.sh` and preview it on loopback.
- Run hardware-free receiver/coordinate checks and the local UDP test sender.

## Explicitly not implemented

- Backend-to-projector publishing or a schema/coordinate adapter.
- A projector WebSocket client or browser-specific live transport.
- Remote installation, file synchronization, process control, or log retrieval.
- A system service, automatic restart manager, or centralized system logging.
- Browser rendering qualification or physical display latency measurement.

Backend integration requires a separately designed and tested adapter. The
local UDP test sender exercises only the receiver contract and must not be
treated as evidence of live backend integration.

## Current source structure

```text
projector/
├── main.lua                    # Native app entry point
├── conf.lua                    # Fullscreen-capable LÖVE configuration
├── conf_windowed.lua           # Local windowed test configuration
├── core/                       # Renderer, calibration, UDP receiver, contracts
├── modules/                    # Trajectory, calibration, and debug drawing
├── build_html.sh               # Local browser package build
└── deploy.sh                   # Legacy-named local run/test/build helper
```

Despite its filename, `deploy.sh` operates only on the current machine. Its
supported commands are `run`, `test`, and `web`; it does not install software,
request credentials, or publish files.

## Local development and verification

Install LÖVE, Python 3, Node.js, and npm as needed through the current machine's
package manager. Then use:

```bash
cd frontend/projector
love . conf_windowed.lua
python3 test_udp_sender.py
./test_local.sh 30 0.5
lua test_coordinate_contract.lua
./build_html.sh
```

The tests and preview are local-only. A successful test run verifies message
handling and rendering behavior; it does not verify production backend
integration, hardware acceleration, long-duration stability, or physical
projection accuracy.
