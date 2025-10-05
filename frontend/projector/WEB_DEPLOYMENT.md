# Web Deployment Guide for Projector Application

This guide explains how to build and deploy the projector application as a web application.

## Overview

The projector application can now be deployed in two modes:

1. **Native Application** (LÖVE2D) - Recommended for production
   - UDP datagram socket communication
   - Full hardware acceleration
   - Lower latency

2. **Web Application** (love.js) - Easier deployment and testing
   - WebSocket communication (UDP not available in browsers)
   - Runs in any modern browser
   - No installation required

## Building the Web Version

### Prerequisites

- Node.js and npm installed
- LÖVE2D installed (for testing native version)

### Build Steps

```bash
cd frontend/projector

# First build - will clone love.js and install dependencies
./build_html.sh

# Output will be in: build/web/
```

The build script will:
1. Clone the love.js repository (first time only)
2. Install npm dependencies (first time only)
3. Package the LÖVE app into a .love file
4. Compile to WebAssembly using love.js
5. Output a complete web application to `build/web/`

## Testing Locally

```bash
cd build/web
python3 -m http.server 8080

# Open browser to: http://localhost:8080
```

## Deploying to Target Environment

### Deploy Web Files

```bash
# From frontend/projector directory
rsync -av build/web/ jchadwick@192.168.1.31:/opt/billiards-trainer/frontend/projector/web/
```

**IMPORTANT**: Never use the `--delete` flag as it will remove important files!

### Serve via Web Server

The web application needs to be served via a web server. Options:

1. **Simple Python Server** (for testing)
   ```bash
   ssh jchadwick@192.168.1.31
   cd /opt/billiards-trainer/frontend/projector/web
   python3 -m http.server 8080
   ```

2. **Nginx** (recommended for production)
   ```nginx
   server {
       listen 8080;
       server_name localhost;
       root /opt/billiards-trainer/frontend/projector/web;

       location / {
           try_files $uri $uri/ /index.html;
       }
   }
   ```

3. **Integrated with main web frontend**
   - Can be served as part of the main web application
   - Mount at `/projector` path

## Network Communication

### Native Application (UDP)

```lua
-- Listens on UDP port 5005
-- Receives JSON messages from API
```

### Web Application (WebSocket)

```lua
-- Connects to: ws://[api-host]:8000/api/v1/game/state/ws
-- Same JSON message format as UDP
-- Automatic reconnection on disconnect
```

## Backend Requirements

For web deployment to work, the backend API must provide:

1. **WebSocket Endpoint**: `/api/v1/game/state/ws`
   - Currently NOT implemented in the API
   - Needs to be added to FastAPI backend
   - Should send same messages as UDP broadcaster

2. **CORS Configuration**
   - Allow WebSocket connections from web client
   - Already configured in API specs

## Implementation Status

### ✅ Completed
- Build script for web compilation
- SPECS updated with deployment options
- Network protocol documentation
- .gitignore for build artifacts

### ⏳ Pending
- WebSocket client implementation in projector app
- WebSocket endpoint implementation in API
- Protocol auto-detection (web vs native)
- Web-specific configuration handling

## Message Format

Both UDP and WebSocket use identical JSON message format:

```json
{
  "type": "state|motion|trajectory|alert|config",
  "timestamp": "2024-01-15T10:30:45.123Z",
  "sequence": 12345,
  "data": {
    // Type-specific payload
  }
}
```

See `SPECS.md` for complete message protocol documentation.

## Browser Compatibility

The web version uses love.js compatibility mode (`-c` flag) for broader browser support:

- ✅ Chrome (latest)
- ✅ Firefox (latest)
- ✅ Safari (latest)
- ✅ Edge (latest)

**Note**: Compatibility mode disables SharedArrayBuffer (pthreads), which may result in slightly different audio behavior, but this doesn't affect the projector application since it doesn't use audio.

## Performance Considerations

- **Network Latency**: WebSocket adds ~5-20ms latency vs UDP
- **CPU Usage**: WebAssembly may use more CPU than native
- **Memory**: Browser overhead adds ~50-100MB memory usage
- **GPU**: WebGL provides good GPU acceleration

For best performance in production, use the native application with UDP.

## Troubleshooting

### Build fails with "npm not found"
Install Node.js: `brew install node` (macOS) or `apt-get install nodejs npm` (Linux)

### Browser shows blank page
- Check browser console for errors
- Verify all files were deployed
- Check if web server is serving files correctly

### WebSocket connection fails
- Verify backend API is running
- Check WebSocket endpoint is implemented
- Verify firewall allows WebSocket connections
- Check API host/port in configuration

### Graphics not rendering
- Check browser WebGL support: visit `about:gpu` in Chrome
- Try different browser
- Check browser console for WebGL errors
