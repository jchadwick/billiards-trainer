# Docker Development Environment

This guide explains how to use Docker Compose for local development with the billiards-trainer project using video files.

## Overview

The Docker Compose setup provides three services:
- **Backend**: FastAPI server with computer vision and ball tracking
- **Frontend**: React web application with hot reload
- **Projector**: LÖVE2D projector application with auto-reload

All services are configured for local development with:
- Source code mounted as volumes for hot reload
- Video files mounted for testing
- Network communication between services
- File watchers for automatic restarts

## Prerequisites

- Docker and Docker Compose installed
- Video files in the project root (`demo.mp4`, `demo2.mp4`, `demo2_short.mp4`)
- X11 server for projector GUI (optional, see Display Options below)

## Quick Start

```bash
# Build and start all services
docker-compose -f docker-compose.dev.yml up --build

# Start in detached mode
docker-compose -f docker-compose.dev.yml up -d

# View logs
docker-compose -f docker-compose.dev.yml logs -f

# Stop all services
docker-compose -f docker-compose.dev.yml down
```

## Individual Services

### Start specific services

```bash
# Backend only
docker-compose -f docker-compose.dev.yml up backend

# Frontend only
docker-compose -f docker-compose.dev.yml up frontend

# Projector only
docker-compose -f docker-compose.dev.yml up projector
```

## Service Details

### Backend (Port 8000)

- **API**: http://localhost:8000
- **WebSocket**: ws://localhost:8000/api/v1/game/state/ws
- **UDP**: localhost:5005 (for projector)
- **Hot reload**: Enabled via uvicorn `--reload`
- **Video source**: `/app/demo2_short.mp4` (configurable via `VIDEO_SOURCE` env var)

Mounted volumes:
- `./backend` → `/app/backend` (source code)
- `./config` → `/app/config` (configuration)
- `./models` → `/app/models` (ML models)
- Video files and data directories

### Frontend (Port 3000)

- **Dev server**: http://localhost:3000
- **Hot reload**: Enabled via Vite
- **API URL**: http://localhost:8000

Mounted volumes:
- `./frontend/web/src` → `/app/src` (source code)
- `./frontend/web/public` → `/app/public` (static files)
- Configuration files (vite.config.ts, tailwind.config.js, etc.)

### Projector (UDP Port 5005)

- **Network**: UDP communication with backend on port 5005
- **Auto-reload**: File watcher restarts LÖVE on changes
- **Display**: Requires X11 (see Display Options below)

Mounted volumes:
- `./frontend/projector` → `/app` (entire projector app)

## Configuration

### Video Source

Change the video file used by the backend:

```bash
# Edit docker-compose.dev.yml
environment:
  - VIDEO_SOURCE=/app/demo.mp4  # or demo2.mp4, demo2_short.mp4
  - VIDEO_LOOP=true
```

Or pass as environment variable:

```bash
VIDEO_SOURCE=/app/demo.mp4 docker-compose -f docker-compose.dev.yml up
```

### Backend Configuration

Edit `config/default.json` or mount a custom config:

```yaml
volumes:
  - ./my-custom-config.json:/app/config/default.json
```

### Network Configuration

Services communicate on the `billiards-network` bridge network:
- Subnet: 172.20.0.0/16
- Services can reach each other by name (e.g., `backend:8000`)

## Display Options for Projector

### Option 1: X11 Socket (Linux/macOS)

Mount the X11 socket and set DISPLAY:

```yaml
projector:
  volumes:
    - /tmp/.X11-unix:/tmp/.X11-unix:rw
  environment:
    - DISPLAY=${DISPLAY:-:0}
```

On macOS, you may need XQuartz:
```bash
# Install XQuartz
brew install --cask xquartz

# Allow connections
xhost + localhost
```

### Option 2: VNC/noVNC

Use a VNC server in the container for remote access:

```dockerfile
# Add to projector Dockerfile
RUN apt-get install -y x11vnc xvfb
CMD Xvfb :0 -screen 0 1920x1080x24 & \
    x11vnc -display :0 -forever & \
    /start.sh
```

### Option 3: Headless Mode

Run without display (for testing network/logic only):

```bash
# Comment out CMD in Dockerfile and use:
CMD ["tail", "-f", "/dev/null"]
```

## Development Workflow

### Making Code Changes

1. **Backend**: Edit files in `./backend/` - uvicorn auto-reloads
2. **Frontend**: Edit files in `./frontend/web/src/` - Vite HMR updates browser
3. **Projector**: Edit files in `./frontend/projector/` - inotifywait restarts LÖVE

### Viewing Logs

```bash
# All services
docker-compose -f docker-compose.dev.yml logs -f

# Specific service
docker-compose -f docker-compose.dev.yml logs -f backend
docker-compose -f docker-compose.dev.yml logs -f frontend
docker-compose -f docker-compose.dev.yml logs -f projector
```

### Debugging

```bash
# Enter container shell
docker-compose -f docker-compose.dev.yml exec backend bash
docker-compose -f docker-compose.dev.yml exec frontend sh
docker-compose -f docker-compose.dev.yml exec projector bash

# Check service health
docker-compose -f docker-compose.dev.yml ps
```

### Restart Services

```bash
# Restart specific service
docker-compose -f docker-compose.dev.yml restart backend

# Rebuild and restart
docker-compose -f docker-compose.dev.yml up --build backend
```

## Troubleshooting

### Backend won't start

```bash
# Check logs
docker-compose -f docker-compose.dev.yml logs backend

# Common issues:
# - Missing video file: Ensure demo files exist in project root
# - Port conflict: Check if port 8000 is already in use
# - GPU access: Remove GPU-specific code or add --device flag
```

### Frontend can't connect to backend

```bash
# Check network
docker-compose -f docker-compose.dev.yml exec frontend ping backend

# Verify API URL
echo $VITE_API_URL  # Should be http://localhost:8000
```

### Projector display issues

```bash
# Test X11 connection
docker-compose -f docker-compose.dev.yml exec projector env | grep DISPLAY

# Check LÖVE installation
docker-compose -f docker-compose.dev.yml exec projector love --version

# View auto-reload script
docker-compose -f docker-compose.dev.yml exec projector cat /start.sh
```

### Hot reload not working

```bash
# Backend: Check uvicorn reload settings
docker-compose -f docker-compose.dev.yml logs backend | grep reload

# Frontend: Check Vite HMR connection
# Open browser console, look for WebSocket errors

# Projector: Check inotifywait
docker-compose -f docker-compose.dev.yml exec projector ps aux | grep inotify
```

## Cleanup

```bash
# Stop and remove containers
docker-compose -f docker-compose.dev.yml down

# Remove volumes (caution: deletes data)
docker-compose -f docker-compose.dev.yml down -v

# Remove images
docker-compose -f docker-compose.dev.yml down --rmi all
```

## Performance Tips

1. **Use .dockerignore**: Speeds up build context transfer
2. **Layer caching**: Keep requirements/package.json separate for better caching
3. **Volume exclusions**: Use `/app/node_modules` to exclude host node_modules
4. **Resource limits**: Add resource constraints if needed:

```yaml
deploy:
  resources:
    limits:
      cpus: '2'
      memory: 4G
```

## Next Steps

- For production deployment, see `DOCKER.md`
- For projector-specific setup, see `frontend/projector/SPECS.md`
- For backend configuration, see `backend/config/SPECS.md`
