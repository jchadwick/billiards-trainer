# Billiards Trainer - Getting Started Guide

This guide walks you through setting up the Billiards Trainer system on a single machine with Docker. The system combines computer vision, physics simulation, and augmented reality to provide AI-powered billiards training.

## Prerequisites

### Required Hardware
- **Pool Table**: Standard 9-foot billiards table
- **Camera**: USB webcam or Microsoft Kinect v2 sensor (recommended)
- **Projector**: HD projector capable of covering the table surface
- **Computer**: Linux/macOS/Windows machine with:
  - USB 3.0 port (for Kinect v2)
  - HDMI/DisplayPort output (for projector)
  - 8GB+ RAM, modern CPU with OpenGL support

### Required Software
- **Docker** with Docker Compose
- **Git** for repository access
- **USB 3.0 drivers** (for Kinect v2)

## Step 1: Repository Setup

### 1.1 Clone the Repository
```bash
git clone <repository-url>
cd billiards-trainer
```

### 1.2 Verify Project Structure
```bash
ls -la
```
**Expected output**: You should see `backend/`, `frontend/`, `docker-compose.yml`, and `Makefile`

**✅ Checkpoint**: Repository cloned successfully with all directories present

## Step 2: Hardware Setup

### 2.1 Connect Hardware
1. **Connect Camera/Kinect**: Plug USB camera or Kinect v2 into USB 3.0 port
2. **Connect Projector**: Connect projector to second display output
3. **Position Equipment**:
   - Camera: Mount above table center, pointing down
   - Projector: Mount above table, aligned with surface

### 2.2 Test Hardware Detection
```bash
# Test camera detection
lsusb | grep -i camera
# or for Kinect v2
lsusb | grep -i kinect

# Test display detection
xrandr  # Linux
# or
system_profiler SPDisplaysDataType  # macOS
```

**✅ Checkpoint**: Hardware devices detected by system

## Step 3: Environment Configuration

### 3.1 Create Environment File
```bash
cp .env.example .env
```

### 3.2 Configure Essential Settings
Edit `.env` file with your settings:
```bash
# Security (REQUIRED - Generate strong secrets)
JWT_SECRET_KEY=your-super-secret-jwt-key-here
DEFAULT_API_KEY=your-api-key-for-services

# Hardware Configuration
CAMERA_INDEX=0  # Usually 0 for first camera, 1 for second
PROJECTOR_DISPLAY_INDEX=1  # Usually 1 for second monitor

# Camera Settings (adjust based on your camera)
CAMERA_WIDTH=1920
CAMERA_HEIGHT=1080
CAMERA_FPS=30

# Projector Settings (match your projector resolution)
PROJECTOR_WIDTH=1920
PROJECTOR_HEIGHT=1080
PROJECTOR_FULLSCREEN=true
```

**Important**: Replace `your-super-secret-jwt-key-here` with a strong random string (32+ characters)

### 3.3 Generate Secure Keys
```bash
# Generate JWT secret
python -c "import secrets; print('JWT_SECRET_KEY=' + secrets.token_urlsafe(32))"
# Generate API key
python -c "import secrets; print('DEFAULT_API_KEY=' + secrets.token_urlsafe(24))"
```

**✅ Checkpoint**: Environment file configured with secure keys

## Step 4: Docker Container Setup

### 4.1 Build Docker Images
```bash
# Build all containers
docker-compose build

# Verify images built successfully
docker images | grep billiards
```

**Expected output**: You should see `billiards-trainer_backend` and related images

### 4.2 Start Supporting Services
```bash
# Start Redis cache first
docker-compose up -d redis

# Verify Redis is running
docker-compose ps
```

**✅ Checkpoint**: Redis container running successfully

## Step 5: System Startup

### 5.1 Start All Services
```bash
# Start entire system
docker-compose up -d

# Check all containers are running
docker-compose ps
```

**Expected services**:
- `backend` (ports 8000, 8001, 9090)
- `redis` (port 6379)
- `monitoring` (port 3000) - optional

### 5.2 Verify API Server
```bash
# Test API health
curl http://localhost:8000/health

# Expected response:
# {"status": "healthy", "timestamp": "...", "modules": {...}}
```

**✅ Checkpoint**: API server responding to health checks

## Step 6: Camera Calibration

### 6.1 Access Web Interface
Open browser to: `http://localhost:8000`

### 6.2 Configure Camera
1. Navigate to **Settings → Vision Configuration**
2. **Camera Selection**: Choose your camera device
3. **Table Detection**:
   - Click **"Detect Table"**
   - Manually adjust corner points if needed
   - Table should be highlighted with green overlay

### 6.3 Test Ball Detection
1. Place a ball on the table
2. Go to **Live View** tab
3. Verify ball is detected with blue circle overlay
4. **Adjust sensitivity** if detection is poor:
   - Increase **Detection Confidence** (0.6-0.9)
   - Modify **Color Thresholds** for different ball colors

**✅ Checkpoint**: Camera detecting table and balls accurately

## Step 7: Projector Calibration

### 7.1 Enable Projector Mode
```bash
# Ensure projector is connected and recognized
docker-compose logs backend | grep -i projector
```

### 7.2 Calibrate Projection Mapping
1. In web interface: **Settings → Projector Configuration**
2. **Display Test Pattern**: Click to project calibration grid
3. **Align Corners**: Drag corner points to match table edges
4. **Test Alignment**: Project circle at known ball position
5. **Fine-tune**: Adjust until projection perfectly overlays physical table

### 7.3 Verify Projection Accuracy
1. Place ball on table
2. Enable **"Show Ball Positions"** in projector settings
3. Projected circle should exactly surround physical ball
4. Move ball around table to test accuracy across surface

**✅ Checkpoint**: Projector accurately aligned with table surface

## Step 8: System Integration Test

### 8.1 Full System Test
1. **Place Multiple Balls**: Put 3-4 balls on table in different positions
2. **Check Detection**: All balls should be detected and highlighted
3. **Test Trajectory**: Use cue stick to aim at a ball
4. **Verify Prediction**: System should project predicted ball path
5. **Execute Shot**: Ball should follow approximately predicted path

### 8.2 WebSocket Connectivity Test
```bash
# Test WebSocket connection
curl -i -N -H "Connection: Upgrade" \
  -H "Upgrade: websocket" \
  -H "Host: localhost:8001" \
  -H "Sec-WebSocket-Version: 13" \
  -H "Sec-WebSocket-Key: test" \
  http://localhost:8001/ws
```

**✅ Checkpoint**: Full system working with real-time updates

## Step 9: Performance Optimization

### 9.1 Monitor System Performance
```bash
# Check container resource usage
docker stats

# View system logs
docker-compose logs -f backend
```

### 9.2 Adjust Performance Settings
Based on system performance, modify `.env`:
```bash
# Reduce camera resolution if system is slow
CAMERA_WIDTH=1280
CAMERA_HEIGHT=720

# Lower frame rate for better processing
CAMERA_FPS=15

# Adjust detection sensitivity
DETECTION_CONFIDENCE_THRESHOLD=0.7
```

### 9.3 GPU Acceleration (Optional)
If available, enable GPU acceleration:
```bash
# Verify GPU support
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi

# Update docker-compose.yml to enable GPU
# (already configured in the compose file)
```

**✅ Checkpoint**: System running smoothly with acceptable frame rates

## Step 10: Production Deployment

### 10.1 Secure Production Settings
```bash
# Use production configuration
cp config.prod.json.example config.prod.json
# Edit config.prod.json with production settings
```

### 10.2 Enable Monitoring (Optional)
```bash
# Start with monitoring stack
docker-compose -f docker-compose.yml -f docker-compose.monitoring.yml up -d

# Access Grafana dashboard
# http://localhost:3000 (admin/admin)
```

### 10.3 Set Up Automatic Startup
```bash
# Enable auto-restart on system boot
docker-compose up -d --restart unless-stopped
```

**✅ Checkpoint**: Production system running with monitoring

## Troubleshooting

### Common Issues

#### Camera Not Detected
```bash
# Check camera permissions
ls -la /dev/video*
# Add user to video group if needed
sudo usermod -a -G video $USER
```

#### Projector Not Found
```bash
# Verify display configuration
xrandr --listmonitors
# Adjust PROJECTOR_DISPLAY_INDEX in .env
```

#### Poor Ball Detection
- **Lighting**: Ensure consistent, bright lighting
- **Contrast**: Use high-contrast balls vs. table felt
- **Settings**: Adjust `DETECTION_CONFIDENCE_THRESHOLD`

#### WebSocket Connection Issues
```bash
# Check firewall settings
sudo ufw status
# Open required ports: 8000, 8001
```

### Performance Issues
- **High CPU**: Reduce camera resolution/frame rate
- **Memory Usage**: Restart containers periodically
- **Network Lag**: Use wired connection for better stability

### Getting Help
- **Logs**: `docker-compose logs backend`
- **System Health**: `curl http://localhost:8000/health`
- **Configuration**: Check `/config` endpoint for current settings

## Next Steps

After successful setup:

1. **Explore Training Modes**: Try different game types in the web interface
2. **Customize Physics**: Adjust ball physics and table characteristics
3. **Advanced Features**: Enable machine learning shot suggestions
4. **Multi-User**: Set up user accounts and progress tracking

## Quick Reference

### Essential Commands
```bash
# Start system
docker-compose up -d

# Stop system
docker-compose down

# View logs
docker-compose logs -f backend

# Restart after config changes
docker-compose restart backend

# System health check
curl http://localhost:8000/health
```

### Key URLs
- **Web Interface**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **WebSocket**: ws://localhost:8001/ws
- **Monitoring**: http://localhost:3000 (if enabled)

### Configuration Files
- **Environment**: `.env`
- **Docker**: `docker-compose.yml`
- **Application**: `config.dev.json` / `config.prod.json`

---

**System Status**: If you've completed all checkpoints successfully, your Billiards Trainer system is ready for use! 🎱
