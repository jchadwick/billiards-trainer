# Projector Deployment Status

## ✅ Deployment Complete

The LÖVE2D projector application has been successfully deployed to the target system at `192.168.1.31:/opt/billiards-trainer/frontend/projector/`.

### What Was Deployed

1. **LÖVE2D Projector Application** (`frontend/projector/`)
   - Main entry point: `main.lua`
   - Configuration: `conf.lua`
   - Core systems: Module manager, networking, calibration, renderer
   - Modules: Trajectory visualization, calibration UI
   - Libraries: JSON parser
   - Documentation: README.md, INSTALL.md, SPECS.md

2. **Updated run.sh Script**
   - Now starts both backend and projector
   - Checks for LÖVE2D installation
   - Handles graceful shutdown of both processes
   - Projector logs to `/tmp/projector.log`

3. **Cleaned Up**
   - Old `backend/projector/` folder removed
   - Python-based projector replaced with LÖVE2D

## 🔧 Manual Installation Required

LÖVE2D needs to be installed on the target system with sudo access:

```bash
# SSH into target system
ssh jchadwick@192.168.1.31

# Install LÖVE2D
sudo apt-get update
sudo apt-get install -y love

# Verify installation
love --version
```

## 🚀 Running the System

### Start Everything

```bash
# On target system
cd /opt/billiards-trainer
./run.sh
```

This will start:
- ✅ Backend API (with auto-reload)
- ✅ Projector (fullscreen) - if LÖVE2D is installed

### Projector Controls

Once running, the projector supports:
- **Ctrl+R**: Hot reload modules
- **C**: Toggle calibration mode
- **P**: Pause/unpause
- **ESC**: Quit

### Calibration Mode Controls

- **Arrow Keys**: Move selected corner
- **Shift+Arrow**: Move corner faster (10px)
- **Tab**: Select next corner
- **R**: Reset to default
- **Mouse**: Click to select, drag to move
- **C**: Save and exit calibration

## 📍 Current Status

### ✅ Completed
- [x] LÖVE2D projector application created
- [x] Module system implemented
- [x] UDP networking on port 9999
- [x] Calibration system
- [x] Trajectory visualization module
- [x] Deployed to target system
- [x] Updated run.sh to start both services
- [x] Documentation created

### ⏳ Pending
- [ ] Install LÖVE2D on target (requires sudo)
- [ ] Add UDP broadcasting to backend
- [ ] Test end-to-end trajectory visualization
- [ ] Calibrate projector to table
- [ ] Create additional modules (effects, training drills, games)

## 🔌 Backend Integration

To send trajectory data from the backend to the projector, you need to add UDP broadcasting to `backend/api/websocket/broadcaster.py`:

```python
import socket
import json
import time

class MessageBroadcaster:
    def __init__(self):
        self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    async def broadcast_trajectory(self, trajectory_data):
        # Send via WebSocket (existing)
        await self._broadcast("trajectory", trajectory_data)

        # Also send via UDP for low-latency projector display
        try:
            msg = json.dumps({
                "type": "trajectory",
                "timestamp": time.time(),
                "data": trajectory_data
            })
            self.udp_socket.sendto(
                msg.encode(),
                ("localhost", 9999)  # Projector UDP port
            )
        except Exception as e:
            print(f"UDP broadcast error: {e}")
```

## 📁 File Locations

### On Target System (192.168.1.31)
- Projector: `/opt/billiards-trainer/frontend/projector/`
- Backend: `/opt/billiards-trainer/backend/`
- Run script: `/opt/billiards-trainer/run.sh`
- Projector logs: `/tmp/projector.log`

### On Local System
- Projector source: `frontend/projector/`
- Deployment script: `frontend/projector/deploy.sh`
- Run script source: `scripts/deploy/run.sh`

## 🛠️ Troubleshooting

### Projector won't start
1. Check LÖVE2D is installed: `love --version`
2. Check projector files exist: `ls -la /opt/billiards-trainer/frontend/projector/`
3. Check logs: `tail -f /tmp/projector.log`

### No trajectory display
1. Verify backend is running
2. Check UDP port 9999 is not blocked
3. Look for network status in projector UI (green = connected)
4. Verify backend is broadcasting UDP messages

### Calibration issues
1. Press 'C' to enter calibration mode
2. Press 'R' to reset to defaults
3. Use arrow keys or mouse to adjust corners
4. Press 'C' again to save

## 🎯 Next Steps

1. **Install LÖVE2D** on target system (requires sudo password)
2. **Add UDP Broadcasting** to backend MessageBroadcaster
3. **Test Projector** by sending test messages via UDP
4. **Calibrate** projector to align with table
5. **Verify** trajectories appear correctly when balls are detected
6. **Create Additional Modules** as needed (effects, games, training)

## 📚 Documentation

- Projector README: `/opt/billiards-trainer/frontend/projector/README.md`
- Installation Guide: `/opt/billiards-trainer/frontend/projector/INSTALL.md`
- Specifications: `/opt/billiards-trainer/frontend/projector/SPECS.md`
- Main Plan: `/opt/billiards-trainer/PLAN.md`
