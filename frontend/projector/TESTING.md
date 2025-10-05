# Testing the Projector Application

## Option 1: Test on Target System (Recommended)

Since LÖVE2D is not installed locally but you have the target system, this is the best approach:

### Step 1: Install LÖVE2D on Target

```bash
ssh jchadwick@192.168.1.31
sudo apt-get update && sudo apt-get install -y love
```

### Step 2: Deploy Test Files

```bash
# From your local machine
cd frontend/projector
rsync -av test_udp_sender.py jchadwick@192.168.1.31:/opt/billiards-trainer/frontend/projector/
```

### Step 3: Run Projector on Target

```bash
# SSH to target
ssh jchadwick@192.168.1.31

# Run projector in windowed test mode
cd /opt/billiards-trainer/frontend/projector
DISPLAY=:0 love . conf_windowed.lua
```

### Step 4: Send Test Data from Another Terminal

```bash
# In another terminal, SSH to target
ssh jchadwick@192.168.1.31

# Run test sender
cd /opt/billiards-trainer/frontend/projector
python3 test_udp_sender.py

# Or send a single test message
python3 test_udp_sender.py single
```

You should see:
- Animated green trajectory line moving across the screen
- Yellow collision markers appearing
- White dashed aim line rotating
- Ghost ball indicator (white circle with "8")

---

## Option 2: Test Locally (Requires Installing LÖVE2D)

### Install LÖVE2D on macOS

```bash
brew install --cask love
```

### Run Projector Locally

```bash
cd frontend/projector

# Run in windowed mode
love . conf_windowed.lua
```

### Send Test Data (in another terminal)

```bash
cd frontend/projector
python3 test_udp_sender.py
```

---

## Test Modes

### Continuous Test (Default)
Sends trajectory updates continuously for 30 seconds:
```bash
python3 test_udp_sender.py
```

### Custom Duration
```bash
python3 test_udp_sender.py continuous 60 0.5
# Args: mode duration(seconds) interval(seconds)
```

### Single Message Test
Send just one message to verify connectivity:
```bash
python3 test_udp_sender.py single
```

---

## What You Should See

### 1. **Network Status** (top-left overlay)
- Green "UDP: Connected (Port 9999)" = working
- Red "UDP: Disconnected" = network issue

### 2. **Trajectory Visualization**
- Green curved line showing ball path
- Fades out after 2 seconds
- Gradient from bright to dim

### 3. **Collision Markers**
- Yellow circles at collision points
- Different shapes for ball/cushion/pocket collisions
- Fade out after 2 seconds

### 4. **Ghost Ball**
- White circle outline
- Shows number "8"
- Indicates aiming target

### 5. **Aim Line**
- White dashed line
- Rotates slowly in test mode
- Shows direction of shot

### 6. **FPS Counter** (top-left)
- Should show 60 FPS
- Lower FPS indicates performance issues

---

## Calibration Testing

### Enter Calibration Mode
Press **C** key

You should see:
- Green calibration grid overlay
- 4 corner markers (circles)
- Selected corner highlighted in yellow
- Help text in bottom-left
- Corner coordinates in top-right

### Test Calibration Controls
1. **Arrow Keys**: Move selected corner
2. **Shift+Arrow**: Move faster
3. **Tab**: Select next corner
4. **Mouse**: Click to select, drag to move
5. **R**: Reset to defaults
6. **C**: Save and exit

The grid should deform as you adjust corners, showing the perspective transformation.

---

## Troubleshooting

### "UDP: Disconnected" but test sender is running
- Check firewall is not blocking port 9999
- Verify sender and receiver are on same machine/network
- Try running both as the same user

### No trajectory visible
- Press **E** to clear any error messages
- Check console/logs for errors
- Verify test sender is sending messages (look for output)

### Projector won't start
```bash
# Check LÖVE2D is installed
love --version

# Check for Lua syntax errors
lua -l main.lua

# View error log
cat /tmp/projector.log
```

### Performance issues (low FPS)
- Reduce MSAA in conf.lua (try msaa = 0)
- Disable vsync (vsync = 0)
- Check GPU utilization

### Calibration not saving
- Check write permissions on projector directory
- Look for `calibration.json` file creation
- Check console for save errors

---

## Expected Output

### Test Sender Output
```
Sending UDP test messages to 127.0.0.1:9999
Duration: 30s, Interval: 0.5s
Press Ctrl+C to stop

Sent trajectory message #1
Sent aim line message #2
Sent trajectory message #3
Sent aim line message #4
Sent collision message #5
...
```

### Projector Console Output
```
Billiards Projector Starting...
Renderer initialized
Calibration loaded from calibration.json
UDP socket listening on port 9999
Loaded module: trajectory (priority: 100)
Loaded module: calibration_ui (priority: 500)
Loaded 2 modules
Billiards Projector Ready
```

---

## Advanced Testing

### Test Custom Messages

Create a custom test message:
```python
import socket, json, time

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

msg = {
    "type": "trajectory",
    "timestamp": time.time(),
    "data": {
        "paths": [{
            "points": [
                {"x": 0.1, "y": 0.1},
                {"x": 0.9, "y": 0.9}
            ],
            "ballType": "cue",
            "confidence": 1.0
        }]
    }
}

sock.sendto(json.dumps(msg).encode(), ("127.0.0.1", 9999))
```

### Monitor Network Traffic
```bash
# Watch UDP packets on port 9999
sudo tcpdump -i lo0 udp port 9999 -A
```

---

## Integration with Backend

Once testing is complete, to integrate with the real backend:

1. Add UDP broadcasting to `backend/api/websocket/broadcaster.py`
2. Send trajectory data when calculated
3. Use same message format as test sender
4. Ensure backend and projector are on same network/machine

See `PROJECTOR_DEPLOYMENT.md` for backend integration code examples.
