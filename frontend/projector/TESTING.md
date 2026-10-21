# Testing the Projector Application Locally

All projector tests run on one Mac or Linux machine. The application binds its
test UDP receiver and sender to loopback (`127.0.0.1`); these instructions do
not use remote login or deployment.

## Run the native app and a test stream

Install LÖVE2D and Python 3 as described in [INSTALL.md](INSTALL.md), then run:

```bash
cd frontend/projector
love . conf_windowed.lua
```

In another terminal on the same machine, send a continuous test stream:

```bash
cd frontend/projector
python3 test_udp_sender.py
```

Or start both locally with the helper script:

```bash
./test_local.sh 30 0.5
```

You should see an animated trajectory, collision markers, a rotating aim line,
and a ghost ball. The local UDP status should show port 9999 as connected.

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

### Coordinate contract

Run the hardware-free receiver and geometry contract tests from the repository
root, then run the Python UDP sender checks:

```bash
lua frontend/projector/test_coordinate_contract.lua
python -m pytest frontend/projector/test_udp_sender.py frontend/projector/test_chessboard.py -q
luac -p frontend/projector/core/coordinate_contract.lua
```

The tests cover nested state points, table corners and pockets, bounds, radii,
motion vectors, event payloads, chessboard pixel extents, frame dimensions, and
meter-tagged physical records. Coordinate-bearing packets without their owning
space and identity scale are rejected before they reach callbacks. Physical
meter values are checked separately and are never converted as table or
projector coordinates. UDP payloads do not accept legacy pixel or unscaled
geometry forms.

### Test Custom Messages

Create a custom test message:
```python
import socket, json, time

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

msg = {
    "type": "trajectory",
    "timestamp": time.time(),
    "data": {
        "coordinate_space": "normalized_table",
        "coordinate_scale": [1, 1],
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

## Backend integration status

The projector currently receives only its custom local UDP message format. A
production backend publisher and schema adapter are not implemented. The test
sender verifies the local receiver contract; it is not evidence of live backend
integration.
