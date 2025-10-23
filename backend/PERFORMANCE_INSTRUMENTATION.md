# Performance Instrumentation Guide

This guide explains the performance instrumentation added to the billiards vision system to diagnose and optimize realtime performance.

## 🎯 Quick Start

### 1. Start Your Backend
```bash
cd /Users/jchadwick/code/billiards-trainer
python backend/main.py
```

### 2. Watch Performance in Real-Time
```bash
# In another terminal
python backend/tools/watch_performance.py
```

This will display a live dashboard showing:
- Current FPS vs target (15 FPS)
- Frame processing time breakdown
- Top bottlenecks with progress bars
- Real-time updates every second

### 3. Check Logs
Logs now include detailed timing for each frame (every 30 frames):
```
=== Performance Summary (last 100 frames) ===
FPS: 12.3 (total: 81.2ms, min: 75.1ms, max: 95.3ms)
Breakdown:
  Preprocessing:    12.5ms
  Masking:          2.3ms
  Table Detection:  18.7ms
  Ball Detection:   38.2ms
    (YOLO only:     28.7ms)
  Cue Detection:    5.8ms
  Tracking:         3.7ms

Top bottlenecks:
  1. ball_detection        : 38.2ms (47.0%)
  2. table_detection       : 18.7ms (23.0%)
  3. preprocessing         : 12.5ms (15.4%)
```

## 📊 What Was Instrumented

### Vision Module (`backend/vision/__init__.py`)
- **Frame-level profiling** - Complete timing for each frame
- **Stage-by-stage breakdown**:
  - Preprocessing (image enhancement)
  - Masking (marker/boundary removal)
  - Table detection
  - Ball detection (YOLO + OpenCV)
  - Tracking
  - Cue detection
  - Result building

### YOLO Detector (`backend/vision/detection/yolo_detector.py`)
- **Inference timing** - Actual model inference time
- **Parsing timing** - Post-processing time
- **Total timing** - End-to-end detection time

### Performance Profiler (`backend/vision/performance_profiler.py`)
- Rolling window statistics (configurable history size)
- Automatic bottleneck identification
- FPS calculation and target comparison
- Aggregate metrics (mean, min, max)

## 🔧 Configuration

Performance profiling is configured in `backend/config.json`:

```json
{
  "vision": {
    "performance": {
      "enable_profiling": true,         // Enable/disable profiling
      "profile_log_interval": 30,       // Log summary every N frames
      "profile_history_size": 100,      // Keep stats for last N frames
      "enable_console_logging": false   // Log to console (default: false, API-only)
    }
  },
  "system": {
    "logging": {
      "level": "INFO",                  // Must be INFO or DEBUG to see perf logs
      "console_logging": true,          // Enable console output
      "log_modules": {
        "vision": "INFO"                // Vision module logging level
      }
    }
  }
}
```

### Disable Profiling
Set `vision.performance.enable_profiling` to `false` in config.json to disable overhead.

### Enable Console Logging
By default, profiler runs in **silent mode** (API-only). To see logs in the backend console:
```json
"vision.performance.enable_console_logging": true
```

### Adjust Logging Frequency
- `profile_log_interval: 30` - Log every 30 frames (default)
- `profile_log_interval: 10` - More frequent logging (every 10 frames)
- `profile_log_interval: 100` - Less frequent logging

**Note**: Console logging only applies when `enable_console_logging: true`. Otherwise, use `watch_performance.py` to monitor.

## 📡 API Endpoints

### Get Real-Time Performance
```bash
curl http://localhost:8000/api/v1/vision/performance
```

Returns:
```json
{
  "profiling_enabled": true,
  "fps": 12.3,
  "target_fps": 15.0,
  "meeting_target": false,
  "frame_time_ms": 81.2,
  "target_frame_time_ms": 66.7,
  "overhead_ms": 14.5,
  "bottlenecks": [
    {"stage": "ball_detection", "time_ms": 38.2},
    {"stage": "table_detection", "time_ms": 18.7}
  ],
  "frame_count": 100,
  "total_frames": 1523,
  "uptime_seconds": 123.4
}
```

### Get Aggregate Summary
```bash
curl http://localhost:8000/api/v1/vision/performance/summary
```

### Get Top Bottlenecks
```bash
curl http://localhost:8000/api/v1/vision/performance/bottlenecks?top_n=5
```

## 🛠️ Tools

### 1. Real-Time Monitor (`watch_performance.py`)
```bash
python backend/tools/watch_performance.py [--url http://localhost:8000] [--interval 1.0]
```
Live dashboard with visual progress bars showing bottlenecks.

### 2. Quick Performance Check (`quick_perf_check.py`)
```bash
python3 -m backend.tools.quick_perf_check
```
One-time check that:
- Verifies device configuration (MPS/CUDA/CPU)
- Tests YOLO inference speed
- Tests full pipeline speed
- Provides recommendations

### 3. Full Diagnostics (`performance_diagnostics.py`)
```bash
python backend/tools/performance_diagnostics.py --frames 100 --output report.json
```
Comprehensive analysis over N frames with JSON output.

## 🔍 Interpreting Results

### Target Performance
- **Target FPS**: 15 FPS
- **Target Frame Time**: 66.7ms

### Common Bottlenecks

1. **Ball Detection (YOLO)**: 28-35ms
   - ✅ **GOOD** if using CoreML/MPS
   - ⚠️ **SLOW** if >50ms (check device config)

2. **Table Detection**: 15-25ms
   - Consider disabling if not needed
   - Or reduce frequency (every N frames)

3. **Preprocessing**: 10-20ms
   - Can be disabled if image quality is good
   - Set `vision.processing.enable_preprocessing: false`

4. **Tracking**: 3-10ms
   - Usually minimal
   - Increases with number of balls

### Quick Optimizations

If **NOT meeting 15 FPS target**:

1. **Reduce queue depth** (immediate impact):
   ```json
   "vision.processing.max_frame_queue_size": 1  // down from 5
   ```
   Saves ~100ms of queue lag

2. **Disable preprocessing** (if image quality is good):
   ```json
   "vision.processing.enable_preprocessing": false
   ```
   Saves ~15ms per frame

3. **Reduce table detection frequency**:
   - Detect table once, cache result
   - Or detect every 10 frames instead of every frame

4. **Verify CoreML is being used**:
   ```bash
   # Should show "CoreML" format and "mps" device
   curl http://localhost:8000/api/v1/vision/detector/info
   ```

## 📈 Performance History

Check recent commits for performance improvements:
- **Before instrumentation**: Unknown bottlenecks, ~200-300ms lag
- **After YOLO verification**: YOLO only takes 28.7ms ✅
- **Next**: Identify and fix remaining bottlenecks

## 🐛 Troubleshooting

### "Performance profiling is disabled"
Set `vision.performance.enable_profiling: true` in config.json

### No logs appearing
1. Check `system.logging.level` is "INFO" or "DEBUG"
2. Check `system.logging.console_logging` is true
3. Check `system.logging.log_modules.vision` is "INFO"

### watch_performance.py fails to connect
1. Make sure backend is running: `python backend/main.py`
2. Check URL is correct: `--url http://localhost:8000`
3. Verify API endpoint: `curl http://localhost:8000/api/v1/vision/performance`

### Profiler shows 0 frames
- Vision module hasn't started yet
- Camera isn't providing frames
- Check integration service is started

## 📝 Example Session

```bash
# Terminal 1: Start backend with profiling
python backend/main.py

# Terminal 2: Watch performance in real-time
python backend/tools/watch_performance.py

# Terminal 3: Check specific metrics
curl http://localhost:8000/api/v1/vision/performance | jq .

# Terminal 4: Run diagnostics
python backend/tools/performance_diagnostics.py --frames 100 --output perf_report.json
```

## 🎓 Understanding the Output

### FPS
- **15+ FPS**: ✅ Meeting target, realtime performance
- **10-15 FPS**: ⚠️ Close, may have occasional lag
- **<10 FPS**: ❌ Too slow, significant lag

### Frame Time
- **<67ms**: ✅ Under budget for 15 FPS
- **67-100ms**: ⚠️ Over budget but may work
- **>100ms**: ❌ Significant delay

### Bottlenecks
Focus optimization efforts on stages taking >20ms or >30% of total time.

## 🚀 Next Steps

1. **Run the monitor** to see current performance
2. **Identify your #1 bottleneck** from the output
3. **Apply targeted optimization** based on findings
4. **Measure improvement** with same tools
5. **Repeat** until meeting 15 FPS target
