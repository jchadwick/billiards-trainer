# Billiards Trainer API Reference

## Overview

The Billiards Trainer API provides RESTful endpoints and WebSocket connections for interacting with the billiards training system. This document provides comprehensive API documentation including endpoints, request/response formats, and usage examples.

## Base Configuration

- **Base URL**: `http://localhost:8000` (development) or your production domain
- **API Version**: v1
- **Content Type**: `application/json`
- **WebSocket URL**: `ws://localhost:8000/ws`

## Authentication

Currently, the API supports development mode without authentication. In production, implement appropriate authentication mechanisms:

```http
Authorization: Bearer <your-access-token>
```

## Health and System Status

### Health Check
Get basic system health status.

```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "service": "billiards-trainer",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### System Status
Get detailed system status including all modules.

```http
GET /api/v1/system/status
```

**Response:**
```json
{
  "system": {
    "state": "running",
    "uptime": 3600.5,
    "startup_time": 1642248600.0
  },
  "modules": {
    "core": {
      "state": "running",
      "health": "healthy",
      "startup_time": 2.1,
      "restart_count": 0,
      "metrics": {
        "total_updates": 1500,
        "avg_update_time": 0.025
      }
    },
    "vision": {
      "state": "running",
      "health": "healthy",
      "startup_time": 3.2,
      "restart_count": 0,
      "metrics": {
        "fps": 28.5,
        "frames_processed": 45000,
        "frames_dropped": 12
      }
    }
  },
  "health": {
    "overall_status": "healthy",
    "performance_score": 0.95
  },
  "performance": {
    "cpu_percent": 45.2,
    "memory_percent": 62.1,
    "disk_percent": 35.7
  },
  "alerts": []
}
```

### Performance Metrics
Get real-time performance metrics.

```http
GET /api/v1/system/metrics
```

**Response:**
```json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "system": {
    "cpu_percent": 45.2,
    "memory_total": 8589934592,
    "memory_used": 5340405555,
    "memory_percent": 62.1,
    "disk_percent": 35.7,
    "network_sent": 1234567890,
    "network_received": 9876543210
  },
  "application": {
    "vision_fps": 28.5,
    "vision_latency": 12.3,
    "core_update_time": 0.025,
    "api_requests_per_second": 45.2,
    "active_connections": 8
  }
}
```

## Configuration Management

### Get Configuration
Retrieve current system configuration.

```http
GET /api/v1/config
```

**Response:**
```json
{
  "system": {
    "environment": "development",
    "debug_mode": true,
    "log_level": "INFO"
  },
  "vision": {
    "camera_device_id": 0,
    "target_fps": 30,
    "enable_tracking": true
  },
  "core": {
    "physics_enabled": true,
    "prediction_enabled": true,
    "assistance_enabled": true
  },
  "api": {
    "host": "0.0.0.0",
    "port": 8000,
    "cors_origins": ["*"]
  }
}
```

### Update Configuration
Update system configuration.

```http
PUT /api/v1/config
Content-Type: application/json

{
  "vision": {
    "target_fps": 60,
    "enable_tracking": true
  },
  "core": {
    "debug_mode": false
  }
}
```

**Response:**
```json
{
  "status": "success",
  "message": "Configuration updated successfully",
  "updated_fields": ["vision.target_fps", "core.debug_mode"]
}
```

### Reload Configuration
Reload configuration from files.

```http
POST /api/v1/config/reload
```

**Response:**
```json
{
  "status": "success",
  "message": "Configuration reloaded successfully"
}
```

## Game Operations

### Get Game State
Get current game state.

```http
GET /api/v1/game/state
```

**Response:**
```json
{
  "timestamp": 1642248600.0,
  "game_type": "practice",
  "balls": [
    {
      "id": "cue",
      "position": {"x": 100.5, "y": 200.3},
      "velocity": {"x": 0.0, "y": 0.0},
      "radius": 28.5,
      "is_cue_ball": true,
      "is_pocketed": false,
      "number": 0
    },
    {
      "id": "ball_1",
      "position": {"x": 300.2, "y": 150.7},
      "velocity": {"x": 0.0, "y": 0.0},
      "radius": 28.5,
      "is_cue_ball": false,
      "is_pocketed": false,
      "number": 1
    }
  ],
  "table": {
    "width": 2540,
    "height": 1270,
    "corners": [
      {"x": 0, "y": 0},
      {"x": 2540, "y": 0},
      {"x": 2540, "y": 1270},
      {"x": 0, "y": 1270}
    ],
    "pockets": [
      {"x": 0, "y": 0, "type": "corner"},
      {"x": 1270, "y": 0, "type": "side"},
      {"x": 2540, "y": 0, "type": "corner"}
    ]
  }
}
```

### Reset Game
Reset game to initial state.

```http
POST /api/v1/game/reset
Content-Type: application/json

{
  "game_type": "practice"
}
```

**Response:**
```json
{
  "status": "success",
  "message": "Game reset successfully",
  "game_type": "practice"
}
```

### Analyze Shot
Analyze current shot setup.

```http
POST /api/v1/game/analyze
Content-Type: application/json

{
  "target_ball": "ball_1",
  "include_alternatives": true
}
```

**Response:**
```json
{
  "analysis": {
    "shot_type": "straight",
    "difficulty": 0.3,
    "success_probability": 0.85,
    "recommended_force": 0.6,
    "recommended_angle": 15.5,
    "contact_point": {"x": 285.2, "y": 142.1}
  },
  "trajectory": [
    {"x": 100.5, "y": 200.3},
    {"x": 150.2, "y": 180.1},
    {"x": 200.1, "y": 160.5},
    {"x": 285.2, "y": 142.1}
  ],
  "alternative_shots": [
    {
      "shot_type": "bank",
      "difficulty": 0.7,
      "success_probability": 0.45
    }
  ]
}
```

### Calculate Trajectory
Calculate ball trajectory for given parameters.

```http
POST /api/v1/game/trajectory
Content-Type: application/json

{
  "ball_id": "cue",
  "initial_velocity": {"x": 50.0, "y": 30.0},
  "time_limit": 5.0
}
```

**Response:**
```json
{
  "trajectory": [
    {"x": 100.5, "y": 200.3, "timestamp": 0.0},
    {"x": 150.2, "y": 230.3, "timestamp": 1.0},
    {"x": 195.1, "y": 257.8, "timestamp": 2.0}
  ],
  "final_position": {"x": 350.2, "y": 380.1},
  "total_time": 4.2,
  "collisions": [
    {
      "timestamp": 2.5,
      "position": {"x": 220.3, "y": 275.1},
      "object_type": "ball",
      "object_id": "ball_3"
    }
  ]
}
```

### Suggest Shots
Get shot suggestions based on current state.

```http
POST /api/v1/game/suggest
Content-Type: application/json

{
  "difficulty_filter": 0.5,
  "shot_type_filter": "straight",
  "max_suggestions": 3
}
```

**Response:**
```json
{
  "suggestions": [
    {
      "target_ball": "ball_1",
      "shot_type": "straight",
      "difficulty": 0.3,
      "success_probability": 0.85,
      "description": "Easy straight shot to corner pocket"
    },
    {
      "target_ball": "ball_5",
      "shot_type": "cut",
      "difficulty": 0.4,
      "success_probability": 0.72,
      "description": "Medium cut shot to side pocket"
    }
  ]
}
```

## Calibration

### Camera Calibration
Perform camera calibration.

```http
POST /api/v1/calibration/camera
```

**Response:**
```json
{
  "status": "success",
  "message": "Camera calibration completed",
  "calibration_data": {
    "camera_matrix": [[1000, 0, 320], [0, 1000, 240], [0, 0, 1]],
    "distortion_coefficients": [0.1, -0.2, 0.0, 0.0, 0.0],
    "reprojection_error": 0.5
  }
}
```

### Color Calibration
Perform automatic color calibration.

```http
POST /api/v1/calibration/colors
Content-Type: application/json

{
  "use_current_frame": true
}
```

**Response:**
```json
{
  "status": "success",
  "message": "Color calibration completed",
  "color_ranges": {
    "table_surface": {
      "hue": [40, 80],
      "saturation": [50, 255],
      "value": [50, 200]
    },
    "cue_ball": {
      "hue": [0, 180],
      "saturation": [0, 30],
      "value": [200, 255]
    }
  }
}
```

### Calibration Status
Get current calibration status.

```http
GET /api/v1/calibration/status
```

**Response:**
```json
{
  "camera_calibrated": true,
  "color_calibrated": true,
  "geometry_calibrated": false,
  "last_calibration": "2024-01-15T09:15:00Z",
  "calibration_quality": {
    "camera": "good",
    "colors": "excellent",
    "geometry": "not_calibrated"
  }
}
```

## Vision System

### Start Vision Capture
Start camera capture and processing.

```http
POST /api/v1/vision/start
```

**Response:**
```json
{
  "status": "success",
  "message": "Vision capture started"
}
```

### Stop Vision Capture
Stop camera capture and processing.

```http
POST /api/v1/vision/stop
```

**Response:**
```json
{
  "status": "success",
  "message": "Vision capture stopped"
}
```

### Vision Statistics
Get vision processing statistics.

```http
GET /api/v1/vision/stats
```

**Response:**
```json
{
  "is_running": true,
  "frames_processed": 45000,
  "frames_dropped": 12,
  "avg_fps": 28.5,
  "avg_processing_time_ms": 12.3,
  "detection_accuracy": {
    "table": 0.95,
    "balls": 0.89,
    "cue": 0.76
  },
  "uptime_seconds": 3600.5,
  "camera_connected": true
}
```

### Set Region of Interest
Set region of interest for vision processing.

```http
POST /api/v1/vision/roi
Content-Type: application/json

{
  "corners": [
    {"x": 100, "y": 50},
    {"x": 900, "y": 50},
    {"x": 900, "y": 650},
    {"x": 100, "y": 650}
  ]
}
```

**Response:**
```json
{
  "status": "success",
  "message": "Region of interest set"
}
```

## WebSocket API

### Connection
Connect to the WebSocket endpoint for real-time updates.

```javascript
const ws = new WebSocket('ws://localhost:8000/ws');

ws.onopen = function(event) {
    console.log('Connected to WebSocket');
};

ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    handleMessage(data);
};
```

### Message Types

#### Subscribe to Updates
```json
{
  "type": "subscribe",
  "channels": ["game_state", "performance", "alerts"]
}
```

#### Game State Updates
```json
{
  "type": "game_state_update",
  "timestamp": 1642248600.0,
  "data": {
    "balls": [...],
    "table": {...}
  }
}
```

#### Performance Metrics
```json
{
  "type": "performance_metrics",
  "timestamp": 1642248600.0,
  "data": {
    "cpu_percent": 45.2,
    "memory_percent": 62.1,
    "vision_fps": 28.5
  }
}
```

#### System Alerts
```json
{
  "type": "system_alert",
  "timestamp": 1642248600.0,
  "data": {
    "alert_type": "warning",
    "source": "vision",
    "message": "Frame rate below threshold",
    "threshold": 15.0,
    "current_value": 12.3
  }
}
```

## Error Handling

### Error Response Format
All API errors follow a consistent format:

```json
{
  "error": {
    "code": "INVALID_REQUEST",
    "message": "The request parameters are invalid",
    "details": {
      "field": "target_ball",
      "reason": "Ball ID not found in current game state"
    },
    "timestamp": "2024-01-15T10:30:00Z"
  }
}
```

### Common Error Codes

- `INVALID_REQUEST`: Request parameters are invalid
- `NOT_FOUND`: Requested resource not found
- `MODULE_UNAVAILABLE`: Required module is not available
- `CALIBRATION_REQUIRED`: Operation requires calibration
- `SYSTEM_ERROR`: Internal system error
- `RATE_LIMITED`: Request rate limit exceeded

### HTTP Status Codes

- `200 OK`: Request successful
- `201 Created`: Resource created successfully
- `400 Bad Request`: Invalid request parameters
- `404 Not Found`: Resource not found
- `429 Too Many Requests`: Rate limit exceeded
- `500 Internal Server Error`: Server error
- `503 Service Unavailable`: Service temporarily unavailable

## Rate Limiting

### Limits
- API endpoints: 60 requests per minute per IP
- WebSocket connections: 5 connections per minute per IP
- Burst allowance: 20 requests for API, 10 for WebSocket

### Rate Limit Headers
```http
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 45
X-RateLimit-Reset: 1642248660
```

## Examples

### Python Client Example
```python
import requests
import json

# Get system status
response = requests.get('http://localhost:8000/api/v1/system/status')
status = response.json()
print(f"System state: {status['system']['state']}")

# Analyze current shot
analysis_request = {
    "target_ball": "ball_1",
    "include_alternatives": True
}
response = requests.post(
    'http://localhost:8000/api/v1/game/analyze',
    json=analysis_request
)
analysis = response.json()
print(f"Shot difficulty: {analysis['analysis']['difficulty']}")
```

### JavaScript WebSocket Example
```javascript
const ws = new WebSocket('ws://localhost:8000/ws');

ws.onopen = function() {
    // Subscribe to game state updates
    ws.send(JSON.stringify({
        type: 'subscribe',
        channels: ['game_state']
    }));
};

ws.onmessage = function(event) {
    const message = JSON.parse(event.data);

    if (message.type === 'game_state_update') {
        updateGameDisplay(message.data);
    }
};

function updateGameDisplay(gameState) {
    // Update UI with new game state
    console.log('Balls:', gameState.balls.length);
}
```

### cURL Examples
```bash
# Get system health
curl http://localhost:8000/health

# Get detailed system status
curl http://localhost:8000/api/v1/system/status

# Reset game
curl -X POST http://localhost:8000/api/v1/game/reset \
  -H "Content-Type: application/json" \
  -d '{"game_type": "practice"}'

# Start vision capture
curl -X POST http://localhost:8000/api/v1/vision/start
```

This API reference provides comprehensive documentation for all available endpoints and WebSocket communications in the Billiards Trainer system.
