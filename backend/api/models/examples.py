"""Example Data for API Documentation.

This module provides comprehensive example data for all API models.
These examples are used in:
- OpenAPI documentation
- Client SDK generation
- Testing and development
- User guides and tutorials

All examples include realistic data that demonstrates proper usage patterns.
"""

from typing import Any

# Example data collections organized by model category
EXAMPLES = {}


# =============================================================================
# Request Examples
# =============================================================================

EXAMPLES["requests"] = {
    "LoginRequest": {
        "description": "User authentication request",
        "examples": {
            "basic_login": {
                "summary": "Basic user login",
                "value": {
                    "username": "operator1",
                    "password": "secure_password123",
                    "remember_me": False,
                },
            },
            "admin_login": {
                "summary": "Administrator login with extended session",
                "value": {
                    "username": "admin",
                    "password": "admin_secure_pass456",
                    "remember_me": True,
                },
            },
        },
    },
    "ConfigUpdateRequest": {
        "description": "System configuration update request",
        "examples": {
            "camera_config": {
                "summary": "Update camera settings",
                "value": {
                    "config_section": "camera",
                    "config_data": {
                        "camera": {
                            "device_id": 0,
                            "resolution": [1920, 1080],
                            "fps": 30,
                            "exposure": "auto",
                            "brightness": 0.5,
                            "contrast": 0.6,
                        }
                    },
                    "validate_only": False,
                    "force_update": False,
                    "client_timestamp": "2024-01-15T10:30:00Z",
                },
            },
            "vision_config": {
                "summary": "Update vision processing settings",
                "value": {
                    "config_section": "vision",
                    "config_data": {
                        "vision": {
                            "ball_detection_sensitivity": 0.85,
                            "min_ball_radius": 15,
                            "max_ball_radius": 35,
                            "cue_detection_threshold": 0.75,
                            "tracking_enabled": True,
                            "color_profiles": {
                                "table_green": {
                                    "hue_range": [40, 80],
                                    "saturation_range": [50, 255],
                                    "value_range": [20, 200],
                                }
                            },
                        }
                    },
                    "validate_only": True,
                    "force_update": False,
                },
            },
            "full_config": {
                "summary": "Complete system configuration update",
                "value": {
                    "config_data": {
                        "system": {
                            "debug": False,
                            "log_level": "INFO",
                            "performance_mode": "balanced",
                        },
                        "camera": {
                            "device_id": 0,
                            "resolution": [1920, 1080],
                            "fps": 30,
                        },
                        "vision": {
                            "ball_detection_sensitivity": 0.8,
                            "tracking_enabled": True,
                        },
                        "network": {
                            "host": "0.0.0.0",
                            "port": 8000,
                            "cors_origins": ["*"],
                            "max_connections": 100,
                        },
                    },
                    "validate_only": False,
                    "force_update": False,
                },
            },
        },
    },
    "CalibrationStartRequest": {
        "description": "Calibration sequence initialization",
        "examples": {
            "standard_calibration": {
                "summary": "Standard calibration procedure",
                "value": {
                    "calibration_type": "standard",
                    "force_restart": False,
                    "timeout_seconds": 300,
                    "client_timestamp": "2024-01-15T10:30:00Z",
                },
            },
            "quick_calibration": {
                "summary": "Quick calibration for minor adjustments",
                "value": {
                    "calibration_type": "quick",
                    "force_restart": True,
                    "timeout_seconds": 120,
                },
            },
            "advanced_calibration": {
                "summary": "Advanced calibration with custom parameters",
                "value": {
                    "calibration_type": "advanced",
                    "force_restart": False,
                    "timeout_seconds": 600,
                },
            },
        },
    },
    "CalibrationPointRequest": {
        "description": "Calibration reference point capture",
        "examples": {
            "corner_point": {
                "summary": "Table corner calibration point",
                "value": {
                    "session_id": "cal_session_abc123",
                    "point_id": "bottom_left_corner",
                    "screen_position": [45.0, 850.0],
                    "world_position": [0.0, 0.0],
                    "confidence": 0.98,
                    "client_timestamp": "2024-01-15T10:32:15Z",
                },
            },
            "pocket_point": {
                "summary": "Pocket center calibration point",
                "value": {
                    "session_id": "cal_session_abc123",
                    "point_id": "corner_pocket_1",
                    "screen_position": [82.0, 820.0],
                    "world_position": [0.0635, 0.0635],
                    "confidence": 0.92,
                },
            },
            "table_center": {
                "summary": "Table center reference point",
                "value": {
                    "session_id": "cal_session_abc123",
                    "point_id": "table_center",
                    "screen_position": [960.0, 540.0],
                    "world_position": [1.42, 0.71],
                    "confidence": 0.95,
                },
            },
        },
    },
    "GameStateResetRequest": {
        "description": "Game state initialization request",
        "examples": {
            "practice_mode": {
                "summary": "Initialize practice mode",
                "value": {
                    "game_type": "practice",
                    "preserve_table": True,
                    "custom_setup": None,
                    "client_timestamp": "2024-01-15T10:30:00Z",
                },
            },
            "eight_ball_game": {
                "summary": "Start new 8-ball game",
                "value": {
                    "game_type": "8ball",
                    "preserve_table": True,
                    "custom_setup": {
                        "rack_type": "triangle",
                        "break_ball_position": [0.71, 0.71],
                        "cue_ball_position": [0.71, 0.36],
                    },
                },
            },
            "custom_setup": {
                "summary": "Custom ball arrangement",
                "value": {
                    "game_type": "practice",
                    "preserve_table": False,
                    "custom_setup": {
                        "balls": [
                            {"id": "cue", "position": [0.5, 0.6], "is_cue_ball": True},
                            {"id": "ball_1", "position": [1.5, 0.7], "number": 1},
                            {"id": "ball_8", "position": [2.0, 0.7], "number": 8},
                        ]
                    },
                },
            },
        },
    },
    "BallPositionUpdateRequest": {
        "description": "Manual ball position adjustment",
        "examples": {
            "single_ball_update": {
                "summary": "Update single ball position",
                "value": {
                    "ball_updates": [
                        {
                            "id": "ball_8",
                            "position": [1.8, 0.8],
                            "velocity": [0.0, 0.0],
                            "is_pocketed": False,
                        }
                    ],
                    "validate_positions": True,
                    "check_collisions": True,
                    "client_timestamp": "2024-01-15T10:35:00Z",
                },
            },
            "multiple_ball_update": {
                "summary": "Update multiple ball positions",
                "value": {
                    "ball_updates": [
                        {
                            "id": "cue",
                            "position": [0.6, 0.7],
                            "velocity": [0.0, 0.0],
                            "is_pocketed": False,
                        },
                        {
                            "id": "ball_1",
                            "position": [1.4, 0.7],
                            "velocity": [0.0, 0.0],
                            "is_pocketed": False,
                        },
                        {
                            "id": "ball_2",
                            "position": [1.6, 0.7],
                            "velocity": [0.0, 0.0],
                            "is_pocketed": False,
                        },
                    ],
                    "validate_positions": True,
                    "check_collisions": True,
                },
            },
        },
    },
    "WebSocketSubscribeRequest": {
        "description": "WebSocket stream subscription",
        "examples": {
            "basic_streams": {
                "summary": "Subscribe to basic data streams",
                "value": {
                    "streams": ["frames", "state"],
                    "quality": "high",
                    "frame_rate": 30,
                    "filters": {},
                },
            },
            "all_streams": {
                "summary": "Subscribe to all available streams",
                "value": {
                    "streams": ["frames", "state", "trajectories", "events", "metrics"],
                    "quality": "medium",
                    "frame_rate": 15,
                    "filters": {
                        "min_confidence": 0.8,
                        "ball_ids": ["cue", "ball_1", "ball_8"],
                    },
                },
            },
            "low_bandwidth": {
                "summary": "Low bandwidth subscription",
                "value": {
                    "streams": ["state", "events"],
                    "quality": "low",
                    "frame_rate": 5,
                    "filters": {"include_moving_only": True},
                },
            },
        },
    },
}


# =============================================================================
# Response Examples
# =============================================================================

EXAMPLES["responses"] = {
    "HealthResponse": {
        "description": "System health status information",
        "examples": {
            "healthy_system": {
                "summary": "All systems operational",
                "value": {
                    "status": "healthy",
                    "timestamp": "2024-01-15T10:30:00Z",
                    "uptime": 86400.5,
                    "version": "1.0.0",
                    "components": {
                        "vision": {
                            "name": "vision",
                            "status": "healthy",
                            "message": "Vision processing operational at 30 FPS",
                            "last_check": "2024-01-15T10:30:00Z",
                            "uptime": 86400.0,
                            "errors": [],
                        },
                        "camera": {
                            "name": "camera",
                            "status": "healthy",
                            "message": "Camera connected and streaming",
                            "last_check": "2024-01-15T10:30:00Z",
                            "uptime": 86350.0,
                            "errors": [],
                        },
                        "projector": {
                            "name": "projector",
                            "status": "healthy",
                            "message": "Projector calibrated and ready",
                            "last_check": "2024-01-15T10:30:00Z",
                            "uptime": 85000.0,
                            "errors": [],
                        },
                    },
                    "metrics": {
                        "cpu_usage": 45.2,
                        "memory_usage": 68.5,
                        "disk_usage": 34.1,
                        "network_io": {
                            "bytes_sent": 15728640,
                            "bytes_received": 31457280,
                        },
                        "api_requests_per_second": 12.3,
                        "websocket_connections": 3,
                        "average_response_time": 25.4,
                    },
                },
            },
            "degraded_system": {
                "summary": "System with performance issues",
                "value": {
                    "status": "degraded",
                    "timestamp": "2024-01-15T10:30:00Z",
                    "uptime": 3600.0,
                    "version": "1.0.0",
                    "components": {
                        "vision": {
                            "name": "vision",
                            "status": "degraded",
                            "message": "Processing slower than expected",
                            "last_check": "2024-01-15T10:30:00Z",
                            "uptime": 3600.0,
                            "errors": ["High processing latency detected"],
                        },
                        "camera": {
                            "name": "camera",
                            "status": "healthy",
                            "message": "Camera operational",
                            "last_check": "2024-01-15T10:30:00Z",
                            "uptime": 3600.0,
                            "errors": [],
                        },
                    },
                    "metrics": {
                        "cpu_usage": 95.8,
                        "memory_usage": 87.2,
                        "disk_usage": 34.1,
                        "network_io": {
                            "bytes_sent": 5242880,
                            "bytes_received": 10485760,
                        },
                        "api_requests_per_second": 45.7,
                        "websocket_connections": 8,
                        "average_response_time": 156.3,
                    },
                },
            },
        },
    },
    "GameStateResponse": {
        "description": "Current game state information",
        "examples": {
            "practice_game": {
                "summary": "Practice mode with scattered balls",
                "value": {
                    "timestamp": "2024-01-15T10:30:00Z",
                    "frame_number": 15647,
                    "balls": [
                        {
                            "id": "cue",
                            "number": None,
                            "position": [0.71, 0.6],
                            "velocity": [0.0, 0.0],
                            "is_cue_ball": True,
                            "is_pocketed": False,
                            "confidence": 0.98,
                            "last_update": "2024-01-15T10:30:00Z",
                        },
                        {
                            "id": "ball_1",
                            "number": 1,
                            "position": [1.2, 0.4],
                            "velocity": [0.0, 0.0],
                            "is_cue_ball": False,
                            "is_pocketed": False,
                            "confidence": 0.95,
                            "last_update": "2024-01-15T10:30:00Z",
                        },
                        {
                            "id": "ball_8",
                            "number": 8,
                            "position": [1.8, 0.8],
                            "velocity": [0.0, 0.0],
                            "is_cue_ball": False,
                            "is_pocketed": False,
                            "confidence": 0.92,
                            "last_update": "2024-01-15T10:30:00Z",
                        },
                    ],
                    "cue": {
                        "tip_position": [0.5, 0.6],
                        "angle": 45.5,
                        "elevation": 5.2,
                        "estimated_force": 12.3,
                        "is_visible": True,
                        "confidence": 0.87,
                    },
                    "table": {
                        "width": 2.84,
                        "height": 1.42,
                        "pocket_positions": [
                            [0, 0],
                            [1.42, 0],
                            [2.84, 0],
                            [0, 1.42],
                            [1.42, 1.42],
                            [2.84, 1.42],
                        ],
                        "pocket_radius": 0.0635,
                        "surface_friction": 0.2,
                    },
                    "game_type": "practice",
                    "is_valid": True,
                    "confidence": 0.94,
                    "events": [
                        {
                            "timestamp": "2024-01-15T10:29:45Z",
                            "event_type": "shot",
                            "description": "Player attempted bank shot",
                            "data": {
                                "success": False,
                                "ball_contacted": "ball_1",
                                "intended_target": "ball_8",
                            },
                        }
                    ],
                },
            },
            "eight_ball_break": {
                "summary": "8-ball game after break shot",
                "value": {
                    "timestamp": "2024-01-15T10:35:00Z",
                    "frame_number": 18234,
                    "balls": [
                        {
                            "id": "cue",
                            "number": None,
                            "position": [0.71, 0.71],
                            "velocity": [0.0, 0.0],
                            "is_cue_ball": True,
                            "is_pocketed": False,
                            "confidence": 0.99,
                            "last_update": "2024-01-15T10:35:00Z",
                        },
                        {
                            "id": "ball_9",
                            "number": 9,
                            "position": [0.0, 0.0],
                            "velocity": [0.0, 0.0],
                            "is_cue_ball": False,
                            "is_pocketed": True,
                            "confidence": 0.0,
                            "last_update": "2024-01-15T10:34:55Z",
                        },
                    ],
                    "cue": {
                        "tip_position": [0.4, 0.71],
                        "angle": 90.0,
                        "elevation": 0.0,
                        "estimated_force": 0.0,
                        "is_visible": True,
                        "confidence": 0.92,
                    },
                    "table": {
                        "width": 2.84,
                        "height": 1.42,
                        "pocket_positions": [
                            [0, 0],
                            [1.42, 0],
                            [2.84, 0],
                            [0, 1.42],
                            [1.42, 1.42],
                            [2.84, 1.42],
                        ],
                        "pocket_radius": 0.0635,
                        "surface_friction": 0.2,
                    },
                    "game_type": "8ball",
                    "is_valid": True,
                    "confidence": 0.96,
                    "events": [
                        {
                            "timestamp": "2024-01-15T10:34:52Z",
                            "event_type": "break",
                            "description": "Break shot executed",
                            "data": {
                                "balls_scattered": 14,
                                "balls_pocketed": ["ball_9"],
                                "valid_break": True,
                            },
                        },
                        {
                            "timestamp": "2024-01-15T10:34:55Z",
                            "event_type": "pocket",
                            "description": "Ball 9 pocketed in corner pocket",
                            "data": {"ball_id": "ball_9", "pocket_id": 0},
                        },
                    ],
                },
            },
        },
    },
    "ErrorResponse": {
        "description": "Error response format",
        "examples": {
            "validation_error": {
                "summary": "Request validation failed",
                "value": {
                    "error": "validation error",
                    "message": "Invalid configuration parameter",
                    "code": "VAL_001",
                    "details": {
                        "field": "camera.fps",
                        "value": -30,
                        "expected": "positive integer",
                    },
                    "timestamp": "2024-01-15T10:30:00Z",
                    "request_id": "req_abc123def456",
                },
            },
            "authentication_error": {
                "summary": "Authentication failed",
                "value": {
                    "error": "authentication failed",
                    "message": "Invalid credentials provided",
                    "code": "AUTH_001",
                    "details": {"username": "attempted_user"},
                    "timestamp": "2024-01-15T10:30:00Z",
                    "request_id": "req_xyz789abc123",
                },
            },
            "system_error": {
                "summary": "Internal system error",
                "value": {
                    "error": "internal server error",
                    "message": "Camera device not available",
                    "code": "HW_001",
                    "details": {
                        "component": "camera",
                        "device_id": 0,
                        "last_working": "2024-01-15T09:45:00Z",
                    },
                    "timestamp": "2024-01-15T10:30:00Z",
                    "request_id": "req_error_456",
                },
            },
        },
    },
}


# =============================================================================
# WebSocket Message Examples
# =============================================================================

EXAMPLES["websocket"] = {
    "FrameMessage": {
        "description": "Video frame data stream",
        "examples": {
            "high_quality_frame": {
                "summary": "High quality video frame",
                "value": {
                    "type": "frame",
                    "timestamp": "2024-01-15T10:30:00.123Z",
                    "sequence": 15647,
                    "frame_number": 15647,
                    "processing_time_ms": 23.5,
                    "data": {
                        "image_data": "/9j/4AAQSkZJRgABAQEAYABgAAD...",
                        "metadata": {
                            "width": 1920,
                            "height": 1080,
                            "fps": 30.0,
                            "quality": "high",
                            "format": "jpeg",
                            "compression_level": 0.85,
                            "encoding_time_ms": 8.2,
                        },
                    },
                    "annotations": {
                        "detected_balls": 12,
                        "detected_cue": True,
                        "processing_quality": "excellent",
                    },
                },
            },
            "low_quality_frame": {
                "summary": "Low quality video frame for bandwidth conservation",
                "value": {
                    "type": "frame",
                    "timestamp": "2024-01-15T10:30:01.156Z",
                    "sequence": 15648,
                    "frame_number": 15648,
                    "processing_time_ms": 12.1,
                    "data": {
                        "image_data": "/9j/4AAQSkZJRgABAQEAYABgAAD...",
                        "metadata": {
                            "width": 640,
                            "height": 480,
                            "fps": 15.0,
                            "quality": "low",
                            "format": "jpeg",
                            "compression_level": 0.6,
                            "encoding_time_ms": 3.8,
                        },
                    },
                },
            },
        },
    },
    "StateMessage": {
        "description": "Game state update stream",
        "examples": {
            "active_game_state": {
                "summary": "Game state during active play",
                "value": {
                    "type": "state",
                    "timestamp": "2024-01-15T10:30:00.234Z",
                    "sequence": 8234,
                    "processing_time_ms": 5.7,
                    "data": {
                        "frame_number": 15647,
                        "balls": [
                            {
                                "id": "cue",
                                "number": None,
                                "position": [0.71, 0.6],
                                "velocity": [1.2, -0.8],
                                "radius": 0.028575,
                                "is_cue_ball": True,
                                "is_pocketed": False,
                                "confidence": 0.98,
                            },
                            {
                                "id": "ball_1",
                                "number": 1,
                                "position": [1.4, 0.7],
                                "velocity": [0.0, 0.0],
                                "radius": 0.028575,
                                "is_cue_ball": False,
                                "is_pocketed": False,
                                "confidence": 0.95,
                            },
                        ],
                        "cue": {
                            "tip_position": [0.5, 0.6],
                            "angle": 45.0,
                            "elevation": 5.0,
                            "estimated_force": 15.2,
                            "is_visible": True,
                            "confidence": 0.89,
                        },
                        "table": {
                            "width": 2.84,
                            "height": 1.42,
                            "pocket_positions": [
                                [0, 0],
                                [1.42, 0],
                                [2.84, 0],
                                [0, 1.42],
                                [1.42, 1.42],
                                [2.84, 1.42],
                            ],
                            "pocket_radius": 0.0635,
                        },
                        "game_type": "practice",
                        "is_valid": True,
                        "confidence": 0.94,
                    },
                    "changes": ["cue_ball_velocity", "cue_position"],
                },
            }
        },
    },
    "AlertMessage": {
        "description": "System alert notifications",
        "examples": {
            "info_alert": {
                "summary": "Informational alert",
                "value": {
                    "type": "alert",
                    "timestamp": "2024-01-15T10:30:00Z",
                    "sequence": 1001,
                    "priority": "normal",
                    "data": {
                        "level": "info",
                        "message": "Calibration session completed successfully",
                        "code": "CAL_200",
                        "component": "calibration",
                        "details": {
                            "session_id": "cal_session_abc123",
                            "accuracy": 0.97,
                            "points_captured": 12,
                        },
                        "auto_dismiss": True,
                        "dismiss_timeout": 30,
                        "actions": ["dismiss"],
                    },
                },
            },
            "warning_alert": {
                "summary": "System warning",
                "value": {
                    "type": "alert",
                    "timestamp": "2024-01-15T10:30:00Z",
                    "sequence": 1002,
                    "priority": "high",
                    "data": {
                        "level": "warning",
                        "message": "Camera frame rate dropping below optimal",
                        "code": "CAM_201",
                        "component": "camera",
                        "details": {
                            "current_fps": 22.3,
                            "target_fps": 30.0,
                            "duration_seconds": 15,
                        },
                        "auto_dismiss": False,
                        "actions": ["check_camera", "restart_camera", "dismiss"],
                    },
                },
            },
            "error_alert": {
                "summary": "System error alert",
                "value": {
                    "type": "alert",
                    "timestamp": "2024-01-15T10:30:00Z",
                    "sequence": 1003,
                    "priority": "critical",
                    "data": {
                        "level": "error",
                        "message": "Vision processing has stopped",
                        "code": "PROC_001",
                        "component": "vision",
                        "details": {
                            "error_type": "processing_timeout",
                            "last_successful_frame": 15645,
                            "timeout_duration": 5.0,
                        },
                        "auto_dismiss": False,
                        "actions": ["restart_vision", "check_logs", "contact_support"],
                    },
                },
            },
        },
    },
    "TrajectoryMessage": {
        "description": "Ball trajectory predictions",
        "examples": {
            "simple_trajectory": {
                "summary": "Simple straight-line trajectory",
                "value": {
                    "type": "trajectory",
                    "timestamp": "2024-01-15T10:30:00.345Z",
                    "sequence": 5678,
                    "processing_time_ms": 12.4,
                    "trajectories": [
                        {
                            "ball_id": "cue",
                            "points": [
                                {
                                    "position": [0.71, 0.6],
                                    "time": 0.0,
                                    "velocity": [1.2, -0.3],
                                },
                                {
                                    "position": [1.0, 0.52],
                                    "time": 0.25,
                                    "velocity": [1.1, -0.28],
                                },
                                {
                                    "position": [1.4, 0.42],
                                    "time": 0.6,
                                    "velocity": [0.8, -0.2],
                                },
                            ],
                            "collisions": [
                                {
                                    "position": [1.42, 0.41],
                                    "time": 0.62,
                                    "type": "ball",
                                    "ball1_id": "cue",
                                    "ball2_id": "ball_1",
                                    "impact_angle": 25.0,
                                    "confidence": 0.89,
                                }
                            ],
                            "will_be_pocketed": False,
                            "pocket_id": None,
                            "time_to_rest": 2.1,
                            "max_velocity": 1.2,
                            "confidence": 0.91,
                        }
                    ],
                    "shot_analysis": {
                        "shot_type": "direct",
                        "difficulty": 0.3,
                        "success_probability": 0.85,
                        "recommended_adjustments": [],
                    },
                },
            }
        },
    },
}


# =============================================================================
# Common Model Examples
# =============================================================================

EXAMPLES["common"] = {
    "Coordinate2D": {
        "description": "2D coordinate representation",
        "examples": {
            "table_center": {
                "summary": "Table center coordinates",
                "value": {"x": 1.42, "y": 0.71},
            },
            "pocket_position": {
                "summary": "Corner pocket position",
                "value": {"x": 0.0, "y": 0.0},
            },
        },
    },
    "PaginationRequest": {
        "description": "Pagination parameters",
        "examples": {
            "first_page": {
                "summary": "First page of results",
                "value": {
                    "page": 1,
                    "size": 20,
                    "sort_by": "timestamp",
                    "sort_order": "desc",
                },
            },
            "large_page": {
                "summary": "Large page size for bulk operations",
                "value": {
                    "page": 1,
                    "size": 100,
                    "sort_by": "frame_number",
                    "sort_order": "asc",
                },
            },
        },
    },
    "TimeRange": {
        "description": "Time range specification",
        "examples": {
            "last_hour": {
                "summary": "Data from the last hour",
                "value": {
                    "start_time": "2024-01-15T09:30:00Z",
                    "end_time": "2024-01-15T10:30:00Z",
                },
            },
            "game_session": {
                "summary": "Complete game session timeframe",
                "value": {
                    "start_time": "2024-01-15T10:00:00Z",
                    "end_time": "2024-01-15T10:45:00Z",
                },
            },
        },
    },
}


def get_example_by_path(
    category: str, model_name: str, example_name: str = None
) -> dict[str, Any]:
    """Get a specific example by category, model, and example name."""
    if category not in EXAMPLES:
        raise ValueError(f"Unknown category: {category}")

    category_examples = EXAMPLES[category]
    if model_name not in category_examples:
        raise ValueError(f"Unknown model: {model_name} in category: {category}")

    model_examples = category_examples[model_name]

    if example_name is None:
        # Return the first example
        examples = model_examples.get("examples", {})
        if examples:
            return list(examples.values())[0]["value"]
        return {}

    examples = model_examples.get("examples", {})
    if example_name not in examples:
        raise ValueError(f"Unknown example: {example_name} for model: {model_name}")

    return examples[example_name]["value"]


def get_all_examples_for_model(category: str, model_name: str) -> dict[str, Any]:
    """Get all examples for a specific model."""
    if category not in EXAMPLES:
        raise ValueError(f"Unknown category: {category}")

    category_examples = EXAMPLES[category]
    if model_name not in category_examples:
        raise ValueError(f"Unknown model: {model_name} in category: {category}")

    return category_examples[model_name]


def list_available_examples() -> dict[str, list[str]]:
    """List all available examples by category."""
    available = {}
    for category, models in EXAMPLES.items():
        available[category] = list(models.keys())
    return available


def export_examples_to_json(filename: str = "api_examples.json") -> str:
    """Export all examples to a JSON file."""
    import json
    from datetime import datetime

    export_data = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "description": "Billiards Trainer API Examples",
            "version": "1.0.0",
        },
        "examples": EXAMPLES,
    }

    with open(filename, "w") as f:
        json.dump(export_data, f, indent=2, default=str)

    return filename


def generate_openapi_examples() -> dict[str, Any]:
    """Generate OpenAPI-compatible examples."""
    openapi_examples = {}

    for _category, models in EXAMPLES.items():
        for model_name, model_data in models.items():
            if "examples" in model_data:
                openapi_examples[model_name] = {}
                for example_name, example_data in model_data["examples"].items():
                    openapi_examples[model_name][example_name] = {
                        "summary": example_data["summary"],
                        "value": example_data["value"],
                    }

    return openapi_examples


if __name__ == "__main__":
    # Export examples when run as script
    filename = export_examples_to_json()
    print(f"Examples exported to {filename}")

    # Print summary
    available = list_available_examples()
    total_examples = sum(
        len(EXAMPLES[cat][model].get("examples", {}))
        for cat in EXAMPLES
        for model in EXAMPLES[cat]
    )

    print(f"\nTotal examples: {total_examples}")
    print("\nAvailable by category:")
    for category, models in available.items():
        print(f"  {category}: {len(models)} models")
        for model in models:
            example_count = len(EXAMPLES[category][model].get("examples", {}))
            print(f"    {model}: {example_count} examples")
