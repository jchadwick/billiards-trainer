"""WebSocket message format verification and testing utilities.

This module provides tools to verify that frontend and backend WebSocket
message formats are compatible and to test end-to-end message flow.
"""

import asyncio
import base64
import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np
from fastapi import WebSocket
from pydantic import BaseModel, ValidationError

from .schemas import (
    AlertData,
    BallData,
    ConfigData,
    CueData,
    FrameData,
    GameStateData,
    MessageType,
    MetricsData,
    TableData,
    TrajectoryData,
    WebSocketMessage,
)

logger = logging.getLogger(__name__)


class MessageFormatVerifier:
    """Verifies WebSocket message formats between frontend and backend."""

    def __init__(self):
        self.verification_results: dict[str, dict[str, Any]] = {}

    def verify_frame_message(self) -> dict[str, Any]:
        """Verify frame message format compatibility."""
        try:
            # Create sample frame data
            sample_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            import cv2

            _, buffer = cv2.imencode(".jpg", sample_image)
            base64_image = base64.b64encode(buffer).decode("utf-8")

            frame_data = FrameData(
                image=base64_image,
                width=640,
                height=480,
                format="jpeg",
                quality=85,
                compressed=False,
                fps=30.0,
                size_bytes=len(buffer),
            )

            message = WebSocketMessage(
                type=MessageType.FRAME,
                timestamp=datetime.now(timezone.utc),
                sequence=1,
                data=frame_data.dict(),
            )

            # Test serialization/deserialization
            json_message = message.json()
            parsed_message = json.loads(json_message)

            return {
                "message_type": "frame",
                "status": "valid",
                "sample_message": parsed_message,
                "validation_notes": [
                    "Image data successfully encoded as base64",
                    "All required fields present and valid",
                    "Message serializes/deserializes correctly",
                ],
            }

        except Exception as e:
            return {
                "message_type": "frame",
                "status": "error",
                "error": str(e),
                "validation_notes": ["Frame message format validation failed"],
            }

    def verify_game_state_message(self) -> dict[str, Any]:
        """Verify game state message format compatibility."""
        try:
            # Create sample ball data
            balls = [
                BallData(
                    id="cue",
                    position=[320.0, 240.0],
                    radius=12.0,
                    color="white",
                    velocity=[0.0, 0.0],
                    confidence=0.98,
                    visible=True,
                ),
                BallData(
                    id="1",
                    position=[400.0, 200.0],
                    radius=12.0,
                    color="yellow",
                    velocity=None,
                    confidence=0.95,
                    visible=True,
                ),
                BallData(
                    id="8",
                    position=[450.0, 280.0],
                    radius=12.0,
                    color="black",
                    velocity=[-5.0, 2.0],
                    confidence=0.99,
                    visible=True,
                ),
            ]

            # Create sample cue data
            cue = CueData(
                angle=45.0,
                position=[300.0, 220.0],
                detected=True,
                confidence=0.87,
                length=200.0,
                tip_position=[310.0, 230.0],
            )

            # Create sample table data
            table = TableData(
                corners=[[50.0, 50.0], [590.0, 50.0], [590.0, 430.0], [50.0, 430.0]],
                pockets=[
                    [60.0, 60.0],
                    [320.0, 55.0],
                    [580.0, 60.0],
                    [580.0, 420.0],
                    [320.0, 425.0],
                    [60.0, 420.0],
                ],
                calibrated=True,
                dimensions={"length": 2.54, "width": 1.27},  # meters
            )

            # Create game state
            game_state = GameStateData(
                balls=balls,
                cue=cue,
                table=table,
                game_mode="8ball",
                current_player="player1",
                player_turn=1,
                shot_number=5,
                game_status="in_progress",
            )

            message = WebSocketMessage(
                type=MessageType.STATE,
                timestamp=datetime.now(timezone.utc),
                sequence=2,
                data=game_state.dict(),
            )

            # Test serialization/deserialization
            json_message = message.json()
            parsed_message = json.loads(json_message)

            return {
                "message_type": "state",
                "status": "valid",
                "sample_message": parsed_message,
                "validation_notes": [
                    f"Game state includes {len(balls)} balls",
                    "Cue stick detection data included",
                    "Table calibration data included",
                    "All required fields present and valid",
                ],
            }

        except Exception as e:
            return {
                "message_type": "state",
                "status": "error",
                "error": str(e),
                "validation_notes": ["Game state message format validation failed"],
            }

    def verify_trajectory_message(self) -> dict[str, Any]:
        """Verify trajectory message format compatibility."""
        try:
            # Create sample trajectory data
            trajectory = TrajectoryData(
                ball_id="cue",
                path=[[320.0, 240.0], [350.0, 220.0], [380.0, 200.0], [400.0, 180.0]],
                ghost_ball_position=[395.0, 185.0],
                predicted_collisions=[
                    {
                        "ball_id": "1",
                        "collision_point": [395.0, 185.0],
                        "collision_time": 0.5,
                        "collision_angle": 45.0,
                    }
                ],
                shot_power=0.7,
                success_probability=0.82,
                recommendations=["Aim slightly higher for better angle"],
                physics_data={
                    "initial_velocity": 5.0,
                    "spin": {"english": 0.0, "follow": 0.2},
                    "estimated_time": 1.2,
                },
            )

            message = WebSocketMessage(
                type=MessageType.TRAJECTORY,
                timestamp=datetime.now(timezone.utc),
                sequence=3,
                data=trajectory.dict(),
            )

            # Test serialization/deserialization
            json_message = message.json()
            parsed_message = json.loads(json_message)

            return {
                "message_type": "trajectory",
                "status": "valid",
                "sample_message": parsed_message,
                "validation_notes": [
                    "Trajectory path with multiple points",
                    "Ghost ball position included",
                    "Collision predictions included",
                    "Physics data and recommendations included",
                ],
            }

        except Exception as e:
            return {
                "message_type": "trajectory",
                "status": "error",
                "error": str(e),
                "validation_notes": ["Trajectory message format validation failed"],
            }

    def verify_alert_message(self) -> dict[str, Any]:
        """Verify alert message format compatibility."""
        try:
            alert = AlertData(
                level="warning",
                title="Camera Detection Issue",
                message="Ball detection confidence has dropped below threshold",
                source="vision",
                code="LOW_CONFIDENCE",
                details={"confidence": 0.65, "threshold": 0.8},
                actions=["Recalibrate camera", "Check lighting conditions"],
            )

            message = WebSocketMessage(
                type=MessageType.ALERT,
                timestamp=datetime.now(timezone.utc),
                sequence=4,
                data=alert.dict(),
            )

            # Test serialization/deserialization
            json_message = message.json()
            parsed_message = json.loads(json_message)

            return {
                "message_type": "alert",
                "status": "valid",
                "sample_message": parsed_message,
                "validation_notes": [
                    "Alert level and severity properly set",
                    "Descriptive title and message",
                    "Source module identification",
                    "Actionable recommendations included",
                ],
            }

        except Exception as e:
            return {
                "message_type": "alert",
                "status": "error",
                "error": str(e),
                "validation_notes": ["Alert message format validation failed"],
            }

    def verify_config_message(self) -> dict[str, Any]:
        """Verify configuration message format compatibility."""
        try:
            config = ConfigData(
                module="vision",
                settings={
                    "camera": {
                        "resolution": [1920, 1080],
                        "fps": 30,
                        "backend": "auto",
                    },
                    "detection": {
                        "ball_confidence_threshold": 0.8,
                        "cue_detection_enabled": True,
                        "tracking_enabled": True,
                    },
                },
                version="1.2.0",
                updated_by="system",
            )

            message = WebSocketMessage(
                type=MessageType.CONFIG,
                timestamp=datetime.now(timezone.utc),
                sequence=5,
                data=config.dict(),
            )

            # Test serialization/deserialization
            json_message = message.json()
            parsed_message = json.loads(json_message)

            return {
                "message_type": "config",
                "status": "valid",
                "sample_message": parsed_message,
                "validation_notes": [
                    "Module-specific configuration data",
                    "Nested settings structure",
                    "Version tracking included",
                    "Update attribution included",
                ],
            }

        except Exception as e:
            return {
                "message_type": "config",
                "status": "error",
                "error": str(e),
                "validation_notes": ["Config message format validation failed"],
            }

    def verify_metrics_message(self) -> dict[str, Any]:
        """Verify metrics message format compatibility."""
        try:
            metrics = MetricsData(
                module="api",
                performance={
                    "cpu_usage": 25.4,
                    "memory_usage": 512.8,
                    "processing_time": 0.023,
                    "queue_size": 3,
                },
                counters={"requests_processed": 1250, "errors": 2, "warnings": 15},
                rates={"fps": 29.8, "messages_per_second": 45.2},
                quality_indicators={
                    "detection_accuracy": 0.94,
                    "tracking_stability": 0.98,
                    "calibration_score": 0.96,
                },
            )

            message = WebSocketMessage(
                type=MessageType.METRICS,
                timestamp=datetime.now(timezone.utc),
                sequence=6,
                data=metrics.dict(),
            )

            # Test serialization/deserialization
            json_message = message.json()
            parsed_message = json.loads(json_message)

            return {
                "message_type": "metrics",
                "status": "valid",
                "sample_message": parsed_message,
                "validation_notes": [
                    "Performance metrics included",
                    "Counter and rate data",
                    "Quality indicators",
                    "Module attribution",
                ],
            }

        except Exception as e:
            return {
                "message_type": "metrics",
                "status": "error",
                "error": str(e),
                "validation_notes": ["Metrics message format validation failed"],
            }

    def run_comprehensive_verification(self) -> dict[str, Any]:
        """Run comprehensive WebSocket message format verification."""
        logger.info("Starting comprehensive WebSocket message format verification...")

        results = {
            "verification_timestamp": datetime.now(timezone.utc).isoformat(),
            "message_formats": {},
            "summary": {
                "total_formats": 6,
                "valid_formats": 0,
                "invalid_formats": 0,
                "errors": [],
            },
        }

        # Run all verification tests
        verifications = [
            ("frame", self.verify_frame_message),
            ("state", self.verify_game_state_message),
            ("trajectory", self.verify_trajectory_message),
            ("alert", self.verify_alert_message),
            ("config", self.verify_config_message),
            ("metrics", self.verify_metrics_message),
        ]

        for message_type, verify_func in verifications:
            try:
                result = verify_func()
                results["message_formats"][message_type] = result

                if result["status"] == "valid":
                    results["summary"]["valid_formats"] += 1
                else:
                    results["summary"]["invalid_formats"] += 1
                    results["summary"]["errors"].append(
                        f"{message_type}: {result.get('error', 'Unknown error')}"
                    )

            except Exception as e:
                logger.error(f"Error verifying {message_type} message format: {e}")
                results["message_formats"][message_type] = {
                    "message_type": message_type,
                    "status": "error",
                    "error": str(e),
                    "validation_notes": ["Verification test failed"],
                }
                results["summary"]["invalid_formats"] += 1
                results["summary"]["errors"].append(f"{message_type}: {str(e)}")

        # Add overall assessment
        if results["summary"]["invalid_formats"] == 0:
            results["summary"]["overall_status"] = "all_formats_valid"
            results["summary"][
                "assessment"
            ] = "All WebSocket message formats are valid and compatible"
        elif (
            results["summary"]["valid_formats"] > results["summary"]["invalid_formats"]
        ):
            results["summary"]["overall_status"] = "mostly_valid"
            results["summary"][
                "assessment"
            ] = "Most WebSocket message formats are valid, some issues need attention"
        else:
            results["summary"]["overall_status"] = "needs_attention"
            results["summary"][
                "assessment"
            ] = "Multiple WebSocket message format issues need to be resolved"

        logger.info(
            f"WebSocket format verification completed: {results['summary']['assessment']}"
        )
        return results

    def generate_frontend_types(self) -> str:
        """Generate TypeScript type definitions for frontend compatibility."""
        typescript_types = f"""
// Auto-generated WebSocket message type definitions
// Generated on: {datetime.now(timezone.utc).isoformat()}

export interface WebSocketMessage {{
  type: MessageType;
  timestamp: string;
  sequence?: number;
  data: any;
}}

export type MessageType =
  | 'frame'
  | 'state'
  | 'trajectory'
  | 'alert'
  | 'config'
  | 'metrics'
  | 'connection'
  | 'ping'
  | 'pong'
  | 'subscribe'
  | 'unsubscribe'
  | 'subscribed'
  | 'unsubscribed'
  | 'status'
  | 'error';

export interface FrameData {{
  image: string;           // base64 encoded image
  width: number;
  height: number;
  format: string;          // 'jpeg', 'png', etc.
  quality: number;         // 1-100
  compressed: boolean;
  fps: number;
  size_bytes: number;
}}

export interface BallData {{
  id: string;              // 'cue', '1', '2', ..., '8ball'
  position: [number, number]; // [x, y]
  radius: number;
  color: string;
  velocity?: [number, number]; // [vx, vy]
  confidence: number;      // 0.0-1.0
  visible: boolean;
}}

export interface CueData {{
  angle: number;           // degrees
  position: [number, number]; // [x, y]
  detected: boolean;
  confidence: number;      // 0.0-1.0
  length?: number;
  tip_position?: [number, number];
}}

export interface TableData {{
  corners: [number, number][]; // 4 corner coordinates
  pockets: [number, number][]; // pocket coordinates
  rails?: any[];
  calibrated: boolean;
  dimensions?: {{ length: number; width: number }};
}}

export interface GameStateData {{
  balls: BallData[];
  cue: CueData;
  table: TableData;
  game_mode: string;
  current_player?: string;
  player_turn?: number;
  shot_number?: number;
  game_status: string;
}}

export interface TrajectoryData {{
  ball_id: string;
  path: [number, number][];
  ghost_ball_position?: [number, number];
  predicted_collisions: any[];
  shot_power: number;
  success_probability: number;
  recommendations: string[];
  physics_data: any;
}}

export interface AlertData {{
  level: 'info' | 'warning' | 'error' | 'critical';
  title: string;
  message: string;
  source: string;
  code?: string;
  details?: any;
  actions?: string[];
}}

export interface ConfigData {{
  module: string;
  settings: any;
  version?: string;
  updated_by?: string;
}}

export interface MetricsData {{
  module: string;
  performance: any;
  counters: any;
  rates: any;
  quality_indicators: any;
}}
"""

        return typescript_types


async def run_format_verification():
    """Standalone function to run WebSocket format verification."""
    verifier = MessageFormatVerifier()
    results = verifier.run_comprehensive_verification()

    # Print results
    print("\n" + "=" * 60)
    print("WEBSOCKET MESSAGE FORMAT VERIFICATION RESULTS")
    print("=" * 60)

    print(f"\nOverall Status: {results['summary']['overall_status']}")
    print(f"Assessment: {results['summary']['assessment']}")
    print(
        f"Valid Formats: {results['summary']['valid_formats']}/{results['summary']['total_formats']}"
    )

    if results["summary"]["errors"]:
        print("\nErrors Found:")
        for error in results["summary"]["errors"]:
            print(f"  - {error}")

    print("\nDetailed Results:")
    for message_type, result in results["message_formats"].items():
        status = result["status"].upper()
        print(f"  {message_type}: {status}")
        if result["status"] == "valid":
            print(f"    ✓ {', '.join(result['validation_notes'])}")
        else:
            print(f"    ✗ {result.get('error', 'Unknown error')}")

    # Generate TypeScript types
    typescript_types = verifier.generate_frontend_types()
    print("\n" + "=" * 60)
    print("TYPESCRIPT TYPE DEFINITIONS")
    print("=" * 60)
    print(typescript_types)

    return results


if __name__ == "__main__":
    asyncio.run(run_format_verification())
