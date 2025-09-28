#!/usr/bin/env python3
"""
WebSocket System Demonstration Script

This script demonstrates the real-time WebSocket streaming capabilities
of the billiards trainer system.
"""

import asyncio
import json
import time
import logging
from datetime import datetime, timezone

import numpy as np
import uvicorn

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def demo_websocket_system():
    """Demonstrate the WebSocket system functionality"""
    print("=" * 60)
    print("Billiards Trainer WebSocket System Demo")
    print("=" * 60)

    try:
        # Import WebSocket components
        from backend.api.websocket import (
            websocket_handler,
            websocket_manager,
            message_broadcaster,
            connection_monitor,
            initialize_websocket_system,
            shutdown_websocket_system
        )

        print("✓ WebSocket components imported successfully")

        # Initialize the WebSocket system
        print("\n1. Initializing WebSocket system...")
        await initialize_websocket_system()
        print("✓ WebSocket system initialized")

        # Test message broadcasting
        print("\n2. Testing message broadcasting...")

        # Broadcast test game state
        test_balls = [
            {
                "id": "cue",
                "position": [100.0, 200.0],
                "radius": 20.0,
                "color": "white",
                "velocity": [0.0, 0.0],
                "confidence": 0.95
            },
            {
                "id": "8ball",
                "position": [300.0, 400.0],
                "radius": 20.0,
                "color": "black",
                "velocity": [5.0, -3.0],
                "confidence": 0.88
            }
        ]

        test_cue = {
            "angle": 45.5,
            "position": [150.0, 250.0],
            "detected": True,
            "confidence": 0.92
        }

        await message_broadcaster.broadcast_game_state(
            balls=test_balls,
            cue=test_cue
        )
        print("✓ Game state broadcast completed")

        # Broadcast test trajectory
        test_lines = [
            {
                "start": [100.0, 200.0],
                "end": [300.0, 400.0],
                "type": "primary"
            },
            {
                "start": [300.0, 400.0],
                "end": [500.0, 200.0],
                "type": "reflection"
            }
        ]

        test_collisions = [
            {
                "position": [300.0, 400.0],
                "ball_id": "8ball",
                "angle": 30.0
            }
        ]

        await message_broadcaster.broadcast_trajectory(
            lines=test_lines,
            collisions=test_collisions,
            confidence=0.85,
            calculation_time_ms=12.5
        )
        print("✓ Trajectory broadcast completed")

        # Broadcast test frame
        print("\n3. Testing frame broadcasting...")
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        # Add some visual elements
        test_image[100:200, 100:200] = [0, 255, 0]  # Green square
        test_image[300:400, 400:500] = [255, 0, 0]  # Red square

        await message_broadcaster.broadcast_frame(
            image_data=test_image,
            width=640,
            height=480,
            quality=85
        )
        print("✓ Frame broadcast completed")

        # Test alert broadcasting
        print("\n4. Testing alert broadcasting...")
        await message_broadcaster.broadcast_alert(
            level="info",
            message="WebSocket system demonstration in progress",
            code="DEMO_001",
            details={
                "demo_step": 4,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "test_data": True
            }
        )
        print("✓ Alert broadcast completed")

        # Display system statistics
        print("\n5. System Performance Statistics:")
        print("-" * 40)

        # Handler stats
        handler_stats = websocket_handler.get_connection_stats()
        print(f"Active connections: {handler_stats['total_connections']}")
        print(f"Total messages sent: {handler_stats['total_messages']}")
        print(f"Total bytes sent: {handler_stats['total_bytes_sent']:,}")

        # Broadcaster stats
        broadcaster_stats = message_broadcaster.get_broadcast_stats()
        print(f"Broadcasting active: {broadcaster_stats['is_streaming']}")
        print(f"Frame buffer size: {broadcaster_stats['frame_buffer_size']}")
        print(f"Current FPS: {broadcaster_stats['current_fps']:.1f}")

        # Monitor stats
        system_health = connection_monitor.get_system_health()
        print(f"System status: {system_health['overall_status']}")
        print(f"Average latency: {system_health['performance']['average_latency_ms']:.2f} ms")

        print("\n6. WebSocket Message Protocol Examples:")
        print("-" * 40)

        # Show example messages
        frame_example = {
            "type": "frame",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "sequence": 12345,
            "data": {
                "image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEAAA...",
                "width": 1920,
                "height": 1080,
                "format": "jpeg",
                "quality": 85,
                "fps": 30.0,
                "size_bytes": 87432
            }
        }

        state_example = {
            "type": "state",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "sequence": 12346,
            "data": {
                "balls": test_balls,
                "cue": test_cue,
                "ball_count": len(test_balls)
            }
        }

        trajectory_example = {
            "type": "trajectory",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "sequence": 12347,
            "data": {
                "lines": test_lines,
                "collisions": test_collisions,
                "confidence": 0.85,
                "calculation_time_ms": 12.5,
                "line_count": len(test_lines),
                "collision_count": len(test_collisions)
            }
        }

        print("Frame message example:")
        print(json.dumps(frame_example, indent=2)[:300] + "...")

        print("\nState message example:")
        print(json.dumps(state_example, indent=2))

        print("\nTrajectory message example:")
        print(json.dumps(trajectory_example, indent=2))

        print("\n7. Client Integration Instructions:")
        print("-" * 40)
        print("To connect to the WebSocket server:")
        print("1. Connect to: ws://localhost:8000/ws")
        print("2. Optional authentication: ws://localhost:8000/ws?token=your_token")
        print("3. Subscribe to streams:")
        print('   {"type": "subscribe", "data": {"streams": ["frame", "state", "trajectory", "alert"]}}')
        print("4. Handle incoming messages based on type field")
        print("5. Send ping messages to maintain connection quality monitoring")

        print("\n8. REST API Endpoints:")
        print("-" * 40)
        print("WebSocket management endpoints:")
        print("• GET /api/v1/websocket/connections - List active connections")
        print("• GET /api/v1/websocket/health - System health status")
        print("• POST /api/v1/websocket/broadcast/frame - Broadcast test frame")
        print("• POST /api/v1/websocket/broadcast/alert - Broadcast test alert")
        print("• POST /api/v1/websocket/system/start - Start WebSocket services")
        print("• POST /api/v1/websocket/system/stop - Stop WebSocket services")

        print("\n" + "=" * 60)
        print("WebSocket System Demo Completed Successfully!")
        print("=" * 60)

        # Keep system running for a bit to allow manual testing
        print("\nWebSocket system will remain active for 30 seconds for manual testing...")
        print("You can now connect with the client example:")
        print("python backend/api/websocket/client_example.py")

        await asyncio.sleep(30)

    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise

    finally:
        # Clean shutdown
        print("\nShutting down WebSocket system...")
        await shutdown_websocket_system()
        print("✓ WebSocket system shutdown completed")


async def run_server_demo():
    """Run the WebSocket server with demo data streaming"""
    print("Starting FastAPI server with WebSocket support...")

    try:
        from backend.api.main import app

        # Configure uvicorn
        config = uvicorn.Config(
            app,
            host="0.0.0.0",
            port=8000,
            log_level="info",
            access_log=True
        )

        server = uvicorn.Server(config)

        print("Server will start at: http://localhost:8000")
        print("WebSocket endpoint: ws://localhost:8000/ws")
        print("API Documentation: http://localhost:8000/docs")
        print("\nPress Ctrl+C to stop the server")

        await server.serve()

    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        raise


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "server":
        print("Running in server mode...")
        asyncio.run(run_server_demo())
    else:
        print("Running WebSocket system demo...")
        asyncio.run(demo_websocket_system())
        print("\nTo run the full server with WebSocket support:")
        print("python websocket_demo.py server")
