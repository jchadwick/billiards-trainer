#!/usr/bin/env python3
"""Basic WebSocket connectivity test."""

import asyncio
import json
import sys

import websockets


async def test_websocket():
    """Test WebSocket connection to backend."""
    uri = "ws://localhost:8001/ws"

    try:
        print(f"Connecting to {uri}...")
        async with websockets.connect(uri) as websocket:
            print("✓ WebSocket connection established")

            # Send a ping message
            ping_message = {"type": "ping", "timestamp": 1234567890}
            await websocket.send(json.dumps(ping_message))
            print("✓ Ping message sent")

            # Wait for response
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                print(f"✓ Received response: {response}")

                # Try to parse as JSON
                try:
                    response_data = json.loads(response)
                    print(f"✓ Response parsed as JSON: {response_data}")
                except json.JSONDecodeError:
                    print(f"⚠ Response not JSON: {response}")

            except asyncio.TimeoutError:
                print("⚠ No response received within 5 seconds")

            print("✓ WebSocket test completed successfully")

    except ConnectionRefusedError:
        print("✗ Connection refused - is the backend server running?")
        return False
    except Exception as e:
        print(f"✗ WebSocket test failed: {e}")
        return False

    return True


if __name__ == "__main__":
    result = asyncio.run(test_websocket())
    sys.exit(0 if result else 1)
