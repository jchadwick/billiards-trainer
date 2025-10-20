"""Manual test script for streaming endpoints.

Tests the shared memory streaming endpoints with:
1. Single client streaming
2. Multiple concurrent client streaming (10+ clients)
3. Client disconnect cleanup
4. Backward compatibility with legacy mode

Usage:
    # Test shared memory streaming (requires Video Module running)
    python backend/tests/manual/test_streaming_endpoints.py --mode shm

    # Test legacy camera module streaming
    python backend/tests/manual/test_streaming_endpoints.py --mode legacy

    # Test with multiple concurrent clients
    python backend/tests/manual/test_streaming_endpoints.py --mode shm --clients 15

    # Test feature flag integration
    python backend/tests/manual/test_streaming_endpoints.py --mode feature-flag

Prerequisites:
    - API server running: python -m backend.api.main
    - For shared memory mode: Video Module running: python -m backend.video
"""

import argparse
import asyncio
import logging
import sys
import time
from typing import List, Optional

import httpx

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class StreamingTestClient:
    """Test client for streaming endpoints."""

    def __init__(
        self,
        client_id: int,
        base_url: str,
        endpoint: str,
        quality: int = 85,
        fps: int = 30,
    ):
        """Initialize test client.

        Args:
            client_id: Client identifier
            base_url: API base URL
            endpoint: Streaming endpoint path
            quality: JPEG quality
            fps: Target frame rate
        """
        self.client_id = client_id
        self.base_url = base_url
        self.endpoint = endpoint
        self.quality = quality
        self.fps = fps

        self.frames_received = 0
        self.bytes_received = 0
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.errors: list[str] = []

    async def stream(self, duration_seconds: int = 10):
        """Stream video for specified duration.

        Args:
            duration_seconds: How long to stream in seconds
        """
        url = f"{self.base_url}{self.endpoint}?quality={self.quality}&fps={self.fps}"
        logger.info(f"Client {self.client_id} connecting to {url}")

        self.start_time = time.time()

        try:
            async with httpx.AsyncClient(timeout=duration_seconds + 5) as client:
                async with client.stream("GET", url) as response:
                    if response.status_code != 200:
                        error = f"HTTP {response.status_code}: {await response.aread()}"
                        self.errors.append(error)
                        logger.error(f"Client {self.client_id} error: {error}")
                        return

                    logger.info(
                        f"Client {self.client_id} connected, streaming for {duration_seconds}s"
                    )

                    # Read frames until duration expires
                    end_time = time.time() + duration_seconds
                    async for chunk in response.aiter_bytes():
                        if time.time() > end_time:
                            break

                        self.bytes_received += len(chunk)

                        # Count frames (look for JPEG boundaries)
                        if b"\xff\xd8" in chunk:  # JPEG start marker
                            self.frames_received += 1

                        if self.frames_received % 30 == 0 and self.frames_received > 0:
                            elapsed = time.time() - self.start_time
                            fps = self.frames_received / elapsed
                            logger.debug(
                                f"Client {self.client_id}: {self.frames_received} frames, "
                                f"{fps:.1f} FPS"
                            )

        except Exception as e:
            error = f"Exception: {str(e)}"
            self.errors.append(error)
            logger.error(f"Client {self.client_id} error: {error}")

        finally:
            self.end_time = time.time()

    def get_stats(self) -> dict:
        """Get client statistics.

        Returns:
            Dictionary with client statistics
        """
        if self.start_time and self.end_time:
            duration = self.end_time - self.start_time
            fps = self.frames_received / duration if duration > 0 else 0
            mbps = (self.bytes_received * 8 / 1000000) / duration if duration > 0 else 0
        else:
            duration = 0
            fps = 0
            mbps = 0

        return {
            "client_id": self.client_id,
            "frames_received": self.frames_received,
            "bytes_received": self.bytes_received,
            "duration_sec": duration,
            "avg_fps": fps,
            "avg_mbps": mbps,
            "errors": self.errors,
        }


async def test_single_client(base_url: str, endpoint: str, duration: int = 10):
    """Test single client streaming.

    Args:
        base_url: API base URL
        endpoint: Streaming endpoint path
        duration: Streaming duration in seconds
    """
    logger.info("=" * 80)
    logger.info("TEST: Single Client Streaming")
    logger.info("=" * 80)

    client = StreamingTestClient(
        client_id=1, base_url=base_url, endpoint=endpoint, quality=85, fps=30
    )

    await client.stream(duration_seconds=duration)
    stats = client.get_stats()

    logger.info("Single Client Results:")
    logger.info(f"  Frames Received: {stats['frames_received']}")
    logger.info(f"  Duration: {stats['duration_sec']:.2f}s")
    logger.info(f"  Average FPS: {stats['avg_fps']:.1f}")
    logger.info(f"  Average Mbps: {stats['avg_mbps']:.2f}")
    logger.info(f"  Errors: {len(stats['errors'])}")

    if stats["errors"]:
        for error in stats["errors"]:
            logger.error(f"    {error}")

    # Validate results
    success = (
        stats["frames_received"] > 0
        and len(stats["errors"]) == 0
        and stats["avg_fps"] > 15  # At least 15 FPS
    )

    logger.info(f"TEST RESULT: {'PASS' if success else 'FAIL'}")
    return success


async def test_concurrent_clients(
    base_url: str, endpoint: str, num_clients: int = 10, duration: int = 10
):
    """Test multiple concurrent client streaming.

    Args:
        base_url: API base URL
        endpoint: Streaming endpoint path
        num_clients: Number of concurrent clients
        duration: Streaming duration in seconds
    """
    logger.info("=" * 80)
    logger.info(f"TEST: Concurrent Streaming ({num_clients} clients)")
    logger.info("=" * 80)

    clients = [
        StreamingTestClient(
            client_id=i, base_url=base_url, endpoint=endpoint, quality=80, fps=30
        )
        for i in range(1, num_clients + 1)
    ]

    logger.info(f"Starting {num_clients} concurrent clients...")
    start_time = time.time()

    # Start all clients concurrently
    tasks = [client.stream(duration_seconds=duration) for client in clients]
    await asyncio.gather(*tasks)

    total_duration = time.time() - start_time

    # Collect statistics
    all_stats = [client.get_stats() for client in clients]

    total_frames = sum(s["frames_received"] for s in all_stats)
    avg_fps = sum(s["avg_fps"] for s in all_stats) / len(all_stats)
    total_errors = sum(len(s["errors"]) for s in all_stats)
    total_mbps = sum(s["avg_mbps"] for s in all_stats)

    logger.info("Concurrent Client Results:")
    logger.info(f"  Total Clients: {num_clients}")
    logger.info(f"  Total Duration: {total_duration:.2f}s")
    logger.info(f"  Total Frames: {total_frames}")
    logger.info(f"  Average FPS per Client: {avg_fps:.1f}")
    logger.info(f"  Total Bandwidth: {total_mbps:.2f} Mbps")
    logger.info(f"  Total Errors: {total_errors}")

    if total_errors > 0:
        logger.warning("Errors encountered:")
        for stats in all_stats:
            if stats["errors"]:
                logger.error(f"  Client {stats['client_id']}: {stats['errors']}")

    # Per-client breakdown
    logger.info("\nPer-Client Breakdown:")
    for stats in all_stats:
        logger.info(
            f"  Client {stats['client_id']}: "
            f"{stats['frames_received']} frames, "
            f"{stats['avg_fps']:.1f} FPS, "
            f"{len(stats['errors'])} errors"
        )

    # Validate results
    success = (
        total_frames > 0
        and total_errors == 0
        and avg_fps > 15  # Average at least 15 FPS per client
    )

    logger.info(f"TEST RESULT: {'PASS' if success else 'FAIL'}")
    return success


async def test_client_disconnect(base_url: str, endpoint: str):
    """Test client disconnect cleanup.

    Args:
        base_url: API base URL
        endpoint: Streaming endpoint path
    """
    logger.info("=" * 80)
    logger.info("TEST: Client Disconnect Cleanup")
    logger.info("=" * 80)

    client = StreamingTestClient(
        client_id=1, base_url=base_url, endpoint=endpoint, quality=85, fps=30
    )

    # Stream for 2 seconds then disconnect
    logger.info("Starting client, will disconnect after 2 seconds...")
    await client.stream(duration_seconds=2)

    stats = client.get_stats()
    logger.info("Client Disconnect Results:")
    logger.info(f"  Frames Received: {stats['frames_received']}")
    logger.info(f"  Duration: {stats['duration_sec']:.2f}s")
    logger.info(f"  Clean Disconnect: {len(stats['errors']) == 0}")

    success = stats["frames_received"] > 0 and len(stats["errors"]) == 0

    logger.info(f"TEST RESULT: {'PASS' if success else 'FAIL'}")
    return success


async def test_feature_flag_integration(base_url: str):
    """Test feature flag integration.

    Args:
        base_url: API base URL
    """
    logger.info("=" * 80)
    logger.info("TEST: Feature Flag Integration")
    logger.info("=" * 80)

    # Test /video endpoint (should respect feature flag)
    logger.info("Testing /stream/video endpoint (feature flag routing)...")
    client = StreamingTestClient(
        client_id=1, base_url=base_url, endpoint="/stream/video", quality=85, fps=30
    )
    await client.stream(duration_seconds=5)
    stats = client.get_stats()

    logger.info("Feature Flag Test Results:")
    logger.info(f"  Frames Received: {stats['frames_received']}")
    logger.info(f"  Errors: {len(stats['errors'])}")

    success = stats["frames_received"] > 0

    logger.info(f"TEST RESULT: {'PASS' if success else 'FAIL'}")
    return success


async def main():
    """Main test runner."""
    parser = argparse.ArgumentParser(description="Test streaming endpoints")
    parser.add_argument(
        "--mode",
        choices=["shm", "legacy", "feature-flag", "all"],
        default="shm",
        help="Test mode",
    )
    parser.add_argument(
        "--clients", type=int, default=10, help="Number of concurrent clients"
    )
    parser.add_argument(
        "--duration", type=int, default=10, help="Streaming duration in seconds"
    )
    parser.add_argument(
        "--base-url", default="http://localhost:8000", help="API base URL"
    )

    args = parser.parse_args()

    results = []

    if args.mode in ["shm", "all"]:
        # Test shared memory endpoint
        logger.info("\n\nTesting Shared Memory Endpoint (/stream/video/shm)")
        logger.info("=" * 80)

        # Single client
        success = await test_single_client(
            args.base_url, "/stream/video/shm", args.duration
        )
        results.append(("Shared Memory - Single Client", success))

        # Concurrent clients
        success = await test_concurrent_clients(
            args.base_url, "/stream/video/shm", args.clients, args.duration
        )
        results.append(("Shared Memory - Concurrent Clients", success))

        # Disconnect cleanup
        success = await test_client_disconnect(args.base_url, "/stream/video/shm")
        results.append(("Shared Memory - Disconnect Cleanup", success))

    if args.mode in ["legacy", "all"]:
        # Test legacy endpoint (requires video.use_shared_memory=false)
        logger.info("\n\nTesting Legacy Endpoint (/stream/video)")
        logger.info("=" * 80)
        logger.info("NOTE: Set video.use_shared_memory=false in config.json")

        success = await test_single_client(
            args.base_url, "/stream/video", args.duration
        )
        results.append(("Legacy - Single Client", success))

    if args.mode in ["feature-flag", "all"]:
        # Test feature flag integration
        logger.info("\n\nTesting Feature Flag Integration")
        success = await test_feature_flag_integration(args.base_url)
        results.append(("Feature Flag Integration", success))

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("TEST SUMMARY")
    logger.info("=" * 80)

    for test_name, success in results:
        status = "PASS" if success else "FAIL"
        logger.info(f"  {test_name}: {status}")

    all_passed = all(success for _, success in results)
    logger.info("=" * 80)
    logger.info(f"OVERALL: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    asyncio.run(main())
