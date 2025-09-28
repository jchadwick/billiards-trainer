#!/usr/bin/env python3
"""Demo script showing complete API integration with backend modules.

This script demonstrates:
1. API integration startup and initialization
2. Backend module communication
3. Service layer usage
4. Event handling and real-time updates
5. Caching and performance optimization
6. Health monitoring and metrics
7. Error handling and recovery
8. Data transformation between layers
"""

import asyncio
import logging
import signal
import sys
import time
from pathlib import Path
from typing import Any, Optional

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from .integration import (
    APIIntegration,
    IntegrationConfig,
    ServiceUnavailableError,
)
from .models.transformers import GameStateTransformer, HealthTransformer

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class IntegrationDemo:
    """Demo class for API integration functionality."""

    def __init__(self):
        """Initialize demo."""
        self.integration: Optional[APIIntegration] = None
        self.running = True
        self.demo_data = self._generate_demo_data()

    def _generate_demo_data(self) -> dict[str, Any]:
        """Generate demo data for testing."""
        return {
            "detection_data": {
                "balls": [
                    {
                        "id": "cue_ball",
                        "center": {"x": 100, "y": 200},
                        "radius": 15,
                        "color": [255, 255, 255],
                        "is_cue_ball": True,
                    },
                    {
                        "id": "ball_1",
                        "center": {"x": 300, "y": 200},
                        "radius": 15,
                        "color": [255, 255, 0],
                        "number": 1,
                    },
                ],
                "table": {
                    "corners": [
                        {"x": 50, "y": 100},
                        {"x": 650, "y": 100},
                        {"x": 650, "y": 300},
                        {"x": 50, "y": 300},
                    ],
                    "width": 600,
                    "height": 200,
                },
            },
            "config_updates": {
                "vision.camera.fps": 30,
                "core.physics.enabled": True,
                "api.cache.enabled": True,
            },
        }

    async def startup(self) -> None:
        """Start the integration demo."""
        try:
            logger.info("🚀 Starting API Integration Demo")

            # Create integration configuration
            config = IntegrationConfig(
                enable_config_module=True,
                enable_core_module=True,
                enable_vision_module=True,
                enable_caching=True,
                cache_ttl=300,
                enable_events=True,
                health_check_interval=10,
                enable_metrics=True,
                auto_recovery=True,
            )

            # Initialize integration
            self.integration = APIIntegration(config)
            await self.integration.startup()

            logger.info("✅ Integration startup completed successfully")

        except Exception as e:
            logger.error(f"❌ Failed to start integration: {e}")
            raise

    async def shutdown(self) -> None:
        """Shutdown the integration demo."""
        try:
            logger.info("🛑 Shutting down API Integration Demo")
            if self.integration:
                await self.integration.shutdown()
            logger.info("✅ Shutdown completed successfully")
        except Exception as e:
            logger.error(f"❌ Error during shutdown: {e}")

    async def demo_module_integration(self) -> None:
        """Demonstrate module integration and communication."""
        logger.info("\n" + "=" * 60)
        logger.info("📡 DEMO: Module Integration and Communication")
        logger.info("=" * 60)

        if not self.integration:
            logger.error("Integration not initialized")
            return

        try:
            # Test Configuration Module
            logger.info("🔧 Testing Configuration Module...")
            config_service = self.integration.get_config_service()

            # Get configuration values
            for key, value in self.demo_data["config_updates"].items():
                await config_service.set_config(key, value)
                retrieved = await config_service.get_config(key)
                logger.info(f"   Config {key}: {retrieved}")

            # Get all configurations
            all_configs = await config_service.get_all_configs()
            logger.info(f"   Total config keys: {len(all_configs)}")

        except ServiceUnavailableError as e:
            logger.warning(f"⚠️  Configuration service unavailable: {e}")
        except Exception as e:
            logger.error(f"❌ Configuration module error: {e}")

        try:
            # Test Core Module
            logger.info("🎱 Testing Core Module...")
            game_service = self.integration.get_game_service()

            # Update game state
            detection_data = self.demo_data["detection_data"]
            game_state = await game_service.update_state(detection_data)
            logger.info(
                f"   Game state updated: {len(game_state.get('balls', []))} balls detected"
            )

            # Get current state
            current_state = await game_service.get_current_state()
            if current_state:
                logger.info(
                    f"   Current state timestamp: {current_state.get('timestamp')}"
                )

            # Analyze shot
            analysis = await game_service.analyze_shot()
            logger.info(f"   Shot analysis: {analysis.get('shot_type', 'unknown')}")

            # Get shot suggestions
            suggestions = await game_service.suggest_shots(difficulty_filter=0.7)
            logger.info(f"   Shot suggestions: {len(suggestions)} options")

        except ServiceUnavailableError as e:
            logger.warning(f"⚠️  Game service unavailable: {e}")
        except Exception as e:
            logger.error(f"❌ Core module error: {e}")

        try:
            # Test Vision Module (if available)
            logger.info("👁️  Testing Vision Module...")
            detection_service = self.integration.get_detection_service()

            # Start detection
            start_result = await detection_service.start_detection()
            logger.info(f"   Detection started: {start_result.get('success')}")

            # Get statistics
            stats = await detection_service.get_detection_statistics()
            logger.info(
                f"   Detection stats: {stats.get('frames_processed', 0)} frames processed"
            )

            # Get latest detection
            detection = await detection_service.get_latest_detection()
            if detection:
                logger.info(
                    f"   Latest detection: Frame {detection.get('frame_number')}"
                )

            # Stop detection
            stop_result = await detection_service.stop_detection()
            logger.info(f"   Detection stopped: {stop_result.get('success')}")

        except ServiceUnavailableError as e:
            logger.warning(f"⚠️  Detection service unavailable: {e}")
        except Exception as e:
            logger.error(f"❌ Vision module error: {e}")

    async def demo_caching_performance(self) -> None:
        """Demonstrate caching and performance optimization."""
        logger.info("\n" + "=" * 60)
        logger.info("⚡ DEMO: Caching and Performance Optimization")
        logger.info("=" * 60)

        if not self.integration:
            return

        try:
            # Test cache operations
            logger.info("💾 Testing cache operations...")

            # Cache performance test
            start_time = time.time()
            for i in range(100):
                await self.integration.set_cached(f"perf_test_{i}", f"value_{i}")
            cache_write_time = time.time() - start_time

            start_time = time.time()
            hits = 0
            for i in range(100):
                value = await self.integration.get_cached(f"perf_test_{i}")
                if value:
                    hits += 1
            cache_read_time = time.time() - start_time

            logger.info(f"   Cache write time (100 ops): {cache_write_time:.4f}s")
            logger.info(f"   Cache read time (100 ops): {cache_read_time:.4f}s")
            logger.info(f"   Cache hit rate: {hits}/100")

            # Test cache expiration
            logger.info("⏰ Testing cache expiration...")
            await self.integration.set_cached("expire_test", "temporary_value")

            # Immediate read
            value = await self.integration.get_cached("expire_test")
            logger.info(f"   Immediate read: {value}")

            # Test cache patterns
            logger.info("🔄 Testing cache patterns...")
            await self.integration.clear_cache("perf_test_")
            remaining = 0
            for i in range(100):
                value = await self.integration.get_cached(f"perf_test_{i}")
                if value:
                    remaining += 1
            logger.info(f"   Remaining after pattern clear: {remaining}")

        except Exception as e:
            logger.error(f"❌ Caching demo error: {e}")

    async def demo_health_monitoring(self) -> None:
        """Demonstrate health monitoring and metrics."""
        logger.info("\n" + "=" * 60)
        logger.info("🏥 DEMO: Health Monitoring and Metrics")
        logger.info("=" * 60)

        if not self.integration:
            return

        try:
            # Get overall health status
            logger.info("📊 Getting overall health status...")
            health = await self.integration.get_health_status()

            logger.info(f"   Overall status: {health['status']}")
            logger.info(f"   Services monitored: {len(health.get('services', {}))}")
            logger.info(f"   Uptime: {health.get('uptime', 0):.2f}s")

            # Check individual services
            logger.info("🔍 Checking individual services...")
            services = ["config", "core", "vision"]

            for service_name in services:
                try:
                    service_health = await self.integration.check_service_health(
                        service_name
                    )
                    status_emoji = (
                        "✅"
                        if service_health.status == "healthy"
                        else "⚠️" if service_health.status == "degraded" else "❌"
                    )
                    response_time = service_health.response_time or 0
                    logger.info(
                        f"   {status_emoji} {service_name}: {service_health.status} ({response_time:.4f}s)"
                    )
                except Exception as e:
                    logger.warning(f"   ❌ {service_name}: Failed to check - {e}")

            # Display metrics
            if "metrics" in health:
                metrics = health["metrics"]
                logger.info("📈 Integration metrics:")
                logger.info(
                    f"   Requests processed: {metrics.get('requests_processed', 0)}"
                )
                logger.info(
                    f"   Average response time: {metrics.get('average_response_time', 0):.4f}s"
                )
                logger.info(f"   Cache hits: {metrics.get('cache_hits', 0)}")
                logger.info(f"   Cache misses: {metrics.get('cache_misses', 0)}")
                logger.info(f"   Errors: {metrics.get('errors_count', 0)}")

        except Exception as e:
            logger.error(f"❌ Health monitoring demo error: {e}")

    async def demo_event_handling(self) -> None:
        """Demonstrate event handling and real-time updates."""
        logger.info("\n" + "=" * 60)
        logger.info("⚡ DEMO: Event Handling and Real-time Updates")
        logger.info("=" * 60)

        if not self.integration:
            return

        try:
            # Simulate events
            logger.info("📡 Simulating backend events...")

            # Game state update event
            await self.integration._queue_event(
                "state_updated",
                {"state": {"balls": 2, "timestamp": time.time()}, "source": "demo"},
            )

            # Detection complete event
            await self.integration._queue_event(
                "detection_complete",
                {
                    "result": {"frame_number": 100, "processing_time": 50.0},
                    "source": "demo",
                },
            )

            # Configuration change event
            await self.integration._queue_event(
                "config_changed",
                {
                    "key": "demo.setting",
                    "old_value": "old",
                    "new_value": "new",
                    "source": "demo",
                },
            )

            logger.info("   Events queued successfully")

            # Process events for a short time
            logger.info("⚙️  Processing events...")
            await asyncio.sleep(2.0)  # Let event processor run

            logger.info(
                f"   Events processed: {self.integration.metrics.events_processed}"
            )

        except Exception as e:
            logger.error(f"❌ Event handling demo error: {e}")

    async def demo_data_transformation(self) -> None:
        """Demonstrate data transformation between layers."""
        logger.info("\n" + "=" * 60)
        logger.info("🔄 DEMO: Data Transformation Between Layers")
        logger.info("=" * 60)

        try:
            # Demo Vector2D transformation
            logger.info("📐 Vector2D transformation...")
            from ..core.models import Vector2D

            # Backend to API
            backend_vector = Vector2D(x=100.5, y=200.7)
            api_vector = GameStateTransformer.vector2d_to_model(backend_vector)
            logger.info(
                f"   Backend Vector2D({backend_vector.x}, {backend_vector.y}) -> API Vector2DModel({api_vector.x}, {api_vector.y})"
            )

            # API to Backend
            converted_back = GameStateTransformer.model_to_vector2d(api_vector)
            logger.info(
                f"   API Vector2DModel({api_vector.x}, {api_vector.y}) -> Backend Vector2D({converted_back.x}, {converted_back.y})"
            )

            # Demo health status transformation
            logger.info("🏥 Health status transformation...")
            health_data = {
                "status": "healthy",
                "response_time": 0.05,
                "uptime": 3600.0,
                "version": "1.0.0",
            }

            health_model = HealthTransformer.service_health_to_model(
                "demo_service", "healthy", health_data
            )
            logger.info(
                f"   Service: {health_model.name}, Status: {health_model.status}, Type: {health_model.type}"
            )

            # Demo performance metrics transformation
            logger.info("📊 Performance metrics transformation...")
            perf_data = {
                "requests_total": 1000,
                "response_time_avg": 0.15,
                "error_rate": 0.02,
                "cache_hit_rate": 0.85,
            }

            perf_model = HealthTransformer.performance_metrics_to_model(perf_data)
            logger.info(
                f"   Requests: {perf_model.requests_total}, Avg Response: {perf_model.response_time_avg}s"
            )
            logger.info(
                f"   Error Rate: {perf_model.error_rate:.2%}, Cache Hit Rate: {perf_model.cache_hit_rate:.2%}"
            )

        except Exception as e:
            logger.error(f"❌ Data transformation demo error: {e}")

    async def demo_error_handling(self) -> None:
        """Demonstrate error handling and recovery."""
        logger.info("\n" + "=" * 60)
        logger.info("🛡️  DEMO: Error Handling and Recovery")
        logger.info("=" * 60)

        if not self.integration:
            return

        try:
            # Test service unavailable handling
            logger.info("⚠️  Testing service unavailable handling...")

            # Temporarily disable a service
            original_game_service = self.integration.game_service
            self.integration.game_service = None

            try:
                self.integration.get_game_service()
                logger.error("   Expected ServiceUnavailableError but got service")
            except ServiceUnavailableError:
                logger.info("   ✅ ServiceUnavailableError handled correctly")

            # Restore service
            self.integration.game_service = original_game_service

            # Test cache error handling
            logger.info("💾 Testing cache error handling...")

            # Try to cache an uncacheable object
            try:
                uncacheable = object()
                await self.integration.set_cached("uncacheable", uncacheable)
                logger.info("   ✅ Cache handled uncacheable object")
            except Exception as e:
                logger.info(f"   ✅ Cache error handled: {type(e).__name__}")

            # Test health check timeout
            logger.info("⏰ Testing health check timeout...")

            # This should complete within timeout
            start_time = time.time()
            await self.integration.check_service_health("config")
            elapsed = time.time() - start_time

            logger.info(
                f"   Health check completed in {elapsed:.4f}s (timeout: {self.integration.config.module_timeout}s)"
            )

            if elapsed < self.integration.config.module_timeout:
                logger.info("   ✅ Health check within timeout")
            else:
                logger.warning("   ⚠️  Health check exceeded timeout")

        except Exception as e:
            logger.error(f"❌ Error handling demo error: {e}")

    async def run_demo(self) -> None:
        """Run the complete integration demo."""
        try:
            await self.startup()

            # Run all demo sections
            demos = [
                self.demo_module_integration,
                self.demo_caching_performance,
                self.demo_health_monitoring,
                self.demo_event_handling,
                self.demo_data_transformation,
                self.demo_error_handling,
            ]

            for demo in demos:
                if not self.running:
                    break
                await demo()
                await asyncio.sleep(1)  # Brief pause between demos

            # Final status report
            logger.info("\n" + "=" * 60)
            logger.info("📋 DEMO: Final Status Report")
            logger.info("=" * 60)

            if self.integration:
                health = await self.integration.get_health_status()
                logger.info(f"🏥 Overall Health: {health['status']}")
                logger.info(f"⏱️  Total Uptime: {health.get('uptime', 0):.2f}s")

                if "metrics" in health:
                    metrics = health["metrics"]
                    logger.info(
                        f"📊 Total Requests: {metrics.get('requests_processed', 0)}"
                    )
                    logger.info(
                        f"⚡ Avg Response Time: {metrics.get('average_response_time', 0):.4f}s"
                    )
                    logger.info(f"❌ Total Errors: {metrics.get('errors_count', 0)}")

            logger.info("\n🎉 Integration demo completed successfully!")

        except KeyboardInterrupt:
            logger.info("\n⚠️  Demo interrupted by user")
        except Exception as e:
            logger.error(f"\n❌ Demo failed: {e}")
        finally:
            self.running = False
            await self.shutdown()


async def main():
    """Main demo function."""
    logger.info("🎱 Billiards Trainer API Integration Demo")
    logger.info("========================================")

    demo = IntegrationDemo()

    # Set up signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        logger.info(f"\n📡 Received signal {signum}, shutting down gracefully...")
        demo.running = False

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        await demo.run_demo()
    except Exception as e:
        logger.error(f"❌ Demo failed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    # Run the demo
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
