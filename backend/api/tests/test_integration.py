"""Integration tests for API backend communication."""

import asyncio
import time
from unittest.mock import AsyncMock, Mock, patch

import pytest

from backend.api.integration import (
    APIIntegration,
    ConfigurationService,
    DetectionService,
    GameStateService,
    IntegrationConfig,
    IntegrationError,
    ServiceUnavailableError,
)
from backend.config import ConfigurationModule
from backend.core import CoreModule
from backend.vision import VisionModule


class TestAPIIntegration:
    """Test suite for API Integration."""

    @pytest.fixture()
    async def integration_config(self):
        """Test integration configuration."""
        return IntegrationConfig(
            enable_config_module=True,
            enable_core_module=True,
            enable_vision_module=False,  # Disable vision for faster tests
            enable_caching=True,
            cache_ttl=60,
            health_check_interval=5,
            auto_recovery=True,
        )

    @pytest.fixture()
    async def mock_modules(self):
        """Mock backend modules."""
        config_module = Mock(spec=ConfigurationModule)
        config_module.get = Mock(return_value="test_value")
        config_module.set = Mock()

        core_module = Mock(spec=CoreModule)
        core_module.get_current_state = Mock(return_value=None)
        core_module.update_state = AsyncMock()
        core_module.analyze_shot = AsyncMock()
        core_module.subscribe_to_events = Mock(return_value="sub_id")
        core_module.get_performance_metrics = Mock(return_value=Mock())

        vision_module = Mock(spec=VisionModule)
        vision_module.start_capture = Mock(return_value=True)
        vision_module.stop_capture = Mock()
        vision_module.process_frame = Mock(return_value=None)
        vision_module.get_statistics = Mock(return_value={})
        vision_module.subscribe_to_events = Mock(return_value="sub_id")

        return {"config": config_module, "core": core_module, "vision": vision_module}

    @pytest.fixture()
    async def integration(self, integration_config, mock_modules):
        """Create integration instance with mocked modules."""
        with patch(
            "backend.api.integration.ConfigurationModule",
            return_value=mock_modules["config"],
        ), patch(
            "backend.api.integration.CoreModule", return_value=mock_modules["core"]
        ), patch(
            "backend.api.integration.VisionModule", return_value=mock_modules["vision"]
        ):
            integration = APIIntegration(integration_config)
            await integration.startup()
            yield integration
            await integration.shutdown()

    async def test_integration_startup(self, integration_config):
        """Test integration startup sequence."""
        with patch("backend.api.integration.ConfigurationModule"), patch(
            "backend.api.integration.CoreModule"
        ):
            integration = APIIntegration(integration_config)

            # Test startup
            await integration.startup()

            # Verify modules were initialized
            assert integration.config_module is not None
            assert integration.core_module is not None

            # Verify services were initialized
            assert integration.game_service is not None
            assert integration.config_service is not None

            await integration.shutdown()

    async def test_integration_startup_failure(self, integration_config):
        """Test integration startup failure handling."""
        with patch(
            "backend.api.integration.ConfigurationModule",
            side_effect=Exception("Config failed"),
        ):
            integration = APIIntegration(integration_config)

            with pytest.raises(IntegrationError):
                await integration.startup()

    async def test_service_access(self, integration):
        """Test service access methods."""
        # Test successful service access
        game_service = integration.get_game_service()
        assert isinstance(game_service, GameStateService)

        config_service = integration.get_config_service()
        assert isinstance(config_service, ConfigurationService)

        # Test service unavailable
        integration.game_service = None
        with pytest.raises(ServiceUnavailableError):
            integration.get_game_service()

    async def test_caching_functionality(self, integration):
        """Test caching operations."""
        # Test cache set and get
        await integration.set_cached("test_key", "test_value")
        cached_value = await integration.get_cached("test_key")
        assert cached_value == "test_value"

        # Test cache miss
        missing_value = await integration.get_cached("nonexistent_key")
        assert missing_value is None

        # Test cache clear
        await integration.clear_cache()
        cleared_value = await integration.get_cached("test_key")
        assert cleared_value is None

    async def test_cache_expiration(self, integration_config):
        """Test cache TTL expiration."""
        config = integration_config
        config.cache_ttl = 1  # 1 second TTL

        integration = APIIntegration(config)
        await integration.startup()

        try:
            # Set a value
            await integration.set_cached("expire_test", "value")

            # Should be available immediately
            value = await integration.get_cached("expire_test")
            assert value == "value"

            # Wait for expiration
            await asyncio.sleep(1.1)

            # Should be expired
            expired_value = await integration.get_cached("expire_test")
            assert expired_value is None

        finally:
            await integration.shutdown()

    async def test_health_checks(self, integration, mock_modules):
        """Test health check functionality."""
        # Test overall health status
        health = await integration.get_health_status()
        assert "status" in health
        assert "services" in health
        assert "timestamp" in health

        # Test individual service health check
        service_health = await integration.check_service_health("config")
        assert service_health.name == "config"
        assert service_health.status in ["healthy", "degraded", "unhealthy"]

    async def test_event_handling(self, integration):
        """Test event subscription and handling."""
        # Test event queue and processing
        event_data = {"test": "data", "timestamp": time.time()}

        # Queue an event
        await integration._queue_event("test_event", event_data)

        # Event should be queued
        assert not integration._event_queue.empty()

    async def test_module_configuration_integration(self, integration, mock_modules):
        """Test configuration integration with modules."""
        # Test getting core config from settings
        config_data = await integration._get_core_config_from_settings()
        assert isinstance(config_data, (dict, type(None)))

        # Test getting vision config from settings
        vision_config = await integration._get_vision_config_from_settings()
        assert isinstance(vision_config, (dict, type(None)))

    async def test_error_handling(self, integration, mock_modules):
        """Test error handling and recovery."""
        # Test service health check with error
        mock_modules["config"].get.side_effect = Exception("Config error")

        service_health = await integration.check_service_health("config")
        assert service_health.status == "unhealthy"
        assert "Config error" in service_health.error_message


class TestGameStateService:
    """Test suite for Game State Service."""

    @pytest.fixture()
    def mock_core_module(self):
        """Mock core module for testing."""
        mock = Mock(spec=CoreModule)
        mock.get_current_state = Mock(return_value=None)
        mock.update_state = AsyncMock()
        mock.analyze_shot = AsyncMock()
        mock.suggest_shots = AsyncMock(return_value=[])
        return mock

    @pytest.fixture()
    def mock_integration(self):
        """Mock integration for testing."""
        mock = Mock(spec=APIIntegration)
        mock.get_cached = AsyncMock(return_value=None)
        mock.set_cached = AsyncMock()
        return mock

    @pytest.fixture()
    def game_service(self, mock_core_module, mock_integration):
        """Create game service with mocks."""
        return GameStateService(mock_core_module, mock_integration)

    async def test_get_current_state_cache_miss(
        self, game_service, mock_core_module, mock_integration
    ):
        """Test getting current state with cache miss."""
        # Mock cache miss
        mock_integration.get_cached.return_value = None

        # Mock core module state
        mock_state = Mock()
        mock_core_module.get_current_state.return_value = mock_state

        # Test getting state
        with patch("backend.api.integration.asdict", return_value={"test": "state"}):
            state = await game_service.get_current_state()
            assert state == {"test": "state"}

        # Verify cache was set
        mock_integration.set_cached.assert_called_once()

    async def test_get_current_state_cache_hit(self, game_service, mock_integration):
        """Test getting current state with cache hit."""
        # Mock cache hit
        cached_state = {"cached": "state"}
        mock_integration.get_cached.return_value = cached_state

        state = await game_service.get_current_state()
        assert state == cached_state

    async def test_update_state(self, game_service, mock_core_module, mock_integration):
        """Test updating game state."""
        detection_data = {"balls": [], "table": {}}
        mock_state = Mock()
        mock_core_module.update_state.return_value = mock_state

        with patch("backend.api.integration.asdict", return_value={"updated": "state"}):
            result = await game_service.update_state(detection_data)
            assert result == {"updated": "state"}

        # Verify core module was called
        mock_core_module.update_state.assert_called_once_with(detection_data)

        # Verify cache was updated
        mock_integration.set_cached.assert_called()

    async def test_analyze_shot(self, game_service, mock_core_module, mock_integration):
        """Test shot analysis."""
        mock_analysis = Mock()
        mock_core_module.analyze_shot.return_value = mock_analysis

        with patch(
            "backend.api.integration.asdict", return_value={"analysis": "result"}
        ):
            result = await game_service.analyze_shot("target_ball")
            assert result == {"analysis": "result"}

        mock_core_module.analyze_shot.assert_called_once_with("target_ball")

    async def test_suggest_shots(
        self, game_service, mock_core_module, mock_integration
    ):
        """Test shot suggestions."""
        mock_suggestions = [Mock(), Mock()]
        mock_core_module.suggest_shots.return_value = mock_suggestions

        with patch(
            "backend.api.integration.asdict", side_effect=[{"shot": "1"}, {"shot": "2"}]
        ):
            result = await game_service.suggest_shots(difficulty_filter=0.5)
            assert result == [{"shot": "1"}, {"shot": "2"}]

        mock_core_module.suggest_shots.assert_called_once_with(difficulty_filter=0.5)


class TestConfigurationService:
    """Test suite for Configuration Service."""

    @pytest.fixture()
    def mock_config_module(self):
        """Mock configuration module."""
        mock = Mock(spec=ConfigurationModule)
        mock.get = Mock(return_value="test_value")
        mock.set = Mock()
        mock.get_all = Mock(return_value={"key1": "value1", "key2": "value2"})
        return mock

    @pytest.fixture()
    def mock_integration(self):
        """Mock integration for testing."""
        mock = Mock(spec=APIIntegration)
        mock.clear_cache = AsyncMock()
        return mock

    @pytest.fixture()
    def config_service(self, mock_config_module, mock_integration):
        """Create configuration service with mocks."""
        return ConfigurationService(mock_config_module, mock_integration)

    async def test_get_config(self, config_service, mock_config_module):
        """Test getting configuration value."""
        result = await config_service.get_config("test.key")
        assert result == "test_value"
        mock_config_module.get.assert_called_once_with("test.key", None)

    async def test_set_config(
        self, config_service, mock_config_module, mock_integration
    ):
        """Test setting configuration value."""
        await config_service.set_config("test.key", "new_value")

        mock_config_module.set.assert_called_once_with("test.key", "new_value")
        mock_integration.clear_cache.assert_called_once()

    async def test_get_all_configs(
        self, config_service, mock_config_module, mock_integration
    ):
        """Test getting all configurations."""
        # Mock cache miss
        mock_integration.get_cached = AsyncMock(return_value=None)
        mock_integration.set_cached = AsyncMock()

        result = await config_service.get_all_configs()
        assert result == {"key1": "value1", "key2": "value2"}

        mock_config_module.get_all.assert_called_once()
        mock_integration.set_cached.assert_called_once()


class TestDetectionService:
    """Test suite for Detection Service."""

    @pytest.fixture()
    def mock_vision_module(self):
        """Mock vision module."""
        mock = Mock(spec=VisionModule)
        mock.start_capture = Mock(return_value=True)
        mock.stop_capture = Mock()
        mock.process_frame = Mock(return_value=None)
        mock.get_statistics = Mock(return_value={"frames_processed": 100})
        return mock

    @pytest.fixture()
    def mock_integration(self):
        """Mock integration for testing."""
        return Mock(spec=APIIntegration)

    @pytest.fixture()
    def detection_service(self, mock_vision_module, mock_integration):
        """Create detection service with mocks."""
        return DetectionService(mock_vision_module, mock_integration)

    async def test_start_detection(self, detection_service, mock_vision_module):
        """Test starting detection."""
        result = await detection_service.start_detection()

        assert result["success"] is True
        mock_vision_module.start_capture.assert_called_once()

    async def test_stop_detection(self, detection_service, mock_vision_module):
        """Test stopping detection."""
        result = await detection_service.stop_detection()

        assert result["success"] is True
        mock_vision_module.stop_capture.assert_called_once()

    async def test_get_latest_detection(self, detection_service, mock_vision_module):
        """Test getting latest detection."""
        mock_result = Mock()
        mock_vision_module.process_frame.return_value = mock_result

        with patch(
            "backend.api.integration.asdict", return_value={"detection": "result"}
        ):
            result = await detection_service.get_latest_detection()
            assert result == {"detection": "result"}

    async def test_get_detection_statistics(
        self, detection_service, mock_vision_module
    ):
        """Test getting detection statistics."""
        result = await detection_service.get_detection_statistics()

        assert "frames_processed" in result
        assert result["frames_processed"] == 100
        mock_vision_module.get_statistics.assert_called_once()


class TestIntegrationPerformance:
    """Test suite for integration performance."""

    async def test_concurrent_cache_operations(self):
        """Test concurrent cache operations."""
        config = IntegrationConfig(cache_ttl=60)
        integration = APIIntegration(config)

        try:
            # Perform concurrent cache operations
            tasks = []
            for i in range(10):
                tasks.append(integration.set_cached(f"key_{i}", f"value_{i}"))

            await asyncio.gather(*tasks)

            # Verify all values were cached
            for i in range(10):
                value = await integration.get_cached(f"key_{i}")
                assert value == f"value_{i}"

        finally:
            await integration.shutdown()

    async def test_cache_lru_eviction(self):
        """Test LRU cache eviction."""
        config = IntegrationConfig(cache_ttl=60, max_cache_size=5)
        integration = APIIntegration(config)

        try:
            # Fill cache beyond capacity
            for i in range(10):
                await integration.set_cached(f"key_{i}", f"value_{i}")

            # Check that only the last 5 items remain
            remaining_count = 0
            for i in range(10):
                value = await integration.get_cached(f"key_{i}")
                if value is not None:
                    remaining_count += 1

            assert remaining_count <= 5

        finally:
            await integration.shutdown()

    async def test_health_check_performance(self):
        """Test health check performance."""
        config = IntegrationConfig(module_timeout=0.1)
        integration = APIIntegration(config)

        with patch("backend.api.integration.ConfigurationModule"), patch(
            "backend.api.integration.CoreModule"
        ), patch("backend.api.integration.VisionModule"):
            await integration.startup()

            try:
                start_time = time.time()
                await integration.check_service_health("config")
                end_time = time.time()

                # Health check should complete within timeout
                assert (end_time - start_time) < 1.0

            finally:
                await integration.shutdown()


# Pytest configuration
@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
