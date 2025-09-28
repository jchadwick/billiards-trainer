"""Comprehensive tests for configuration schemas.

This module tests all configuration schemas to ensure they:
- Validate correctly with valid data
- Reject invalid data appropriately
- Have proper defaults and constraints
- Support JSON Schema generation
- Work with configuration inheritance and profiles
"""

from pathlib import Path

import pytest
from pydantic import SecretStr, ValidationError

from .schemas import (  # Base classes; System configuration; Vision configuration; Core configuration; API configuration; Projector configuration; Application configuration; Utility functions
    APIConfig,
    ApplicationConfig,
    AssistanceConfig,
    AssistanceDisplayConfig,
    AuthenticationConfig,
    BaseConfig,
    CameraBackend,
    CameraSettings,
    ColorThresholds,
    ConfigFormat,
    ConfigMetadata,
    ConfigProfileEnhanced,
    ConfigSource,
    CoreConfig,
    CoreValidationConfig,
    CorsConfig,
    DetectionMethod,
    DetectionSettings,
    DisplayConfig,
    DisplayMode,
    ExposureMode,
    LogLevel,
    NetworkConfig,
    PerformanceMode,
    PhysicsConfig,
    PredictionConfig,
    ProcessingSettings,
    ProjectorCalibrationConfig,
    ProjectorConfig,
    ProjectorNetworkConfig,
    RateLimitConfig,
    RenderingConfig,
    RenderQuality,
    SystemConfig,
    SystemLogging,
    SystemPaths,
    SystemPerformance,
    VisionConfig,
    VisualRenderingConfig,
    create_default_config,
    create_development_config,
    create_production_config,
    generate_json_schemas,
)


class TestBaseConfig:
    """Test base configuration functionality."""

    def test_base_config_validation(self):
        """Test that BaseConfig validates correctly."""
        # BaseConfig should reject extra fields due to Extra.forbid
        with pytest.raises(ValidationError):

            class TestConfig(BaseConfig):
                test_field: str = "test"

            TestConfig(test_field="valid", extra_field="invalid")

    def test_enum_validation(self):
        """Test enum validation works correctly."""
        # Valid enum values should work
        assert ConfigSource.DEFAULT == "default"
        assert ConfigFormat.JSON == "json"
        assert LogLevel.INFO == "INFO"

        # Invalid enum values should be rejected in validation
        class TestConfig(BaseConfig):
            source: ConfigSource = ConfigSource.DEFAULT

        with pytest.raises(ValidationError):
            TestConfig(source="invalid_source")


class TestSystemConfig:
    """Test system configuration schemas."""

    def test_system_paths_defaults(self):
        """Test SystemPaths default values."""
        paths = SystemPaths()
        assert paths.config_dir == Path("config")
        assert paths.data_dir == Path("data")
        assert paths.log_dir == Path("logs")
        assert paths.cache_dir == Path(".cache")
        assert paths.profiles_dir == Path("config/profiles")
        assert paths.temp_dir == Path("/tmp")

    def test_system_logging_defaults(self):
        """Test SystemLogging default values."""
        logging = SystemLogging()
        assert logging.level == LogLevel.INFO
        assert logging.file_logging is True
        assert logging.console_logging is True
        assert logging.max_file_size == 10 * 1024 * 1024
        assert logging.backup_count == 5
        assert "vision" in logging.log_modules
        assert "core" in logging.log_modules

    def test_system_performance_validation(self):
        """Test SystemPerformance validation."""
        # Valid values should work
        perf = SystemPerformance(
            mode=PerformanceMode.HIGH,
            max_memory_mb=4096,
            max_cpu_percent=90,
            thread_pool_size=8,
        )
        assert perf.mode == PerformanceMode.HIGH
        assert perf.max_memory_mb == 4096

        # Invalid values should be rejected
        with pytest.raises(ValidationError):
            SystemPerformance(max_memory_mb=100)  # Below minimum

        with pytest.raises(ValidationError):
            SystemPerformance(max_cpu_percent=120)  # Above maximum

    def test_system_config_complete(self):
        """Test complete SystemConfig validation."""
        config = SystemConfig(
            debug=True, environment="development", timezone="America/New_York"
        )
        assert config.debug is True
        assert config.environment == "development"
        assert isinstance(config.paths, SystemPaths)
        assert isinstance(config.logging, SystemLogging)
        assert isinstance(config.performance, SystemPerformance)

    def test_timezone_validation(self):
        """Test timezone validation."""
        # Valid timezones should work
        config = SystemConfig(timezone="UTC")
        assert config.timezone == "UTC"

        config = SystemConfig(timezone="GMT")
        assert config.timezone == "GMT"

        # Invalid timezone should be rejected (if pytz is available)
        try:
            import pytz

            with pytest.raises(ValidationError):
                SystemConfig(timezone="Invalid/Timezone")
        except ImportError:
            # If pytz is not available, only basic validation is performed
            pass


class TestVisionConfig:
    """Test vision configuration schemas."""

    def test_camera_settings_defaults(self):
        """Test CameraSettings default values."""
        camera = CameraSettings()
        assert camera.device_id == 0
        assert camera.backend == CameraBackend.AUTO
        assert camera.resolution == (1920, 1080)
        assert camera.fps == 30
        assert camera.exposure_mode == ExposureMode.AUTO
        assert camera.gain == 1.0
        assert camera.brightness == 0.5
        assert camera.buffer_size == 1
        assert camera.auto_focus is True

    def test_camera_resolution_validation(self):
        """Test camera resolution validation."""
        # Valid resolutions should work
        camera = CameraSettings(resolution=(1920, 1080))
        assert camera.resolution == (1920, 1080)

        # Invalid resolutions should be rejected
        with pytest.raises(ValidationError):
            CameraSettings(resolution=(320, 240))  # Too small

        with pytest.raises(ValidationError):
            CameraSettings(resolution=(8192, 8192))  # Too large

    def test_color_thresholds_validation(self):
        """Test ColorThresholds validation."""
        # Valid thresholds should work
        color = ColorThresholds(
            hue_min=0,
            hue_max=179,
            saturation_min=50,
            saturation_max=255,
            value_min=0,
            value_max=255,
        )
        assert color.hue_min == 0
        assert color.hue_max == 179

        # Invalid range should be rejected
        with pytest.raises(ValidationError):
            ColorThresholds(hue_min=100, hue_max=50)  # Max < Min

    def test_detection_settings_defaults(self):
        """Test DetectionSettings default values."""
        detection = DetectionSettings()
        assert isinstance(detection.table_color, ColorThresholds)
        assert detection.table_edge_threshold == 0.7
        assert detection.min_table_area == 0.3
        assert detection.ball_detection_method == DetectionMethod.HOUGH
        assert detection.ball_sensitivity == 0.8
        assert detection.cue_detection_enabled is True
        assert detection.min_cue_length == 100

    def test_processing_settings_validation(self):
        """Test ProcessingSettings validation."""
        # Valid settings should work
        processing = ProcessingSettings(
            blur_kernel_size=5, morphology_kernel_size=3, tracking_max_distance=50
        )
        assert processing.blur_kernel_size == 5

        # Invalid kernel sizes should be rejected (must be odd)
        with pytest.raises(ValidationError):
            ProcessingSettings(blur_kernel_size=4)  # Even number

        with pytest.raises(ValidationError):
            ProcessingSettings(morphology_kernel_size=6)  # Even number

    def test_vision_config_complete(self):
        """Test complete VisionConfig validation."""
        config = VisionConfig(
            debug=True, save_debug_images=True, debug_output_path=Path("/tmp/debug")
        )
        assert config.debug is True
        assert isinstance(config.camera, CameraSettings)
        assert isinstance(config.detection, DetectionSettings)
        assert isinstance(config.processing, ProcessingSettings)


class TestCoreConfig:
    """Test core configuration schemas."""

    def test_physics_config_defaults(self):
        """Test PhysicsConfig default values."""
        physics = PhysicsConfig()
        assert physics.gravity == 9.81
        assert physics.air_resistance == 0.01
        assert physics.rolling_friction == 0.01
        assert physics.sliding_friction == 0.2
        assert physics.cushion_coefficient == 0.85
        assert physics.spin_decay_rate == 0.95
        assert physics.max_iterations == 1000
        assert physics.time_step == 0.001
        assert physics.enable_spin_effects is True
        assert physics.enable_cushion_compression is True

    def test_prediction_config_validation(self):
        """Test PredictionConfig validation."""
        # Valid values should work
        prediction = PredictionConfig(
            max_prediction_time=15.0,
            prediction_resolution=200,
            collision_threshold=0.002,
        )
        assert prediction.max_prediction_time == 15.0

        # Invalid values should be rejected
        with pytest.raises(ValidationError):
            PredictionConfig(max_prediction_time=0.5)  # Below minimum

        with pytest.raises(ValidationError):
            PredictionConfig(prediction_resolution=5)  # Below minimum

    def test_assistance_config_defaults(self):
        """Test AssistanceConfig default values."""
        assistance = AssistanceConfig()
        assert "beginner" in assistance.difficulty_levels
        assert assistance.difficulty_levels["expert"] == 1.0
        assert assistance.show_alternative_shots is True
        assert assistance.max_alternatives == 3
        assert assistance.highlight_best_shot is True
        assert assistance.enable_shot_analysis is True

    def test_core_validation_config(self):
        """Test CoreValidationConfig validation."""
        validation = CoreValidationConfig(
            max_ball_velocity=15.0,
            min_ball_separation=0.002,
            enable_physics_validation=True,
        )
        assert validation.max_ball_velocity == 15.0
        assert validation.enable_physics_validation is True

    def test_core_config_complete(self):
        """Test complete CoreConfig validation."""
        config = CoreConfig(
            state_history_size=500,
            event_buffer_size=50,
            update_frequency=30.0,
            enable_game_rules=True,
        )
        assert config.state_history_size == 500
        assert isinstance(config.physics, PhysicsConfig)
        assert isinstance(config.prediction, PredictionConfig)
        assert isinstance(config.assistance, AssistanceConfig)
        assert isinstance(config.validation, CoreValidationConfig)


class TestAPIConfig:
    """Test API configuration schemas."""

    def test_authentication_config_validation(self):
        """Test AuthenticationConfig validation."""
        # Valid configuration should work
        auth = AuthenticationConfig(
            jwt_secret_key=SecretStr("secret-key-123"),
            jwt_algorithm="HS256",
            jwt_expiration_hours=48,
        )
        assert auth.jwt_algorithm == "HS256"
        assert auth.jwt_expiration_hours == 48

        # Invalid values should be rejected
        with pytest.raises(ValidationError):
            AuthenticationConfig(
                jwt_secret_key=SecretStr("secret"),
                jwt_expiration_hours=200,  # Above maximum
            )

    def test_cors_config_defaults(self):
        """Test CorsConfig default values."""
        cors = CorsConfig()
        assert cors.enabled is True
        assert "*" in cors.allow_origins
        assert "GET" in cors.allow_methods
        assert "POST" in cors.allow_methods
        assert cors.allow_credentials is True
        assert cors.max_age == 600

    def test_rate_limit_config_validation(self):
        """Test RateLimitConfig validation."""
        rate_limit = RateLimitConfig(
            enabled=True, requests_per_minute=200, burst_size=50
        )
        assert rate_limit.requests_per_minute == 200
        assert rate_limit.burst_size == 50

        # Invalid values should be rejected
        with pytest.raises(ValidationError):
            RateLimitConfig(requests_per_minute=20000)  # Above maximum

    def test_network_config_validation(self):
        """Test NetworkConfig validation."""
        network = NetworkConfig(host="127.0.0.1", port=8080, max_connections=200)
        assert network.host == "127.0.0.1"
        assert network.port == 8080

        # Invalid port should be rejected
        with pytest.raises(ValidationError):
            NetworkConfig(port=70000)  # Above maximum

    def test_api_config_complete(self):
        """Test complete APIConfig validation."""
        config = APIConfig(
            authentication=AuthenticationConfig(
                jwt_secret_key=SecretStr("test-secret")
            ),
            enable_docs=False,
            api_prefix="/api/v2",
        )
        assert config.enable_docs is False
        assert config.api_prefix == "/api/v2"
        assert isinstance(config.network, NetworkConfig)
        assert isinstance(config.cors, CorsConfig)
        assert isinstance(config.rate_limiting, RateLimitConfig)


class TestProjectorConfig:
    """Test projector configuration schemas."""

    def test_display_config_defaults(self):
        """Test DisplayConfig default values."""
        display = DisplayConfig()
        assert display.mode == DisplayMode.FULLSCREEN
        assert display.monitor_index == 0
        assert display.resolution == (1920, 1080)
        assert display.refresh_rate == 60
        assert display.vsync is True
        assert display.gamma == 1.0
        assert display.brightness == 1.0
        assert display.contrast == 1.0

    def test_projector_calibration_validation(self):
        """Test ProjectorCalibrationConfig validation."""
        # Valid calibration should work
        calibration = ProjectorCalibrationConfig(
            calibration_points=[(0, 0), (1920, 0), (1920, 1080), (0, 1080)],
            keystone_horizontal=0.1,
            rotation=5.0,
        )
        assert len(calibration.calibration_points) == 4
        assert calibration.keystone_horizontal == 0.1

        # Invalid calibration points should be rejected
        with pytest.raises(ValidationError):
            ProjectorCalibrationConfig(
                calibration_points=[(0, 0), (1920, 0)]  # Only 2 points
            )

    def test_visual_rendering_config_defaults(self):
        """Test VisualRenderingConfig default values."""
        visual = VisualRenderingConfig()
        assert visual.trajectory_width == 3.0
        assert visual.trajectory_color == (0, 255, 0)
        assert visual.collision_color == (255, 0, 0)
        assert visual.enable_glow is True
        assert visual.glow_intensity == 0.5
        assert visual.enable_animations is True
        assert visual.font_family == "Arial"
        assert visual.font_size == 24

    def test_rendering_config_validation(self):
        """Test RenderingConfig validation."""
        rendering = RenderingConfig(
            renderer="vulkan", quality=RenderQuality.ULTRA, max_fps=120, use_gpu=True
        )
        assert rendering.renderer == "vulkan"
        assert rendering.quality == RenderQuality.ULTRA
        assert rendering.max_fps == 120

        # Invalid values should be rejected
        with pytest.raises(ValidationError):
            RenderingConfig(max_fps=200)  # Above maximum

    def test_projector_config_complete(self):
        """Test complete ProjectorConfig validation."""
        config = ProjectorConfig(debug=True, debug_overlay=True)
        assert config.debug is True
        assert config.debug_overlay is True
        assert isinstance(config.display, DisplayConfig)
        assert isinstance(config.calibration, ProjectorCalibrationConfig)
        assert isinstance(config.visual, VisualRenderingConfig)
        assert isinstance(config.rendering, RenderingConfig)
        assert isinstance(config.network, ProjectorNetworkConfig)
        assert isinstance(config.assistance, AssistanceDisplayConfig)


class TestApplicationConfig:
    """Test complete application configuration."""

    def test_application_config_defaults(self):
        """Test ApplicationConfig creates with proper defaults."""
        config = ApplicationConfig(
            api=APIConfig(
                authentication=AuthenticationConfig(
                    jwt_secret_key=SecretStr("test-secret")
                )
            )
        )
        assert isinstance(config.metadata, ConfigMetadata)
        assert isinstance(config.system, SystemConfig)
        assert isinstance(config.vision, VisionConfig)
        assert isinstance(config.core, CoreConfig)
        assert isinstance(config.api, APIConfig)
        assert isinstance(config.projector, ProjectorConfig)

    def test_profile_validation(self):
        """Test configuration profile validation."""
        # Valid profile should work
        profile = ConfigProfileEnhanced(
            name="test-profile",
            description="Test profile",
            settings={"system.debug": True},
        )
        assert profile.name == "test-profile"
        assert profile.settings["system.debug"] is True

        # Application config with valid profile should work
        config = ApplicationConfig(
            api=APIConfig(
                authentication=AuthenticationConfig(jwt_secret_key=SecretStr("test"))
            ),
            profiles={"test": profile},
            active_profile="test",
        )
        assert config.active_profile == "test"

    def test_profile_inheritance_validation(self):
        """Test profile inheritance validation."""
        parent_profile = ConfigProfileEnhanced(
            name="parent", settings={"system.debug": False}
        )
        child_profile = ConfigProfileEnhanced(
            name="child", parent="parent", settings={"system.environment": "test"}
        )

        # Valid inheritance should work
        config = ApplicationConfig(
            api=APIConfig(
                authentication=AuthenticationConfig(jwt_secret_key=SecretStr("test"))
            ),
            profiles={"parent": parent_profile, "child": child_profile},
        )
        assert "parent" in config.profiles
        assert "child" in config.profiles

        # Invalid inheritance should be rejected
        with pytest.raises(ValidationError):
            ApplicationConfig(
                api=APIConfig(
                    authentication=AuthenticationConfig(
                        jwt_secret_key=SecretStr("test")
                    )
                ),
                profiles={"child": child_profile},  # Missing parent
            )

    def test_profile_merging(self):
        """Test profile merging functionality."""
        profile = ConfigProfileEnhanced(
            name="test",
            settings={
                "system.debug": True,
                "vision.debug": True,
                "api.enable_docs": False,
            },
        )

        config = ApplicationConfig(
            api=APIConfig(
                authentication=AuthenticationConfig(jwt_secret_key=SecretStr("test"))
            ),
            profiles={"test": profile},
        )

        # Merge profile should apply settings
        merged = config.merge_profile("test")
        assert merged.system.debug is True
        assert merged.vision.debug is True
        assert merged.api.enable_docs is False

    def test_json_schema_generation(self):
        """Test JSON schema generation."""
        config = ApplicationConfig(
            api=APIConfig(
                authentication=AuthenticationConfig(jwt_secret_key=SecretStr("test"))
            )
        )

        schema = config.get_json_schema()
        assert "type" in schema
        assert "properties" in schema
        assert "system" in schema["properties"]
        assert "vision" in schema["properties"]
        assert "core" in schema["properties"]
        assert "api" in schema["properties"]
        assert "projector" in schema["properties"]


class TestConfigurationFactories:
    """Test configuration factory functions."""

    def test_create_default_config(self):
        """Test default configuration creation."""
        config = create_default_config()
        assert isinstance(config, ApplicationConfig)
        assert config.system.debug is False
        assert config.system.environment == "development"
        assert isinstance(config.api.authentication.jwt_secret_key, SecretStr)

    def test_create_development_config(self):
        """Test development configuration creation."""
        config = create_development_config()
        assert config.system.debug is True
        assert config.system.logging.level == LogLevel.DEBUG
        assert config.api.enable_docs is True
        assert config.vision.debug is True
        assert config.projector.debug is True

    def test_create_production_config(self):
        """Test production configuration creation."""
        config = create_production_config()
        assert config.system.debug is False
        assert config.system.logging.level == LogLevel.INFO
        assert config.api.enable_docs is False
        assert config.api.network.ssl_enabled is True
        assert config.system.performance.mode == PerformanceMode.HIGH


class TestJSONSchemaGeneration:
    """Test JSON schema generation utilities."""

    def test_generate_json_schemas(self):
        """Test generate_json_schemas function."""
        schemas = generate_json_schemas()

        # Check all expected schemas are present
        expected_schemas = [
            "system",
            "vision",
            "core",
            "api",
            "projector",
            "application",
            "profile",
        ]

        for schema_name in expected_schemas:
            assert schema_name in schemas
            schema = schemas[schema_name]
            assert "type" in schema
            assert "properties" in schema

    def test_individual_schema_generation(self):
        """Test individual schema generation."""
        # Test SystemConfig schema
        schema = SystemConfig.model_json_schema()
        assert schema["type"] == "object"
        assert "debug" in schema["properties"]
        assert "environment" in schema["properties"]
        assert "paths" in schema["properties"]
        assert "logging" in schema["properties"]
        assert "performance" in schema["properties"]

        # Test VisionConfig schema
        schema = VisionConfig.model_json_schema()
        assert "camera" in schema["properties"]
        assert "detection" in schema["properties"]
        assert "processing" in schema["properties"]

        # Test field descriptions are present
        debug_field = schema["properties"]["debug"]
        assert "description" in debug_field
        assert debug_field["description"] == "Enable vision debug mode"


class TestConfigurationCompatibility:
    """Test backward compatibility and aliases."""

    def test_backward_compatibility_aliases(self):
        """Test that backward compatibility aliases work."""
        # Import aliases to ensure they exist
        from .schemas import CameraConfig, ConfigurationSettings

        # ConfigurationSettings should be an alias for ApplicationConfig
        assert ConfigurationSettings is ApplicationConfig

        # CameraConfig should be an alias for CameraSettings
        assert CameraConfig is CameraSettings

    def test_legacy_dataclass_compatibility(self):
        """Test that legacy dataclasses still work."""
        from .schemas import ConfigChange, ConfigProfile, ConfigValue

        # Test ConfigValue dataclass
        config_value = ConfigValue(
            key="test.key",
            value="test_value",
            source=ConfigSource.DEFAULT,
            timestamp=1234567890.0,
            validated=True,
        )
        assert config_value.key == "test.key"
        assert config_value.source == ConfigSource.DEFAULT

        # Test ConfigChange dataclass
        config_change = ConfigChange(
            key="test.key",
            old_value="old",
            new_value="new",
            source=ConfigSource.RUNTIME,
            timestamp=1234567890.0,
            applied=True,
        )
        assert config_change.old_value == "old"
        assert config_change.new_value == "new"

        # Test ConfigProfile dataclass
        config_profile = ConfigProfile(
            name="test", description="Test profile", settings={"key": "value"}
        )
        assert config_profile.name == "test"
        assert config_profile.settings["key"] == "value"


# Integration tests
class TestFullSystemIntegration:
    """Test full system configuration integration."""

    def test_complete_configuration_validation(self):
        """Test that a complete configuration validates properly."""
        config = ApplicationConfig(
            metadata=ConfigMetadata(version="2.0.0", environment="test"),
            system=SystemConfig(
                debug=True,
                environment="test",
                performance=SystemPerformance(
                    mode=PerformanceMode.HIGH, max_memory_mb=4096
                ),
            ),
            vision=VisionConfig(
                camera=CameraSettings(device_id=1, resolution=(2560, 1440), fps=60),
                debug=True,
            ),
            core=CoreConfig(
                physics=PhysicsConfig(gravity=9.81, enable_spin_effects=True),
                enable_game_rules=True,
            ),
            api=APIConfig(
                authentication=AuthenticationConfig(
                    jwt_secret_key=SecretStr("test-secret-key-123")
                ),
                network=NetworkConfig(port=8080, max_connections=200),
            ),
            projector=ProjectorConfig(
                display=DisplayConfig(
                    mode=DisplayMode.FULLSCREEN, resolution=(1920, 1080)
                ),
                debug=True,
            ),
        )

        # All configurations should be valid
        assert config.metadata.version == "2.0.0"
        assert config.system.debug is True
        assert config.vision.camera.fps == 60
        assert config.core.enable_game_rules is True
        assert config.api.network.port == 8080
        assert config.projector.debug is True

    def test_configuration_serialization(self):
        """Test configuration serialization to/from dict."""
        config = create_development_config()

        # Convert to dict
        config_dict = config.model_dump()
        assert isinstance(config_dict, dict)
        assert "system" in config_dict
        assert "vision" in config_dict

        # Recreate from dict
        recreated = ApplicationConfig(**config_dict)
        assert recreated.system.debug == config.system.debug
        assert recreated.vision.debug == config.vision.debug

    def test_configuration_json_export(self):
        """Test configuration JSON export."""
        import json

        config = create_default_config()

        # Convert to JSON
        config_json = config.model_dump_json()
        parsed = json.loads(config_json)

        assert isinstance(parsed, dict)
        assert "system" in parsed
        assert "api" in parsed

        # Recreate from JSON
        recreated = ApplicationConfig.model_validate_json(config_json)
        assert isinstance(recreated, ApplicationConfig)


if __name__ == "__main__":
    # Run basic validation tests
    print("Running basic configuration schema validation tests...")

    try:
        # Test all main configuration classes can be instantiated
        system_config = SystemConfig()
        print("✓ SystemConfig validation passed")

        vision_config = VisionConfig()
        print("✓ VisionConfig validation passed")

        core_config = CoreConfig()
        print("✓ CoreConfig validation passed")

        api_config = APIConfig(
            authentication=AuthenticationConfig(jwt_secret_key=SecretStr("test-secret"))
        )
        print("✓ APIConfig validation passed")

        projector_config = ProjectorConfig()
        print("✓ ProjectorConfig validation passed")

        # Test complete application configuration
        app_config = create_default_config()
        print("✓ ApplicationConfig validation passed")

        # Test JSON schema generation
        schemas = generate_json_schemas()
        print(f"✓ JSON schema generation passed - {len(schemas)} schemas generated")

        # Test configuration factories
        dev_config = create_development_config()
        prod_config = create_production_config()
        print("✓ Configuration factory functions passed")

        print("\nAll basic validation tests passed! ✅")
        print(
            "Run 'pytest backend/config/models/test_schemas.py' for comprehensive tests."
        )

    except Exception as e:
        print(f"❌ Validation test failed: {e}")
        raise
