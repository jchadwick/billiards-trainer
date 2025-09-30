"""Unit tests for the projector module."""

import time
from unittest.mock import MagicMock, mock_open, patch

import numpy as np
import pytest
from projector.calibration.geometric import GeometricCalibrator
from projector.config.manager import ProjectorConfigManager
from projector.models import Overlay, ProjectorState, RenderObject
from projector.rendering.opengl.renderer import OpenGLRenderer
from projector.rendering.opengl.shaders import ShaderManager
from projector.rendering.opengl.textures import TextureManager
from projector.utils.math import create_projection_matrix, transform_point
from vision.calibration.color import ColorCalibrator as ProjectorColorCalibrator


@pytest.mark.unit()
class TestProjectorState:
    """Test the projector state model."""

    def test_state_creation(self):
        """Test creating projector state."""
        state = ProjectorState(
            position=(0, 0, 2.0),
            rotation=(0, 0, 0),
            brightness=1.0,
            contrast=1.0,
            is_active=True,
        )

        assert state.position == (0, 0, 2.0)
        assert state.rotation == (0, 0, 0)
        assert state.brightness == 1.0
        assert state.is_active is True

    def test_state_validation(self):
        """Test projector state validation."""
        # Valid state
        state = ProjectorState(
            position=(0, 0, 2.0), rotation=(0, 0, 0), brightness=0.8, contrast=1.2
        )
        assert state.brightness == 0.8
        assert state.contrast == 1.2

        # Test brightness bounds
        with pytest.raises(ValueError):
            ProjectorState(brightness=-0.1)  # Below 0

        with pytest.raises(ValueError):
            ProjectorState(brightness=1.1)  # Above 1

    def test_state_update(self):
        """Test updating projector state."""
        state = ProjectorState()

        state.update_position(1.0, 0.5, 2.5)
        assert state.position == (1.0, 0.5, 2.5)

        state.update_rotation(10, 5, 0)
        assert state.rotation == (10, 5, 0)

        state.update_brightness(0.7)
        assert state.brightness == 0.7

    def test_state_serialization(self):
        """Test projector state serialization."""
        state = ProjectorState(
            position=(1, 2, 3), rotation=(10, 20, 30), brightness=0.8, contrast=1.1
        )

        serialized = state.to_dict()
        assert serialized["position"] == [1, 2, 3]
        assert serialized["rotation"] == [10, 20, 30]
        assert serialized["brightness"] == 0.8

        # Test deserialization
        new_state = ProjectorState.from_dict(serialized)
        assert new_state.position == state.position
        assert new_state.brightness == state.brightness


@pytest.mark.unit()
class TestRenderObject:
    """Test the render object model."""

    def test_render_object_creation(self):
        """Test creating render objects."""
        obj = RenderObject(
            object_type="circle",
            position=(100, 200),
            size=(50, 50),
            color=(255, 0, 0),
            z_index=1,
        )

        assert obj.object_type == "circle"
        assert obj.position == (100, 200)
        assert obj.color == (255, 0, 0)
        assert obj.z_index == 1

    def test_render_object_with_animation(self):
        """Test render object with animation."""
        obj = RenderObject(
            object_type="line",
            position=(0, 0),
            end_position=(100, 100),
            color=(0, 255, 0),
            animation={"type": "fade", "duration": 2.0, "start_time": time.time()},
        )

        assert obj.animation["type"] == "fade"
        assert obj.animation["duration"] == 2.0

    def test_render_object_visibility(self):
        """Test render object visibility calculations."""
        obj = RenderObject(
            object_type="circle",
            position=(100, 200),
            size=(50, 50),
            color=(255, 0, 0),
            visible=True,
        )

        assert obj.is_visible()

        obj.visible = False
        assert not obj.is_visible()

    def test_render_object_bounds(self):
        """Test render object bounds calculation."""
        obj = RenderObject(
            object_type="rectangle",
            position=(100, 100),
            size=(50, 30),
            color=(0, 0, 255),
        )

        bounds = obj.get_bounds()
        assert bounds["x"] == 100
        assert bounds["y"] == 100
        assert bounds["width"] == 50
        assert bounds["height"] == 30

    def test_render_object_collision(self):
        """Test render object collision detection."""
        obj1 = RenderObject(
            object_type="circle", position=(100, 100), size=(20, 20), color=(255, 0, 0)
        )

        obj2 = RenderObject(
            object_type="circle", position=(110, 110), size=(20, 20), color=(0, 255, 0)
        )

        obj3 = RenderObject(
            object_type="circle", position=(200, 200), size=(20, 20), color=(0, 0, 255)
        )

        assert obj1.overlaps(obj2)  # Close objects overlap
        assert not obj1.overlaps(obj3)  # Distant objects don't overlap


@pytest.mark.unit()
class TestOverlay:
    """Test the overlay model."""

    def test_overlay_creation(self):
        """Test creating overlay."""
        overlay = Overlay(name="shot_prediction", objects=[], visible=True, z_index=10)

        assert overlay.name == "shot_prediction"
        assert overlay.visible is True
        assert overlay.z_index == 10

    def test_overlay_add_objects(self):
        """Test adding objects to overlay."""
        overlay = Overlay(name="test")

        circle = RenderObject(
            object_type="circle", position=(100, 100), size=(20, 20), color=(255, 0, 0)
        )

        overlay.add_object(circle)
        assert len(overlay.objects) == 1
        assert overlay.objects[0] == circle

    def test_overlay_remove_objects(self):
        """Test removing objects from overlay."""
        overlay = Overlay(name="test")

        circle = RenderObject(
            object_type="circle", position=(100, 100), size=(20, 20), color=(255, 0, 0)
        )

        overlay.add_object(circle)
        overlay.remove_object(circle)
        assert len(overlay.objects) == 0

    def test_overlay_clear(self):
        """Test clearing overlay."""
        overlay = Overlay(name="test")

        for i in range(5):
            obj = RenderObject(
                object_type="circle",
                position=(i * 50, i * 50),
                size=(20, 20),
                color=(255, 0, 0),
            )
            overlay.add_object(obj)

        assert len(overlay.objects) == 5

        overlay.clear()
        assert len(overlay.objects) == 0

    def test_overlay_filtering(self):
        """Test filtering objects in overlay."""
        overlay = Overlay(name="test")

        # Add different types of objects
        circle = RenderObject(object_type="circle", position=(0, 0), color=(255, 0, 0))
        line = RenderObject(object_type="line", position=(0, 0), color=(0, 255, 0))
        rect = RenderObject(object_type="rectangle", position=(0, 0), color=(0, 0, 255))

        overlay.add_object(circle)
        overlay.add_object(line)
        overlay.add_object(rect)

        circles = overlay.get_objects_by_type("circle")
        assert len(circles) == 1
        assert circles[0] == circle

        lines = overlay.get_objects_by_type("line")
        assert len(lines) == 1
        assert lines[0] == line


@pytest.mark.unit()
class TestOpenGLRenderer:
    """Test the OpenGL renderer."""

    def test_renderer_creation(self, mock_opengl_context):
        """Test creating OpenGL renderer."""
        renderer = OpenGLRenderer(width=1920, height=1080)
        assert renderer is not None
        assert renderer.width == 1920
        assert renderer.height == 1080

    @patch("moderngl.create_context")
    def test_renderer_initialization(self, mock_create_context):
        """Test renderer initialization."""
        mock_context = MagicMock()
        mock_create_context.return_value = mock_context

        renderer = OpenGLRenderer()
        renderer.initialize()

        # Should have created context and shaders
        assert renderer.context is not None
        mock_create_context.assert_called_once()

    def test_renderer_viewport_setup(self, mock_opengl_context):
        """Test viewport setup."""
        renderer = OpenGLRenderer(width=1920, height=1080)
        renderer.context = mock_opengl_context

        renderer.set_viewport(0, 0, 1920, 1080)

        # Verify viewport was set
        mock_opengl_context.viewport = (0, 0, 1920, 1080)

    def test_renderer_clear_screen(self, mock_opengl_context):
        """Test clearing screen."""
        renderer = OpenGLRenderer()
        renderer.context = mock_opengl_context

        renderer.clear(color=(0.0, 0.0, 0.0, 1.0))

        # Should have called clear
        mock_opengl_context.clear.assert_called()

    def test_render_circle(self, mock_opengl_context):
        """Test rendering circle."""
        renderer = OpenGLRenderer()
        renderer.context = mock_opengl_context

        circle = RenderObject(
            object_type="circle", position=(100, 100), size=(50, 50), color=(255, 0, 0)
        )

        # Mock shader program
        mock_program = MagicMock()
        renderer.shader_manager = MagicMock()
        renderer.shader_manager.get_program.return_value = mock_program

        renderer.render_object(circle)

        # Should have used shader program
        renderer.shader_manager.get_program.assert_called_with("circle")

    def test_render_line(self, mock_opengl_context):
        """Test rendering line."""
        renderer = OpenGLRenderer()
        renderer.context = mock_opengl_context

        line = RenderObject(
            object_type="line",
            position=(0, 0),
            end_position=(100, 100),
            color=(0, 255, 0),
            thickness=2.0,
        )

        # Mock shader program
        mock_program = MagicMock()
        renderer.shader_manager = MagicMock()
        renderer.shader_manager.get_program.return_value = mock_program

        renderer.render_object(line)

        # Should have rendered line
        renderer.shader_manager.get_program.assert_called_with("line")

    def test_render_text(self, mock_opengl_context):
        """Test rendering text."""
        renderer = OpenGLRenderer()
        renderer.context = mock_opengl_context

        text = RenderObject(
            object_type="text",
            position=(100, 100),
            color=(255, 255, 255),
            text="Test Text",
            font_size=24,
        )

        # Mock font rendering
        renderer.font_manager = MagicMock()

        renderer.render_object(text)

        # Should have used font manager
        assert renderer.font_manager is not None

    def test_render_overlay(self, mock_opengl_context):
        """Test rendering overlay."""
        renderer = OpenGLRenderer()
        renderer.context = mock_opengl_context
        renderer.shader_manager = MagicMock()

        overlay = Overlay(name="test")
        circle = RenderObject(
            object_type="circle", position=(100, 100), size=(20, 20), color=(255, 0, 0)
        )
        overlay.add_object(circle)

        renderer.render_overlay(overlay)

        # Should have rendered all objects in overlay
        renderer.shader_manager.get_program.assert_called()

    def test_performance_monitoring(self, mock_opengl_context):
        """Test rendering performance monitoring."""
        renderer = OpenGLRenderer()
        renderer.context = mock_opengl_context

        # Enable performance monitoring
        renderer.enable_profiling()

        # Render some objects
        for i in range(10):
            circle = RenderObject(
                object_type="circle",
                position=(i * 50, i * 50),
                size=(20, 20),
                color=(255, 0, 0),
            )
            renderer.render_object(circle)

        stats = renderer.get_performance_stats()
        assert "render_time" in stats
        assert "objects_rendered" in stats


@pytest.mark.unit()
class TestShaderManager:
    """Test the shader manager."""

    def test_manager_creation(self):
        """Test creating shader manager."""
        manager = ShaderManager()
        assert manager is not None

    @patch("moderngl.Context")
    def test_load_shader_program(self, mock_context):
        """Test loading shader program."""
        manager = ShaderManager()
        manager.context = mock_context

        # Mock shader source
        vertex_source = """
        #version 330 core
        layout (location = 0) in vec3 position;
        void main() {
            gl_Position = vec4(position, 1.0);
        }
        """

        with patch("builtins.open", mock_open(read_data=vertex_source)):
            manager.load_program("test", "vertex.glsl", "fragment.glsl")

        assert "test" in manager.programs

    def test_get_shader_program(self):
        """Test getting shader program."""
        manager = ShaderManager()

        # Mock program
        mock_program = MagicMock()
        manager.programs["test"] = mock_program

        program = manager.get_program("test")
        assert program == mock_program

        # Test non-existent program
        with pytest.raises(KeyError):
            manager.get_program("nonexistent")

    def test_shader_compilation_error(self):
        """Test handling shader compilation errors."""
        manager = ShaderManager()

        # Mock context that throws compilation error
        mock_context = MagicMock()
        mock_context.program.side_effect = Exception("Compilation error")
        manager.context = mock_context

        with pytest.raises(Exception):
            manager.load_program("test", "vertex.glsl", "fragment.glsl")

    def test_uniform_setting(self):
        """Test setting shader uniforms."""
        manager = ShaderManager()

        mock_program = MagicMock()
        manager.programs["test"] = mock_program

        manager.set_uniform("test", "resolution", (1920, 1080))

        # Should have set uniform on program
        mock_program.__setitem__.assert_called_with("resolution", (1920, 1080))


@pytest.mark.unit()
class TestTextureManager:
    """Test the texture manager."""

    def test_manager_creation(self):
        """Test creating texture manager."""
        manager = TextureManager()
        assert manager is not None

    @patch("PIL.Image.open")
    def test_load_texture_from_file(self, mock_image_open):
        """Test loading texture from file."""
        # Mock PIL Image
        mock_image = MagicMock()
        mock_image.size = (256, 256)
        mock_image.mode = "RGBA"
        mock_image.tobytes.return_value = b"\x00" * (256 * 256 * 4)
        mock_image_open.return_value = mock_image

        manager = TextureManager()
        manager.context = MagicMock()

        texture = manager.load_from_file("test.png")
        assert texture is not None

    def test_create_texture_from_data(self):
        """Test creating texture from data."""
        manager = TextureManager()
        manager.context = MagicMock()

        # Create test image data
        width, height = 256, 256
        data = np.random.randint(0, 256, (height, width, 4), dtype=np.uint8)

        texture = manager.create_from_data(data, width, height)
        assert texture is not None

    def test_texture_caching(self):
        """Test texture caching."""
        manager = TextureManager()

        # Mock texture
        mock_texture = MagicMock()
        manager.textures["test"] = mock_texture

        # Should return cached texture
        texture = manager.get_texture("test")
        assert texture == mock_texture

    def test_texture_binding(self):
        """Test texture binding."""
        manager = TextureManager()

        mock_texture = MagicMock()
        manager.textures["test"] = mock_texture

        manager.bind_texture("test", unit=0)

        # Should have bound texture
        mock_texture.use.assert_called_with(0)

    def test_texture_cleanup(self):
        """Test texture cleanup."""
        manager = TextureManager()

        mock_texture = MagicMock()
        manager.textures["test"] = mock_texture

        manager.cleanup()

        # Should have released all textures
        mock_texture.release.assert_called()


@pytest.mark.unit()
class TestGeometryCalibrator:
    """Test the geometry calibrator."""

    def test_calibrator_creation(self):
        """Test creating geometry calibrator."""
        from projector.calibration.geometric import TableDimensions

        table_dims = TableDimensions(length=2.84, width=1.42)
        calibrator = GeometricCalibrator(
            table_dims, display_width=1920, display_height=1080
        )
        assert calibrator is not None

    def test_add_calibration_point(self):
        """Test adding calibration points."""
        from projector.calibration.geometric import TableDimensions

        table_dims = TableDimensions(length=2.84, width=1.42)
        calibrator = GeometricCalibrator(
            table_dims, display_width=1920, display_height=1080
        )

        # Add real-world to screen mappings
        calibrator.add_calibration_target(
            table_x=0, table_y=0, display_x=100, display_y=100, label="Table corner"
        )

        assert len(calibrator.calibration_targets) == 1

    def test_calculate_transformation_matrix(self):
        """Test calculating transformation matrix."""
        from projector.calibration.geometric import TableDimensions

        table_dims = TableDimensions(length=2.84, width=1.42)
        calibrator = GeometricCalibrator(
            table_dims, display_width=1920, display_height=1080
        )

        # Add four corner points for perspective transform
        calibrator.add_calibration_target(0, 0, 100, 100, "Top-left")  # Top-left
        calibrator.add_calibration_target(2.84, 0, 500, 100, "Top-right")  # Top-right
        calibrator.add_calibration_target(
            2.84, 1.42, 500, 300, "Bottom-right"
        )  # Bottom-right
        calibrator.add_calibration_target(
            0, 1.42, 100, 300, "Bottom-left"
        )  # Bottom-left

        success = calibrator.calculate_transform()
        assert success is True
        assert calibrator.transform_matrix is not None
        assert calibrator.transform_matrix.shape == (3, 3)

    def test_transform_point(self):
        """Test transforming points."""
        from projector.calibration.geometric import TableDimensions

        table_dims = TableDimensions(length=2.84, width=1.42)
        calibrator = GeometricCalibrator(
            table_dims, display_width=1920, display_height=1080
        )

        # Mock transformation matrix
        calibrator.transform_matrix = np.array([[1, 0, 100], [0, 1, 100], [0, 0, 1]])

        # Transform point
        screen_point = calibrator.table_to_display(1.0, 0.5)
        assert screen_point is not None

    def test_inverse_transform(self):
        """Test inverse transformation."""
        from projector.calibration.geometric import TableDimensions

        table_dims = TableDimensions(length=2.84, width=1.42)
        calibrator = GeometricCalibrator(
            table_dims, display_width=1920, display_height=1080
        )

        # Set up simple transformation
        calibrator.transform_matrix = np.array([[100, 0, 0], [0, 100, 0], [0, 0, 1]])
        calibrator.inverse_transform = np.linalg.inv(calibrator.transform_matrix)

        # Transform and inverse transform
        screen_point = calibrator.table_to_display(1.0, 0.5)
        real_point = calibrator.display_to_table(screen_point[0], screen_point[1])

        # Should get back original point (approximately)
        assert abs(real_point[0] - 1.0) < 0.01
        assert abs(real_point[1] - 0.5) < 0.01

    def test_calibration_accuracy(self):
        """Test calibration accuracy measurement."""
        from projector.calibration.geometric import TableDimensions

        table_dims = TableDimensions(length=2.84, width=1.42)
        calibrator = GeometricCalibrator(
            table_dims, display_width=1920, display_height=1080
        )

        # Add calibration points
        for i in range(4):
            calibrator.add_calibration_target(
                i * 0.5, i * 0.5, i * 100, i * 100, f"point_{i}"
            )

        calibrator.calculate_transform()

        # Calibration error should be a reasonable value
        assert calibrator.calibration_error >= 0

    def test_save_load_calibration(self, temp_dir):
        """Test saving and loading calibration."""
        from projector.calibration.geometric import TableDimensions

        table_dims = TableDimensions(length=2.84, width=1.42)
        calibrator = GeometricCalibrator(
            table_dims, display_width=1920, display_height=1080
        )

        # Add some calibration data
        calibrator.add_calibration_target(0, 0, 100, 100, "point_1")
        calibrator.add_calibration_target(1, 1, 200, 200, "point_2")

        # Get calibration data
        calibration_data = calibrator.get_calibration_data()

        # Load calibration
        new_calibrator = GeometricCalibrator(
            table_dims, display_width=1920, display_height=1080
        )
        success = new_calibrator.load_calibration_data(calibration_data)

        assert success is True
        assert len(new_calibrator.calibration_targets) == 2


@pytest.mark.unit()
class TestProjectorColorCalibrator:
    """Test the projector color calibrator."""

    def test_calibrator_creation(self):
        """Test creating color calibrator."""
        calibrator = ProjectorColorCalibrator()
        assert calibrator is not None

    def test_measure_projected_color(self):
        """Test measuring projected color."""
        calibrator = ProjectorColorCalibrator()

        # Mock camera capture
        with patch.object(calibrator, "capture_camera_frame") as mock_capture:
            mock_frame = np.full((480, 640, 3), [128, 128, 128], dtype=np.uint8)
            mock_capture.return_value = mock_frame

            measured_color = calibrator.measure_color(
                projected_color=(255, 0, 0), measurement_region=(100, 100, 50, 50)
            )

            assert measured_color is not None
            assert len(measured_color) == 3

    def test_color_correction_matrix(self):
        """Test color correction matrix calculation."""
        calibrator = ProjectorColorCalibrator()

        # Add color measurements
        calibrator.add_measurement(projected=(255, 0, 0), measured=(200, 50, 50))
        calibrator.add_measurement(projected=(0, 255, 0), measured=(50, 200, 50))
        calibrator.add_measurement(projected=(0, 0, 255), measured=(50, 50, 200))

        correction_matrix = calibrator.calculate_correction_matrix()
        assert correction_matrix is not None
        assert correction_matrix.shape == (3, 3)

    def test_apply_color_correction(self):
        """Test applying color correction."""
        calibrator = ProjectorColorCalibrator()

        # Mock correction matrix
        calibrator.correction_matrix = np.array([[1.2, 0, 0], [0, 1.1, 0], [0, 0, 1.3]])

        corrected_color = calibrator.correct_color([100, 100, 100])
        assert corrected_color is not None
        assert len(corrected_color) == 3

    def test_gamma_correction(self):
        """Test gamma correction."""
        calibrator = ProjectorColorCalibrator()

        original_color = [128, 128, 128]
        gamma_corrected = calibrator.apply_gamma_correction(original_color, gamma=2.2)

        assert gamma_corrected is not None
        assert len(gamma_corrected) == 3
        # Gamma correction should change the values
        assert gamma_corrected != original_color


@pytest.mark.unit()
class TestProjectorMath:
    """Test projector math utilities."""

    def test_transform_point(self):
        """Test point transformation."""
        # Identity matrix
        matrix = np.eye(3)
        point = (100, 200)

        transformed = transform_point(point, matrix)
        assert transformed == point

        # Translation matrix
        matrix = np.array([[1, 0, 50], [0, 1, 100], [0, 0, 1]])

        transformed = transform_point(point, matrix)
        assert transformed == (150, 300)

    def test_create_projection_matrix(self):
        """Test creating projection matrix."""
        # Perspective projection
        matrix = create_projection_matrix(
            fov=60, aspect_ratio=16 / 9, near=0.1, far=100.0
        )

        assert matrix.shape == (4, 4)
        assert matrix[3, 3] == 0  # Perspective matrix has 0 in bottom-right

        # Orthographic projection
        ortho_matrix = create_projection_matrix(
            left=-10,
            right=10,
            bottom=-10,
            top=10,
            near=0.1,
            far=100.0,
            orthographic=True,
        )

        assert ortho_matrix.shape == (4, 4)
        assert ortho_matrix[3, 3] == 1  # Orthographic matrix has 1 in bottom-right

    def test_viewport_transformation(self):
        """Test viewport transformation."""
        from projector.utils.math import viewport_transform

        # Normalized coordinates to screen coordinates
        normalized_point = (0.5, 0.5)  # Center
        screen_point = viewport_transform(
            normalized_point, viewport_width=1920, viewport_height=1080
        )

        assert screen_point == (960, 540)  # Center of screen

    def test_perspective_division(self):
        """Test perspective division."""
        from projector.utils.math import perspective_divide

        # Homogeneous coordinates
        homogeneous_point = (100, 200, 50, 2)  # w = 2
        cartesian_point = perspective_divide(homogeneous_point)

        assert cartesian_point == (50, 100, 25)  # Divided by w

    def test_frustum_culling(self):
        """Test frustum culling."""
        from projector.utils.math import point_in_frustum

        # Point inside frustum
        assert point_in_frustum((0, 0, -5), near=1, far=10)

        # Point outside frustum (too close)
        assert not point_in_frustum((0, 0, -0.5), near=1, far=10)

        # Point outside frustum (too far)
        assert not point_in_frustum((0, 0, -15), near=1, far=10)


@pytest.mark.unit()
class TestProjectorConfigManager:
    """Test the projector configuration manager."""

    def test_manager_creation(self):
        """Test creating projector config manager."""
        manager = ProjectorConfigManager()
        assert manager is not None

    def test_default_configuration(self):
        """Test default configuration values."""
        manager = ProjectorConfigManager()

        config = manager.get_default_config()
        assert "resolution" in config
        assert "position" in config
        assert "brightness" in config

    def test_validate_configuration(self):
        """Test configuration validation."""
        manager = ProjectorConfigManager()

        valid_config = {
            "resolution": {"width": 1920, "height": 1080},
            "position": {"x": 0, "y": 0, "z": 2.0},
            "brightness": 0.8,
            "contrast": 1.1,
        }

        assert manager.validate_config(valid_config)

        invalid_config = {
            "resolution": {"width": -100, "height": 1080},  # Invalid width
            "brightness": 1.5,  # Brightness too high
        }

        assert not manager.validate_config(invalid_config)

    def test_apply_configuration(self):
        """Test applying configuration."""
        manager = ProjectorConfigManager()

        config = {
            "brightness": 0.7,
            "contrast": 1.2,
            "position": {"x": 1, "y": 0, "z": 2.5},
        }

        # Mock projector state
        mock_state = MagicMock()
        manager.apply_config(config, mock_state)

        # Should have updated projector state
        mock_state.update_brightness.assert_called_with(0.7)
        mock_state.update_position.assert_called_with(1, 0, 2.5)

    def test_save_load_configuration(self, temp_dir):
        """Test saving and loading configuration."""
        manager = ProjectorConfigManager()

        config = {"resolution": {"width": 1920, "height": 1080}, "brightness": 0.8}

        config_file = temp_dir / "projector_config.json"
        manager.save_config(config, str(config_file))

        loaded_config = manager.load_config(str(config_file))
        assert loaded_config == config
