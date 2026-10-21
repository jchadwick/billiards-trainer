"""Smoke and coordinate-contract tests for executable backend examples."""

import ast
import importlib
from pathlib import Path

from backend.core.resolution_converter import ResolutionConverter

EXAMPLES_DIR = Path(__file__).parents[2] / "examples"


def test_every_example_module_imports_without_hardware() -> None:
    """Import all examples without opening camera, video, or accelerator devices."""
    module_names = sorted(
        f"backend.examples.{path.stem}"
        for path in EXAMPLES_DIR.glob("*.py")
        if not path.stem.startswith("__")
    )

    assert module_names
    for module_name in module_names:
        importlib.import_module(module_name)


def test_resolution_examples_scale_points_boundaries_and_physical_dimensions(
    capsys,
) -> None:
    module = importlib.import_module("backend.examples.resolution_config_examples")

    point_4k, bottom_right_4k, radius_4k = module.example_2_scale_points_and_bounds()
    table = module.example_4_calibrated_table_units()
    capsys.readouterr()

    assert point_4k.to_dict() == {"x": 1920.0, "y": 1080.0, "scale": [1.0, 1.0]}
    assert bottom_right_4k.to_dict() == {
        "x": 3840.0,
        "y": 2160.0,
        "scale": [1.0, 1.0],
    }
    assert radius_4k == 36.0
    assert [
        (corner.x, corner.y, corner.scale)
        for corner in table.playing_area_corners or []
    ] == [
        (200.0, 200.0, (1.0, 1.0)),
        (3640.0, 200.0, (1.0, 1.0)),
        (3640.0, 1960.0, (1.0, 1.0)),
        (200.0, 1960.0, (1.0, 1.0)),
    ]
    assert table.real_width_meters == 2.54
    assert table.real_height_meters == 1.27


def test_cue_pointing_examples_run_and_use_canonical_positions(capsys) -> None:
    module = importlib.import_module("backend.examples.example_cue_pointing")
    module.example_with_backend_models()
    module.example_with_cue_direction_vector()
    module.example_with_custom_tolerance()
    module.example_with_vision_models()
    module.example_multiple_balls_in_line()
    output = capsys.readouterr().out

    assert "Cue is pointing at: ball_1" in output
    assert "With 80px tolerance: None" in output
    assert "With 120px tolerance: 0" in output
    assert (
        ResolutionConverter.scale_distance_to_4k(40.0, module.CAMERA_RESOLUTION) == 80
    )


def test_refinement_example_runs_with_explicit_source_and_canonical_geometry(
    capsys,
) -> None:
    module = importlib.import_module("backend.examples.demo_hybrid_validator")
    source, refined = module.run_example()
    output = capsys.readouterr().out

    assert source.coordinate_space == "4k_canonical"
    assert source.position == (1920.0, 1080.0)
    assert source.radius == 288.0
    assert refined.coordinate_space == "4k_canonical"
    assert 0 <= refined.position[0] <= 3840
    assert 0 <= refined.position[1] <= 2160
    assert "Frame resolution: 320x180 pixels" in output


def test_example_code_does_not_construct_unscaled_vector2d_values() -> None:
    """Image geometry must use explicit canonical or source-resolution factories."""
    for path in EXAMPLES_DIR.glob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "Vector2D"
            ):
                has_scale = any(keyword.arg == "scale" for keyword in node.keywords)
                assert (
                    len(node.args) >= 3 or has_scale
                ), f"{path}:{node.lineno} constructs Vector2D without scale metadata"
