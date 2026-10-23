# Legacy still-image vision fixtures

This directory contains still images, CVAT source annotations, and JSON labels
from the retired Python vision workflow. They are retained as reference data;
they are not an active quality gate and are not scored by a supported
standalone still-image evaluator. The previously documented `>98%` thresholds
and `test_vision_with_test_data.py` command were not backed by a current test
suite and must not be used as evidence of live localization quality.

The current Rust temporal evaluator scores video-backed observations against
reviewed temporal annotations. Use the tools in
[`docs/PERFORMANCE_AUDIT.md`](../../../docs/PERFORMANCE_AUDIT.md) for that
workflow. The still-image JSON files below do not establish movement tracking,
freshness, occlusion recovery, or live camera accuracy.

## Fixture coordinate convention

The checked-in JSON fixtures pair with 3840×2160 PNG images. Their top-level
`coordinate_space` is `4k_canonical` and `coordinate_scale` is `[1.0, 1.0]`.
This applies to image-space points, box edges, radii, and extents. The identity
scale records canonical pixels; it is not derived from camera calibration. For
another frame size, scale x by `frame_width / 3840` and y by
`frame_height / 2160`. Nonuniform resizing turns a circular radius into x/y
extents.

The CVAT `annotations.xml` file retains native annotation coordinates. No
current tool converts these still-image labels into the Rust temporal video
annotation contract. Review and explicitly convert them before using them for
any accuracy claim.

## Included examples

Fixtures include empty, multi-ball, clustered, full-table, cue, aiming,
motion-blur, calibration-view, and visible-table-corner cases. Their filenames
and labels describe the recorded examples only; they do not imply a measured
quality threshold. The active detector's class contract is generic `cue` and
`ball`; numbered-ball classification is not supported.
