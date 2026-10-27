# Frozen native-localization image corpus, version 1

`native-localization-corpus-v1.json` indexes the original saved JPEG bytes,
their image sizes and SHA-256 digests, source annotations, optics provenance,
known physical references, and experimental use. It does not activate a camera
profile or a ball-center calibration.

The measured 15-ball scene has 15 manually constructed square proposal boxes.
For each annotation, the center is `segmentation_bbox_midpoint_px`, and the
side length is the larger `gradient_ellipse.diameters_px` value plus 4 pixels.
These reproduce the saved identity-audit proposals. They are not detector
outputs and do not independently establish the true ball boundary. Review flags
are preserved, including a known highlight-fragment proposal.

Only nine ball IDs in the exact measured scene have user-confirmed physical
X/Y references; their measurement uncertainties were not independently
established. The six pocket-adjacent balls have no physical X/Y reference.
Checkerboard, alternate-grid, and framing images do not inherit those nine
coordinates, regardless of whether balls appear in the image.

The alternate grid and native 4K image are reserved from *further* tuning.
Both were previously inspected, and the 4K image was previously evaluated.
They are not blind or previously unseen holdouts. The earlier 4K three-ball
check did not retain its proposal boxes; the manifest marks them as not yet
reviewed rather than reconstructing them from the reported acceptance count.

The 720p zero-added-radial matrix is the exact saved eight-view exploratory
fit. The 4K matrix is a direct 3× diagnostic scaling of that matrix. It is not
a qualified 4K runtime profile; near-matching framing does not prove identical
intrinsics or measured millimeter accuracy.

## Validation and reproducibility

From the repository root, the following dependency-free check verifies the
referenced file hashes, JPEG dimensions, exact saved optics matrices, 15 boxes,
nine measured references, six unmeasured cases, and unreviewed proposals in
other scenes. It reads files only and requires no camera:

```sh
python3 - <<'PY'
from pathlib import Path
import hashlib
import json
import math

def jpeg_dimensions(path):
    data = path.read_bytes()
    assert data[:2] == b'\xff\xd8', path
    pos = 2
    frame_markers = {0xc0, 0xc1, 0xc2, 0xc3, 0xc5, 0xc6, 0xc7,
                     0xc9, 0xca, 0xcb, 0xcd, 0xce, 0xcf}
    while pos < len(data):
        assert data[pos] == 0xff, (path, pos)
        while pos < len(data) and data[pos] == 0xff:
            pos += 1
        marker = data[pos]
        pos += 1
        if marker in (0x01, 0xd8, 0xd9) or 0xd0 <= marker <= 0xd7:
            continue
        assert marker != 0xda, (path, 'scan before frame dimensions')
        length = int.from_bytes(data[pos:pos + 2], 'big')
        assert length >= 2 and pos + length <= len(data), path
        if marker in frame_markers:
            width = int.from_bytes(data[pos + 5:pos + 7], 'big')
            height = int.from_bytes(data[pos + 3:pos + 5], 'big')
            return [width, height]
        pos += length
    raise AssertionError((path, 'frame dimensions not found'))

def check_hash(record):
    path = Path(record['path'])
    assert hashlib.sha256(path.read_bytes()).hexdigest() == record['sha256'], path

root = Path('thoughts/calibration')
manifest = json.loads((root / 'native-localization-corpus-v1.json').read_text())
assert manifest['schema_version'] == 1
for record in manifest['source_data'].values():
    check_hash(record)
for image in manifest['images']:
    check_hash(image)
    assert jpeg_dimensions(Path(image['path'])) == image['dimensions_px']
    if 'checkerboard_corner_data' in image:
        check_hash(image['checkerboard_corner_data'])

assert len(manifest['images']) == 13
images = {image['id']: image for image in manifest['images']}
assert len(images) == 13
baseline = json.loads(Path(manifest['source_data']['checkerboard_frozen_baseline']['path']).read_text())
full = baseline['models']['zero']['full_fit']
models = manifest['optics_models']
assert models['frozen_zero_radial_720']['camera_matrix'] == full['camera_matrix']
assert models['frozen_zero_radial_720']['distortion_k1_k2_p1_p2_k3'] == full['distortion_k1_k2_p1_p2_k3']
k = full['camera_matrix']
scaled = [[v * 3.0 for v in k[0]], [v * 3.0 for v in k[1]], k[2][:]]
assert models['scaled_frozen_zero_radial_4k_diagnostic']['camera_matrix'] == scaled

annotations = json.loads(Path(manifest['source_data']['ball_outline_annotations']['path']).read_text())
measurements = json.loads(Path(manifest['source_data']['measured_grid_physical_references']['path']).read_text())
physical = {entry['id']: entry for entry in measurements['points']}
grid = images['measured-grid-720']
assert measurements['snapshot_path'] == grid['path']
proposals = grid['ball_proposals']['proposals']
assert len(proposals) == len(annotations['balls']) == 15
assert len(physical) == 9
for index, (proposal, ball) in enumerate(zip(proposals, annotations['balls'])):
    x, y = ball['segmentation_bbox_midpoint_px']
    side = max(ball['gradient_ellipse']['diameters_px']) + 4.0
    expected = [x - side / 2, y - side / 2, x + side / 2, y + side / 2]
    assert proposal['ball_id'] == ball['id']
    assert proposal['detection_index_in_frozen_audit'] == index
    assert all(math.isclose(a, b, rel_tol=0, abs_tol=1e-12)
               for a, b in zip(proposal['bbox_xyxy_source_px'], expected))
    assert proposal['localization_review_flags'] == ball['localization_review_flags']
    assert proposal['physical_reference_present'] == (ball['id'] in physical)
    assert proposal['physical_reference_id'] == (ball['id'] if ball['id'] in physical else None)

refs = manifest['physical_reference_sets']['measured_grid_nine']
assert refs['image_id'] == 'measured-grid-720'
assert refs['coordinate_convention'] == measurements['coordinate_convention']
assert {entry['id'] for entry in refs['observations']} == set(physical)
for entry in refs['observations']:
    source = physical[entry['id']]
    assert entry['center_from_upper_left_mm'] == [source['center_from_upper_left_mm']['x'], source['center_from_upper_left_mm']['y']]
    assert entry['measurement_uncertainty_mm'] is None
    assert source['measurement_uncertainty_mm'] is None
assert sum(p['physical_reference_present'] for p in proposals) == 9
assert sum(not p['physical_reference_present'] for p in proposals) == 6
for image in manifest['images']:
    if image['id'] != 'measured-grid-720':
        assert image['ball_proposals']['status'] == 'not_yet_reviewed'
        assert not image['ball_proposals']['proposals']
        assert image['physical_reference_status'].startswith('no_matching_')

print('Verified 13 saved JPEGs, 15 proposals, 9 measured references, 6 without physical XY, and exact optics provenance.')
PY
```

The original implementation and numerical audit are preserved at the
experiment checkpoint recorded in the manifest. Future code changes should
retain that original result and write newly versioned outputs with the changed
source hash, parameters, proposals, and image hashes.
