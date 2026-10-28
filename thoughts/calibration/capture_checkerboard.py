"""Offline checkerboard capture, independent of production perception.

Run using /tmp/billiards-checkerboard-env/bin/python. Install dependencies into
that isolated environment only: opencv-python-headless (provides NumPy).
Example:
  python capture_checkerboard.py --file /path/frame.jpg --scale-verified --rigid-backed
  python capture_checkerboard.py --url http://192.168.1.31:8001/snapshot --note 'tilted left'
  python capture_checkerboard.py --self-test

A session is bound to image dimensions, board pitch and inner-corner count.
Detection runs at original resolution; no invented pose/ball Z data is produced.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile
import urllib.request

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def active_pattern_size_inches(pattern, pitch_mm):
    """Return the outside dimensions of the printed square grid.

    A checkerboard with N-by-M inner corners has N+1 by M+1 squares. Keep
    scale provenance tied to the active black-and-white pattern, not the
    paper or rigid backing margins.
    """
    columns, rows = pattern
    return ((columns + 1) * pitch_mm / 25.4,
            (rows + 1) * pitch_mm / 25.4)


def _detect_in_region(image, pattern):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # The SB algorithm supplies subpixel coordinates directly, in source pixels.
    success, corners = cv2.findChessboardCornersSB(
        gray, pattern, flags=cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_ACCURACY
    )
    method = 'findChessboardCornersSB'
    if not success:
        success, corners = cv2.findChessboardCorners(
            gray, pattern, flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
        )
        if success:
            corners = cv2.cornerSubPix(
                gray, corners, (5, 5), (-1, -1),
                (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 40, .001),
            )
            method = 'findChessboardCorners + cornerSubPix'
    if not success or corners is None or len(corners) != pattern[0] * pattern[1]:
        raise ValueError(f'Complete {pattern[0]}x{pattern[1]} corner pattern not detected; nothing appended')
    if not np.isfinite(corners).all():
        raise ValueError('Detector returned nonfinite corners')
    return corners.astype(np.float64).reshape(-1, 2), method


def bright_component_rois(image, pattern):
    """Find plausible bright-paper regions to reduce background clutter.

    The original image remains the source of truth: callers detect only on a
    crop, then translate corner coordinates back to the full source frame.
    Multiple thresholds and candidates allow for exposure changes, shadows,
    and unrelated bright objects without selecting one candidate by color
    alone.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    minimum_area = max(4096, int(width * height * .002))
    expected_aspect = (pattern[0] + 1) / (pattern[1] + 1)
    regions = {}
    for threshold in (175, 195, 215, 235):
        _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for x, y, region_width, region_height, area in stats[1:count]:
            x, y, region_width, region_height, area = map(
                int, (x, y, region_width, region_height, area)
            )
            if area < minimum_area or region_width < (pattern[0] + 1) * 8 \
                    or region_height < (pattern[1] + 1) * 8:
                continue
            aspect = region_width / region_height
            if not .35 * expected_aspect <= aspect <= 2.85 * expected_aspect:
                continue
            padding = max(8, int(max(region_width, region_height) * .08))
            x0, y0 = max(0, x - padding), max(0, y - padding)
            x1 = min(width, x + region_width + padding)
            y1 = min(height, y + region_height + padding)
            key = (x0, y0, x1, y1)
            regions[key] = max(regions.get(key, 0), area)
    # Try larger plausible paper regions first; detections are still required
    # to contain the complete requested corner pattern.
    return sorted(regions, key=lambda roi: (-(roi[2] - roi[0]) * (roi[3] - roi[1]), roi))[:32]


def detect_in_bright_regions(image, pattern):
    """Detect within bright-region crops and return full-frame coordinates."""
    failures = []
    for x0, y0, x1, y1 in bright_component_rois(image, pattern):
        try:
            corners, method = _detect_in_region(image[y0:y1, x0:x1], pattern)
        except ValueError as error:
            failures.append(str(error))
            continue
        corners += np.array([x0, y0], dtype=np.float64)
        return corners, f'{method}; bright-region ROI ({x0},{y0})-({x1},{y1})'
    detail = f'; tried {len(failures)} bright-region crops' if failures else ''
    raise ValueError(
        f'Complete {pattern[0]}x{pattern[1]} corner pattern not detected{detail}; nothing appended'
    )


def detect(image, pattern):
    """Detect a full checkerboard, falling back to bright-paper crops."""
    try:
        return _detect_in_region(image, pattern)
    except ValueError as full_frame_error:
        try:
            return detect_in_bright_regions(image, pattern)
        except ValueError:
            raise full_frame_error


def quality(image, corners, pattern):
    cols, rows = pattern
    grid = corners.reshape(rows, cols, 2)
    distances = np.r_[np.linalg.norm(np.diff(grid, axis=1), axis=2).ravel(),
                      np.linalg.norm(np.diff(grid, axis=0), axis=2).ravel()]
    # Extrapolate one square beyond internal corners to estimate black/white
    # rectangle extent. This is a projective estimate, not measured white border.
    objects = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2).astype(np.float64)
    h, _ = cv2.findHomography(objects, corners, method=0)
    outer = np.array([[[-1., -1.], [cols, -1.], [cols, rows], [-1., rows]]])
    outer = cv2.perspectiveTransform(outer, h).reshape(-1, 2)
    height, width = image.shape[:2]
    clearance = float(np.min(np.r_[outer[:, 0], outer[:, 1], width - 1 - outer[:, 0], height - 1 - outer[:, 1]]))
    lo = np.maximum(np.floor(outer.min(0)).astype(int), 0)
    hi = np.minimum(np.ceil(outer.max(0)).astype(int), [width, height])
    crop = image[lo[1]:hi[1], lo[0]:hi[0]]
    contrast = float(np.percentile(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), 95)
                     - np.percentile(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), 5))
    return {'adjacent_corner_pitch_px_min': float(distances.min()),
            'adjacent_corner_pitch_px_median': float(np.median(distances)),
            'estimated_pattern_outer_corners_px': outer.tolist(),
            'estimated_pattern_to_image_border_clearance_px_min': clearance,
            'white_border_clearance': 'not measured; inspect annotated source image',
            'pattern_roi_grayscale_p95_minus_p05': contrast,
            'warnings': ['Small target: native square pitch below 15 px; collect additional clear views']
                        if distances.min() < 15 else []}


def atomic_json(path, data):
    fd, temp = tempfile.mkstemp(prefix=path.name + '.', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(data, stream, indent=2)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def append_capture(args):
    capture_started_at = utc_now()
    if args.url:
        with urllib.request.urlopen(args.url, timeout=15) as response:
            raw = response.read(20_000_001)
        source = {'kind': 'snapshot_url', 'value': args.url}
    else:
        raw = args.file.read_bytes()
        source = {'kind': 'source_file', 'value': str(args.file.resolve())}
    if len(raw) > 20_000_000:
        raise ValueError('Source exceeds offline capture size limit')
    image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError('Source is not a decodable image')
    received_at = utc_now()
    pattern = (args.columns, args.rows)
    corners, method = detect(image, pattern)
    diagnostics = quality(image, corners, pattern)
    if diagnostics['estimated_pattern_to_image_border_clearance_px_min'] <= 0:
        raise ValueError('Estimated checkerboard pattern crosses image border; nothing appended')
    height, width = image.shape[:2]
    encoded_hash = hashlib.sha256(raw).hexdigest()
    pixel_hash = hashlib.sha256(image.tobytes()).hexdigest()
    root = args.session
    root.mkdir(parents=True, exist_ok=True)
    # Lock numbering and manifest update together; atomic manifest replacement
    # prevents concurrent captures from overwriting each other's session records.
    with (root / '.capture.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = root / 'manifest.json'
        if path.exists():
            manifest = json.loads(path.read_text())
            if manifest['image_size_px'] != [width, height] or manifest['pattern_inner_corners'] != list(pattern) or manifest['square_pitch_mm'] != args.pitch_mm:
                raise ValueError('Source geometry or target differs from session; create a new session')
            if any(v['decoded_pixel_sha256'] == pixel_hash or v['source_sha256'] == encoded_hash for v in manifest['views']):
                raise ValueError('Exact duplicate frame rejected; move/tilt target and capture again')
        else:
            manifest = {'schema_version': 1, 'created_at_utc': capture_started_at,
                        'image_size_px': [width, height], 'pattern_inner_corners': list(pattern),
                        'pattern_squares': [pattern[0] + 1, pattern[1] + 1],
                        'square_pitch_mm': args.pitch_mm, 'printed_scale_verification': 'pending',
                        'rigid_flat_backing_verification': 'pending',
                        'camera_geometry': 'fixed camera; do not change crop/resolution/zoom/focus settings during session',
                        'calibration_status': 'captures only; not calibrated or activated',
                        'opencv_version': cv2.__version__, 'views': []}
        active_width_in, active_height_in = active_pattern_size_inches(pattern, args.pitch_mm)
        active_size_mm = [(pattern[0] + 1) * args.pitch_mm,
                          (pattern[1] + 1) * args.pitch_mm]
        manifest['pattern_active_size_mm'] = active_size_mm
        if args.scale_verified:
            manifest['printed_scale_verification'] = (
                'user confirmed active checkerboard measures '
                f'{active_width_in:g} x {active_height_in:g} inches'
            )
        if args.rigid_backed:
            manifest['rigid_flat_backing_verification'] = 'user confirmed flat rigid backing'
        number = max([v['number'] for v in manifest['views']] + [0]) + 1
        while any(root.glob(f'view-{number:03d}*')):
            number += 1
        stem = f'view-{number:03d}'
        # Preserve exact source bytes; extension follows detected JPEG/PNG magic.
        extension = '.jpg' if raw[:2] == b'\xff\xd8' else '.png'
        if extension == '.png' and not raw.startswith(b'\x89PNG'):
            raise ValueError('Capture source must be JPEG or PNG')
        source_path = root / (stem + extension)
        with source_path.open('xb') as stream:
            stream.write(raw)
        record = {'number': number, 'id': stem,
                  'source_file': source_path.name, 'source': source,
                  'capture_request_started_at_utc': capture_started_at,
                  'image_received_at_utc': received_at,
                  'capture_time_note': 'Receipt/import time; camera exposure time unavailable',
                  'source_sha256': encoded_hash, 'decoded_pixel_sha256': pixel_hash,
                  'source_width_px': width, 'source_height_px': height,
                  'pattern_inner_corners': list(pattern), 'square_pitch_mm': args.pitch_mm,
                  'pattern_active_size_mm': active_size_mm,
                  'corner_count': len(corners), 'corners_source_image_px': corners.tolist(),
                  'corner_method': method, 'corner_coordinates': 'original source resolution; no upsampling',
                  'physical_measurement_uncertainty': 'unknown', 'quality': diagnostics,
                  'notes': args.note, 'corner_file': stem + '-corners.json',
                  'annotated_file': stem + '-annotated.png'}
        atomic_json(root / record['corner_file'], record)
        annotated = image.copy()
        cv2.drawChessboardCorners(annotated, pattern, corners.astype(np.float32).reshape(-1, 1, 2), True)
        if not cv2.imwrite(str(root / record['annotated_file']), annotated):
            raise OSError('Failed writing annotated image; manifest not updated')
        manifest['views'].append(record)
        atomic_json(path, manifest)
    print(json.dumps({'id': stem, 'corner_count': len(corners), 'quality': diagnostics,
                      'manifest': str(path.resolve())}, indent=2))


def self_test():
    pattern = (9, 6)
    assert active_pattern_size_inches(pattern, 50.8) == (20.0, 14.0)
    image = np.full((440, 600, 3), 255, np.uint8)
    pitch = 40
    for row in range(7):
        for col in range(10):
            if (row + col) % 2 == 0:
                image[80 + row*pitch:80 + (row+1)*pitch, 80 + col*pitch:80 + (col+1)*pitch] = 0
    corners, _ = detect(image, pattern)
    assert len(corners) == 54 and np.isfinite(corners).all()
    assert abs(quality(image, corners, pattern)['adjacent_corner_pitch_px_median'] - pitch) < .1
    try:
        detect(image[:, :300], pattern)
    except ValueError:
        pass
    else:
        raise AssertionError('Partial target incorrectly accepted')

    # Exercise the camera-frame fallback: a small checkerboard on a large,
    # cluttered green field must be found inside its paper crop, with its
    # detected corner coordinates translated back into the full image.
    cluttered = np.full((720, 1280, 3), (70, 145, 85), np.uint8)
    board = np.full((440, 600, 3), 255, np.uint8)
    pitch = 48
    for row in range(7):
        for col in range(10):
            if (row + col) % 2 == 0:
                board[80 + row*pitch:80 + (row+1)*pitch,
                      60 + col*pitch:60 + (col+1)*pitch] = 0
    cluttered[120:560, 340:940] = board
    # Bright distractors ensure the crop search cannot assume one white object.
    cv2.rectangle(cluttered, (20, 20), (320, 620), (255, 255, 255), -1)
    fallback_corners, fallback_method = detect_in_bright_regions(cluttered, pattern)
    assert len(fallback_corners) == 54 and 'bright-region ROI' in fallback_method
    assert np.all(fallback_corners[:, 0] > 340) and np.all(fallback_corners[:, 1] > 120)

    # Exercise session metadata with a 2-inch-square print so the verified
    # scale record cannot silently retain the former 10 x 7 inch dimensions.
    with tempfile.TemporaryDirectory(prefix='checkerboard-scale-self-test-') as temp:
        root = Path(temp)
        source = root / 'synthetic-20x14.jpg'
        if not cv2.imwrite(str(source), image):
            raise AssertionError('Could not write synthetic checkerboard JPEG')
        session = root / 'session'
        append_capture(argparse.Namespace(
            url=None, file=source, columns=9, rows=6, pitch_mm=50.8,
            session=session, scale_verified=True, rigid_backed=True,
            note='synthetic scale metadata test',
        ))
        manifest = json.loads((session / 'manifest.json').read_text())
        assert np.allclose(manifest['pattern_active_size_mm'], [508.0, 355.6])
        assert manifest['printed_scale_verification'] == (
            'user confirmed active checkerboard measures 20 x 14 inches'
        )
    print('Synthetic detection, pitch, crop rejection, and 20 x 14 inch scale metadata passed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--file', type=Path)
    source.add_argument('--url')
    parser.add_argument('--session', type=Path, default=HERE / 'checkerboard-session')
    parser.add_argument('--columns', type=int, default=9)
    parser.add_argument('--rows', type=int, default=6)
    parser.add_argument('--pitch-mm', type=float, default=25.4)
    parser.add_argument('--scale-verified', action='store_true',
                        help='record user confirmation of the active grid dimensions')
    parser.add_argument('--rigid-backed', action='store_true')
    parser.add_argument('--note', default='')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not (args.file or args.url):
        parser.error('Provide --file or --url')
    if not np.isfinite(args.pitch_mm) or args.pitch_mm <= 0 or min(args.columns, args.rows) < 3:
        parser.error('Positive finite pitch and at least 3x3 inner corners required')
    try:
        append_capture(args)
    except (ValueError, OSError) as error:
        parser.exit(1, str(error) + '\n')


if __name__ == '__main__':
    main()
