"""Fit the runtime's zero-distortion intrinsics from bounded checkerboard stills."""
import json
import sys

import cv2
import numpy as np


def fit(columns, rows, square_mm, paths):
    cv2.setNumThreads(1)
    world = np.zeros((columns * rows, 3), np.float32)
    world[:, :2] = np.mgrid[:columns, :rows].T.reshape(-1, 2) * square_mm
    observations = []
    size = None
    for index, path in enumerate(paths, 1):
        gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise ValueError(f"Checkerboard capture {index} cannot be decoded.")
        if size is not None and size != gray.shape[::-1]:
            raise ValueError("All captures must use the same image size.")
        size = gray.shape[::-1]
        found, corners = cv2.findChessboardCornersSB(gray, (columns, rows))
        if not found:
            raise ValueError(f"Checkerboard corners were not found in capture {index}. Replace that capture.")
        observations.append(corners)
    flags = cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3

    def calibrate(views):
        return cv2.calibrateCamera([world] * len(views), views, size, None, None, flags=flags)

    rms, camera, _, rotations, _ = calibrate(observations)
    normals = np.array([cv2.Rodrigues(rotation)[0][:, 2] for rotation in rotations])
    if np.min(normals @ normals.T) > np.cos(np.deg2rad(20)):
        raise ValueError("Checkerboard poses are too similar. Tilt the board in different directions and recapture.")
    coverage = np.ptp(np.concatenate(observations).reshape(-1, 2), axis=0) / size
    if np.any(coverage < 0.5):
        raise ValueError("Move the board across the image. Captures must span at least half of each image dimension.")

    # Hold out each complete view from the intrinsics fit, and estimate its
    # pose using only half of its corners. Score the other half.
    parity = np.indices((rows, columns)).sum(axis=0).ravel() % 2 == 0
    squared_errors = []
    for index, points in enumerate(observations):
        _, matrix, distortion, _, _ = calibrate(observations[:index] + observations[index + 1:])
        for selection in (parity, ~parity):
            ok, rotation, translation = cv2.solvePnP(world[selection], points[selection], matrix, distortion)
            if not ok:
                raise ValueError(f"Could not estimate held-out checkerboard pose {index + 1}.")
            projection, _ = cv2.projectPoints(world[~selection], rotation, translation, matrix, distortion)
            squared_errors.extend(np.sum((projection - points[~selection]).reshape(-1, 2) ** 2, axis=1))
    holdout = float(np.sqrt(np.mean(squared_errors)))
    if not np.isfinite(holdout) or holdout > max(size) * 0.002:
        raise ValueError(f"Held-out reprojection error is {holdout:.2f} px. Recheck board size, flatness, focus, and pose coverage.")
    if not (0 < camera[0, 2] < size[0] and 0 < camera[1, 2] < size[1]):
        raise ValueError("The fitted optical center is outside the image. Capture more varied board poses.")
    return dict(
        views=len(paths), rms_px=rms, holdout_rms_px=holdout,
        intrinsics=dict(source_width_px=size[0], source_height_px=size[1],
                        fx=camera[0, 0], fy=camera[1, 1], cx=camera[0, 2], cy=camera[1, 2]),
    )


if __name__ == "__main__":
    try:
        columns, rows, square_mm, *images = sys.argv[1:]
        print(json.dumps(fit(int(columns), int(rows), float(square_mm), images), allow_nan=False))
    except (ValueError, cv2.error) as error:
        sys.exit(str(error))
