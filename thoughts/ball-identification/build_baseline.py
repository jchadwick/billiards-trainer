#!/usr/bin/env python3
"""Build the immutable, user-confirmed ball-identity corpus manifest.

Manual regions are deliberately authored by table position and visible appearance;
source detector boxes are recorded separately and never become identity labels.
"""
from __future__ import annotations

import argparse
import cv2
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


# Coordinates from the original user-reviewed 1280x720 measured grid.
GRID = [
    (6, 109.0, 99.0), (5, 352.0, 96.5), (13, 642.0, 93.5),
    (11, 928.0, 93.5), ("cue", 1198.0, 93.0),
    (9, 95.5, 362.0), (12, 357.5, 363.5), (15, 643.5, 361.5),
    (14, 925.0, 361.5), (4, 1198.5, 363.0),
    (3, 103.5, 621.0), (8, 362.0, 627.0), (7, 641.0, 633.5),
    (1, 925.0, 630.5), (2, 1196.0, 632.0),
]


def labels(ids: list[int | str], coordinates: dict[int | str, tuple[float, float]], side: float) -> list[dict]:
    result = []
    for identity in ids:
        x, y = coordinates[identity]
        result.append({
            "identity": identity,
            "center_xy_px": [round(x, 3), round(y, 3)],
            "region_xyxy_px": [round(x-side/2, 3), round(y-side/2, 3), round(x+side/2, 3), round(y+side/2, 3)],
            "region_source": "manual_user_confirmed_identity_and_position",
        })
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    root = args.root.resolve()

    grid_coords = {identity: (x, y) for identity, x, y in GRID}
    scenes: list[dict] = []

    def add_image(scene_id: str, relative_path: str, ids: list[int | str], coords: dict, side: float = 56.0) -> None:
        path = root / relative_path
        image = __import__("PIL.Image", fromlist=["Image"]).open(path)
        width, height = image.size
        group = ("grid-pair" if scene_id in ("measured-grid-720", "alternate-grid-720") else
                 "small-checkerboard-session" if scene_id.startswith("small-checkerboard-") else
                 "framing-pair" if scene_id.startswith("framing-") else
                 "large-checkerboard-session")
        scenes.append({
            "id": scene_id,
            "capture_group": group,
            "source": {"kind": "image", "path": relative_path, "sha256": sha256(path)},
            "dimensions_px": [width, height],
            "manual_regions": labels(ids, coords, side),
        })

    def add_video(scene_id: str, frame_index: int, ids: list[int | str], coords: dict, side: float = 104.0) -> None:
        path = root / "test_data/demo3.mp4"
        cap = cv2.VideoCapture(str(path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise RuntimeError(f"could not decode {path} frame {frame_index}")
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        decoded_pixel_hash = hashlib.sha256(frame_rgb.tobytes()).hexdigest()
        scenes.append({
            "id": scene_id,
            "capture_group": "demo3-video-session",
            "source": {
                "kind": "video_frame", "path": "test_data/demo3.mp4",
                "sha256": sha256(path), "frame_index": frame_index,
                "timestamp_seconds": frame_index / 30.0,
                "timestamp_basis": "exact frame index decoded from constant 30 fps source; ffprobe verified PTS at frame 0 and CFR interval",
                "decoded_rgb_sha256": decoded_pixel_hash,
            },
            "dimensions_px": [3840, 2160],
            "manual_regions": labels(ids, coords, side),
        })

    add_image("measured-grid-720", "thoughts/calibration/table-grid-20261026-114241.jpg", [row[0] for row in GRID], grid_coords)
    add_image("alternate-grid-720", "thoughts/calibration/table-grid-2026-10-26.jpg", [row[0] for row in GRID], grid_coords)
    add_image("small-checkerboard-001", "thoughts/calibration/checkerboard-session/view-001.jpg", [row[0] for row in GRID], grid_coords)

    retained = {
        2: [5, 13, 11, 9, 12, 15, 14, 4, 8, 7, 1],
        3: [5, 13, 11, 9, 12, 15, 14, 4, 8, 7, 1],
        4: [5, 13, 11, 9, 12, 15, 14, 4, 8, 7, 1],
        5: [5, 13, 11, 9, 12, 15, 14, 4, 8, 7, 1],
        6: [5, 13, 11, 9, 12, 4, 1, 14, 8, 7],
        7: [5, 13, 11, 9, 12, 4, 15, 1, 14, 8, 7],
        8: [5, 13, 11, 9, 12, 15, 1, 14, 8, 7],
        9: [5, 13, 11],
    }
    # Balls that stayed in their grid positions retain the photographed centers.
    # Displaced balls in views 006-008 have separately reviewed center locations.
    displaced = {
        6: {1: (610.0, 400.0), 14: (748.0, 408.0)},
        7: {15: (210.0, 488.0), 1: (181.0, 530.0), 14: (204.0, 553.0)},
        8: {15: (210.0, 488.0), 1: (181.0, 530.0), 14: (204.0, 553.0)},
    }
    for view, ids in retained.items():
        coords = dict(grid_coords)
        coords.update(displaced.get(view, {}))
        add_image(f"small-checkerboard-{view:03}", f"thoughts/calibration/checkerboard-session/view-{view:03}.jpg", ids, coords)

    before = {5: (359.0, 95.0), 13: (642.0, 96.0), 11: (929.0, 95.0)}
    add_image("framing-before-720", "thoughts/calibration/stream-4k-verification/before.jpg", [5, 13, 11], before)
    add_image("framing-after-4k", "thoughts/calibration/stream-4k-verification/after.jpg", [5, 13, 11], {5: (1054, 287), 13: (1928, 290), 11: (2790, 285)}, side=170)
    for view in range(1, 9):
        relative = f"thoughts/calibration/checkerboard-20x14-session/view-{view:03}.jpg"
        add_image(f"large-checkerboard-{view:03}", relative, [], {})

    # Exact, user-reviewed ordering and approximate ball centers in the rack.
    rack = {
        1: (2760, 1085), 6: (2818, 1048), 15: (2835, 1140),
        13: (2885, 1010), 8: (2910, 1090), 3: (2910, 1175),
        7: (2980, 950), 11: (2981, 1070), 9: (2980, 1125), 4: (2970, 1200),
        10: (3060, 900), 5: (3045, 1025), 12: (3030, 1100),
        2: (3070, 1160), 14: (3070, 1265), "cue": (1054, 1635),
    }
    add_video("demo3-rack-frame-000", 0, list(range(1, 16)) + ["cue"], rack)

    scattered = {
        6: (959, 309), 14: (1294, 340), 5: (2584, 369), 7: (2862, 405),
        12: (3479, 297), "cue": (2419, 493), 10: (816, 1030),
        11: (3027, 884), 8: (2970, 1258), 1: (466, 1659),
        13: (1252, 1683), 15: (2231, 1677), 3: (2539, 1633),
        4: (2622, 1840), 2: (3017, 1847), 9: (3490, 1880),
    }
    add_video("demo3-scattered-frame-900", 900, [6, 14, 5, 7, 12, "cue", 10, 11, 8, 1, 13, 15, 3, 4, 2, 9], scattered)

    total = sum(len(scene["manual_regions"]) for scene in scenes)
    empty = sum(not scene["manual_regions"] for scene in scenes)
    assert len(scenes) == 23, len(scenes)
    assert total == 161, total
    assert empty == 8, empty
    assert len({identity for scene in scenes for identity in (r["identity"] for r in scene["manual_regions"])}) == 16

    manifest = {
        "schema_version": 1,
        "name": "confirmed-ball-identity-corpus-v1",
        "purpose": "User-confirmed identity baseline for isolated experiments; boxes are identity crops, not localization ground truth.",
        "confirmation": {
            "grid-scenes": "User confirmed both 15-ball grids and that the 10 ball is absent.",
            "video-scenes": "User confirmed both demo3 frames: complete opening rack and dispersed 30-second layout.",
            "small-checkerboard-view-006": "User confirmed mostly white ball immediately below and right of board is the 14 ball.",
            "remaining-stills": "User confirmed all remaining still-image annotations and all eight large-checkerboard images empty.",
        },
        "scene_count": len(scenes),
        "visible_ball_count": total,
        "empty_scene_count": empty,
        "identity_classes": ["cue", *range(1, 16), "unknown"],
        "annotation_limits": [
            "User-confirmed identities; source still regions use preserved visual centers or manually estimated centers, with region boxes padded to include ball appearance.",
            "The two video-frame regions are manually estimated from full-resolution frames; they are not physical center annotations.",
            "Repeated frames from one camera arrangement are correlated and must not be described as independent validation scenes.",
        ],
        "detector_reference": {
            "path": "thoughts/ball-identification/detector-proposals.json",
            "sha256": sha256(root / "thoughts/ball-identification/detector-proposals.json"),
            "model_sha256": "f04470378b10f02d7467dad046ad26fb48eaeb1002e7866cfc37f501b478dc0e",
            "settings": {"backend": "Ultralytics PyTorch CPU", "image_size": 1280, "confidence": 0.3, "iou": 0.45, "max_detections": 16},
            "limitation": "Detection proposals are separately measured, not ground-truth boxes; proposals on videos/stills are matched to manual regions only for the detection-conditioned score.",
        },
        "scenes": scenes,
    }
    output = root / "thoughts/ball-identification/corpus-v1.json"
    output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {output}: {len(scenes)} scenes, {total} labels, {empty} empty scenes")


if __name__ == "__main__":
    main()
