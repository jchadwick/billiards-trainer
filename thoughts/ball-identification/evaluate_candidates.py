#!/usr/bin/env python3
"""Common, detection-fixed scoring harness for experimental crop classifiers.

Training is limited to the reviewed still-image captures. The two exact video
frames come from the same file and are held out together; neither should be
reported as an independent video-session validation set.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import math
import resource
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = ROOT / "thoughts/ball-identification/corpus-v1.json"
PROPOSALS_PATH = ROOT / "thoughts/ball-identification/detector-proposals.json"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def process_resources() -> dict[str, float | int]:
    """Return process CPU totals and peak RSS normalized to bytes."""
    usage = resource.getrusage(resource.RUSAGE_SELF)
    peak_rss = int(usage.ru_maxrss if sys.platform == "darwin" else usage.ru_maxrss * 1024)
    return {
        "cpu_seconds": float(usage.ru_utime + usage.ru_stime),
        "peak_rss_bytes": peak_rss,
    }


def approach_training_metadata(model: Any) -> dict[str, Any]:
    """Expose train-only calibration decisions without serializing model weights."""
    metadata = getattr(model, "metadata", None)
    if isinstance(metadata, dict):
        return metadata
    if isinstance(model, dict):
        keys = (
            "training_calibration",
            "identity_oof_support",
            "training_class_counts",
            "identity_min_probability",
        )
        return {key: model[key] for key in keys if key in model}
    keys = (
        "oof_summary",
        "confidence_threshold",
        "minimum_boundary_quality",
        "train_scene_count",
        "train_sample_count",
    )
    return {key: getattr(model, key) for key in keys if hasattr(model, key)}


def approach_config_path(module: Any, explicit_path: Path | None) -> Path | None:
    """Find the exact settings file used by an approach, when it has one."""
    if explicit_path is not None:
        return explicit_path.resolve()
    for name in ("CONFIG_PATH", "_CONFIG_PATH"):
        candidate = getattr(module, name, None)
        if candidate is not None and Path(candidate).is_file():
            return Path(candidate).resolve()
    return None


def crop(frame: np.ndarray, bbox: list[float]) -> np.ndarray:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    left, top = max(0, math.floor(x1)), max(0, math.floor(y1))
    right, bottom = min(width, math.ceil(x2)), min(height, math.ceil(y2))
    if right <= left or bottom <= top:
        raise ValueError(f"invalid crop box {bbox} for frame {width}x{height}")
    return cv2.cvtColor(frame[top:bottom, left:right], cv2.COLOR_BGR2RGB)


def frame_for(scene: dict, root: Path, video_frames: dict[tuple[str, int], np.ndarray]) -> np.ndarray:
    source = scene["source"]
    path = root / source["path"]
    if source["kind"] == "image":
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(path)
        return image
    key = (str(path), int(source["frame_index"]))
    if key not in video_frames:
        cap = cv2.VideoCapture(str(path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, key[1])
        ok, image = cap.read()
        cap.release()
        if not ok:
            raise RuntimeError(f"failed reading {path} frame {key[1]}")
        video_frames[key] = image
    return video_frames[key]


def dataset(root: Path, manifest: dict) -> tuple[list[dict], list[dict]]:
    frames: dict[tuple[str, int], np.ndarray] = {}
    train, test = [], []
    for scene in manifest["scenes"]:
        is_video = scene["source"]["kind"] == "video_frame"
        target = test if is_video else train
        frame = frame_for(scene, root, frames)
        for region in scene["manual_regions"]:
            x1, y1, x2, y2 = region["region_xyxy_px"]
            target.append({
                "scene_id": scene["id"],
                "capture_group": scene["capture_group"],
                "capture_kind": scene["source"]["kind"],
                "identity": region["identity"],
                "center_xy_px": region["center_xy_px"],
                "crop_rgb": crop(frame, [x1, y1, x2, y2]),
            })
    return train, test


def module_at(path: Path):
    spec = importlib.util.spec_from_file_location("ball_identity_experiment", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load experiment module {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def predictions(module, model, samples: list[dict]) -> tuple[list[dict], dict]:
    crops = [sample["crop_rgb"] for sample in samples]
    start = time.perf_counter()
    if hasattr(module, "predict_batch"):
        results = list(module.predict_batch(model, crops))
    else:
        results = [module.predict(model, image) for image in crops]
    elapsed = time.perf_counter() - start
    if len(results) != len(samples):
        raise ValueError(f"predict returned {len(results)} records for {len(samples)} crops")
    durations = [float(result.get("latency_ms", 0.0)) for result in results]
    return results, {
        "batch_elapsed_ms": elapsed * 1000,
        "crops": len(crops),
        "batch_amortized_ms_per_crop": elapsed * 1000 / max(1, len(crops)),
        "reported_per_crop_p50_ms": statistics.median(durations) if durations else None,
        "reported_per_crop_p95_ms": float(np.percentile(durations, 95)) if durations else None,
    }


def summarise(samples: list[dict], results: list[dict]) -> dict:
    assert len(samples) == len(results)
    accepted = correct = errors = 0
    by_identity: dict[str, dict] = {}
    confusion: dict[str, int] = {}
    for sample, result in zip(samples, results):
        truth = sample["identity"]
        pred = result.get("identity")
        key = str(truth)
        row = by_identity.setdefault(key, {"truth": truth, "n": 0, "accepted": 0, "correct": 0, "errors": 0})
        row["n"] += 1
        if pred is None or pred == "unknown":
            continue
        accepted += 1
        row["accepted"] += 1
        if pred == truth:
            correct += 1
            row["correct"] += 1
        else:
            errors += 1
            row["errors"] += 1
            confusion[f"{truth}->{pred}"] = confusion.get(f"{truth}->{pred}", 0) + 1
    return {
        "observations": len(samples),
        "accepted": accepted,
        "correct_accepted": correct,
        "wrong_accepted": errors,
        "coverage": accepted / len(samples) if samples else 0.0,
        "selective_accuracy": correct / accepted if accepted else None,
        "by_identity": by_identity,
        "confusion_pairs": confusion,
    }


def detector_matches(root: Path, scenes: list[dict], proposals: dict) -> tuple[list[dict], list[dict], dict]:
    proposal_scenes = {}
    for proposal in proposals:
        if proposal.get("frame_index") is not None:
            proposal_scenes[(str(Path(proposal["path"]).resolve()), proposal["frame_index"])] = proposal
        else:
            proposal_scenes[str(Path(proposal["path"]).resolve())] = proposal
    matched_samples: list[dict] = []
    unmatched_samples: list[dict] = []
    counts = {}
    for scene in scenes:
        source = scene["source"]
        source_path = str((root / source["path"]).resolve())
        proposal_key = (source_path, source["frame_index"]) if source["kind"] == "video_frame" else source_path
        pred_scene = proposal_scenes.get(proposal_key)
        if pred_scene is None:
            continue
        # Greedy one-to-one match on center distance, with a gate equal to the
        # annotated ball radius. The same metric is used for all approaches.
        remaining_regions = set(range(len(scene["manual_regions"])))
        remaining_detections = set(range(len(pred_scene["detections"])))
        edges = []
        for di, detection in enumerate(pred_scene["detections"]):
            dx, dy = detection["center"]
            for gi, region in enumerate(scene["manual_regions"]):
                gx, gy = region["center_xy_px"]
                distance = math.hypot(dx-gx, dy-gy)
                gate = (region["region_xyxy_px"][2]-region["region_xyxy_px"][0]) / 2
                if distance <= gate:
                    edges.append((distance, di, gi))
        matches = []
        for distance, di, gi in sorted(edges):
            if di not in remaining_detections or gi not in remaining_regions:
                continue
            remaining_detections.remove(di)
            remaining_regions.remove(gi)
            matches.append((di, gi, distance))
        counts[scene["id"]] = {
            "ground_truth": len(scene["manual_regions"]),
            "detections": len(pred_scene["detections"]),
            "matched": len(matches),
            "missed_ground_truth": len(remaining_regions),
            "extra_detections": len(remaining_detections),
        }
        matched_samples.extend({
            "scene_id": scene["id"],
            "identity": scene["manual_regions"][gi]["identity"],
            "crop_rgb": None,
            "detection_index": di,
            "bbox": pred_scene["detections"][di]["bbox"],
        } for di, gi, _ in matches)
        unmatched_samples.extend({
            "scene_id": scene["id"],
            "identity": None,
            "crop_rgb": None,
            "detection_index": di,
            "bbox": pred_scene["detections"][di]["bbox"],
        } for di in remaining_detections)
    return matched_samples, unmatched_samples, counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--approach", type=Path, required=True)
    parser.add_argument("--config", type=Path, help="Optional approach-specific JSON settings")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--seed", type=int, default=20261030)
    args = parser.parse_args()
    root = args.root.resolve()
    manifest_path = root / "thoughts/ball-identification/corpus-v1.json"
    proposals_path = root / "thoughts/ball-identification/detector-proposals.json"
    manifest, proposals = read_json(manifest_path), read_json(proposals_path)
    assert len(manifest["scenes"]) == 23
    assert manifest["visible_ball_count"] == 161
    assert manifest["empty_scene_count"] == 8
    train, heldout = dataset(root, manifest)
    assert len(train) == 129 and len(heldout) == 32
    assert {sample["scene_id"] for sample in heldout} == {"demo3-rack-frame-000", "demo3-scattered-frame-900"}
    training_ids = {sample["identity"] for sample in train}
    assert set(range(1, 16)) - training_ids == {10}, "unexpected still-image class support"

    module = module_at(args.approach.resolve())
    config_path = approach_config_path(module, args.config)
    resources_before_fit = process_resources()
    fit_start = time.perf_counter()
    if args.config is None:
        model = module.fit(train, seed=args.seed)
    else:
        if "config" not in inspect.signature(module.fit).parameters:
            raise ValueError(f"{args.approach} does not accept --config")
        model = module.fit(train, seed=args.seed, config=read_json(args.config))
    fit_seconds = time.perf_counter() - fit_start
    resources_after_fit = process_resources()

    train_preds, train_timing = predictions(module, model, train)
    heldout_preds, heldout_timing = predictions(module, model, heldout)
    detector_regions, unmatched_detections, detector_counts = detector_matches(root, manifest["scenes"], proposals)
    frames = {}
    for scene in manifest["scenes"]:
        if scene["source"]["kind"] == "video_frame":
            frames[(str(root / scene["source"]["path"]), scene["source"]["frame_index"])] = frame_for(scene, root, {})
    scene_by_id = {scene["id"]: scene for scene in manifest["scenes"]}
    grouped_detections: dict[str, list[dict]] = {}
    for item in [*detector_regions, *unmatched_detections]:
        grouped_detections.setdefault(item["scene_id"], []).append(item)
    for scene_id, samples in grouped_detections.items():
        scene = scene_by_id[scene_id]
        frame = frame_for(scene, root, frames)
        for sample in samples:
            sample["crop_rgb"] = crop(frame, sample["bbox"])
    detection_train = [item for item in detector_regions if not item["scene_id"].startswith("demo3-")]
    detection_heldout = [item for item in detector_regions if item["scene_id"].startswith("demo3-")]
    false_detections = unmatched_detections
    del frames
    train_detection_preds, train_detection_timing = predictions(module, model, detection_train)
    detection_preds, detection_timing = predictions(module, model, detection_heldout)
    false_detection_preds, false_detection_timing = predictions(module, model, false_detections)
    resources_after_evaluation = process_resources()

    report = {
        "schema_version": 1,
        "approach": args.approach.name,
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "proposal_sha256": hashlib.sha256(proposals_path.read_bytes()).hexdigest(),
        "approach_sha256": hashlib.sha256(args.approach.resolve().read_bytes()).hexdigest(),
        "config_path": str(config_path) if config_path else None,
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest() if config_path else None,
        "seed": args.seed,
        "resources": {
            "platform": sys.platform,
            "cpu_seconds_fit_and_model_load": resources_after_fit["cpu_seconds"] - resources_before_fit["cpu_seconds"],
            "cpu_seconds_postfit_evaluation": resources_after_evaluation["cpu_seconds"] - resources_after_fit["cpu_seconds"],
            "peak_rss_bytes_before_fit": resources_before_fit["peak_rss_bytes"],
            "peak_rss_bytes_after_fit": resources_after_fit["peak_rss_bytes"],
            "peak_rss_bytes_after_evaluation": resources_after_evaluation["peak_rss_bytes"],
            "peak_rss_increase_over_pre_fit_bytes": max(0, resources_after_evaluation["peak_rss_bytes"] - resources_before_fit["peak_rss_bytes"]),
        },
        "approach_training_metadata": approach_training_metadata(model),
        "training": {"samples": len(train), "scenes": sorted({s["scene_id"] for s in train}), "fit_seconds": fit_seconds, "train_diagnostic": summarise(train, train_preds), "timing": train_timing},
        "heldout_manual_regions": {**summarise(heldout, heldout_preds), "scene_scope": "two frames from one video, correlated; provisional only", "timing": heldout_timing},
        "training_actual_detection_regions_diagnostic": {**summarise(detection_train, train_detection_preds), "matched_regions": len(detection_train), "timing": train_detection_timing},
        "heldout_actual_detection_regions": {**summarise(detection_heldout, detection_preds), "matched_regions": len(detection_heldout), "timing": detection_timing},
        "unmatched_detection_crops": {
            "regions": len(false_detections),
            "predicted_identity_count": sum(result.get("identity") not in (None, "unknown") for result in false_detection_preds),
            "by_scene": {scene_id: sum(item["scene_id"] == scene_id and result.get("identity") not in (None, "unknown") for item, result in zip(false_detections, false_detection_preds)) for scene_id in sorted({item["scene_id"] for item in false_detections})},
            "timing": false_detection_timing,
        },
        "detector_coverage": {"all_labeled_positive_scenes": detector_counts, "heldout_video_frames": {k: v for k, v in detector_counts.items() if k.startswith("demo3-")}},
        "candidate_outputs": [
            {"scene_id": sample["scene_id"], "truth": sample["identity"], "prediction": result.get("identity"), "confidence": result.get("confidence"), "evidence": result.get("evidence", {})}
            for sample, result in zip(heldout, heldout_preds)
        ],
        "limitations": manifest["annotation_limits"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "approach": report["approach"],
        "heldout_manual": {k: report["heldout_manual_regions"][k] for k in ("observations", "accepted", "correct_accepted", "wrong_accepted", "coverage", "selective_accuracy")},
        "heldout_detection": {k: report["heldout_actual_detection_regions"][k] for k in ("observations", "accepted", "correct_accepted", "wrong_accepted", "coverage", "selective_accuracy")},
        "detector_video": report["detector_coverage"]["heldout_video_frames"],
        "fit_seconds": fit_seconds,
        "output": str(args.output.resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
