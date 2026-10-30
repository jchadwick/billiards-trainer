"""Frozen-trace, fixed-physical-size feasibility audit; never production acceptance.
Run with /tmp/billiards-checkerboard-env/bin/python. No camera access.
"""
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
BASE = ROOT / 'thoughts/calibration/actual-detector-refiner-final-20261029'
ASSET = ROOT / 'thoughts/calibration/runtime-localization-20261029/ball-center-720.json'


def project(h, p):
    q = h @ np.r_[p, 1.0]
    return q[:2] / q[2]


def fit_fixed(points, radius, initial):
    """Bounded two-parameter Cauchy IRLS, fitting only observed points."""
    c = np.array(initial, dtype=float)
    for _ in range(20):
        delta = c - points
        distances = np.linalg.norm(delta, axis=1)
        residual = distances - radius
        jacobian = delta / np.maximum(distances[:, None], 1e-9)
        scale = max(0.25, np.median(np.abs(residual)) * 1.4826)
        weights = 1 / (1 + (residual / (2.5 * scale)) ** 2)
        step = np.linalg.lstsq(jacobian * np.sqrt(weights[:, None]), -residual * np.sqrt(weights), rcond=None)[0]
        if np.linalg.norm(step) > radius:
            raise ValueError('unbounded update')
        c += step
        if np.linalg.norm(step) < 1e-8:
            break
    residual = np.linalg.norm(points - c, axis=1) - radius
    jacobian = (c - points) / np.maximum(np.linalg.norm(c - points, axis=1)[:, None], 1e-9)
    return c, float(np.sqrt(np.mean(residual ** 2))), float(np.linalg.cond(jacobian))


def synthetic_checks():
    cases = []
    rng = np.random.default_rng(121)
    truth = np.array([1.3, -0.7])
    for arc in [60, 120, 180, 240, 300]:
        t = np.linspace(-arc / 2, arc / 2, 60) * np.pi / 180
        points = truth + 10 * np.c_[np.cos(t), np.sin(t)]
        points += rng.normal(0, 0.08, points.shape)
        fits = []
        for ratio in [0.96, 1.0, 1.04]:
            c, rms, cond = fit_fixed(points, 10 * ratio, [0, 0])
            fits.append(dict(radius_ratio=ratio, center_error=float(np.linalg.norm(c-truth)), rms=rms, condition=cond))
        assert fits[1]['center_error'] < 0.15, (arc, fits)
        cases.append(dict(arc_degrees=arc, fits=fits))
    # A wrong radius can bias the center while producing a small residual.
    assert cases[0]['fits'][0]['center_error'] > 0.3
    return cases


def main():
    asset = json.loads(ASSET.read_text())
    mapping = np.array(asset['image_to_table_mm'])
    results = []
    for path in sorted((BASE / 'cases').glob('*.json')):
        d = json.loads(path.read_text())
        intr = d['intrinsics']
        k = np.array([[intr['fx'], 0, intr['cx']], [0, intr['fy'], intr['cy']], [0, 0, 1]])
        q = np.linalg.inv(k) @ np.linalg.inv(mapping)
        q /= (np.linalg.norm(q[:, 0]) + np.linalg.norm(q[:, 1])) / 2
        events = d['trace']['events']
        first = [e['data'] for e in events if e['iteration'] == 1]
        candidates = [e for e in first if e['kind'] == 'boundary_candidate']
        row = dict(trace_path=str(path.relative_to(ROOT)), trace_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), ball_id=d['ball_id'], runtime_status=d['refinement']['status'], runtime_reason=d['refinement'].get('reason'), runtime_stage=d['refinement']['quality']['stage'], observed_candidates=len(candidates), paths=[])
        if not candidates:
            row['conclusion'] = 'No recorded boundary candidates: a radius prior cannot repair this upstream rejection.'
            results.append(row)
            continue
        local = np.array([c['point_local_px'] for c in candidates])
        image = np.array([c['point_image_px'] for c in candidates])
        h, _ = cv2.findHomography(local, image, method=0)
        assert max(np.linalg.norm(project(h, p)-v) for p,v in zip(local,image)) < 1e-4
        # K^-1 H maps patch pixels into camera rays. Rotation columns have common
        # norm1/focal after arbitrary projective scaling has been removed.
        a = np.linalg.inv(k) @ h
        a /= np.linalg.norm(a[:, 2])
        focal = 2 / (np.linalg.norm(a[:, 0]) + np.linalg.norm(a[:, 1]))
        source = d['proposal']['bbox_center_source_px']
        physical = project(mapping, source)
        camera_ray = q @ np.r_[physical, 1]
        radius = focal * (asset['ball_diameter_mm']/2) / np.sqrt(np.dot(camera_ray, camera_ray) - (asset['ball_diameter_mm']/2)**2)
        row['predicted_local_radius'] = float(radius)
        row['local_focal'] = float(focal)
        for hypothesis in [e for e in first if e['kind'] == 'cyclic_hypothesis']:
            points = local[hypothesis['candidate_indices']]
            variants = []
            for ratio in [0.96, 1.0, 1.04]:
                c, rms, cond = fit_fixed(points, radius*ratio, hypothesis['center_local_px'])
                variants.append(dict(radius_ratio=ratio, center_source_px=project(h,c).tolist(), local_rms=rms, jacobian_condition=cond))
            centers = np.array([v['center_source_px'] for v in variants])
            row['paths'].append(dict(fit_id=hypothesis['fit_id'], unconstrained_local_radius=hypothesis['radius_local_px'], fixed_radius_variants=variants, source_center_spread_4percent=float(max(np.linalg.norm(a-b) for a in centers for b in centers))))
        row['conclusion'] = 'Geometry-only sensitivity; no pixel, coverage, quadrant, or source-validity qualification is performed.'
        results.append(row)
    report = dict(purpose='Research feasibility, not production acceptance or independent accuracy', dependencies=dict(numpy=np.__version__,opencv=cv2.__version__), calibration_sha256=hashlib.sha256(ASSET.read_bytes()).hexdigest(), synthetic=synthetic_checks(), cases=results)
    (HERE/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps([dict(id=r['ball_id'],status=r['runtime_status'],stage=r['runtime_stage'],radius=r.get('predicted_local_radius'),spread=max([p['source_center_spread_4percent'] for p in r['paths']],default=None)) for r in results],indent=2))

if __name__ == '__main__':
    main()
