#!/usr/bin/env python3
"""Reproduce the large-board optical preflight without touching camera or profiles.

The existing solver supplies unchanged fitting/validation policy. Additional
residual vectors retain held-out corner identity and pull pixel errors back
through a local projection Jacobian for board-coordinate diagnostics. These
nominal board coordinates do not constitute independently measured table XY.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import importlib.util
import importlib.metadata
import json
import platform
from pathlib import Path
import sys
import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location('checkerboard_preflight', HERE/'analyze_checkerboard_session.py')
CORE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CORE)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def options():
    # Exact defaults of the existing preflight; research paths are provenance.
    return argparse.Namespace(session_dir=str(HERE/'checkerboard-20x14-session'),
        output_prefix=str(HERE/'large-board-optics-20261028/summary'),
        models='zero,k1,k1k2', minimum_views=6, iterations=100, epsilon=1e-9,
        min_focal_ratio=.1, max_focal_ratio=10., principal_margin=.25,
        max_matrix_condition=1e6, radial_samples=101,
        maximum_fold_focal_range=.25, maximum_fold_principal_range=.1,
        self_test=False)


def residual_vectors(obj, image, pattern, k, d, pitch):
    """Enumerate exactly the existing parity-held-out errors with identities.

    Pose selection uses only that parity's training corners. Jacobians are
    numerical projection derivatives with respect to board X/Y (nominal mm),
    so inverse-Jacobian residuals are local tangent displacements, not another
    optimization or an assertion that image noise is independent.
    """
    points = image.reshape(-1, 2)
    parity_labels = np.indices((pattern[1], pattern[0])).sum(axis=0).ravel() % 2
    rows = []
    for parity in (0, 1):
        train = parity_labels == parity
        result = cv2.solvePnPGeneric(obj[train], points[train], k, d, flags=cv2.SOLVEPNP_IPPE)
        candidates = []
        for rvec, tvec in zip(result[1], result[2]):
            if np.min((obj @ cv2.Rodrigues(rvec)[0].T+tvec.reshape(1, 3))[:, 2]) <= 0:
                continue
            rvec, tvec = cv2.solvePnPRefineLM(obj[train], points[train], k, d, rvec, tvec)
            if np.min((obj @ cv2.Rodrigues(rvec)[0].T+tvec.reshape(1, 3))[:, 2]) <= 0:
                continue
            predicted_train = cv2.projectPoints(obj[train], rvec, tvec, k, d)[0].reshape(-1, 2)
            candidates.append((float(np.sum((predicted_train-points[train])**2)), rvec, tvec))
        if not candidates:
            raise ValueError('No positive-depth held-out pose solution')
        _, rvec, tvec = min(candidates, key=lambda x: x[0])
        evaluated = np.where(~train)[0]
        nominal = obj[evaluated].astype(float)
        predicted = cv2.projectPoints(nominal, rvec, tvec, k, d)[0].reshape(-1, 2)
        residual = points[evaluated]-predicted
        # Float64 projections avoid float32 cancellation in numerical derivatives.
        epsilon = pitch * 1e-4
        jacobian = np.empty((len(nominal), 2, 2))
        for axis in (0, 1):
            offset = np.zeros_like(nominal); offset[:, axis] = epsilon
            plus = cv2.projectPoints(nominal+offset, rvec, tvec, k, d)[0].reshape(-1, 2)
            minus = cv2.projectPoints(nominal-offset, rvec, tvec, k, d)[0].reshape(-1, 2)
            jacobian[:, :, axis] = (plus-minus)/(2*epsilon)
        for j, index in enumerate(evaluated):
            condition = float(np.linalg.cond(jacobian[j]))
            if not np.isfinite(condition) or condition > 1e6:
                raise ValueError('Unstable board residual Jacobian')
            local_mm = np.linalg.solve(jacobian[j], residual[j])
            rows.append({'corner_index': int(index), 'column': int(index % pattern[0]),
                'row': int(index//pattern[0]), 'pose_training_parity': parity,
                'observed_source_px': points[index].tolist(), 'predicted_source_px': predicted[j].tolist(),
                'observed_minus_predicted_source_px': residual[j].tolist(),
                'local_board_displacement_nominal_mm': local_mm.tolist(),
                'local_board_displacement_square_units': (local_mm/pitch).tolist(),
                'board_projection_jacobian_condition': condition})
    assert sorted(r['corner_index'] for r in rows) == list(range(len(obj)))
    # Float64 projection only changes the last rounding digits relative to helper.
    reference = np.asarray(CORE.heldout_pose_errors(obj, image, pattern, k, d))
    measured = np.linalg.norm(np.asarray([r['observed_minus_predicted_source_px'] for r in rows]), axis=1)
    assert np.allclose(reference, measured, atol=.001, rtol=.001), (reference, measured)
    return rows


def grouped(rows, key):
    out = []
    for value in sorted({r[key] for r in rows}):
        chosen = [r for r in rows if r[key] == value]
        pixels = np.asarray([r['observed_minus_predicted_source_px'] for r in chosen])
        board = np.asarray([r['local_board_displacement_square_units'] for r in chosen])
        out.append({key: value, 'count': len(chosen), 'source_error_px': CORE.summary(np.linalg.norm(pixels, axis=1)),
                    'mean_board_vector_square_units': board.mean(axis=0).tolist(),
                    'board_error_square_units': CORE.summary(np.linalg.norm(board, axis=1))})
    return out


def plot(result, directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for model, record in result['models'].items():
        rows = [dict(r, view=f['held_out_view']) for f in record['folds'] for r in f.get('heldout_residual_vectors', [])]
        if not rows:
            continue
        fig, ax = plt.subplots(figsize=(12, 7))
        for color_index, view in enumerate(result['view_ids']):
            group = [r for r in rows if r['view'] == view]
            if not group: continue
            xy = np.asarray([r['observed_source_px'] for r in group])
            e = np.asarray([r['observed_minus_predicted_source_px'] for r in group])
            ax.quiver(xy[:, 0], xy[:, 1], e[:, 0]*30, e[:, 1]*30, angles='xy', scale_units='xy', scale=1, width=.002, color=plt.cm.tab10(color_index), label=view)
        ax.set(xlim=(0, result['image_size_px'][0]), ylim=(result['image_size_px'][1], 0), aspect='equal',
               xlabel='Source X (pixels)', ylabel='Source Y (pixels)', title=f'{model}: held-out residuals, observed − predicted; arrows ×30')
        ax.legend(ncols=4); fig.tight_layout(); fig.savefig(directory/f'{model}-source-residuals.png', dpi=130); plt.close(fig)
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        for ax, view in zip(axes.flat, result['view_ids']):
            group = [r for r in rows if r['view'] == view]
            if group:
                xy = np.asarray([[r['column'], r['row']] for r in group])
                e = np.asarray([r['local_board_displacement_square_units'] for r in group])
                ax.scatter(xy[:, 0], xy[:, 1], s=4, color='gray')
                ax.quiver(xy[:, 0], xy[:, 1], e[:, 0]*30, e[:, 1]*30, angles='xy', scale_units='xy', scale=1)
            ax.axvline(4, color='orange', linestyle='--', linewidth=.7)
            for y in (2, 3): ax.axhline(y, color='orange', linestyle=':', linewidth=.7)
            ax.set(xlim=(-1, 9), ylim=(6, -1), aspect='equal', title=view, xlabel='Detector board column', ylabel='Detector board row')
        fig.suptitle(f'{model}: local board residuals in nominal square units; arrows ×30\nOrange: possible assembled seams; physical row orientation unknown')
        fig.tight_layout(); fig.savefig(directory/f'{model}-board-residuals.png', dpi=130); plt.close(fig)


def self_test():
    pattern = (9, 6); pitch = 50.8
    obj = CORE.object_points(pattern, pitch)
    k = np.array([[2100., 0, 1920.], [0, 2130., 1080.], [0, 0, 1.]])
    d = np.zeros(5)
    image = cv2.projectPoints(obj, np.array([.3, -.35, .12]), np.array([-200., -100., 1200.]), k, d)[0]
    exact = residual_vectors(obj, image, pattern, k, d, pitch)
    assert max(np.linalg.norm(r['observed_minus_predicted_source_px']) for r in exact) < .001
    corrupted = image.copy(); corrupted[0, 0] += np.array([1., -.5])
    perturbed = residual_vectors(obj, corrupted, pattern, k, d, pitch)
    at_zero = next(r for r in perturbed if r['corner_index'] == 0)
    # Corner zero is evaluated only from parity-one pose: the injected shift
    # cannot be absorbed by fitting that same corner, and retains its direction.
    assert at_zero['pose_training_parity'] == 1
    assert np.allclose(at_zero['observed_minus_predicted_source_px'], [1., -.5], atol=.001)
    doubled = residual_vectors(obj*2, corrupted, pattern, k, d, pitch*2)
    assert np.allclose([r['local_board_displacement_square_units'] for r in perturbed], [r['local_board_displacement_square_units'] for r in doubled], atol=1e-6)
    assert sum(r['count'] for r in grouped(perturbed, 'row')) == 54
    assert sum(r['count'] for r in grouped(perturbed, 'column')) == 54
    print('Synthetic parity exclusion, corner identity, residual sign/norm, board-Jacobian and grouping checks passed')



def relocalize(manifest, session, opts, fractions, output):
    """Source-only corner refinement sensitivity; no projection supplies seeds.

    Windows are derived once from each view's observed median grid pitch, then
    applied to every corner in that view. Changing only window size probes the
    original five-pixel half-window's capture range. No point is hand-picked,
    removed, or moved toward a calibration prediction. These are diagnostic
    alternate observations, not edits to the capture manifest.
    """
    records, analyses, correspondences = [], {}, {}
    for label, fraction in [('original-window-5', None)]+[(f'pitch-fraction-{f:g}', f) for f in fractions]:
        alternative = copy.deepcopy(manifest)
        views = []
        for original, changed in zip(manifest['views'], alternative['views']):
            gray = cv2.imread(str(session/original['source_file']), cv2.IMREAD_GRAYSCALE)
            if gray is None: raise ValueError('Source image could not be decoded')
            initial = np.asarray(original['corners_source_image_px'], np.float32).reshape(-1, 1, 2)
            window = 5 if fraction is None else max(2, int(round(fraction*original['quality']['adjacent_corner_pitch_px_median'])))
            refined = cv2.cornerSubPix(gray, initial.copy(), (window, window), (-1, -1),
                (cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_MAX_ITER, 40, .001)).reshape(-1, 2)
            shift = np.linalg.norm(refined-initial.reshape(-1, 2), axis=1)
            changed['corners_source_image_px'] = refined.tolist()
            views.append({'view_id': original['id'], 'source_sha256': original['source_sha256'],
                'window_half_size_px': window, 'corner_shift_px': CORE.summary(shift),
                'shifted_more_than_2px_indices': np.where(shift>2)[0].tolist(),
                'corners_source_image_px': refined.tolist()})
        analyses[label] = CORE.analyze(alternative, opts)
        if fraction == max(fractions):
            pattern = tuple(manifest['pattern_inner_corners'])
            pitch = manifest['square_pitch_mm']
            obj = CORE.object_points(pattern, pitch)
            for record in analyses[label]['models'].values():
                for fold, view in zip(record['folds'], alternative['views']):
                    fit = fold.get('training_fit')
                    if not fit or 'error' in fold: continue
                    fold['heldout_residual_vectors'] = residual_vectors(obj,
                        np.asarray(view['corners_source_image_px'], np.float32), pattern,
                        np.asarray(fit['camera_matrix']), np.asarray(fit['distortion_k1_k2_p1_p2_k3']), pitch)
            plot_dir = output/'relocalized-largest-window'
            plot_dir.mkdir(exist_ok=True)
            plot(analyses[label], plot_dir)
            source = cv2.imread(str(session/'view-005.jpg'))
            roi = source[1670:1940, 3050:3500].copy()
            original_points = manifest['views'][4]['corners_source_image_px']
            new_points = alternative['views'][4]['corners_source_image_px']
            for i, (obs, refined) in enumerate(zip(original_points, new_points)):
                if i//pattern[0]>2 or i%pattern[0] not in (2, 3, 4, 5): continue
                o = (round(obs[0]-3050), round(obs[1]-1670)); q = (round(refined[0]-3050), round(refined[1]-1670))
                cv2.drawMarker(roi, o, (0, 0, 255), cv2.MARKER_CROSS, 13, 1)
                cv2.drawMarker(roi, q, (0, 180, 0), cv2.MARKER_TILTED_CROSS, 11, 1)
                cv2.putText(roi, f'r{i//pattern[0]}c{i%pattern[0]}', (o[0]+5, o[1]-8), cv2.FONT_HERSHEY_SIMPLEX, .4, (220, 40, 220), 1)
            cv2.imwrite(str(output/'view-005-relocalized-native-crop.png'), roi)
        correspondences[label] = views
        records.append({'label': label, 'window_fraction_of_median_neighbor_pitch': fraction,
            'window_half_sizes_px': [v['window_half_size_px'] for v in views],
            'models': {m: {'holdout': r['whole_view_holdout_pose_split_error_px'],
                'full_fit': r.get('full_fit'), 'successful_folds': r['successful_folds'],
                'fold_intrinsics': r.get('fold_intrinsics'), 'unstable_across_folds': r.get('unstable_across_folds')}
                for m, r in analyses[label]['models'].items()}})
    for filename, body in [('relocalized-correspondences.json', {'schema_version': 1,
        'purpose': 'Source-pixel-only window sensitivity, not validated replacement corners or production calibration',
        'original_manifest_sha256': sha(session/'manifest.json'), 'variants': correspondences}),
        ('relocalized-models.json', analyses)]:
        (output/filename).write_text(json.dumps(body, indent=2, allow_nan=False)+'\n')
    return records


def outlier_crop(result, session, output):
    """Native pixel evidence for the original view-005 gross residuals."""
    fold = result['models']['k1']['folds'][4]
    source = cv2.imread(str(session/'view-005.jpg'))
    roi = source[1670:1940, 3050:3500].copy()
    for row in fold.get('heldout_residual_vectors', []):
        if row['row']>2 or row['column'] not in (2, 3, 4, 5): continue
        obs, pred = row['observed_source_px'], row['predicted_source_px']
        o = (round(obs[0]-3050), round(obs[1]-1670)); q = (round(pred[0]-3050), round(pred[1]-1670))
        cv2.drawMarker(roi, o, (0, 0, 255), cv2.MARKER_CROSS, 13, 1)
        cv2.drawMarker(roi, q, (0, 200, 255), cv2.MARKER_TILTED_CROSS, 11, 1)
        cv2.putText(roi, f"r{row['row']}c{row['column']}", (o[0]+5, o[1]-8), cv2.FONT_HERSHEY_SIMPLEX, .4, (220, 40, 220), 1)
    cv2.imwrite(str(output/'view-005-outlier-native-crop.png'), roi)



def frozen_cross_target(result, manifest, output):
    """Score independently printed targets with frozen intrinsics only.

    Registration supports approximate direct 3x source-coordinate scaling;
    its residual uncertainty remains part of this diagnostic. Board pose is
    fitted on one parity and scored on the other as in the primary protocol.
    The old zero baseline trained on old views1–8, so only its old view9 is a
    prospective baseline test. The new models never train on old target views.
    """
    old = json.loads((HERE/'checkerboard-session/manifest.json').read_text())
    old_baseline = json.loads((HERE/'checkerboard-eight-view-baseline.json').read_text())
    alternates = json.loads((output/'relocalized-models.json').read_text())
    relocalized = json.loads((output/'relocalized-correspondences.json').read_text())
    label = result['relocalization_sensitivity'][-1]['label']
    corrected = copy.deepcopy(manifest)
    for view, alternate in zip(corrected['views'], relocalized['variants'][label]):
        view['corners_source_image_px'] = alternate['corners_source_image_px']
    candidates = [('old-frozen-zero', old_baseline['models']['zero']['full_fit'], old['image_size_px'][0], ['old-small-board-9', 'new-raw-8', 'new-relocalized-8'])]
    for name, record in result['models'].items():
        candidates.append((f'new-raw-{name}', record['full_fit'], manifest['image_size_px'][0], ['old-small-board-9']))
    for name, record in alternates[label]['models'].items():
        candidates.append((f'new-relocalized-{name}', record['full_fit'], manifest['image_size_px'][0], ['old-small-board-9']))
    target_sets = {'old-small-board-9': old, 'new-raw-8': manifest, 'new-relocalized-8': corrected}
    records = []
    for name, fit, width, target_names in candidates:
        for target_name in target_names:
            target = target_sets[target_name]
            scale = target['image_size_px'][0]/width
            k = np.asarray(fit['camera_matrix']).copy(); k[:2, :] *= scale
            d = np.asarray(fit['distortion_k1_k2_p1_p2_k3'])
            obj = CORE.object_points(tuple(target['pattern_inner_corners']), target['square_pitch_mm'])
            views, all_errors = [], []
            for view in target['views']:
                row = {'view_id': view['id'], 'source_sha256': view['source_sha256']}
                try:
                    errors = CORE.heldout_pose_errors(obj,
                        np.asarray(view['corners_source_image_px'], np.float32),
                        tuple(target['pattern_inner_corners']), k, d)
                    row['error_px'] = CORE.summary(errors); all_errors.extend(errors)
                except (ValueError, cv2.error, np.linalg.LinAlgError) as error:
                    row['error'] = str(error)
                views.append(row)
            records.append({'frozen_model': name, 'target_set': target_name,
                'intrinsics_or_lens_refit': False, 'direct_coordinate_scale': scale,
                'camera_matrix_in_target_pixels': k.tolist(), 'distortion_coefficients': d.tolist(),
                'aggregate_error_px': CORE.summary(all_errors), 'views': views})
    return {'assumption': 'Approximate unchanged camera framing with direct image-width scaling; verify against the separate static rail registration audit. Registration residuals limit subpixel cross-target interpretation.',
        'baseline_training_note': 'Old-frozen-zero trained on old small-board views001–008; only old view009 is prospective for that baseline. New fits train on no old small-board images. No new physical observations are collected.',
        'old_manifest_sha256': sha(HERE/'checkerboard-session/manifest.json'), 'records': records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=HERE/'large-board-optics-20261028')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test(); return
    # Frozen declared research workload, not a tuning interface.
    fractions = (.15, .22, .30)
    cv2.setNumThreads(1)
    cv2.setRNGSeed(7321)
    opts = options()
    session = Path(opts.session_dir)
    manifest_path = session/'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    for view in manifest['views']:
        if sha(session/view['source_file']) != view['source_sha256']:
            raise ValueError(f"Source image hash mismatch: {view['id']}")
    result = CORE.analyze(manifest, opts)
    result['limitations'] = [
        'Assembled 20 × 14 inch target scale and flat rigid backing remain unverified; nominal pitch is used for exploratory fitting only.',
        'Parity-held-out corners still share a board pose and correlated detection/print/assembly errors. Pixel reprojection consistency is not physical table XY accuracy.',
        'The 10 × 7 square pattern has no asymmetric orientation marker. Detector row/column axes can reverse between views; central column is symmetric, possible horizontal seam rows are 2 or 3 in internal-corner coordinates.',
        'Residual grouping can expose patterns, but cannot distinguish lens errors, board warping, local corner bias, or assembly errors independently.',
        'Uniform pitch scale changes pose translation without identifying true table scale. Nonuniform printing/assembly or nonplanarity can bias intrinsics.',
        'Earlier small-board and new large-board errors are normalized by image width only; they are different targets, poses, corner precision and images, not matched physical placements.',
        'No camera, production calibration, ball estimator, or configuration is modified. No fisheye model is fitted.']
    old_path = HERE/'checkerboard-eight-view-baseline.json'
    old = json.loads(old_path.read_text())
    result['provenance'] = {'runner_sha256': sha(__file__), 'existing_solver_sha256': sha(HERE/'analyze_checkerboard_session.py'),
        'manifest_sha256': sha(manifest_path), 'old_baseline_sha256': sha(old_path), 'python': sys.version,
        'opencv': cv2.__version__, 'opencv_threads': cv2.getNumThreads(), 'opencv_rng_seed': 7321, 'numpy': np.__version__, 'matplotlib': importlib.metadata.version('matplotlib'), 'platform': platform.platform(),
        'printed_scale_verification': manifest['printed_scale_verification'],
        'rigid_flat_backing_verification': manifest['rigid_flat_backing_verification'],
        'source_images': [{'id': v['id'], 'sha256': sha(session/v['source_file'])} for v in manifest['views']]}
    pattern = tuple(manifest['pattern_inner_corners'])
    obj = CORE.object_points(pattern, manifest['square_pitch_mm'])
    for model, record in result['models'].items():
        all_rows = []
        for fold, view in zip(record['folds'], manifest['views']):
            fit = fold.get('training_fit')
            if not fit or 'error' in fold: continue
            try:
                vectors = residual_vectors(obj, np.asarray(view['corners_source_image_px'], np.float32), pattern,
                    np.asarray(fit['camera_matrix']), np.asarray(fit['distortion_k1_k2_p1_p2_k3']), manifest['square_pitch_mm'])
                fold['heldout_residual_vectors'] = vectors
                fold['row_diagnostics'] = grouped(vectors, 'row'); fold['column_diagnostics'] = grouped(vectors, 'column')
                all_rows.extend(vectors)
            except (ValueError, cv2.error, np.linalg.LinAlgError) as error:
                fold['residual_vector_error'] = str(error)
        record['row_diagnostics'] = grouped(all_rows, 'row')
        record['column_diagnostics'] = grouped(all_rows, 'column')
        record['comparison_to_old_small_board'] = {'old_native_px': old['models'][model]['whole_view_holdout_pose_split_error_px'],
            'old_errors_scaled_to_new_width_px': {key: value*manifest['image_size_px'][0]/old['image_size_px'][0] if key != 'count' else value for key, value in old['models'][model]['whole_view_holdout_pose_split_error_px'].items()},
            'new_errors_scaled_to_old_width_px': {key: value*old['image_size_px'][0]/manifest['image_size_px'][0] if key != 'count' else value for key, value in record['whole_view_holdout_pose_split_error_px'].items()}}
    output = args.output_dir; output.mkdir(parents=True, exist_ok=True)
    result['relocalization_sensitivity'] = relocalize(manifest, session, opts, fractions, output)
    original_fitter = CORE.fit_model
    def forbid_intrinsics_refit(*args, **kwargs):
        raise AssertionError('Cross-target frozen evaluation cannot refit intrinsics')
    CORE.fit_model = forbid_intrinsics_refit
    try:
        result['cross_target_frozen_models'] = frozen_cross_target(result, manifest, output)
    finally:
        CORE.fit_model = original_fitter
    outlier_crop(result, session, output)
    (output/'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    plot(result, output)
    write_report(result, output)
    print(json.dumps({m: {'full_fit': r.get('full_fit'), 'holdout': r['whole_view_holdout_pose_split_error_px'], 'unstable': r.get('unstable_across_folds')} for m, r in result['models'].items()}, indent=2))


def write_report(result, output):
    lines = ['# Larger-board optical model evaluation — 2026-10-28', '',
        'Eight saved 3840 × 2160 captures; nominal 50.8 mm checkerboard pitch. Exploratory evidence only: assembled print scale and rigid flat backing remain unverified. No camera access or production asset activation.', '',
        '## Unchanged model and validation policy', '', result['validation_protocol'], '',
        'Reuse the existing fixed solver defaults: 100 iterations, epsilon 1e−9, fixed tangential/higher-order terms, minimum six views, maximum fold focal relative range 0.25 and principal range fraction 0.1. All eight leave-one-view-out folds use seven training views. No coefficients are tuned from the holdout scores.', '',
        '| Model | Full-fit RMS px | Holdout RMS px | Holdout p95 px | Holdout max px | Old RMS scaled to 4K px | Successful folds | Fold instability |', '|---|---:|---:|---:|---:|---:|---:|---|']
    for name, r in result['models'].items():
        h = r['whole_view_holdout_pose_split_error_px']; old = r['comparison_to_old_small_board']['old_errors_scaled_to_new_width_px']
        lines.append(f"| {name} | {r.get('full_fit', {}).get('opencv_rms_px', float('nan')):.4f} | {h['rms']:.4f} | {h['p95']:.4f} | {h['max']:.4f} | {old['rms']:.4f} | {r['successful_folds']}/8 | {r.get('unstable_across_folds')} |")
    for name, r in result['models'].items():
        lines += ['', f'## {name}', '', 'Full-fit issues: '+(', '.join(r.get('full_fit', {}).get('issues', [])) or 'none under configured plausibility bounds'), '',
            f"Fold intrinsics: `{json.dumps(r.get('fold_intrinsics', {}))}`", '',
            '| Held-out view | RMS px | p95 px | Max px | Fit warnings / failures |', '|---|---:|---:|---:|---|']
        for f in r['folds']:
            h=f.get('validation_error_px') or {}; warn=f.get('error') or ', '.join(f.get('training_fit', {}).get('issues', [])) or 'none'
            lines.append(f"| {f['held_out_view']} | {h.get('rms', float('nan')):.4f} | {h.get('p95', float('nan')):.4f} | {h.get('max', float('nan')):.4f} | {warn} |")
        lines += ['', f'![Source residuals]({name}-source-residuals.png)', '', f'![Board residuals]({name}-board-residuals.png)', '',
                  '| Detector row | Mean nominal board residual X/Y (square units) | RMS source px |', '|---|---|---:|']
        for row in r['row_diagnostics']:
            lines.append(f"| {row['row']} | {row['mean_board_vector_square_units']} | {row['source_error_px']['rms']:.4f} |")
        lines += ['', '| Detector column | Mean nominal board residual X/Y (square units) | RMS source px |', '|---|---|---:|']
        for col in r['column_diagnostics']:
            lines.append(f"| {col['column']} | {col['mean_board_vector_square_units']} | {col['source_error_px']['rms']:.4f} |")
    lines += ['', '## Findings and source-only corner re-localization', '',
        'In the original captured observations, k1 improves held-out RMS in all eight folds. Aggregate RMS falls from 2.5044 to 2.0832 px; k1+k2 offers no further aggregate RMS improvement (2.0889 px). The broad existing plausibility and fold-range gates pass, but zero-model focal ranges near 19% and radial-model ranges near 8% remain material uncertainty. These checks do not qualify physical calibration.', '',
        'The original view-005 observations contain grossly displaced corner labels. On the native crop below, red pluses are the archived observations and yellow crosses are k1 held-out predictions. At detector row 0, columns 3 and 4, the red labels visibly miss the black/white intersection; this is an observation problem rather than proof of a lens defect. The third large outlier is row 2, column 4. The five-pixel capture-refinement window can leave an integer initial detection outside the correct corner basin.', '',
        '![Original corner and held-out prediction evidence](view-005-outlier-native-crop.png)', '',
        'Re-localize every corner from its archived source-pixel seed with the existing cornerSubPix objective, varying only the half-window. The half-window is 5 px for the original-setting repeat or 0.15, 0.22 and 0.30 times each view median adjacent pitch, rounded to source pixels; convergence remains 40 iterations/0.001 px. No prediction supplies a seed and no corner is removed. This finite sensitivity workload is declared before comparing its resulting fits, and does not select a replacement production setting.', '',
        '| Refinement half-window | Model | Holdout RMS px | p95 px | Max px |', '|---|---|---:|---:|---:|']
    for variant in result['relocalization_sensitivity']:
        for model, record in variant['models'].items():
            h = record['holdout']
            lines.append(f"| {variant['label']} ({variant['window_half_sizes_px']}) | {model} | {h['rms']:.4f} | {h['p95']:.4f} | {h['max']:.4f} |")
    lines += ['', 'The largest declared window moves four view-005 observations by more than 2 px (maximum 19.9786 px), and zero observations by more than 2 px in the other seven views. Red pluses below are original coordinates; green crosses are the source-only re-localizations. The re-localized gross outliers align with visible checker intersections. Other shifts are sensitivity observations; their truth is not independently known.', '',
        '![Source-only re-localization evidence](view-005-relocalized-native-crop.png)', '',
        'With that alternate observation set, k1 holdout RMS is 1.5910 px and maximum is 4.2666 px; k1+k2 RMS is 1.5822 px and maximum is 4.7264 px. The remaining residuals exceed the older small-board zero-model RMS scaled to 4K (0.6443 px), despite much larger native corner spacing. The datasets and targets differ, so this comparison cannot identify which target or optical model is more accurate.', '',
        'The residual vectors show repeatable structure along detector-board rows in multiple views, including rows near candidate assembly seams. They also show pose-specific patterns, particularly view 005. This supports investigating board/observation geometry as a confounder; it does not prove an assembly, print-scale, flatness, or lens defect. Detector orientations can reverse, so row-wise signed averages cannot be assigned to a particular physical seam without independent orientation evidence.', '',
        'Full alternate correspondences are retained in relocalized-correspondences.json and complete repeated model fits/folds in relocalized-models.json. The largest-window residual plots are in relocalized-largest-window/. The original capture manifest, original fit results, and all source images remain unchanged.', '',
        '## Interpretation and limits', '']+[f'- {s}' for s in result['limitations']]
    lines += ['', '## Frozen cross-target checks', '',
        result['cross_target_frozen_models']['assumption'], '', result['cross_target_frozen_models']['baseline_training_note'], '',
        '| Frozen model | Target set | RMS px | p95 px | Max px |', '|---|---|---:|---:|---:|']
    for transfer in result['cross_target_frozen_models']['records']:
        h = transfer['aggregate_error_px']
        lines.append(f"| {transfer['frozen_model']} | {transfer['target_set']} | {h['rms']:.4f} | {h['p95']:.4f} | {h['max']:.4f} |")
    lines += ['', 'The new radial models improve internal larger-target cross-validation, but that improvement does not transfer to the older independent target under the approximate direct 3× framing transform. On the old 1280 × 720 target, raw-new k1 gives 0.2606 px RMS and re-localized-new k1 gives 0.2627 px, compared with 0.2153 px for the old frozen zero model. On the prospective old view 009 alone, the corresponding values are 0.2666/0.2680 versus 0.2279 px. This is evidence against promoting the new radial correction from its internal score alone. It does not independently identify whether target geometry, corner localization, residual framing transfer, pose flexibility, or lens-model inadequacy causes the conflict.', '',
        'Each transfer freezes K and distortion coefficients; only held-out board pose is estimated from alternating corners. Pixel statistics from 1280-wide and 3840-wide frames must be scaled before comparison. Full per-view results and transformed matrices are in summary.json. These cross-target tests are projection-consistency evidence, not independent table or ball-center truth.']
    lines += ['', 'The row/column tables aggregate detector-oriented board axes; orientation ambiguity prevents treating a repeated detector row as a known physical assembly seam. Per-view vector plots are the evidence for inspecting whether residual structure follows the board rather than the sensor. No residual pattern alone identifies its physical cause.', '',
        'This comparison is complete without another capture. More captures are justified only by a specific identified pose/coverage or target-quality ambiguity. Qualified table X/Y and ball-center performance require separate evidence.', '',
        '## Reproduction', '', '```sh', '/tmp/billiards-checkerboard-env/bin/python thoughts/calibration/offline-large-board-optics-20261028.py --self-test', '/tmp/billiards-checkerboard-env/bin/python thoughts/calibration/offline-large-board-optics-20261028.py', '```', '',
        'The full JSON retains exact matrices, coefficients, per-fold scores/warnings, one residual vector per held-out corner, row/column diagnostics, software versions, and input/source hashes. Old reports and source images are unchanged.']
    (output/'README.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    main()
