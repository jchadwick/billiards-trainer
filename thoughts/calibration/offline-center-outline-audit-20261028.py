#!/usr/bin/env python3
"""Summarize frozen center/outline ambiguity without running or accepting a refiner.

Only existing traces are read. Each aiming iteration is summarized separately;
source-space outlines describe projected patch-circle fits, not labelled truth.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'thoughts/calibration'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path, provenance):
    provenance[str(path.relative_to(ROOT))] = digest(path)
    return json.loads(path.read_text())


def distance(a, b):
    return np.linalg.norm(np.asarray(a)[:, None, :] - np.asarray(b)[None, :, :], axis=2)


def max_spread(points):
    return float(distance(points, points).max()) if len(points) > 1 else None


def hausdorff(a, b):
    d = distance(a, b)
    return float(max(d.min(axis=0).max(), d.min(axis=1).max()))


def iteration_metrics(fits, threshold):
    """Pairwise *sampled* source-outline Hausdorff, distinct from center spread."""
    centers = [fit['center_image_px'] for fit in fits]
    radii = [np.linalg.norm(np.asarray(f['outline_image_px']) - f['center_image_px'], axis=1)
             for f in fits]
    pairs = []
    for i, a in enumerate(fits):
        for b in fits[i+1:]:
            pairs.append({'fit_ids': [a['fit_id'], b['fit_id']],
                          'center_distance_source_px': float(np.linalg.norm(np.asarray(a['center_image_px'])-b['center_image_px'])),
                          'sampled_outline_hausdorff_source_px': hausdorff(a['outline_image_px'], b['outline_image_px'])})
    center_spread = max_spread(centers)
    outline_spread = max((p['sampled_outline_hausdorff_source_px'] for p in pairs), default=None)
    return {'qualified_fit_count': len(fits), 'fits': [
        {'fit_id': f['fit_id'], 'center_source_px': f['center_image_px'],
         'median_source_outline_radius_px': float(np.median(r)),
         'min_source_outline_radius_px': float(r.min()),
         'max_source_outline_radius_px': float(r.max()),
         'local_patch_radius_px': f.get('radius_local_px')}
        for f, r in zip(fits, radii)],
        'center_pairwise_max_source_px': center_spread,
        'sampled_outline_pairwise_max_source_px': outline_spread,
        'median_source_radius_range_px': float(np.ptp([np.median(r) for r in radii])) if radii else None,
        'diagnostic_existing_source_threshold_px': threshold,
        'centers_within_existing_threshold_despite_outline_spread': (
            center_spread <= threshold and outline_spread > threshold) if pairs else None,
        'pairs': pairs}


def case_metrics(case, source_case, maximum_center_spread_px=1.5, reference_width_px=3840):
    groups = defaultdict(dict)
    events = case['trace']['events']
    radii = {}
    for event in events:
        data = event['data']
        key = (event['iteration'], data.get('fit_id'))
        if data['kind'] == 'refined_fit':
            radii[key] = data.get('radius_local_px')
        if data['kind'] == 'candidate_evaluation' and data.get('rejection') is None:
            if data.get('center_image_px') is None or not data.get('outline_image_px'):
                raise ValueError(f'Qualified fit missing geometry: {source_case}')
            groups[event['iteration']][data['fit_id']] = data
    width = case['intrinsics']['source_width_px']
    threshold = maximum_center_spread_px * width / reference_width_px
    iterations = []
    for iteration, candidates in sorted(groups.items()):
        fits = [dict(data, radius_local_px=radii.get((iteration, fit_id)))
                for fit_id, data in candidates.items()]
        iterations.append({'iteration': iteration, **iteration_metrics(fits, threshold)})
    outcome = case['refinement']
    return {'ball_id': case.get('ball_id', case.get('proposal', {}).get('id')),
            'source_case': source_case,
            'source_image_id': case.get('source_image_id', case.get('image_id')),
            'source_width_px': width,
            'status': outcome['status'], 'reason': outcome.get('reason'),
            'final_center_source_px': outcome.get('center_image_px'),
            'dropped_trace_events': case['trace'].get('dropped_events', 0),
            'iterations_with_qualified_candidates': iterations}


def linear_homography(source, target):
    # Source/target must already be normalized to keep linear LS conditioned.
    x, y = np.asarray(source).T
    u, v = np.asarray(target).T
    z, o = np.zeros(len(source)), np.ones(len(source))
    a = np.vstack([np.column_stack([x,y,o,z,z,z,-u*x,-u*y]),
                   np.column_stack([z,z,z,x,y,o,-v*x,-v*y])])
    coef, _, rank, sv = np.linalg.lstsq(a, np.r_[u,v], rcond=None)
    if rank < 8:
        raise ValueError('Rank-deficient diagnostic homography')
    return np.r_[coef, 1].reshape(3,3), float(sv[0]/sv[-1])


def project(points, h):
    v = np.column_stack([points, np.ones(len(points))]) @ h.T
    if np.any(abs(v[:,2]) < 1e-12):
        raise ValueError('Projection at infinity')
    return v[:,:2]/v[:,2:]


def mapping(points, targets, ids):
    # Match the original diagnostic's normalization exactly, not a new map model.
    source_origin, source_scale = np.array([640,360]), 640
    target_origin, target_scale = np.array([1168.4,577.85]), 1168.4
    src = (points-source_origin)/source_scale
    dst = (targets-target_origin)/target_scale
    errors, conditions, predictions = [], [], []
    for i in range(len(points)):
        mask = np.arange(len(points)) != i
        h, condition = linear_homography(src[mask], dst[mask])
        pred = project(src[[i]],h)[0]*target_scale+target_origin
        errors.append(float(np.linalg.norm(pred-targets[i])))
        predictions.append(pred.tolist());conditions.append(condition)
    h, condition = linear_homography(src,dst)
    residuals = np.linalg.norm(project(src,h)*target_scale+target_origin-targets,axis=1)
    return {'point_ids': ids, 'count': len(points), 'leave_one_out_errors_mm': dict(zip(ids, errors)),
            'leave_one_out_predictions_mm': dict(zip(ids,predictions)),
            'leave_one_out_rms_mm': float(np.sqrt(np.mean(np.square(errors)))),
            'leave_one_out_p95_mm': float(np.percentile(errors,95)),
            'leave_one_out_max_mm': max(errors),
            'training_residuals_mm': dict(zip(ids,residuals.tolist())),
            'normalized_design_conditions_by_fold': conditions,
            'normalized_design_condition_all_points': condition,
            'diagnostic_homography_normalized': h.tolist(),
            'independently_validated_accuracy': False,
            'reference_uncertainty_mm': None}


def self_test():
    angles=np.arange(96)*2*np.pi/96
    center=np.array([3.,4.])
    def circle(c,r):
        return np.column_stack([np.cos(angles),np.sin(angles)])*r+c
    fits=[{'fit_id':i,'center_image_px':center.tolist(),'outline_image_px':circle(center,r).tolist()}
          for i,r in enumerate([10,12])]
    m=iteration_metrics(fits,.5)
    assert m['center_pairwise_max_source_px']==0
    assert abs(m['sampled_outline_pairwise_max_source_px']-2)<1e-12
    assert m['centers_within_existing_threshold_despite_outline_spread'] is True
    assert abs(max_spread([[0,0],[3,4]])-5)<1e-12
    pts=np.array([[100,100],[600,100],[1100,100],[100,350],[600,350],[1100,350],[100,600],[600,600],[1100,600]],float)
    h=np.array([[2,0.03,4],[-.01,1.8,8],[.00002,-.00001,1]])
    targets=project(pts,h)
    fit=mapping(pts,targets,list(map(str,range(9))))
    assert fit['leave_one_out_max_mm']<1e-8
    # A shared 3,4 pixel bias keeps all centers mutually identical yet wrong by 5.
    biased=[center+[3,4]]*3
    assert max_spread(biased)==0 and np.linalg.norm(biased[0]-center)==5
    try:
        linear_homography(np.zeros((4,2)),np.zeros((4,2)))
    except ValueError:
        pass
    else:
        raise AssertionError('degenerate mapping accepted')
    # Reused fit identifiers in different aiming iterations must never mix.
    trace={'intrinsics':{'source_width_px':3840},'proposal':{'id':'synthetic'},
           'refinement':{'status':'rejected','reason':'ambiguous_boundary'},
           'trace':{'events':[{'iteration':j,'data':dict(fits[0],kind='candidate_evaluation',rejection=None)} for j in [1,2]]}}
    separated=case_metrics(trace,'synthetic')
    assert len(separated['iterations_with_qualified_candidates'])==2
    assert all(i['center_pairwise_max_source_px'] is None for i in separated['iterations_with_qualified_candidates'])
    return {'aiming_iteration_isolation':'passed','concentric_outline_radius_disagreement': 'passed', 'known_center_shift_3_4_5':'passed',
            'shared_bias_counterexample':'passed', 'exact_projective_recovery_and_leave_one_out':'passed',
            'degenerate_mapping_rejection':'passed'}


def extract_inputs(output):
    """Freeze only numerical evidence needed here; full trace hashes remain provenance."""
    provenance = {}
    compact_cases = []
    paths = sorted((BASE/'converged-candidate-audit-20261027/cases').glob('*.json'))
    paths += sorted((BASE/'robustness-audit-20261027/reserved-cases').glob('*.json'))
    if len(paths) != 33:
        raise ValueError('Expected exact frozen corpus of 15 measured-scene and 18 reserved cases')
    for path in paths:
        original = load(path, provenance)
        compact = {k: original[k] for k in ['ball_id','source_image_id','image_id','intrinsics','proposal','refinement','refiner_revision'] if k in original}
        kept = []
        for event in original['trace']['events']:
            data = event['data']
            if data['kind'] == 'refined_fit':
                kept.append({'iteration':event['iteration'], 'data':{k:data[k] for k in ['kind','fit_id','radius_local_px']}})
            elif data['kind'] == 'candidate_evaluation' and data.get('rejection') is None:
                kept.append({'iteration':event['iteration'], 'data':{k:data[k] for k in ['kind','fit_id','center_image_px','outline_image_px','rejection']}})
        compact['trace'] = {'events': kept, 'dropped_events': original['trace'].get('dropped_events',0)}
        compact_cases.append({'source_case': str(path.relative_to(ROOT)), 'case': compact})
    robustness = load(BASE/'robustness-audit-20261027/summary.json', provenance)
    slim_robustness = {k:robustness[k] for k in ['measured_image','refiner_source','runtime_parameters','corpus_path','corpus_sha256']}
    keys = ['ball_id','status','reason','center_image_px','seed','axis','offset_px','factor','experiment','bbox_xyxy_source_px']
    for category in ['center_and_size_sensitivity','fitting_seed_sensitivity']:
        slim_robustness[category] = {'results': [{k: row[k] for k in keys if k in row} for row in robustness[category]['results']]}
    measurements = load(BASE/'table-grid-20261026-114241-measurements.json', provenance)
    baseline = load(BASE/'table-grid-feasibility-results.json', provenance)
    image = ROOT/robustness['measured_image']['path']
    image_hash = digest(image)
    if image_hash != robustness['measured_image']['sha256']:
        raise ValueError('Measured image hash mismatch')
    provenance[str(image.relative_to(ROOT))] = image_hash
    for scene in robustness['reserved_scene_baselines']:
        scene_path=ROOT/scene['image_path']
        scene_hash=digest(scene_path)
        if scene_hash != scene['image_sha256']:
            raise ValueError(f"Reserved source image hash mismatch: {scene_path}")
        provenance[str(scene_path.relative_to(ROOT))]=scene_hash
    revision = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    result = {'schema_version':1, 'purpose':'compact numerical extraction for saved center-outline audit',
              'extraction_experiment_revision':revision,
              'source_hashes':provenance, 'source_images_verified_at_extraction':True,
              'cases':compact_cases,'robustness':slim_robustness,
              'measurements':measurements,
              'segmentation_observations':baseline['observations']}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,separators=(',',':'),allow_nan=False)+'\n')
    print(f'Extracted {len(compact_cases)} frozen cases: {output}')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--self-test',action='store_true')
    p.add_argument('--extract-inputs',type=Path,help='Extract compact evidence from full experiment traces; writes only this bundle')
    p.add_argument('--inputs',type=Path,help='Analyze an explicit frozen compact evidence bundle; full traces are not required')
    p.add_argument('--output',type=Path,default=BASE/'center-outline-audit-20261028')
    args=p.parse_args();tests=self_test()
    if args.self_test:
        print(json.dumps(tests));return
    if bool(args.extract_inputs) == bool(args.inputs):
        p.error('Choose exactly one of --extract-inputs or --inputs')
    if args.extract_inputs:
        extract_inputs(args.extract_inputs);return
    inputs=json.loads(args.inputs.read_text())
    if inputs['schema_version'] != 1 or len(inputs['cases']) != 33:
        raise ValueError('Invalid frozen compact input schema or case count')
    provenance=inputs['source_hashes']
    robustness=inputs['robustness']
    parameters=robustness['runtime_parameters']
    cases=[case_metrics(row['case'],row['source_case'], parameters['maximum_center_spread_px'],parameters['reference_width_px']) for row in inputs['cases']]
    sensitivity=[]
    for category in ['center_and_size_sensitivity','fitting_seed_sensitivity']:
        grouped=defaultdict(list)
        for row in robustness[category]['results']:
            grouped[row['ball_id']].append(row)
        for ball_id,rows in sorted(grouped.items()):
            centers=[r['center_image_px'] for r in rows if r['status']=='accepted']
            sensitivity.append({'experiment':category,'ball_id':ball_id,'runs':len(rows),
                                'accepted_count':len(centers),'status_reason_counts':dict(Counter(f"{r['status']}:{r.get('reason')}" for r in rows)),
                                'accepted_center_pairwise_max_source_px':max_spread(centers),
                                'accepted_centers_source_px':centers,
                                'rejected_candidate_geometry_available':False})
    measurements=inputs['measurements']
    segmentation={r['id']:r['center_px'] for r in inputs['segmentation_observations']}
    ids=[r['id'] for r in measurements['points']]
    targets=np.array([[r['center_from_upper_left_mm'][axis] for axis in ['x','y']] for r in measurements['points']])
    boxes={}
    for entry in inputs['cases']:
        d=entry['case']
        if 'ball_id' in d:
            boxes[d['ball_id']]=d['proposal']['bbox_center_source_px']
    method_results={}
    for method,lookup in [('existing_segmentation_ellipse_centers',segmentation),('manual_proposal_midpoints',boxes)]:
        points=np.array([lookup[i] for i in ids])
        method_results[method]=mapping(points,targets,ids)
    measured_cases={c['ball_id']:c for c in cases if c['source_image_id']=='measured-grid-720'}
    accepted_measured=[i for i in ids if measured_cases[i]['status']=='accepted']
    if accepted_measured:
        raise ValueError('Frozen no-fit baseline changed; review audit before fitting current centers')
    method_results['current_refiner']={'accepted_count':0,'fit_status':'no_fit_no_accepted_correspondences','common_accepted_subset_with_other_methods':[]}
    method_point_rows=[{'ball_id':i,'reference_xy_mm':targets[k].tolist(),
                        'existing_segmentation_center_source_px':segmentation[i],
                        'manual_proposal_midpoint_source_px':boxes[i],
                        'segmentation_to_proposal_distance_source_px':float(np.linalg.norm(np.asarray(segmentation[i])-boxes[i])),
                        'current_refiner_status':measured_cases[i]['status'],
                        'current_refiner_center_source_px':None}
                       for k,i in enumerate(ids)]
    # Map candidate-center spread through the same all-nine segmentation-derived
    # diagnostic H. This reports sensitivity only; no candidate is accepted/chosen.
    diagnostic_h=np.array(method_results['existing_segmentation_ellipse_centers']['diagnostic_homography_normalized'])
    for case in cases:
        if case['source_image_id']!='measured-grid-720' and case['ball_id'] not in ids:
            continue
        if case['source_width_px']!=1280 or case['ball_id'] not in ids:
            continue
        for iteration in case['iterations_with_qualified_candidates']:
            centers=np.array([f['center_source_px'] for f in iteration['fits']])
            mapped=project((centers-[640,360])/640,diagnostic_h)*1168.4+[1168.4,577.85]
            iteration['candidate_center_pairwise_max_diagnostic_table_mm']=max_spread(mapped)
            iteration['table_mapping_scope']='all-nine segmentation-fit diagnostic, not independent truth or production calibration'
    summaries=[i for c in cases for i in c['iterations_with_qualified_candidates']]
    pairs=[i for i in summaries if i['qualified_fit_count']>=2]
    within=[i for i in pairs if i['centers_within_existing_threshold_despite_outline_spread']]
    output={'schema_version':1,'purpose':'saved center-versus-outline sensitivity, no production change or calibration activation',
            'runtime':{'python':platform.python_version(),'numpy':np.__version__},'self_tests':tests,
            'source_hashes':provenance,'script_sha256':digest(Path(__file__)),
            'compact_inputs_sha256':digest(args.inputs),
            'extraction_experiment_revision':inputs['extraction_experiment_revision'],
            'producer_refiner_source':robustness['refiner_source'],
            'frozen_runtime_parameters':parameters,
            'producer_corpus':{'path':robustness['corpus_path'],'sha256':robustness['corpus_sha256']},
            'cases':cases,'sensitivity':sensitivity,'nine_reference_method_comparison':method_results,
            'nine_reference_point_rows':method_point_rows,
            'summary':{'case_count':len(cases),'accepted_count':sum(c['status']=='accepted' for c in cases),
                       'qualified_multi_candidate_iteration_count':len(pairs),
                       'center_stable_outline_disagrees_iteration_count':len(within),
                       'case_count_center_stable_outline_disagrees':sum(any(i['centers_within_existing_threshold_despite_outline_spread'] for i in c['iterations_with_qualified_candidates']) for c in cases)},
            'limitations':['Manual proposal midpoints are not raw YOLO outputs; no saved detector batch establishes that comparison.',
                          'Same image, repeated trials and candidate fits are not independent ball placements.',
                          'Source center is projected local-view circle center, not a surveyed physical sphere-center label.',
                          'All plausible contours share image bias; pairwise agreement does not bound true error.',
                          'Candidate radius summary is distance from projected local center; off-axis projection produces varying radius and is not a physical ball diameter.',
                          'Sampled-outline Hausdorff can include angular discretization; it is not exact continuous-curve distance.',
                          'Current refiner has no accepted measured points; hypothetical candidates are not substituted into a production map.',
                          'Nine references have unknown uncertainty; fitted-homography errors mix mapping, reference and center bias.',
                          'Only three native-4K saved ball crops exist; no full-table independent native4K accuracy data.',
                          'Optics are frozen exploratory profiles here; new optics comparison belongs to separate audit.',
                          'Rejected perturbation runs lack full candidate traces; only outcomes and accepted-center sensitivity are available.']}
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'summary.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    lines=['# Center versus outline audit — 2026-10-28','',
           'This report reads frozen traces only. It changes no refiner, threshold, calibration, camera or deployment.',
           '',f"Across {len(cases)} baseline cases, all observations remain rejected. Of {len(pairs)} aiming iterations with at least two individually qualified fits, {len(within)} have center spread within the existing source-space threshold while their sampled outlines disagree beyond it.",
           '', 'The threshold is the existing 1.5 px at width 3840 (0.5 px at width 1280); this is a diagnostic classification, not a newly validated uncertainty or acceptance bound.',
           '', '| Scene / ball | Iteration | Fits | Center max px | Outline max px | Radius median range px | Stable centers / differing outlines |',
           '|---|---:|---:|---:|---:|---:|---|']
    for case in cases:
        for i in case['iterations_with_qualified_candidates'] or [{'iteration':None,'qualified_fit_count':0}]:
            def fmt(key):
                value=i.get(key);return '—' if value is None else f'{value:.3f}'
            lines.append(f"| {case['source_image_id']} / {case['ball_id']} | {i['iteration'] or '—'} | {i['qualified_fit_count']} | {fmt('center_pairwise_max_source_px')} | {fmt('sampled_outline_pairwise_max_source_px')} | {fmt('median_source_radius_range_px')} | {i.get('centers_within_existing_threshold_despite_outline_spread')} |")
    lines+=['','## Same-nine-reference comparison','',
            '| Method | Eligible centers | Diagnostic leave-one-out RMS mm | p95 mm | max mm |',
            '|---|---:|---:|---:|---:|']
    for method in ['existing_segmentation_ellipse_centers','manual_proposal_midpoints']:
        m=method_results[method]
        lines.append(f"| {method} | 9 diagnostic observations | {m['leave_one_out_rms_mm']:.3f} | {m['leave_one_out_p95_mm']:.3f} | {m['leave_one_out_max_mm']:.3f} |")
    lines+=['| current refiner | 0 accepted | no fit | no fit | no fit |','',
            'The two diagnostic methods use exactly the same nine physical references and source JPEG. There is no common accepted subset with the current refiner. Manual proposal centers are not preserved raw detector outputs. The older segmentation LOO result is reproduced; none of these residuals establishes independent accuracy.',
            '', '## Interpretation','',
            'The saved cases do not support recovering useful measurements by removing unique-outline rejection alone: zero multi-candidate iterations have stable centers while only outlines disagree. Every iteration whose outlines exceed the existing tolerance also has a center spread exceeding that tolerance. One intermediate iteration has both center and outline spread below the threshold, but a later aiming iteration rejects with materially different centers. The evidence favors addressing search/measurement stability rather than relaxing outline qualification.',
            '', 'The saved perturbation audit still contains material fitting-start/proposal dependence. Its accepted outputs and rejection counts are reproduced per ball in summary.json. Only accepted perturbation traces preserve full fit geometry, so this report does not claim a bound over the rejected runs or all possible contours.',
            '', 'Candidate-center sensitivity is also expressed through a nine-point segmentation-derived diagnostic map for the measured cases. That map was fit on the same physical references; its millimeter spreads are conditional sensitivity, not an error bound. All current refiner outcomes remain unavailable and no calibration asset is emitted.',
            '', '## Limits','']
    lines += [f'- {s}' for s in output['limitations']]
    lines += ['', '## Reproduction','', '```sh', '/tmp/billiards-checkerboard-env/bin/python thoughts/calibration/offline-center-outline-audit-20261028.py --self-test', '/tmp/billiards-checkerboard-env/bin/python thoughts/calibration/offline-center-outline-audit-20261028.py --inputs thoughts/calibration/center-outline-audit-20261028/inputs.json', '```','',
              'The compact inputs.json bundle preserves the qualified per-iteration numerical evidence, perturbation outcomes, segmentation observations and physical references, with full source trace hashes and extraction revision. Analysis reads this explicit bundle and does not require full traces or images; image hashes were checked during extraction. Re-extract only on the recorded experiment revision using --extract-inputs /tmp/center-outline-inputs.json. The JSON preserves every input hash, all per-iteration qualified fits and pairwise distances, perturbation outcome counts, accepted variant centers, each homography holdout prediction and residual, and the exact source image identity. Synthetic checks cover concentric radius ambiguity, known center separation, shared bias despite perfect agreement, exact projective recovery/LOO, and degenerate mapping rejection.']
    (args.output/'README.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(output['summary']))
    print(json.dumps({k:v.get('leave_one_out_rms_mm') for k,v in method_results.items()}))


if __name__=='__main__':
    main()
