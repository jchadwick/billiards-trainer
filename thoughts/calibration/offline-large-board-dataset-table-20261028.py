#!/usr/bin/env python3
"""Saved-image audit of the larger board and candidate table-map transfer.

Research only: no camera access, production calibration, or ball coordinates.
The two stages have separate artifacts so data integrity can be reviewed before
using any optical fit. The nominal print dimensions are not physical truth.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import platform
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CAL = ROOT / 'thoughts/calibration'
OUT = CAL / 'large-board-dataset-table-20261028'
SESSION = CAL / 'checkerboard-20x14-session'


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, CAL / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


REG = load_module('table_registration_reference', 'offline-table-feature-registration-audit-20261027.py')
GRID = load_module('mark_grid_reference', 'offline-mark-grid-calibration-audit-20261027.py')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stats(values):
    values = np.asarray(values, float)
    return {'count': int(values.size), 'rms': float(np.sqrt(np.mean(values**2))),
            'median': float(np.median(values)), 'p95': float(np.percentile(values,95)),
            'max': float(np.max(values))}


def provenance(paths):
    return {'inputs_sha256': {str(Path(p).relative_to(ROOT)): digest(p) for p in paths},
            'script_sha256': digest(__file__), 'python': platform.python_version(),
            'opencv': cv2.__version__, 'numpy': np.__version__,
            'camera_access': False, 'production_calibration_emitted': False}


def save(name, record):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / (name+'.json')).write_text(json.dumps(record, indent=2, allow_nan=False)+'\n')


def grid_topology(points, pattern):
    grid = np.asarray(points).reshape(pattern[1], pattern[0], 2)
    a = grid[:-1,1:] - grid[:-1,:-1]
    b = grid[1:,:-1] - grid[:-1,:-1]
    cross = a[...,0]*b[...,1]-a[...,1]*b[...,0]
    valid = bool(np.all(cross>0) or np.all(cross<0))
    if not valid:
        raise ValueError('Corner grid has a degenerate/reversed cell')
    pitch = np.concatenate((np.linalg.norm(np.diff(grid,axis=0),axis=2).ravel(),
                            np.linalg.norm(np.diff(grid,axis=1),axis=2).ravel()))
    return {'all_cells_same_nonzero_winding': valid, 'cell_signed_area_px2_min_abs': float(np.min(abs(cross))),
            'adjacent_pitch_px': stats(pitch)}


def dataset_stage():
    path = SESSION/'manifest.json'
    m = json.loads(path.read_text())
    rows, hashes, pixels, tiles = [], set(), set(), []
    inputs = [path, CAL/'capture_checkerboard.py']
    for view in m['views']:
        p = SESSION/view['source_file']; inputs.append(p)
        raw = p.read_bytes(); encoded = hashlib.sha256(raw).hexdigest()
        im = cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR)
        assert im is not None
        pix = hashlib.sha256(im.tobytes()).hexdigest()
        assert encoded == view['source_sha256'] and pix == view['decoded_pixel_sha256']
        assert encoded not in hashes and pix not in pixels
        hashes.add(encoded); pixels.add(pix)
        assert [im.shape[1],im.shape[0]] == m['image_size_px'] == [3840,2160]
        corner_path=SESSION/view['corner_file']; inputs.append(corner_path)
        corner_record=json.loads(corner_path.read_text())
        pts=np.asarray(view['corners_source_image_px'],float)
        assert np.array_equal(pts,np.asarray(corner_record['corners_source_image_px']))
        assert pts.shape == (54,2) and np.isfinite(pts).all()
        assert np.all(pts>=0) and np.all(pts<np.array(m['image_size_px']))
        annotation_path=SESSION/view['annotated_file']; inputs.append(annotation_path)
        annotation=cv2.imread(str(annotation_path)); assert annotation.shape==im.shape
        gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        low=np.floor(pts.min(0)).astype(int); high=np.ceil(pts.max(0)).astype(int)
        roi=gray[low[1]:high[1]+1,low[0]:high[0]+1]
        topology=grid_topology(pts,m['pattern_inner_corners'])
        row={'id':view['id'],'source_sha256':encoded,'decoded_pixel_sha256':pix,
             'corner_count':len(pts),'corner_file_matches_manifest':True,
             'topology':topology,'center_fraction_image':(pts.mean(0)/m['image_size_px']).tolist(),
             'internal_corner_bbox_fraction_image':((high-low)/m['image_size_px']).tolist(),
             'board_roi_laplacian_variance_proxy':float(cv2.Laplacian(roi,cv2.CV_64F).var()),
             'board_roi_gray_p95_minus_p05':float(np.percentile(roi,95)-np.percentile(roi,5)),
             'capture_quality':view['quality'],'capture_note':view['notes']}
        rows.append(row)
        tile=cv2.resize(annotation,(640,360),interpolation=cv2.INTER_AREA)
        cv2.putText(tile,view['id'],(10,25),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,0,255),2)
        tiles.append(tile)
    OUT.mkdir(parents=True,exist_ok=True)
    cv2.imwrite(str(OUT/'dataset-montage.jpg'),np.vstack([np.hstack(tiles[i:i+2]) for i in range(0,len(tiles),2)]))
    report={'purpose':'dataset integrity, image support and coverage; exploratory only',
            'image_size_px':m['image_size_px'],'view_count':len(rows),'views':rows,
            'printed_scale_verification':m['printed_scale_verification'],
            'rigid_flat_backing_verification':m['rigid_flat_backing_verification'],
            'nominal_pitch_mm':m['square_pitch_mm'],'provenance':provenance(inputs),
            'limits':['Laplacian and contrast are descriptive proxies, not a calibrated sharpness/noise or corner-accuracy measurement.',
                      'Cell winding verifies topology, not independent correct detection or physical target geometry.',
                      'Board seams are visibly present; mounting flatness, tiled alignment and print scale are not verified.',
                      'All views have previously been inspected; leave-one-view-out results are cross-validation, not prospective unseen validation.',
                      'Board positions and perspective diversity are visible in the montage; image geometry alone cannot prove a physically flat target.']}
    save('dataset-audit',report)
    lines=['# Larger-board dataset audit — 2026-10-28','',
           'All eight retained JPEG and decoded-pixel hashes match the manifest and are unique. All sources and annotations are 3840 × 2160; every corner file matches its 54-corner manifest record. Every grid is finite, in bounds and has consistent nonzero cell winding.','',
           '| View | Median pitch px | Minimum pitch px | Contrast proxy | Laplacian variance proxy |','|---|---:|---:|---:|---:|']
    for r in rows:
        q=r['capture_quality']; lines.append(f"| {r['id']} | {q['adjacent_corner_pitch_px_median']:.2f} | {q['adjacent_corner_pitch_px_min']:.2f} | {r['board_roi_gray_p95_minus_p05']:.1f} | {r['board_roi_laplacian_variance_proxy']:.1f} |")
    lines+=['','The montage shows center, left/right and upper/lower coverage, several perspective views and a near-front-facing portrait view 8. View 5 has only about 21 source pixels of estimated pattern margin at the image edge; its minimum pitch is 42 px. Other images have larger margins.','',
            'The board has visible page seams. These images cannot verify physical flatness, precise square dimensions, or alignment of assembled pages. Preserve both physical verification fields as pending. Exploratory fitting may test numerical consistency using nominal geometry; no loadable calibration is justified by this integrity audit.','',
            'The contrast and sharpness proxies include the checkerboard edges and printing; values cannot rank actual corner accuracy across differing poses. The full-resolution originals and annotated images remain unchanged.','',
            'Reproduce with `/tmp/billiards-checkerboard-env/bin/python thoughts/calibration/offline-large-board-dataset-table-20261028.py --stage dataset`. Inspect `dataset-montage.jpg`; see JSON for exact inputs, checks and limitations.']
    (OUT/'dataset-audit.md').write_text('\n'.join(lines)+'\n')
    return report


def undistort(points,k,d):
    return cv2.undistortPoints(np.asarray(points,float).reshape(-1,1,2),k,d,P=k).reshape(-1,2)


def self_test():
    points=np.mgrid[:9,:6].T.reshape(-1,2).astype(float)*30+100
    grid_topology(points,[9,6])
    bad=points.copy(); bad[[0,10]]=bad[[10,0]]
    try: grid_topology(bad,[9,6])
    except ValueError: pass
    else: raise AssertionError('bad winding not rejected')
    k=np.array([[2100.,0,1920],[0,2090,1080],[0,0,1]])
    d=np.array([.025,-.003,0,0,0])
    xy=np.array([[-.5,-.2],[.4,.2],[0,0]],float)
    projected=cv2.projectPoints(np.column_stack((xy,np.ones(3))),np.zeros(3),np.zeros(3),k,d)[0].reshape(-1,2)
    corrected=undistort(projected,k,d)
    expected=(k@np.column_stack((xy,np.ones(3))).T).T[:,:2]
    assert np.max(abs(corrected-expected))<1e-6
    affine=np.array([[1.001,.002,.3],[-.001,.999,-.2],[0,0,1]])
    mapped=GRID.project_homography(affine,points)
    fitted=GRID.normalized_dlt(points,mapped)
    assert np.max(abs(GRID.project_homography(fitted,points)-mapped))<1e-7
    # Uniform image scaling transforms K and points together exactly.
    scale=np.diag([1/3,1/3,1]); scaled=scale@k
    assert np.max(abs(undistort(projected/3,scaled,d)-corrected/3))<1e-7
    # Recover a known rail/cloth separation after applying nonzero distortion.
    world=np.array([[100,-90],[600,-90],[1200,-90],[2200,-90],[300,1250],[2000,1250]],float)
    rotation=cv2.Rodrigues(np.array([.03,-.015,.01]))[0]
    translation=np.array([-1100.,-600.,2400.])
    ideal=GRID.project_camera(k,rotation,translation,np.column_stack((world,np.zeros(len(world)))))
    observed=distort(ideal,k,d)
    h=GRID.normalized_dlt(world,undistort(observed,k,d))
    recovered_r,recovered_t=GRID.decompose_pose(h,k)
    spots=np.array([[584.2,577.85],[1752.6,577.85]])
    spot_ideal=GRID.project_camera(k,rotation,translation,np.column_stack((spots,[55.,55.])))
    spot_raw=distort(spot_ideal,k,d)
    separation,_,errors=GRID.solve_plane_separation(k,recovered_r,recovered_t,spots,undistort(spot_raw,k,d))
    assert abs(separation-55)<1e-5 and max(errors)<1e-5
    for xy,uv in zip(spots,spot_ideal):
        recovered=GRID.unproject_to_fixed_plane(k,rotation,translation,uv,55.)
        assert np.max(abs(recovered-xy))<1e-7
    print('Synthetic topology rejection, registration, distortion/scale and known rail/cloth separation recovery passed')


def distort(points, k, d):
    homogeneous = np.column_stack((points, np.ones(len(points))))
    rays = (np.linalg.inv(k) @ homogeneous.T).T
    return cv2.projectPoints(rays, np.zeros(3), np.zeros(3), k, d)[0].reshape(-1, 2)


def table_geometry(registration, k, d, offset, width):
    # Keep measured pixels fixed. Distorted observations must be corrected
    # before fitting/decomposing a pinhole homography. No H is applied to K.
    ids, world, raw = GRID.get_rail_data(registration, 'framing_before_720')
    world[:, 1] = [-offset if name.startswith('top_') else width + offset for name in ids]
    ideal = undistort(raw, k, d)
    fit = GRID.fit_and_score_rail_plane(ids, world, ideal)
    h = np.asarray(fit['homography_world_mm_to_image_px'])
    rotation, translation = GRID.decompose_pose(h, k)
    annotations = registration['cloth_marker_annotations_by_image']['framing_before_720']
    cloth_raw = np.asarray([item['center_px'] for item in annotations], float)
    cloth_ideal = undistort(cloth_raw, k, d)
    xy = np.asarray([[GRID.TABLE_LENGTH_MM/4, width/2], [GRID.TABLE_LENGTH_MM*3/4, width/2]])
    separation, predicted, errors = GRID.solve_plane_separation(k, rotation, translation, xy, cloth_ideal)
    cross = []
    for index in range(2):
        other = 1-index
        single, _, _ = GRID.solve_plane_separation(k, rotation, translation, xy[index:index+1], cloth_ideal[index:index+1])
        point = GRID.project_camera(k, rotation, translation, np.column_stack((xy[other:other+1], [single])))
        raw_prediction = distort(point, k, d)[0]
        cross.append({'fit_marker': index+1, 'heldout_marker': other+1,
                      'diagnostic_plane_separation_mm': single,
                      'heldout_prediction_distorted_px': raw_prediction.tolist(),
                      'heldout_error_distorted_px': float(np.linalg.norm(raw_prediction-cloth_raw[other])),
                      'heldout_error_ideal_px': float(np.linalg.norm(point[0]-cloth_ideal[other]))})
    raw_predictions = distort(predicted, k, d)
    raw_fit = distort(GRID.project_homography(h, world), k, d)
    raw_loo=[]
    for i in range(len(ids)):
        keep=np.arange(len(ids))!=i
        loo_h=GRID.normalized_dlt(world[keep],ideal[keep])
        pred=distort(GRID.project_homography(loo_h,world[i:i+1]),k,d)[0]
        raw_loo.append(float(np.linalg.norm(pred-raw[i])))
    return {'sight_offset_mm': offset, 'width_assumption_mm': width,
            'rail_fit_ideal_px': fit['fit_error_px'],
            'rail_loo_ideal_px': fit['leave_one_sight_out_error_px'],
            'rail_fit_distorted_px': stats(np.linalg.norm(raw_fit-raw,axis=1)),
            'rail_loo_distorted_px': stats(raw_loo),
            'cloth_joint_error_ideal_px': errors.tolist(),
            'cloth_joint_error_distorted_px': np.linalg.norm(raw_predictions-cloth_raw,axis=1).tolist(),
            'diagnostic_joint_plane_separation_mm': separation,
            'one_spot_to_other': cross,
            'warning': 'Diagnostic rail/cloth nuisance separation only; no ball height or ball position is estimated.'}



def candidate_envelopes(registration, models):
    """Conditional map sensitivity of frozen candidates, never acceptance.

    The original refiner candidates came from a different exploratory optical
    profile. Holding source coordinates fixed is essential here: this tests
    coordinate conversion only, not estimator performance under new optics.
    """
    path=CAL/'center-outline-audit-20261028/summary.json'
    if not path.exists():
        return {'status':'not_evaluated_missing_center_audit'}, []
    evidence=json.loads(path.read_text())
    corpus_path=CAL/'native-localization-corpus-v1.json'
    assert digest(corpus_path)==evidence['producer_corpus']['sha256']
    # The compact numerical bundle is retained on main; the much larger
    # original trace archive remains on the experiment branch. Verify the
    # exact extraction bundle while retaining original trace hashes through
    # the center audit's provenance rather than requiring those archives.
    compact_path=CAL/'center-outline-audit-20261028/inputs.json'
    assert digest(compact_path)==evidence['compact_inputs_sha256']
    compact=json.loads(compact_path.read_text())
    assert compact['source_hashes']==evidence['source_hashes']
    measured_ids={row['ball_id'] for row in evidence['nine_reference_point_rows']}
    cases=[row for row in evidence['cases'] if row['source_image_id']=='measured-grid-720' and row['ball_id'] in measured_ids]
    assert len(cases)==9
    entries=[]
    ids, base_world, raw_rail=GRID.get_rail_data(registration,'measured_720')
    maps=[]
    for model_name,model in models.items():
        for evaluation in model['evaluations']:
            if evaluation['status']!='diagnostic_only': continue
            k=np.asarray(evaluation['K_720_from_verified_approximate_3x_framing'],float)
            d=np.asarray(evaluation['distortion_coefficients'],float)
            ideal=undistort(raw_rail,k,d)
            for variant in evaluation['variants'][:3]:
                world=base_world.copy()
                offset=variant['sight_offset_mm']; width=variant['width_assumption_mm']
                world[:,1]=[-offset if name.startswith('top_') else width+offset for name in ids]
                h=GRID.normalized_dlt(world,ideal)
                rotation,translation=GRID.decompose_pose(h,k)
                maps.append((model_name,evaluation['fit'],offset,k,d,rotation,translation,
                             variant['diagnostic_joint_plane_separation_mm']-GRID.BALL_RADIUS_MM))
    for case in cases:
        for iteration in case['iterations_with_qualified_candidates']:
            fits=iteration['fits']
            points=np.asarray([fit['center_source_px'] for fit in fits],float)
            trials=[]
            for model_name,fit_label,offset,k,d,r,t,plane in maps:
                corrected=undistort(points,k,d)
                xy=np.asarray([GRID.unproject_to_fixed_plane(k,r,t,p,plane) for p in corrected])
                distances=np.linalg.norm(xy[:,None]-xy[None,:],axis=2)
                trials.append({'model':model_name,'optics_fit':fit_label,'sight_offset_mm':offset,
                               'fit_ids':[fit['fit_id'] for fit in fits],
                               'conditional_xy_mm':xy.tolist(),
                               'within_iteration_candidate_max_pairwise_mm':float(distances.max())})
            all_xy=np.asarray([xy for trial in trials for xy in trial['conditional_xy_mm']])
            distances=np.linalg.norm(all_xy[:,None]-all_xy[None,:],axis=2)
            nominal=[trial for trial in trials if trial['optics_fit']=='full' and abs(trial['sight_offset_mm']-GRID.RAIL_SIGHT_OFFSET_MM)<1e-8]
            nominal_xy=np.asarray([xy for trial in nominal for xy in trial['conditional_xy_mm']])
            nominal_spread=float(np.linalg.norm(nominal_xy[:,None]-nominal_xy[None,:],axis=2).max())
            entries.append({'ball_id':case['ball_id'],'aiming_iteration':iteration['iteration'],
                            'candidate_count':len(fits),'precise_status_remains':case['status'],
                            'conditional_nominal_full_model_envelope_mm':nominal_spread,
                            'conditional_all_models_folds_offset_envelope_mm':float(distances.max()),
                            'trials':trials})
    return {'status':'conditional_sensitivity_only','cases_by_iteration':entries,
            'limits':['Frozen source candidate centers are rejected refiner outputs; no measurement or acceptance is created.',
                      'Each aiming iteration remains separate; no envelope pools fitting stages.',
                      'Changing optics is applied only to coordinate conversion; candidates are not a refiner rerun with new K/D.',
                      'Physical XY depends on the provisional rail lattice, two cloth spots and nominal fixed ball diameter.',
                      'The fixed radius projection offset is a constant shared by all balls, not a ball-height estimate.',
                      'Full-model envelope mixes incompatible exploratory optical hypotheses and is sensitivity, not a statistical confidence bound.',
                      'Folds/offset variants measure limited calibration sensitivity and do not bound unknown target or table construction error.']}, [path,compact_path,corpus_path]

def table_stage():
    manifest_path=SESSION/'manifest.json'
    registration_path=CAL/'table-feature-registration-audit-20261027/summary.json'
    optics_path=CAL/'large-board-optics-20261028/summary.json'
    manifest=json.loads(manifest_path.read_text())
    registration=json.loads(registration_path.read_text())
    optics=json.loads(optics_path.read_text())
    optical_models=dict(optics['models'])
    relocalized_path=CAL/'large-board-optics-20261028/relocalized-models.json'
    if relocalized_path.exists():
        corrected=json.loads(relocalized_path.read_text())['pitch-fraction-0.3']['models']
        optical_models.update({'relocalized-0.3/'+name:model for name,model in corrected.items()})
    inputs=[manifest_path,registration_path,optics_path,
            CAL/'offline-table-feature-registration-audit-20261027.py',
            CAL/'offline-mark-grid-calibration-audit-20261027.py']
    if relocalized_path.exists(): inputs.append(relocalized_path)
    for source in registration['image_sources'].values():
        source_path=ROOT/source['path']
        assert digest(source_path)==source['sha256'], source_path
        inputs.append(source_path)
    holdout={'top_rail_surface_1','top_rail_surface_6','bottom_rail_surface_1','bottom_rail_surface_4'}
    records=[]; tiles=[]
    # A declared research consistency check, not a camera calibration tolerance.
    # At native 4K this permits at most 3px sight displacement. Report the
    # actual discrepancy rather than treating this gate as physical accuracy.
    maximum_framing_displacement_720=1.0
    for view in manifest['views']:
        p=SESSION/view['source_file']; inputs.append(p)
        gray=cv2.imread(str(p),cv2.IMREAD_GRAYSCALE)
        assert gray.shape==(2160,3840)
        small=cv2.resize(gray,(1280,720),interpolation=cv2.INTER_AREA)
        detected={}
        for plane,seeds in REG.RAIL_SEEDS.items():
            if plane not in ('top_rail_surface','bottom_rail_surface'): continue
            for number, seed in enumerate(seeds,1):
                item=REG.detect_seeded_rail_sight(small,seed)
                if item: detected[f'{plane}_{number}']=item
        comparisons=[]
        for reference in ('framing_after_4k','measured_720','framing_before_720'):
            old=registration['rail_dot_annotations_by_image'][reference]
            names=[name for name in detected if old[name]['status']=='localized']
            new=np.asarray([detected[name]['center_px'] for name in names])
            ref=np.asarray([old[name]['center_px'] for name in names])
            train=np.asarray([name not in holdout for name in names])
            if train.sum()<4 or (~train).sum()<2:
                comparisons.append({'reference':reference,'status':'insufficient_visible_sights'}); continue
            # Estimate only the affine geometry needed to test unchanged
            # framing. A noisy projective fit with only one lower training
            # sight is weakly constrained and can invent corner motion.
            design=np.column_stack((new[train],np.ones(train.sum())))
            coefficients, _, rank, _=np.linalg.lstsq(design,ref[train],rcond=None)
            if rank!=3: raise ValueError('degenerate affine registration')
            h=np.vstack((coefficients.T,[0,0,1]))
            pred=GRID.project_homography(h,new[~train])
            direct=np.linalg.norm(new-ref,axis=1)
            samples=np.array([[0,0],[1279,0],[1279,719],[0,719],[640,360]],float)
            deviation=np.linalg.norm(GRID.project_homography(h,samples)-samples,axis=1)
            comparisons.append({'reference':reference,'status':'scored','train_ids':[name for name,t in zip(names,train) if t],
                                'heldout_ids':[name for name,t in zip(names,train) if not t],
                                'affine_new720_to_reference720':h.tolist(),
                                'holdout_transfer_error_720_px':stats(np.linalg.norm(pred-ref[~train],axis=1)),
                                'direct_same_pixel_displacement_720_px':stats(direct),
                                'affine_deviation_from_identity_at_frame_corners_and_center_720_px':deviation.tolist(),
                                'matches': [{'id':name,'new720':a.tolist(),'reference720':b.tolist()} for name,a,b in zip(names,new,ref)],
                                'supports_approximate_same_framing':bool(max(direct)<=maximum_framing_displacement_720 and max(deviation)<=maximum_framing_displacement_720)})
        record={'id':view['id'],'localized_sights':len(detected),'detected_annotations_720':detected,'comparisons':comparisons}
        records.append(record)
        tile=cv2.cvtColor(cv2.resize(small,(640,360)),cv2.COLOR_GRAY2BGR)
        for name,entry in detected.items():
            center=tuple(np.round(np.array(entry['center_px'])/2).astype(int))
            cv2.circle(tile,center,4,(0,0,255) if name in holdout else (0,255,0),1)
        cv2.putText(tile,view['id'],(10,25),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,0,255),2)
        tiles.append(tile)
    supported=all(len(row['comparisons'])==3 and all(c.get('supports_approximate_same_framing',False) for c in row['comparisons']) for row in records)
    OUT.mkdir(parents=True,exist_ok=True)
    cv2.imwrite(str(OUT/'registration-montage.jpg'),np.vstack([np.hstack(tiles[i:i+2]) for i in range(0,len(tiles),2)]))
    models={}
    if supported:
        for name, model in optical_models.items():
            fits=[('full',model.get('full_fit'))]+[(fold['held_out_view'],fold.get('training_fit')) for fold in model['folds']]
            entries=[]
            for label, fit in fits:
                if not fit or fit.get('issues'):
                    entries.append({'fit':label,'status':'blocked_implausible_or_missing_optics'}); continue
                k=np.diag([1/3,1/3,1])@np.asarray(fit['camera_matrix'],float)
                d=np.asarray(fit['distortion_k1_k2_p1_p2_k3'],float)
                variants=[]
                for offset_delta in (-GRID.RAIL_SIGHT_OFFSET_SENSITIVITY_MM,0,GRID.RAIL_SIGHT_OFFSET_SENSITIVITY_MM):
                    result=table_geometry(registration,k,d,GRID.RAIL_SIGHT_OFFSET_MM+offset_delta,GRID.TABLE_WIDTH_MM)
                    result['assumption']='user-measured 45.5in width; standard fractional sight layout'
                    variants.append(result)
                if label=='full':
                    width_variant=table_geometry(registration,k,d,GRID.RAIL_SIGHT_OFFSET_MM,46*25.4)
                    width_variant['assumption']='published 46in width sensitivity only; not an alternative measurement or production dimension'
                    variants.append(width_variant)
                entries.append({'fit':label,'status':'diagnostic_only','K_720_from_verified_approximate_3x_framing':k.tolist(),
                                'distortion_coefficients':d.tolist(),'variants':variants})
            models[name]={'optical_model_unstable_across_folds':model.get('unstable_across_folds'),'evaluations':entries}
    envelopes, extra_inputs=candidate_envelopes(registration,models) if models else ({'status':'blocked_no_transfer'},[])
    inputs+=extra_inputs
    report={'purpose':'same-framing support and provisional table-mark transfer under original and source-relocalized optics',
            'registration':records,'approximate_3x_scaling_supported':supported,
            'research_framing_consistency_limit_720_px':maximum_framing_displacement_720,
            'intrinsics_transfer':'K720 = diag(1/3,1/3,1) K4K only after approximate same framing checks; fitted projective rail homographies are never used to transform K',
            'models':models,'frozen_candidate_coordinate_sensitivity':envelopes,'provenance':provenance(inputs),
            'limits':['Rail-mark registration checks geometry; it cannot establish unchanged optical focus or exact subpixel pixel conventions.',
                      'Image resampling and binary sight localization produce small coordinate discrepancies; scaled intrinsics are diagnostic only.',
                      'Two cloth spots are correlated checks, not a full-table qualification. Their head/foot identities and standard rail layout are provisional.',
                      'Plane-separation fitting is a nuisance parameter for rail/cloth transfer; it does not estimate ball Z or publish any physical ball positions.',
                      'Optics retain pending target physical verification and finite-data uncertainty.',
                      'The 46in sensitivity variant probes dependence on standard layout; the authoritative table width remains 45.5in.',
                      'Existing width/offset sensitivity cannot cover arbitrary table-specific sight deviations. No loadable metric asset is emitted.']}
    save('table-results',report)
    lines=['# Saved-table transfer under larger-board optics — 2026-10-28','',
           'All registration uses ten existing long-rail white sight neighborhoods at 720-equivalent scale. Six fixed sights fit an affine transform and four outer sights are held out; when a sight is not detected, the remaining matching observations retain the same split. The additional direct-coordinate and corner checks constrain the unchanged-framing hypothesis. An affine transform is sufficient for this check and avoids an unstable projective estimate when only one lower-rail training sight remains. Green/red points in the registration montage denote fitting/held-out sights.','',
           '| New frame | Reference | Direct displacement RMS / max px720 | Held-out transfer RMS px720 |','|---|---|---:|---:|']
    for row in records:
        for c in row['comparisons']:
            if c['status']=='scored':
                e=c['direct_same_pixel_displacement_720_px']; lines.append(f"| {row['id']} | {c['reference']} | {e['rms']:.3f} / {e['max']:.3f} | {c['holdout_transfer_error_720_px']['rms']:.3f} |")
    lines+=['',f'Approximate same-framing support: **{supported}**. The fitted affine transforms and individual correspondences are retained in JSON. Intrinsics are scaled by exactly 1/3 only for this diagnostic comparison; no arbitrary rail-plane projective transform is applied to a camera matrix.','',
            '## Mark-grid comparison','',
            'Original and source-relocalized (0.30-pitch corner windows) optical candidates each undistort the same saved pixel observations with their own K and D, fit the standard fractional rail grid, decompose the resulting ideal-image homography, and fit a common rail-to-cloth nuisance separation using the two existing cloth spots. Held-out prediction fits the separation from one spot and predicts the other; scores below are re-distorted back to the original observed pixel domain. The separation fit minimizes ideal-image pixel error.','',
            '| Model | Rail fit RMS px720 | Rail leave-one-out RMS px720 | Cloth joint RMS px720 | Spot 1→2 / 2→1 px720 |','|---|---:|---:|---:|---:|']
    for name,model in models.items():
        nominal=model['evaluations'][0]['variants'][1]
        joint=stats(nominal['cloth_joint_error_distorted_px'])
        cross=nominal['one_spot_to_other']
        lines.append(f"| {name} | {nominal['rail_fit_distorted_px']['rms']:.3f} | {nominal['rail_loo_distorted_px']['rms']:.3f} | {joint['rms']:.3f} | {cross[0]['heldout_error_distorted_px']:.3f} / {cross[1]['heldout_error_distorted_px']:.3f} |")
    lines+=['','## Sensitivity and interpretation','',
            'Every usable full fit and withheld-board fit is evaluated at the nominal sight offset and ±3.175 mm. A separate full-fit-only variant substitutes the published 46-inch width to expose reliance on a standard-table assumption. It does not replace the user’s 45.5-inch measurement. JSON retains each residual and inferred rail/cloth separation; none is an application Z estimate.','']
    for name,model in models.items():
        variants=[v for e in model['evaluations'] if e['status']=='diagnostic_only' for v in e['variants'][:3]]
        cross=[c['heldout_error_distorted_px'] for v in variants for c in v['one_spot_to_other']]
        nominal=model['evaluations'][0]['variants'][1]
        lines.append(f"- {name}: across withheld-board optics and offset variants, one-spot prediction errors range {min(cross):.3f}–{max(cross):.3f} px720. Nominal joint rail/cloth nuisance separation is {nominal['diagnostic_joint_plane_separation_mm']:.3f} mm.")
    if envelopes.get('status')=='conditional_sensitivity_only':
        lines+=['','## Frozen-candidate coordinate sensitivity','',
                'The following envelopes convert the original rejected source centers through the provisional maps. They do not rerun the refiner under new optics, select an outline, qualify coordinates, or estimate ball height. Each aiming iteration is kept separate. The shared known-radius projection offset is constant for every ball.','',
                '| Ball | Aiming iteration | Nominal full-model envelope mm | Full/fold/offset envelope mm |',
                '|---|---:|---:|---:|']
        for entry in envelopes['cases_by_iteration']:
            lines.append(f"| {entry['ball_id']} | {entry['aiming_iteration']} | {entry['conditional_nominal_full_model_envelope_mm']:.3f} | {entry['conditional_all_models_folds_offset_envelope_mm']:.3f} |")
        lines+=['','These envelopes expose uncertainty in the coordinate transform as well as differences among frozen candidates. They are not confidence bounds or physical accuracy estimates. Unknown board geometry, home-table layout and reference uncertainty remain outside the envelope.']
    lines+=['','A low checkerboard score alone cannot qualify the table map. Compare the rail/felt behavior above with the optical holdouts. Standard sight identities, unknown home-table offsets, two cloth spots and unverified target construction prevent independent full-table millimeter accuracy claims. Any improvement or degradation is diagnostic rather than permission to activate a model.','',
            'No ball positions are fitted from references and no production position is emitted. Conditional rejected-candidate coordinates are sensitivity diagnostics only. The source images and original annotations are read-only. Reproduce with `/tmp/billiards-checkerboard-env/bin/python thoughts/calibration/offline-large-board-dataset-table-20261028.py --stage table`; run `--self-test` for synthetic topology, registration and distortion/scale consistency.']
    (OUT/'table-results.md').write_text('\n'.join(lines)+'\n')
    return report


def main():
    global OUT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['dataset','table','all'],default='all')
    parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--output',type=Path,default=OUT,help='Independent output directory for reproducibility checks')
    args=parser.parse_args()
    OUT=args.output
    if args.self_test: self_test(); return
    if args.stage in ('dataset','all'): dataset_stage()
    if args.stage in ('table','all'): table_stage()


if __name__=='__main__': main()
