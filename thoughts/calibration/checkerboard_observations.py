#!/usr/bin/env python3
"""Versioned source-pixel checkerboard observations, with explicit review gating.

No camera access or camera-model fitting exists in this tool. Original source
images and manifests are read-only. Neighborhood agreement and alternating
quadrant appearance are image-support diagnostics, never physical accuracy.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import platform
import sys

import cv2
import numpy as np

REVISION = 1
ROOT = Path(__file__).resolve().parents[2]
CAL = ROOT / 'thoughts/calibration'


@dataclass(frozen=True)
class Config:
    window_fractions: tuple[float, ...] = (.15, .22, .30)
    baseline_half_window: int = 5
    maximum_iterations: int = 40
    convergence_epsilon_px: float = .001
    agreement_reference_px: float = .5
    pixel_reference_width: int = 3840
    maximum_identity_shift_fraction: float = .45
    support_radii_fractions: tuple[float, ...] = (.10, .16, .22)
    support_patch_fraction: float = .025
    minimum_alternating_contrast_gray: float = 12.
    minimum_alternation_consistency: float = .55
    crop_half_pitch_fraction: float = .48
    contact_sheet_columns: int = 9
    display_minimum_crop_side_px: int = 64
    minimum_parity_corners: int = 8
    minimum_parity_hull_fraction: float = .30
    minimum_axis_span_fraction: float = .50


def digest(data):
    return hashlib.sha256(data).hexdigest()


def topology(points, pattern, size):
    p = np.asarray(points, float)
    if p.shape != (pattern[0]*pattern[1], 2) or not np.isfinite(p).all():
        raise ValueError('invalid_corner_array')
    if np.any(p < 0) or np.any(p >= np.asarray(size)):
        raise ValueError('corner_outside_source')
    grid = p.reshape(pattern[1], pattern[0], 2)
    u = grid[:-1, 1:] - grid[:-1, :-1]
    v = grid[1:, :-1] - grid[:-1, :-1]
    cross = u[..., 0]*v[..., 1]-u[..., 1]*v[..., 0]
    if not (np.all(cross > 0) or np.all(cross < 0)):
        raise ValueError('inconsistent_or_degenerate_cell_winding')
    if len(np.unique(p, axis=0)) != len(p):
        raise ValueError('duplicate_corner_identity')
    return grid


def neighborhood(grid, row, col):
    """Estimate two local grid directions without a fitted optical model."""
    rows, cols = grid.shape[:2]
    u = grid[row, min(col+1, cols-1)]-grid[row, max(col-1, 0)]
    v = grid[min(row+1, rows-1), col]-grid[max(row-1, 0), col]
    distances = []
    for rr, cc in ((row-1,col),(row+1,col),(row,col-1),(row,col+1)):
        if 0 <= rr < rows and 0 <= cc < cols:
            distances.append(float(np.linalg.norm(grid[row,col]-grid[rr,cc])))
    return u/np.linalg.norm(u), v/np.linalg.norm(v), min(distances)


def support(gray, point, u, v, pitch, config):
    """Sample four alternating sectors in local row/column directions.

    A complete checker intersection has similar opposite sectors and different
    neighboring sectors. This is a conservative appearance check; a stripe,
    print seam or blur can still bias it, so review is independently required.
    """
    result = []
    half = max(1, int(np.floor(pitch*config.support_patch_fraction)))
    patch_width = 2*half+1
    for fraction in config.support_radii_fractions:
        samples = []
        for su, sv in ((-1,-1),(1,-1),(-1,1),(1,1)):
            xy = np.asarray(point)+(su*u+sv*v)*pitch*fraction
            if (xy[0]-half < 0 or xy[1]-half < 0 or
                    xy[0]+half >= gray.shape[1] or xy[1]+half >= gray.shape[0]):
                return {'supported': False, 'reason': 'support_patch_outside_source', 'scales': result}
            patch = cv2.getRectSubPix(gray, (patch_width,patch_width), tuple(map(float,xy)))
            samples.append(float(np.mean(patch)))
        a,b,c,d = samples
        contrast = abs((a+d-b-c)/2)
        opposite_difference = (abs(a-d)+abs(b-c))/2
        consistency = contrast/(contrast+opposite_difference+np.finfo(float).eps)
        result.append({'radius_fraction': fraction, 'quadrant_mean_gray': samples,
                       'alternating_contrast_gray': contrast,
                       'alternation_consistency': consistency})
    supported = all(r['alternating_contrast_gray'] >= config.minimum_alternating_contrast_gray
                    and r['alternation_consistency'] >= config.minimum_alternation_consistency
                    for r in result)
    return {'supported': supported, 'reason': None if supported else 'insufficient_alternating_sector_support',
            'scales': result}


def validate_observation(image, raw, pattern, config):
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    size = [gray.shape[1], gray.shape[0]]
    grid = topology(raw, pattern, size)
    distances = np.r_[np.linalg.norm(np.diff(grid,axis=0),axis=2).ravel(),
                      np.linalg.norm(np.diff(grid,axis=1),axis=2).ravel()]
    median_pitch = float(np.median(distances))
    specifications = [('baseline',config.baseline_half_window)] + [
        (f'pitch-{f:g}',max(1,int(np.floor(f*median_pitch)))) for f in config.window_fractions]
    variants = []
    for label, half_window in specifications:
        # Every refinement starts from the immutable raw observation. Refining
        # one result from another would hide basin dependence and large errors.
        value = cv2.cornerSubPix(gray,np.asarray(raw,np.float32).reshape(-1,1,2).copy(),
                                (half_window,half_window),(-1,-1),
                                (cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_MAX_ITER,
                                 config.maximum_iterations,config.convergence_epsilon_px)).reshape(-1,2)
        reason = None
        try:
            topology(value,pattern,size)
        except ValueError as error:
            reason = str(error)
        variants.append({'label': label,'half_window_px': half_window,
                         'corners_source_px': value.tolist(),'topology_rejection': reason})
    corners = []
    agreement_limit = config.agreement_reference_px*size[0]/config.pixel_reference_width
    for i, original in enumerate(raw):
        row,col = divmod(i,pattern[0]);u,v,pitch = neighborhood(grid,row,col)
        supports = {'raw':support(gray,original,u,v,pitch,config)}
        positions = []
        changes = []
        reasons = []
        for variant in variants:
            point = np.asarray(variant['corners_source_px'][i])
            positions.append(point)
            supports[variant['label']] = support(gray,point,u,v,pitch,config)
            shift = float(np.linalg.norm(point-original));changes.append(shift)
            if shift > config.maximum_identity_shift_fraction*pitch:
                reasons.append('refinement_exceeds_identity_neighborhood')
            if variant['topology_rejection']:
                reasons.append('refined_grid_'+variant['topology_rejection'])
        positions = np.asarray(positions)
        spread = float(np.max(np.linalg.norm(positions[:,None]-positions[None,:],axis=2)))
        if spread > agreement_limit:
            reasons.append('neighborhood_disagreement')
        if not all(record['supported'] for record in supports.values()):
            reasons.append('insufficient_image_intersection_support')
        corners.append({'index':i,'row':row,'column':col,'raw_source_px':list(map(float,original)),
                        'refinements_source_px':{x['label']:x['corners_source_px'][i] for x in variants},
                        'raw_to_refinement_shifts_px':dict(zip((x['label'] for x in variants),changes)),
                        'maximum_neighborhood_spread_px':spread,'agreement_limit_source_px':agreement_limit,
                        'local_neighbor_pitch_min_px':pitch,'image_support':supports,
                        'machine_supported':not reasons,'rejection_reasons':sorted(set(reasons))})
    return {'median_neighbor_pitch_px':median_pitch,'variants':variants,'corners':corners,
            'machine_supported_corner_count':sum(c['machine_supported'] for c in corners),
            'machine_supported_view':all(c['machine_supported'] for c in corners)}


def contact_sheets(image, record, pattern, config, directory):
    """Source crop contact sheets; tiny crops enlarge by nearest-neighbor for display only."""
    half = max(8,int(record['median_neighbor_pitch_px']*config.crop_half_pitch_fraction))
    label_height = 30
    source_side=2*half+1
    display_scale=max(1,int(np.ceil(config.display_minimum_crop_side_px/source_side)))
    tile_width, tile_height = source_side*display_scale, source_side*display_scale+label_height
    columns = config.contact_sheet_columns
    rows = int(np.ceil(len(record['corners'])/columns))
    original = np.full((rows*tile_height,columns*tile_width,3),35,np.uint8)
    refined = original.copy()
    last_label = record['variants'][-1]['label']
    for c in record['corners']:
        x,y = np.rint(c['raw_source_px']).astype(int)
        x0,y0 = x-half,y-half
        tile = np.full((2*half+1,2*half+1,3),35,np.uint8)
        xa,ya,xb,yb=max(0,x0),max(0,y0),min(image.shape[1],x+half+1),min(image.shape[0],y+half+1)
        tile[ya-y0:yb-y0,xa-x0:xb-x0]=image[ya:yb,xa:xb]
        rr,cc = divmod(c['index'],columns); xx,yy=cc*tile_width,rr*tile_height
        display_tile=cv2.resize(tile,(tile_width,tile_width),interpolation=cv2.INTER_NEAREST)
        for sheet, point, color in ((original,c['raw_source_px'],(0,0,255)),
                                     (refined,c['refinements_source_px'][last_label],(0,255,0))):
            sheet[yy+label_height:yy+tile_height,xx:xx+tile_width]=display_tile
            label=f"{c['row']},{c['column']} {'OK?' if c['machine_supported'] else 'FLAG'}"
            cv2.putText(sheet,label,(xx+2,yy+17),cv2.FONT_HERSHEY_SIMPLEX,.33,(240,240,240),1)
            mark=tuple(np.rint((np.asarray(point)-[x0,y0]+.5)*display_scale-.5+[xx,yy+label_height]).astype(int))
            cv2.drawMarker(sheet,mark,color,cv2.MARKER_CROSS,7,1)
    paths=[]
    for label,sheet in [('raw',original),('largest-window',refined)]:
        path=directory/f'{record["view_id"]}-{label}-corners.png'
        if not cv2.imwrite(str(path),sheet):raise OSError(f'Could not write {path}')
        paths.append(path.name)
    return paths


def make_session(session, output, config, reviews=None):
    path=session/'manifest.json';manifest_bytes=path.read_bytes();manifest=json.loads(manifest_bytes)
    pattern=tuple(manifest['pattern_inner_corners']);records=[]
    output.mkdir(parents=True,exist_ok=True)
    for view in manifest['views']:
        source=session/view['source_file'];raw=source.read_bytes()
        if digest(raw)!=view['source_sha256']:raise ValueError(f'Source hash mismatch: {source}')
        image=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR)
        if image is None:raise ValueError(f'Undecodable source: {source}')
        if [image.shape[1],image.shape[0]]!=manifest['image_size_px']:raise ValueError('Mixed source dimensions')
        if digest(image.tobytes())!=view['decoded_pixel_sha256']:raise ValueError('Decoded pixel hash mismatch')
        record={'view_id':view['id'],'source_sha256':view['source_sha256'],
                'decoded_pixel_sha256':view['decoded_pixel_sha256'],'source_file':view['source_file'],
                'image_size_px':manifest['image_size_px'],
                'review_state':'pending','eligible_for_frozen_scoring':False,
                'eligibility_reasons':['independent_pixel_review_pending']}
        try:
            record.update(validate_observation(image,view['corners_source_image_px'],pattern,config))
        except (ValueError,cv2.error) as error:
            record.update({'machine_supported_view':False,'machine_supported_corner_count':0,
                           'validation_error':str(error),'corners':[],'variants':[]})
        if not record['machine_supported_view']:record['eligibility_reasons'].append('machine_validation_rejected')
        if record['corners']:record['native_corner_contact_sheets']=contact_sheets(image,record,pattern,config,output)
        review = next((r for r in (reviews or []) if r['session'] == session.name
                       and r['view_id'] == record['view_id']), None)
        if review is not None:
            apply_review(record,review,pattern,config)
        records.append(record)
    result={'schema_version':1,'observation_refiner_revision':REVISION,
            'purpose':'Source-only observation support diagnostics; no calibrated accuracy or production asset',
            'source_manifest_sha256':digest(manifest_bytes),'source_session':str(session.relative_to(ROOT)) if session.is_relative_to(ROOT) else str(session),
            'image_size_px':manifest['image_size_px'],'pattern_inner_corners':list(pattern),
            'nominal_square_pitch_mm':manifest['square_pitch_mm'],
            'printed_scale_verification':manifest['printed_scale_verification'],
            'rigid_flat_backing_verification':manifest['rigid_flat_backing_verification'],
            'configuration':asdict(config),'configuration_sha256':digest(json.dumps(asdict(config),sort_keys=True).encode()),'views':records,'eligible_view_count':sum(v['eligible_for_frozen_scoring'] for v in records),
            'provenance':{'script_sha256':digest(Path(__file__).read_bytes()),'opencv':cv2.__version__,
                          'numpy':np.__version__,'python':platform.python_version()},
            'camera_access':False,'intrinsics_fitted':False,'source_mutation':False}
    (output/'observations.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    lines=['# Source-only checkerboard observations','',
           'All views require independent pixel review. Machine agreement is not corner accuracy. Window disagreement is retained as a flag; source-bound review can select a visibly supported existing refinement. Missing intersection support or invalid geometry cannot be bypassed. Pending review is ineligible for scoring. Sources and raw coordinates are preserved.', '',
           '| View | Machine-supported corners | Maximum window spread px | Scoring |',
           '|---|---:|---:|---|']
    for r in records:
        maximum=max((c['maximum_neighborhood_spread_px'] for c in r['corners']),default=0)
        state='eligible' if r['eligible_for_frozen_scoring'] else ', '.join(r['eligibility_reasons'])
        lines.append(f"| {r['view_id']} | {r['machine_supported_corner_count']}/{len(r['corners'])} | {maximum:.3f} | {state} |")
    lines += ['', 'Raw and largest-window contact sheets show every source corner. Tiny source crops are enlarged only for display with nearest-neighbor interpolation (minimum 64-pixel cells); all measurements retain native source coordinates. Red crosses are original observations, green crosses show the largest declared refinement window. `FLAG` indicates at least one image-support, identity or window-agreement rejection; `OK?` is only machine support pending review.', '',
              'Quadrant contrast and consistency are appearance heuristics. Paper seams, blur, shadows and shared localization bias can invalidate them. No image-derived number here establishes physical target scale, flatness or a true location uncertainty.', '',
              'Original images, capture manifests, original corners and camera models remain unchanged. JSON stores every neighborhood result, quadrant sample, per-corner rejection and review state. A frozen-model evaluator requires reviewed source-bound observations and at least eight supported corners in each parity with spatial coverage. Every excluded corner remains in the denominator and record. Reviewer selections cannot depend on camera residuals. Review records bind source hash, revision and configuration.']
    (output/'README.md').write_text('\n'.join(lines)+'\n')
    return result


def apply_review(record, review, pattern, config):
    """Source-bound pixel review may resolve window flags, never missing support.

    Review chooses an existing source-only refinement, not a coordinate inferred
    from a camera fit. All alternative locations and original flags remain.
    """
    if review['source_sha256'] != record['source_sha256']:
        raise ValueError('Review source hash does not match immutable JPEG')
    config_hash=digest(json.dumps(asdict(config),sort_keys=True).encode())
    if review.get('observation_refiner_revision') != REVISION or review.get('configuration_sha256') != config_hash:
        raise ValueError('Review revision/configuration does not match observations')
    if not review.get('reviewer') or not review.get('reviewed_at_utc'):
        raise ValueError('Review requires reviewer identity and timestamp')
    selections=review.get('accepted_corners',[])
    ids=[c['index'] for c in selections]
    if len(ids)!=len(set(ids)) or any(not isinstance(i,int) or i<0 or i>=len(record['corners']) for i in ids):
        raise ValueError('Review has invalid or duplicate corner identities')
    selected={}
    for selection in selections:
        i=selection['index'];label=selection['variant_label'];corner=record['corners'][i]
        if label not in corner['refinements_source_px']:
            raise ValueError('Review selected unknown source-only neighborhood')
        variant=next(v for v in record['variants'] if v['label']==label)
        if variant['topology_rejection'] or not corner['image_support'][label]['supported']:
            raise ValueError('Review cannot override missing intersection support or invalid geometry')
        selected[i]=corner['refinements_source_px'][label]
    mixed=np.asarray(record['variants'][-1]['corners_source_px'],float)
    for i,point in selected.items():mixed[i]=point
    # The complete coordinate identity grid remains inspectable and valid even
    # though rejected corners are excluded from the pose/scoring subsets.
    topology(mixed,pattern,record['image_size_px'])
    record['review_state']='reviewed';record['review']=review
    record['accepted_corner_indices']=ids
    record['reviewed_corners_source_px']={str(i):point for i,point in selected.items()}
    record['scoring_coverage']=scoring_coverage(record,pattern,config)
    record['eligible_for_frozen_scoring']=record['scoring_coverage']['eligible']
    record['eligibility_reasons']=record['scoring_coverage']['rejection_reasons']


def scoring_coverage(record, pattern, config):
    """Require spatially distributed, reviewed observations in both parities."""
    raw = np.asarray([c['raw_source_px'] for c in record['corners']],float)
    hull = float(cv2.contourArea(cv2.convexHull(raw.astype(np.float32))))
    accepted = record.get('accepted_corner_indices', [])
    reasons=[]; parities=[]
    for parity in (0,1):
        ids=[i for i in accepted if sum(divmod(i,pattern[0]))%2==parity]
        fraction=0. if len(ids)<3 or hull<=0 else float(cv2.contourArea(cv2.convexHull(raw[ids].astype(np.float32))))/hull
        if len(ids)<config.minimum_parity_corners:reasons.append(f'parity_{parity}_insufficient_reviewed_corners')
        if fraction<config.minimum_parity_hull_fraction:reasons.append(f'parity_{parity}_insufficient_spatial_coverage')
        spans=[0.,0.] if not ids else (np.ptp(raw[ids],axis=0)/np.maximum(np.ptp(raw,axis=0),np.finfo(float).eps)).tolist()
        if min(spans)<config.minimum_axis_span_fraction:reasons.append(f'parity_{parity}_insufficient_axis_spread')
        mid=np.mean(np.array([[0,0],[pattern[0]-1,pattern[1]-1]]),axis=0)
        occupied=sorted(set((int(divmod(i,pattern[0])[1]>=mid[0]),int(divmod(i,pattern[0])[0]>=mid[1])) for i in ids))
        parities.append({'parity':parity,'count':len(ids),'hull_fraction_of_raw_grid':fraction,
                         'source_axis_span_fractions':spans,'occupied_board_quadrants':[list(v) for v in occupied]})
    return {'eligible':not reasons,'rejection_reasons':reasons,'parities':parities}


def training_membership(observations, frozen, view):
    """Distinguish fit reuse using target/session identity, never view ID alone."""
    training_ids=set(frozen.get('view_ids',[]))
    hashes={v['sha256'] for v in frozen.get('provenance',{}).get('source_images',[])
            if not training_ids or v.get('id') in training_ids}
    if hashes:
        return ('training_reuse' if view['source_sha256'] in hashes else 'not_in_frozen_intrinsic_fit','source_sha256')
    source=frozen.get('settings',{}).get('session_dir')
    if source and Path(source).name==Path(observations.get('source_session','')).name and training_ids:
        return ('training_reuse' if view['view_id'] in training_ids else 'not_in_frozen_intrinsic_fit',
                'recorded_session_identity_and_training_view_ids')
    return ('unknown','no_source_bound_training_ledger')


def score_frozen(observations, frozen):
    """Score a frozen K/D using reviewed, supported parity masks only.

    No calibrateCamera call is present. Intrinsics are not fitted or selected;
    each parity estimates only board pose, then scores the other parity.
    """
    if observations['image_size_px'] != frozen['image_size_px']:
        raise ValueError('Frozen model source dimensions differ; explicit qualified transfer required')
    cols,rows=observations['pattern_inner_corners']
    obj=np.zeros((cols*rows,3),np.float32)
    obj[:,:2]=np.mgrid[:cols,:rows].T.reshape(-1,2)*observations['nominal_square_pitch_mm']
    models={}
    for name,model in frozen['models'].items():
        fit=model.get('full_fit')
        if fit is None:continue
        k=np.asarray(fit['camera_matrix'],float);d=np.asarray(fit['distortion_k1_k2_p1_p2_k3'],float)
        perview=[]
        for view in observations['views']:
            membership,basis=training_membership(observations,frozen,view)
            identity={'source_sha256':view['source_sha256'],'source_session':observations.get('source_session'),
                      'intrinsic_training_membership':membership,'membership_basis':basis,
                      'independent_physical_accuracy_validation':False}
            if not view['eligible_for_frozen_scoring']:
                perview.append({'view_id':view['view_id'],'status':'unavailable','reasons':view['eligibility_reasons'],**identity});continue
            try:
                points=np.asarray(view['variants'][-1]['corners_source_px'],np.float32).copy()
                for i,xy in view['reviewed_corners_source_px'].items():points[int(i)]=xy
                ids=np.asarray(view['accepted_corner_indices'],int)
                parity=np.array([sum(divmod(i,cols))%2 for i in ids])
                residuals=[]
                for side in (0,1):
                    train=ids[parity==side]; test=ids[parity!=side]
                    solution=cv2.solvePnPGeneric(obj[train],points[train],k,d,flags=cv2.SOLVEPNP_IPPE)
                    candidates=[]
                    for r,t in zip(solution[1],solution[2]):
                        if np.min((obj @ cv2.Rodrigues(r)[0].T+t.reshape(1,3))[:,2])<=0:continue
                        r,t=cv2.solvePnPRefineLM(obj[train],points[train],k,d,r,t)
                        if np.min((obj @ cv2.Rodrigues(r)[0].T+t.reshape(1,3))[:,2])<=0:continue
                        predicted=cv2.projectPoints(obj[train],r,t,k,d)[0].reshape(-1,2)
                        candidates.append((float(np.sum((predicted-points[train])**2)),r,t))
                    if not candidates:raise ValueError('No positive-depth board pose for reviewed observation')
                    _,r,t=min(candidates,key=lambda c:c[0])
                    predicted=cv2.projectPoints(obj[test],r,t,k,d)[0].reshape(-1,2)
                    residuals += [{'corner_index':int(i),'error_px':float(np.linalg.norm(p-q)),
                                   'pose_training_parity':side,'observed_source_px':q.tolist(),
                                   'predicted_source_px':p.tolist()} for i,p,q in zip(test,predicted,points[test])]
                errors=np.asarray([r['error_px'] for r in residuals])
                perview.append({'view_id':view['view_id'],'status':'scored',**identity,'corner_count':len(errors),
                                'rms_px':float(np.sqrt(np.mean(errors**2))),
                                'max_px':float(np.max(errors)),'residuals':residuals,
                                'excluded_corner_indices':sorted(set(range(len(obj)))-set(ids.tolist()))})
            except (cv2.error,ValueError,FloatingPointError) as error:
                perview.append({'view_id':view['view_id'],'status':'unavailable',**identity,
                                'reasons':['board_pose_failed'], 'detail':str(error)})
        models[name]={'frozen_camera_matrix':k.tolist(),'frozen_distortion':d.tolist(),'views':perview}
    if not models:raise ValueError('Frozen asset has no full-fit models')
    return {'schema_version':1,'purpose':'Frozen-model parity scoring on explicitly reviewed source observations; no intrinsics fit',
            'intrinsics_fitted':False,'models':models}


def synthetic_image(pitch=40, blur=0):
    image=np.full((400,520),255,np.uint8)
    for row in range(7):
        for col in range(10):
            image[40+row*pitch:40+(row+1)*pitch,40+col*pitch:40+(col+1)*pitch]=255*((row+col)%2)
    if blur:image=cv2.GaussianBlur(image,(0,0),blur)
    coords=np.array([[40+(col+1)*pitch-.5,40+(row+1)*pitch-.5]
                     for row in range(6) for col in range(9)],float)
    return image,coords


def self_test():
    config=Config();pattern=(9,6);image,coords=synthetic_image()
    result=validate_observation(image,coords,pattern,config)
    assert result['machine_supported_view']
    assert np.max(np.linalg.norm(np.asarray(result['variants'][-1]['corners_source_px'])-coords,axis=1))<.05
    displaced=coords.copy();displaced[20]+=[8,0]
    changed=validate_observation(image,displaced,pattern,config)
    assert 'neighborhood_disagreement' in changed['corners'][20]['rejection_reasons']
    blurred,bc=synthetic_image(blur=2)
    br=validate_observation(blurred,bc,pattern,config)
    assert br['machine_supported_view']
    missing=image.copy();x,y=np.rint(coords[20]).astype(int);missing[y-15:y+16,x-15:x+16]=255
    mr=validate_observation(missing,coords,pattern,config)
    assert not mr['corners'][20]['machine_supported']
    jump=coords.copy();jump[20]+=[35,0]
    try:
        jr=validate_observation(image,jump,pattern,config)
        assert not jr['corners'][20]['machine_supported']
    except ValueError:pass
    reverse=coords.copy();reverse[[20,21]]=reverse[[21,20]]
    try:topology(reverse,pattern,[520,400])
    except ValueError:pass
    else:raise AssertionError('Topology swap accepted')
    record={'view_id':'synthetic','source_sha256':'synthetic-source',
            'image_size_px':[520,400],**result}
    review={'source_sha256':'synthetic-source','reviewer':'synthetic-test',
            'reviewed_at_utc':'synthetic-test','observation_refiner_revision':REVISION,
            'configuration_sha256':digest(json.dumps(asdict(config),sort_keys=True).encode()),
            'accepted_corners':[{'index':i,'variant_label':'pitch-0.3'} for i in range(54)]}
    apply_review(record,review,pattern,config)
    assert record['eligible_for_frozen_scoring']
    missing_review=dict(review,accepted_corners=review['accepted_corners'][:8])
    apply_review(record,missing_review,pattern,config)
    assert not record['eligible_for_frozen_scoring']
    for key,value in [('source_sha256','wrong'),('configuration_sha256','wrong'),('observation_refiner_revision',999)]:
        wrong=dict(review,**{key:value})
        try:apply_review(record,wrong,pattern,config)
        except ValueError:pass
        else:raise AssertionError('Stale/mismatched review accepted')
    unsupported={'view_id':'missing','source_sha256':'synthetic-source','image_size_px':[520,400],**mr}
    try:apply_review(unsupported,review,pattern,config)
    except ValueError:pass
    else:raise AssertionError('Review overrode missing image evidence')
    apply_review(record,review,pattern,config)
    frozen={'image_size_px':[520,400],'models':{'known':{'full_fit':{
        'camera_matrix':[[500.,0,260.],[0,500.,200.],[0,0,1]],
        'distortion_k1_k2_p1_p2_k3':[0.,0.,0.,0.,0.]}}}}
    observations={'image_size_px':[520,400],'pattern_inner_corners':[9,6],
                  'nominal_square_pitch_mm':40.,'views':[record]}
    original_calibrate=cv2.calibrateCamera
    original_extended=cv2.calibrateCameraExtended
    def forbidden(*args,**kwargs):raise AssertionError('Frozen scoring refitted intrinsics')
    cv2.calibrateCamera=forbidden
    cv2.calibrateCameraExtended=forbidden
    try:scored=score_frozen(observations,frozen)
    finally:
        cv2.calibrateCamera=original_calibrate
        cv2.calibrateCameraExtended=original_extended
    score=scored['models']['known']['views'][0]
    assert score['corner_count']==54 and score['rms_px']<.05
    assert len({r['corner_index'] for r in score['residuals']})==54
    assert all(sum(divmod(r['corner_index'],9))%2 != r['pose_training_parity'] for r in score['residuals'])
    try:score_frozen(dict(observations,image_size_px=[1280,720]),frozen)
    except ValueError:pass
    else:raise AssertionError('Mismatched image-size model accepted')
    collision_frozen={'view_ids':['view-001'],'settings':{'session_dir':'old-board'},
                      'provenance':{'source_images':[{'id':'view-001','sha256':'old-hash'}]}}
    membership,basis=training_membership({'source_session':'new-board'},collision_frozen,
                                         {'view_id':'view-001','source_sha256':'new-hash'})
    assert membership=='not_in_frozen_intrinsic_fit' and basis=='source_sha256'
    old_ledger={'view_ids':['view-001'],'settings':{'session_dir':'old-board'}}
    assert training_membership({'source_session':'old-board'},old_ledger,
                               {'view_id':'view-001','source_sha256':'old-hash'})[0]=='training_reuse'
    assert training_membership({'source_session':'old-board'},old_ledger,
                               {'view_id':'view-009','source_sha256':'other-hash'})[0]=='not_in_frozen_intrinsic_fit'
    assert training_membership({'source_session':'new-board'},old_ledger,
                               {'view_id':'view-001','source_sha256':'new-hash'})[0]=='unknown'
    print('Known intersections, >5px displaced start, blur, missing intersection, neighbor attraction, topology, source/config-bound reviews, coverage, frozen parity scoring and training membership checks passed')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session',type=Path,action='append',help='Read-only saved session; repeat for both targets')
    parser.add_argument('--output',type=Path,default=CAL/'validated-observations-20261028')
    parser.add_argument('--config',type=Path,help='JSON overrides for declared Config fields')
    parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--reviews',type=Path,help='Source-bound reviewer records; hard failures cannot be overridden')
    parser.add_argument('--frozen-models',type=Path,help='Score reviewed observations only, with matching-size frozen model JSON')
    args=parser.parse_args()
    cv2.setNumThreads(1);cv2.setRNGSeed(0)
    if args.self_test:self_test();return
    values=json.loads(args.config.read_text()) if args.config else {}
    try:config=Config(**values)
    except TypeError as error:parser.error(str(error))
    if (config.maximum_iterations<1 or config.convergence_epsilon_px<=0 or
        not config.window_fractions or any(not 0<f<.5 for f in config.window_fractions) or
        config.minimum_parity_corners<4 or not 0<config.minimum_parity_hull_fraction<=1 or
        not 0<config.minimum_axis_span_fraction<=1 or config.display_minimum_crop_side_px<1 or
        config.baseline_half_window<1 or config.support_patch_fraction<=0 or config.crop_half_pitch_fraction<=0 or
        any(a>=b for a,b in zip(config.window_fractions,config.window_fractions[1:])) or
        config.agreement_reference_px<=0 or config.pixel_reference_width<1 or
        not 0<config.maximum_identity_shift_fraction<.5 or config.contact_sheet_columns<1 or
        not config.support_radii_fractions or any(not 0<f<.5 for f in config.support_radii_fractions) or
        config.minimum_alternating_contrast_gray<=0 or not 0<config.minimum_alternation_consistency<=1):
        parser.error('Invalid observation-validation configuration')
    reviews=json.loads(args.reviews.read_text())['reviews'] if args.reviews else []
    review_keys=[(r['session'],r['view_id']) for r in reviews]
    if len(set(review_keys))!=len(review_keys):parser.error('Duplicate reviewer records')
    frozen=json.loads(args.frozen_models.read_text()) if args.frozen_models else None
    sessions=args.session or [CAL/'checkerboard-20x14-session',CAL/'checkerboard-session']
    summaries=[]
    for session in sessions:
        session=session.resolve();result=make_session(session,args.output/session.name,config,reviews)
        if frozen is not None:
            scored=score_frozen(result,frozen)
            scored['frozen_models_sha256']=digest(args.frozen_models.read_bytes())
            scored['observations_sha256']=digest((args.output/session.name/'observations.json').read_bytes())
            (args.output/session.name/'frozen-scores.json').write_text(json.dumps(scored,indent=2,allow_nan=False)+'\n')
        summaries.append({'session':session.name,'views':len(result['views']),
                          'machine_supported_views':sum(v['machine_supported_view'] for v in result['views']),
                          'eligible_views':result['eligible_view_count']})
    print(json.dumps(summaries))
    # A successful export is not permission to score unreviewed observations.
    # An explicit nonzero exit makes accidental pipeline continuation fail closed.
    if not any(s['eligible_views'] for s in summaries):
        print('No eligible scoring views: inspect per-view review/support/coverage reasons.',file=sys.stderr)
        raise SystemExit(2)


if __name__=='__main__':main()
