#!/usr/bin/env python3
"""Offline straight-edge diagnostic, not a surveyed cushion or table calibration.

The source-pixel ROIs below were inspected visually. Pocket jaws, balls and the
view-009 checkerboard are omitted. Visible nose-associated transitions and the
stronger bottom seam are kept distinct rather than assuming they are one edge.
Only extracted points are undistorted; image pixels and runtime stay unchanged.
"""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


# Offline annotation defaults, configurable as a complete JSON ROI file.
# Axis x: scan along x, search y; axis y: scan y, search x. Signed gradient
# selects the visually inspected transition instead of the strongest hardware.
DEFAULT_ROIS = [
    dict(id='top-left', axis='x', spans=[[145, 315], [390, 575]], band=[75, 89], sign=-1, feature='nose-associated brightness transition'),
    dict(id='top-right', axis='x', spans=[[705, 875], [980, 1135]], band=[73, 88], sign=-1, feature='nose-associated brightness transition'),
    dict(id='bottom-left-nose', axis='x', spans=[[160, 320], [405, 570]], band=[615, 635], sign=-1, feature='weak inner cushion transition; not dark outer seam'),
    dict(id='bottom-right-nose', axis='x', spans=[[715, 880], [975, 1120]], band=[615, 637], sign=-1, feature='weak inner cushion transition; not dark outer seam'),
    dict(id='left-nose', axis='y', spans=[[145, 280], [455, 565]], band=[97, 111], sign=1, feature='inner shadow-to-cloth transition; board and middle-row ball excluded'),
    dict(id='right-nose', axis='y', spans=[[145, 295], [435, 565]], band=[1198, 1213], sign=-1, feature='cloth-to-inner-shadow transition; middle-row ball excluded'),
    dict(id='bottom-left-seam', axis='x', spans=[[160, 320], [405, 570]], band=[637, 649], sign=1, feature='outer dark seam rising edge; alternative reference, not nose'),
    dict(id='bottom-right-seam', axis='x', spans=[[715, 880], [975, 1120]], band=[637, 652], sign=1, feature='outer dark seam rising edge; alternative reference, not nose'),
]


def robust_poly(t, values, degree):
    design = np.vander(t, degree+1)
    weights = np.ones(len(t))
    for _ in range(20):
        coefficients = np.linalg.lstsq(design*np.sqrt(weights[:, None]), values*np.sqrt(weights), rcond=None)[0]
        residual = values-design@coefficients
        scale = max(1.4826*np.median(np.abs(residual-np.median(residual))), .03)
        weights = np.minimum(1, 1.5*scale/np.maximum(np.abs(residual), 1e-12))
    return coefficients, residual, weights


def line_metrics(points):
    center = points.mean(axis=0)
    _, _, vt = np.linalg.svd(points-center, full_matrices=False)
    direction = vt[0]
    t = (points-center)@direction
    normal = np.array([-direction[1], direction[0]])
    r = (points-center)@normal
    normalized = (t-(t.min()+t.max())/2)/max((t.max()-t.min())/2, 1)
    _, residual, weights = robust_poly(normalized, r, 1)
    quadratic, qr, _ = robust_poly(normalized, r, 2)
    return {'rms_px': float(np.sqrt(np.mean(residual**2))),
            'robust_weighted_rms_px': float(np.sqrt(np.average(residual**2, weights=weights))),
            'p95_abs_px': float(np.percentile(np.abs(residual), 95)),
            'quadratic_midpoint_chord_bow_px': float(abs(quadratic[0])),
            'quadratic_residual_rms_px': float(np.sqrt(np.mean(qr**2))),
            'points': len(points), 'span_px': float(np.ptp(t))}


def extract(gray, roi, sigma, band_shift, options):
    # Median along a short tangent strip suppresses texture without filling
    # gaps or connecting observations across board/ball exclusions.
    image = gray if roi['axis']=='x' else gray.T
    coords, strengths = [], []
    lo, hi = np.asarray(roi['band'])+band_shift
    for begin, end in roi['spans']:
        for t in range(begin, end, options.step):
            profile = np.median(image[:, max(0,t-options.tangent_halfwidth):t+options.tangent_halfwidth+1], axis=1)
            smooth = cv2.GaussianBlur(profile.reshape(-1,1), (1,0), sigmaY=sigma, sigmaX=0).ravel()
            gradient = np.gradient(smooth)*roi['sign']
            index = int(lo+np.argmax(gradient[lo:hi+1]))
            strength = float(gradient[index])
            if strength < options.minimum_gradient or index in (lo,hi):
                continue
            denom = gradient[index-1]-2*gradient[index]+gradient[index+1]
            delta = np.clip(.5*(gradient[index-1]-gradient[index+1])/denom, -.5, .5) if abs(denom)>1e-12 else 0
            xy = [float(t),float(index+delta)]
            coords.append(xy if roi['axis']=='x' else xy[::-1]); strengths.append(strength)
    return np.asarray(coords), strengths


def point_correct(points, model):
    k = np.asarray(model['camera_matrix'], dtype=float)
    d = np.asarray(model['distortion_k1_k2_p1_p2_k3'], dtype=float)
    return cv2.undistortPoints(points.astype(np.float64).reshape(-1,1,2), k, d,
                               P=k, criteria=(cv2.TERM_CRITERIA_COUNT|cv2.TERM_CRITERIA_EPS, 50, 1e-10)).reshape(-1,2)


def synthetic_test():
    k=np.array([[700.,0,640.],[0,700.,360.],[0,0,1.]])
    true_d=np.array([.10,0,0,0,0.])
    ideal=np.column_stack([np.linspace(120,1160,200),np.full(200,90.)])
    rays=np.column_stack([(ideal[:,0]-640)/700,(ideal[:,1]-360)/700,np.ones(200)])
    raw=cv2.projectPoints(rays,np.zeros(3),np.zeros(3),k,true_d)[0].reshape(-1,2)
    def model(d): return {'camera_matrix': k.tolist(),'distortion_k1_k2_p1_p2_k3': d.tolist()}
    raw_metric=line_metrics(raw)
    good=line_metrics(point_correct(raw,model(true_d)))
    wrong=line_metrics(point_correct(raw,model(-true_d)))
    assert raw_metric['quadratic_midpoint_chord_bow_px']>5
    assert good['rms_px']<1e-5 and wrong['rms_px']>raw_metric['rms_px']
    assert line_metrics(ideal)['rms_px']<1e-8
    # Extraction verifies a blurred, sloped transition with deterministic noise.
    y,x=np.indices((100,220));rng=np.random.default_rng(83)
    frame=(60+80*(y>40+.01*x)+rng.normal(0,.4,(100,220))).astype(np.float32)
    class O: step=2;tangent_halfwidth=2;minimum_gradient=2
    roi=dict(axis='x',spans=[[15,205]],band=[33,50],sign=1)
    pts,_=extract(frame,roi,.8,0,O())
    assert len(pts)>80 and line_metrics(pts)['rms_px']<.5
    print(json.dumps({'synthetic':'passed','raw_bow_px':raw_metric['quadratic_midpoint_chord_bow_px'],'corrected_rms_px':good['rms_px'],'wrong_rms_px':wrong['rms_px']}))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--images',nargs='+',default=['thoughts/calibration/checkerboard-session/view-009.jpg','thoughts/calibration/table-grid-20261026-114241.jpg'])
    p.add_argument('--frozen-models',default='thoughts/calibration/checkerboard-eight-view-baseline.json')
    p.add_argument('--output-prefix',default='thoughts/calibration/cushion-straightness')
    p.add_argument('--roi-json',help='Override visually annotated ROI list')
    p.add_argument('--step',type=int,default=2);p.add_argument('--tangent-halfwidth',type=int,default=2)
    p.add_argument('--minimum-gradient',type=float,default=2.0)
    p.add_argument('--sigmas',type=float,nargs='+',default=[.6,1.0,1.4])
    p.add_argument('--minimum-points',type=int,default=30)
    p.add_argument('--self-test',action='store_true')
    options=p.parse_args()
    if options.self_test: synthetic_test();return
    if options.step<1 or options.minimum_points<4 or min(options.sigmas)<=0: p.error('Invalid sampling settings')
    baseline_bytes=Path(options.frozen_models).read_bytes();baseline=json.loads(baseline_bytes)
    models={name:fit['full_fit'] for name,fit in baseline['models'].items()}
    rois=json.loads(Path(options.roi_json).read_text()) if options.roi_json else DEFAULT_ROIS
    results={'purpose':'offline visible-edge straightness diagnostics; no physical nose survey or activation',
             'frozen_models_sha256':hashlib.sha256(baseline_bytes).hexdigest(),'settings':vars(options),'rois':rois,'images':[]}
    panels=[]
    palette=[(0,255,255),(255,180,30),(60,200,255),(255,100,255),(100,255,100),(255,255,0),(0,150,255),(180,200,255)]
    for filename in options.images:
        image_bytes=Path(filename).read_bytes();image=cv2.imdecode(np.frombuffer(image_bytes,np.uint8),cv2.IMREAD_COLOR)
        if image is None or tuple(image.shape[1::-1])!=tuple(baseline['image_size_px']):raise ValueError('Source image geometry differs from model')
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY).astype(np.float32)
        record={'source':filename,'source_sha256':hashlib.sha256(image_bytes).hexdigest(),'features':[]}
        panel=image.copy()
        cv2.putText(panel,Path(filename).name+' | raw observed points; no image warp',(120,25),0,.55,(255,255,255),1,cv2.LINE_AA)
        for roi,color in zip(rois,palette):
            feature={'id':roi['id'],'feature':roi['feature'],'variants':[]}
            for sigma in options.sigmas:
                for shift in [-2,0,2]:
                    points,strength=extract(gray,roi,sigma,shift,options)
                    if len(points)<options.minimum_points:
                        feature['variants'].append({'sigma':sigma,'band_shift':shift,'error':'insufficient_points','points':len(points)});continue
                    variant={'sigma':sigma,'band_shift':shift,'points_raw_px':points.tolist(),'median_gradient':float(np.median(strength)),
                             'models':{'raw':line_metrics(points)}}
                    for name,fit in models.items():
                        corrected=point_correct(points,fit)
                        if not np.isfinite(corrected).all():raise ValueError('Nonfinite candidate correction')
                        variant['models'][name]=line_metrics(corrected)
                    feature['variants'].append(variant)
                    if sigma==options.sigmas[len(options.sigmas)//2] and shift==0:
                        for point in points:cv2.circle(panel,tuple(np.rint(point).astype(int)),1,color,-1)
                        anchor=tuple(np.rint(points[len(points)//2]).astype(int))
                        cv2.putText(panel,roi['id'],(anchor[0]+4,anchor[1]-6),0,.38,color,1,cv2.LINE_AA)
            valid=[v for v in feature['variants'] if 'models'in v]
            if valid:
                feature['sensitivity_ranges']={name:{metric:[min(v['models'][name][metric] for v in valid),max(v['models'][name][metric] for v in valid)] for metric in ['rms_px','quadratic_midpoint_chord_bow_px']} for name in ['raw',*models]}
                expected=sum(len(range(a,b,options.step)) for a,b in roi['spans'])
                feature['sample_coverage_range']=[min(v['models']['raw']['points']/expected for v in valid),max(v['models']['raw']['points']/expected for v in valid)]
                feature['interpretation_warnings']=[]
                if feature['sample_coverage_range'][0]<.65:
                    feature['interpretation_warnings'].append('Weak edge: fewer than65% of planned samples survive in some variants')
                raw_range=feature['sensitivity_ranges']['raw']['rms_px']
                if raw_range[1]-raw_range[0]>.5 or raw_range[1]>1:
                    feature['interpretation_warnings'].append('Ambiguous/local edge jumps: do not use this feature alone to choose a lens model')
            record['features'].append(feature)
        results['images'].append(record);panels.append(panel)
    prefix=Path(options.output_prefix);prefix.parent.mkdir(parents=True,exist_ok=True)
    results['limitations']=['Observed image edges may trace shadow, cloth texture or a seam instead of the exact physical nose; their straightness is not a physical survey.',
        'Search-band and smoothing variants measure extraction sensitivity, not physical confidence intervals; these variants are correlated.',
        'Long segments split around pockets and balls; separate rail sections are not assumed collinear. The left-board region is excluded.',
        'Point-undistortion comparison reuses exactly the same extracted raw samples for every candidate; no refitting lens coefficients or image warping.',
        'Camera mode matching is assumed for these two1280x720 images; sources are hashed. No millimeter ball-center accuracy follows.',
        'Small residuals do not prove zero lens distortion; larger smooth bow can reflect extraction bias or physical edge geometry.']
    prefix.with_suffix('.json').write_text(json.dumps(results,indent=2,allow_nan=False)+'\n')
    lines=['# Cushion-associated edge straightness', '', 'Offline diagnostic only. Frozen eight-view lens candidates are compared by transforming points; source images, runtime settings and calibration stay unchanged.', '', 'ROIs were inspected on original images and enlarged rail strips. The bottom inner transition and stronger outer seam are reported separately. Pockets, balls and the left-middle checkerboard are excluded. Colored pixels in the PNG show the nominal extracted samples on unwarped images.']
    for record in results['images']:
        lines+=['',f"## {Path(record['source']).name}",'','Ranges below span three smoothing scales and search bands shifted ±2px. They quantify extraction sensitivity, not confidence bounds.','','| Feature | Raw RMS px | Raw smooth bow px | k1 RMS px | k1 smooth bow px | k1+k2 RMS px |','|---|---:|---:|---:|---:|---:|']
        for f in record['features']:
            if 'sensitivity_ranges'not in f:lines.append(f"| {f['id']} | insufficient edge support | | | | |");continue
            r=f['sensitivity_ranges'];fmt=lambda name,metric:'–'.join(f'{x:.2f}'for x in r[name][metric])
            lines.append(f"| {f['id']} | {fmt('raw','rms_px')} | {fmt('raw','quadratic_midpoint_chord_bow_px')} | {fmt('k1','rms_px')} | {fmt('k1','quadratic_midpoint_chord_bow_px')} | {fmt('k1k2','rms_px')} |")
        for f in record['features']:
            for warning in f.get('interpretation_warnings',[]):
                lines.append(f"\n- {f['id']}: {warning}.")
    lines+=['','## Interpretation and limits','','The raw-versus-candidate differences are a model-adequacy diagnostic. If a candidate consistently increases smooth bow on independently observed rail transitions, that argues against applying that candidate; it does not identify the cause of ball elongation. Nose and seam disagreement or unstable extraction calls for reviewing the annotated evidence, not selecting whichever edge favors a lens model.','']+['- '+x for x in results['limitations']]
    lines+=['','In these two images, both top rail sections, the left inner transition and the independent bottom outer seams become more bowed under the frozen positive-k1 candidate; k1+k2 behaves similarly. This is complementary evidence against applying these particular frozen radial corrections. The right-side extraction is ambiguous and the view-009 bottom-left nose transition has weak support, so neither is treated as decisive. The zero-distortion candidate leaves points unchanged; its metrics match raw coordinates to numerical precision.','', 'Keep production geometry unchanged. Coordinate calibration and actual ball-outline refinement remain separate unresolved work; these edge diagnostics do not prove zero residual lens distortion, physical cushion straightness or accurate ball X/Y.']
    lines+=['','## Reproduce','','```sh','/tmp/billiards-checkerboard-env/bin/python thoughts/calibration/analyze_cushion_straightness.py --self-test','/tmp/billiards-checkerboard-env/bin/python thoughts/calibration/analyze_cushion_straightness.py','```','','Synthetic tests cover a true straight line, known radial distortion, successful point correction, worsened straightness under wrong correction, and noisy sloped-edge extraction.']
    prefix.with_suffix('.md').write_text('\n'.join(lines)+'\n')
    cv2.imwrite(str(prefix.with_suffix('.png')),np.vstack(panels))
    print(json.dumps({'report':str(prefix.with_suffix('.md')),'images':len(panels)}))


if __name__=='__main__':main()
