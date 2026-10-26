"""Offline local image-coordinate normalization; no depth/height fit or production change.
Run /tmp/billiards-checkerboard-env/bin/python thoughts/calibration/ball-local-view-normalization.py.
"""
from pathlib import Path
import importlib.util
import json
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('outline_analysis',ROOT/'analyze_ball_outline_consistency.py')
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
K=np.array([[696.17,0,644.18],[0,695.36,349.67],[0,0,1.]])
KI=np.linalg.inv(K);KP=np.diag([696.,696.,1.])


def warp(points,h):
    p=np.column_stack((points,np.ones(len(points))))@h.T
    return p[:,:2]/p[:,2,None]


def homography(center):
    d=KI@np.r_[center,1.];d/=np.linalg.norm(d)
    # Rodrigues maps source center ray onto virtual optical axis. This rotates
    # image rays only; it does not estimate a sphere depth or physical Z value.
    z=np.array([0.,0.,1.]);v=np.cross(d,z);c=d@z
    skew=np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0.]])
    R=np.eye(3)+skew+skew@skew/(1+c)
    assert np.linalg.norm(R@d-z)<1e-10
    return KP@R@KI


def circle(points):
    center0=points.mean(0);q=points-center0
    design=np.column_stack((2*q[:,0],2*q[:,1],np.ones(len(q))))
    sol=np.linalg.lstsq(design,np.sum(q*q,axis=1),rcond=None)[0]
    center=center0+sol[:2];radius=np.median(np.linalg.norm(points-center,axis=1))
    for _ in range(15):
        delta=points-center;dist=np.maximum(np.linalg.norm(delta,axis=1),1e-9)
        residual=dist-radius
        scale=max(np.median(np.abs(residual-np.median(residual)))*1.4826,.15)
        weights=1/np.sqrt(1+(residual/(2.5*scale))**2)
        jac=np.column_stack((-delta[:,0]/dist,-delta[:,1]/dist,-np.ones(len(points))))
        step=np.linalg.lstsq(jac*weights[:,None],-residual*weights,rcond=None)[0]
        center+=step[:2];radius+=step[2]
        if np.linalg.norm(step)<1e-8:break
    return center,float(radius),float(np.sqrt(np.mean((np.linalg.norm(points-center,axis=1)-radius)**2)))


def normalize(points,seed):
    axis=np.array(seed,dtype=float);history=[]
    for _ in range(3):
        H=homography(axis);local=warp(points,H);center,radius,rms=circle(local)
        recovered=warp(center[None,:],np.linalg.inv(H))[0]
        history.append(float(np.linalg.norm(recovered-axis)));axis=recovered
    # Reevaluate in final aimed patch, retaining circle origin for rendering.
    H=homography(axis);local=warp(points,H);center,radius,rms=circle(local)
    source=warp(center[None,:],np.linalg.inv(H))[0]
    assert np.max(np.abs(warp(warp(points,H),np.linalg.inv(H))-points))<1e-8
    e=cv2.fitEllipse(local.astype(np.float32))
    return {'source_center_px':source.tolist(),'local_circle_radius_px':radius,'local_circle_rms_px':rms,'local_ellipse_axis_ratio':float(max(e[1])/min(e[1])),'iteration_center_shifts_px':history},H,center,radius


def synthetic():
    """Generate exact tangent-cone contours solely to provide known test centers.
    Arbitrary scene range and radius exist only in this generator. The tested
    estimator receives 2D contour pixels and K; it never receives that range.
    """
    rng=np.random.default_rng(931);out=[]
    for pixel in ([644.18,349.67],[95,362],[1198,364],[105,618]):
        d=KI@np.r_[pixel,1.];d/=np.linalg.norm(d)
        ref=np.array([0.,1.,0.]);e=np.cross(d,ref);e/=np.linalg.norm(e);f=np.cross(d,e)
        alpha=np.arcsin(2./100.) # arbitrary sphere radius/range in synthetic generator only
        theta=np.linspace(0,2*np.pi,180,endpoint=False)
        rays=np.cos(alpha)*d+np.sin(alpha)*(np.cos(theta)[:,None]*e+np.sin(theta)[:,None]*f)
        p=rays@K.T;clean=p[:,:2]/p[:,2,None]
        raw=cv2.fitEllipse(clean.astype(np.float32))
        for noise in (0.,.3):
            noisy=clean+rng.normal(0,noise,clean.shape)
            fit,_,_,_=normalize(noisy,pixel)
            error=float(np.linalg.norm(np.array(fit['source_center_px'])-pixel))
            if noise==0:assert error<1e-5 and abs(fit['local_ellipse_axis_ratio']-1)<1e-5
            else:assert error<.15
            out.append({'known_center_px':pixel,'noise_sigma_px':noise,'source_aspect':float(max(raw[1])/min(raw[1])),'normalized_aspect':fit['local_ellipse_axis_ratio'],'recovered_center_error_px':error})
    return out


def main():
    image=cv2.imread(str(a.IMAGE));measurements=json.loads(a.MEASUREMENTS.read_text());truth={p['id']:p for p in measurements['points']}
    previous=json.loads((ROOT/'ball-outline-consistency.json').read_text());prev={p['id']:p for p in previous['balls']}
    annotation=image.copy();records=[]
    for ident,seed in a.SEEDS:
        se,bbox,contour,hull=a.segmentation(image,seed,.11)
        grad,gpoints,inliers,coverage,arc=a.robust_gradient_fit(image,seed)
        rec={'id':ident,'physical_ground_truth_available':ident in truth,'source_segmentation_aspect':float(max(se[1])/min(se[1])),'source_gradient_aspect':float(max(grad[1])/min(grad[1])),'review_flags':prev[ident]['localization_review_flags']}
        for method,points,ellipse,color in [('segmentation',hull,se,(255,255,0)),('gradient',gpoints[inliers],grad,(0,180,255))]:
            result,H,c,r=normalize(points,seed)
            result['source_center_vs_free_ellipse_offset_px']=float(np.linalg.norm(np.array(result['source_center_px'])-ellipse[0]))
            result['source_free_ellipse_center_px']=list(ellipse[0]);rec[method]=result
            t=np.linspace(0,2*np.pi,100);circ=c+r*np.column_stack((np.cos(t),np.sin(t)))
            boundary=warp(circ,np.linalg.inv(H));cv2.polylines(annotation,[np.round(boundary).astype(int)],True,color,1,cv2.LINE_AA)
            cv2.drawMarker(annotation,tuple(np.round(result['source_center_px']).astype(int)),color,cv2.MARKER_CROSS,5,1)
        cv2.putText(annotation,str(len(records)+1),(seed[0]+17,seed[1]-12),cv2.FONT_HERSHEY_SIMPLEX,.35,(255,255,255),1,cv2.LINE_AA);records.append(rec)
    known=[r for r in records if r['physical_ground_truth_available']];dst=np.array([[truth[r['id']]['center_from_upper_left_mm'][axis] for axis in ('x','y')] for r in known]);metrics={}
    for method in ('segmentation','gradient'):
        for local in (False,True):
            key=method+('_local_circle' if local else '_free_ellipse');field='source_center_px' if local else 'source_free_ellipse_center_px'
            src=np.array([r[method][field] for r in known]);H=a.fit_h(src,dst);err=np.linalg.norm(a.project(src,H)-dst,axis=1);held=[]
            for idx in range(len(src)):
                keep=np.arange(len(src))!=idx;held.append(float(np.linalg.norm(a.project(src[idx:idx+1],a.fit_h(src[keep],dst[keep]))[0]-dst[idx])))
            metrics[key]={'training_rms_mm':float(np.sqrt(np.mean(err**2))),'leave_one_position_out_rms_mm':float(np.sqrt(np.mean(np.array(held)**2))),'held_out_errors_mm':dict(zip([r['id'] for r in known],held))}
    tests=synthetic();output={'K_frozen_zero_radial':K.tolist(),'physical_diameter_mm':57.15,'scope':'local 2D point homography only; no ball depth/height fit, no variable physical radius','method_note':'Gradient points are preselected by previous free-ellipse consensus; comparisons isolate circle constraint rather than independent edge selection. Segment convex hull can bridge invalid arcs. Local radius is image pixels only.','balls':records,'nine_point_diagnostics':metrics,'synthetic_exact_tangent_cone_tests':tests,'source_image':str(a.IMAGE),'production_changes':False}
    (ROOT/'ball-local-view-normalization.json').write_text(json.dumps(output,indent=2)+'\n');cv2.imwrite(str(ROOT/'ball-local-view-normalization.png'),annotation)
    # Real pixel montage: same 65x65 native canvas and identical 3x enlargement.
    # Homography rotates source image rays; no circular mask, per-ball scaling,
    # or radius-based resizing is applied. OpenCV inverse-samples source pixels.
    selected=['far-left-yellow-middle-row-ball','middle-row-center-ball','far-right-purple-middle-row-ball']
    montage=np.full((3*250,2*235,3),245,np.uint8)
    for row,ident in enumerate(selected):
        rec=next(r for r in records if r['id']==ident)
        aim=np.array(rec['segmentation']['source_center_px'])
        native=65;half=32
        # Source crop stays centered at the same inferred aim direction.
        source_map=np.array([[1.,0.,half-aim[0]],[0.,1.,half-aim[1]],[0.,0.,1.]])
        target_map=np.array([[1.,0.,half],[0.,1.,half],[0.,0.,1.]])@homography(aim)
        for col,mapping in enumerate((source_map,target_map)):
            patch=cv2.warpPerspective(image,mapping,(native,native),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=(0,0,0))
            enlarged=cv2.resize(patch,(195,195),interpolation=cv2.INTER_NEAREST)
            x=col*235+20;y=row*250+44;montage[y:y+195,x:x+195]=enlarged
            title=('Original source' if col==0 else 'Local virtual view')
            label=('Left yellow' if row==0 else 'Center control' if row==1 else 'Right purple')
            cv2.putText(montage,label+' - '+title,(col*235+7,row*250+16),cv2.FONT_HERSHEY_SIMPLEX,.35,(20,20,20),1,cv2.LINE_AA)
            cv2.putText(montage,'65 px patch; enlarged 3x',(col*235+12,row*250+33),cv2.FONT_HERSHEY_SIMPLEX,.35,(20,20,20),1,cv2.LINE_AA)
            # Validate OpenCV's transformed pixel against explicit inverse-map
            # bilinear sampling at the central output point (within rounding).
            source_point=warp(np.array([[half,half]],float),np.linalg.inv(mapping))[0]
            sample=cv2.getRectSubPix(image,(1,1),tuple(map(float,source_point)))[0,0]
            assert np.max(np.abs(patch[half,half].astype(int)-sample.astype(int)))<=3
    cv2.imwrite(str(ROOT/'ball-local-view-normalization-crops.png'),montage)

    lines=['# Local-view circle normalization - offline test','','The fixed physical ball diameter remains 57.15 mm. This experiment changes only local 2D image coordinates. No sphere depth/height is estimated, no application Z output is produced, and no correction is activated.','','## Method','','Frozen zero-radial intrinsics: fx=696.17, fy=695.36, cx=644.18, cy=349.67. A Rodrigues rotation maps the coarse ball image ray onto a virtual optical axis; H=Kpatch R K^-1 maps contour pixels into a local virtual patch. A three-parameter robust circle is fitted and its center projected back. Three aiming iterations refine the patch direction. Pixel radii may differ between patches; physical diameter does not change. No full-frame image warp is performed.','','Segmented convex-envelope points and previously selected gradient/ellipse-consensus points are reused from the source-image analysis. This isolates fitting constraints; it does not repair bad edge selection or supply an independent center truth. White stripes, rail shadows, and bridged missing arcs remain problematic.','','## Source-image comparison','','| Ball | Raw segment aspect | Local segment aspect | Segment center shift px | Raw gradient aspect | Local gradient aspect | Gradient center shift px |','|---|---:|---:|---:|---:|---:|---:|']
    for r in records:lines.append(f"| {r['id']} | {r['source_segmentation_aspect']:.3f} | {r['segmentation']['local_ellipse_axis_ratio']:.3f} | {r['segmentation']['source_center_vs_free_ellipse_offset_px']:.3f} | {r['source_gradient_aspect']:.3f} | {r['gradient']['local_ellipse_axis_ratio']:.3f} | {r['gradient']['source_center_vs_free_ellipse_offset_px']:.3f} |")
    lines+=['','## Actual source-pixel patches','', '[Original versus local-view crop montage](ball-local-view-normalization-crops.png) shows left yellow, center red control, and right purple, each on the same 65 x 65 native-pixel canvas enlarged exactly 3x by nearest-neighbor for inspection. The local patch is inverse-sampled from source RGB through the camera rotation homography with the same 696 px virtual focal scale for every ball. There is no circular mask, diameter equalization, or per-ball resizing. Neighbor clutter and shadows remain visible. Center directions come from the fitted source observations and remain uncertain.','', '## Nine-point center diagnostics','','Each method is evaluated with its own simple H and leave-one-position-out refitting. These discrepancies include unknown physical measurement/contact error and are not verified mm accuracy. Pocket balls remain outside physical ground truth.','','| Fit | Training RMS mm | Held-out RMS mm |','|---|---:|---:|']
    for method,r in metrics.items():lines.append(f"| {method} | {r['training_rms_mm']:.3f} | {r['leave_one_position_out_rms_mm']:.3f} |")
    lines+=['','## Exact synthetic sphere-contour check','','Synthetic generation uses an arbitrary sphere radius/range only to construct exact tangent rays and known projected center. The tested estimator receives pixels and K only. At four image positions, noiseless local normalization circularizes the exact contour and recovers its known center within 0.00001 px. With independently seeded 0.3 px contour noise, errors remain below 0.15 px in these four tests. This verifies transformation/inversion and limited recovery, not real-image physical accuracy. JSON retains every test and shape ratio.','','## Limits and practical conclusion','','Clean off-axis ball outlines become closer to circular in local coordinates, consistent with ordinary viewing geometry. This supports an image-domain local circularity prior without changing physical diameter. The fitted-center differences and held-out results determine whether it actually improves localization here; stronger shape constraints cannot fix contours contaminated by stripe, shadow, cushion, or pocket edges. Near-center points should change little. Review the source-image cyan segmentation-constrained circle projections versus orange gradient-constrained projections.','','The top-left pocket segmentation remains a known highlight-fragment failure and is retained/flagged rather than promoted to a valid circle. Bad contour fits may normalize poorly or have large center shifts; successful circle fitting is not an acceptance criterion by itself. Corner/pocket physical accuracy remains unverified. No new measurement is requested. Local normalization is a bounded candidate for future robust ROI localization, not a reason to activate global radial lens warping.','','## Verification','','Exact synthetic tangent-contour recovery with and without pixel noise, finite transforms, image-homography inversion round trips, original source dimensions and JSON output are verified by the script. No production file, dependency, calibration state or existing report is changed.']
    (ROOT/'ball-local-view-normalization.md').write_text('\n'.join(lines)+'\n');print(json.dumps(metrics,indent=2));print('Synthetic tests passed:',len(tests))

if __name__=='__main__':main()
