"""Offline source-RGB ball contour diagnostics; no production changes or Z model.
Run with /tmp/billiards-checkerboard-env/bin/python. Outputs only ball-outline-consistency.*.
"""
from pathlib import Path
import json
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parent
IMAGE=ROOT/'table-grid-20261026-114241.jpg'
MEASUREMENTS=ROOT/'table-grid-20261026-114241-measurements.json'
SEEDS=[('top-left-pocket-ball',(102,105)),('top-orange-solid-second-left-ball',(352,97)),('top-middle-pocket-ball',(642,97)),('top-red-striped-second-right-ball',(928,94)),('top-right-cue-pocket-ball',(1196,97)),('far-left-yellow-middle-row-ball',(96,362)),('interior-purple-second-left-middle-row-ball',(359,362)),('middle-row-center-ball',(643,361)),('interior-green-second-right-middle-row-ball',(925,361)),('far-right-purple-middle-row-ball',(1198,364)),('bottom-left-pocket-ball',(105,618)),('bottom-black-second-left-ball',(363,627)),('bottom-middle-pocket-ball',(641,630)),('bottom-yellow-second-right-ball',(925,630)),('bottom-right-pocket-ball',(1195,628))]


def ellipse_parameters(e):
    (x,y),(a,b),angle=e
    return {'center_px':[float(x),float(y)],'diameters_px':[float(a),float(b)],'major_axis_angle_deg':float((angle+(90 if b>a else 0))%180),'axis_ratio':float(max(a,b)/min(a,b))}


def residual(points,e):
    (x,y),(a,b),angle=e
    t=np.deg2rad(angle);rot=np.array([[np.cos(t),np.sin(t)],[-np.sin(t),np.cos(t)]])
    q=(points-[x,y])@rot.T
    rho=np.sqrt((q[:,0]/(a/2))**2+(q[:,1]/(b/2))**2)
    return np.abs(rho-1)*min(a,b)/2


def segmentation(image,seed,threshold):
    x,y=seed;r=25;x0=x-r;y0=y-r;roi=image[y0:y+r+1,x0:x+r+1].astype(float)
    yy,xx=np.indices(roi.shape[:2]);distance=np.hypot(xx-r,yy-r)
    cloth=np.median(roi[(distance>21)&(distance<25)],axis=0)
    chroma=roi/np.maximum(roi.sum(2,keepdims=True),1);clothchroma=cloth/cloth.sum()
    mask=(((np.linalg.norm(chroma-clothchroma,axis=2)>threshold)|(roi.sum(2)<cloth.sum()*.47))&(distance<23)).astype(np.uint8)
    count,labels,stats,centroids=cv2.connectedComponentsWithStats(mask,connectivity=8)
    candidates=[k for k in range(1,count) if stats[k,cv2.CC_STAT_AREA]>35]
    if not candidates:raise ValueError('no segmented component')
    k=min(candidates,key=lambda k:np.linalg.norm(centroids[k]-[r,r]));mask=(labels==k).astype(np.uint8)
    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    contour=max(contours,key=cv2.contourArea).reshape(-1,2)+[x0,y0]
    hull=cv2.convexHull(contour.astype(np.float32)).reshape(-1,2)
    dense=[]
    for a,b in zip(hull,np.roll(hull,-1,axis=0)):
        for f in np.arange(max(1,int(np.ceil(np.linalg.norm(b-a)))))/max(1,int(np.ceil(np.linalg.norm(b-a)))):dense.append(a+f*(b-a))
    points=np.array(dense,np.float32);e=cv2.fitEllipse(points)
    lo=contour.min(0);hi=contour.max(0)
    return e,(lo+hi)/2,contour,points


def gradient_candidates(image,seed):
    """Independent outer-radius gradient proposals; channels compete by vector norm.
    Seeds only identify ball ROI. A wide 8..22px radial interval permits shadow
    edges to win; robustness is measured, not assumed to eliminate that ambiguity.
    """
    pixels=image.astype(np.float32)/255
    gx=cv2.Sobel(pixels,cv2.CV_32F,1,0,ksize=3);gy=cv2.Sobel(pixels,cv2.CV_32F,0,1,ksize=3)
    theta=np.linspace(0,2*np.pi,96,endpoint=False);radii=np.arange(8,22.01,.25)
    points=[];strength=[]
    for t in theta:
        x=seed[0]+radii*np.cos(t);y=seed[1]+radii*np.sin(t)
        ax=cv2.remap(gx,x.astype(np.float32)[None,:],y.astype(np.float32)[None,:],cv2.INTER_LINEAR)[0]
        ay=cv2.remap(gy,x.astype(np.float32)[None,:],y.astype(np.float32)[None,:],cv2.INTER_LINEAR)[0]
        # Radial channel-vector gradient suppresses tangential stripe markings.
        s=np.linalg.norm(ax*np.cos(t)+ay*np.sin(t),axis=1)
        idx=int(np.argmax(s));points.append([x[idx],y[idx]]);strength.append(s[idx])
    return np.array(points,np.float32),np.array(strength)


def robust_gradient_fit(image,seed,iterations=350):
    points,strength=gradient_candidates(image,seed);keep=strength>max(.12,np.percentile(strength,15))
    points=points[keep];rng=np.random.default_rng(716);best=None
    for _ in range(iterations):
        if len(points)<6:break
        try:e=cv2.fitEllipse(points[rng.choice(len(points),6,replace=False)])
        except cv2.error:continue
        (x,y),(a,b),angle=e
        if min(a,b)<15 or max(a,b)>45 or max(a,b)/min(a,b)>2 or np.linalg.norm(np.array([x,y])-seed)>7:continue
        err=residual(points,e);inliers=err<1.1
        score=(int(inliers.sum()),-float(np.median(err)))
        if best is None or score>best[0]:best=(score,e,inliers)
    if best is None:raise ValueError('gradient consensus failed')
    e=best[1];inliers=best[2]
    for _ in range(3):
        if inliers.sum()<6:break
        e=cv2.fitEllipse(points[inliers]);inliers=residual(points,e)<1.1
    angles=np.mod(np.arctan2(points[inliers,1]-e[0][1],points[inliers,0]-e[0][0]),2*np.pi)
    coverage=len(set((angles/(2*np.pi)*12).astype(int)))/12
    # Drop each quadrant to expose sensitivity to incomplete arcs/shadows.
    centers=[]
    for quadrant in range(4):
        q=(np.mod(np.arctan2(points[:,1]-e[0][1],points[:,0]-e[0][0]),2*np.pi)/(np.pi/2)).astype(int)
        subset=points[inliers&(q!=quadrant)]
        if len(subset)>=8:
            try:centers.append(cv2.fitEllipse(subset)[0])
            except cv2.error:pass
    sensitivity=max([np.linalg.norm(np.array(c)-e[0]) for c in centers]+[0.])
    return e,points,inliers,float(coverage),float(sensitivity)


def fit_h(src,dst):
    h,_=cv2.findHomography(src.astype(float),dst.astype(float),method=0)
    return h


def project(points,h):return cv2.perspectiveTransform(np.array(points,dtype=float).reshape(-1,1,2),np.asarray(h,dtype=float)).reshape(-1,2)


def synthetic_checks():
    # Known ellipse with shadow, white internal stripe/hole. This is a limited
    # segmentation/localization regression, not a realism or generality claim.
    image=np.full((100,100,3),[145,170,20],np.uint8);center=(49,48)
    cv2.ellipse(image,(54,54),(19,12),10,0,360,(75,88,10),-1)
    cv2.ellipse(image,center,(16,12),20,0,360,(25,60,220),-1)
    cv2.rectangle(image,(36,45),(62,49),(230,230,230),-1)
    seg,_,_,_=segmentation(image,center,.11)
    grad,_,_,_,arc=robust_gradient_fit(image,center)
    a=float(np.linalg.norm(np.array(seg[0])-center));b=float(np.linalg.norm(np.array(grad[0])-center))
    assert a<2 and b<2,(a,b)
    return {'known_center_px':center,'segmented_center_error_px':a,'gradient_center_error_px':b,'gradient_leave_quadrant_out_sensitivity_px':arc,'scope':'single idealized ellipse plus shadow and stripe; does not validate real JPEG silhouettes'}


def main():
    image=cv2.imread(str(IMAGE));assert image.shape==(720,1280,3)
    measured=json.loads(MEASUREMENTS.read_text());truth={p['id']:p for p in measured['points']}
    records=[];annotated=image.copy()
    for number,(ident,seed) in enumerate(SEEDS,1):
        fits=[]
        for threshold in (.08,.11,.14):
            try:
                e,bbox,contour,hull=segmentation(image,seed,threshold);fits.append((threshold,e,bbox,contour,hull))
            except (ValueError,cv2.error):pass
        if not fits:raise ValueError(ident)
        threshold,e,bbox,contour,hull=min(fits,key=lambda f:abs(f[0]-.11))
        center=np.array(e[0]);shift=max(np.linalg.norm(np.array(f[1][0])-center) for f in fits)
        radial_vector=center-np.array([644.18,349.67]); radial_angle=float(np.degrees(np.arctan2(radial_vector[1],radial_vector[0]))%180)
        ep=ellipse_parameters(e); angle_difference=abs((ep['major_axis_angle_deg']-radial_angle+90)%180-90)
        rec={'image_center_radius_px':float(np.linalg.norm(radial_vector)),'radial_angle_deg':radial_angle,'major_axis_radial_angle_difference_deg':float(angle_difference),'small_sphere_apparent_aspect_approximation':float(np.sqrt(1+(np.linalg.norm(radial_vector)/696)**2)),'id':ident,'seed_px':seed,'segmentation_ellipse':ellipse_parameters(e),'segmentation_bbox_midpoint_px':bbox.tolist(),'bbox_to_segmentation_ellipse_offset_px':float(np.linalg.norm(bbox-center)),'threshold_center_sensitivity_px':float(shift),'segmentation_threshold':threshold,'physical_ground_truth_available':ident in truth}
        cv2.ellipse(annotated,e,(255,255,0),1,cv2.LINE_AA);cv2.drawMarker(annotated,tuple(np.round(bbox).astype(int)),(255,0,255),cv2.MARKER_CROSS,5,1)
        try:
            g,points,inliers,coverage,arc=robust_gradient_fit(image,seed)
            rec.update({'gradient_ellipse':ellipse_parameters(g),'gradient_inlier_count':int(inliers.sum()),'gradient_candidate_count':len(points),'gradient_angular_sector_coverage':coverage,'gradient_leave_quadrant_out_center_sensitivity_px':arc,'gradient_vs_segmentation_center_offset_px':float(np.linalg.norm(np.array(g[0])-center))})
            cv2.ellipse(annotated,g,(0,200,255),1,cv2.LINE_AA)
            for p in points[inliers]:cv2.circle(annotated,tuple(np.round(p).astype(int)),0,(0,255,0),1)
        except (ValueError,cv2.error) as error:rec['gradient_failure']=str(error)
        flags=[]
        if min(e[1])<16:flags.append('segmentation likely selected an internal highlight/fragment rather than full ball')
        if shift>1:flags.append('segmentation threshold sensitivity exceeds 1 px')
        if rec.get('gradient_vs_segmentation_center_offset_px',0)>2:flags.append('methods disagree by more than 2 px; neither establishes true center')
        if rec.get('gradient_leave_quadrant_out_center_sensitivity_px',0)>1:flags.append('gradient center sensitive to omitted arc')
        if ident not in truth:flags.append('no physical XY ground truth; pocket/jaw silhouette may be occluded')
        rec['localization_review_flags']=flags
        cv2.putText(annotated,str(number),(seed[0]+17,seed[1]-12),cv2.FONT_HERSHEY_SIMPLEX,.35,(255,255,255),1,cv2.LINE_AA)
        records.append(rec)
    known=[r for r in records if r['physical_ground_truth_available']]
    dst=np.array([[truth[r['id']]['center_from_upper_left_mm'][a] for a in ('x','y')] for r in known])
    assert len(truth)==9
    for point in truth.values():
        for axis in ('x','y'):
            assert np.isclose(point['center_from_upper_left_inches'][axis]*25.4,point['center_from_upper_left_mm'][axis])
    assert all(np.isfinite(r['segmentation_ellipse']['center_px']).all() for r in records)
    models={}
    fields={'segmentation_bbox':'segmentation_bbox_midpoint_px','segmentation_ellipse':None,'gradient_ellipse':None}
    for method,field in fields.items():
        src=np.array([r[field] if field else r[method]['center_px'] for r in known])
        h=fit_h(src,dst);train=np.linalg.norm(project(src,h)-dst,axis=1);held=[]
        for j in range(len(src)):
            keep=np.arange(len(src))!=j;held.append(float(np.linalg.norm(project(src[j:j+1],fit_h(src[keep],dst[keep]))[0]-dst[j])))
        models[method]={'training_rms_mm':float(np.sqrt(np.mean(train**2))),'leave_one_position_out_rms_mm':float(np.sqrt(np.mean(np.array(held)**2))),'leave_one_position_out_error_mm':dict(zip([r['id'] for r in known],held)),'homography':h.tolist()}
    baseline=models['segmentation_ellipse']['homography']
    for r in records:
        r['baseline_H_bbox_vs_segmentation_center_displacement_mm_diagnostic']=float(np.linalg.norm(project([r['segmentation_bbox_midpoint_px']],baseline)-project([r['segmentation_ellipse']['center_px']],baseline)))
        if 'gradient_ellipse' in r:
            r['baseline_H_gradient_vs_segmentation_center_displacement_mm_diagnostic']=float(np.linalg.norm(project([r['gradient_ellipse']['center_px']],baseline)-project([r['segmentation_ellipse']['center_px']],baseline)))
    result={'source_image':str(IMAGE),'physical_diameter_mm':57.15,'scope':'2D XY; apparent axes are image measurements, never variable physical ball radii','legend':'cyan segmented convex-envelope ellipse; magenta segmentation bbox midpoint; orange gradient-consensus ellipse; green consensus gradient proposals','balls':records,'nine_point_homography_comparison':models,'synthetic_checks':synthetic_checks(),'limits':['No method proves true sphere center from this approximately 13px-radius JPEG silhouette.','Convex envelopes bridge stripe gaps and possibly shadow/rail errors. Gradient consensus can prefer shadow or internal stripe boundaries.','Leave-one-position-out includes shared unknown physical measurement and contact errors; no physical accuracy claim.','Pocket-adjacent balls have no physical ground truth and are never used as calibration targets.','Original measured XY applies only to this exact source snapshot, not subsequent moved-ball checkerboard images.']}
    (ROOT/'ball-outline-consistency.json').write_text(json.dumps(result,indent=2)+'\n');cv2.imwrite(str(ROOT/'ball-outline-consistency.png'),annotated)
    lines=['# Ball outline and center consistency - original source image','','No production correction is activated. All balls retain one fixed 57.15 mm physical diameter. Apparent ellipse diameters below describe source pixels only. Source: `table-grid-20261026-114241.jpg`; original nine physical measurements are used only with that associated snapshot.','','## Methods','','Three chromaticity thresholds segment each source-RGB ROI against locally estimated cloth. A convex envelope bridges internal stripe gaps; its ellipse center is compared with the segmented bounding-box midpoint. These are segmentation-derived boxes, not archived YOLO detector boxes; this experiment does not directly measure current detector midpoint bias. This can also bridge invalid shadow/rail arcs and is not a true-center estimator by definition. Independently, channel-vector Sobel gradients are sampled radially over 8-22 px from coarse ROI seeds. RANSAC-like six-point ellipse hypotheses select consensus edges, then refit. This explicitly allows failure or disagreement. Dropping each quadrant probes arc sensitivity; this is not a statistical confidence interval. No upsampling or learned precision is assumed.','','## Source-image findings','','| Ball | Apparent ellipse diameters px | Major angle deg | Bbox/ellipse offset px | Threshold sensitivity px | Gradient/seg offset px | Gradient arc sensitivity px |','|---|---|---:|---:|---:|---:|---:|']
    for i,r in enumerate(records,1):
        e=r['segmentation_ellipse'];lines.append(f"| {i}: {r['id']} | {e['diameters_px'][0]:.1f} x {e['diameters_px'][1]:.1f} | {e['major_axis_angle_deg']:.0f} | {r['bbox_to_segmentation_ellipse_offset_px']:.2f} | {r['threshold_center_sensitivity_px']:.2f} | {r.get('gradient_vs_segmentation_center_offset_px',float('nan')):.2f} | {r.get('gradient_leave_quadrant_out_center_sensitivity_px',float('nan')):.2f} |")
    lines+=['','Off-axis diagnostic uses frozen optical principal point (644.18,349.67) and approximate focal length 696 px. JSON records major-axis alignment with the radial image direction, radius and sqrt(1+(radius/f)^2), a small-sphere apparent-aspect approximation only. This characterizes a 2D image pattern without estimating ball height or changing physical diameter; segmentation/occlusion can confound it. Near-circular angles are unreliable.','','## Nine-point homography diagnostic','','Each method gets its own simple homography fit to the same nine measured positions; leave-one-position-out refits only the other eight. These are discrepancy diagnostics against unknown-uncertainty measurements, not accuracy estimates. They cannot select a physically correct center method without independent validation. The baseline segmentation-ellipse H also converts inter-method pixel shifts to illustrative table-distance shifts in the JSON; that conversion is not physical truth.','','| Center method | Training RMS mm | Held-out RMS mm |','|---|---:|---:|']
    for method,r in models.items():lines.append(f"| {method} | {r['training_rms_mm']:.2f} | {r['leave_one_position_out_rms_mm']:.2f} |")
    lines+=['','| Measured ball | Bbox held-out mm | Seg ellipse held-out mm | Gradient held-out mm |','|---|---:|---:|---:|']
    for r in known:lines.append('| '+r['id']+' | '+' | '.join(f"{models[m]['leave_one_position_out_error_mm'][r['id']]:.2f}" for m in models)+' |')
    lines+=['','## Interpretation','', 'Visual review finds concrete failure modes: top-left pocket segmentation selected a tiny internal highlight/fragment (7.5 x 12.5 px ellipse), not the whole ball, with 11.5 px threshold sensitivity. This result is retained and flagged as a localization failure. Top-middle and corner balls show 3-10 px inter-method disagreements and can be contaminated by pocket/shadow edges. Even the measured striped middle balls can move 2-3 px under independent gradient fitting, consistent with internal-edge ambiguity. Bottom black threshold sensitivity is about 1.94 px and its segmented-ellipse held-out discrepancy is 14.64 mm; this deserves review but is not automatically discarded.','', 'For a clean radial-pattern example, the far-left measured yellow ellipse aspect is about 1.27 and major-axis direction about 5 degrees from horizontal, close to the image-radial direction; its small-sphere apparent-aspect approximation is about 1.27. The far-right measured purple ellipse aspect is about 1.25 and its major axis is nearly horizontal. These observations are compatible with a fixed-diameter round ball having position-dependent apparent shape; they do not justify changing physical radius or estimating ball Z. Green stripe segmentation and near-circular balls provide less reliable orientation evidence.','','The earlier 8.71 mm homography held-out RMS was obtained with another robust conic implementation on convex-envelope segmentations. Differences here quantify method dependence rather than a new physical truth. A 1 px center shift maps to roughly 2 mm at this image scale, so several pixels of shadow/rail ambiguity can explain a material portion of an 8-14 mm discrepancy. It cannot establish that all error is localization: physical measurements, contact assumptions, and the global map also contribute. No difficult ball is discarded.','','Near-round ellipse angles are unstable and should not be interpreted as meaningful orientation. Rail-contact silhouettes and corner-pocket shadows remain difficult; the six unmeasured pocket balls are diagnostic only. White cue surfaces and white stripes require outer-contour reasoning rather than treating white patches or numbers as complete balls. Gradient methods can reduce color dependence but may instead lock onto stripes, shadow borders, or cushion edges. Review cyan/orange disagreement and green inliers in the annotated original image.','','The synthetic test confirms only limited known-center recovery on an ideal ellipse with a shadow and stripe. Real JPEG anti-aliasing, rail contact, and partial silhouettes remain unvalidated. At approximately 13 px radius, subpixel fitted output is not proof of subpixel correctness.','','Recommendation: retain native-resolution ROIs, publish raw bbox midpoint separately from refined outer-boundary observation, use robust arc selection and uncertainty/occlusion gating, and validate localization separately from XY calibration. Keep physical diameter fixed in table coordinates. Do not add radial image warping to force these raw outlines into circles. More reliable boundary evidence and independent validation should precede production integration.','','## Verification','','JSON decoding, nine physical-coordinate inch conversions, source dimensions, finite fits, and a synthetic known-center/shadow/stripe test are checked by the script. No physical measurement, production file, configuration, or dependency is changed.']
    (ROOT/'ball-outline-consistency.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'models':{k:{a:b for a,b in v.items() if a.endswith('rms_mm')} for k,v in models.items()},'synthetic':result['synthetic_checks']},indent=2))

if __name__=='__main__':main()
