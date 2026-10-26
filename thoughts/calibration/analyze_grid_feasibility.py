"""Offline nine-position calibration experiment. Uses bundled numpy/Pillow only.
Seeds select source-image regions; centers are fitted to color-segmented boundaries.
Not production calibration. Run with the bundled Python runtime; outputs beside script.
"""
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

ROOT=Path(__file__).resolve().parent
META=ROOT/'table-grid-20261026-114241-measurements.json'
DATA=json.loads(META.read_text())
IMAGE=np.asarray(Image.open(ROOT/'table-grid-20261026-114241.jpg').convert('RGB'))
SEEDS={
'middle-row-center-ball':(643,361), 'far-right-purple-middle-row-ball':(1198,364),
'far-left-yellow-middle-row-ball':(96,362), 'interior-purple-second-left-middle-row-ball':(359,362),
'interior-green-second-right-middle-row-ball':(925,361),'top-orange-solid-second-left-ball':(352,97),
 'top-red-striped-second-right-ball':(928,94),'bottom-black-second-left-ball':(363,627),
 'bottom-yellow-second-right-ball':(925,630)}

def component(mask,seed):
    """Connected component closest to region seed; cloth and shadows suppressed first."""
    seen=set(); components=[]
    for y,x in zip(*np.where(mask)):
        if (y,x) in seen: continue
        stack=[(y,x)];seen.add((y,x)); comp=[]
        while stack:
            qy,qx=stack.pop();comp.append((qy,qx))
            for dy,dx in ((1,0),(-1,0),(0,1),(0,-1)):
                p=(qy+dy,qx+dx)
                if 0<=p[0]<mask.shape[0] and 0<=p[1]<mask.shape[1] and mask[p] and p not in seen:
                    seen.add(p);stack.append(p)
        if len(comp)>30:components.append(np.array(comp))
    if not components:raise ValueError('No ball component')
    return min(components,key=lambda c: np.linalg.norm(c.mean(0)-np.array(seed)[::-1]))

def ellipse_fit(boundary):
    """IRLS geometric-distance approximation to a conic; no circular-shape assumption."""
    origin=boundary.mean(0); q=(boundary-origin)/20
    x,y=q.T; design=np.column_stack((x*x,x*y,y*y,x,y));w=np.ones(len(q))
    for _ in range(8):
        a=np.linalg.lstsq(design*w[:,None],w,rcond=None)[0]
        gradient=np.column_stack((2*a[0]*x+a[1]*y+a[3],2*a[2]*y+a[1]*x+a[4]))
        residual=(design@a-1)/np.maximum(np.linalg.norm(gradient,axis=1),.01)
        scale=max(np.median(np.abs(residual-np.median(residual)))*1.4826,.03)
        w=1/np.sqrt(1+(residual/(2*scale))**2)
    A=np.array([[a[0],a[1]/2],[a[1]/2,a[2]]]); c=-np.linalg.solve(2*A,a[3:])
    level=1+c@A@c; vals,vecs=np.linalg.eigh(A/level)
    if np.any(vals<=0):raise ValueError('Nonellipse boundary')
    axes=20/np.sqrt(vals)
    return origin+20*c,axes,vecs,float(np.sqrt(np.mean(residual**2))*20)

def localize(seed,threshold):
    sx,sy=seed; span=24;x0=sx-span;y0=sy-span
    crop=IMAGE[y0:sy+span+1,x0:sx+span+1].astype(float)
    yy,xx=np.indices(crop.shape[:2]);rad=np.hypot(xx-span,yy-span)
    cloth=np.median(crop[(rad>19)&(rad<24)],axis=0)
    chroma=crop/np.maximum(crop.sum(2,keepdims=True),1)
    clothchroma=cloth/cloth.sum()
    deviation=np.linalg.norm(chroma-clothchroma,axis=2)
    dark=crop.sum(2)<cloth.sum()*.47
    mask=((deviation>threshold)|dark)&(rad<22)
    c=component(mask,(span,span)); m=np.zeros(mask.shape,bool);m[c[:,0],c[:,1]]=True
    # Flood exterior background; fill internal stripe/highlight holes before tracing outer boundary.
    exterior=np.zeros(m.shape,bool); stack=[]
    for j in range(m.shape[0]):
        for k in (0,m.shape[1]-1):
            if not m[j,k]: exterior[j,k]=True;stack.append((j,k))
    for k in range(m.shape[1]):
        for j in (0,m.shape[0]-1):
            if not m[j,k]: exterior[j,k]=True;stack.append((j,k))
    while stack:
        j,k=stack.pop()
        for dj,dk in ((1,0),(-1,0),(0,1),(0,-1)):
            a,b=j+dj,k+dk
            if 0<=a<m.shape[0] and 0<=b<m.shape[1] and not m[a,b] and not exterior[a,b]:
                exterior[a,b]=True;stack.append((a,b))
    m=~exterior
    interior=m.copy();interior[1:-1,1:-1]&=m[:-2,1:-1]&m[2:,1:-1]&m[1:-1,:-2]&m[1:-1,2:]
    by,bx=np.where(m&~interior);boundary=np.column_stack((bx+x0,by+y0))
    # Ball silhouette is convex; use its convex envelope to exclude stripe gaps
    # connected to the exterior. This can bridge missing visible boundary arcs.
    pts=sorted(set(map(tuple,boundary)))
    def cross(o,a,b):return (a[0]-o[0])*(b[1]-o[1])-(a[1]-o[1])*(b[0]-o[0])
    lower=[]
    for p in pts:
        while len(lower)>=2 and cross(lower[-2],lower[-1],p)<=0:lower.pop()
        lower.append(p)
    upper=[]
    for p in reversed(pts):
        while len(upper)>=2 and cross(upper[-2],upper[-1],p)<=0:upper.pop()
        upper.append(p)
    hull=np.array(lower[:-1]+upper[:-1]); samples=[]
    for a,b in zip(hull,np.roll(hull,-1,axis=0)):
        n=max(1,int(np.ceil(np.linalg.norm(b-a))))
        samples.extend(a+(b-a)*f for f in np.arange(n)/n)
    boundary=np.array(samples)
    fit=ellipse_fit(boundary)
    return fit,boundary

# Normalize source and target coordinates to make fitting numerically conditioned.
def radial(q,k):return q*(1+k*np.sum(q*q,axis=1)[:,None])
def fit_h(q,t):
    x,y=q.T;u,v=t.T;z=np.zeros(len(q));o=np.ones(len(q))
    A=np.vstack((np.column_stack((x,y,o,z,z,z,-u*x,-u*y)),np.column_stack((z,z,z,x,y,o,-v*x,-v*y))))
    h=np.r_[np.linalg.lstsq(A,np.r_[u,v],rcond=None)[0],1].reshape(3,3)
    return h

def project(q,h):
    p=np.column_stack((q,np.ones(len(q))))@h.T
    return p[:,:2]/p[:,2,None]

def model_fit(q,t,use_radial):
    # One fixed image-centered radial coefficient. Chosen only from each training fold.
    best=None
    for k in np.linspace(-.18,.18,361) if use_radial else [0.]:
        h=fit_h(radial(q,k),t);error=np.sum((project(radial(q,k),h)-t)**2)
        if best is None or error<best[0]:best=(error,float(k),h)
    return best[1:]

points=DATA['points']; observations=[]; canvas=Image.fromarray(IMAGE);draw=ImageDraw.Draw(canvas)
for i,p in enumerate(points):
    seed=SEEDS[p['id']];fits=[]
    for threshold in (.09,.11,.13):
        try:fits.append((threshold,*localize(seed,threshold)))
        except (ValueError,np.linalg.LinAlgError):pass
    if not fits: raise ValueError(p['id'])
    chosen=min(fits,key=lambda r:abs(r[0]-.11));threshold,fit,boundary=chosen
    center,axes,vecs,residual=fit
    spread=max(np.linalg.norm(f[1][0]-center) for f in fits)
    observations.append({'id':p['id'],'seed_px':seed,'center_px':center.tolist(),'ellipse_semiaxes_px':axes.tolist(),'boundary_fit_rms_px':residual,'threshold_center_sensitivity_px':float(spread),'mask_threshold':threshold})
    for a,b in boundary:draw.point((int(a),int(b)),fill='cyan')
    a,b=center;draw.line((a-5,b,a+5,b),fill='red',width=1);draw.line((a,b-5,a,b+5),fill='red',width=1)
    draw.text((a+14,b-16),f'{i+1}',fill='white',stroke_width=1,stroke_fill='black')
canvas.save(ROOT/'table-grid-feasibility-annotated.png')
src=np.array([o['center_px'] for o in observations]); target=np.array([[p['center_from_upper_left_mm'][a] for a in ('x','y')] for p in points])
q=(src-np.array([640,360]))/640;t=(target-np.array([1168.4,577.85]))/1168.4
results={}
for name,use_radial in [('homography',False),('homography_plus_one_radial_parameter',True)]:
    k,h=model_fit(q,t,use_radial);pred=project(radial(q,k),h);training=np.linalg.norm(pred-t,axis=1)*1168.4
    held=[];foldks=[]
    for idx in range(len(q)):
        keep=np.arange(len(q))!=idx;fk,fh=model_fit(q[keep],t[keep],use_radial)
        held.append(float(np.linalg.norm(project(radial(q[idx:idx+1],fk),fh)[0]-t[idx])*1168.4));foldks.append(fk)
    results[name]={'radial_coefficient':k,'homography_normalized':h.tolist(),'training_error_mm':training.tolist(),'training_rms_mm':float(np.sqrt(np.mean(training**2))),'leave_one_position_out_error_mm':held,'leave_one_position_out_rms_mm':float(np.sqrt(np.mean(np.array(held)**2))),'leave_one_position_out_median_mm':float(np.median(held)),'leave_one_position_out_max_mm':max(held),'fold_radial_coefficients':foldks}

# Synthetic recovery verifies the numerical homography/projection implementation.
synthetic=np.array([[1.1,.02,.01],[-.01,.9,.03],[.02,-.03,1.]])
assert np.max(np.abs(project(q,fit_h(q,project(q,synthetic)))-project(q,synthetic)))<1e-10
known_k=-.03
recovered_k,recovered_h=model_fit(q,project(radial(q,known_k),synthetic),True)
assert abs(recovered_k-known_k)<1e-10
assert np.max(np.abs(project(radial(q,recovered_k),recovered_h)-project(radial(q,known_k),synthetic)))<1e-10
assert len(points)==9 and all(p['accepted_as_complete_ground_truth'] for p in points)
for p in points:
 for axis in ('x','y'):
  assert np.isclose(p['center_from_upper_left_inches'][axis]*25.4,p['center_from_upper_left_mm'][axis])
output={'image_path':str(ROOT/'table-grid-20261026-114241.jpg'),'image_size':[1280,720],'observations':observations,'results':results,'synthetic_homography_recovery':'passed','synthetic_radial_recovery':'passed','physical_uncertainty':'unknown; conversion decimals are not measurement precision','excluded_unmeasured_pocket_adjacent_balls':6}
(ROOT/'table-grid-feasibility-results.json').write_text(json.dumps(output,indent=2)+'\n')
lines=['# Offline nine-position feasibility analysis','', 'This is exploratory analysis, not an activated calibration. Six unmeasured pocket-adjacent balls are excluded; no further pocket measurements are required now. Corner and pocket accuracy remains unverified.','', '## Method','', 'Seeds select 49×49 source-RGB regions. Local cloth chromaticity is estimated from the outer annulus; chromaticity difference or substantial darkness selects ball pixels. A connected component nearest the seed supplies outer-boundary pixels after flood-filling internal stripe/highlight holes and taking a convex envelope to bridge stripe gaps connected to the exterior. Envelope segments can bridge missing arcs and should not be mistaken for independently observed edges. An iteratively reweighted conic fit estimates ellipse centers without forcing circles. The annotated image shows the actual selected boundaries and centers. Three segmentation thresholds probe sensitivity, not physical uncertainty. Shadows, rail contact, stripes, JPEG edges and nonelliptic silhouettes can bias these centers. The source resolution is used throughout.','', 'Models compare an 8-parameter homography with a homography plus one image-centered radial coordinate correction. Image center and normalization are fixed rather than estimated. The radial coefficient is searched over [-0.18, 0.18]; each leave-one-position-out fold fits only its eight training positions. This is a restricted 2D empirical model, not a lens-intrinsics calibration. No Z model is used.','', '## Results','', '| Model | Training RMS mm | Leave-one-out RMS mm | Median mm | Max mm |','|---|---:|---:|---:|---:|']
for name,r in results.items():lines.append(f"| {name} | {r['training_rms_mm']:.2f} | {r['leave_one_position_out_rms_mm']:.2f} | {r['leave_one_position_out_median_mm']:.2f} | {r['leave_one_position_out_max_mm']:.2f} |")
lines+=['','| Ball | Fitted pixel center | Threshold sensitivity px | H held-out mm | Radial held-out mm |','|---|---|---:|---:|---:|']
for i,o in enumerate(observations):lines.append(f"| {i+1}: {o['id']} | {o['center_px'][0]:.2f}, {o['center_px'][1]:.2f} | {o['threshold_center_sensitivity_px']:.2f} | {results['homography']['leave_one_position_out_error_mm'][i]:.2f} | {results['homography_plus_one_radial_parameter']['leave_one_position_out_error_mm'][i]:.2f} |")
lines+=['','## Interpretation and limits','', 'The extra radial parameter does not earn its complexity: its held-out RMS is worse despite slightly lower training error. Keep the homography as the comparison baseline; do not activate either model from this experiment. Bottom black has the largest segmentation-threshold center sensitivity (see the sensitivity table), and left/top/bottom held-out discrepancies deserve review without automatic removal. All recorded inch-to-mm conversions pass consistency checks; that cannot establish that physical measurements are correct.','', 'Held-out discrepancies mix image localization, physical placement/measurement error, table geometry and model inadequacy; they are not verified camera accuracy. Training errors are not validation. Measurement uncertainty is unknown; shared table scale and rail-contact assumptions create correlated errors. Eighth-inch reporting granularity does not establish a numerical error bound. No point was automatically removed. Large per-point errors should prompt a repeat measurement and image capture, not automatic rejection.','', 'Nine points occupy only three rows. The extreme corner/pocket regions lie outside their convex hull, and some leave-one-out folds extrapolate at edge positions. An unconstrained thin-plate spline could interpolate the nine training points but is unsupported by this dataset. Independent off-grid validation and repeated frames are needed before selecting or deploying a nonlinear model. Fixed physical diameter is a modeling input, not established by these center fits; a center mapping alone does not guarantee circular rectified silhouettes.','', 'Next: review the annotated contours, repeat any anomalous measured positions, use one movable ball for independent accessible off-grid center/edge and interior corner-quadrant positions safely away from pockets, and accept a more complex mapping only if it improves held-out error reliably. Pocket measurements can remain excluded.','', '## Reproduction','', 'Run this file with the bundled Python runtime (NumPy and Pillow). No production dependencies or configuration are modified. JSON conversion checks and synthetic homography and radial-model recovery pass. Raw results include model matrices, fold coefficients and boundary-fit diagnostics.']
(ROOT/'table-grid-feasibility-report.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({name:{k:v for k,v in r.items() if k in ('training_rms_mm','leave_one_position_out_rms_mm','leave_one_position_out_max_mm','radial_coefficient')} for name,r in results.items()},indent=2))
