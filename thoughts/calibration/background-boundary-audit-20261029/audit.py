"""Saved-pixel comparison of unimodal versus observed multimodal background.
Never changes production or accepts a ball. Sampling and threshold defaults are
read from frozen runtime traces, with the same source-resolution scaling.
"""
from pathlib import Path
import json
import hashlib
import importlib.util
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
BASE=ROOT/'thoughts/calibration/actual-detector-refiner-roi-20261029'
spec=importlib.util.spec_from_file_location('cyclic',ROOT/'thoughts/calibration/offline-cyclic-boundary-20261028.py')
cyclic=importlib.util.module_from_spec(spec);spec.loader.exec_module(cyclic)


def project(h,p):
 q=h@np.r_[p,1];return q[:2]/q[2]


def background_modes(pixels):
 """Deterministic two-median RGB clustering; empty clusters retain seed."""
 centers=pixels[[np.argmin(pixels.mean(axis=1)),np.argmax(pixels.mean(axis=1))]].copy()
 for _ in range(20):
  labels=np.argmin(((pixels[:,None]-centers[None])**2).sum(axis=2),axis=1)
  new=np.array([np.median(pixels[labels==i],axis=0) if np.any(labels==i) else centers[i] for i in range(2)])
  if np.max(abs(new-centers))<1e-10:break
  centers=new
 return centers


def main():
 imagepath=ROOT/'thoughts/calibration/table-grid-20261026-114241.jpg'
 image=cv2.cvtColor(cv2.imread(str(imagepath)),cv2.COLOR_BGR2RGB).astype(float)/255
 summary=json.loads((BASE/'summary.json').read_text());cfg=summary['runtime_parameters'];scale=image.shape[1]/cfg['reference_width_px']
 config=dict(cyclic.DEFAULT);config['paths']=cfg['maximum_cyclic_paths']; results=[]
 for file in sorted((BASE/'cases').glob('*.json')):
  data=json.loads(file.read_text());ki=data['intrinsics'];k=np.array([[ki['fx'],0,ki['cx']],[0,ki['fy'],ki['cy']],[0,0,1]])
  bbox=data['proposal']['bbox_xyxy_source_px'];left,top,right,bottom=bbox;axis=np.array([(left+right)/2,(top+bottom)/2])
  d=np.linalg.inv(k)@np.r_[axis,1];d/=np.linalg.norm(d);v=np.array([d[1],-d[0],0]);s=np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
  r=np.eye(3)+s+s@s/(1+d[2]);h=np.linalg.inv(np.diag([ki['fx'],ki['fy'],1])@r@np.linalg.inv(k))
  radius=np.median([np.linalg.norm(project(np.linalg.inv(h),p)) for p in [[left,axis[1]],[right,axis[1]],[axis[0],top],[axis[0],bottom]]])
  limit=(np.ceil(cfg['patch_size_px']*scale)-1)/2
  def sample(local):
   if max(abs(local))>limit:return None
   x,y=project(h,local)
   if x<0 or y<0 or x>=image.shape[1]-1 or y>=image.shape[0]-1:return None
   ix,iy=int(x),int(y);wx,wy=x-ix,y-iy
   return image[iy,ix]*(1-wx)*(1-wy)+image[iy,ix+1]*wx*(1-wy)+image[iy+1,ix]*(1-wx)*wy+image[iy+1,ix+1]*wx*wy
  backgrounds=[];byray=[]
  for ray in range(cfg['angular_samples']):
   t=2*np.pi*ray/cfg['angular_samples'];direction=np.array([np.cos(t),np.sin(t)])
   samples=[sample(direction*radius*factor) for factor in [cfg['annulus_inner_factor'],cfg['annulus_outer_factor']]]
   samples=[p for p in samples if p is not None];backgrounds.extend(samples);byray.append(samples)
  backgrounds=np.array(backgrounds);median=np.median(backgrounds,axis=0);modes=background_modes(backgrounds)
  row=dict(ball_id=data['ball_id'],trace_sha256=hashlib.sha256(file.read_bytes()).hexdigest(),runtime_status=data['refinement']['status'],variants=[])
  firstfits=[e['data'] for e in data['trace']['events'] if e['iteration']==1 and e['data']['kind']=='refined_fit']
  reference=firstfits[-1] if firstfits else None
  for model in ['single','two_median','ray_local']:
   candidates=[]
   dist=lambda a,b:float(np.sqrt(np.mean((a-b)**2)))
   for ray in range(cfg['angular_samples']):
    t=2*np.pi*ray/cfg['angular_samples'];direction=np.array([np.cos(t),np.sin(t)])
    bg=np.atleast_2d(median) if model=='single' else modes if model=='two_median' else np.atleast_2d(np.median(byray[ray],axis=0)) if byray[ray] else np.atleast_2d(median)
    # Both inner and outer use their nearest observed background class. This
    # is intentionally a stricter foreground test than selecting a convenient
    # distant class for the inner sample.
    bgdist=lambda p:float(np.sqrt(np.min(np.mean((bg-p)**2,axis=1))))
    options=[]
    for rad in np.linspace(radius*cfg['radius_min_fraction'],radius*cfg['radius_max_fraction'],cfg['radial_samples']):
     samples=[sample(direction*(rad+offset*scale)) for offset in [-cfg['background_probe_px'],cfg['background_probe_px'],-cfg['gradient_step_px'],cfg['gradient_step_px']]]
     if any(p is None for p in samples):continue
     inner,outer,a,b=samples;strength=dist(a,b)
     if strength<cfg['minimum_gradient'] or bgdist(inner)<cfg['minimum_foreground_background_contrast'] or bgdist(outer)>cfg['maximum_outer_background_distance'] or bgdist(outer)>bgdist(inner):continue
     options.append(dict(ray=ray,local=(direction*rad).tolist(),source=project(h,direction*rad).tolist(),strength=strength))
    chosen=[]
    for option in sorted(options,key=lambda c:(-c['strength'],np.linalg.norm(c['local']))):
     if all(np.linalg.norm(np.array(option['local'])-o['local'])>=2*cfg['gradient_step_px']*scale for o in chosen):chosen.append(option)
     if len(chosen)>=cfg['candidates_per_ray']:break
    candidates.extend(chosen)
   paths,diag=cyclic.cyclic_paths(candidates,config);fits=[]
   for path in paths:
    try:fits.append(cyclic.path_fit(path,h,config,image.shape[1],bbox,axis))
    except (ValueError,np.linalg.LinAlgError) as error:fits.append(dict(error=str(error)))
   near=[]
   if reference:
    center=np.array(reference['center_local_px']);ref_radius=reference['radius_local_px']
    near=sorted(set(c['ray'] for c in candidates if abs(np.linalg.norm(np.array(c['local'])-center)-ref_radius)<=cfg['inlier_threshold_px']*scale))
   row['variants'].append(dict(model=model,candidates=len(candidates),supported_rays=len(set(c['ray'] for c in candidates)),rays_near_existing_first_aim_circle=near,path_diagnostics=diag,paths=fits))
  results.append(row)
 report=dict(purpose='Pixel candidate and geometric diagnostic only, no runtime acceptance or physical truth',image_sha256=hashlib.sha256(imagepath.read_bytes()).hexdigest(),dependencies=dict(numpy=np.__version__,opencv=cv2.__version__),cases=results)
 (HERE/'report.json').write_text(json.dumps(report,indent=2)+'\n')
 for row in results:print(row['ball_id'],row['runtime_status'],[(v['model'],v['candidates'],v['supported_rays'],len(v['rays_near_existing_first_aim_circle']),sum(p.get('geometric_diagnostic_pass',False) for p in v['paths'])) for v in row['variants']])

if __name__=='__main__':main()
