#!/usr/bin/env python3
"""Deterministic cyclic-path experiment on frozen boundary candidates.

Research comparison only. It does not rerun pixel sampling/aiming, qualify physical
truth, or substitute a diagnostic point into runtime tracking. Trace extraction
and standalone compact-input replay are explicit separate operations.
"""
from __future__ import annotations
import argparse
from collections import defaultdict, Counter
import hashlib
import importlib.util
import itertools
import json
import math
from pathlib import Path
import platform
import subprocess
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
CAL=ROOT/'thoughts/calibration'
DEFAULT={'paths':8,'continuity_weight':1.0,'strength_floor':1e-9,'angular_samples':96,
 'minimum_supported_fraction':.5,'maximum_gap_fraction':1/3,'angular_sectors':16,
 'minimum_coverage':2/3,'minimum_inliers':28,'irls_iterations':10,'irls_tuning':2.5,
 'irls_scale_floor_px':.25,'convergence_px':.01,'numerical_epsilon':1e-9,
 'maximum_residual_px':1.5,'maximum_center_spread_px':1.5,'reference_width_px':3840,
 'radius_min_fraction':.6,'radius_max_fraction':1.4,'maximum_center_shift_fraction':.2,
 'outline_samples':48}

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):p.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
def project(h,p):
 a=np.column_stack((np.atleast_2d(p),np.ones(len(np.atleast_2d(p)))))@h.T
 if np.any(abs(a[:,2])<1e-12):raise ValueError('Projection at infinity')
 return a[:,:2]/a[:,2:]

def homography(a,b):
 def norm(p):
  c=p.mean(0);s=np.sqrt(2)/np.linalg.norm(p-c,axis=1).mean()
  t=np.array([[s,0,-s*c[0]],[0,s,-s*c[1]],[0,0,1.]])
  return project(t,p),t
 a,ta=norm(np.asarray(a));b,tb=norm(np.asarray(b));rows=[]
 for (x,y),(u,v) in zip(a,b):
  rows.extend([[-x,-y,-1,0,0,0,u*x,u*y,u],[0,0,0,-x,-y,-1,v*x,v*y,v]])
 _,sv,vh=np.linalg.svd(np.asarray(rows),full_matrices=False)
 if sv[-2]<1e-10:raise ValueError('Rank deficient source projection')
 h=np.linalg.inv(tb)@vh[-1].reshape(3,3)@ta
 return h/h[2,2]

def canonical(states):
 # Candidate indices come from this geometric ordering, never trace event order.
 return sorted(states,key=lambda c:(c['ray'],np.linalg.norm(c['local']),*c['source'],c['strength']))

def cyclic_paths(candidates,cfg):
 groups=defaultdict(list)
 for c in canonical(candidates):groups[c['ray']].append(c)
 rays=sorted(groups)
 if not rays:return [],{'supported_fraction':0,'largest_gap_fraction':1,'reason':'no_candidates'}
 if any(len(groups[r])>3 for r in rays):raise ValueError('Frozen candidate bound exceeded')
 n=cfg['angular_samples'];gaps=np.diff(rays+[rays[0]+n])/n
 diag={'supported_fraction':len(rays)/n,'largest_gap_fraction':float(max(gaps)),'reason':None}
 if diag['supported_fraction']<cfg['minimum_supported_fraction'] or max(gaps)>cfg['maximum_gap_fraction']:
  diag['reason']='insufficient_angular_support';return [],diag
 r0=float(np.median([np.linalg.norm(c['local']) for c in candidates]))
 if r0<=0:raise ValueError('Invalid radial scale')
 def unary(c):return -math.log(max(c['strength'],cfg['strength_floor']))
 def edge(a,b,delta):
  return cfg['continuity_weight']*(np.linalg.norm(a['local'])-np.linalg.norm(b['local']))**2/(r0*r0*delta)
 cycles=[];k=cfg['paths']
 for first,start in enumerate(groups[rays[0]]):
  prefixes={first:[(unary(start),(first,))]}
  for i in range(1,len(rays)):
   prev=groups[rays[i-1]];cur=groups[rays[i]];delta=2*math.pi*(rays[i]-rays[i-1])/n
   next_layer={}
   for dest,c in enumerate(cur):
    options=[(cost+unary(c)+edge(prev[src],c,delta),path+(dest,))
             for src,seq in prefixes.items() for cost,path in seq]
    next_layer[dest]=sorted(options,key=lambda q:(q[0],q[1]))[:k]
   prefixes=next_layer
  delta=2*math.pi*(rays[0]+n-rays[-1])/n
  for last,seq in prefixes.items():
   for cost,path in seq:cycles.append((cost+edge(groups[rays[-1]][last],start,delta),path))
 cycles=sorted(cycles,key=lambda q:(q[0],q[1]))[:k]
 return [{'cost':float(cost),'state_path':list(path),'rays':rays,
          'points_local_px':[groups[r][j]['local'] for r,j in zip(rays,path)],
          'points_source_px':[groups[r][j]['source'] for r,j in zip(rays,path)]} for cost,path in cycles],diag

def fit_circle(points,cfg,scale,initial=None):
 """Centered algebraic start, then Rust's Cauchy IRLS normal equations.

 residual=distance-radius, scale=max(median(abs(residual)), scaled floor),
 weight=1/(1+(residual/(tuning*scale))²); same Jacobian and stop criterion.
 """
 p=np.asarray(points,float)
 if len(p)<3:raise ValueError('Too few circle observations')
 if initial is None:
  o=p.mean(0);q=p-o;a=np.column_stack((2*q,np.ones(len(q))))
  sol,_,rank,_=np.linalg.lstsq(a,np.sum(q*q,axis=1),rcond=None)
  if rank<3:raise ValueError('Degenerate circle observations')
  fit=np.r_[sol[:2]+o,math.sqrt(max(0,sol[2]+sol[:2]@sol[:2]))]
 else:fit=np.array(initial,float)
 for _ in range(cfg['irls_iterations']):
  delta=fit[:2]-p;d=np.linalg.norm(delta,axis=1);res=d-fit[2]
  robust=max(float(np.median(abs(res))),cfg['irls_scale_floor_px']*scale)
  j=np.column_stack((delta/np.maximum(d[:,None],cfg['numerical_epsilon']),-np.ones(len(p))))
  w=1/(1+(res/(cfg['irls_tuning']*robust))**2)
  a=j.T@(w[:,None]*j);rhs=-j.T@(w*res)
  if np.linalg.matrix_rank(a,tol=cfg['numerical_epsilon'])<3:raise ValueError('Singular IRLS normal equations')
  step=np.linalg.solve(a,rhs);fit+=step
  if not np.isfinite(fit).all() or fit[2]<=0:raise ValueError('Invalid circle')
  if np.linalg.norm(step)<cfg['convergence_px']*scale:break
 return fit

def path_fit(path,h,cfg,width,bbox,axis):
 scale=width/cfg['reference_width_px'];p=np.asarray(path['points_local_px']);fit=fit_circle(p,cfg,scale)
 center=project(h,fit[:2])[0];angles=np.mod(np.arctan2(p[:,1]-fit[1],p[:,0]-fit[0]),2*math.pi)
 coverage=len(set(np.minimum((angles/(2*math.pi)*cfg['angular_sectors']).astype(int),cfg['angular_sectors']-1)))/cfg['angular_sectors']
 ordered=np.sort(angles);gap=float(max(np.diff(np.r_[ordered,ordered[0]+2*math.pi]))/(2*math.pi))
 residual=np.linalg.norm(p-fit[:2],axis=1)-fit[2];rms=float(np.sqrt(np.mean(residual**2)))
 theta=np.arange(cfg['outline_samples'])*2*math.pi/cfg['outline_samples']
 outline=project(h,fit[:2]+fit[2]*np.column_stack((np.cos(theta),np.sin(theta))))
 spread=0.;fail=[]
 for omit in range(4):
  subset=p[(angles/(math.pi/2)).astype(int)!=omit]
  other=fit_circle(subset,cfg,scale,fit);other_center=project(h,other[:2])[0]
  spread=max(spread,float(np.linalg.norm(other_center-center)))
 if len(p)<cfg['minimum_inliers']:fail.append('insufficient_inliers')
 if coverage<cfg['minimum_coverage'] or gap>cfg['maximum_gap_fraction']:fail.append('inadequate_coverage')
 if rms>cfg['maximum_residual_px']*scale:fail.append('excessive_residual')
 if spread>cfg['maximum_center_spread_px']*scale:fail.append('unstable_center')
 if bbox is not None:
  side=min(bbox[2]-bbox[0],bbox[3]-bbox[1]); proposal_radius=side/2
  # Radius is in local patch units, whose magnification varies spatially.
  # Compare source outline radius to the source proposal only as a diagnostic.
  radius_source=float(np.median(np.linalg.norm(outline-center,axis=1)))
  if not cfg['radius_min_fraction']*proposal_radius<=radius_source<=cfg['radius_max_fraction']*proposal_radius:fail.append('source_radius_ratio_diagnostic')
  if np.linalg.norm(center-np.array([(bbox[0]+bbox[2])/2,(bbox[1]+bbox[3])/2]))>cfg['maximum_center_shift_fraction']*side:fail.append('excessive_center_shift')
 return {'center_local_px':fit[:2].tolist(),'radius_local_px':float(fit[2]),'center_source_px':center.tolist(),
 'outline_source_px':outline.tolist(),'residual_rms_local_px':rms,'coverage':coverage,'largest_gap_fraction':gap,
 'source_residual_rms_px':float(np.sqrt(np.mean(np.sum((project(h,fit[:2]+fit[2]*(p-fit[:2])/np.linalg.norm(p-fit[:2],axis=1)[:,None])-np.asarray(path['points_source_px']))**2,axis=1)))),
 'quadrant_removal_center_spread_source_px':spread,'geometric_diagnostic_failures':fail,
 'geometric_diagnostic_pass':not fail,'not_checked':['pixel foreground/background gates','new sampling or aiming','physical truth','global contour ambiguity completeness']}

def extract(output):
 paths=sorted((CAL/'converged-candidate-audit-20261027/cases').glob('*.json'))+sorted((CAL/'robustness-audit-20261027/reserved-cases').glob('*.json'))
 if len(paths)!=33:raise ValueError('Expected exact 33-case corpus')
 center=json.loads((CAL/'center-outline-audit-20261028/inputs.json').read_text());cases=[];hashes={}
 for path in paths:
  d=json.loads(path.read_text());hashes[str(path.relative_to(ROOT))]=sha(path);groups=defaultdict(list);axes={}
  if d['trace'].get('dropped_events',0):raise ValueError('Incomplete candidate trace')
  for e in d['trace']['events']:
   v=e['data'];i=e['iteration']
   if v['kind']=='iteration_started':axes[i]=v['axis_image_px']
   if v['kind']=='boundary_candidate':groups[i].append({'ray':v['ray'],'local':v['point_local_px'],'source':v['point_image_px'],'strength':v['strength']})
  scene=d.get('source_image_id',d.get('image_id'));ball=d.get('ball_id',d.get('proposal',{}).get('id'))
  proposal=d.get('proposal',{});bbox=proposal.get('bbox_xyxy_source_px')
  if bbox is None and 'center_px_approx' in proposal:
   c=np.array(proposal['center_px_approx']);r=proposal['bbox_side_px']/2;bbox=np.r_[c-r,c+r].tolist()
  iterations=[]
  for iteration,cs in sorted(groups.items()):
   if len(cs)<4:continue
   h=homography(np.array([v['local'] for v in cs]),np.array([v['source'] for v in cs]))
   err=float(np.linalg.norm(project(h,[v['local'] for v in cs])-np.array([v['source'] for v in cs]),axis=1).max())
   if err>1e-6:raise ValueError(f'Source projection mismatch {path}: {err}')
   iterations.append({'iteration':iteration,'axis_source_px':axes[iteration],'local_to_source_homography':h.tolist(),'projection_reconstruction_max_px':err,'candidates':canonical(cs)})
  cases.append({'ball_id':ball,'scene_id':scene,'source_case':str(path.relative_to(ROOT)),'width':d['intrinsics']['source_width_px'],'bbox':bbox,'baseline':d['refinement'],'iterations':iterations})
 image_hashes={name:expected for name,expected in center['source_hashes'].items() if Path(name).suffix.lower() in ('.jpg','.png')}
 for name,expected in image_hashes.items():
  if sha(ROOT/name)!=expected:raise ValueError(f'Source image mismatch {name}')
 result={'schema_version':1,'image_hashes':image_hashes,'purpose':'Frozen candidate extraction; manual proposals, not detections or truth','extraction_revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'source_hashes':hashes,'refiner_source_sha256':sha(ROOT/'perception/src/ball_localization.rs'),'cases':cases,'reference_evidence':{k:center[k] for k in ['measurements','segmentation_observations']}}
 output.parent.mkdir(parents=True,exist_ok=True);write(output,result)

def ring(radius, count=12,strength=.8,center=(0,0)):
 return [{'ray':i,'local':(np.array(center)+radius*np.array([math.cos(2*math.pi*i/count),math.sin(2*math.pi*i/count)])).tolist(),'source':[radius*math.cos(2*math.pi*i/count),radius*math.sin(2*math.pi*i/count)],'strength':strength} for i in range(count)]

def tests():
 cfg={**DEFAULT,'angular_samples':4,'paths':8,'minimum_inliers':3};cs=ring(10,4)+ring(12,4)
 got,_=cyclic_paths(cs,cfg)
 groups=defaultdict(list)
 for c in canonical(cs):groups[c['ray']].append(c)
 expected=[];r0=11
 for path in itertools.product(range(2),repeat=4):
  score=sum(-math.log(groups[i][j]['strength']) for i,j in enumerate(path))
  score+=sum((np.linalg.norm(groups[i][path[i]]['local'])-np.linalg.norm(groups[(i+1)%4][path[(i+1)%4]]['local']))**2/(r0*r0*(math.pi/2)) for i in range(4))
  expected.append((score,list(path)))
 expected.sort(key=lambda q:(q[0],q[1]));assert [(p['cost'],p['state_path']) for p in got]==expected[:8]
 assert got[0]['state_path'] in [[0]*4,[1]*4] and got[1]['state_path'] in [[0]*4,[1]*4]
 reverse,_=cyclic_paths(list(reversed(cs)),cfg);assert got==reverse
 assert got[0]['cost']==got[1]['cost'] # coherent concentric contours both optimal
 shadow=ring(10,4,.4)+ring(12,4,.9);best,_=cyclic_paths(shadow,cfg);assert best[0]['state_path']==[1]*4
 sparse=[v for v in ring(10,12) if v['ray']<5];_,quality=cyclic_paths(sparse,{**cfg,'angular_samples':12});assert quality['reason']=='insufficient_angular_support'
 p=np.array([v['local'] for v in ring(13,96,center=(1.2,-.8))]);fit=fit_circle(p,DEFAULT,1);assert np.linalg.norm(fit-[1.2,-.8,13])<1e-8
 noise=.05*np.column_stack((np.sin(np.arange(96)*1.7),np.cos(np.arange(96)*2.1)));fit=fit_circle(p+noise,DEFAULT,1);assert np.linalg.norm(fit[:2]-[1.2,-.8])<.02
 h=np.array([[1.2,.1,100],[.05,.9,200],[.0001,.0002,1.]])
 assert np.max(abs(project(homography(p,project(h,p)),p)-project(h,p)))<1e-8
 return ['exact top-K cycles vs exhaustive tiny graph including wrap closure','candidate/event order invariance and deterministic equal-cost tie','concentric-ring ambiguity retained','strong coherent shadow beats true contour counterexample','missing-angle support rejection','known offset-circle recovery and noise robustness','source homography reconstruction']

def render(inputs,results,out):
 # Source-image mapping is frozen from the prior corpus, not new camera reads.
 image_paths={'measured-grid-720':CAL/'table-grid-20261026-114241.jpg','alternate-grid-720':CAL/'table-grid-2026-10-26.jpg','framing-after-4k':CAL/'stream-4k-verification/after.jpg'}
 for p in image_paths.values():
  if sha(p)!=inputs['image_hashes'][str(p.relative_to(ROOT))]:raise ValueError(f'Source image mismatch {p}')
 images={k:cv2.imread(str(p)) for k,p in image_paths.items()}
 for scene,im in images.items():
  if im is None:raise ValueError(f'Missing source {scene}')
  tiles=[]
  for case,res in zip(inputs['cases'],results):
   if case['scene_id']!=scene:continue
   iteration=next((r for r in res['iterations'] if r['paths']),None)
   if case['bbox'] is not None:b=np.array(case['bbox']);c=(b[:2]+b[2:])/2;side=max(b[2:]-b[:2])*1.7
   elif case['iterations']:c=np.array(case['iterations'][0]['axis_source_px']);side=100
   else:continue
   x0,y0=np.maximum(np.floor(c-side/2).astype(int),0);x1,y1=np.minimum(np.ceil(c+side/2).astype(int),[im.shape[1],im.shape[0]])
   crop=im[y0:y1,x0:x1].copy()
   if iteration:
    for i,p in enumerate(iteration['paths']):
     if 'fit' not in p:continue
     pts=np.array(p['fit']['outline_source_px'])-[x0,y0]
     cv2.polylines(crop,[np.round(pts).astype(np.int32)],True,(0,0,255) if i==0 else (255,180,0),1,cv2.LINE_AA)
     cc=np.round(np.array(p['fit']['center_source_px'])-[x0,y0]).astype(int);cv2.drawMarker(crop,tuple(cc),(255,0,255),cv2.MARKER_CROSS,5,1)
   tile=cv2.resize(crop,(240,240),interpolation=cv2.INTER_NEAREST);tile=cv2.copyMakeBorder(tile,30,0,0,0,cv2.BORDER_CONSTANT,value=(245,245,245))
   cv2.putText(tile,case['ball_id'][:29],(3,19),cv2.FONT_HERSHEY_SIMPLEX,.35,(0,0,0),1);tiles.append(tile)
  if tiles:
   while len(tiles)%3:tiles.append(np.full_like(tiles[0],245))
   canvas=np.vstack([np.hstack(tiles[i:i+3]) for i in range(0,len(tiles),3)])
   cv2.imwrite(str(out/f'{scene}-diagnostic-crops.png'),canvas)

def physical_comparison(inputs,results):
 # Same report-only least-squares homography as the earlier nine-point study.
 # Include the all-nine best-path result only as a hypothetical comparison;
 # do not relabel geometrically rejected paths or baseline unavailable centers.
 spec=importlib.util.spec_from_file_location('center_reference',CAL/'offline-center-outline-audit-20261028.py')
 core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)
 refs=inputs['reference_evidence']['measurements']['points'];ids=[p['id'] for p in refs]
 targets=np.array([[p['center_from_upper_left_mm']['x'],p['center_from_upper_left_mm']['y']] for p in refs])
 seg={p['id']:p['center_px'] for p in inputs['reference_evidence']['segmentation_observations']}
 chosen={r['ball_id']:r for r in results if r['scene_id']=='measured-grid-720'}
 points=[];passed=[]
 for name in ids:
  row=chosen[name];it=row['iterations'][0];fit=it['paths'][0].get('fit') if it['paths'] else None
  if fit is None:return {'status':'insufficient_finite_best_paths','missing':name}
  points.append(fit['center_source_px']);passed.append(fit['geometric_diagnostic_pass'])
 result={'scope':'Hypothetical diagnostic centers, not runtime acceptance or independent physical truth',
  'helper_sha256':sha(CAL/'offline-center-outline-audit-20261028.py'),
  'all_nine_best_path_diagnostic':core.mapping(np.array(points),targets,ids),
  'same_nine_existing_segmentation':core.mapping(np.array([seg[name] for name in ids]),targets,ids),
  'geometrically_rejected_point_ids':[name for name,ok in zip(ids,passed) if not ok],
  'current_refiner':{'accepted_count':0,'mapping':None}}
 selected=np.flatnonzero(passed)
 if len(selected)>=5:
  try:
   result['common_geometric_pass_subset']={'point_ids':[ids[i] for i in selected],
    'cyclic':core.mapping(np.array(points)[selected],targets[selected],[ids[i] for i in selected]),
    'segmentation':core.mapping(np.array([seg[name] for name in ids])[selected],targets[selected],[ids[i] for i in selected]),
    'selection_bias':'Subset selected by geometric diagnostic; reduced spatial coverage and unknown reference uncertainty.'}
  except (ValueError,np.linalg.LinAlgError) as error:result['subset_error']=str(error)
 return result

def run(args):
 inp=json.loads(args.inputs.read_text());out=args.output;out.mkdir(parents=True,exist_ok=True);cfg=dict(DEFAULT)
 results=[]
 for case in inp['cases']:
  row={'ball_id':case['ball_id'],'scene_id':case['scene_id'],'baseline_status':case['baseline']['status'],'baseline_reason':case['baseline'].get('reason'),'iterations':[]}
  for iteration in case['iterations']:
   paths,quality=cyclic_paths(iteration['candidates'],cfg);fits=[]
   for path in paths:
    try:path['fit']=path_fit(path,np.array(iteration['local_to_source_homography']),cfg,case['width'],case['bbox'],iteration['axis_source_px'])
    except (ValueError,np.linalg.LinAlgError) as error:path['fit_error']=str(error)
    fits.append(path)
   centers=[p['fit']['center_source_px'] for p in fits if 'fit' in p]
   spread=None if len(centers)<2 else float(np.linalg.norm(np.array(centers)[:,None]-np.array(centers)[None,:],axis=2).max())
   row['iterations'].append({'iteration':iteration['iteration'],'support':quality,'paths':fits,'topK_fitted_center_max_spread_source_px':spread,'best_second_cost_gap':None if len(paths)<2 else float(paths[1]['cost']-paths[0]['cost'])})
  results.append(row)
 stats={'case_count':len(results),'baseline_accepted':sum(r['baseline_status']=='accepted' for r in results),'iterations':sum(len(r['iterations']) for r in results),'best_path_geometric_diagnostic_pass':sum(bool(r['iterations'] and r['iterations'][0]['paths'] and r['iterations'][0]['paths'][0].get('fit',{}).get('geometric_diagnostic_pass')) for r in results),'production_accepted':0}
 summary={'schema_version':1,'purpose':'Deterministic cyclic search comparison; no production measurement accepted','config':cfg,'input_sha256':sha(args.inputs),'runner_sha256':sha(__file__),'python':platform.python_version(),'numpy':np.__version__,'opencv':cv2.__version__,'self_tests':tests(),'summary':stats,'cases':results,'physical_reference_comparison':physical_comparison(inp,results),'limits':['Top8 paths are exact for the declared additive graph objective only, not all plausible physical contours.','No new boundary sampling or full normal aiming flow; each saved iteration is fixed.','Only geometric diagnostic gates; foreground/background qualification not rerun.','Best cost and determinism do not prove true contour or physical X/Y.',inp.get('purpose', 'Proposal provenance unavailable'),'No source image or runtime/calibration configuration changes.']}
 write(out/'summary.json',summary);render(inp,results,out)
 lines=['# Deterministic cyclic boundary-path comparison — 2026-10-28','','Research only. Exact top-eight paths optimize a declared continuity/gradient objective on frozen source candidates. All original runtime rejections remain unchanged. No point is accepted for production.','',str(stats),'','| Scene / ball | Iteration | Best local RMS px | Best source center | Geometric failures | Top8 center spread source px |','|---|---:|---:|---|---|---:|']
 for row in results:
  if not row['iterations']:lines.append(f"| {row['scene_id']} / {row['ball_id']} | — | — | — | no retained boundary candidates | — |")
  for it in row['iterations']:
   best=it['paths'][0].get('fit') if it['paths'] else None
   lines.append(f"| {row['scene_id']} / {row['ball_id']} | {it['iteration']} | {best['residual_rms_local_px'] if best else '—'} | {best['center_source_px'] if best else '—'} | {best['geometric_diagnostic_failures'] if best else it['support']['reason']} | {it['topK_fitted_center_max_spread_source_px']} |")
 lines+=['','## Interpretation','','The synthetic concentric-circle case has two equally optimal coherent contours. A coherent stronger shadow ring outranks the true silhouette in another counterexample. Thus this search supplies deterministic hypotheses and a test of continuity, not independent silhouette truth. Candidate families beyond the top eight may remain important. Small top-eight center spread is not a calibrated error bound.','', 'The full outputs preserve all eight paths, their sampled points, source projection, circle quality, and per-iteration center spread. Pixel foreground/background support and full iterative aiming are intentionally untested, so a geometric diagnostic pass is not comparable to a fully accepted runtime measurement.','', 'Current refiner accepts zero referenced measurements. This experiment does not fit a physical homography from rejected or diagnostic centers; the existing nine-reference comparison remains a no-fit production result.','', 'Reproduce: `python thoughts/calibration/offline-cyclic-boundary-20261028.py --inputs thoughts/calibration/cyclic-boundary-audit-20261028/inputs.json --output /tmp/cyclic-replay`; run `--self-test` for exact/exhaustive search, ambiguity counterexamples, known-center recovery and support checks. Re-extract compact inputs only from the recorded experiment checkout using `--extract-inputs PATH`. Source hashes are retained in inputs.json.']
 comparison=summary['physical_reference_comparison']
 lines+=['','## Conditional nine-reference comparison','','The all-nine best-path homography is hypothetical and includes geometrically rejected paths. It is not a production map. All nine physical reference uncertainties are unknown.','', '| Diagnostic method | Same-nine LOO RMS mm |','|---|---:|',f"| Cyclic best path, including failed geometric diagnostics | {comparison.get('all_nine_best_path_diagnostic',{}).get('leave_one_out_rms_mm')} |",f"| Earlier segmentation, same nine points | {comparison.get('same_nine_existing_segmentation',{}).get('leave_one_out_rms_mm')} |",'', f"Geometrically rejected IDs in the hypothetical fit: {comparison.get('geometrically_rejected_point_ids')}. Full per-point residuals and common passing-subset comparisons are in JSON; neither reduced residual nor subset selection proves accuracy."]
 (out/'README.md').write_text('\n'.join(lines)+'\n');print(json.dumps(stats));print(json.dumps({k:v.get('leave_one_out_rms_mm') for k,v in comparison.items() if isinstance(v,dict)}))

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--inputs',type=Path,default=CAL/'cyclic-boundary-audit-20261028/inputs.json');p.add_argument('--extract-inputs',type=Path);p.add_argument('--output',type=Path,default=CAL/'cyclic-boundary-audit-20261028');p.add_argument('--self-test',action='store_true');args=p.parse_args()
 if args.self_test:print(json.dumps(tests()));return
 if args.extract_inputs:extract(args.extract_inputs);return
 run(args)
if __name__=='__main__':main()
