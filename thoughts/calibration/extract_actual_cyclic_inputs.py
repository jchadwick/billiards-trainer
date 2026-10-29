"""Extract actual-detector refiner traces for deterministic contour comparison."""
import importlib.util,json,hashlib,subprocess
from pathlib import Path
from collections import defaultdict
import numpy as np
ROOT=Path(__file__).resolve().parents[2];CAL=ROOT/'thoughts/calibration'
spec=importlib.util.spec_from_file_location('cyclic',CAL/'offline-cyclic-boundary-20261028.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
data=json.loads((CAL/'cyclic-boundary-audit-20261028/inputs.json').read_text());data['cases']=[];data['source_hashes']={};data['purpose']='Actual PyTorch detector boxes, frozen refiner candidate replay; no precise measurement accepted'
for path in sorted((CAL/'actual-detector-refiner-20261028/cases').glob('*.json')):
 d=json.loads(path.read_text());groups=defaultdict(list);axes={}
 assert d['trace']['dropped_events']==0
 for e in d['trace']['events']:
  v=e['data'];i=e['iteration']
  if v['kind']=='iteration_started':axes[i]=v['axis_image_px']
  if v['kind']=='boundary_candidate':groups[i].append({'ray':v['ray'],'local':v['point_local_px'],'source':v['point_image_px'],'strength':v['strength']})
 its=[]
 for i,cs in sorted(groups.items()):
  if len(cs)<4:continue
  h=m.homography(np.array([v['local'] for v in cs]),np.array([v['source'] for v in cs]));err=float(np.linalg.norm(m.project(h,[v['local'] for v in cs])-np.array([v['source'] for v in cs]),axis=1).max());assert err<1e-6
  its.append({'iteration':i,'axis_source_px':axes[i],'local_to_source_homography':h.tolist(),'projection_reconstruction_max_px':err,'candidates':m.canonical(cs)})
 data['cases'].append({'ball_id':d['ball_id'],'scene_id':'measured-grid-720','source_case':str(path.relative_to(ROOT)),'width':1280,'bbox':d['proposal']['bbox_xyxy_source_px'],'baseline':d['refinement'],'iterations':its});data['source_hashes'][str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
assert len(data['cases'])==15
data['extraction_revision']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();out=CAL/'actual-cyclic-boundary-20261028';out.mkdir(exist_ok=True);(out/'inputs.json').write_text(json.dumps(data,sort_keys=True)+'\n')
