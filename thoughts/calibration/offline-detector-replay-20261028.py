#!/usr/bin/env python3
"""Replay verified checkpoint on saved pixels; no camera or production changes.

This is the floating-point PyTorch source model on the development CPU, not a
claim of numerical equivalence to the compiled Coral backend. Manual boxes are
crop references only. Video tracking is a small image-space diagnostic, not the
application tracker or validated physical identity/position.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import platform
import time
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
CAL=ROOT/'thoughts/calibration'

@dataclass(frozen=True)
class Config:
    matching_iou: float=.10
    video_sample_hz: float=5.
    video_duration_seconds: float=4.
    association_distance_reference_px: float=18.
    association_reference_width: int=1280


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(1024*1024),b''):h.update(part)
    return h.hexdigest()


def iou(a,b):
    x0,y0=max(a[0],b[0]),max(a[1],b[1]);x1,y1=min(a[2],b[2]),min(a[3],b[3])
    overlap=max(0.,x1-x0)*max(0.,y1-y0)
    union=max(0.,a[2]-a[0])*max(0.,a[3]-a[1])+max(0.,b[2]-b[0])*max(0.,b[3]-b[1])-overlap
    return overlap/union if union>0 else 0.


def match(manual,detections,threshold):
    """Deterministic highest-IoU greedy matching; never a truth assignment."""
    candidates=sorted([(-iou(m['bbox_xyxy'],d['bbox_xyxy_source_px']),m['id'],d['index'])
                       for m in manual for d in detections if d['class_name']=='ball'])
    used_m,used_d=set(),set();pairs=[]
    for negative,mid,did in candidates:
        if -negative < threshold:continue
        if mid in used_m or did in used_d:continue
        used_m.add(mid);used_d.add(did)
        pairs.append({'manual_proposal_id':mid,'detection_index':did,'iou':-negative})
    return {'method':'highest-IoU greedy; not physical identity or detection ground truth',
            'minimum_iou':threshold,'pairs':pairs,
            'unmatched_manual_proposals':[m['id'] for m in manual if m['id'] not in used_m],
            'unmatched_ball_detections':[d['index'] for d in detections if d['class_name']=='ball' and d['index'] not in used_d]}


def source_letterbox_parameters(width,height,size):
    """Test helper documenting square-letterbox geometry; outputs already unpad."""
    ratio=min(size/width,size/height)
    new_w,new_h=round(width*ratio),round(height*ratio)
    dw,dh=(size-new_w)/2,(size-new_h)/2
    return ratio,round(dw-.1),round(dh-.1)


def inverse_letterbox(points,width,height,size):
    ratio,left,top=source_letterbox_parameters(width,height,size)
    return (np.asarray(points,float)-[left,top])/ratio


def frame_identity(image):
    return {'dimensions_px':[image.shape[1],image.shape[0]],
            'decoded_bgr_sha256':hashlib.sha256(image.tobytes()).hexdigest()}


def proposals(result,image):
    assert tuple(result.orig_shape)==image.shape[:2], 'Result does not match source geometry'
    values=result.boxes.data.cpu().numpy()
    assert values.ndim==2 and values.shape[1]==6 and np.isfinite(values).all()
    rows=[]
    for index,(x0,y0,x1,y1,confidence,category) in enumerate(values.tolist()):
        assert 0<=x0<=x1<=image.shape[1] and 0<=y0<=y1<=image.shape[0]
        cls=int(category)
        rows.append({'index':index,'class_id':cls,'class_name':str(result.names[cls]),
                     'confidence':confidence,'bbox_xyxy_source_px':[x0,y0,x1,y1],
                     'raw_bbox_midpoint_source_px':[(x0+x1)/2,(y0+y1)/2]})
    return rows


def associate(previous,current,maximum,next_id):
    """One-step nearest-center association; no semantic ID or predicted position."""
    pairs=[]
    for pi,p in enumerate(previous):
        for ci,c in enumerate(current):
            if p['class_id']!=c['class_id']:continue
            distance=float(np.linalg.norm(np.asarray(p['raw_bbox_midpoint_source_px'])-c['raw_bbox_midpoint_source_px']))
            if distance<=maximum:pairs.append((distance,p['track_id'],pi,ci))
    taken_p,taken_c=set(),set();out=[dict(c) for c in current]
    for distance,tid,pi,ci in sorted(pairs):
        if pi in taken_p or ci in taken_c:continue
        taken_p.add(pi);taken_c.add(ci);out[ci]['track_id']=tid;out[ci]['previous_center_distance_px']=distance
    for ci,c in enumerate(out):
        if ci not in taken_c:c['track_id']=next_id;next_id+=1
    return out,next_id,{'continued':len(taken_c),'new':len(current)-len(taken_c),
                       'missing_previous':len(previous)-len(taken_p)}


def self_test():
    assert iou([0,0,2,2],[1,1,3,3])==1/7
    assert iou([0,0,0,0],[0,0,1,1])==0
    manual=[{'id':'b','bbox_xyxy':[0,0,10,10]},{'id':'a','bbox_xyxy':[20,0,30,10]}]
    detections=[{'index':1,'class_name':'ball','bbox_xyxy_source_px':[0,0,10,10]},
                {'index':0,'class_name':'ball','bbox_xyxy_source_px':[20,0,30,10]}]
    assert match(manual,detections,.1)==match(manual[::-1],detections[::-1],.1)
    xy=np.array([[10,20],[300,170]],float);r,left,top=source_letterbox_parameters(320,180,1280)
    assert np.max(abs(inverse_letterbox(xy*r+[left,top],320,180,1280)-xy))<1e-12
    im=np.zeros((3,4,3),np.uint8); first=frame_identity(im);im[0,0]=1
    assert frame_identity(im)!=first
    base={'class_id':1,'raw_bbox_midpoint_source_px':[10,10]}
    curr=[dict(base)];a,tid,s=associate([],curr,5.,0);assert a[0]['track_id']==0 and tid==1
    b,tid,s=associate(a,[{**base,'raw_bbox_midpoint_source_px':[12,10]}],5.,tid)
    assert b[0]['track_id']==0 and s['continued']==1
    c,tid,s=associate(b,[{**base,'raw_bbox_midpoint_source_px':[30,10]}],5.,tid)
    assert c[0]['track_id']==1 and s['missing_previous']==1
    return {'iou_and_degenerate_boxes':'passed','matching_order_invariance':'passed',
            'source_letterbox_roundtrip':'passed','frame_identity_change':'passed',
            'association_distance_and_identity':'passed'}


def load_scene_inputs():
    corpus=json.loads((CAL/'native-localization-corpus-v1.json').read_text())
    reserved=json.loads((CAL/'reserved-scene-coarse-proposals-20261027.json').read_text())
    measured=next(i for i in corpus['images'] if i['id']=='measured-grid-720')
    scenes=[{'id':measured['id'],'path':measured['path'],'sha256':measured['sha256'],
             'manual':[{'id':p['ball_id'],'bbox_xyxy':p['bbox_xyxy_source_px']} for p in measured['ball_proposals']['proposals']]}]
    for item in reserved['proposal_sets']:
        manual=[]
        for p in item['proposals']:
            x,y=p['center_px_approx'];half=p['bbox_side_px']/2
            manual.append({'id':p['id'],'bbox_xyxy':[x-half,y-half,x+half,y+half]})
        scenes.append({'id':item['image_id'],'path':item['path'],'sha256':item['sha256'],'manual':manual})
    return scenes


def overlay(image,detections,label):
    view=image.copy();scale=max(1.,image.shape[1]/1280.)
    for d in detections:
        x0,y0,x1,y1=map(lambda n:int(round(n)),d['bbox_xyxy_source_px'])
        color=(0,220,255) if d['class_name']=='ball' else (255,100,50)
        cv2.rectangle(view,(x0,y0),(x1,y1),color,max(1,round(scale)))
        text=f"{d.get('track_id',d['index'])}:{d['class_name']} {d['confidence']:.2f}"
        cv2.putText(view,text,(x0,max(12,y0-3)),cv2.FONT_HERSHEY_SIMPLEX,.30*scale,color,max(1,round(scale)))
    cv2.putText(view,label,(12,round(24*scale)),cv2.FONT_HERSHEY_SIMPLEX,.6*scale,(0,0,255),max(1,round(scale)))
    return view


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=CAL/'actual-detector-replay-20261028')
    parser.add_argument('--checkpoint',type=Path,default=ROOT/'dist/models/billiards-yolov8n.pt')
    parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--skip-video',action='store_true',help='Research workload control, not a production backend mode')
    args=parser.parse_args()
    if args.self_test:print(json.dumps(self_test()));return
    import torch,ultralytics
    from ultralytics import YOLO
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    defaults_path=ROOT/'perception/contracts/configuration/v1/defaults.json'
    defaults=json.loads(defaults_path.read_text())['perception']['rust_live']
    manifest_path=ROOT/'dist/models/manifest.json';manifest=json.loads(manifest_path.read_text())
    model_hash=sha(args.checkpoint)
    assert model_hash==manifest['source']['checkpoint']['sha256'], 'Checkpoint differs from verified conversion source'
    model=YOLO(str(args.checkpoint));config=Config();out=args.output;out.mkdir(parents=True,exist_ok=True)
    kwargs={'imgsz':defaults['image_size'],'rect':False,'device':'cpu','conf':defaults['confidence'],
            'iou':defaults['iou'],'max_det':defaults['max_detections'],'verbose':False}
    timings=[]
    def infer(image,identity):
        before=time.perf_counter();result=model.predict(image,**kwargs)[0]
        timings.append({'source_id':identity,'elapsed_seconds':time.perf_counter()-before,'ultralytics_speed_ms':result.speed})
        return proposals(result,image)
    scenes=[]
    for item in load_scene_inputs():
        path=ROOT/item['path'];assert sha(path)==item['sha256'],f"Changed saved image: {path}"
        im=cv2.imread(str(path));assert im is not None
        rows=infer(im,item['id']); repeat=infer(im,item['id']+'-repeat')
        assert rows==repeat, 'Identical source pixels changed actual proposals in this replay'
        scenes.append({'id':item['id'],'source_path':item['path'],'source_sha256':item['sha256'],
                       **frame_identity(im),'detections':rows,'manual_crop_comparison':match(item['manual'],rows,config.matching_iou),
                       'manual_proposals':item['manual'],'identical_pixel_repeat_proposals_equal':True})
        cv2.imwrite(str(out/(item['id']+'-detector.png')),overlay(im,rows,item['id']))
    videos=[];tiles=[]
    for name in ['demo.mp4','demo2.mp4','demo2_short.mp4','demo3.mp4']:
        path=ROOT/'test_data'/name;c=cv2.VideoCapture(str(path));ret,im=c.read()
        assert ret,f'Cannot read saved video {name}'
        meta={'path':str(path.relative_to(ROOT)),'dimensions_px':[int(c.get(cv2.CAP_PROP_FRAME_WIDTH)),int(c.get(cv2.CAP_PROP_FRAME_HEIGHT))],
              'fps':c.get(cv2.CAP_PROP_FPS),'frame_count':int(c.get(cv2.CAP_PROP_FRAME_COUNT)),
              'first_frame_identity':frame_identity(im)}
        c.release();videos.append(meta)
        tile=cv2.resize(im,(640,360));cv2.putText(tile,name,(8,22),cv2.FONT_HERSHEY_SIMPLEX,.65,(0,0,255),2);tiles.append(tile)
    cv2.imwrite(str(out/'video-first-frame-montage.jpg'),np.vstack([np.hstack(tiles[:2]),np.hstack(tiles[2:])]))
    temporal=None
    if not args.skip_video:
        # All first-frame table surfaces were visually reviewed as the same home
        # table. Native4K demo3 starts with rack contact/occlusion: intentionally
        # retain that hard scene rather than cherry-pick a clean still segment.
        path=ROOT/'test_data/demo3.mp4';cap=cv2.VideoCapture(str(path));fps=cap.get(cv2.CAP_PROP_FPS)
        samples=[int(round(n*fps/config.video_sample_hz)) for n in range(math.ceil(config.video_duration_seconds*config.video_sample_hz))]
        target=set(samples);frame_index=0;frames=[];previous=[];next_id=0;last_image=None
        while frame_index<=samples[-1]:
            ret,im=cap.read()
            if not ret:raise ValueError('Saved clip ended before declared bounded workload')
            if frame_index in target:
                rows=[d for d in infer(im,f'demo3-frame-{frame_index}') if d['class_name']=='ball']
                tracks,next_id,association=associate(previous,rows,config.association_distance_reference_px*im.shape[1]/config.association_reference_width,next_id)
                frames.append({'frame_index':frame_index,'source_time_seconds':frame_index/fps,**frame_identity(im),
                               'detections':rows,'diagnostic_tracks':tracks,'association':association})
                if len(frames)==1:cv2.imwrite(str(out/'demo3-first-sampled-frame.png'),overlay(im,tracks,'demo3 frame '+str(frame_index)))
                previous=tracks;last_image=im
            frame_index+=1
        cap.release();assert len(frames)==len(samples)
        cv2.imwrite(str(out/'demo3-last-sampled-frame.png'),overlay(last_image,previous,'demo3 final sampled frame'))
        temporal={'source_path':'test_data/demo3.mp4','source_sha256':sha(path),'source_fps':fps,
                  'declared_sample_hz':config.video_sample_hz,'declared_duration_seconds':config.video_duration_seconds,
                  'frame_selection':'Sequential decode, exact nearest source frame at n / sample_hz; no interpolation',
                  'frames':frames,'association_method':'Greedy same-class nearest source-center, no prediction, no semantic identity',
                  'association_distance_reference_px':config.association_distance_reference_px,
                  'association_reference_width':config.association_reference_width,
                  'missing_event_count':sum(f['association']['missing_previous'] for f in frames),
                  'new_track_event_count':sum(f['association']['new'] for f in frames),
                  'limitation':'Touching rack balls may merge or swap IDs. No physical truth, stationarity label, tracking accuracy, real-time availability or target latency is established.'}
    record={'schema_version':1,'purpose':'Actual source-checkpoint detector proposals from saved pixels; no precise localization or production rollout',
            'model':{'checkpoint_path':str(args.checkpoint.relative_to(ROOT)),'sha256':model_hash,'class_names':model.names,
                     'backend':'PyTorch CPU development-host reference; not compiled Coral numerical parity',
                     'conversion_manifest_sha256':sha(manifest_path)},
            'runtime':{'python':platform.python_version(),'torch':torch.__version__,'ultralytics':ultralytics.__version__,
                       'opencv':cv2.__version__,'numpy':np.__version__,'torch_threads':1,'deterministic_algorithms':True},
            'configuration':{'source_defaults_path':str(defaults_path.relative_to(ROOT)),'source_defaults_sha256':sha(defaults_path),
                             'inference':kwargs,'research':asdict(config)},
            'script_sha256':sha(__file__),'camera_access':False,'production_change':False,'self_tests':self_test(),
            'scenes':scenes,'saved_video_first_frames':videos,'temporal_replay':temporal,
            'limitations':['Manual crop boxes are neither detector truth nor physical center labels.',
                          'Saved videos were used for INT8 calibration; this is not independent heldout detector validation.',
                          'PyTorch source-model proposals may differ from Coral conversion, quantization and deployment preprocessing.',
                          'Source-box midpoint preserves detector geometry; it is not a validated physical ball center.']}
    (out/'results.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    (out/'timings.json').write_text(json.dumps({'scope':'Development-host source-checkpoint timing only; nondeterministic per-run telemetry','records':timings},indent=2)+'\n')
    lines=['# Actual saved-frame detector replay — 2026-10-28','','The verified source checkpoint runs on the development CPU with the configuration defaults: 1280-square input, confidence .3, IoU .45, maximum 16 detections. These are real model proposals rather than the previous manually supplied crops. This does not establish compiled Coral parity, precise ball centers, physical X/Y or target performance.','',
           '| Saved image | Ball detections | Other detections | Manual crop matches | Unmatched manual crops |','|---|---:|---:|---:|---|']
    for s in scenes:
        comparison=s['manual_crop_comparison'];lines.append(f"| {s['id']} | {sum(d['class_name']=='ball' for d in s['detections'])} | {sum(d['class_name']!='ball' for d in s['detections'])} | {len(comparison['pairs'])} | {', '.join(comparison['unmatched_manual_proposals']) or 'none'} |")
    lines+=['','Manual matching is deterministic greedy IoU ≥ .10 against previously reviewed coarse crops; its matches/misses are not independent detection accuracy. All boxes, classes, confidence values, source dimensions, encoded/decoded hashes and exact midpoints are saved. Identical-pixel repeated inference produced identical proposal lists on all three stills.','']
    if temporal:
        counts=[len(f['detections']) for f in temporal['frames']]
        lines+=['## Bounded saved native4K video replay','',f"The first {config.video_duration_seconds:g} seconds of demo3 are sampled at {config.video_sample_hz:g} Hz ({len(counts)} source frames). First-frame montage confirms the same home table, with touching rack balls; no clean segment was selected after scoring. Ball counts are `{counts}`. The simple image-space association records {temporal['missing_event_count']} missing-previous and {temporal['new_track_event_count']} new-track events (including initial births). IDs may swap in the rack; no semantic identity or temporal accuracy is claimed.",'', 'The selected clip and the other saved clips were used during INT8 representative calibration, so these frames are not independent heldout model data. Sampling at 5 Hz is a replay workload, not proof that the deployed system can sustain 5 FPS. The input video is preserved locally and its SHA-256 and sampled decoded-pixel hashes are recorded.']
    lines+=['','## Reproduce','', 'Use `/tmp/billiards-detector-replay-env/bin/python thoughts/calibration/offline-detector-replay-20261028.py --self-test`, then the same script with `--output /tmp/actual-detector-replay-check`. The checkpoint/video dependencies are intentionally not copied into research commits. The exact checkpoint hash must match the local conversion manifest. The compact `results.json` preserves actual proposals for later refiner/contour replay. `timings.json` records variable development-host telemetry separately. No camera, model training, runtime configuration or deployment was used.','']
    (out/'README.md').write_text('\n'.join(lines))
    print(json.dumps({'scenes':[(s['id'],len(s['detections']),len(s['manual_crop_comparison']['pairs'])) for s in scenes],
                      'video_samples':len(temporal['frames']) if temporal else 0,'results':str(out/'results.json')}))

if __name__=='__main__':main()
