"""Saved-pixel annulus geometry/color audit. No runtime edits or acceptance."""
from pathlib import Path
import json
import cv2
import numpy as np
from audit import ROOT, BASE, ASSET, HERE, project


def main():
    mapping = np.array(json.loads(ASSET.read_text())['image_to_table_mm'])
    image = cv2.cvtColor(cv2.imread(str(ROOT/'thoughts/calibration/table-grid-20261026-114241.jpg')), cv2.COLOR_BGR2RGB).astype(float)/255
    summary = json.loads((BASE/'summary.json').read_text())
    cfg = summary['runtime_parameters']
    cfg = dict(cfg)
    scale = image.shape[1] / cfg['reference_width_px']
    for key in ['gradient_step_px', 'background_probe_px']:
        cfg[key] *= scale
    cfg['patch_size_px'] = int(np.ceil(cfg['patch_size_px'] * scale))
    output=[]
    for name in ['bottom-right-pocket-ball','bottom-left-pocket-ball','top-left-pocket-ball']:
        data=json.loads((BASE/'cases'/f'{name}.json').read_text())
        ki=data['intrinsics']; k=np.array([[ki['fx'],0,ki['cx']],[0,ki['fy'],ki['cy']],[0,0,1]])
        left,top,right,bottom=data['proposal']['bbox_xyxy_source_px']; axis=np.array([(left+right)/2,(top+bottom)/2])
        half=max(right-left,bottom-top)*cfg['roi_padding_factor']/2
        d=np.linalg.inv(k)@np.r_[axis,1];d/=np.linalg.norm(d)
        v=np.array([d[1],-d[0],0]);skew=np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
        rotation=np.eye(3)+skew+skew@skew/(1+d[2])
        local_h=np.diag([ki['fx'],ki['fy'],1])@rotation@np.linalg.inv(k)
        source_h=np.linalg.inv(local_h)
        edges=[[left,axis[1]],[right,axis[1]],[axis[0],top],[axis[0],bottom]]
        proposal_radius=float(np.median([np.linalg.norm(project(local_h,p)) for p in edges]))
        q=np.linalg.inv(k)@np.linalg.inv(mapping);q/=(np.linalg.norm(q[:,0])+np.linalg.norm(q[:,1]))/2
        c=q@np.r_[project(mapping,axis),1]
        physical_radius=(ki['fx']+ki['fy'])/2*28.575/np.sqrt(c@c-28.575**2)
        row=dict(ball_id=name,proposal_local_radius=proposal_radius,physical_local_radius=float(physical_radius),roi_half=half,runtime_quality=data['refinement']['quality'],sampling=[])
        def sample(point):
            p=project(source_h,point); x,y=p
            if max(abs(point))>(cfg['patch_size_px']-1)/2:return None,'patch'
            if constrain_roi and (np.any(p<axis-half) or np.any(p>axis+half)):return None,'roi'
            if x<0 or y<0 or x>=image.shape[1]-1 or y>=image.shape[0]-1:return None,'frame'
            ix,iy=int(x),int(y);wx,wy=x-ix,y-iy
            return (image[iy,ix]*(1-wx)*(1-wy)+image[iy,ix+1]*wx*(1-wy)+image[iy+1,ix]*(1-wx)*wy+image[iy+1,ix+1]*wx*wy),'valid'
        for label,radius,constrain_roi in [('proposal',proposal_radius,True),('proposal_annulus_roi_expanded',proposal_radius,False),('physical_sampling_only',physical_radius,True)]:
            pixels=[];counts={};angular=[]
            for ray in range(cfg['angular_samples']):
                angle=2*np.pi*ray/cfg['angular_samples']
                for factor in [cfg['annulus_inner_factor'],cfg['annulus_outer_factor']]:
                    pixel,why=sample(np.array([np.cos(angle),np.sin(angle)])*radius*factor)
                    counts[why]=counts.get(why,0)+1
                    if pixel is not None:pixels.append(pixel);angular.append(ray)
            pixels=np.array(pixels);median=np.median(pixels,axis=0)
            distances=np.sqrt(np.mean((pixels-median)**2,axis=1));supported=distances<=cfg['maximum_outer_background_distance']
            # Deterministic two-cluster comparison only, no production proposal.
            seeds=[pixels[np.argmin(np.mean(pixels,axis=1))],pixels[np.argmax(np.mean(pixels,axis=1))]]
            centers=np.array(seeds)
            for _ in range(20):
                membership=np.argmin(((pixels[:,None,:]-centers[None,:,:])**2).sum(axis=2),axis=1)
                next_centers=np.array([np.median(pixels[membership==i],axis=0) if np.any(membership==i) else centers[i] for i in range(2)])
                if np.max(abs(next_centers-centers))<1e-10:break
                centers=next_centers
            mixed=np.sqrt(np.min(np.mean((pixels[:,None,:]-centers[None,:,:])**2,axis=2),axis=1))<=cfg['maximum_outer_background_distance']
            # Count actual radial transitions under the exact production tests.
            stages=dict(sampled=0,gradient=0,foreground=0,outer_background=0,direction=0);rays=set()
            for ray in range(cfg['angular_samples']):
                direction=np.array([np.cos(2*np.pi*ray/cfg['angular_samples']),np.sin(2*np.pi*ray/cfg['angular_samples'])])
                for rad in np.linspace(radius*cfg['radius_min_fraction'],radius*cfg['radius_max_fraction'],cfg['radial_samples']):
                    samples=[sample(direction*(rad+offset))[0] for offset in [-cfg['background_probe_px'],cfg['background_probe_px'],-cfg['gradient_step_px'],cfg['gradient_step_px']]]
                    if any(p is None for p in samples):continue
                    inner,outer,a,b=samples;stages['sampled']+=1
                    dist=lambda a,b:float(np.sqrt(np.mean((a-b)**2)))
                    if dist(a,b)<cfg['minimum_gradient']:continue
                    stages['gradient']+=1
                    if dist(inner,median)<cfg['minimum_foreground_background_contrast']:continue
                    stages['foreground']+=1
                    if dist(outer,median)>cfg['maximum_outer_background_distance']:continue
                    stages['outer_background']+=1
                    if dist(outer,median)>dist(inner,median):continue
                    stages['direction']+=1;rays.add(ray)
            row['sampling'].append(dict(seed=label,radius=float(radius),validity_counts=counts,median_rgb=median.tolist(),supported_count=int(sum(supported)),supported_fraction=float(sum(supported)/192),unsupported_rays=[int(x) for x in np.array(angular)[~supported]],two_cluster_centers_rgb=centers.tolist(),two_cluster_supported_count=int(sum(mixed)),radial_test_counts=stages,supported_ray_count=len(rays)))
        output.append(row)
    (HERE/'annulus-report.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(output,indent=2))

if __name__=='__main__':main()
