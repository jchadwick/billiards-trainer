import json,time,subprocess,hashlib
from pathlib import Path
import numpy as np
import openvino as ov
root=Path('/home/jchadwick/billiards-qualification-20261029');core=ov.Core();available=core.available_devices
x=np.transpose(np.load(root/'measured-input.npy'),(0,3,1,2)).copy()
report={'scope':'saved input comparison, no camera capture','openvino_version':ov.__version__,'available_devices':available,'onnx_sha256':hashlib.sha256((root/'model.onnx').read_bytes()).hexdigest(),'devices':{}}
for device in ['CPU','GPU']:
 if device not in available:report['devices'][device]={'status':'unavailable'};continue
 try:
  start=time.perf_counter();compiled=core.compile_model(str(root/'model.onnx'),device,{'PERFORMANCE_HINT':'LATENCY'});compilems=(time.perf_counter()-start)*1000
  subprocess.run(['systemctl','--user','stop','billiards-trainer.service'],check=True)
  try:
   times=[]
   for i in range(7):
    start=time.perf_counter();result=compiled([x]);y=np.array(result[compiled.output(0)],copy=True);ms=(time.perf_counter()-start)*1000
    if i>=2:times.append(ms)
   assert np.isfinite(y).all();np.save(root/f'openvino-{device}-output.npy',y)
   report['devices'][device]={'status':'success','compile_ms':compilems,'times_ms':times,'p95_ms':float(np.percentile(times,95)),'output_shape':list(y.shape),'execution_devices':str(compiled.get_property('EXECUTION_DEVICES'))}
  finally:subprocess.run(['systemctl','--user','start','billiards-trainer.service'],check=True)
 except Exception as error:report['devices'][device]={'status':'error','error':str(error)}
 (root/'openvino-saved-report.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report))
