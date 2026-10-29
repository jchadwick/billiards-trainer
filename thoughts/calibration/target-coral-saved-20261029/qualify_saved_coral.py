import json,time,hashlib
from pathlib import Path
import numpy as np
from ai_edge_litert.interpreter import Interpreter,load_delegate
root=Path('/home/jchadwick/billiards-qualification-20261029')
model=Path('/home/jchadwick/apps/billiards-trainer/models/billiards-yolov8n_edgetpu.tflite')
x=np.load(root/'measured-input.npy')
i=Interpreter(model_path=str(model),experimental_delegates=[load_delegate('libedgetpu.so.1')]);i.allocate_tensors()
a=i.get_input_details()[0];b=i.get_output_details()[0]
assert list(x.shape)==a['shape'].tolist() and x.dtype==a['dtype']
times=[]
for step in range(22):
 start=time.perf_counter();i.set_tensor(a['index'],x);i.invoke();y=i.get_tensor(b['index']);elapsed=(time.perf_counter()-start)*1000
 if step>=2:times.append(elapsed)
assert np.isfinite(y).all()
np.save(root/'measured-output.npy',y)
indices=np.argsort(y[0,5])[-16:][::-1]
report={'scope':'saved JPEG input only; no camera capture','model_sha256':hashlib.sha256(model.read_bytes()).hexdigest(),'input_shape':list(x.shape),'output_shape':list(y.shape),'times_ms':times,'p95_ms':float(np.percentile(times,95)),'ball_top16':[{'index':int(j),'xywh':y[0,:4,j].tolist(),'confidence':float(y[0,5,j])} for j in indices]}
(root/'coral-saved-report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
