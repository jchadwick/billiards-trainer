#!/usr/bin/env bash
set -euo pipefail
root=/home/jchadwick/billiards-qualification-20261029
export LD_LIBRARY_PATH="$root/openvino-env/lib/python3.12/site-packages/openvino/libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
systemctl --user stop billiards-trainer.service
trap 'systemctl --user start billiards-trainer.service' EXIT
cd "$root/saved-workload"
../bin-r9/runtime_gpu_batch_replay_20261029 ../target-refiner-gpu-batch-r9-isolated.json
../bin-r9/billiards-perception-video-replay --video measured-5fps.mkv --onnx ../models/billiards-yolov8n.onnx --source-checkpoint /home/jchadwick/apps/billiards-trainer/models/billiards-yolov8n.pt --provider openvino --image-size 1280 --confidence 0.3 --iou 0.45 --max-detections 16 --mode realtime-latest-frame --maximum-frame-count 20 --output-dir ../native-gpu-replay-r9 --ball-localization-settings-json ../billiards-target-settings-r9.json --phase-schedule-json ../billiards-target-phase-r9.json --detection-batches-jsonl batches.jsonl --engine-instance-id target-native-gpu-r9 --source-session-id saved-measured --scene-snapshots-jsonl scenes.jsonl --tracking-ball-class-id 1 --tracking-camera-profile-id saved-mokose-720 --tracking-association-gate-px 18 --tracking-association-prediction-horizon-ms 250 --tracking-lost-after-ms 1000 --tracking-maximum-track-age-ms 3000 --tracking-maximum-active-tracks 32 --tracking-recent-observation-capacity 128 --tracking-configuration-revision 1 --tracking-rectification-revision 0
