#!/usr/bin/env bash
set -euo pipefail
root=/home/jchadwick/billiards-qualification-20261029
export LD_LIBRARY_PATH="$root/openvino-env/lib/python3.12/site-packages/openvino/libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
systemctl --user stop billiards-trainer.service
trap 'systemctl --user start billiards-trainer.service' EXIT
cd "$root/saved-workload"
../bin/runtime_batch_replay_20261029 ../target-refiner-batch-r8-isolated.json
../bin/billiards-perception-video-replay --video measured-5fps.mkv --onnx ../models/billiards-yolov8n.onnx --source-checkpoint /home/jchadwick/apps/billiards-trainer/models/billiards-yolov8n.pt --provider openvino --image-size 1280 --confidence 0.3 --iou 0.45 --max-detections 16 --mode realtime-latest-frame --maximum-frame-count 20 --output-dir ../native-gpu-replay-r8 --ball-localization-settings-json thoughts/calibration/runtime-localization-20261029/settings-1280-r8.json --detection-batches-jsonl batches.jsonl --engine-instance-id target-native-gpu-r8 --source-session-id saved-measured --scene-snapshots-jsonl scenes.jsonl --tracking-ball-class-id 1 --tracking-camera-profile-id saved-mokose-720 --tracking-association-gate-px 18 --tracking-association-prediction-horizon-ms 250 --tracking-lost-after-ms 1000 --tracking-maximum-track-age-ms 3000 --tracking-maximum-active-tracks 32 --tracking-recent-observation-capacity 128 --tracking-configuration-revision 1 --tracking-rectification-revision 0
