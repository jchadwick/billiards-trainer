#!/usr/bin/env bash
set -euo pipefail

root=/home/jchadwick/billiards-qualification-20261029
distribution="$root/distribution-r10-schedfix"
config="$distribution/config-offline-spread16-motion15.json"
mode="${1:-run}"
duration="${2:-60}"
startup_minimum="${3:-15}"
export LD_LIBRARY_PATH="$root/openvino-env/lib/python3.12/site-packages/openvino/libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

case "$mode" in
    seed|run) ;;
    *) printf 'usage: %s [seed|run] [seconds]\n' "$0" >&2; exit 2 ;;
esac

was_active=false
if systemctl --user is-active --quiet billiards-trainer.service; then
    was_active=true
fi
source_pid=0
server_pid=0
cleanup() {
    status=$?
    trap - EXIT INT TERM
    if (( server_pid > 0 )); then
        kill -TERM "$server_pid" 2>/dev/null || true
        wait "$server_pid" 2>/dev/null || true
    fi
    if (( source_pid > 0 )); then
        kill -TERM "$source_pid" 2>/dev/null || true
        wait "$source_pid" 2>/dev/null || true
    fi
    if [[ "$was_active" == true ]]; then
        systemctl --user start billiards-trainer.service
    fi
    exit "$status"
}
trap cleanup EXIT INT TERM

if [[ "$was_active" == true ]]; then
    systemctl --user stop billiards-trainer.service
fi
python3 "$root/billiards-saved-mjpeg-server.py" \
    --jpeg "$root/billiards-measured-upscaled-4k.jpg" \
    --fps 5 --port 18001 > "$root/saved-source-r10-spread16-motion15.log" 2>&1 &
source_pid=$!
BILLIARDS_MOTION_DIAGNOSTICS=1 \
    BILLIARDS_CONFIG="$config" \
    BILLIARDS_HTTP_BIND=127.0.0.1 \
    BILLIARDS_HTTP_PORT=18000 \
    BILLIARDS_SCENE_PORT=18321 \
    "$distribution/run.sh" > "$root/saved-live-r10-spread16-motion15.log" 2>&1 &
server_pid=$!

case "$mode" in
    seed)
        python3 "$root/seed_saved_visibility_reference.py" \
            --port 18322 > "$root/saved-visibility-seed-r10.json"
        ;;
    run)
        python3 "$root/observe_saved_pipeline_r10.py" \
            --seconds "$duration" \
            --startup-minimum "$startup_minimum" \
            --target-precise 15 \
            --output "$root/saved-live-r10-spread16-motion15-observation-${duration}s-min${startup_minimum}"
        ;;
esac
