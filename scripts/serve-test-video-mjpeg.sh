#!/usr/bin/env bash
# Pre-encode a saved clip once, then serve its JPEG frames to multiple clients.
# The Rust fixture shares one latest-frame value, allowing detector/viewer tests
# without FFmpeg decode/encode work inside a measured audit window.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"

video_path="${BILLIARDS_TEST_VIDEO:-${repo_root}/test_data/demo2_short.mp4}"
bind_host="${BILLIARDS_TEST_VIDEO_HOST:-127.0.0.1}"
port="${BILLIARDS_TEST_VIDEO_PORT:-18001}"
fps="${BILLIARDS_TEST_VIDEO_FPS:-}"
quality="${BILLIARDS_TEST_VIDEO_QUALITY:-5}"
maximum_clients="${BILLIARDS_TEST_VIDEO_MAX_CLIENTS:-8}"
maximum_frames="${BILLIARDS_TEST_VIDEO_MAX_FRAMES:-10000}"
maximum_frame_bytes="${BILLIARDS_TEST_VIDEO_MAX_FRAME_BYTES:-16777216}"
maximum_total_bytes="${BILLIARDS_TEST_VIDEO_MAX_TOTAL_BYTES:-536870912}"
shutdown_grace_seconds="${BILLIARDS_TEST_VIDEO_SHUTDOWN_GRACE_SECONDS:-3}"
ffmpeg_bin="${FFMPEG_BIN:-ffmpeg}"
ffprobe_bin="${FFPROBE_BIN:-ffprobe}"
cargo_bin="${CARGO_BIN:-cargo}"
target_dir="${CARGO_TARGET_DIR:-${repo_root}/perception/target}"
if [[ "${target_dir}" != /* ]]; then
  target_dir="${repo_root}/${target_dir}"
fi
fanout_bin="${target_dir}/release/billiards-mjpeg-fanout"
frames_dir=""
server_pid=""

usage() {
  cat <<'EOF'
Serve a saved video as shared, paced HTTP multipart MJPEG for detector and viewer tests.

Usage: scripts/serve-test-video-mjpeg.sh [--help]

Environment:
  BILLIARDS_TEST_VIDEO                 Input video (default: test_data/demo2_short.mp4)
  BILLIARDS_TEST_VIDEO_HOST            IP address to bind (default: 127.0.0.1)
  BILLIARDS_TEST_VIDEO_PORT            HTTP port (default: 18001)
  BILLIARDS_TEST_VIDEO_FPS             Frame cadence (default: source average frame rate)
  BILLIARDS_TEST_VIDEO_QUALITY         FFmpeg JPEG quality, 2-31 (default: 5)
  BILLIARDS_TEST_VIDEO_MAX_CLIENTS     Concurrent subscribers (default: 8)
  BILLIARDS_TEST_VIDEO_MAX_FRAMES      Preloaded frame-count limit, 1..1000000 (default: 10000)
  BILLIARDS_TEST_VIDEO_MAX_FRAME_BYTES Maximum encoded JPEG size (default: 16 MiB)
  BILLIARDS_TEST_VIDEO_MAX_TOTAL_BYTES Maximum preloaded JPEG corpus (default: 512 MiB)
  BILLIARDS_TEST_VIDEO_SHUTDOWN_GRACE_SECONDS Graceful client shutdown (default: 3)
  FFMPEG_BIN                           FFmpeg executable
  FFPROBE_BIN                          FFprobe executable
  CARGO_BIN                            Cargo executable
  CARGO_TARGET_DIR                     Cargo target directory

Frames are decoded and encoded before the server starts. The server then paces
the pre-encoded corpus and publishes a capacity-one latest frame shared by all
clients. Stop with Ctrl+C. This validates software networking and viewer
interference; it does not represent camera exposure time or physical capture.
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi
if (($# > 0)); then
  echo "Unexpected argument: $1" >&2
  usage >&2
  exit 2
fi

if [[ ! -f "${video_path}" || ! -r "${video_path}" ]]; then
  echo "Video file is missing or unreadable: ${video_path}" >&2
  exit 2
fi
for executable in "${ffmpeg_bin}" "${ffprobe_bin}" "${cargo_bin}"; do
  if ! command -v "${executable}" >/dev/null 2>&1; then
    echo "Required executable not found: ${executable}" >&2
    exit 2
  fi
done
if [[ ! "${port}" =~ ^[0-9]+$ ]] || ((port < 1 || port > 65535)); then
  echo "BILLIARDS_TEST_VIDEO_PORT must be an integer from 1 to 65535" >&2
  exit 2
fi
if [[ ! "${quality}" =~ ^[0-9]+$ ]] || ((quality < 2 || quality > 31)); then
  echo "BILLIARDS_TEST_VIDEO_QUALITY must be an integer from 2 to 31" >&2
  exit 2
fi
if [[ ! "${maximum_clients}" =~ ^[0-9]+$ ]] || ((maximum_clients < 1)); then
  echo "BILLIARDS_TEST_VIDEO_MAX_CLIENTS must be a positive integer" >&2
  exit 2
fi
if [[ ! "${maximum_frames}" =~ ^[0-9]+$ ]] || ! awk -v value="${maximum_frames}" 'BEGIN { exit !(value >= 1 && value <= 1000000) }'; then
  echo "BILLIARDS_TEST_VIDEO_MAX_FRAMES must be an integer from 1 to 1000000" >&2
  exit 2
fi
maximum_frames_plus_one="$(awk -v value="${maximum_frames}" 'BEGIN { printf "%.0f", value + 1 }')"
if [[ ! "${maximum_frame_bytes}" =~ ^[0-9]+$ ]] || ((maximum_frame_bytes < 1)); then
  echo "BILLIARDS_TEST_VIDEO_MAX_FRAME_BYTES must be a positive integer" >&2
  exit 2
fi
if [[ ! "${maximum_total_bytes}" =~ ^[0-9]+$ ]] || ((maximum_total_bytes < 1)); then
  echo "BILLIARDS_TEST_VIDEO_MAX_TOTAL_BYTES must be a positive integer" >&2
  exit 2
fi
if [[ ! "${shutdown_grace_seconds}" =~ ^[0-9]+$ ]] || ((shutdown_grace_seconds < 1 || shutdown_grace_seconds > 60)); then
  echo "BILLIARDS_TEST_VIDEO_SHUTDOWN_GRACE_SECONDS must be an integer from 1 to 60" >&2
  exit 2
fi
if [[ ! "${bind_host}" =~ ^[0-9A-Fa-f:.]+$ ]]; then
  echo "BILLIARDS_TEST_VIDEO_HOST must be an IP address" >&2
  exit 2
fi

if [[ -z "${fps}" ]]; then
  frame_rate="$("${ffprobe_bin}" \
    -v error \
    -select_streams v:0 \
    -show_entries stream=avg_frame_rate \
    -of default=noprint_wrappers=1:nokey=1 \
    "${video_path}")"
  fps="$(awk -v value="${frame_rate}" 'BEGIN {
    count = split(value, ratio, "/")
    if (count == 1) {
      numerator = ratio[1]
      denominator = 1
    } else {
      numerator = ratio[1]
      denominator = ratio[2]
    }
    if (numerator !~ /^[0-9]+$/ || denominator !~ /^[0-9]+$/ || denominator == 0) {
      exit 1
    }
    rate = numerator / denominator
    if (rate < 0.1 || rate > 120) exit 1
    printf "%.6f", rate
  }')" || {
    echo "Could not derive a supported frame rate from ${video_path}; set BILLIARDS_TEST_VIDEO_FPS" >&2
    exit 2
  }
fi
if [[ ! "${fps}" =~ ^[0-9]+([.][0-9]+)?$ ]] || ! awk -v value="${fps}" 'BEGIN { exit !(value >= 0.1 && value <= 120) }'; then
  echo "BILLIARDS_TEST_VIDEO_FPS must be between 0.1 and 120" >&2
  exit 2
fi

frames_dir="$(mktemp -d "${TMPDIR:-/tmp}/billiards-test-mjpeg.XXXXXX")"
cleanup() {
  if [[ -n "${server_pid}" ]]; then
    kill -TERM "${server_pid}" 2>/dev/null || true
    wait "${server_pid}" 2>/dev/null || true
    server_pid=""
  fi
  if [[ -n "${frames_dir}" ]]; then
    rm -rf -- "${frames_dir}"
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

echo "Pre-encoding ${video_path} at ${fps} fps before measurement"
"${ffmpeg_bin}" \
  -hide_banner \
  -loglevel error \
  -i "${video_path}" \
  -map 0:v:0 \
  -an \
  -vf "fps=${fps}" \
  -q:v "${quality}" \
  -frames:v "${maximum_frames_plus_one}" \
  -start_number 0 \
  "${frames_dir}/frame-%08d.jpg"

(cd -- "${repo_root}" && "${cargo_bin}" build --release --locked \
  --manifest-path perception/Cargo.toml --bin billiards-mjpeg-fanout)

echo "Serving paced shared-frame MJPEG at http://${bind_host}:${port}/stream.mjpg"
"${fanout_bin}" \
  --frames-dir "${frames_dir}" \
  --bind "${bind_host}" \
  --port "${port}" \
  --fps "${fps}" \
  --max-clients "${maximum_clients}" \
  --max-frames "${maximum_frames}" \
  --max-frame-bytes "${maximum_frame_bytes}" \
  --max-total-bytes "${maximum_total_bytes}" \
  --shutdown-grace-seconds "${shutdown_grace_seconds}" &
server_pid=$!
wait "${server_pid}"
server_pid=""
