#!/usr/bin/env bash
# Serve a paced saved video as HTTP multipart MJPEG for local Rust pipeline tests.
# FFmpeg accepts one client per listener, so the listener restarts after a client
# disconnects. This is development tooling only; production capture stays in Rust.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"

video_path="${BILLIARDS_TEST_VIDEO:-${repo_root}/test_data/demo2_short.mp4}"
bind_host="${BILLIARDS_TEST_VIDEO_HOST:-127.0.0.1}"
port="${BILLIARDS_TEST_VIDEO_PORT:-18001}"
quality="${BILLIARDS_TEST_VIDEO_QUALITY:-5}"
restart_delay="${BILLIARDS_TEST_VIDEO_RESTART_DELAY_SECONDS:-1}"
ffmpeg_bin="${FFMPEG_BIN:-ffmpeg}"

usage() {
  cat <<'EOF'
Serve a paced saved video as HTTP multipart MJPEG for local Rust capture tests.

Usage: scripts/serve-test-video-mjpeg.sh [--help]

Environment:
  BILLIARDS_TEST_VIDEO                    Input file (default: test_data/demo2_short.mp4)
  BILLIARDS_TEST_VIDEO_HOST               Bind host (default: 127.0.0.1)
  BILLIARDS_TEST_VIDEO_PORT               HTTP port (default: 18001)
  BILLIARDS_TEST_VIDEO_QUALITY            MJPEG quality, 2-31 (default: 5)
  BILLIARDS_TEST_VIDEO_RESTART_DELAY_SECONDS  Delay after a client disconnects (default: 1)
  FFMPEG_BIN                              FFmpeg executable (default: ffmpeg)

The stream loops at the source video's recorded frame rate. Stop with Ctrl+C.
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
if ! command -v "${ffmpeg_bin}" >/dev/null 2>&1; then
  echo "FFmpeg executable not found: ${ffmpeg_bin}" >&2
  exit 2
fi
if [[ ! "${port}" =~ ^[0-9]+$ ]] || ((port < 1 || port > 65535)); then
  echo "BILLIARDS_TEST_VIDEO_PORT must be an integer from 1 to 65535" >&2
  exit 2
fi
if [[ ! "${quality}" =~ ^[0-9]+$ ]] || ((quality < 2 || quality > 31)); then
  echo "BILLIARDS_TEST_VIDEO_QUALITY must be an integer from 2 to 31" >&2
  exit 2
fi
if [[ ! "${restart_delay}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  echo "BILLIARDS_TEST_VIDEO_RESTART_DELAY_SECONDS must be a non-negative number" >&2
  exit 2
fi
if [[ ! "${bind_host}" =~ ^[A-Za-z0-9.-]+$ ]]; then
  echo "BILLIARDS_TEST_VIDEO_HOST must be a hostname or IP address" >&2
  exit 2
fi

stopping=0
listener_pid=""
stop() {
  stopping=1
  if [[ -n "${listener_pid}" ]]; then
    kill -TERM "${listener_pid}" 2>/dev/null || true
  fi
}
trap stop INT TERM

echo "Serving ${video_path} at source frame rate on http://${bind_host}:${port}/stream.mjpg"
echo "The listener is intended for local development and accepts one client at a time."

while ((stopping == 0)); do
  listener_status=0
  "${ffmpeg_bin}" \
    -hide_banner \
    -loglevel warning \
    -re \
    -stream_loop -1 \
    -i "${video_path}" \
    -map 0:v:0 \
    -an \
    -c:v mjpeg \
    -q:v "${quality}" \
    -f mpjpeg \
    -boundary_tag frame \
    -listen 1 "http://${bind_host}:${port}/stream.mjpg" &
  listener_pid=$!
  if wait "${listener_pid}"; then
    listener_status=0
  else
    listener_status=$?
  fi
  # A signal trap can interrupt Bash's first wait before the child has exited.
  # Wait again after forwarding TERM so shutdown does not leave FFmpeg behind.
  if ((stopping != 0)); then
    wait "${listener_pid}" || true
  fi
  listener_pid=""

  if ((stopping != 0)); then
    break
  fi
  echo "FFmpeg listener ended with status ${listener_status}; reopening in ${restart_delay}s."
  sleep "${restart_delay}" || true
done
