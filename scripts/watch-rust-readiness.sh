#!/bin/sh
# Poll the Rust gateway's readiness endpoint. This reports localization
# availability and freshness status; it is not a performance profiler.

set -u

usage() {
  cat <<'EOF'
Usage: scripts/watch-rust-readiness.sh [--url URL] [--interval SECONDS] [--timeout SECONDS] [--once]

Poll the Rust gateway readiness endpoint. Defaults can be supplied with
BILLIARDS_READINESS_URL, BILLIARDS_POLL_INTERVAL_SECONDS, and
BILLIARDS_REQUEST_TIMEOUT_SECONDS.
EOF
}

readiness_url=${BILLIARDS_READINESS_URL:-http://127.0.0.1:8001/api/v1/health/ready}
poll_interval=${BILLIARDS_POLL_INTERVAL_SECONDS:-1}
request_timeout=${BILLIARDS_REQUEST_TIMEOUT_SECONDS:-2}
once=false

while [ "$#" -gt 0 ]; do
  case "$1" in
    --url)
      [ "$#" -ge 2 ] || { usage >&2; exit 2; }
      readiness_url=$2
      shift 2
      ;;
    --interval)
      [ "$#" -ge 2 ] || { usage >&2; exit 2; }
      poll_interval=$2
      shift 2
      ;;
    --timeout)
      [ "$#" -ge 2 ] || { usage >&2; exit 2; }
      request_timeout=$2
      shift 2
      ;;
    --once)
      once=true
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      printf 'Unknown argument: %s\n' "$1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

for duration in "$poll_interval" "$request_timeout"; do
  case "$duration" in
    ''|*[!0-9]*)
      printf 'Polling interval and timeout must be positive whole seconds.\n' >&2
      exit 2
      ;;
  esac
  case "$duration" in
    *[1-9]*) ;;
    *)
      printf 'Polling interval and timeout must be positive whole seconds.\n' >&2
      exit 2
      ;;
  esac
done

command -v curl >/dev/null 2>&1 || {
  printf 'curl is required to query the Rust readiness endpoint.\n' >&2
  exit 2
}

while :; do
  if response=$(curl --silent --show-error --max-time "$request_timeout" \
    --write-out '\n__BILLIARDS_HTTP_STATUS__%{http_code}' "$readiness_url" 2>&1); then
    case "$response" in
      *__BILLIARDS_HTTP_STATUS__*)
        body=${response%__BILLIARDS_HTTP_STATUS__*}
        status=${response##*__BILLIARDS_HTTP_STATUS__}
        body=$(printf '%s' "$body" | tr '\012' ' ')
        printf '[%s] readiness HTTP %s: %s\n' "$(date '+%Y-%m-%dT%H:%M:%S%z')" "$status" "$body"
        ;;
      *)
        status=000
        printf '[%s] readiness request returned no HTTP status: %s\n' \
          "$(date '+%Y-%m-%dT%H:%M:%S%z')" "$response"
        ;;
    esac
  else
    status=000
    printf '[%s] readiness request failed: %s\n' \
      "$(date '+%Y-%m-%dT%H:%M:%S%z')" "$response"
  fi

  if [ "$once" = true ]; then
    [ "$status" = 200 ] && exit 0
    exit 1
  fi
  sleep "$poll_interval"
done
