#!/usr/bin/env bash
# Compare two explicitly selected YOLO models on the same video or image input.

set -euo pipefail

INPUT=""
REFERENCE_MODEL=""
CANDIDATE_MODEL=""
FRAMES="100"
TOLERANCE="2.0"
OUTPUT_DIR="thoughts/performance-audits/equivalence-tests"
LOG_LEVEL="INFO"
IMGSZ="1280"
CONFIDENCE="0.4"
IOU="0.45"
DEVICE="cpu"

usage() {
    cat <<'EOF'
Usage: tools/run_equivalence_test.sh --video INPUT --reference-model MODEL --candidate-model MODEL [OPTIONS]

INPUT may be a video, one image, or a directory of images.

Options:
  -v, --video PATH             Input video/image/image directory (required)
  -r, --reference-model PATH   Reference model, usually a .pt checkpoint (required)
  -c, --candidate-model PATH   Candidate export/provider model (required)
  -f, --frames N               Maximum decoded frames (default: 100)
  -t, --tolerance PX           Center-distance association radius (default: 2.0)
      --imgsz N                Square inference size (default: 1280)
      --confidence VALUE       Confidence threshold (default: 0.4)
      --iou VALUE              NMS IoU threshold (default: 0.45)
      --device DEVICE          Requested Ultralytics device (default: cpu)
  -o, --output-dir DIR         Report directory (default: thoughts/performance-audits/equivalence-tests)
  -l, --log-level LEVEL        DEBUG, INFO, WARNING, or ERROR (default: INFO)
  -h, --help                   Show this help

The tolerance only associates detections. It is not a production accuracy gate.
The generated report records actual model hashes and observed runtime placement.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -v|--video|--input)
            [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 2; }
            INPUT="$2"
            shift 2
            ;;
        -r|--reference-model)
            [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 2; }
            REFERENCE_MODEL="$2"
            shift 2
            ;;
        -c|--candidate-model)
            [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 2; }
            CANDIDATE_MODEL="$2"
            shift 2
            ;;
        -f|--frames)
            [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 2; }
            FRAMES="$2"
            shift 2
            ;;
        -t|--tolerance)
            [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 2; }
            TOLERANCE="$2"
            shift 2
            ;;
        --imgsz|--confidence|--iou|--device|-o|--output-dir|-l|--log-level)
            [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 2; }
            option="$1"
            value="$2"
            case "$option" in
                --imgsz) IMGSZ="$value" ;;
                --confidence) CONFIDENCE="$value" ;;
                --iou) IOU="$value" ;;
                --device) DEVICE="$value" ;;
                -o|--output-dir) OUTPUT_DIR="$value" ;;
                -l|--log-level) LOG_LEVEL="$value" ;;
            esac
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ -z "$INPUT" || -z "$REFERENCE_MODEL" || -z "$CANDIDATE_MODEL" ]]; then
    echo "Input, reference model, and candidate model are required." >&2
    usage >&2
    exit 2
fi
if [[ ! -e "$INPUT" ]]; then
    echo "Input does not exist: $INPUT" >&2
    exit 2
fi
if [[ ! -e "$REFERENCE_MODEL" ]]; then
    echo "Reference model does not exist: $REFERENCE_MODEL" >&2
    exit 2
fi
if [[ ! -e "$CANDIDATE_MODEL" ]]; then
    echo "Candidate model does not exist: $CANDIDATE_MODEL" >&2
    exit 2
fi

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" ]]; then
    if [[ -x ".venv/bin/python" ]]; then
        PYTHON_BIN=".venv/bin/python"
    else
        PYTHON_BIN="python3"
    fi
fi

exec "$PYTHON_BIN" tools/test_backend_equivalence.py "$INPUT" \
    --reference-model "$REFERENCE_MODEL" \
    --candidate-model "$CANDIDATE_MODEL" \
    --frames "$FRAMES" \
    --tolerance "$TOLERANCE" \
    --imgsz "$IMGSZ" \
    --confidence "$CONFIDENCE" \
    --iou "$IOU" \
    --device "$DEVICE" \
    --output-dir "$OUTPUT_DIR" \
    --log-level "$LOG_LEVEL"
