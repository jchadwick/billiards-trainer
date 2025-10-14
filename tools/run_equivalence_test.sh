#!/bin/bash
# Convenience script to run backend equivalence test

set -e

# Default values
VIDEO=""
FRAMES=100
TOLERANCE=2.0
OUTPUT_DIR="./equivalence-test-results"
LOG_LEVEL="INFO"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -v|--video)
            VIDEO="$2"
            shift 2
            ;;
        -f|--frames)
            FRAMES="$2"
            shift 2
            ;;
        -t|--tolerance)
            TOLERANCE="$2"
            shift 2
            ;;
        -o|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -l|--log-level)
            LOG_LEVEL="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 --video VIDEO_FILE [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -v, --video FILE        Video file to test (required)"
            echo "  -f, --frames N          Max frames to test (default: 100)"
            echo "  -t, --tolerance T       Position tolerance in pixels (default: 2.0)"
            echo "  -o, --output-dir DIR    Output directory (default: ./equivalence-test-results)"
            echo "  -l, --log-level LEVEL   Logging level (default: INFO)"
            echo "  -h, --help              Show this help message"
            echo ""
            echo "Example:"
            echo "  $0 --video demo.mkv --frames 200 --tolerance 3.0"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Check required arguments
if [ -z "$VIDEO" ]; then
    echo "Error: --video argument is required"
    echo "Use --help for usage information"
    exit 1
fi

# Check video file exists
if [ ! -f "$VIDEO" ]; then
    echo "Error: Video file not found: $VIDEO"
    exit 1
fi

# Run test
echo "========================================"
echo "Backend Equivalence Test"
echo "========================================"
echo "Video: $VIDEO"
echo "Max Frames: $FRAMES"
echo "Tolerance: ${TOLERANCE}px"
echo "Output Dir: $OUTPUT_DIR"
echo "Log Level: $LOG_LEVEL"
echo "========================================"
echo ""

python tools/test_backend_equivalence.py "$VIDEO" \
    --frames "$FRAMES" \
    --tolerance "$TOLERANCE" \
    --output-dir "$OUTPUT_DIR" \
    --log-level "$LOG_LEVEL"

exit_code=$?

echo ""
echo "========================================"
if [ $exit_code -eq 0 ]; then
    echo "✓ Test completed successfully"
else
    echo "✗ Test failed or found significant differences"
fi
echo "========================================"

exit $exit_code
