#!/bin/bash
# Quick fix script for GitHub Actions build failures
# Run ID: 18117672492
# Generated: 2025-09-29

set -e  # Exit on any error

BACKEND_DIR="/Users/jchadwick/code/billiards-trainer/backend"
cd "$BACKEND_DIR"

echo "════════════════════════════════════════════════════════════"
echo "  Build Failure Fix Script"
echo "  Fixing import errors and code quality issues"
echo "════════════════════════════════════════════════════════════"
echo ""

# Backup files before modification
BACKUP_DIR="./backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

echo "Creating backups in $BACKUP_DIR..."
cp -r tests/ "$BACKUP_DIR/"
cp system/orchestrator.py "$BACKUP_DIR/"

echo ""
echo "────────────────────────────────────────────────────────────"
echo "Phase 1: Fixing Import Errors (7 issues)"
echo "────────────────────────────────────────────────────────────"
echo ""

# 1.1 Fix api.middleware.auth -> api.middleware.authentication
echo "✓ Fixing api.middleware.auth import..."
sed -i '' 's/from api\.middleware\.auth import AuthMiddleware/from api.middleware.authentication import AuthMiddleware/' tests/unit/test_api.py

# 1.2 Fix PersistenceManager -> ConfigPersistence
echo "✓ Fixing PersistenceManager -> ConfigPersistence..."
sed -i '' 's/from config\.storage\.persistence import PersistenceManager/from config.storage.persistence import ConfigPersistence/' tests/unit/test_config.py
sed -i '' 's/PersistenceManager/ConfigPersistence/g' tests/unit/test_config.py

# 1.3 Fix ShotAssistant -> AssistanceEngine
echo "✓ Fixing ShotAssistant -> AssistanceEngine..."
sed -i '' 's/from core\.analysis\.assistance import ShotAssistant/from core.analysis.assistance import AssistanceEngine\nShotAssistant = AssistanceEngine  # Backward compatibility alias/' tests/unit/test_core.py

# 1.4 Fix projector color calibration import
echo "✓ Fixing projector.calibration.color import..."
sed -i '' 's/from projector\.calibration\.color import ColorCalibrator as ProjectorColorCalibrator/from vision.calibration.color import ColorCalibrator as VisionColorCalibrator/' tests/unit/test_projector.py

# 1.5 Fix CameraFrame import - needs manual investigation
echo "⚠ CameraFrame import needs manual review:"
echo "  File: tests/unit/test_vision.py:13"
echo "  Issue: CameraFrame not found in vision.models"
echo "  Available: FrameStatistics"
echo "  Action: Skipping - needs decision on whether to implement CameraFrame or use FrameStatistics"

# 1.6 Fix Ball -> BallState
echo "✓ Fixing Ball -> BallState in integration tests..."
for file in tests/integration/test_config_core_integration.py tests/integration/test_vision_core_integration.py; do
    if [ -f "$file" ]; then
        sed -i '' 's/from core\.models import Ball/from core.models import BallState\nBall = BallState  # Backward compatibility alias/' "$file"
        sed -i '' 's/\bBall(/BallState(/g' "$file"
    fi
done

# 1.7 Fix BallTracker -> ObjectTracker
echo "✓ Fixing BallTracker -> ObjectTracker in performance tests..."
find tests/performance -name "*.py" -type f -exec sed -i '' 's/from vision\.tracking\.tracker import BallTracker/from vision.tracking.tracker import ObjectTracker\nBallTracker = ObjectTracker  # Backward compatibility alias/' {} \;

echo ""
echo "────────────────────────────────────────────────────────────"
echo "Phase 2: Fixing Code Quality Issues"
echo "────────────────────────────────────────────────────────────"
echo ""

# 2.1 Run black formatter
echo "✓ Running black formatter on system/orchestrator.py..."
python -m black system/orchestrator.py

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  Fix Summary"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "✓ Fixed: api.middleware.auth import"
echo "✓ Fixed: PersistenceManager -> ConfigPersistence"
echo "✓ Fixed: ShotAssistant -> AssistanceEngine"
echo "✓ Fixed: projector.calibration.color import"
echo "⚠ Skipped: CameraFrame (needs manual review)"
echo "✓ Fixed: Ball -> BallState"
echo "✓ Fixed: BallTracker -> ObjectTracker"
echo "✓ Fixed: Black formatting"
echo ""
echo "════════════════════════════════════════════════════════════"
echo ""
echo "Next Steps:"
echo "1. Review CameraFrame issue manually in tests/unit/test_vision.py"
echo "2. Run tests locally:"
echo "   cd $BACKEND_DIR"
echo "   pytest tests/unit/ -v"
echo "3. Check formatting:"
echo "   black --check --diff ."
echo "4. If all passes, commit and push:"
echo "   git add tests/ system/"
echo "   git commit -m 'fix: resolve import errors and formatting issues from CI'"
echo "   git push"
echo ""
echo "Backup created at: $BACKUP_DIR"
echo "To restore: cp -r $BACKUP_DIR/* ."
echo ""
