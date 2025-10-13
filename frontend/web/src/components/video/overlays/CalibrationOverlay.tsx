/**
 * Calibration overlay component - displays table calibration data when available
 */

import React from 'react';
import type { CalibrationData, CoordinateTransform, OverlayConfig, Point2D } from '../../../types/video';

interface CalibrationOverlayProps {
  calibrationData: CalibrationData | null;
  transform: CoordinateTransform;
  config: OverlayConfig['calibration'];
  ctx: CanvasRenderingContext2D;
}

/**
 * Renders calibration overlay showing the calibration points and boundaries
 * Only displays when calibration data exists and is valid
 */
export function renderCalibrationOverlay({
  calibrationData,
  transform,
  config,
  ctx,
}: CalibrationOverlayProps): void {
  // Don't render if not visible, no data, or data is invalid
  if (!config.visible || !calibrationData || !calibrationData.isValid) {
    return;
  }

  ctx.save();
  ctx.globalAlpha = config.opacity;

  const corners = calibrationData.corners;

  // Draw calibration boundary lines
  if (config.showLines && corners.length >= 4) {
    ctx.strokeStyle = '#FF00FF'; // Magenta for calibration
    ctx.lineWidth = config.lineWidth;
    ctx.setLineDash([8, 4]); // Dashed line to distinguish from table detection

    ctx.beginPath();
    const firstCorner = transform.videoToCanvas(corners[0].screenPosition);
    ctx.moveTo(firstCorner.x, firstCorner.y);

    // Draw lines connecting all corners
    for (let i = 1; i < corners.length; i++) {
      const corner = transform.videoToCanvas(corners[i].screenPosition);
      ctx.lineTo(corner.x, corner.y);
    }

    // Close the path
    ctx.closePath();
    ctx.stroke();
    ctx.setLineDash([]); // Reset dash pattern
  }

  // Draw calibration corner points
  if (config.showPoints && corners.length > 0) {
    corners.forEach((corner, index) => {
      const cornerPoint = transform.videoToCanvas(corner.screenPosition);

      // Outer circle (glow effect)
      ctx.fillStyle = 'rgba(255, 0, 255, 0.3)';
      ctx.beginPath();
      ctx.arc(cornerPoint.x, cornerPoint.y, 12, 0, 2 * Math.PI);
      ctx.fill();

      // Main corner point
      ctx.fillStyle = '#FF00FF';
      ctx.strokeStyle = '#FFFFFF';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(cornerPoint.x, cornerPoint.y, 7, 0, 2 * Math.PI);
      ctx.fill();
      ctx.stroke();

      // Inner dot
      ctx.fillStyle = '#FFFFFF';
      ctx.beginPath();
      ctx.arc(cornerPoint.x, cornerPoint.y, 2, 0, 2 * Math.PI);
      ctx.fill();

      // Corner label
      if (config.showLabels) {
        ctx.fillStyle = '#FF00FF';
        ctx.strokeStyle = '#000000';
        ctx.lineWidth = 3;
        ctx.font = 'bold 14px Arial';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';

        // Draw text with outline for better visibility
        const label = `C${index + 1}`;
        ctx.strokeText(label, cornerPoint.x, cornerPoint.y - 20);
        ctx.fillText(label, cornerPoint.x, cornerPoint.y - 20);

        // Show confidence if available
        if (corner.confidence !== undefined && corner.confidence > 0) {
          ctx.font = '10px Arial';
          const confidenceText = `${Math.round(corner.confidence * 100)}%`;
          ctx.strokeText(confidenceText, cornerPoint.x, cornerPoint.y - 32);
          ctx.fillText(confidenceText, cornerPoint.x, cornerPoint.y - 32);
        }
      }
    });
  }

  // Draw calibration accuracy indicator
  if (config.showAccuracy && calibrationData.accuracy !== undefined) {
    const firstCorner = corners.length > 0 ? transform.videoToCanvas(corners[0].screenPosition) : { x: 20, y: 20 };

    // Background box
    const boxWidth = 150;
    const boxHeight = 50;
    const boxX = firstCorner.x + 30;
    const boxY = firstCorner.y;

    ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
    ctx.fillRect(boxX, boxY, boxWidth, boxHeight);

    ctx.strokeStyle = '#FF00FF';
    ctx.lineWidth = 2;
    ctx.strokeRect(boxX, boxY, boxWidth, boxHeight);

    // Title
    ctx.fillStyle = '#FF00FF';
    ctx.font = 'bold 12px Arial';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'top';
    ctx.fillText('Calibration', boxX + 8, boxY + 6);

    // Accuracy value
    ctx.fillStyle = '#FFFFFF';
    ctx.font = '11px Arial';
    const accuracyPercent = Math.round(calibrationData.accuracy * 100);
    const accuracyColor = accuracyPercent >= 90 ? '#00FF00' : accuracyPercent >= 70 ? '#FFFF00' : '#FF6666';
    ctx.fillStyle = accuracyColor;
    ctx.fillText(`Accuracy: ${accuracyPercent}%`, boxX + 8, boxY + 24);

    // Timestamp
    if (calibrationData.calibratedAt) {
      const date = new Date(calibrationData.calibratedAt);
      const timeStr = date.toLocaleTimeString();
      ctx.fillStyle = '#CCCCCC';
      ctx.font = '9px Arial';
      ctx.fillText(`@ ${timeStr}`, boxX + 8, boxY + 38);
    }
  }

  // Draw calibration info badge in corner
  if (calibrationData.isValid) {
    const badgeX = 10;
    const badgeY = 10;
    const badgeWidth = 140;
    const badgeHeight = 28;

    // Semi-transparent background
    ctx.fillStyle = 'rgba(0, 0, 0, 0.6)';
    ctx.fillRect(badgeX, badgeY, badgeWidth, badgeHeight);

    // Border
    ctx.strokeStyle = '#FF00FF';
    ctx.lineWidth = 2;
    ctx.strokeRect(badgeX, badgeY, badgeWidth, badgeHeight);

    // Text
    ctx.fillStyle = '#FF00FF';
    ctx.font = 'bold 11px Arial';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';
    ctx.fillText('✓ Calibrated', badgeX + 10, badgeY + 14);

    // Point count
    ctx.fillStyle = '#FFFFFF';
    ctx.font = '10px Arial';
    ctx.textAlign = 'right';
    ctx.fillText(`${corners.length} pts`, badgeX + badgeWidth - 10, badgeY + 14);
  }

  // Draw center crosshair if we have all 4 corners
  if (corners.length >= 4) {
    // Calculate center point
    let centerX = 0;
    let centerY = 0;
    corners.forEach(corner => {
      centerX += corner.screenPosition.x;
      centerY += corner.screenPosition.y;
    });
    centerX /= corners.length;
    centerY /= corners.length;

    const center = transform.videoToCanvas({ x: centerX, y: centerY });

    // Draw crosshair
    ctx.strokeStyle = '#FF00FF';
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);

    const crosshairSize = 15;
    ctx.beginPath();
    ctx.moveTo(center.x - crosshairSize, center.y);
    ctx.lineTo(center.x + crosshairSize, center.y);
    ctx.moveTo(center.x, center.y - crosshairSize);
    ctx.lineTo(center.x, center.y + crosshairSize);
    ctx.stroke();

    // Center dot
    ctx.fillStyle = '#FF00FF';
    ctx.setLineDash([]);
    ctx.beginPath();
    ctx.arc(center.x, center.y, 3, 0, 2 * Math.PI);
    ctx.fill();
  }

  // Draw dimensions if available
  if (config.showLabels && calibrationData.dimensions) {
    const { width, height } = calibrationData.dimensions;
    if (corners.length >= 4) {
      // Calculate center for dimension display
      let centerX = 0;
      let centerY = 0;
      corners.forEach(corner => {
        centerX += corner.screenPosition.x;
        centerY += corner.screenPosition.y;
      });
      centerX /= corners.length;
      centerY /= corners.length;

      const center = transform.videoToCanvas({ x: centerX, y: centerY });

      // Display dimensions below center
      ctx.fillStyle = '#FF00FF';
      ctx.strokeStyle = '#000000';
      ctx.lineWidth = 3;
      ctx.font = 'bold 12px Arial';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'top';

      const dimensionText = `${width.toFixed(1)}m × ${height.toFixed(1)}m`;
      ctx.strokeText(dimensionText, center.x, center.y + 20);
      ctx.fillText(dimensionText, center.x, center.y + 20);
    }
  }

  ctx.restore();
}

// React wrapper component for testing/isolation
interface CalibrationOverlayComponentProps {
  calibrationData: CalibrationData | null;
  transform: CoordinateTransform;
  config: OverlayConfig['calibration'];
}

export const CalibrationOverlayComponent: React.FC<CalibrationOverlayComponentProps> = ({
  calibrationData,
  transform,
  config,
}) => {
  const canvasRef = React.useRef<HTMLCanvasElement>(null);

  React.useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    renderCalibrationOverlay({ calibrationData, transform, config, ctx });
  }, [calibrationData, transform, config]);

  return (
    <canvas
      ref={canvasRef}
      className="absolute inset-0 pointer-events-none"
      style={{ zIndex: 2 }}
    />
  );
};
