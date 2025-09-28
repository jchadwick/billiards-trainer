/**
 * Cue stick detection overlay component
 */

import React from 'react';
import type { CueStick, CoordinateTransform, OverlayConfig } from '../../../types/video';

interface CueOverlayProps {
  cue: CueStick | null;
  transform: CoordinateTransform;
  config: OverlayConfig['cue'];
  ctx: CanvasRenderingContext2D;
}

export function renderCueOverlay({ cue, transform, config, ctx }: CueOverlayProps): void {
  if (!config.visible || !cue || !cue.detected) return;

  ctx.save();
  ctx.globalAlpha = config.opacity;

  const tipPoint = transform.videoToCanvas(cue.tipPosition);
  const tailPoint = transform.videoToCanvas(cue.tailPosition);

  // Draw main cue stick
  ctx.strokeStyle = '#FF8800';
  ctx.lineWidth = config.lineWidth;
  ctx.lineCap = 'round';

  ctx.beginPath();
  ctx.moveTo(tipPoint.x, tipPoint.y);
  ctx.lineTo(tailPoint.x, tailPoint.y);
  ctx.stroke();

  // Draw cue tip
  ctx.fillStyle = '#FF0000';
  ctx.strokeStyle = '#000000';
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.arc(tipPoint.x, tipPoint.y, 6, 0, 2 * Math.PI);
  ctx.fill();
  ctx.stroke();

  // Draw cue butt
  ctx.fillStyle = '#8B4513';
  ctx.strokeStyle = '#000000';
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.arc(tailPoint.x, tailPoint.y, 4, 0, 2 * Math.PI);
  ctx.fill();
  ctx.stroke();

  // Draw cue grip area (textured section)
  const gripStart = 0.7; // 70% along the cue from tip to tail
  const gripStartPoint = {
    x: tipPoint.x + (tailPoint.x - tipPoint.x) * gripStart,
    y: tipPoint.y + (tailPoint.y - tipPoint.y) * gripStart,
  };

  ctx.strokeStyle = '#654321';
  ctx.lineWidth = config.lineWidth + 2;
  ctx.beginPath();
  ctx.moveTo(gripStartPoint.x, gripStartPoint.y);
  ctx.lineTo(tailPoint.x, tailPoint.y);
  ctx.stroke();

  // Draw angle information if enabled
  if (config.showAngle) {
    const angleText = `${Math.round(cue.angle * 180 / Math.PI)}°`;
    const elevationText = `Elev: ${Math.round(cue.elevation * 180 / Math.PI)}°`;

    ctx.fillStyle = '#000000';
    ctx.font = '12px Arial';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'bottom';

    // Angle text
    ctx.fillText(angleText, tipPoint.x + 15, tipPoint.y - 10);

    // Elevation text
    ctx.font = '10px Arial';
    ctx.fillText(elevationText, tipPoint.x + 15, tipPoint.y + 5);

    // Visual angle indicator (arc)
    const referenceAngle = 0; // Horizontal reference
    const arcRadius = 25;

    ctx.strokeStyle = '#FF8800';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.arc(
      tipPoint.x,
      tipPoint.y,
      arcRadius,
      referenceAngle,
      cue.angle
    );
    ctx.stroke();

    // Angle arc label
    const midAngle = (referenceAngle + cue.angle) / 2;
    const labelX = tipPoint.x + Math.cos(midAngle) * (arcRadius + 10);
    const labelY = tipPoint.y + Math.sin(midAngle) * (arcRadius + 10);

    ctx.fillStyle = '#FF8800';
    ctx.font = '8px Arial';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('θ', labelX, labelY);
  }

  // Draw guide lines if enabled
  if (config.showGuideLines) {
    // Extension line showing cue direction
    const extensionLength = 200;
    const extendedTip = {
      x: cue.tipPosition.x + Math.cos(cue.angle) * extensionLength,
      y: cue.tipPosition.y + Math.sin(cue.angle) * extensionLength,
    };
    const extendedPoint = transform.videoToCanvas(extendedTip);

    // Dashed line for prediction
    ctx.strokeStyle = '#FF8800';
    ctx.lineWidth = 2;
    ctx.setLineDash([10, 5]);
    ctx.beginPath();
    ctx.moveTo(tipPoint.x, tipPoint.y);
    ctx.lineTo(extendedPoint.x, extendedPoint.y);
    ctx.stroke();
    ctx.setLineDash([]);

    // Draw aim guide dots along the line
    const dotSpacing = 30;
    const numDots = Math.floor(extensionLength / dotSpacing);

    for (let i = 1; i <= numDots; i++) {
      const dotPosition = {
        x: cue.tipPosition.x + Math.cos(cue.angle) * (dotSpacing * i),
        y: cue.tipPosition.y + Math.sin(cue.angle) * (dotSpacing * i),
      };
      const dotPoint = transform.videoToCanvas(dotPosition);

      ctx.fillStyle = '#FF8800';
      const dotSize = Math.max(1, 3 - (i * 0.3)); // Decreasing size
      ctx.beginPath();
      ctx.arc(dotPoint.x, dotPoint.y, dotSize, 0, 2 * Math.PI);
      ctx.fill();
    }

    // Power/force indicator (if available)
    if ((cue as any).force !== undefined) {
      const force = (cue as any).force;
      const powerBarWidth = 60;
      const powerBarHeight = 8;
      const powerBarX = tipPoint.x + 20;
      const powerBarY = tipPoint.y + 20;

      // Power bar background
      ctx.fillStyle = '#333333';
      ctx.fillRect(powerBarX, powerBarY, powerBarWidth, powerBarHeight);

      // Power bar fill
      const fillWidth = powerBarWidth * Math.min(force, 1.0);
      const powerColor = force > 0.8 ? '#FF0000' : force > 0.5 ? '#FFAA00' : '#00FF00';
      ctx.fillStyle = powerColor;
      ctx.fillRect(powerBarX, powerBarY, fillWidth, powerBarHeight);

      // Power bar border
      ctx.strokeStyle = '#000000';
      ctx.lineWidth = 1;
      ctx.strokeRect(powerBarX, powerBarY, powerBarWidth, powerBarHeight);

      // Power label
      ctx.fillStyle = '#000000';
      ctx.font = '10px Arial';
      ctx.textAlign = 'left';
      ctx.textBaseline = 'bottom';
      ctx.fillText(`Power: ${Math.round(force * 100)}%`, powerBarX, powerBarY - 2);
    }
  }

  // Draw cue detection confidence
  ctx.fillStyle = '#000000';
  ctx.font = '10px Arial';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  const confidenceText = `Cue: ${Math.round(cue.confidence * 100)}%`;
  ctx.fillText(confidenceText, tipPoint.x, tipPoint.y + 15);

  // Draw cue length if available
  if (cue.length > 0) {
    const lengthText = `${Math.round(cue.length)}px`;
    ctx.fillStyle = '#666666';
    ctx.font = '9px Arial';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    ctx.fillText(lengthText, (tipPoint.x + tailPoint.x) / 2, (tipPoint.y + tailPoint.y) / 2 + 10);
  }

  // Draw cue position coordinates (for debugging)
  if ((cue as any).showCoordinates) {
    const tipCoords = `(${Math.round(cue.tipPosition.x)}, ${Math.round(cue.tipPosition.y)})`;
    const tailCoords = `(${Math.round(cue.tailPosition.x)}, ${Math.round(cue.tailPosition.y)})`;

    ctx.fillStyle = '#000000';
    ctx.font = '8px monospace';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'top';

    ctx.fillText(`Tip: ${tipCoords}`, tipPoint.x + 10, tipPoint.y + 25);
    ctx.fillText(`Tail: ${tailCoords}`, tailPoint.x + 10, tailPoint.y + 25);
  }

  // Draw cue shadow (optional visual enhancement)
  if (config.opacity > 0.5) {
    const shadowOffset = 3;
    ctx.globalAlpha = 0.3;
    ctx.strokeStyle = '#000000';
    ctx.lineWidth = config.lineWidth;
    ctx.beginPath();
    ctx.moveTo(tipPoint.x + shadowOffset, tipPoint.y + shadowOffset);
    ctx.lineTo(tailPoint.x + shadowOffset, tailPoint.y + shadowOffset);
    ctx.stroke();
  }

  // Draw trajectory prediction cone (where cue would hit)
  if (config.showGuideLines) {
    const hitConeLength = 50;
    const hitConeWidth = 15;

    const coneEnd = {
      x: cue.tipPosition.x + Math.cos(cue.angle) * hitConeLength,
      y: cue.tipPosition.y + Math.sin(cue.angle) * hitConeLength,
    };

    const perpAngle = cue.angle + Math.PI / 2;
    const cone1 = {
      x: coneEnd.x + Math.cos(perpAngle) * hitConeWidth,
      y: coneEnd.y + Math.sin(perpAngle) * hitConeWidth,
    };
    const cone2 = {
      x: coneEnd.x - Math.cos(perpAngle) * hitConeWidth,
      y: coneEnd.y - Math.sin(perpAngle) * hitConeWidth,
    };

    const conePoint1 = transform.videoToCanvas(cone1);
    const conePoint2 = transform.videoToCanvas(cone2);
    const coneEndPoint = transform.videoToCanvas(coneEnd);

    ctx.globalAlpha = 0.2;
    ctx.fillStyle = '#FF8800';
    ctx.beginPath();
    ctx.moveTo(tipPoint.x, tipPoint.y);
    ctx.lineTo(conePoint1.x, conePoint1.y);
    ctx.lineTo(coneEndPoint.x, coneEndPoint.y);
    ctx.lineTo(conePoint2.x, conePoint2.y);
    ctx.closePath();
    ctx.fill();
  }

  ctx.restore();
}

// React wrapper component for testing/isolation
interface CueOverlayComponentProps {
  cue: CueStick | null;
  transform: CoordinateTransform;
  config: OverlayConfig['cue'];
}

export const CueOverlayComponent: React.FC<CueOverlayComponentProps> = ({
  cue,
  transform,
  config
}) => {
  const canvasRef = React.useRef<HTMLCanvasElement>(null);

  React.useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    renderCueOverlay({ cue, transform, config, ctx });
  }, [cue, transform, config]);

  return (
    <canvas
      ref={canvasRef}
      className="absolute inset-0 pointer-events-none"
      style={{ zIndex: 15 }}
    />
  );
};
