/**
 * Table detection overlay component
 */

import React from 'react';
import type { Table, CoordinateTransform, OverlayConfig } from '../../../types/video';

interface TableOverlayProps {
  table: Table | null;
  transform: CoordinateTransform;
  config: OverlayConfig['table'];
  ctx: CanvasRenderingContext2D;
}

export function renderTableOverlay({ table, transform, config, ctx }: TableOverlayProps): void {
  if (!config.visible || !table || !table.detected) return;

  ctx.save();
  ctx.globalAlpha = config.opacity;

  // Draw table outline
  if (table.corners.length >= 4) {
    ctx.strokeStyle = '#00FF00';
    ctx.lineWidth = config.lineWidth;
    ctx.setLineDash([]);

    ctx.beginPath();
    const startCorner = transform.videoToCanvas(table.corners[0]);
    ctx.moveTo(startCorner.x, startCorner.y);

    // Draw table perimeter
    for (let i = 1; i < table.corners.length; i++) {
      const corner = transform.videoToCanvas(table.corners[i]);
      ctx.lineTo(corner.x, corner.y);
    }

    // Close the path
    ctx.closePath();
    ctx.stroke();

    // Draw corner markers
    table.corners.forEach((corner, index) => {
      const cornerPoint = transform.videoToCanvas(corner);

      // Corner circle
      ctx.fillStyle = '#00FF00';
      ctx.strokeStyle = '#000000';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.arc(cornerPoint.x, cornerPoint.y, 6, 0, 2 * Math.PI);
      ctx.fill();
      ctx.stroke();

      // Corner number
      ctx.fillStyle = '#000000';
      ctx.font = '10px Arial';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText((index + 1).toString(), cornerPoint.x, cornerPoint.y);
    });
  }

  // Draw pockets if enabled
  if (config.showPockets && table.pockets.length > 0) {
    table.pockets.forEach((pocket, index) => {
      const pocketPoint = transform.videoToCanvas(pocket);

      // Pocket hole (black circle)
      ctx.fillStyle = '#000000';
      ctx.strokeStyle = '#FFFFFF';
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.arc(pocketPoint.x, pocketPoint.y, 12, 0, 2 * Math.PI);
      ctx.fill();
      ctx.stroke();

      // Pocket rim (yellow circle)
      ctx.strokeStyle = '#FFD700';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(pocketPoint.x, pocketPoint.y, 18, 0, 2 * Math.PI);
      ctx.stroke();

      // Pocket number/label
      ctx.fillStyle = '#FFD700';
      ctx.font = '12px Arial';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'bottom';
      ctx.fillText(`P${index + 1}`, pocketPoint.x, pocketPoint.y - 25);

      // Draw pocket influence zone (optional)
      ctx.strokeStyle = '#FFD700';
      ctx.lineWidth = 1;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.arc(pocketPoint.x, pocketPoint.y, 35, 0, 2 * Math.PI);
      ctx.stroke();
      ctx.setLineDash([]);
    });
  }

  // Draw rails if enabled
  if (config.showRails && table.rails.length > 0) {
    table.rails.forEach((rail, railIndex) => {
      if (rail.length < 2) return;

      // Rail line
      ctx.strokeStyle = '#0080FF';
      ctx.lineWidth = config.lineWidth + 1;
      ctx.lineCap = 'round';

      ctx.beginPath();
      const startPoint = transform.videoToCanvas(rail[0]);
      ctx.moveTo(startPoint.x, startPoint.y);

      rail.slice(1).forEach(point => {
        const canvasPoint = transform.videoToCanvas(point);
        ctx.lineTo(canvasPoint.x, canvasPoint.y);
      });

      ctx.stroke();

      // Rail markers at endpoints
      [rail[0], rail[rail.length - 1]].forEach((endpoint, endIndex) => {
        const endPoint = transform.videoToCanvas(endpoint);

        ctx.fillStyle = '#0080FF';
        ctx.strokeStyle = '#000000';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.arc(endPoint.x, endPoint.y, 4, 0, 2 * Math.PI);
        ctx.fill();
        ctx.stroke();
      });

      // Rail label
      if (rail.length >= 2) {
        const midIndex = Math.floor(rail.length / 2);
        const midPoint = transform.videoToCanvas(rail[midIndex]);

        ctx.fillStyle = '#0080FF';
        ctx.font = '10px Arial';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(`R${railIndex + 1}`, midPoint.x, midPoint.y - 15);
      }
    });
  }

  // Draw table center mark
  if (table.corners.length >= 4) {
    // Calculate table center from corners
    let centerX = 0;
    let centerY = 0;
    table.corners.forEach(corner => {
      centerX += corner.x;
      centerY += corner.y;
    });
    centerX /= table.corners.length;
    centerY /= table.corners.length;

    const tableCenter = transform.videoToCanvas({ x: centerX, y: centerY });

    // Center cross
    ctx.strokeStyle = '#FFFFFF';
    ctx.lineWidth = 2;
    const crossSize = 10;

    ctx.beginPath();
    ctx.moveTo(tableCenter.x - crossSize, tableCenter.y);
    ctx.lineTo(tableCenter.x + crossSize, tableCenter.y);
    ctx.moveTo(tableCenter.x, tableCenter.y - crossSize);
    ctx.lineTo(tableCenter.x, tableCenter.y + crossSize);
    ctx.stroke();

    // Center circle
    ctx.strokeStyle = '#FFFFFF';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.arc(tableCenter.x, tableCenter.y, 5, 0, 2 * Math.PI);
    ctx.stroke();
  }

  // Draw table bounds (bounding box)
  if (table.bounds) {
    const bounds = table.bounds;
    const topLeft = transform.videoToCanvas({ x: bounds.x, y: bounds.y });
    const bottomRight = transform.videoToCanvas({
      x: bounds.x + bounds.width,
      y: bounds.y + bounds.height,
    });

    ctx.strokeStyle = '#FF8000';
    ctx.lineWidth = 1;
    ctx.setLineDash([5, 5]);
    ctx.beginPath();
    ctx.rect(
      topLeft.x,
      topLeft.y,
      bottomRight.x - topLeft.x,
      bottomRight.y - topLeft.y
    );
    ctx.stroke();
    ctx.setLineDash([]);

    // Bounds label
    ctx.fillStyle = '#FF8000';
    ctx.font = '10px Arial';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'top';
    ctx.fillText(
      `${Math.round(bounds.width)}×${Math.round(bounds.height)}`,
      topLeft.x + 5,
      topLeft.y + 5
    );
  }

  // Draw detection confidence
  if (table.corners.length > 0) {
    const firstCorner = transform.videoToCanvas(table.corners[0]);

    ctx.fillStyle = '#00FF00';
    ctx.font = '12px Arial';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'bottom';
    const confidenceText = `Table: ${Math.round(table.confidence * 100)}%`;
    ctx.fillText(confidenceText, firstCorner.x + 10, firstCorner.y - 10);
  }

  // Draw playing area indicators (if available)
  if (table.corners.length >= 4) {
    // Draw head string (breaking area)
    const corner1 = transform.videoToCanvas(table.corners[0]);
    const corner2 = transform.videoToCanvas(table.corners[1]);
    const corner3 = transform.videoToCanvas(table.corners[2]);
    const corner4 = transform.videoToCanvas(table.corners[3]);

    // Assume corners are in order: top-left, top-right, bottom-right, bottom-left
    // Head string is typically 1/4 from one end
    const headStringY1 = corner1.y + (corner4.y - corner1.y) * 0.25;
    const headStringY2 = corner2.y + (corner3.y - corner2.y) * 0.25;

    ctx.strokeStyle = '#FFFF00';
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(corner1.x, headStringY1);
    ctx.lineTo(corner2.x, headStringY2);
    ctx.stroke();

    // Foot string (racking area)
    const footStringY1 = corner1.y + (corner4.y - corner1.y) * 0.75;
    const footStringY2 = corner2.y + (corner3.y - corner2.y) * 0.75;

    ctx.beginPath();
    ctx.moveTo(corner1.x, footStringY1);
    ctx.lineTo(corner2.x, footStringY2);
    ctx.stroke();

    // Center string (long axis)
    const centerX1 = (corner1.x + corner2.x) / 2;
    const centerX2 = (corner3.x + corner4.x) / 2;

    ctx.beginPath();
    ctx.moveTo(centerX1, corner1.y);
    ctx.lineTo(centerX2, corner4.y);
    ctx.stroke();

    ctx.setLineDash([]);

    // Area labels
    ctx.fillStyle = '#FFFF00';
    ctx.font = '10px Arial';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';

    // Head area label
    const headCenterX = (corner1.x + corner2.x) / 2;
    const headCenterY = (corner1.y + headStringY1) / 2;
    ctx.fillText('HEAD', headCenterX, headCenterY);

    // Foot area label
    const footCenterX = (corner3.x + corner4.x) / 2;
    const footCenterY = (footStringY1 + corner4.y) / 2;
    ctx.fillText('FOOT', footCenterX, footCenterY);
  }

  ctx.restore();
}

// React wrapper component for testing/isolation
interface TableOverlayComponentProps {
  table: Table | null;
  transform: CoordinateTransform;
  config: OverlayConfig['table'];
}

export const TableOverlayComponent: React.FC<TableOverlayComponentProps> = ({
  table,
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
    renderTableOverlay({ table, transform, config, ctx });
  }, [table, transform, config]);

  return (
    <canvas
      ref={canvasRef}
      className="absolute inset-0 pointer-events-none"
      style={{ zIndex: 1 }}
    />
  );
};
