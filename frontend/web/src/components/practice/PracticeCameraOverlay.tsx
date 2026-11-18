import { useEffect, useRef, useState } from 'react';
import type { MouseEvent } from 'react';
import { centerToTable, imageToTableMm, tableMmToImage, tableToCenter } from '../video/live-coaching-geometry';
import type { TableProjection } from '../../types/api';
import type { PotCandidate } from '../video/shot-prediction';

export interface PracticeOverlayTarget {
  id: string;
  label: string;
  /** Positive-origin physical table coordinates, in millimeters. */
  x: number;
  y: number;
  placed: boolean;
}

export interface PracticeOverlayBall {
  /** Physical table coordinates from the producer, centered on the playing surface. */
  x: number;
  y: number;
  radiusMm: number;
  matched: boolean;
  /** Extrapolated centers are shown dimly and never count as a placement. */
  extrapolated: boolean;
}

interface PracticeCameraOverlayProps {
  image: HTMLImageElement | null;
  imageAvailable: boolean;
  mode: string | null | undefined;
  dimensions: { width: number; height: number } | null;
  projection: TableProjection | null;
  tableRadiusMm: number;
  toleranceMm: number;
  targets: readonly PracticeOverlayTarget[];
  balls: readonly PracticeOverlayBall[];
  canEditOnImage: boolean;
  ready: boolean;
  notice: string;
  readinessMessage: string;
  shot: PotCandidate | null;
  onPlaceAt: (x: number, y: number) => void;
  onImageDimensions: (width: number, height: number) => void;
}

interface ProjectedEllipse {
  x: number;
  y: number;
  rx: number;
  ry: number;
}

function projectEllipse(
  projection: TableProjection,
  centerMm: { x: number; y: number },
  radiusMm: number,
  dimensions: { width: number; height: number },
): ProjectedEllipse | null {
  const center = tableMmToImage(projection, centerMm);
  const edgeX = tableMmToImage(projection, { x: centerMm.x + radiusMm, y: centerMm.y });
  const edgeY = tableMmToImage(projection, { x: centerMm.x, y: centerMm.y + radiusMm });
  if (!center || !edgeX || !edgeY) return null;
  const scaleX = dimensions.width / projection.source_width_px;
  const scaleY = dimensions.height / projection.source_height_px;
  const x = center.x * scaleX;
  const y = center.y * scaleY;
  if (x < 0 || y < 0 || x > dimensions.width || y > dimensions.height) return null;
  return {
    x,
    y,
    rx: Math.hypot(edgeX.x - center.x, edgeX.y - center.y) * scaleX,
    ry: Math.hypot(edgeY.x - center.x, edgeY.y - center.y) * scaleY,
  };
}

function projectPositivePoint(
  projection: TableProjection,
  point: { x: number; y: number },
  dimensions: { width: number; height: number },
): { x: number; y: number } | null {
  const centerOrigin = tableToCenter(point, projection.table_width_mm, projection.table_height_mm);
  const projected = tableMmToImage(projection, centerOrigin);
  if (!projected) return null;
  return {
    x: projected.x * dimensions.width / projection.source_width_px,
    y: projected.y * dimensions.height / projection.source_height_px,
  };
}

/**
 * Presents the editable practice layout on the one app-owned live image. The
 * shared image is borrowed only from its parking host and returned there on
 * unmount, matching the live coach and calibration ownership contract.
 */
export function PracticeCameraOverlay({
  image,
  imageAvailable,
  mode,
  dimensions,
  projection,
  tableRadiusMm,
  toleranceMm,
  targets,
  balls,
  canEditOnImage,
  ready,
  notice,
  readinessMessage,
  shot,
  onPlaceAt,
  onImageDimensions,
}: PracticeCameraOverlayProps) {
  const imageSlotRef = useRef<HTMLDivElement>(null);
  const [ownershipConflict, setOwnershipConflict] = useState(false);
  const usableDimensions = dimensions && dimensions.width > 0 && dimensions.height > 0
    ? dimensions
    : projection ? { width: projection.source_width_px, height: projection.source_height_px } : null;

  useEffect(() => {
    const slot = imageSlotRef.current;
    if (!image || mode !== 'native' || !slot) {
      setOwnershipConflict(false);
      return undefined;
    }
    const originalParent = image.parentElement;
    // A second visible surface may already own this single native image. Do
    // not detach it from that surface; practice remains explicitly unverified.
    if (originalParent !== slot && originalParent?.id !== 'shared-video-receiver-parking') {
      setOwnershipConflict(true);
      return undefined;
    }
    setOwnershipConflict(false);
    const loaded = () => onImageDimensions(image.naturalWidth, image.naturalHeight);
    image.addEventListener('load', loaded);
    if (originalParent !== slot) slot.appendChild(image);
    loaded();
    return () => {
      image.removeEventListener('load', loaded);
      if (image.parentElement === slot && originalParent) originalParent.appendChild(image);
    };
  }, [image, mode, onImageDimensions]);

  const projectedTargets = projection && usableDimensions
    ? targets.map((target) => ({
      target,
      tolerance: projectEllipse(
        projection,
        tableToCenter(target, projection.table_width_mm, projection.table_height_mm),
        toleranceMm,
        usableDimensions,
      ),
      ellipse: projectEllipse(
        projection,
        tableToCenter(target, projection.table_width_mm, projection.table_height_mm),
        tableRadiusMm,
        usableDimensions,
      ),
    })).filter((entry): entry is { target: PracticeOverlayTarget; ellipse: ProjectedEllipse; tolerance: ProjectedEllipse } => !!entry.ellipse && !!entry.tolerance)
    : [];
  const projectedBalls = projection && usableDimensions
    ? balls.map((ball, inputIndex) => ({
      inputIndex,
      ball,
      ellipse: projectEllipse(projection, { x: ball.x, y: ball.y }, ball.radiusMm, usableDimensions),
    })).filter((entry): entry is { inputIndex: number; ball: PracticeOverlayBall; ellipse: ProjectedEllipse } => !!entry.ellipse)
    : [];
  const projectedShot = projection && usableDimensions && shot ? {
    cueStart: projectPositivePoint(projection, shot.cueBall, usableDimensions),
    ghostCenter: projectPositivePoint(projection, shot.contactCenter, usableDimensions),
    objectStart: projectPositivePoint(projection, shot.objectPathStart, usableDimensions),
    pocketEnd: projectPositivePoint(projection, shot.objectPathEnd, usableDimensions),
    contact: projectPositivePoint(projection, shot.contactPoint, usableDimensions),
  } : null;

  function placeFromImage(event: MouseEvent<SVGSVGElement>): void {
    if (!canEditOnImage || !projection || !usableDimensions) return;
    const inverse = event.currentTarget.getScreenCTM()?.inverse();
    if (!inverse) return;
    const displayedX = inverse.a * event.clientX + inverse.c * event.clientY + inverse.e;
    const displayedY = inverse.b * event.clientX + inverse.d * event.clientY + inverse.f;
    const physical = imageToTableMm(projection, {
      x: displayedX * projection.source_width_px / usableDimensions.width,
      y: displayedY * projection.source_height_px / usableDimensions.height,
    });
    if (!physical) return;
    onPlaceAt(centerToTable(physical, projection.table_width_mm, projection.table_height_mm).x,
      centerToTable(physical, projection.table_width_mm, projection.table_height_mm).y);
  }

  const cameraLoaded = imageAvailable && mode === 'native' && !!image && !!dimensions
    && image.naturalWidth > 0 && image.naturalHeight > 0
    && dimensions.width > 0 && dimensions.height > 0;
  const stateMessage = ownershipConflict
    ? 'Camera is already displayed in another view. Placement is unverified.'
    : ready ? readinessMessage : notice;
  const stageAspect = usableDimensions
    ? `${usableDimensions.width} / ${usableDimensions.height}`
    : '16 / 9';

  return <figure className="practice-camera-figure" aria-label="Live camera with practice placement overlay">
    <div className="practice-camera-stage" style={{ aspectRatio: stageAspect }} data-testid="practice-camera-stage">
      <div className="native-video-image-slot practice-camera-image-slot" ref={imageSlotRef} />
      {!cameraLoaded && <div className="practice-camera-empty" data-testid="practice-camera-unavailable">
        {ownershipConflict ? 'Camera is being shown in another view.' : mode === 'native' && image ? 'Waiting for the shared camera image…' : 'Live camera video is unavailable.'}
      </div>}
      {cameraLoaded && projection && usableDimensions && (
        <svg
          className="practice-camera-overlay"
          viewBox={`0 0 ${usableDimensions.width} ${usableDimensions.height}`}
          preserveAspectRatio="none"
          role="application"
          aria-label={canEditOnImage ? 'Live camera table placement overlay. Click to place the selected target.' : 'Live camera placement overlay is unverified.'}
          onClick={placeFromImage}
          data-testid="practice-camera-overlay"
        >
          {projectedShot?.cueStart && projectedShot.ghostCenter && <line
            x1={projectedShot.cueStart.x} y1={projectedShot.cueStart.y}
            x2={projectedShot.ghostCenter.x} y2={projectedShot.ghostCenter.y}
            className="practice-shot-cue-line"
          />}
          {projectedShot?.objectStart && projectedShot.pocketEnd && <line
            x1={projectedShot.objectStart.x} y1={projectedShot.objectStart.y}
            x2={projectedShot.pocketEnd.x} y2={projectedShot.pocketEnd.y}
            className="practice-shot-object-line"
          />}
          {projectedShot?.ghostCenter && <>
            <circle cx={projectedShot.ghostCenter.x} cy={projectedShot.ghostCenter.y}
              r={projectEllipse(projection, tableToCenter(shot!.contactCenter, projection.table_width_mm, projection.table_height_mm), tableRadiusMm, usableDimensions)?.rx ?? 8}
              className="practice-shot-ghost" />
            {projectedShot.contact && <circle cx={projectedShot.contact.x} cy={projectedShot.contact.y} r="5" className="practice-shot-contact" />}
          </>}
          {projectedTargets.map(({ target, tolerance, ellipse }) => <g key={target.id} data-testid={`practice-target-${target.id}`}>
            <ellipse
              cx={tolerance.x} cy={tolerance.y} rx={tolerance.rx} ry={tolerance.ry}
              className={`practice-target-tolerance${target.placed ? ' is-placed' : ''}`}
            />
            <ellipse
              cx={ellipse.x} cy={ellipse.y} rx={ellipse.rx} ry={ellipse.ry}
              className={`practice-target-marker${target.placed ? ' is-placed' : ''}`}
            />
            <text x={ellipse.x} y={ellipse.y - ellipse.ry - 7} className="practice-overlay-label">
              {target.label}
            </text>
          </g>)}
          {projectedBalls.map(({ inputIndex, ball, ellipse }) => <ellipse
            key={`detected-${inputIndex}`}
            cx={ellipse.x} cy={ellipse.y} rx={ellipse.rx} ry={ellipse.ry}
            className={`practice-detected-marker${ball.matched ? ' is-matched' : ''}${ball.extrapolated ? ' is-extrapolated' : ''}`}
            data-testid="practice-detected-center"
          />)}
        </svg>
      )}
      <div
        className={`practice-camera-state${ready ? ' is-ready' : ''}`}
        data-testid="practice-camera-state"
        role="status"
        aria-live="polite"
      >
        <strong>{ready ? 'Ready' : 'Unverified'}</strong>
        <span>{stateMessage}</span>
      </div>
    </div>
    <figcaption className={ready ? 'practice-placement-caption is-ready' : 'practice-placement-caption'}>
      {stateMessage}
    </figcaption>
  </figure>;
}
