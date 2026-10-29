import { canonicalToImagePoint, normalizeImagePointToCanonical } from '../../utils/coordinates';
import type { BallData } from '../../types/api';
import type { FramePairingIdentity } from './use-frame-aware-mjpeg';
// Wire resource bound shared with the scene/public-state contracts.
const MAX_REFINED_OUTLINE_VERTICES = 512;

/** Preserve fitted geometry only for its exact observed acquisition. Canvas
 * clipping handles vertices outside the frame; clamping would change shape. */
export function observedBallOutline(
  ball: Pick<BallData, 'outline' | 'source_observation' | 'is_observed'>,
  frame: FramePairingIdentity | null,
  imageSize: { width: number; height: number },
): { x: number; y: number }[] | null {
  const source = ball.source_observation;
  const outline = ball.outline;
  if (!frame || ball.is_observed !== true || !source || !outline
    || source.frame_id !== frame.frameId
    || source.engine_instance_id !== frame.engineInstanceId
    || String(source.session_id) !== frame.sourceSessionId
    || !Array.isArray(outline.points)
    || outline.points.length < 3 || outline.points.length > MAX_REFINED_OUTLINE_VERTICES) return null;
  const points = outline.points.map((point) => {
    const canonical = normalizeImagePointToCanonical({ ...point, scale: outline.scale });
    return canonical ? canonicalToImagePoint(canonical, imageSize) : null;
  });
  return points.every((point) => point !== null) ? points as { x: number; y: number }[] : null;
}
