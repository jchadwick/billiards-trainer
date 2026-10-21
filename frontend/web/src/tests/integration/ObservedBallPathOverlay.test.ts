import { renderObservedBallPathOverlay } from '../../components/video/overlays/ObservedBallPathOverlay';

describe('observed ball path overlay', () => {
  test('does not connect measurements across a marked observation gap', () => {
    const ctx = {
      save: vi.fn(),
      restore: vi.fn(),
      setLineDash: vi.fn(),
      beginPath: vi.fn(),
      moveTo: vi.fn(),
      lineTo: vi.fn(),
      stroke: vi.fn(),
      fill: vi.fn(),
      arc: vi.fn(),
    } as unknown as CanvasRenderingContext2D;
    const identity = (point: { x: number; y: number }) => point;

    renderObservedBallPathOverlay({
      trails: [
        {
          ballId: 'cue-track',
          points: [
            { x: 10, y: 20, timestamp: 1, timestampKind: 'source', gapBefore: false },
            { x: 20, y: 20, timestamp: 1.1, timestampKind: 'source', gapBefore: false },
            { x: 60, y: 20, timestamp: 1.8, timestampKind: 'source', gapBefore: true },
          ],
        },
      ],
      transform: {
        videoToCanvas: identity,
        canvasToVideo: identity,
        screenToCanvas: identity,
        canvasToScreen: identity,
      },
      config: { visible: true, showProbability: false, lineWidth: 2, opacity: 1, maxLength: 0 },
      ctx,
    });

    // Only the first contiguous pair gets a line. The final point is marked by
    // a dashed amber ring and remains spatially separate from the earlier path.
    expect(ctx.lineTo).toHaveBeenCalledTimes(1);
    expect(ctx.lineTo).toHaveBeenCalledWith(20, 20);
    expect(ctx.arc).toHaveBeenCalledTimes(4);
    expect(ctx.setLineDash).toHaveBeenCalledWith([2, 3]);
    expect(ctx.restore).toHaveBeenCalledOnce();
  });
});
