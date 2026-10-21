import { renderTrajectoryOverlay } from '../../components/video/overlays/TrajectoryOverlay';
import type { Trajectory } from '../../types/video';

describe('frozen trajectory display', () => {
  test('renders a retained prediction as a labeled dashed reference', () => {
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
      fillText: vi.fn(),
      closePath: vi.fn(),
    } as unknown as CanvasRenderingContext2D;
    const trajectory: Trajectory = {
      ballId: 'cue',
      points: [{ x: 10, y: 20 }, { x: 30, y: 40 }],
      collisions: [],
      type: 'primary',
      probability: 0.9,
      color: '#00FF00',
      predictionStatus: 'frozen_reference',
    };
    const transform = {
      videoToCanvas: (point: { x: number; y: number }) => point,
      canvasToVideo: (point: { x: number; y: number }) => point,
      screenToCanvas: (point: { x: number; y: number }) => point,
      canvasToScreen: (point: { x: number; y: number }) => point,
    };

    renderTrajectoryOverlay({
      trajectories: [trajectory],
      transform,
      config: {
        visible: true,
        showProbability: true,
        lineWidth: 2,
        opacity: 1,
        maxLength: 0,
      },
      ctx,
    });

    expect(ctx.setLineDash).toHaveBeenCalledWith([8, 6]);
    expect(ctx.fillText).toHaveBeenCalledWith('Previous prediction', 15, 15);
    expect(ctx.restore).toHaveBeenCalledOnce();
  });
});
