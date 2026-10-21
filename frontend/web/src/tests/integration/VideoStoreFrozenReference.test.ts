import { apiClient } from '../../api/client';
import { VideoStore } from '../../stores/VideoStore';
import { vi } from 'vitest';
import type { GameStateData, TrajectoryData, WebSocketMessage } from '../../types/api';

function stateMessage(data: GameStateData): WebSocketMessage {
  return {
    type: 'state',
    timestamp: new Date().toISOString(),
    data: data as unknown as Record<string, any>,
  };
}

function trajectoryMessage(data: TrajectoryData): WebSocketMessage {
  return {
    type: 'trajectory',
    timestamp: new Date().toISOString(),
    data: data as unknown as Record<string, any>,
  };
}

function prediction(
  predictionId: string,
  metadata: Partial<TrajectoryData> = {}
): TrajectoryData {
  return {
    lines: [
      {
        start: { x: 100, y: 120, scale: [1, 1] },
        end: { x: 300, y: 220, scale: [1, 1] },
        type: 'primary',
        confidence: 0.9,
      },
    ],
    collisions: [],
    confidence: 0.9,
    calculation_time_ms: 2,
    line_count: 1,
    collision_count: 0,
    prediction_status: 'live',
    prediction_id: predictionId,
    map_revision: 4,
    calibration_revision: 7,
    ...metadata,
  };
}

describe('VideoStore frozen trajectory reference', () => {
  let store: VideoStore;

  beforeEach(() => {
    // dispose() intentionally stops capture; keep this store-only test isolated
    // from a backend process while still exercising its cleanup path.
    vi.spyOn(apiClient, 'stopVideoCapture').mockResolvedValue({
      success: true,
      timestamp: new Date().toISOString(),
    });
    store = new VideoStore();
  });

  afterEach(async () => {
    store.dispose();
    // dispose() starts a dynamic API import before it stops capture. Let that
    // cleanup finish while the API spy is still installed.
    await new Promise<void>((resolve) => setTimeout(resolve, 0));
    vi.restoreAllMocks();
  });

  function sendState(phase: string, overrides: Partial<GameStateData> = {}) {
    (store as any).handleGameStateMessage(
      stateMessage({
        balls: [],
        ball_count: 0,
        phase,
        calibration_revision: 7,
        map_revision: 4,
        map_trusted: true,
        aim_calculation_eligible: phase === 'aiming',
        ...overrides,
      })
    );
  }

  function sendBallObservation(
    phase: string,
    frameNumber: number,
    sourceTimestamp: number,
    position: { x: number; y: number },
    status: 'observed' | 'cached' | 'unavailable' = 'observed',
    maximumPoints = 120
  ) {
    (store as any).handleGameStateMessage(
      stateMessage({
        balls: [
          {
            id: 'cue-track',
            position: { ...position, scale: [1, 1] },
            radius: 10,
            color: 'white',
            confidence: 1,
            visible: true,
          },
        ],
        ball_count: 1,
        frame_number: frameNumber,
        phase,
        calibration_revision: 7,
        map_revision: 4,
        map_trusted: false,
        aim_calculation_eligible: false,
        observation_status: {
          phase,
          phase_aware: true,
          ball_positions_status: status,
          ball_observation_source_timestamp: sourceTimestamp,
          ball_observation_source_frame_number: frameNumber,
          ball_observation_source_timestamp_kind: 'shared_memory_publication',
          maximum_ball_observation_gap_seconds: 0.2,
          actual_path_history_max_points: maximumPoints,
        },
      })
    );
  }

  function sendPrediction(data: TrajectoryData) {
    (store as any).handleTrajectoryMessage(trajectoryMessage(data));
  }

  test('keeps and freezes the last prediction through motion, recovery, settlement, and idle', () => {
    sendState('aiming');
    sendPrediction(prediction('prediction-1'));
    expect(store.currentTrajectories[0].predictionStatus).toBe('live');

    sendState('limited_motion', { aim_calculation_eligible: false });
    expect(store.currentTrajectories[0].predictionStatus).toBe('frozen_reference');
    expect(store.currentTrajectories[0].predictionId).toBe('prediction-1');

    sendState('many_moving', { aim_calculation_eligible: false });
    sendState('recovering', { aim_calculation_eligible: false });
    sendState('idle', { aim_calculation_eligible: false });
    expect(store.currentTrajectories).toHaveLength(1);
    expect(store.currentTrajectories[0].predictionStatus).toBe('frozen_reference');
  });

  test('ignores live predictions during motion and replaces the reference on valid aiming output', () => {
    sendState('aiming');
    sendPrediction(prediction('prediction-1'));
    sendState('many_moving', { aim_calculation_eligible: false });

    sendPrediction(prediction('invalid-during-motion'));
    expect(store.currentTrajectories[0].predictionId).toBe('prediction-1');
    expect(store.currentTrajectories[0].predictionStatus).toBe('frozen_reference');

    sendState('idle', {
      aim_calculation_eligible: true,
      map_trusted: true,
    });
    sendPrediction(prediction('premature-idle-prediction'));
    expect(store.currentTrajectories[0].predictionId).toBe('prediction-1');

    sendState('aiming', { aim_calculation_eligible: true });
    sendPrediction(prediction('prediction-2'));
    expect(store.currentTrajectories[0].predictionId).toBe('prediction-2');
    expect(store.currentTrajectories[0].predictionStatus).toBe('live');
  });

  test('hides geometry when calibration revisions differ or backend marks it spatially invalid', () => {
    sendState('aiming');
    sendPrediction(prediction('prediction-1'));
    expect(store.currentTrajectories).toHaveLength(1);

    sendState('recovering', {
      calibration_revision: 8,
      aim_calculation_eligible: false,
    });
    expect(store.currentTrajectories).toHaveLength(0);

    sendState('aiming', { calibration_revision: 8 });
    sendPrediction(prediction('recalibrated', { calibration_revision: 8 }));
    expect(store.currentTrajectories).toHaveLength(1);

    sendPrediction(
      prediction('spatially-invalid', {
        prediction_status: 'frozen_reference',
        spatial_valid: false,
        calibration_revision: 8,
        lines: [],
      })
    );
    expect(store.currentTrajectories).toHaveLength(0);
  });

  test('ignores predictions without an explicit live or frozen lifecycle status', () => {
    sendState('aiming');
    sendPrediction(prediction('prediction-1'));

    const unlabelled = prediction('unlabelled') as any;
    delete unlabelled.prediction_status;
    (store as any).handleTrajectoryMessage(trajectoryMessage(unlabelled));

    const legacyAliasOnly = prediction('legacy-alias') as any;
    delete legacyAliasOnly.prediction_status;
    legacyAliasOnly.is_reference = true;
    (store as any).handleTrajectoryMessage(trajectoryMessage(legacyAliasOnly));

    expect(store.currentTrajectories).toHaveLength(1);
    expect(store.currentTrajectories[0].predictionId).toBe('prediction-1');
    expect(store.currentTrajectories[0].predictionStatus).toBe('live');
  });

  test('uses line identity, then collision identity, and ignores the removed root alias', () => {
    sendState('aiming');
    const withLineIdentity = prediction('line-identity', {
      lines: [
        {
          start: { x: 100, y: 120, scale: [1, 1] },
          end: { x: 300, y: 220, scale: [1, 1] },
          type: 'primary',
          confidence: 0.9,
          ball_id: 'object-line',
        },
        {
          start: { x: 300, y: 220, scale: [1, 1] },
          end: { x: 400, y: 250, scale: [1, 1] },
          type: 'reflection',
          confidence: 0.8,
        },
      ],
      collisions: [
        {
          position: { x: 300, y: 220, scale: [1, 1] },
          ball1_id: 'collision-cue',
          angle: 0,
        },
      ],
    });

    sendPrediction(withLineIdentity);
    expect(store.currentTrajectories.map((trajectory) => trajectory.ballId)).toEqual([
      'object-line',
      'collision-cue',
    ]);

    const legacyPayload = {
      ...prediction('legacy-root-only', { collisions: [] }),
      ball_id: 'removed-root-alias',
    } as unknown as TrajectoryData;
    sendPrediction(legacyPayload);
    expect(store.currentTrajectories.map((trajectory) => trajectory.ballId)).toEqual([
      'trajectory_0',
    ]);
  });

  test('assigns deterministic per-line IDs when no identity metadata is available', () => {
    sendState('aiming');
    sendPrediction(
      prediction('unidentified-lines', {
        lines: [
          {
            start: { x: 100, y: 120, scale: [1, 1] },
            end: { x: 300, y: 220, scale: [1, 1] },
            type: 'primary',
            confidence: 0.9,
          },
          {
            start: { x: 300, y: 220, scale: [1, 1] },
            end: { x: 400, y: 250, scale: [1, 1] },
            type: 'reflection',
            confidence: 0.8,
          },
        ],
        collisions: [],
      })
    );

    expect(store.currentTrajectories.map((trajectory) => trajectory.ballId)).toEqual([
      'trajectory_0',
      'trajectory_1',
    ]);
  });

  test('stores only measured positions and marks skipped intervals as gaps', () => {
    sendState('aiming');
    sendPrediction(prediction('prediction-1'));
    sendBallObservation('limited_motion', 20, 100, { x: 100, y: 100 });
    sendBallObservation('limited_motion', 21, 100.1, { x: 120, y: 100 });
    sendBallObservation('limited_motion', 22, 100.2, { x: 180, y: 100 }, 'cached');
    sendBallObservation('limited_motion', 23, 100.3, { x: 220, y: 100 });

    const trail = store.observedBallTrails.find((item) => item.ballId === 'cue-track');
    expect(trail?.points.map((point) => point.x)).toEqual([100, 120, 220]);
    expect(trail?.points.map((point) => point.gapBefore)).toEqual([false, false, true]);
    expect(store.currentTrajectories[0].predictionStatus).toBe('frozen_reference');
  });

  test('bounds measured path history using the backend profile setting', () => {
    sendState('aiming');
    sendPrediction(prediction('prediction-1'));
    for (let frame = 1; frame <= 4; frame += 1) {
      sendBallObservation('limited_motion', frame, 200 + frame / 10, {
        x: frame * 10,
        y: 90,
      }, 'observed', 2);
    }

    expect(store.observedBallTrails[0].points.map((point) => point.x)).toEqual([30, 40]);
  });
});
