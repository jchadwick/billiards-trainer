import { VideoStore } from '../../stores/VideoStore';
import { apiClient } from '../../api/client';
import { DataProcessingService } from '../../services/data-handlers';
import type { GameStateData, TrajectoryData, WebSocketMessage } from '../../types/api';
import { vi } from 'vitest';
import {
  createCoordinateTransform,
  normalizeImagePointToCanonical,
} from '../../utils/coordinates';

const frame1080Scale = [2, 2] as [number, number];
const frame4kScale = [1, 1] as [number, number];

function point(x: number, y: number, scale: [number, number]) {
  return { x, y, scale };
}

function state(scale: [number, number], canonical = false): GameStateData {
  const position = canonical ? point(1920, 1080, scale) : point(960, 540, scale);
  const tableCorners = canonical
    ? [[200, 100], [1000, 100], [1000, 500], [200, 500]]
    : [[100, 50], [500, 50], [500, 250], [100, 250]];
  const pocketCenters = canonical
    ? [[200, 100], [600, 100], [1000, 100]]
    : [[100, 50], [300, 50], [500, 50]];
  const cueTip = canonical ? point(1800, 1200, scale) : point(900, 600, scale);
  const cueTail = canonical ? point(2000, 1200, scale) : point(1000, 600, scale);

  return {
    balls: [{
      id: 'cue',
      position,
      radius: canonical ? 15 : 7.5,
      radius_scale: canonical ? frame4kScale : frame1080Scale,
      color: '#fff',
      velocity: canonical ? point(20, -10, scale) : point(10, -5, scale),
      confidence: 1,
      visible: true,
    }],
    cue: {
      position: cueTail,
      tip_position: cueTip,
      angle: 180,
      length: canonical ? 200 : 100,
      length_scale: scale,
      detected: true,
      confidence: 1,
    },
    table: {
      corners: tableCorners.map(([x, y]) => point(x, y, scale)),
      pockets: pocketCenters.map(([x, y]) => ({
        position: point(x, y, scale),
        radius: canonical ? 20 : 10,
        radius_scale: scale,
      })),
      rails: [[
        point(tableCorners[0][0], tableCorners[0][1], scale),
        point(tableCorners[1][0], tableCorners[1][1], scale),
      ]],
      calibrated: true,
      dimensions: { width: 2.54, height: 1.27, unit: 'meters' },
    },
    ball_count: 1,
    phase: 'aiming',
    map_trusted: true,
    aim_calculation_eligible: true,
  };
}

function apiTableState(scale: [number, number], canonical = false): GameStateData {
  const sourcePositions = [
    [100, 50], [600, 50], [1100, 50],
    [100, 550], [600, 550], [1100, 550],
  ];
  const coordinate = (value: number) => canonical ? value * 2 : value;
  const positionScale = canonical ? frame4kScale : frame1080Scale;
  const radiusScale = canonical ? frame4kScale : frame1080Scale;
  return {
    balls: [{
      id: 'cue',
      number: 0,
      is_cue_ball: true,
      is_pocketed: false,
      position: point(canonical ? 1920 : 960, canonical ? 1080 : 540, positionScale),
      velocity: point(canonical ? 20 : 10, canonical ? -10 : -5, positionScale),
      radius: canonical ? 15 : 7.5,
      radius_scale: radiusScale,
      radius_unit: 'canonical_pixels',
      confidence: 1,
    }],
    table: {
      width: canonical ? 2000 : 1000,
      height: canonical ? 1000 : 500,
      dimensions_scale: canonical ? frame4kScale : frame1080Scale,
      dimensions_unit: 'canonical_pixels',
      pocket_positions: sourcePositions.map(([x, y]) => point(coordinate(x), coordinate(y), positionScale)),
      pocket_radius: canonical ? 20 : 10,
      pocket_radius_scale: radiusScale,
      pocket_radius_unit: 'canonical_pixels',
      real_world_dimensions: { width: 2.54, height: 1.27, unit: 'meters' },
    },
    ball_count: 1,
    frame_number: 1,
  };
}

function message(type: 'state' | 'trajectory', data: Record<string, unknown>): WebSocketMessage {
  return { type, timestamp: new Date(0).toISOString(), data };
}

function trajectory(scale: [number, number], canonical = false): TrajectoryData {
  return {
    lines: [{
      start: canonical ? point(1920, 1080, scale) : point(960, 540, scale),
      end: canonical ? point(2400, 1200, scale) : point(1200, 600, scale),
      type: 'primary',
      confidence: 1,
    }],
    collisions: [{
      position: canonical ? point(2400, 1200, scale) : point(1200, 600, scale),
      ball2_id: '1',
      angle: 45,
    }],
    confidence: 1,
    calculation_time_ms: 1,
    line_count: 1,
    collision_count: 1,
    prediction_status: 'live',
  };
}

describe('canonical image coordinate consumption', () => {
  let store: VideoStore;

  beforeEach(() => {
    vi.spyOn(apiClient, 'stopVideoCapture').mockResolvedValue({
      success: true,
      timestamp: new Date().toISOString(),
    });
    store = new VideoStore();
  });

  afterEach(async () => {
    store.dispose();
    await new Promise<void>((resolve) => setTimeout(resolve, 0));
    vi.restoreAllMocks();
  });

  function populate(scale: [number, number], is4k: boolean) {
    (store as any).handleGameStateMessage(message('state', state(scale, is4k) as unknown as Record<string, unknown>));
    (store as any).handleTrajectoryMessage(message('trajectory', trajectory(scale, is4k) as unknown as Record<string, unknown>));
    return store.currentFrame!;
  }

  test('normalizes ball, velocity, radius, cue, table boundaries, rails, trajectory and collisions once', () => {
    const from1080 = populate(frame1080Scale, false);
    const first = {
      ball: from1080.balls[0],
      cue: from1080.cue,
      table: from1080.table,
      trajectory: from1080.trajectories[0],
    };

    store.dispose();
    store = new VideoStore();
    const from4k = populate(frame4kScale, true);

    expect(first.ball.position).toEqual(from4k.balls[0].position);
    expect(first.ball.velocity).toEqual(from4k.balls[0].velocity);
    expect(first.ball.radius).toBe(15);
    expect(first.ball.radiusX).toBe(15);
    expect(first.ball.radiusY).toBe(15);
    expect(first.cue?.tipPosition).toEqual(from4k.cue?.tipPosition);
    expect(first.cue?.tailPosition).toEqual(from4k.cue?.tailPosition);
    expect(first.cue?.length).toBe(200);
    expect(first.table?.corners).toEqual(from4k.table?.corners);
    expect(first.table?.bounds).toEqual(from4k.table?.bounds);
    expect(first.table?.rails).toEqual(from4k.table?.rails);
    expect(first.table?.pockets).toEqual(from4k.table?.pockets);
    expect(first.table?.pocketRadii).toEqual(from4k.table?.pocketRadii);
    expect(first.table?.dimensions).toEqual({ width: 2.54, height: 1.27, unit: 'meters' });
    expect(first.trajectory?.points).toEqual(from4k.trajectories[0].points);
    expect(first.trajectory?.collisions).toEqual(from4k.trajectories[0].collisions);
  });

  test('uses explicit identity and converts nanosecond source times for Rust scenes', () => {
    const sceneState = (
      ticks: string,
      sessionId = 'scene-session-1',
      engineInstanceId = 'scene-engine-1',
    ): GameStateData => ({
      balls: [
        {
          id: '8',
          number: null,
          position: point(1882.4, 1024.7, frame4kScale),
          radius: 36,
          radius_scale: frame4kScale,
          radius_unit: 'canonical_pixels',
          confidence: 0.97,
          is_observed: true,
          tracking_status: 'observed',
          position_kind: 'measured',
          image_space: 'raw',
          image_width_px: 3840,
          image_height_px: 2160,
          identity_status: 'unknown',
          identity_confidence: null,
          identity_evidence: [],
          source_observation: {
            engine_instance_id: engineInstanceId,
            session_id: sessionId,
            frame_id: `frame-${ticks}`,
          },
        },
        {
          id: 'track-6',
          number: 6,
          position: point(1200, 800, frame4kScale),
          radius: 36,
          radius_scale: frame4kScale,
          radius_unit: 'canonical_pixels',
          confidence: 0.91,
          is_observed: true,
          tracking_status: 'observed',
          position_kind: 'measured',
          image_space: 'raw',
          image_width_px: 3840,
          image_height_px: 2160,
          table_position: {
            x: 18.25,
            y: -42.5,
            unit: 'millimeters',
            origin: 'playing_surface_center',
          },
        },
      ],
      ball_count: 2,
      observation_status: {
        ball_positions_status: 'observed',
        source_timestamp: ticks,
        source_timestamp_unit: 'nanoseconds',
        source_timestamp_kind: 'monotonic:rust-run:shared_memory_publication',
        ball_observation_source_timestamp: ticks,
        ball_observation_source_timestamp_unit: 'nanoseconds',
        ball_observation_source_timestamp_kind: 'monotonic:rust-run:shared_memory_publication',
        phase: 'rust_scene_observation',
        observation_identity: {
          engine_instance_id: engineInstanceId,
          session_id: sessionId,
          frame_id: `frame-${ticks}`,
        },
      },
    });

    (store as any).handleGameStateMessage(
      message('state', sceneState('1000000000') as unknown as Record<string, unknown>),
    );
    const first = store.currentBalls;
    (store as any).handleGameStateMessage(
      message('state', sceneState('1100000000') as unknown as Record<string, unknown>),
    );

    const unknownTrack = store.currentBalls.find((ball) => ball.id === '8');
    const numberedTrack = store.currentBalls.find((ball) => ball.id === 'track-6');
    expect(unknownTrack?.number).toBeUndefined();
    expect(unknownTrack?.type).toBe('unknown');
    expect(unknownTrack?.identityStatus).toBe('unknown');
    expect(numberedTrack?.number).toBe(6);
    expect(numberedTrack?.type).toBe('solid');
    expect(numberedTrack?.tablePositionMm).toEqual({ x: 18.25, y: -42.5 });
    expect(first.find((ball) => ball.id === '8')?.type).toBe('unknown');
    expect(store.observedBallTrails).toHaveLength(2);
    expect(store.observedBallTrails[0].points.map((point) => point.timestamp)).toEqual([
      1,
      1.1,
    ]);

    const missingUnit = sceneState('1200000000');
    missingUnit.observation_status!.ball_observation_source_timestamp_unit = undefined;
    missingUnit.observation_status!.source_timestamp_unit = undefined;
    (store as any).handleGameStateMessage(
      message('state', missingUnit as unknown as Record<string, unknown>),
    );
    expect(store.observedBallTrails[0].points).toHaveLength(2);
  });

  test('starts a new measured trail when the Rust engine or source session restarts', () => {
    const sceneState = (
      ticks: string,
      sessionId: string,
      engineInstanceId = 'engine-1',
    ): GameStateData => ({
      balls: [{
        id: 'track-reused',
        number: null,
        position: point(1882.4, 1024.7, frame4kScale),
        radius: 36,
        radius_scale: frame4kScale,
        radius_unit: 'canonical_pixels',
        confidence: 0.8,
        is_observed: true,
        tracking_status: 'observed',
        position_kind: 'measured',
        image_space: 'raw',
        image_width_px: 3840,
        image_height_px: 2160,
        source_observation: {
          engine_instance_id: engineInstanceId,
          session_id: sessionId,
          frame_id: `frame-${ticks}`,
        },
      }],
      ball_count: 1,
      observation_status: {
        phase: 'rust_scene_observation',
        ball_positions_status: 'observed',
        source_timestamp: ticks,
        source_timestamp_unit: 'nanoseconds',
        ball_observation_source_timestamp: ticks,
        ball_observation_source_timestamp_unit: 'nanoseconds',
        ball_observation_source_timestamp_kind: 'monotonic:clock:shared_memory_publication',
        observation_identity: {
          engine_instance_id: engineInstanceId,
          session_id: sessionId,
        },
      },
    });

    (store as any).handleGameStateMessage(
      message('state', sceneState('1000000000', 'session-old') as unknown as Record<string, unknown>),
    );
    (store as any).handleGameStateMessage(
      message('state', sceneState('500000000', 'session-new') as unknown as Record<string, unknown>),
    );

    expect(store.observedBallTrails).toHaveLength(1);
    expect(store.observedBallTrails[0].points).toHaveLength(1);
    expect(store.observedBallTrails[0].points[0].timestamp).toBe(0.5);
    expect(store.observedBallTrails[0].points[0].gapBefore).toBe(false);

    (store as any).handleGameStateMessage(
      message(
        'state',
        sceneState('250000000', 'session-new', 'engine-restarted') as unknown as Record<string, unknown>,
      ),
    );
    expect(store.observedBallTrails).toHaveLength(1);
    expect(store.observedBallTrails[0].points).toHaveLength(1);
    expect(store.observedBallTrails[0].points[0].timestamp).toBe(0.25);
    expect(store.currentBalls[0].sourceObservation?.engine_instance_id).toBe(
      'engine-restarted',
    );
  });

  test('rejects rectified Rust image points from the raw-frame overlay path', () => {
    const rectifiedRustState = state(frame4kScale, true) as any;
    rectifiedRustState.observation_status = {
      phase: 'rust_scene_observation',
      ball_positions_status: 'observed',
    };
    rectifiedRustState.balls[0].image_space = 'rectified';

    (store as any).handleGameStateMessage(
      message('state', rectifiedRustState as Record<string, unknown>),
    );

    expect(store.currentBalls).toHaveLength(0);
  });

  test('converts explicit timestamp units and rejects unknown Rust units', () => {
    const toSeconds = (store as any).toTimestampSeconds.bind(store);
    expect(toSeconds('1500', 'milliseconds', true)).toBe(1.5);
    expect(toSeconds('1.5', 'seconds', true)).toBe(1.5);
    expect(toSeconds('1500000000', 'nanoseconds', true)).toBe(1.5);
    expect(toSeconds('1500', 'ticks', true)).toBeNull();
    expect(toSeconds('1500000000', undefined, true)).toBeNull();
  });

  test('health failure retains the last location as explicitly stale', () => {
    const failedState: GameStateData = {
      balls: [{
        id: 'track-1',
        number: null,
        position: point(1882.4, 1024.7, frame4kScale),
        radius: 36,
        radius_scale: frame4kScale,
        radius_unit: 'canonical_pixels',
        confidence: 0,
        confidence_kind: 'unavailable',
        last_observation_confidence: 0.97,
        is_observed: false,
        tracking_status: 'observed',
        position_kind: 'measured',
        image_space: 'raw',
        image_width_px: 3840,
        image_height_px: 2160,
        source_observation: {
          engine_instance_id: 'engine-1',
          session_id: 'session-1',
          frame_id: 'frame-1',
        },
      }],
      ball_count: 1,
      observation_status: {
        phase: 'rust_scene_health',
        ball_positions_status: 'cached',
        ball_detection_status: 'failed',
        health_reason: 'inference_timeout',
        observation_identity: {
          engine_instance_id: 'engine-1',
          session_id: 'session-1',
          frame_id: 'frame-2',
        },
      },
    };

    (store as any).handleGameStateMessage(
      message('state', failedState as unknown as Record<string, unknown>),
    );

    expect(store.currentBalls[0]).toMatchObject({
      isObserved: false,
      confidence: 0,
      confidenceKind: 'unavailable',
      lastObservationConfidence: 0.97,
    });
    expect(store.latestObservationStatus?.ball_positions_status).toBe('cached');
    expect(store.observedBallTrails).toHaveLength(0);
  });

  test('maps canonical points to equal canvas positions over 1080p and 4K frames', () => {
    const canonical = { x: 1920, y: 1080 };
    const view = { width: 960, height: 540 };
    const transform1080 = createCoordinateTransform(
      { width: 1920, height: 1080 }, view,
    );
    const transform4k = createCoordinateTransform(
      { width: 3840, height: 2160 }, view,
    );

    expect(transform1080.videoToCanvas(canonical)).toEqual({ x: 480, y: 270 });
    expect(transform4k.videoToCanvas(canonical)).toEqual({ x: 480, y: 270 });
    expect(transform1080.canvasToVideo({ x: 480, y: 270 })).toEqual(canonical);
    expect(transform4k.canvasToVideo({ x: 480, y: 270 })).toEqual(canonical);
    expect(transform1080.videoExtentToCanvasX(20)).toBe(transform4k.videoExtentToCanvasX(20));
  });

  test('data processing normalizes ball, velocity, radii, table bounds and trajectory endpoints', () => {
    function process(scale: [number, number], is4k: boolean) {
      const service = new DataProcessingService();
      let processedState: ReturnType<DataProcessingService['getCurrentGameState']> = null;
      let processedTrajectory: Parameters<Parameters<DataProcessingService['onTrajectory']>[0]>[0] | null = null;
      service.onGameState((next) => { processedState = next; });
      service.onTrajectory((next) => { processedTrajectory = next; });
      service.processMessage(message('state', state(scale, is4k) as unknown as Record<string, unknown>));
      service.processMessage(message('trajectory', trajectory(scale, is4k) as unknown as Record<string, unknown>));
      return { processedState, processedTrajectory };
    }

    const from1080 = process(frame1080Scale, false);
    const from4k = process(frame4kScale, true);

    expect(from1080.processedState?.balls[0].position).toEqual(from4k.processedState?.balls[0].position);
    expect(from1080.processedState?.balls[0].velocity).toEqual(from4k.processedState?.balls[0].velocity);
    expect(from1080.processedState?.balls[0].radiusX).toBe(15);
    expect(from1080.processedState?.balls[0].radiusY).toBe(15);
    expect(from1080.processedState?.table?.bounds).toEqual(from4k.processedState?.table?.bounds);
    expect(from1080.processedState?.table?.bounds).toEqual({ minX: 200, maxX: 1000, minY: 100, maxY: 500 });
    expect(from1080.processedTrajectory?.smoothedLines).toEqual(from4k.processedTrajectory?.smoothedLines);
    expect(from1080.processedTrajectory?.collisionPredictions).toEqual(from4k.processedTrajectory?.collisionPredictions);
  });

  test('consumes API/Core table units and ball radius scale consistently in both frontend paths', () => {
    const from1080 = apiTableState(frame1080Scale);
    const from4k = apiTableState(frame4kScale, true);

    function process(data: GameStateData) {
      const service = new DataProcessingService();
      let result: ReturnType<DataProcessingService['getCurrentGameState']> = null;
      service.onGameState((next) => { result = next; });
      service.processMessage(message('state', data as unknown as Record<string, unknown>));
      return result;
    }

    const processed1080 = process(from1080);
    const processed4k = process(from4k);
    expect(processed1080?.balls[0].position).toEqual(processed4k?.balls[0].position);
    expect(processed1080?.balls[0].radius).toBe(15);
    expect(processed1080?.balls[0].radiusX).toBe(15);
    expect(processed1080?.table?.width).toBe(2000);
    expect(processed1080?.table?.height).toBe(1000);
    expect(processed1080?.table?.width).toBe(processed4k?.table?.width);
    expect(processed1080?.table?.height).toBe(processed4k?.table?.height);
    expect(processed1080?.table?.pockets?.[0]).toMatchObject({
      radius: 20,
      radius_extent: { x: 20, y: 20, scale: [1, 1] },
    });
    expect(processed1080?.table).toMatchObject({
      pocket_radius: 20,
      pocket_radius_scale: [1, 1],
      pocket_radius_unit: 'canonical_pixels',
    });
    expect(processed1080?.table?.dimensions).toEqual({
      width: 2.54,
      height: 1.27,
      unit: 'meters',
    });

    (store as any).handleGameStateMessage(message('state', from1080 as unknown as Record<string, unknown>));
    const stored1080 = store.currentFrame!;
    store.dispose();
    store = new VideoStore();
    (store as any).handleGameStateMessage(message('state', from4k as unknown as Record<string, unknown>));
    const stored4k = store.currentFrame!;
    expect(stored1080.balls[0].position).toEqual(stored4k.balls[0].position);
    expect(stored1080.balls[0].radius).toBe(15);
    expect(stored1080.table?.width).toBe(2000);
    expect(stored1080.table?.height).toBe(1000);
    expect(stored1080.table?.corners).toEqual([]);
    expect(stored1080.table?.bounds).toBeUndefined();
    expect(stored1080.table?.bounds).toEqual(stored4k.table?.bounds);
    expect(stored1080.table?.pocketRadii?.[0]).toEqual({ x: 20, y: 20 });
    expect(stored1080.table?.dimensions).toEqual({
      width: 2.54,
      height: 1.27,
      unit: 'meters',
    });
  });

  test('rejects canonical coordinate labels and pixel units without explicit scale', () => {
    const invalidState = state(frame4kScale, true) as any;
    const missingScalePoint = { ...invalidState.balls[0].position };
    delete missingScalePoint.scale;
    missingScalePoint.coordinate_space = '4k_canonical';
    expect(normalizeImagePointToCanonical(missingScalePoint)).toBeNull();
    invalidState.balls = [];
    invalidState.table.width = 1000;
    invalidState.table.height = 500;
    delete invalidState.table.width_scale;
    delete invalidState.table.height_scale;
    delete invalidState.table.dimensions_scale;
    invalidState.table.dimensions_unit = 'canonical_pixels';

    const service = new DataProcessingService();
    let processed: ReturnType<DataProcessingService['getCurrentGameState']> = null;
    service.onGameState((next) => { processed = next; });
    service.processMessage(message('state', invalidState));
    expect(processed?.balls).toHaveLength(0);
    expect(processed?.table?.width).toBeUndefined();
    expect(processed?.table?.height).toBeUndefined();

    (store as any).handleGameStateMessage(message('state', invalidState));
    expect(store.currentBalls).toHaveLength(0);
    expect(store.currentTable?.width).toBeUndefined();
    expect(store.currentTable?.height).toBeUndefined();
  });

  test('uses explicit frame dimensions only at source-frame coordinate boundaries', () => {
    const sourceResolution = { width: 1920, height: 1080 };
    expect(normalizeImagePointToCanonical([960, 540], sourceResolution)).toEqual({
      x: 1920,
      y: 1080,
    });
    expect(normalizeImagePointToCanonical(
      { x: 960, y: 540, coordinate_space: 'pixel' },
      sourceResolution,
    )).toEqual({ x: 1920, y: 1080 });
    expect(normalizeImagePointToCanonical(
      { x: 960, y: 540, coordinate_space: '4k_canonical' },
      sourceResolution,
    )).toBeNull();
    expect(normalizeImagePointToCanonical({ x: 960, y: 540 }, sourceResolution)).toBeNull();
  });

  test('drops extents and radii with explicit unsupported units', () => {
    const invalid = apiTableState(frame1080Scale) as any;
    invalid.balls[0].radius_unit = 'meters';
    invalid.table.dimensions_unit = 'meters';
    invalid.table.pocket_radius_unit = 'meters';

    const service = new DataProcessingService();
    let processed: ReturnType<DataProcessingService['getCurrentGameState']> = null;
    service.onGameState((next) => { processed = next; });
    service.processMessage(message('state', invalid));
    expect(processed?.balls).toHaveLength(0);
    expect(processed?.table?.width).toBeUndefined();
    expect(processed?.table?.height).toBeUndefined();
    expect((processed?.table?.pockets?.[0] as any)?.radius).toBeUndefined();

    (store as any).handleGameStateMessage(message('state', invalid));
    expect(store.currentBalls).toHaveLength(0);
    expect(store.currentTable?.width).toBeUndefined();
    expect(store.currentTable?.height).toBeUndefined();
    expect(store.currentTable?.pocketRadii?.[0]).toEqual({ x: 0, y: 0 });
  });

  test('keeps explicit independent width and height scales and supplied playing-area corners', () => {
    const legacy = state(frame1080Scale) as GameStateData;
    legacy.table = {
      ...legacy.table!,
      width: 300,
      width_scale: [2, 3],
      height: 100,
      height_scale: [4, 5],
      playing_area_corners: [
        point(10, 10, frame1080Scale),
        point(310, 10, frame1080Scale),
        point(310, 110, frame1080Scale),
        point(10, 110, frame1080Scale),
      ],
    };
    const service = new DataProcessingService();
    let processed: ReturnType<DataProcessingService['getCurrentGameState']> = null;
    service.onGameState((next) => { processed = next; });
    service.processMessage(message('state', legacy as unknown as Record<string, unknown>));
    expect(processed?.table?.width).toBe(600);
    expect(processed?.table?.height).toBe(500);
    expect(processed?.table?.corners?.[0]).toEqual({ x: 20, y: 20, scale: [1, 1] });

    (store as any).handleGameStateMessage(message('state', legacy as unknown as Record<string, unknown>));
    expect(store.currentTable?.width).toBe(600);
    expect(store.currentTable?.height).toBe(500);
    expect(store.currentTable?.corners[0]).toEqual({ x: 20, y: 20 });
    expect(store.currentTable?.bounds).toEqual({ x: 20, y: 20, width: 600, height: 200 });
  });

  test('rejects invalid scales instead of drawing unscaled or non-finite geometry', () => {
    const invalid = state(frame4kScale, true) as any;
    invalid.balls[0].position.scale = [0, 1];
    invalid.table.corners[0].scale = [Number.NaN, 1];
    (store as any).handleGameStateMessage(message('state', invalid));

    expect(store.currentBalls).toHaveLength(0);
    expect(store.currentTable).toBeNull();

    (store as any).handleTrajectoryMessage(message('trajectory', {
      ...trajectory(frame4kScale, true),
      lines: [{
        start: { x: 1, y: 2, scale: [0, 1] },
        end: point(20, 30, frame4kScale),
        type: 'primary',
        confidence: 1,
      }],
    }));
    expect(store.currentTrajectories).toHaveLength(0);
  });
});
