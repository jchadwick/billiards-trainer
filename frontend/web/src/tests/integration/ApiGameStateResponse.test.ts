import { afterEach, describe, expect, it, vi } from 'vitest';
import axiosClient from '../../api/axios-client';
import { ApiClient } from '../../api/client';
import type { SceneGameStateResponse } from '../../types/api';

const sceneResponse = {
  timestamp: '2026-10-20T14:30:00Z',
  frame_number: 1,
  balls: [
    {
      id: 'track-1',
      position: { x: 1882.4, y: 1024.7, scale: [1, 1] as [number, number] },
      velocity: null,
      is_cue_ball: null,
      is_pocketed: null,
      is_observed: true,
      confidence: 0.97,
      last_update: '2026-10-20T14:29:59.900Z',
      velocity_status: 'unavailable' as const,
      cue_ball_status: 'unavailable' as const,
      pocket_status: 'unavailable' as const,
      confidence_kind: 'detector_observation' as const,
      identity_status: 'unknown' as const,
      identity_evidence: [],
      ball_type: 'unknown' as const,
      tracking_status: 'observed' as const,
      position_kind: 'measured' as const,
      source_observation: {
        engine_instance_id: 'engine-1',
        session_id: 'session-1',
        frame_id: 'frame-1',
        timestamp: {
          ticks: '1000000000',
          unit: 'nanoseconds',
          clock: 'monotonic',
          domain_id: 'host-a',
          stage: 'shared_memory_publication',
        },
      },
      radius: 36,
      radius_unit: 'canonical_pixels' as const,
      radius_source: 'legacy_standard_pool_ball_assumption' as const,
    },
  ],
  cue: null,
  table: null,
  game_type: null,
  is_valid: true,
  confidence: 0.97,
  confidence_status: 'mean_detector_confidence' as const,
  core_metadata_status: 'unavailable' as const,
  core_metadata_timestamp: null,
  table_metadata_status: 'unavailable' as const,
  scene_calibration_revision: null,
  table_calibration_revision: null,
  events: [],
  observation_status: { ball_positions_status: 'observed' },
  perception_source: 'rust_scene' as const,
} satisfies SceneGameStateResponse;

const staleSceneResponse = {
  ...sceneResponse,
  frame_number: null,
  balls: sceneResponse.balls.map((ball) => ({
    ...ball,
    is_observed: false,
    confidence: 0,
    confidence_kind: 'unavailable' as const,
    position_kind: 'last_observed' as const,
  })),
  is_valid: false,
  confidence: null,
  confidence_status: 'unavailable' as const,
  observation_status: {
    phase: 'rust_scene_health',
    ball_positions_status: 'cached',
    freshness_status: 'source_silent',
    health_reason: 'source_silence_timeout',
  },
} satisfies SceneGameStateResponse;

describe('ApiClient current game-state response', () => {
  afterEach(() => vi.restoreAllMocks());

  it('requests the state endpoint and preserves unknown scene fields as null', async () => {
    const get = vi
      .spyOn(axiosClient, 'get')
      .mockResolvedValue({ data: sceneResponse } as never);

    const result = await new ApiClient().getCurrentGameState();

    expect(get).toHaveBeenCalledWith('/api/v1/game/state');
    expect(result.success).toBe(true);
    expect(result.data).toMatchObject({ perception_source: 'rust_scene' });
    if (result.data && 'perception_source' in result.data) {
      expect(result.data.balls[0].velocity).toBeNull();
      expect(result.data.balls[0].is_cue_ball).toBeNull();
      expect(result.data.balls[0].is_pocketed).toBeNull();
      expect(result.data.table).toBeNull();
    } else {
      throw new Error('Expected the scene response variant');
    }
  });

  it('accepts cached localization after the source goes silent', async () => {
    vi.spyOn(axiosClient, 'get').mockResolvedValue({
      data: staleSceneResponse,
    } as never);

    const result = await new ApiClient().getCurrentGameState();

    expect(result.success).toBe(true);
    expect(result.data).toMatchObject({
      perception_source: 'rust_scene',
      frame_number: null,
      is_valid: false,
      observation_status: { freshness_status: 'source_silent' },
      balls: [{ is_observed: false, position_kind: 'last_observed' }],
    });
  });
});
