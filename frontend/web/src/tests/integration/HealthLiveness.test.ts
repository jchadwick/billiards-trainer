import { afterEach, describe, expect, it, vi } from 'vitest';
import axiosClient from '../../api/axios-client';
import { apiClient } from '../../api/client';

describe('Rust gateway liveness check', () => {
  afterEach(() => vi.restoreAllMocks());

  it('checks gateway liveness without requiring camera readiness', async () => {
    const get = vi.spyOn(axiosClient, 'get').mockResolvedValue({
      data: { status: 'alive' },
    } as never);

    const response = await apiClient.healthCheck();

    expect(get).toHaveBeenCalledWith('/api/v1/health/live');
    expect(response.success).toBe(true);
  });
});
