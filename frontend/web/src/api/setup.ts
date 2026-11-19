import client from './axios-client';

export type CalibrationCheck = {
  state: 'ready' | 'missing' | 'invalid' | 'blocked' | 'unverified';
  detail?: string;
};

export type SetupStatus = {
  state: 'idle' | 'starting' | 'running' | 'error' | 'disabled';
  error?: string | null;
  configuration_revision?: number;
  configured_source?: string | null;
  calibrated?: boolean;
  precision_calibration?: { optics: CalibrationCheck; ball_centers: CalibrationCheck };
};

export async function getSetupStatus(signal?: AbortSignal): Promise<SetupStatus> {
  const response = await client.get<SetupStatus>('/api/v1/setup', { signal });
  const status = response.data;
  if (!status || !['idle', 'starting', 'running', 'error', 'disabled'].includes(status.state)) {
    throw new Error('The gateway did not return camera setup status.');
  }
  return status;
}
