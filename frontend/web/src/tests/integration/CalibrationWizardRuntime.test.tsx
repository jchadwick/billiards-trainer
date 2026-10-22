import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { apiClient } from '../../api/client';
import { CalibrationWizard } from '../../components/config/calibration/CalibrationWizard';

vi.mock('../../stores/context', () => ({ useRootStore: vi.fn() }));

const descriptor = {
  schema_version: 1 as const,
  engine_instance_id: 'engine-a',
  capture_session_id: 'session-a',
  source_frame_id: 'frame-4',
  width_px: 1920,
  height_px: 1080,
  pixel_format: 'rgb8' as const,
  image_space: 'raw' as const,
  camera_profile_id: 'camera-a',
  configuration_revision: 1,
  calibration_revision: 0,
  rectification_revision: 0,
  source_timestamp: null,
  source_timestamp_semantics: { kind: 'read_completion_only' as const },
  host_read_completed: {
    ticks: '1000',
    unit: 'nanoseconds' as const,
    clock: 'monotonic' as const,
    domain_id: 'engine-a',
    stage: 'camera_read_complete' as const,
    semantics: 'read_completion_only' as const,
  },
};

class DecodedImage {
  complete = true;
  naturalWidth = descriptor.width_px;
  naturalHeight = descriptor.height_px;
  src = '';

  async decode(): Promise<void> {}
}

describe('CalibrationWizard runtime ownership', () => {
  let originalCreateObjectURL: typeof URL.createObjectURL | undefined;
  let originalRevokeObjectURL: typeof URL.revokeObjectURL | undefined;

  beforeEach(() => {
    originalCreateObjectURL = URL.createObjectURL;
    originalRevokeObjectURL = URL.revokeObjectURL;
    Object.defineProperty(URL, 'createObjectURL', {
      configurable: true,
      value: vi.fn(() => 'blob:rust-preview'),
    });
    Object.defineProperty(URL, 'revokeObjectURL', {
      configurable: true,
      value: vi.fn(),
    });
    vi.stubGlobal('Image', DecodedImage);
  });

  afterEach(() => {
    cleanup();
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
    if (originalCreateObjectURL) {
      Object.defineProperty(URL, 'createObjectURL', {
        configurable: true,
        value: originalCreateObjectURL,
      });
    } else {
      delete (URL as URL & { createObjectURL?: typeof URL.createObjectURL }).createObjectURL;
    }
    if (originalRevokeObjectURL) {
      Object.defineProperty(URL, 'revokeObjectURL', {
        configurable: true,
        value: originalRevokeObjectURL,
      });
    } else {
      delete (URL as URL & { revokeObjectURL?: typeof URL.revokeObjectURL }).revokeObjectURL;
    }
  });

  it('uses Rust current-frame data without touching Python calibration routes', async () => {
    const getStatus = vi.spyOn(apiClient, 'getPerceptionRuntimeStatus').mockResolvedValue({
      success: true,
      data: {
        provider: 'rust',
        mode: 'live',
        preview: 'rust_current_frame',
        live_camera: true,
      },
      timestamp: '2026-10-22T00:00:00Z',
    });
    const getPreview = vi.spyOn(apiClient, 'getRustCalibrationFrame')
      .mockResolvedValueOnce({
        success: true,
        data: { image: new Blob(['jpeg'], { type: 'image/jpeg' }), descriptor },
        timestamp: '2026-10-22T00:00:00Z',
      })
      .mockResolvedValueOnce({
        success: true,
        data: {
          image: new Blob(['jpeg'], { type: 'image/jpeg' }),
          descriptor: { ...descriptor, source_frame_id: 'frame-5' },
        },
        timestamp: '2026-10-22T00:00:01Z',
      });
    const getPlayingArea = vi.spyOn(apiClient, 'getPlayingArea');
    const getCalibrationData = vi.spyOn(apiClient, 'getCalibrationData');
    const updatePlayingArea = vi.spyOn(apiClient, 'updatePlayingArea');

    render(<CalibrationWizard />);

    expect(await screen.findByText(/Frame frame-4/)).toBeTruthy();
    expect(getStatus).toHaveBeenCalledOnce();
    expect(getPreview).toHaveBeenCalledOnce();
    expect(getPlayingArea).not.toHaveBeenCalled();
    expect(getCalibrationData).not.toHaveBeenCalled();
    expect(updatePlayingArea).not.toHaveBeenCalled();
    expect(screen.getByRole('button', { name: 'Rust Calibration Save Not Connected' }).hasAttribute('disabled')).toBe(true);

    const canvas = document.querySelector('canvas');
    expect(canvas).not.toBeNull();
    Object.defineProperties(canvas, {
      clientWidth: { configurable: true, value: 640 },
      clientHeight: { configurable: true, value: 360 },
      clientLeft: { configurable: true, value: 1 },
      clientTop: { configurable: true, value: 1 },
    });
    vi.spyOn(canvas!, 'getBoundingClientRect').mockReturnValue({
      left: 100,
      top: 200,
      right: 742,
      bottom: 562,
      width: 642,
      height: 362,
      x: 100,
      y: 200,
      toJSON: () => ({}),
    });
    for (const [x, y] of [[50, 50], [590, 50], [590, 310], [50, 310]]) {
      fireEvent.click(canvas!, { clientX: 101 + x, clientY: 201 + y });
    }
    expect(screen.getByRole('button', { name: 'Set Corners (4/4)' })).toBeTruthy();

    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: 'Refresh Current Frame' }));
    });
    await waitFor(() => expect(getPreview).toHaveBeenCalledTimes(2));
    expect(await screen.findByText(/A newer camera frame was loaded/)).toBeTruthy();
    expect(screen.getByRole('button', { name: 'Set Corners (0/4)' })).toBeTruthy();
    expect(getPlayingArea).not.toHaveBeenCalled();
  });

  it('keeps the existing Python camera and calibration flow selected for the Python provider', async () => {
    vi.spyOn(apiClient, 'getPerceptionRuntimeStatus').mockResolvedValue({
      success: true,
      data: {
        provider: 'python',
        mode: 'live',
        preview: 'python_provider',
        live_camera: true,
      },
      timestamp: '2026-10-22T00:00:00Z',
    });
    const getPlayingArea = vi.spyOn(apiClient, 'getPlayingArea').mockResolvedValue({
      success: true,
      data: { corners: [], marker_dots: [], calibrated: false },
      timestamp: '2026-10-22T00:00:00Z',
    });
    vi.spyOn(apiClient, 'getCalibrationData').mockResolvedValue({
      success: true,
      data: { corners: [], is_valid: false },
      timestamp: '2026-10-22T00:00:00Z',
    });
    const getPreview = vi.spyOn(apiClient, 'getRustCalibrationFrame');

    render(<CalibrationWizard />);

    expect(await screen.findByText('Connecting to camera stream...')).toBeTruthy();
    expect(getPlayingArea).toHaveBeenCalledOnce();
    expect(getPreview).not.toHaveBeenCalled();
  });
});
