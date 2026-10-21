import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import axiosClient from '../../api/axios-client';
import { apiClient, ApiClient } from '../../api/client';
import { ScreenReaderOnly } from '../../components/accessibility/ScreenReaderOnly';
import { ConfirmModal } from '../../components/ui/Modal';
import { LiveView } from '../../components/video/LiveView';
import { getConfiguredLiveStreamUrl } from '../../components/video/live-stream-url';

vi.mock('../../stores/context', () => ({ useRootStore: vi.fn() }));
import { useRootStore } from '../../stores/context';

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
  vi.unstubAllEnvs();
  vi.mocked(useRootStore).mockReset();
});

describe('retained frontend contracts', () => {
  it('uses versioned monitoring paths and preserves payloads and failures', async () => {
    const get = vi.spyOn(axiosClient, 'get').mockResolvedValueOnce({ data: { components: {} } });
    const client = new ApiClient();
    expect(await client.get('/health?include_details=true')).toEqual({ components: {} });
    expect(get).toHaveBeenCalledWith('/api/v1/health?include_details=true');

    const failure = new Error('unavailable');
    get.mockRejectedValueOnce(failure);
    await expect(client.get('/health/metrics')).rejects.toBe(failure);
  });

  it('forwards a screen-reader status role and live-region attributes', () => {
    render(<ScreenReaderOnly role="status" live="polite" atomic>Scene received</ScreenReaderOnly>);
    const status = screen.getByRole('status');
    expect(status.textContent).toBe('Scene received');
    expect(status.getAttribute('aria-live')).toBe('polite');
    expect(status.getAttribute('aria-atomic')).toBe('true');
  });

  it('waits for an asynchronous confirmation before closing', async () => {
    let complete!: () => void;
    const pending = new Promise<void>(resolve => { complete = resolve; });
    const onClose = vi.fn();
    const onConfirm = vi.fn(() => pending);
    render(<ConfirmModal isOpen onClose={onClose} onConfirm={onConfirm} title="Delete profile" message="Confirm deletion" />);
    fireEvent.click(screen.getByRole('button', { name: 'Confirm' }));
    expect(onConfirm).toHaveBeenCalledOnce();
    expect(onClose).not.toHaveBeenCalled();
    await act(async () => { complete(); await pending; });
    expect(onClose).toHaveBeenCalledOnce();
  });

  it('keeps confirmation controls disabled while the operation is loading', () => {
    const onConfirm = vi.fn();
    render(<ConfirmModal isOpen loading onClose={vi.fn()} onConfirm={onConfirm} title="Delete profile" message="Confirm deletion" />);
    const confirm = screen.getByRole('button', { name: 'Confirm' });
    expect(confirm.hasAttribute('disabled')).toBe(true);
    expect(screen.getByRole('button', { name: 'Cancel' }).hasAttribute('disabled')).toBe(true);
    fireEvent.click(confirm);
    expect(onConfirm).not.toHaveBeenCalled();
  });

  it('uses the API video stream when no runtime viewer URI is configured', async () => {
    const createSocket = vi.spyOn(apiClient, 'createWebSocket');
    vi.spyOn(apiClient, 'getVideoStreamUrl').mockReturnValue('http://configured.test/api/v1/stream/video');
    vi.spyOn(apiClient, 'getConfiguration').mockResolvedValue({
      success: true,
      data: { values: { api: { stream: { viewer_url: null } } } },
      timestamp: new Date().toISOString(),
    } as never);
    const store = {
      system: { status: { isConnected: true } },
      videoStore: { latestGameState: null },
    };
    vi.mocked(useRootStore).mockReturnValue(store as never);
    const view = render(<LiveView autoConnect={false} />);
    expect(createSocket).not.toHaveBeenCalled();
    expect(screen.getByAltText('Video stream').hasAttribute('src')).toBe(false);
    view.rerender(<LiveView />);
    expect(createSocket).not.toHaveBeenCalled();
    await waitFor(() => {
      expect(screen.getByAltText('Video stream').getAttribute('src')).toBe(
        'http://configured.test/api/v1/stream/video',
      );
    });
    view.rerender(<LiveView autoConnect={false} />);
    expect(createSocket).not.toHaveBeenCalled();
    expect(screen.getByAltText('Video stream').hasAttribute('src')).toBe(false);
  });

  it('loads the camera image directly from the runtime viewer URL', async () => {
    const getApiVideoUrl = vi.spyOn(apiClient, 'getVideoStreamUrl');
    vi.spyOn(apiClient, 'getConfiguration').mockResolvedValue({
      success: true,
      data: { values: { api: { stream: { viewer_url: 'http://camera.test:8080/stream' } } } },
      timestamp: new Date().toISOString(),
    } as never);
    vi.mocked(useRootStore).mockReturnValue({
      system: { status: { isConnected: true } },
      videoStore: { latestGameState: null },
    } as never);

    render(<LiveView />);

    await waitFor(() => {
      expect(screen.getByAltText('Video stream').getAttribute('src')).toBe(
        'http://camera.test:8080/stream',
      );
    });
    expect(getApiVideoUrl).not.toHaveBeenCalled();
  });

  it('rejects credentials, fragments, and non-HTTP camera stream URLs', () => {
    expect(getConfiguredLiveStreamUrl('https://viewer:secret@camera.test/stream')).toBeUndefined();
    expect(getConfiguredLiveStreamUrl('http://camera.test/stream#frame')).toBeUndefined();
    expect(getConfiguredLiveStreamUrl('javascript:alert(1)')).toBeUndefined();
    expect(getConfiguredLiveStreamUrl('http://camera.test/stream')).toBe('http://camera.test/stream');
  });
});
