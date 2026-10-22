import { act, cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { apiClient } from '../../api/client';
import { LiveView } from '../../components/video/LiveView';

vi.mock('../../stores/context', () => ({ useRootStore: vi.fn() }));
import { useRootStore } from '../../stores/context';

afterEach(() => {
  cleanup();
  vi.useRealTimers();
  vi.restoreAllMocks();
  vi.mocked(useRootStore).mockReset();
});

describe('live view freshness', () => {
  it('ages a source observation without new messages and hides stale locations', async () => {
    vi.useFakeTimers();
    vi.spyOn(apiClient, 'getConfiguration').mockResolvedValue({
      success: true,
      data: {
        values: {
          api: {
            stream: { viewer_url: 'http://camera.test/stream' },
          },
          perception: { freshness: { maximum_observation_age_seconds: 0.4 } },
        },
      },
      timestamp: new Date().toISOString(),
    } as never);
    vi.mocked(useRootStore).mockReturnValue({
      system: { status: { isConnected: true } },
      videoStore: {
        latestGameState: {
          frame_number: 12,
          timestamp: '2026-10-21T12:00:00Z',
          game_type: 'practice',
          balls: [{
            id: '6',
            number: 6,
            position: { x: 100, y: 100, scale: [1, 1] },
            radius: 10,
            confidence: 0.95,
            is_observed: true,
          }],
          cue: null,
          observation_status: {
            source_frame_number: 12,
            frame_publication_age_seconds: 0.1,
            ball_positions_status: 'observed',
          },
        },
      },
    } as never);

    render(<LiveView />);
    await act(async () => { await Promise.resolve(); });
    expect(screen.getByText(/State age: 100 ms/).textContent).not.toContain('STALE');
    expect(screen.getByText('Ball 6')).toBeTruthy();
    expect(screen.getByRole('status').textContent).toBe('1 ball detected');
    expect(screen.getByRole('grid', { name: 'Detected ball slots' })
      .querySelectorAll('[role="gridcell"]')).toHaveLength(16);
    expect(screen.getByTestId('live-video-pane').style.aspectRatio).toBe('16 / 9');
    expect(screen.getByRole('img', { name: 'Video stream' }).getAttribute('src'))
      .toBe('http://camera.test/stream');

    // A loaded source frame determines the pane ratio. The overlay shares the
    // pane's full bounds, so markers stay aligned without an offset transform.
    vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(null as never);
    const image = screen.getByRole('img', { name: 'Video stream' });
    Object.defineProperty(image, 'naturalWidth', { configurable: true, value: 640 });
    Object.defineProperty(image, 'naturalHeight', { configurable: true, value: 480 });
    fireEvent.load(image);
    expect(screen.getByTestId('live-video-pane').style.aspectRatio).toBe('640 / 480');
    expect(document.querySelector('canvas')?.style.width).toBe('100%');
    expect(document.querySelector('canvas')?.style.height).toBe('100%');

    await act(async () => { vi.advanceTimersByTime(300); });

    expect(screen.getByText(/State age: 400 ms · STALE/)).toBeTruthy();
    expect(screen.queryByText('Ball 6')).toBeNull();
    expect(screen.getByRole('status').textContent).toBe('0 balls detected');
    expect(screen.getByText('Last known state (stale)')).toBeTruthy();
    expect(document.querySelector('canvas')?.getAttribute('data-freshness')).toBe('stale');
  });
});
