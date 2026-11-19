import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { ConfigurationPage } from './ConfigurationPage';
const mocks = vi.hoisted(() => ({ get: vi.fn(), put: vi.fn(), notifyVideoConfigurationApplied: vi.fn(), recordConfigurationApplied: vi.fn() }));
vi.mock('../../api/axios-client', () => ({ default: { get: mocks.get, getInstance: () => ({ put: mocks.put }) } }));
vi.mock('../../stores/context', () => ({ useRootStore: () => ({ notifyVideoConfigurationApplied: mocks.notifyVideoConfigurationApplied, recordConfigurationApplied: mocks.recordConfigurationApplied }) }));
vi.mock('./MonacoConfigurationEditor', () => ({ MonacoConfigurationEditor: ({ value, original, onChange }: any) => original === undefined ? <textarea aria-label="config editor" value={value} onChange={event => onChange(event.target.value)} /> : <div>Diff review</div> }));
beforeEach(() => vi.clearAllMocks());
describe('configuration page lifecycle', () => {
  it('shows a recoverable message when the server returns a non-configuration response', async () => {
    mocks.get.mockResolvedValue({ data: '<!doctype html><html><body>Vite</body></html>' });
    render(<ConfigurationPage />);

    expect(await screen.findByText(/Configuration API returned an unexpected response/)).toBeTruthy();
    expect(screen.getByText('Configuration unavailable. Reload active to retry.')).toBeTruthy();
    expect(screen.queryByRole('textbox', { name: 'config editor' })).toBeNull();
  });

  it('loads schema, preserves failed validation/save drafts, and applies stripped data then reloads canonical config', async () => {
    let values = { metadata: { configuration_revision: 7 }, logging: { level: 'info' } };
    mocks.get.mockImplementation(async (url: string) => ({ data: url.endsWith('/schema') ? { type: 'object' } : { values } }));
    mocks.put.mockResolvedValueOnce({ data: { success: false, validation_errors: [{ field: 'logging.level', message: 'Invalid logging level' }], warnings: [] } });
    render(<ConfigurationPage />);
    const editor = await screen.findByRole('textbox', { name: 'config editor' });
    expect(mocks.get).toHaveBeenCalledWith('/api/v1/config/schema');
    // An idle editor must not silently replace drafts or poll the configuration.
    const initialReads = mocks.get.mock.calls.length;
    vi.useFakeTimers();
    try {
      act(() => vi.advanceTimersByTime(60_000));
      expect(mocks.get).toHaveBeenCalledTimes(initialReads);
      expect(mocks.put).not.toHaveBeenCalled();
    } finally {
      vi.useRealTimers();
    }
    const draft = JSON.stringify({ metadata: { configuration_revision: 7 }, logging: { level: 'debug' } });
    fireEvent.change(editor, { target: { value: draft } });
    fireEvent.click(screen.getByRole('button', { name: 'Review changes' }));
    await screen.findByText('logging.level: Invalid logging level');
    expect((screen.getByRole('textbox', { name: 'config editor' }) as HTMLTextAreaElement).value).toBe(draft);
    mocks.put.mockResolvedValueOnce({ data: { success: true, validation_errors: [], warnings: [] } });
    fireEvent.click(screen.getByRole('button', { name: 'Review changes' }));
    await screen.findByText('Diff review');
    expect(mocks.put).toHaveBeenNthCalledWith(2, '/api/v1/config', { metadata: {}, logging: { level: 'debug' } }, { params: { validate_only: true } });
    expect(values.metadata.configuration_revision).toBe(7);
    mocks.put.mockResolvedValueOnce({ data: { success: false, validation_errors: [{ field: 'logging.level', message: 'Rejected by server' }], warnings: [] } });
    fireEvent.click(screen.getByRole('button', { name: 'Apply configuration' }));
    await screen.findByText('Configuration was rejected. Your draft is preserved.');
    expect(mocks.recordConfigurationApplied).not.toHaveBeenCalled();
    mocks.put.mockRejectedValueOnce(new Error('Connection lost'));
    fireEvent.click(screen.getByRole('button', { name: 'Apply configuration' }));
    await screen.findByText(/Apply was not confirmed: Connection lost/);
    fireEvent.click(screen.getByRole('button', { name: 'Back to editor' }));
    expect((screen.getByRole('textbox', { name: 'config editor' }) as HTMLTextAreaElement).value).toBe(draft);
    mocks.put.mockImplementation(async (_url: string, _body: unknown, options: any) => {
      if (!options.params.validate_only) values = { metadata: { configuration_revision: 8 }, logging: { level: 'debug' } };
      return { data: { success: true, validation_errors: [], warnings: [], restart_required: true } };
    });
    fireEvent.click(screen.getByRole('button', { name: 'Review changes' }));
    await screen.findByText('Diff review');
    fireEvent.click(screen.getByRole('button', { name: 'Apply configuration' }));
    await screen.findByText('Applied · restart required. Arrange restart outside this interface.');
    await waitFor(() => expect((screen.getByRole('textbox', { name: 'config editor' }) as HTMLTextAreaElement).value).toContain('8'));
    expect(mocks.put).toHaveBeenLastCalledWith('/api/v1/config', { metadata: {}, logging: { level: 'debug' } }, { params: { force_update: false } });
    expect(mocks.notifyVideoConfigurationApplied).toHaveBeenCalledTimes(1);
    expect(mocks.recordConfigurationApplied).toHaveBeenCalledTimes(1);
    expect(mocks.recordConfigurationApplied).toHaveBeenCalledWith({
      changedFieldPaths: ['/logging/level'],
      restartRequired: true,
      previousRevision: 7,
      appliedRevision: undefined,
    });
  });

  it('records one confirmed interface action before a failed canonical reload and omits setting values', async () => {
    const baseline = { metadata: { configuration_revision: 8 }, logging: { level: 'debug' } };
    let configReads = 0;
    mocks.get.mockImplementation(async (url: string) => {
      if (url.endsWith('/schema')) return { data: { type: 'object', properties: { metadata: { type: 'object', properties: { configuration_revision: { readOnly: true } } }, logging: { type: 'object', properties: { level: { type: 'string' } } } } } };
      configReads += 1;
      if (configReads <= 1) return { data: { values: baseline } };
      throw new Error('Canonical reload unavailable');
    });
    mocks.put.mockImplementation(async (_url: string, _body: unknown, options: any) => options.params.validate_only
      ? { data: { success: true, validation_errors: [], warnings: [] } }
      : { data: { success: true, restart_required: false, updated_fields: ['logging.level'], validation_errors: [], warnings: [] } });

    render(<ConfigurationPage />);
    const editor = await screen.findByRole('textbox', { name: 'config editor' });
    fireEvent.change(editor, { target: { value: JSON.stringify({ metadata: { configuration_revision: 8 }, logging: { level: 'super-secret-setting' } }) } });
    fireEvent.click(screen.getByRole('button', { name: 'Review changes' }));
    await screen.findByText('Diff review');
    fireEvent.click(screen.getByRole('button', { name: 'Apply configuration' }));
    await screen.findByText(/Apply was confirmed, but the active document could not be reloaded/);

    expect(mocks.recordConfigurationApplied).toHaveBeenCalledTimes(1);
    const eventArgument = mocks.recordConfigurationApplied.mock.calls[0][0];
    expect(eventArgument).toEqual({
      changedFieldPaths: ['logging.level'],
      restartRequired: false,
      previousRevision: 8,
      appliedRevision: undefined,
    });
    expect(JSON.stringify(eventArgument)).not.toContain('super-secret-setting');
    expect(mocks.notifyVideoConfigurationApplied).toHaveBeenCalledTimes(1);
  });
});
