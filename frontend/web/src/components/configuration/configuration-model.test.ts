import { describe, expect, it } from 'vitest';
import { changedConfigurationPaths, deletedPaths, editableConfiguration, parseConfiguration } from './configuration-model';
describe('configuration merge safety', () => {
  it('detects removed nested keys but allows atomic array replacement', () => {
    expect(deletedPaths({ camera: { source: 'a', dimensions: [1, 2] } }, { camera: { dimensions: [3] } })).toEqual(['camera.source']);
  });
  it('strips authority-owned revision and schema-readonly values without changing the draft', () => {
    const draft = { metadata: { configuration_revision: 7, label: 'table' }, camera: { source: 'a', locked: 2 } };
    const schema = { properties: { camera: { properties: { locked: { readOnly: true } } } } };
    expect(editableConfiguration(draft, schema)).toEqual({ metadata: { label: 'table' }, camera: { source: 'a' } });
    expect(draft.metadata.configuration_revision).toBe(7);
  });
  it('requires an object rather than JSON scalars, arrays or empty data', () => {
    for (const value of ['null', '[]', '42', '{}']) expect(() => parseConfiguration(value)).toThrow();
    expect(parseConfiguration('{"camera": {}}')).toEqual({ camera: {} });
  });
  it('returns changed JSON Pointer paths without including configuration values', () => {
    const before = {
      metadata: { configuration_revision: 7 },
      camera: { viewer_url: 'http://user:secret@camera/live', mode: 'mjpeg' },
      'custom/key': { value: 1 },
    };
    const after = {
      metadata: { configuration_revision: 8 },
      camera: { viewer_url: 'http://user:secret@camera/live', mode: 'identified' },
      'custom/key': { value: 2 },
    };
    const paths = changedConfigurationPaths(before, after);
    expect(paths).toEqual(['/camera/mode', '/custom~1key/value']);
    expect(changedConfigurationPaths({ retained: 1, untouched: 2 }, { retained: 1 })).toEqual([]);
    expect(JSON.stringify(paths)).not.toContain('secret');
    expect(JSON.stringify(paths)).not.toContain('identified');
  });
});
