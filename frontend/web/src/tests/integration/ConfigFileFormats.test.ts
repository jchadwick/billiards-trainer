import {
  parseConfigurationFile,
  serializeConfigurationFile,
} from '../../utils/config-file-formats';

describe('configuration file formats', () => {
  const configuration = {
    camera: { resolution: { width: 1920, height: 1080 } },
    detection: { enabledClasses: ['cue ball', 'eight ball'] },
    game: { foulRules: { scratch: true } },
    ui: { theme: 'dark' },
  };

  test.each(['json', 'yaml', 'yml'])('parses nested .%s configuration files', (extension) => {
    const format = extension === 'json' ? 'json' : 'yaml';
    const serialized = serializeConfigurationFile(configuration, format);
    expect(parseConfigurationFile(`config.${extension}`, serialized.content)).toEqual(configuration);
  });

  test('emits the selected file format with a matching extension and MIME type', () => {
    expect(serializeConfigurationFile(configuration, 'yaml')).toMatchObject({
      extension: 'yaml',
      mimeType: 'application/yaml',
      content: expect.stringContaining('enabledClasses:'),
    });
  });

  test('rejects unsupported TOML rather than pretending to parse it', () => {
    expect(() => parseConfigurationFile('config.toml', '[camera]')).toThrow(
      'Unsupported configuration file format: .toml',
    );
  });
});
