import { parse, stringify } from 'yaml';

export type ConfigurationFileFormat = 'json' | 'yaml';

export interface SerializedConfigurationFile {
  extension: ConfigurationFileFormat;
  mimeType: string;
  content: string;
}

/** Parse only the file formats that the configuration screen fully supports. */
export function parseConfigurationFile(filename: string, content: string): unknown {
  const extension = filename.split('.').pop()?.toLowerCase();
  if (extension === 'json') return JSON.parse(content);
  if (extension === 'yaml' || extension === 'yml') return parse(content);

  throw new Error(`Unsupported configuration file format: .${extension ?? ''}`);
}

/** Serialize configurations with standards-compliant JSON or YAML output. */
export function serializeConfigurationFile(
  config: unknown,
  format: ConfigurationFileFormat,
): SerializedConfigurationFile {
  if (format === 'json') {
    return {
      extension: 'json',
      mimeType: 'application/json',
      content: JSON.stringify(config, null, 2),
    };
  }

  return {
    extension: 'yaml',
    mimeType: 'application/yaml',
    content: stringify(config),
  };
}
