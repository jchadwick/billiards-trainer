import { describe, expect, it } from 'vitest';
import { resolveApiBaseUrl } from '../../api/axios-client';

describe('API base URL selection', () => {
  it('uses the configured backend when frontend and backend are split', () => {
    expect(
      resolveApiBaseUrl('http://localhost:8000', 'http://localhost:3000')
    ).toBe('http://localhost:8000');
  });

  it('uses the serving origin for a portable Rust-hosted frontend bundle', () => {
    expect(resolveApiBaseUrl(undefined, 'http://192.168.1.31:8080')).toBe(
      'http://192.168.1.31:8080'
    );
  });

  it('ignores a whitespace-only override and uses the serving origin', () => {
    expect(resolveApiBaseUrl('   ', 'http://127.0.0.1:18080')).toBe(
      'http://127.0.0.1:18080'
    );
  });
});
