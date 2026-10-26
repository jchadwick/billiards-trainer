import { describe, expect, it } from 'vitest';
import { readIdentifiedMjpegFrames } from '../../components/video/mjpeg-frame-reader';
import { frameIdentityMatches } from '../../components/video/use-frame-aware-mjpeg';

const limits = { maximumFrameBytes: 32, maximumPartHeaderBytes: 512 };

function streamInChunks(bytes: Uint8Array, chunkSize: number): ReadableStream<Uint8Array> {
  return new ReadableStream({
    start(controller) {
      for (let offset = 0; offset < bytes.length; offset += chunkSize) {
        controller.enqueue(bytes.slice(offset, offset + chunkSize));
      }
      controller.close();
    },
  });
}

function joinBytes(chunks: Uint8Array[]): Uint8Array {
  const result = new Uint8Array(chunks.reduce((total, chunk) => total + chunk.length, 0));
  let offset = 0;
  for (const chunk of chunks) {
    result.set(chunk, offset);
    offset += chunk.length;
  }
  return result;
}

describe('identified multipart MJPEG reader', () => {
  it('matches only the same frame, source session, engine, and geometry revisions', () => {
    const expected = {
      frameId: 'frame-42',
      engineInstanceId: 'engine-a',
      sourceSessionId: 'session-a',
      configurationRevision: 4,
      calibrationRevision: 2,
      rectificationRevision: 3,
    };
    expect(frameIdentityMatches(expected, expected)).toBe(true);
    expect(frameIdentityMatches({ ...expected, frameId: 'frame-43' }, expected)).toBe(false);
    expect(frameIdentityMatches({ ...expected, sourceSessionId: 'session-b' }, expected)).toBe(false);
    expect(frameIdentityMatches({ ...expected, calibrationRevision: 3 }, expected)).toBe(false);
    expect(frameIdentityMatches(expected, null)).toBe(false);
  });

  it('preserves exact frame IDs and JPEG bytes across arbitrary chunk boundaries', async () => {
    // Multipart data is binary; construct the JPEG bytes explicitly rather
    // than relying on UTF-8 encodings for the non-ASCII markers.
    const headers = (frameId: string) => new TextEncoder().encode(
      `--table\r\nContent-Type: image/jpeg\r\nContent-Length: 5\r\nX-Engine-Instance-Id: engine-a\r\nX-Source-Session-Id: session-a\r\nX-Configuration-Revision: 4\r\nX-Calibration-Revision: 2\r\nX-Rectification-Revision: 3\r\nX-Frame-Id: ${frameId}\r\n\r\n`,
    );
    const headerAndFirst = headers('cam-1:42');
    const secondHeader = new TextEncoder().encode(
      '\r\n',
    );
    const header2 = headers('cam-1:43');
    const suffix = new TextEncoder().encode('\r\n--table--\r\n');
    const binary = joinBytes([
      headerAndFirst,
      new Uint8Array([0xff, 0xd8, 65, 0xff, 0xd9]),
      secondHeader,
      header2,
      new Uint8Array([0xff, 0xd8, 66, 0xff, 0xd9]),
      suffix,
    ]);

    const frames: Array<Record<string, unknown>> = [];
    for await (const frame of readIdentifiedMjpegFrames(
      streamInChunks(binary, 3),
      'multipart/x-mixed-replace; boundary=table',
      limits,
    )) {
      frames.push({ ...frame, jpeg: [...frame.jpeg] });
    }

    expect(frames).toEqual([
      {
        frameId: 'cam-1:42', engineInstanceId: 'engine-a', sourceSessionId: 'session-a',
        configurationRevision: 4, calibrationRevision: 2, rectificationRevision: 3,
        jpeg: [0xff, 0xd8, 65, 0xff, 0xd9],
      },
      {
        frameId: 'cam-1:43', engineInstanceId: 'engine-a', sourceSessionId: 'session-a',
        configurationRevision: 4, calibrationRevision: 2, rectificationRevision: 3,
        jpeg: [0xff, 0xd8, 66, 0xff, 0xd9],
      },
    ]);
  });

  it('fails closed when the stream omits frame identity or exceeds configured limits', async () => {
    const missingIdentity = joinBytes([
      new TextEncoder().encode('--table\r\nContent-Length: 5\r\nX-Frame-Id: cam-1:1\r\n\r\n'),
      new Uint8Array([0xff, 0xd8, 65, 0xff, 0xd9]),
      new TextEncoder().encode('\r\n--table--\r\n'),
    ]);
    const consume = async (bytes: Uint8Array, maximumFrameBytes: number) => {
      for await (const _frame of readIdentifiedMjpegFrames(
        streamInChunks(bytes, 16),
        'multipart/x-mixed-replace; boundary=table',
        { ...limits, maximumFrameBytes },
      )) { /* exhaust the stream to surface protocol errors */ }
    };

    await expect(consume(missingIdentity, 32)).rejects.toThrow('frame identity');

    const oversized = joinBytes([
      new TextEncoder().encode('--table\r\nContent-Length: 5\r\nX-Engine-Instance-Id: engine-a\r\nX-Source-Session-Id: session-a\r\nX-Configuration-Revision: 4\r\nX-Calibration-Revision: 2\r\nX-Rectification-Revision: 3\r\nX-Frame-Id: cam-1:1\r\n\r\n'),
      new Uint8Array([0xff, 0xd8, 65, 0xff, 0xd9]),
      new TextEncoder().encode('\r\n--table--\r\n'),
    ]);
    await expect(consume(oversized, 4)).rejects.toThrow('configured byte limit');
  });
});
