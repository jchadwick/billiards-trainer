/** Frame-identity-preserving reader for bounded multipart MJPEG streams. */

export interface MjpegFrameLimits {
  maximumFrameBytes: number;
  maximumPartHeaderBytes: number;
}

export interface IdentifiedMjpegFrame {
  frameId: string;
  engineInstanceId: string;
  sourceSessionId: string;
  configurationRevision: number;
  calibrationRevision: number;
  rectificationRevision: number;
  jpeg: Uint8Array;
}

const CRLF = new Uint8Array([13, 10]);
const HEADER_END = new Uint8Array([13, 10, 13, 10]);

/**
 * Read each multipart part without buffering the whole stream. Content-Length
 * bounds the JPEG allocation and X-Frame-Id is mandatory so consumers can
 * refuse to paint state from another camera frame.
 */
export async function* readIdentifiedMjpegFrames(
  body: ReadableStream<Uint8Array>,
  contentType: string,
  limits: MjpegFrameLimits,
  signal?: AbortSignal,
): AsyncGenerator<IdentifiedMjpegFrame> {
  if (
    !Number.isSafeInteger(limits.maximumFrameBytes)
    || limits.maximumFrameBytes < 1
    || !Number.isSafeInteger(limits.maximumPartHeaderBytes)
    || limits.maximumPartHeaderBytes < 128
  ) throw new Error('Invalid multipart frame limits');

  const boundary = multipartBoundary(contentType);
  const reader = new BoundedMultipartReader(
    body.getReader(),
    limits.maximumFrameBytes + limits.maximumPartHeaderBytes + boundary.length + 8,
    signal,
  );
  const marker = new TextEncoder().encode(`--${boundary}`);
  try {
    while (true) {
      // Discard only the bounded CRLF or preamble before the next MIME marker.
      await reader.readUntil(marker, limits.maximumPartHeaderBytes);
      const partStart = await reader.readExact(2);
      if (partStart[0] === 45 && partStart[1] === 45) return;
      if (partStart[0] !== 13 || partStart[1] !== 10) {
        throw new Error('MJPEG multipart boundary is malformed');
      }
      const encodedHeaders = await reader.readUntil(HEADER_END, limits.maximumPartHeaderBytes);
      const headers = decodeHeaders(encodedHeaders, limits.maximumPartHeaderBytes);
      const frameId = headers.get('x-frame-id');
      const engineInstanceId = headers.get('x-engine-instance-id');
      const sourceSessionId = headers.get('x-source-session-id');
      const configurationRevision = parseRevision(headers.get('x-configuration-revision'));
      const calibrationRevision = parseRevision(headers.get('x-calibration-revision'));
      const rectificationRevision = parseRevision(headers.get('x-rectification-revision'));
      const lengthText = headers.get('content-length');
      if (
        !frameId || !isValidFrameId(frameId)
        || !engineInstanceId || !isValidFrameId(engineInstanceId)
        || !sourceSessionId || !isValidFrameId(sourceSessionId)
        || configurationRevision === null
        || calibrationRevision === null
        || rectificationRevision === null
      ) {
        throw new Error('MJPEG part has incomplete or invalid frame identity');
      }
      if (!lengthText || !/^\d+$/.test(lengthText)) {
        throw new Error('MJPEG part has no valid Content-Length');
      }
      const contentLength = Number(lengthText);
      if (
        !Number.isSafeInteger(contentLength)
        || contentLength < 4
        || contentLength > limits.maximumFrameBytes
      ) throw new Error('MJPEG frame exceeds its configured byte limit');

      const jpeg = await reader.readExact(contentLength);
      if (jpeg[0] !== 0xff || jpeg[1] !== 0xd8 || jpeg.at(-2) !== 0xff || jpeg.at(-1) !== 0xd9) {
        throw new Error('MJPEG part does not contain a complete JPEG frame');
      }
      yield {
        frameId,
        engineInstanceId,
        sourceSessionId,
        configurationRevision,
        calibrationRevision,
        rectificationRevision,
        jpeg,
      };
    }
  } finally {
    await reader.cancel();
  }
}

function multipartBoundary(contentType: string): string {
  const match = /(?:^|;)\s*boundary=(?:"([^"]+)"|([^;\s]+))/i.exec(contentType);
  const boundary = match?.[1] ?? match?.[2];
  if (!boundary || boundary.length > 70 || /[\r\n]/.test(boundary)) {
    throw new Error('Viewer response is not a bounded multipart MJPEG stream');
  }
  return boundary;
}

function decodeHeaders(bytes: Uint8Array, maximumBytes: number): Map<string, string> {
  if (bytes.length > maximumBytes) throw new Error('MJPEG part headers exceed their limit');
  const text = new TextDecoder('latin1', { fatal: true }).decode(bytes).replace(/^\r\n/, '');
  const headers = new Map<string, string>();
  for (const line of text.split('\r\n')) {
    const separator = line.indexOf(':');
    if (separator <= 0) throw new Error('MJPEG part contains a malformed header');
    const name = line.slice(0, separator).trim().toLowerCase();
    const value = line.slice(separator + 1).trim();
    if (!name || headers.has(name) || /[\r\n]/.test(value)) {
      throw new Error('MJPEG part contains an invalid or duplicate header');
    }
    headers.set(name, value);
  }
  return headers;
}

function isValidFrameId(frameId: string): boolean {
  return frameId.length > 0
    && frameId.length <= 128
    && !/[\u0000-\u001f\u007f]/.test(frameId);
}

function parseRevision(value: string | undefined): number | null {
  if (value === undefined || !/^\d+$/.test(value)) return null;
  const revision = Number(value);
  return Number.isSafeInteger(revision) && revision >= 0 ? revision : null;
}

class BoundedMultipartReader {
  private buffer = new Uint8Array(0);
  private start = 0;
  private end = 0;
  private pendingChunk: Uint8Array | null = null;
  private pendingOffset = 0;

  constructor(
    private readonly reader: ReadableStreamDefaultReader<Uint8Array>,
    private readonly maximumBufferedBytes: number,
    private readonly signal?: AbortSignal,
  ) {}

  async readUntil(delimiter: Uint8Array, maximumPrefixBytes: number): Promise<Uint8Array> {
    while (true) {
      const match = findBytes(this.buffer, this.start, this.end, delimiter);
      if (match >= 0) {
        const prefixLength = match - this.start;
        if (prefixLength > maximumPrefixBytes) throw new Error('MJPEG multipart section is too large');
        const prefix = this.buffer.slice(this.start, match);
        this.start = match + delimiter.length;
        this.compact();
        return prefix;
      }
      if (this.end - this.start > maximumPrefixBytes + delimiter.length - 1) {
        throw new Error('MJPEG multipart section is too large');
      }
      await this.appendNextChunk();
    }
  }

  async readExact(length: number): Promise<Uint8Array> {
    const result = new Uint8Array(length);
    let written = 0;
    while (written < length) {
      if (this.start === this.end) await this.appendNextChunk();
      const count = Math.min(length - written, this.end - this.start);
      result.set(this.buffer.subarray(this.start, this.start + count), written);
      this.start += count;
      written += count;
      this.compact();
    }
    return result;
  }

  async cancel(): Promise<void> {
    try {
      if (this.signal?.aborted) await this.reader.cancel(this.signal.reason);
      else await this.reader.cancel();
    } catch {
      // Abort and network shutdown races are already represented by the
      // caller's lifecycle; cleanup must not mask the parsing result.
    }
  }

  private async appendNextChunk(): Promise<void> {
    if (this.signal?.aborted) throw new DOMException('Stream aborted', 'AbortError');
    if (!this.pendingChunk || this.pendingOffset >= this.pendingChunk.length) {
      const next = await this.reader.read();
      if (next.done) {
        throw new Error('MJPEG stream ended between frames');
      }
      this.pendingChunk = next.value;
      this.pendingOffset = 0;
    }
    const available = this.end - this.start;
    const capacity = this.maximumBufferedBytes - available;
    if (capacity <= 0) throw new Error('MJPEG receive buffer reached its configured limit');
    const count = Math.min(capacity, this.pendingChunk.length - this.pendingOffset);
    this.makeRoom(available + count);
    this.buffer.set(
      this.pendingChunk.subarray(this.pendingOffset, this.pendingOffset + count),
      this.end,
    );
    this.pendingOffset += count;
    this.end += count;
  }

  private makeRoom(required: number): void {
    const available = this.end - this.start;
    if (this.buffer.length - this.end >= required - available) return;
    if (this.start > 0) {
      this.buffer.copyWithin(0, this.start, this.end);
      this.end = available;
      this.start = 0;
      if (this.buffer.length >= required) return;
    }
    const nextCapacity = Math.min(
      this.maximumBufferedBytes,
      Math.max(required, this.buffer.length === 0 ? 1024 : this.buffer.length * 2),
    );
    if (nextCapacity < required) throw new Error('MJPEG receive buffer reached its configured limit');
    const replacement = new Uint8Array(nextCapacity);
    replacement.set(this.buffer.subarray(this.start, this.end));
    this.end = available;
    this.start = 0;
    this.buffer = replacement;
  }

  private compact(): void {
    if (this.start === this.end) {
      this.start = 0;
      this.end = 0;
    } else if (this.start > this.buffer.length / 2) {
      this.buffer.copyWithin(0, this.start, this.end);
      this.end -= this.start;
      this.start = 0;
    }
  }
}

function findBytes(haystack: Uint8Array, start: number, end: number, needle: Uint8Array): number {
  outer: for (let index = start; index <= end - needle.length; index += 1) {
    for (let offset = 0; offset < needle.length; offset += 1) {
      if (haystack[index + offset] !== needle[offset]) continue outer;
    }
    return index;
  }
  return -1;
}
