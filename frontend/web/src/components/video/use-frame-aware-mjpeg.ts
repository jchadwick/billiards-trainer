import { useEffect, useRef, useState } from 'react';
import {
  readIdentifiedMjpegFrames,
  type IdentifiedMjpegFrame,
} from './mjpeg-frame-reader';

export interface FramePairingIdentity {
  frameId: string;
  engineInstanceId: string;
  sourceSessionId: string;
  configurationRevision: number;
  calibrationRevision: number;
  rectificationRevision: number;
}

export interface FrameAwareStreamPolicy {
  maximumFrameBytes: number;
  maximumPartHeaderBytes: number;
  bufferedFrameCount: number;
}

export interface DisplayMjpegFrame extends FramePairingIdentity {
  imageUrl: string;
}

interface FrameAwareStreamState {
  latest: DisplayMjpegFrame | null;
  matched: DisplayMjpegFrame | null;
  error: string | null;
}

const EMPTY_STATE: FrameAwareStreamState = { latest: null, matched: null, error: null };

/** Match every acquisition and geometry revision. Nearest-frame matches are unsafe. */
export function frameIdentityMatches(
  frame: FramePairingIdentity,
  expected: FramePairingIdentity | null,
): boolean {
  return expected !== null
    && frame.frameId === expected.frameId
    && frame.engineInstanceId === expected.engineInstanceId
    && frame.sourceSessionId === expected.sourceSessionId
    && frame.configurationRevision === expected.configurationRevision
    && frame.calibrationRevision === expected.calibrationRevision
    && frame.rectificationRevision === expected.rectificationRevision;
}

/**
 * Read an MJPEG stream as identified stills so the canvas is paired with the
 * exact displayed pixels. A latest-only ring bounds browser memory; the most
 * recently matched still remains in that same ring until a newer exact pair
 * arrives or the expected identity changes.
 */
export function useFrameAwareMjpeg(
  url: string | undefined,
  policy: FrameAwareStreamPolicy,
  expected: FramePairingIdentity | null,
  restartNonce = 0,
  shutdownSignal?: AbortSignal,
): FrameAwareStreamState {
  const [state, setState] = useState<FrameAwareStreamState>(EMPTY_STATE);
  const [ring, setRing] = useState<DisplayMjpegFrame[]>([]);
  const ringRef = useRef<DisplayMjpegFrame[]>([]);
  const matchedRef = useRef<DisplayMjpegFrame | null>(null);
  const expectedRef = useRef<FramePairingIdentity | null>(expected);
  const ownedUrlsRef = useRef(new Set<string>());
  expectedRef.current = expected;

  const expectedKey = expected
    ? [
      expected.frameId,
      expected.engineInstanceId,
      expected.sourceSessionId,
      expected.configurationRevision,
      expected.calibrationRevision,
      expected.rectificationRevision,
    ].join('\u0000')
    : '';

  useEffect(() => {
    const selected = expectedRef.current
      ? ringRef.current.find((frame) => frameIdentityMatches(frame, expectedRef.current)) ?? null
      : null;
    matchedRef.current = selected;
    setState((previous) => ({ ...previous, matched: selected }));
  }, [expectedKey]);

  useEffect(() => {
    ringRef.current = [];
    matchedRef.current = null;
    setRing([]);
    setState(EMPTY_STATE);
    if (!url) return undefined;

    const controller = new AbortController();
    const abortForShutdown = () => controller.abort();
    if (shutdownSignal?.aborted) controller.abort();
    else shutdownSignal?.addEventListener('abort', abortForShutdown, { once: true });
    let disposed = false;
    const revoke = (imageUrl: string) => {
      if (!ownedUrlsRef.current.delete(imageUrl)) return;
      URL.revokeObjectURL(imageUrl);
    };

    const consume = async () => {
      try {
        const response = await fetch(url, {
          cache: 'no-store',
          signal: controller.signal,
        });
        if (!response.ok || !response.body) {
          throw new Error(`Frame stream returned HTTP ${response.status}`);
        }
        const frames: AsyncGenerator<IdentifiedMjpegFrame> = readIdentifiedMjpegFrames(
          response.body,
          response.headers.get('content-type') ?? '',
          {
            maximumFrameBytes: policy.maximumFrameBytes,
            maximumPartHeaderBytes: policy.maximumPartHeaderBytes,
          },
          controller.signal,
        );
        for await (const incoming of frames) {
          if (disposed) return;
          const jpegBuffer = new Uint8Array(incoming.jpeg).buffer as ArrayBuffer;
          const imageUrl = URL.createObjectURL(new Blob([jpegBuffer], { type: 'image/jpeg' }));
          ownedUrlsRef.current.add(imageUrl);
          const displayed: DisplayMjpegFrame = {
            frameId: incoming.frameId,
            engineInstanceId: incoming.engineInstanceId,
            sourceSessionId: incoming.sourceSessionId,
            configurationRevision: incoming.configurationRevision,
            calibrationRevision: incoming.calibrationRevision,
            rectificationRevision: incoming.rectificationRevision,
            imageUrl,
          };
          const expectedNow = expectedRef.current;
          const priorMatched = matchedRef.current;
          const matched = frameIdentityMatches(displayed, expectedNow)
            ? displayed
            : priorMatched && frameIdentityMatches(priorMatched, expectedNow)
              ? priorMatched
              : null;
          const maximumCount = Math.min(64, Math.max(2, Math.floor(policy.bufferedFrameCount)));
          const candidates = [
            ...ringRef.current.filter((frame) => frame.frameId !== displayed.frameId),
            displayed,
          ];
          if (matched && !candidates.some((frame) => frame.imageUrl === matched.imageUrl)) {
            candidates.unshift(matched);
          }
          while (candidates.length > maximumCount) {
            const removableIndex = candidates.findIndex((frame) => frame !== matched);
            if (removableIndex < 0) break;
            candidates.splice(removableIndex, 1);
          }
          const retainedUrls = new Set(candidates.map((frame) => frame.imageUrl));
          for (const previous of ringRef.current) {
            if (!retainedUrls.has(previous.imageUrl)) revoke(previous.imageUrl);
          }
          if (priorMatched && !retainedUrls.has(priorMatched.imageUrl)) revoke(priorMatched.imageUrl);

          ringRef.current = candidates;
          matchedRef.current = matched;
          setRing(candidates);
          setState({ latest: displayed, matched, error: null });
        }
        if (!disposed) setState((previous) => ({
          ...previous,
          error: 'Frame stream ended before another identified frame arrived.',
        }));
      } catch (error) {
        if (disposed || controller.signal.aborted) return;
        const message = error instanceof Error ? error.message : 'Frame stream could not be read.';
        setState((previous) => ({ ...previous, error: message }));
      }
    };

    void consume();
    return () => {
      disposed = true;
      shutdownSignal?.removeEventListener('abort', abortForShutdown);
      controller.abort();
      for (const imageUrl of ownedUrlsRef.current) URL.revokeObjectURL(imageUrl);
      ownedUrlsRef.current.clear();
      ringRef.current = [];
      matchedRef.current = null;
    };
  }, [
    url,
    restartNonce,
    policy.maximumFrameBytes,
    policy.maximumPartHeaderBytes,
    policy.bufferedFrameCount,
    shutdownSignal,
  ]);

  // `ring` intentionally participates so a state event that arrives after its
  // frame is decoded can select the saved exact still without waiting for a
  // duplicate/retransmitted camera part.
  useEffect(() => {
    const selected = expectedRef.current
      ? ring.find((frame) => frameIdentityMatches(frame, expectedRef.current)) ?? null
      : null;
    const prior = matchedRef.current;
    if (prior === selected) return;
    matchedRef.current = selected;
    if (prior && !ring.some((frame) => frame.imageUrl === prior.imageUrl)) {
      if (ownedUrlsRef.current.delete(prior.imageUrl)) URL.revokeObjectURL(prior.imageUrl);
    }
    setState((previous) => ({ ...previous, matched: selected }));
  }, [expectedKey, ring]);

  return state;
}
