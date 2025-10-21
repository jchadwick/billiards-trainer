/**
 * Main video streaming component for displaying MJPEG video feed
 */

import React, { useRef, useEffect, useState, useCallback } from 'react';
import { observer } from 'mobx-react-lite';
import type { VideoStore } from '../../stores/VideoStore';
import type { Size2D, VideoError, ViewportTransform } from '../../types/video';
import { calculateBestFit, constrainTransform } from '../../utils/coordinates';

interface VideoStreamProps {
  videoStore: VideoStore;
  className?: string;
  width?: number;
  height?: number;
  autoPlay?: boolean;
  controls?: boolean;
  onError?: (error: VideoError) => void;
  onLoad?: () => void;
  onResize?: (size: Size2D) => void;
}

export const VideoStream = observer(React.forwardRef<any, VideoStreamProps>(({
  videoStore,
  className = '',
  width,
  height,
  autoPlay = true,
  controls = false,
  onError,
  onLoad,
  onResize,
}, ref) => {
  const imgRef = useRef<HTMLImageElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [containerSize, setContainerSize] = useState<Size2D>({ width: 0, height: 0 });
  const [videoSize, setVideoSize] = useState<Size2D>({ width: 0, height: 0 });
  const [isLoaded, setIsLoaded] = useState(false);
  const [currentSrc, setCurrentSrc] = useState<string>('');

  // Viewport transform for zoom/pan
  const [transform, setTransform] = useState<ViewportTransform>({
    x: 0,
    y: 0,
    scale: 1,
    rotation: 0,
  });

  // Update video source when store changes
  useEffect(() => {
    if (videoStore.isConnected && autoPlay) {
      const newSrc = videoStore.getStreamUrl();
      if (newSrc && newSrc !== currentSrc) {
        console.log('Updating video stream source:', newSrc);
        setCurrentSrc(newSrc);
        setIsLoaded(false);
      }
    } else {
      if (currentSrc) {
        console.log('Clearing video stream source');
        setCurrentSrc('');
        setIsLoaded(false);
      }
    }
  }, [videoStore.isConnected, videoStore.config.quality, videoStore.config.fps, autoPlay, currentSrc]);

  // Handle container resize
  const updateContainerSize = useCallback(() => {
    if (containerRef.current) {
      const rect = containerRef.current.getBoundingClientRect();
      const newSize = {
        width: width || rect.width,
        height: height || rect.height,
      };

      setContainerSize(newSize);
      onResize?.(newSize);
    }
  }, [width, height, onResize]);

  useEffect(() => {
    updateContainerSize();

    const resizeObserver = new ResizeObserver(updateContainerSize);
    if (containerRef.current) {
      resizeObserver.observe(containerRef.current);
    }

    return () => {
      resizeObserver.disconnect();
    };
  }, [updateContainerSize]);

  // Handle image load
  const handleImageLoad = useCallback(() => {
    if (imgRef.current) {
      const img = imgRef.current;
      const newVideoSize = {
        width: img.naturalWidth,
        height: img.naturalHeight,
      };

      setVideoSize(newVideoSize);
      setIsLoaded(true);
      onLoad?.();

      // Update video store status
      videoStore.setStatus({ streaming: true });
    }
  }, [onLoad, videoStore]);

  // Handle image error
  const handleImageError = useCallback((event: React.SyntheticEvent<HTMLImageElement>) => {
    console.error('Video stream image error:', event);

    const error: VideoError = {
      code: 'STREAM_ERROR',
      message: 'Failed to load video stream - check if camera is connected and backend is running',
      timestamp: Date.now(),
      recoverable: true,
    };

    videoStore.addError(error);
    videoStore.setStatus({ streaming: false });
    onError?.(error);
    setIsLoaded(false);

    // Attempt to reconnect if auto-reconnect is enabled
    if (videoStore.config.autoReconnect && videoStore.isConnected) {
      console.log('Attempting to reconnect video stream...');
      setTimeout(() => {
        const newSrc = videoStore.getStreamUrl();
        if (newSrc && newSrc !== currentSrc) {
          setCurrentSrc(newSrc);
        }
      }, videoStore.config.reconnectDelay);
    }
  }, [onError, videoStore, currentSrc]);

  // Calculate image styling for best fit
  const imageStyle = React.useMemo(() => {
    if (!isLoaded || !containerSize.width || !containerSize.height) {
      return {
        display: 'none',
      };
    }

    const fit = calculateBestFit(videoSize, containerSize, 'contain');

    return {
      width: fit.size.width,
      height: fit.size.height,
      transform: `translate(${fit.offset.x + transform.x}px, ${fit.offset.y + transform.y}px) scale(${transform.scale}) rotate(${transform.rotation}rad)`,
      transformOrigin: 'center center',
      transition: 'transform 0.1s ease-out',
    };
  }, [isLoaded, containerSize, videoSize, transform]);

  // Mouse interaction handlers for zoom/pan
  const [isDragging, setIsDragging] = useState(false);
  const [lastMousePos, setLastMousePos] = useState<{ x: number; y: number } | null>(null);

  const handleMouseDown = useCallback((event: React.MouseEvent) => {
    if (event.button === 0) { // Left mouse button
      setIsDragging(true);
      setLastMousePos({ x: event.clientX, y: event.clientY });
      event.preventDefault();
    }
  }, []);

  const handleMouseMove = useCallback((event: React.MouseEvent) => {
    if (isDragging && lastMousePos) {
      const deltaX = event.clientX - lastMousePos.x;
      const deltaY = event.clientY - lastMousePos.y;

      setTransform(prev => constrainTransform({
        ...prev,
        x: prev.x + deltaX,
        y: prev.y + deltaY,
      }, containerSize));

      setLastMousePos({ x: event.clientX, y: event.clientY });
    }
  }, [isDragging, lastMousePos, containerSize]);

  const handleMouseUp = useCallback(() => {
    setIsDragging(false);
    setLastMousePos(null);
  }, []);

  const handleWheel = useCallback((event: React.WheelEvent) => {
    event.preventDefault();

    const scaleFactor = event.deltaY > 0 ? 0.9 : 1.1;
    const newScale = transform.scale * scaleFactor;

    setTransform(prev => constrainTransform({
      ...prev,
      scale: newScale,
    }, containerSize, 0.1, 5));
  }, [transform.scale, containerSize]);

  // Reset transform
  const resetTransform = useCallback(() => {
    setTransform({
      x: 0,
      y: 0,
      scale: 1,
      rotation: 0,
    });
  }, []);

  // Double-click to reset
  const handleDoubleClick = useCallback(() => {
    resetTransform();
  }, [resetTransform]);

  // Provide transform control methods to parent
  React.useImperativeHandle(ref, () => ({
    resetTransform,
    setTransform,
    getTransform: () => transform,
    takeScreenshot: async (): Promise<Blob> => {
      if (!imgRef.current) {
        throw new Error('Video not loaded');
      }

      const canvas = document.createElement('canvas');
      const ctx = canvas.getContext('2d');

      if (!ctx) {
        throw new Error('Cannot create canvas context');
      }

      canvas.width = videoSize.width;
      canvas.height = videoSize.height;

      return new Promise((resolve, reject) => {
        const img = new Image();
        img.crossOrigin = 'anonymous';
        img.onload = () => {
          ctx.drawImage(img, 0, 0);
          canvas.toBlob((blob) => {
            if (blob) {
              resolve(blob);
            } else {
              reject(new Error('Failed to create screenshot'));
            }
          }, 'image/jpeg', 0.9);
        };
        img.onerror = () => reject(new Error('Failed to load image for screenshot'));
        img.src = imgRef.current!.src;
      });
    },
  }), [resetTransform, setTransform, transform, videoSize]);

  return (
    <div
      ref={containerRef}
      className={`relative overflow-hidden bg-black ${className}`}
      style={{
        width: width || '100%',
        height: height || '100%',
        cursor: isDragging ? 'grabbing' : 'grab',
      }}
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      onMouseLeave={handleMouseUp}
      onWheel={handleWheel}
      onDoubleClick={handleDoubleClick}
    >
      {/* Loading indicator */}
      {videoStore.isLoading && (
        <div className="absolute inset-0 flex items-center justify-center bg-gray-900 bg-opacity-75 z-10">
          <div className="text-white text-lg">Loading video stream...</div>
        </div>
      )}

      {/* Error display */}
      {videoStore.hasErrors && !isLoaded && (
        <div className="absolute inset-0 flex items-center justify-center bg-red-900 bg-opacity-75 z-10">
          <div className="text-white text-center p-4">
            <div className="text-lg font-semibold mb-2">Stream Error</div>
            <div className="text-sm">{videoStore.latestError?.message}</div>
            {videoStore.config.autoReconnect && (
              <div className="text-xs mt-2">Attempting to reconnect...</div>
            )}
          </div>
        </div>
      )}

      {/* Video stream */}
      {currentSrc && (
        <img
          ref={imgRef}
          src={currentSrc}
          alt="Video stream"
          style={imageStyle}
          onLoad={handleImageLoad}
          onError={handleImageError}
          draggable={false}
          className="absolute select-none"
        />
      )}

      {/* Connection status indicator */}
      {!videoStore.isConnected && (
        <div className="absolute top-2 left-2 px-3 py-1 bg-red-600 text-white text-sm rounded z-20">
          Disconnected
        </div>
      )}

      {/* Stream info overlay */}
      {controls && isLoaded && (
        <div className="absolute bottom-2 left-2 px-3 py-1 bg-black bg-opacity-75 text-white text-sm rounded z-20">
          {Math.round(videoStore.status.fps)} FPS | {videoStore.status.quality} | {videoSize.width}×{videoSize.height}
        </div>
      )}

      {/* Zoom info */}
      {transform.scale !== 1 && (
        <div className="absolute top-2 right-2 px-3 py-1 bg-black bg-opacity-75 text-white text-sm rounded z-20">
          {Math.round(transform.scale * 100)}%
        </div>
      )}
    </div>
  );
}));

VideoStream.displayName = 'VideoStream';
