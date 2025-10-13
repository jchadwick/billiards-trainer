/**
 * Main live view component combining video stream with overlays
 */

import React, { useState, useRef, useCallback, useEffect } from 'react';
import { observer } from 'mobx-react-lite';
import { VideoStore } from '../../stores/VideoStore';
import { VideoStream } from './VideoStream';
import { StreamControls } from './StreamControls';
import { OverlayCanvas } from './OverlayCanvas';
import { SkipLinkTarget } from '../accessibility/SkipLinks';
import { VideoStatusRegion, GameStatusRegion } from '../accessibility/AnnouncementRegion';
import { ScreenReaderOnly, VisualDescription } from '../accessibility/ScreenReaderOnly';
import { FocusManager } from '../accessibility/FocusManager';
import { useAccessibility } from '../../hooks/useAccessibility';
import { createVideoControlAria, createVisualDescription } from '../../utils/accessibility';
import type { Size2D, ViewportTransform, OverlayConfig, Point2D, VideoError } from '../../types/video';

interface LiveViewProps {
  className?: string;
  autoConnect?: boolean;
}

// Default overlay configuration
const defaultOverlayConfig: OverlayConfig = {
  balls: {
    visible: true,
    showLabels: true,
    showIds: false,
    showVelocity: true,
    showConfidence: false,
    radius: 15,
    opacity: 0.9,
  },
  trajectories: {
    visible: true,
    showProbability: true,
    lineWidth: 3,
    opacity: 0.8,
    maxLength: 50,
  },
  table: {
    visible: true,
    showPockets: true,
    showRails: true,
    lineWidth: 2,
    opacity: 0.7,
  },
  cue: {
    visible: true,
    showAngle: true,
    showGuideLines: true,
    lineWidth: 4,
    opacity: 0.9,
  },
  grid: {
    visible: false,
    spacing: 50,
    opacity: 0.3,
  },
  calibration: {
    visible: true,
    showPoints: true,
    showLines: true,
    showLabels: true,
    showAccuracy: true,
    lineWidth: 2,
    opacity: 0.8,
  },
};

export const LiveView = observer<LiveViewProps>(({
  className = '',
  autoConnect = true,
}) => {
  const [videoStore] = useState(() => new VideoStore());
  const [containerSize, setContainerSize] = useState<Size2D>({ width: 0, height: 0 });
  const [videoSize, setVideoSize] = useState<Size2D>({ width: 0, height: 0 });
  const [overlayConfig, setOverlayConfig] = useState<OverlayConfig>(defaultOverlayConfig);
  const [isFullscreen, setIsFullscreen] = useState(false);

  const containerRef = useRef<HTMLDivElement>(null);
  const videoStreamRef = useRef<any>(null);
  const { announce, reducedMotion } = useAccessibility();

  // Viewport transform for zoom/pan
  const [transform, setTransform] = useState<ViewportTransform>({
    x: 0,
    y: 0,
    scale: 1,
    rotation: 0,
  });

  // Auto-connect on mount
  useEffect(() => {
    if (autoConnect) {
      videoStore.connect().catch(console.error);
    }

    return () => {
      videoStore.dispose();
    };
  }, [autoConnect, videoStore]);

  // Handle container resize
  const handleContainerResize = useCallback((size: Size2D) => {
    setContainerSize(size);
  }, []);

  // Handle video load
  const handleVideoLoad = useCallback(() => {
    console.log('Video stream loaded');
  }, []);

  // Handle video errors
  const handleVideoError = useCallback((error: VideoError) => {
    console.error('Video stream error:', error);
  }, []);

  // Handle fullscreen toggle
  const handleFullscreen = useCallback(() => {
    if (!containerRef.current) return;

    if (!isFullscreen) {
      if (containerRef.current.requestFullscreen) {
        containerRef.current.requestFullscreen();
        announce('Entered fullscreen mode. Press Escape to exit.');
      }
    } else {
      if (document.exitFullscreen) {
        document.exitFullscreen();
        announce('Exited fullscreen mode.');
      }
    }
  }, [isFullscreen, announce]);

  // Handle fullscreen change events
  useEffect(() => {
    const handleFullscreenChange = () => {
      setIsFullscreen(!!document.fullscreenElement);
    };

    document.addEventListener('fullscreenchange', handleFullscreenChange);
    return () => {
      document.removeEventListener('fullscreenchange', handleFullscreenChange);
    };
  }, []);

  // Handle screenshot
  const handleScreenshot = useCallback(async () => {
    try {
      if (videoStreamRef.current?.takeScreenshot) {
        const blob = await videoStreamRef.current.takeScreenshot();

        // Create download link
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = `billiards-screenshot-${new Date().toISOString()}.jpg`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);

        console.log('Screenshot saved');
      }
    } catch (error) {
      console.error('Screenshot failed:', error);
    }
  }, []);

  // Handle overlay canvas clicks
  const handleCanvasClick = useCallback((point: Point2D, event: React.MouseEvent) => {
    console.log('Canvas clicked at:', point);

    // You can add click handlers here, such as:
    // - Ball selection
    // - Trajectory point addition
    // - Table calibration points
  }, []);

  // Handle overlay canvas double clicks
  const handleCanvasDoubleClick = useCallback((point: Point2D, event: React.MouseEvent) => {
    console.log('Canvas double-clicked at:', point);

    // Reset transform on double-click
    setTransform({
      x: 0,
      y: 0,
      scale: 1,
      rotation: 0,
    });
  }, []);

  // Overlay configuration controls
  const updateOverlayConfig = useCallback((section: keyof OverlayConfig, updates: Partial<OverlayConfig[keyof OverlayConfig]>) => {
    setOverlayConfig(prev => ({
      ...prev,
      [section]: {
        ...prev[section],
        ...updates,
      },
    }));
  }, []);

  // Handle keyboard shortcuts
  const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
    if (e.target !== document.body && !(e.target as Element).closest('.video-container')) return;

    switch (e.key) {
      case 'f':
      case 'F':
        if (!e.ctrlKey && !e.altKey && !e.metaKey) {
          e.preventDefault();
          handleFullscreen();
        }
        break;
      case 's':
      case 'S':
        if (!e.ctrlKey && !e.altKey && !e.metaKey) {
          e.preventDefault();
          handleScreenshot();
        }
        break;
      case 'b':
      case 'B':
        if (!e.ctrlKey && !e.altKey && !e.metaKey) {
          e.preventDefault();
          updateOverlayConfig('balls', { visible: !overlayConfig.balls.visible });
          announce(`Ball overlay ${overlayConfig.balls.visible ? 'hidden' : 'shown'}`);
        }
        break;
      case 't':
      case 'T':
        if (!e.ctrlKey && !e.altKey && !e.metaKey) {
          e.preventDefault();
          updateOverlayConfig('trajectories', { visible: !overlayConfig.trajectories.visible });
          announce(`Trajectory overlay ${overlayConfig.trajectories.visible ? 'hidden' : 'shown'}`);
        }
        break;
      case 'Escape':
        if (isFullscreen) {
          handleFullscreen();
        }
        break;
    }
  }, [handleFullscreen, handleScreenshot, overlayConfig, updateOverlayConfig, announce, isFullscreen]);

  return (
    <div
      className={`flex flex-col h-full ${className}`}
      onKeyDown={handleKeyDown}
      tabIndex={-1}
    >
      {/* Accessibility Status Regions */}
      <VideoStatusRegion
        isPlaying={videoStore.isStreaming}
        isMuted={false}
      />
      <GameStatusRegion
        ballCount={videoStore.currentBalls?.length}
        detectedObjects={{
          balls: videoStore.currentBalls?.length || 0,
          trajectories: videoStore.currentTrajectories?.length || 0,
          table: !!videoStore.currentTable,
          cue: !!videoStore.currentCue?.detected
        }}
      />

      {/* Controls */}
      <SkipLinkTarget id="video-controls" className="flex-none p-4 bg-gray-100 border-b">
        <StreamControls
          videoStore={videoStore}
          showAdvanced={true}
          onFullscreen={handleFullscreen}
          onScreenshot={handleScreenshot}
        />
        <ScreenReaderOnly>
          Video controls. Use keyboard shortcuts: F for fullscreen, S for screenshot, B to toggle ball overlay, T to toggle trajectory overlay
        </ScreenReaderOnly>
      </SkipLinkTarget>

      {/* Main view area */}
      <div className="flex-1 flex">
        {/* Video and overlay container */}
        <SkipLinkTarget
          id="video-player"
          className={`relative flex-1 bg-black video-container ${isFullscreen ? 'fixed inset-0 z-50' : ''}`}
          ref={containerRef}
        >
          <FocusManager
            trapFocus={isFullscreen}
            onEscape={isFullscreen ? handleFullscreen : undefined}
          >
            {/* Screen reader description */}
            <ScreenReaderOnly>
              <div role="img" aria-label="Billiards video stream with overlay analysis">
                {videoStore.isConnected ? (
                  `Live video stream showing billiards table. ${videoStore.currentBalls?.length || 0} balls detected.
                   ${videoStore.currentTrajectories?.length || 0} trajectory predictions available.
                   ${videoStore.currentTable ? 'Table boundaries detected.' : 'Table boundaries not detected.'}
                   ${videoStore.currentCue?.detected ? 'Cue stick detected.' : 'Cue stick not detected.'}`
                ) : (
                  'Video stream not connected. Use controls to connect to video source.'
                )}
              </div>
            </ScreenReaderOnly>

            {/* Video stream */}
            <VideoStream
              ref={videoStreamRef}
              videoStore={videoStore}
              className="absolute inset-0"
              onError={handleVideoError}
              onLoad={handleVideoLoad}
              onResize={(size) => {
                handleContainerResize(size);
                setVideoSize(size);
              }}
            />

            {/* Overlay canvas */}
            {videoStore.isStreaming && containerSize.width > 0 && containerSize.height > 0 && (
              <OverlayCanvas
                videoStore={videoStore}
                videoSize={videoSize}
                canvasSize={containerSize}
                transform={transform}
                config={overlayConfig}
                className="absolute inset-0"
                onClick={handleCanvasClick}
                onDoubleClick={handleCanvasDoubleClick}
                role="img"
                aria-label="Interactive overlay showing ball positions, trajectories, and table analysis"
              />
            )}

            {/* Status overlay */}
            {!videoStore.isConnected && (
              <div className="absolute inset-0 flex items-center justify-center bg-gray-900 bg-opacity-75">
                <div className="text-white text-center" role="status" aria-live="polite">
                  <div className="text-xl mb-2">Not Connected</div>
                  <div className="text-sm">Use controls above to connect to video stream</div>
                </div>
              </div>
            )}
          </FocusManager>
        </SkipLinkTarget>

        {/* Side panel with overlay controls */}
        <div className={`flex-none w-80 bg-white border-l overflow-y-auto ${isFullscreen ? 'hidden' : ''}`}>
          <div className="p-4">
            <h3 className="text-lg font-semibold mb-4" id="overlay-settings-heading">
              Overlay Settings
            </h3>
            <ScreenReaderOnly>
              Control panel for configuring video overlay display options
            </ScreenReaderOnly>

            {/* Ball overlays */}
            <fieldset className="mb-6">
              <legend className="font-medium mb-2">Ball Display Options</legend>
              <div className="space-y-2" role="group" aria-labelledby="ball-options-heading">
                <h4 id="ball-options-heading" className="sr-only">Ball overlay controls</h4>

                <label className="flex items-center">
                  <input
                    type="checkbox"
                    checked={overlayConfig.balls.visible}
                    onChange={(e) => {
                      updateOverlayConfig('balls', { visible: e.target.checked });
                      announce(`Ball overlay ${e.target.checked ? 'enabled' : 'disabled'}`);
                    }}
                    className="mr-2 focus:ring-2 focus:ring-blue-500"
                    aria-describedby="show-balls-desc"
                  />
                  <span>Show balls</span>
                  <ScreenReaderOnly id="show-balls-desc">
                    Toggle visibility of ball position markers on the video
                  </ScreenReaderOnly>
                </label>

                <label className="flex items-center">
                  <input
                    type="checkbox"
                    checked={overlayConfig.balls.showLabels}
                    onChange={(e) => {
                      updateOverlayConfig('balls', { showLabels: e.target.checked });
                      announce(`Ball numbers ${e.target.checked ? 'shown' : 'hidden'}`);
                    }}
                    className="mr-2 focus:ring-2 focus:ring-blue-500"
                    aria-describedby="show-numbers-desc"
                  />
                  <span>Show numbers</span>
                  <ScreenReaderOnly id="show-numbers-desc">
                    Display ball numbers on each detected ball
                  </ScreenReaderOnly>
                </label>

                <label className="flex items-center">
                  <input
                    type="checkbox"
                    checked={overlayConfig.balls.showVelocity}
                    onChange={(e) => updateOverlayConfig('balls', { showVelocity: e.target.checked })}
                    className="mr-2"
                  />
                  Show velocity
                </label>

                <label className="flex items-center">
                  <input
                    type="checkbox"
                    checked={overlayConfig.balls.showConfidence}
                    onChange={(e) => updateOverlayConfig('balls', { showConfidence: e.target.checked })}
                    className="mr-2"
                  />
                  Show confidence
                </label>

                <div className="flex items-center space-x-2">
                  <label className="text-sm">Opacity:</label>
                  <input
                    type="range"
                    min="0"
                    max="1"
                    step="0.1"
                    value={overlayConfig.balls.opacity}
                    onChange={(e) => updateOverlayConfig('balls', { opacity: parseFloat(e.target.value) })}
                    className="flex-1"
                  />
                  <span className="text-sm w-10">{Math.round(overlayConfig.balls.opacity * 100)}%</span>
                </div>
              </div>
            </fieldset>

            {/* Trajectory overlays */}
            <div className="mb-6">
              <h4 className="font-medium mb-2">Trajectories</h4>
              <div className="space-y-2">
                <label className="flex items-center">
                  <input
                    type="checkbox"
                    checked={overlayConfig.trajectories.visible}
                    onChange={(e) => updateOverlayConfig('trajectories', { visible: e.target.checked })}
                    className="mr-2"
                  />
                  Show trajectories
                </label>

                <label className="flex items-center">
                  <input
                    type="checkbox"
                    checked={overlayConfig.trajectories.showProbability}
                    onChange={(e) => updateOverlayConfig('trajectories', { showProbability: e.target.checked })}
                    className="mr-2"
                  />
                  Show probability
                </label>

                <div className="flex items-center space-x-2">
                  <label className="text-sm">Line width:</label>
                  <input
                    type="range"
                    min="1"
                    max="10"
                    value={overlayConfig.trajectories.lineWidth}
                    onChange={(e) => updateOverlayConfig('trajectories', { lineWidth: parseInt(e.target.value) })}
                    className="flex-1"
                  />
                  <span className="text-sm w-6">{overlayConfig.trajectories.lineWidth}</span>
                </div>

                <div className="flex items-center space-x-2">
                  <label className="text-sm">Opacity:</label>
                  <input
                    type="range"
                    min="0"
                    max="1"
                    step="0.1"
                    value={overlayConfig.trajectories.opacity}
                    onChange={(e) => updateOverlayConfig('trajectories', { opacity: parseFloat(e.target.value) })}
                    className="flex-1"
                  />
                  <span className="text-sm w-10">{Math.round(overlayConfig.trajectories.opacity * 100)}%</span>
                </div>
              </div>
            </div>

            {/* Table overlays */}
            <div className="mb-6">
              <h4 className="font-medium mb-2">Table</h4>
              <div className="space-y-2">
                <label className="flex items-center">
                  <input
                    type="checkbox"
                    checked={overlayConfig.table.visible}
                    onChange={(e) => updateOverlayConfig('table', { visible: e.target.checked })}
                    className="mr-2"
                  />
                  Show table
                </label>

                <label className="flex items-center">
                  <input
                    type="checkbox"
                    checked={overlayConfig.table.showPockets}
                    onChange={(e) => updateOverlayConfig('table', { showPockets: e.target.checked })}
                    className="mr-2"
                  />
                  Show pockets
                </label>

                <label className="flex items-center">
                  <input
                    type="checkbox"
                    checked={overlayConfig.table.showRails}
                    onChange={(e) => updateOverlayConfig('table', { showRails: e.target.checked })}
                    className="mr-2"
                  />
                  Show rails
                </label>

                <div className="flex items-center space-x-2">
                  <label className="text-sm">Opacity:</label>
                  <input
                    type="range"
                    min="0"
                    max="1"
                    step="0.1"
                    value={overlayConfig.table.opacity}
                    onChange={(e) => updateOverlayConfig('table', { opacity: parseFloat(e.target.value) })}
                    className="flex-1"
                  />
                  <span className="text-sm w-10">{Math.round(overlayConfig.table.opacity * 100)}%</span>
                </div>
              </div>
            </div>

            {/* Cue overlays */}
            <div className="mb-6">
              <h4 className="font-medium mb-2">Cue Stick</h4>
              <div className="space-y-2">
                <label className="flex items-center">
                  <input
                    type="checkbox"
                    checked={overlayConfig.cue.visible}
                    onChange={(e) => updateOverlayConfig('cue', { visible: e.target.checked })}
                    className="mr-2"
                  />
                  Show cue
                </label>

                <label className="flex items-center">
                  <input
                    type="checkbox"
                    checked={overlayConfig.cue.showAngle}
                    onChange={(e) => updateOverlayConfig('cue', { showAngle: e.target.checked })}
                    className="mr-2"
                  />
                  Show angle
                </label>

                <label className="flex items-center">
                  <input
                    type="checkbox"
                    checked={overlayConfig.cue.showGuideLines}
                    onChange={(e) => updateOverlayConfig('cue', { showGuideLines: e.target.checked })}
                    className="mr-2"
                  />
                  Show guide lines
                </label>

                <div className="flex items-center space-x-2">
                  <label className="text-sm">Opacity:</label>
                  <input
                    type="range"
                    min="0"
                    max="1"
                    step="0.1"
                    value={overlayConfig.cue.opacity}
                    onChange={(e) => updateOverlayConfig('cue', { opacity: parseFloat(e.target.value) })}
                    className="flex-1"
                  />
                  <span className="text-sm w-10">{Math.round(overlayConfig.cue.opacity * 100)}%</span>
                </div>
              </div>
            </div>

            {/* Grid overlay */}
            <div className="mb-6">
              <h4 className="font-medium mb-2">Grid</h4>
              <div className="space-y-2">
                <label className="flex items-center">
                  <input
                    type="checkbox"
                    checked={overlayConfig.grid.visible}
                    onChange={(e) => updateOverlayConfig('grid', { visible: e.target.checked })}
                    className="mr-2"
                  />
                  Show grid
                </label>

                <div className="flex items-center space-x-2">
                  <label className="text-sm">Spacing:</label>
                  <input
                    type="range"
                    min="20"
                    max="100"
                    value={overlayConfig.grid.spacing}
                    onChange={(e) => updateOverlayConfig('grid', { spacing: parseInt(e.target.value) })}
                    className="flex-1"
                  />
                  <span className="text-sm w-8">{overlayConfig.grid.spacing}</span>
                </div>
              </div>
            </div>

            {/* Calibration overlay */}
            {videoStore.hasCalibration && (
              <div className="mb-6 border-t pt-4">
                <h4 className="font-medium mb-2 text-purple-600">Calibration</h4>
                <div className="space-y-2">
                  <label className="flex items-center">
                    <input
                      type="checkbox"
                      checked={overlayConfig.calibration.visible}
                      onChange={(e) => {
                        updateOverlayConfig('calibration', { visible: e.target.checked });
                        announce(`Calibration overlay ${e.target.checked ? 'enabled' : 'disabled'}`);
                      }}
                      className="mr-2 focus:ring-2 focus:ring-purple-500"
                      aria-describedby="show-calibration-desc"
                    />
                    <span>Show calibration</span>
                    <ScreenReaderOnly id="show-calibration-desc">
                      Toggle visibility of table calibration boundaries and points
                    </ScreenReaderOnly>
                  </label>

                  <label className="flex items-center">
                    <input
                      type="checkbox"
                      checked={overlayConfig.calibration.showPoints}
                      onChange={(e) => updateOverlayConfig('calibration', { showPoints: e.target.checked })}
                      className="mr-2"
                      disabled={!overlayConfig.calibration.visible}
                    />
                    Show corner points
                  </label>

                  <label className="flex items-center">
                    <input
                      type="checkbox"
                      checked={overlayConfig.calibration.showLines}
                      onChange={(e) => updateOverlayConfig('calibration', { showLines: e.target.checked })}
                      className="mr-2"
                      disabled={!overlayConfig.calibration.visible}
                    />
                    Show boundary lines
                  </label>

                  <label className="flex items-center">
                    <input
                      type="checkbox"
                      checked={overlayConfig.calibration.showLabels}
                      onChange={(e) => updateOverlayConfig('calibration', { showLabels: e.target.checked })}
                      className="mr-2"
                      disabled={!overlayConfig.calibration.visible}
                    />
                    Show labels
                  </label>

                  <label className="flex items-center">
                    <input
                      type="checkbox"
                      checked={overlayConfig.calibration.showAccuracy}
                      onChange={(e) => updateOverlayConfig('calibration', { showAccuracy: e.target.checked })}
                      className="mr-2"
                      disabled={!overlayConfig.calibration.visible}
                    />
                    Show accuracy info
                  </label>

                  <div className="flex items-center space-x-2">
                    <label className="text-sm">Opacity:</label>
                    <input
                      type="range"
                      min="0"
                      max="1"
                      step="0.1"
                      value={overlayConfig.calibration.opacity}
                      onChange={(e) => updateOverlayConfig('calibration', { opacity: parseFloat(e.target.value) })}
                      className="flex-1"
                      disabled={!overlayConfig.calibration.visible}
                    />
                    <span className="text-sm w-10">{Math.round(overlayConfig.calibration.opacity * 100)}%</span>
                  </div>

                  {videoStore.calibrationData && (
                    <div className="mt-2 p-2 bg-purple-50 rounded text-xs">
                      <div className="font-medium text-purple-800">Calibration Info:</div>
                      <div className="text-gray-600 mt-1">
                        <div>Points: {videoStore.calibrationData.corners.length}</div>
                        {videoStore.calibrationData.accuracy && (
                          <div>Accuracy: {Math.round(videoStore.calibrationData.accuracy * 100)}%</div>
                        )}
                        {videoStore.calibrationData.calibratedAt && (
                          <div>
                            Calibrated: {new Date(videoStore.calibrationData.calibratedAt).toLocaleString()}
                          </div>
                        )}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>

          {/* Debug info */}
          <div className="p-4 border-t bg-gray-50 text-xs">
            <h4 className="font-medium mb-2">Debug Info</h4>
            <div className="space-y-1 text-gray-600">
              <div>Balls: {videoStore.currentBalls.length}</div>
              <div>Trajectories: {videoStore.currentTrajectories.length}</div>
              <div>Table: {videoStore.currentTable ? 'Detected' : 'Not detected'}</div>
              <div>Cue: {videoStore.currentCue?.detected ? 'Detected' : 'Not detected'}</div>
              <div className={videoStore.hasCalibration ? 'text-purple-600 font-medium' : ''}>
                Calibration: {videoStore.hasCalibration ? 'Active' : 'Not available'}
              </div>
              <div>Video: {videoSize.width}×{videoSize.height}</div>
              <div>Canvas: {containerSize.width}×{containerSize.height}</div>
              <div>Transform: {Math.round(transform.scale * 100)}% scale</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
});

LiveView.displayName = 'LiveView';
