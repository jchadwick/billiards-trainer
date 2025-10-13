/**
 * Stream controls component for managing video stream settings and playback
 */

import React, { useCallback, useState } from 'react';
import { observer } from 'mobx-react-lite';
import type { VideoStore } from '../../stores/VideoStore';
import type { VideoQuality } from '../../types/video';

interface StreamControlsProps {
  videoStore: VideoStore;
  className?: string;
  showAdvanced?: boolean;
  onFullscreen?: () => void;
  onScreenshot?: () => void;
}

const qualityOptions: { value: VideoQuality; label: string; description: string }[] = [
  { value: 'low', label: 'Low', description: '50% quality, faster streaming' },
  { value: 'medium', label: 'Medium', description: '70% quality, balanced' },
  { value: 'high', label: 'High', description: '85% quality, good detail' },
  { value: 'ultra', label: 'Ultra', description: '95% quality, best detail' },
];

const fpsOptions = [15, 24, 30, 60];

export const StreamControls = observer<StreamControlsProps>(({
  videoStore,
  className = '',
  showAdvanced = false,
  onFullscreen,
  onScreenshot,
}) => {
  const [isAdvancedOpen, setIsAdvancedOpen] = useState(false);

  // Connection controls
  const handleConnect = useCallback(async () => {
    try {
      // Use centralized configuration from axios client
      await videoStore.connect();
    } catch (error) {
      console.error('Failed to connect:', error);
    }
  }, [videoStore]);

  const handleDisconnect = useCallback(() => {
    videoStore.disconnect();
  }, [videoStore]);

  // Quality controls
  const handleQualityChange = useCallback((quality: VideoQuality) => {
    videoStore.setQuality(quality);
  }, [videoStore]);

  // FPS controls
  const handleFPSChange = useCallback((fps: number) => {
    videoStore.setFPS(fps);
  }, [videoStore]);

  // Auto-reconnect toggle
  const handleAutoReconnectChange = useCallback((enabled: boolean) => {
    videoStore.setConfig({ autoReconnect: enabled });
  }, [videoStore]);

  // Error management
  const handleClearErrors = useCallback(() => {
    videoStore.clearErrors();
  }, [videoStore]);

  // Screenshot handler
  const handleScreenshot = useCallback(async () => {
    try {
      onScreenshot?.();
    } catch (error) {
      console.error('Screenshot failed:', error);
    }
  }, [onScreenshot]);

  return (
    <div className={`bg-white shadow-lg rounded-lg border ${className}`}>
      {/* Main Controls */}
      <div className="p-4">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-gray-900">Stream Controls</h3>

          {/* Connection Status */}
          <div className="flex items-center space-x-2">
            <div className={`w-3 h-3 rounded-full ${
              videoStore.isConnected
                ? videoStore.isStreaming
                  ? 'bg-green-500'
                  : 'bg-yellow-500'
                : 'bg-red-500'
            }`} />
            <span className="text-sm text-gray-600">
              {videoStore.isConnected
                ? videoStore.isStreaming
                  ? 'Streaming'
                  : 'Connected'
                : 'Disconnected'
              }
            </span>
          </div>
        </div>

        {/* Primary Controls */}
        <div className="flex items-center space-x-3 mb-4">
          {/* Connect/Disconnect */}
          {!videoStore.isConnected ? (
            <button
              onClick={handleConnect}
              disabled={videoStore.isLoading}
              className="px-4 py-2 bg-green-600 hover:bg-green-700 disabled:bg-gray-400
                       text-white rounded-md font-medium transition-colors"
            >
              {videoStore.isLoading ? 'Connecting...' : 'Connect'}
            </button>
          ) : (
            <button
              onClick={handleDisconnect}
              className="px-4 py-2 bg-red-600 hover:bg-red-700 text-white rounded-md font-medium transition-colors"
            >
              Disconnect
            </button>
          )}

          {/* Quality Selector */}
          <div className="flex items-center space-x-2">
            <label className="text-sm font-medium text-gray-700">Quality:</label>
            <select
              value={videoStore.config.quality}
              onChange={(e) => handleQualityChange(e.target.value as VideoQuality)}
              disabled={!videoStore.isConnected}
              className="px-3 py-1 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:bg-gray-100"
            >
              {qualityOptions.map(option => (
                <option key={option.value} value={option.value} title={option.description}>
                  {option.label}
                </option>
              ))}
            </select>
          </div>

          {/* FPS Selector */}
          <div className="flex items-center space-x-2">
            <label className="text-sm font-medium text-gray-700">FPS:</label>
            <select
              value={videoStore.config.fps}
              onChange={(e) => handleFPSChange(parseInt(e.target.value))}
              disabled={!videoStore.isConnected}
              className="px-3 py-1 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:bg-gray-100"
            >
              {fpsOptions.map(fps => (
                <option key={fps} value={fps}>{fps}</option>
              ))}
            </select>
          </div>

          {/* Action Buttons */}
          <div className="flex items-center space-x-2 ml-auto">
            {onScreenshot && (
              <button
                onClick={handleScreenshot}
                disabled={!videoStore.isStreaming}
                className="p-2 text-gray-600 hover:text-gray-800 disabled:text-gray-400 transition-colors"
                title="Take Screenshot"
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                        d="M3 9a2 2 0 012-2h.93a2 2 0 001.664-.89l.812-1.22A2 2 0 0110.07 4h3.86a2 2 0 011.664.89l.812 1.22A2 2 0 0018.07 7H19a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V9z" />
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                        d="M15 13a3 3 0 11-6 0 3 3 0 016 0z" />
                </svg>
              </button>
            )}

            {onFullscreen && (
              <button
                onClick={onFullscreen}
                disabled={!videoStore.isStreaming}
                className="p-2 text-gray-600 hover:text-gray-800 disabled:text-gray-400 transition-colors"
                title="Fullscreen"
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                        d="M4 8V4m0 0h4M4 4l5 5m11-1V4m0 0h-4m4 0l-5 5M4 16v4m0 0h4m-4 0l5-5m11 5l-5-5m5 5v-4m0 4h-4" />
                </svg>
              </button>
            )}

            {showAdvanced && (
              <button
                onClick={() => setIsAdvancedOpen(!isAdvancedOpen)}
                className="p-2 text-gray-600 hover:text-gray-800 transition-colors"
                title="Advanced Settings"
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                        d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                        d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                </svg>
              </button>
            )}
          </div>
        </div>

        {/* Stream Statistics */}
        {videoStore.isConnected && (
          <div className="grid grid-cols-4 gap-4 text-center border-t pt-3">
            <div>
              <div className="text-sm font-medium text-gray-900">
                {Math.round(videoStore.status.fps)}
              </div>
              <div className="text-xs text-gray-500">FPS</div>
            </div>
            <div>
              <div className="text-sm font-medium text-gray-900">
                {videoStore.status.latency}ms
              </div>
              <div className="text-xs text-gray-500">Latency</div>
            </div>
            <div>
              <div className="text-sm font-medium text-gray-900">
                {videoStore.status.errors}
              </div>
              <div className="text-xs text-gray-500">Errors</div>
            </div>
            <div>
              <div className="text-sm font-medium text-gray-900">
                {Math.round(videoStore.performance.renderFPS)}
              </div>
              <div className="text-xs text-gray-500">Render FPS</div>
            </div>
          </div>
        )}
      </div>

      {/* Advanced Settings */}
      {showAdvanced && isAdvancedOpen && (
        <div className="border-t bg-gray-50 p-4">
          <h4 className="text-md font-medium text-gray-900 mb-3">Advanced Settings</h4>

          <div className="space-y-3">
            {/* Auto-reconnect */}
            <div className="flex items-center justify-between">
              <label className="text-sm font-medium text-gray-700">
                Auto-reconnect
              </label>
              <input
                type="checkbox"
                checked={videoStore.config.autoReconnect}
                onChange={(e) => handleAutoReconnectChange(e.target.checked)}
                className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
              />
            </div>

            {/* Reconnect Delay */}
            <div className="flex items-center justify-between">
              <label className="text-sm font-medium text-gray-700">
                Reconnect Delay (ms)
              </label>
              <input
                type="number"
                value={videoStore.config.reconnectDelay}
                onChange={(e) => videoStore.setConfig({ reconnectDelay: parseInt(e.target.value) || 2000 })}
                min="1000"
                max="30000"
                step="1000"
                className="w-20 px-2 py-1 border border-gray-300 rounded text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>

            {/* Error Management */}
            {videoStore.hasErrors && (
              <div className="pt-2 border-t">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-medium text-red-600">
                    {videoStore.errors.length} error(s)
                  </span>
                  <button
                    onClick={handleClearErrors}
                    className="text-sm text-blue-600 hover:text-blue-800"
                  >
                    Clear
                  </button>
                </div>

                {videoStore.latestError && (
                  <div className="text-xs text-gray-600 bg-red-50 p-2 rounded">
                    <div className="font-medium">{videoStore.latestError.code}</div>
                    <div>{videoStore.latestError.message}</div>
                    <div className="text-gray-500">
                      {new Date(videoStore.latestError.timestamp).toLocaleTimeString()}
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
});

StreamControls.displayName = 'StreamControls';
