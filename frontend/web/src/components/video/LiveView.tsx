/**
 * Simple live view component showing video stream with ball detection overlay
 */

import React, { useEffect, useRef, useState } from 'react';

interface Ball {
  id: string;
  position: { x: number; y: number };
  radius: number;
  number?: number;
  type: 'cue' | 'solid' | 'stripe' | 'eight';
  confidence: number;
}

interface LiveViewProps {
  className?: string;
}

export const LiveView: React.FC<LiveViewProps> = ({ className = '' }) => {
  const videoRef = useRef<HTMLImageElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const [balls, setBalls] = useState<Ball[]>([]);
  const [videoSize, setVideoSize] = useState({ width: 0, height: 0 });
  const [isConnected, setIsConnected] = useState(false);

  // Connect to WebSocket for ball detection data
  useEffect(() => {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.hostname}:8000/api/v1/websocket/ws`;

    console.log('Connecting to WebSocket:', wsUrl);
    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;

    ws.onopen = () => {
      console.log('WebSocket connected');
      setIsConnected(true);

      // Subscribe to state stream for ball detection
      ws.send(JSON.stringify({
        type: 'subscribe',
        data: {
          streams: ['state']
        }
      }));
    };

    ws.onmessage = (event) => {
      try {
        const message = JSON.parse(event.data);
        console.log('WebSocket message received:', message.type, message);

        // Handle state messages containing ball data
        if (message.type === 'state' && message.data?.balls) {
          console.log(`Received ${message.data.balls.length} balls`);
          setBalls(message.data.balls);
        }
      } catch (error) {
        console.error('Failed to parse WebSocket message:', error);
      }
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
    };

    ws.onclose = () => {
      console.log('WebSocket disconnected');
      setIsConnected(false);
    };

    return () => {
      ws.close();
    };
  }, []);

  // Track video size when image loads
  useEffect(() => {
    const img = videoRef.current;
    if (!img) return;

    const handleLoad = () => {
      setVideoSize({
        width: img.naturalWidth,
        height: img.naturalHeight
      });
    };

    img.addEventListener('load', handleLoad);
    return () => img.removeEventListener('load', handleLoad);
  }, []);

  // Draw balls on canvas
  useEffect(() => {
    const canvas = canvasRef.current;
    const video = videoRef.current;

    if (!canvas || !video || !videoSize.width || !videoSize.height) return;

    // Match canvas size to video display size
    const rect = video.getBoundingClientRect();
    canvas.width = rect.width;
    canvas.height = rect.height;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Calculate scale factor from video coordinates to canvas coordinates
    const scaleX = rect.width / videoSize.width;
    const scaleY = rect.height / videoSize.height;

    // Draw each ball
    balls.forEach(ball => {
      // Transform ball position to canvas coordinates
      const x = ball.position.x * scaleX;
      const y = ball.position.y * scaleY;
      const radius = ball.radius * Math.min(scaleX, scaleY);

      // Draw ball circle
      ctx.beginPath();
      ctx.arc(x, y, radius, 0, 2 * Math.PI);

      // Set ball color based on type
      const colors = {
        cue: 'rgba(255, 255, 255, 0.7)',
        solid: 'rgba(255, 107, 107, 0.7)',
        stripe: 'rgba(78, 205, 196, 0.7)',
        eight: 'rgba(44, 62, 80, 0.7)',
      };
      ctx.fillStyle = colors[ball.type] || 'rgba(200, 200, 200, 0.7)';
      ctx.fill();

      // Draw outline
      ctx.strokeStyle = '#000000';
      ctx.lineWidth = 2;
      ctx.stroke();

      // Draw ball number if available
      if (ball.number !== undefined) {
        ctx.fillStyle = ball.type === 'eight' ? '#FFFFFF' : '#000000';
        ctx.font = `${Math.max(14, radius * 0.8)}px Arial`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(ball.number.toString(), x, y);
      }
    });
  }, [balls, videoSize]);

  return (
    <div className={`flex flex-col h-screen ${className}`}>
      {/* Header */}
      <div className="flex-none bg-gray-800 text-white p-4">
        <h1 className="text-xl font-bold">Live View</h1>
        <div className="text-sm mt-1">
          <span className={`inline-block w-2 h-2 rounded-full mr-2 ${isConnected ? 'bg-green-500' : 'bg-red-500'}`}></span>
          {isConnected ? 'Connected' : 'Disconnected'} | {balls.length} balls detected
        </div>
      </div>

      {/* Video container */}
      <div className="flex-1 relative bg-black flex items-center justify-center">
        {/* MJPEG video stream */}
        <img
          ref={videoRef}
          src="http://localhost:8000/api/v1/stream/video"
          alt="Video stream"
          className="max-w-full max-h-full object-contain"
          style={{ display: 'block' }}
        />

        {/* Ball detection overlay canvas */}
        <canvas
          ref={canvasRef}
          className="absolute top-0 left-0 pointer-events-none"
          style={{
            width: videoRef.current?.clientWidth || 0,
            height: videoRef.current?.clientHeight || 0,
          }}
        />

        {/* Status overlay when not connected */}
        {!isConnected && (
          <div className="absolute inset-0 flex items-center justify-center bg-gray-900 bg-opacity-75">
            <div className="text-white text-center">
              <div className="text-xl mb-2">Connecting...</div>
              <div className="text-sm">Waiting for video stream and ball detection</div>
            </div>
          </div>
        )}
      </div>

      {/* Ball list */}
      <div className="flex-none bg-gray-100 p-4 border-t">
        <h2 className="font-semibold mb-2">Detected Balls</h2>
        <div className="flex gap-2 flex-wrap">
          {balls.length === 0 ? (
            <div className="text-gray-500 text-sm">No balls detected</div>
          ) : (
            balls.map(ball => (
              <div
                key={ball.id}
                className="px-3 py-1 bg-white border rounded shadow-sm text-sm"
              >
                {ball.number !== undefined ? `Ball ${ball.number}` : ball.type}
                <span className="text-gray-500 ml-2">
                  {Math.round(ball.confidence * 100)}%
                </span>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
};

LiveView.displayName = 'LiveView';
