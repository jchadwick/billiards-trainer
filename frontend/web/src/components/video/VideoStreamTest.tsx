/**
 * Test component for video streaming functionality
 */

import React, { useState, useEffect } from 'react';
import { VideoStore } from '../../stores/VideoStore';
import { LiveView } from './LiveView';
import type { DetectionFrame, Ball, Trajectory, Table, CueStick } from '../../types/video';

// Mock data for testing when no backend is available
const generateMockData = (): DetectionFrame => {
  const balls: Ball[] = [
    {
      id: 'cue-ball',
      position: { x: 400, y: 300 },
      radius: 12,
      type: 'cue',
      velocity: { x: 0, y: 0 },
      confidence: 0.95,
      color: '#FFFFFF',
    },
    {
      id: 'ball-1',
      position: { x: 500, y: 250 },
      radius: 12,
      type: 'solid',
      number: 1,
      velocity: { x: 0, y: 0 },
      confidence: 0.92,
      color: '#FFD700',
    },
    {
      id: 'ball-8',
      position: { x: 600, y: 300 },
      radius: 12,
      type: 'eight',
      number: 8,
      velocity: { x: 0, y: 0 },
      confidence: 0.88,
      color: '#000000',
    },
    {
      id: 'ball-9',
      position: { x: 450, y: 350 },
      radius: 12,
      type: 'stripe',
      number: 9,
      velocity: { x: 0, y: 0 },
      confidence: 0.91,
      color: '#FFD700',
    },
  ];

  const table: Table = {
    corners: [
      { x: 100, y: 100 },
      { x: 700, y: 100 },
      { x: 700, y: 500 },
      { x: 100, y: 500 },
    ],
    pockets: [
      { x: 100, y: 100 },
      { x: 400, y: 90 },
      { x: 700, y: 100 },
      { x: 700, y: 500 },
      { x: 400, y: 510 },
      { x: 100, y: 500 },
    ],
    bounds: { x: 100, y: 100, width: 600, height: 400 },
    rails: [
      [{ x: 100, y: 100 }, { x: 700, y: 100 }],
      [{ x: 700, y: 100 }, { x: 700, y: 500 }],
      [{ x: 700, y: 500 }, { x: 100, y: 500 }],
      [{ x: 100, y: 500 }, { x: 100, y: 100 }],
    ],
    detected: true,
    confidence: 0.85,
  };

  const cue: CueStick = {
    tipPosition: { x: 350, y: 280 },
    tailPosition: { x: 250, y: 260 },
    angle: Math.atan2(280 - 260, 350 - 250),
    elevation: 0.1,
    detected: true,
    confidence: 0.87,
    length: Math.sqrt((350 - 250) ** 2 + (280 - 260) ** 2),
  };

  const trajectories: Trajectory[] = [
    {
      ballId: 'cue-ball',
      points: [
        { x: 400, y: 300 },
        { x: 450, y: 275 },
        { x: 500, y: 250 },
        { x: 550, y: 225 },
        { x: 600, y: 200 },
      ],
      collisions: [
        {
          position: { x: 500, y: 250 },
          type: 'ball',
          targetId: 'ball-1',
          angle: 0.5,
          impulse: 10,
        },
      ],
      type: 'primary',
      probability: 0.85,
      color: '#00FF00',
    },
  ];

  return {
    balls,
    cue,
    table,
    trajectories,
    timestamp: Date.now(),
    frameNumber: Math.floor(Date.now() / 16), // ~60 FPS
    processingTime: 16.7,
  };
};

interface VideoStreamTestProps {
  useMockData?: boolean;
  mockUpdateInterval?: number;
}

export const VideoStreamTest: React.FC<VideoStreamTestProps> = ({
  useMockData = false,
  mockUpdateInterval = 100,
}) => {
  const [videoStore] = useState(() => new VideoStore());
  const [testMode, setTestMode] = useState(useMockData);

  // Mock data simulation
  useEffect(() => {
    if (!testMode) return;

    const interval = setInterval(() => {
      const mockFrame = generateMockData();

      // Animate balls slightly
      mockFrame.balls.forEach(ball => {
        ball.position.x += (Math.random() - 0.5) * 2;
        ball.position.y += (Math.random() - 0.5) * 2;

        // Add some velocity to cue ball occasionally
        if (ball.id === 'cue-ball' && Math.random() < 0.1) {
          ball.velocity.x = (Math.random() - 0.5) * 20;
          ball.velocity.y = (Math.random() - 0.5) * 20;
        }
      });

      // Animate cue stick
      if (mockFrame.cue) {
        const time = Date.now() / 1000;
        mockFrame.cue.angle += Math.sin(time) * 0.01;
        mockFrame.cue.tipPosition.x = 350 + Math.sin(time * 2) * 5;
        mockFrame.cue.tipPosition.y = 280 + Math.cos(time * 2) * 5;
      }

      // Update trajectory
      if (mockFrame.trajectories.length > 0) {
        const trajectory = mockFrame.trajectories[0];
        trajectory.points = trajectory.points.map((point, index) => ({
          x: point.x + Math.sin(Date.now() / 1000 + index) * 2,
          y: point.y + Math.cos(Date.now() / 1000 + index) * 2,
        }));
      }

      videoStore.setCurrentFrame(mockFrame);
    }, mockUpdateInterval);

    // Simulate connection
    videoStore.setStatus({ connected: true, streaming: true });

    return () => {
      clearInterval(interval);
    };
  }, [testMode, mockUpdateInterval, videoStore]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      videoStore.dispose();
    };
  }, [videoStore]);

  return (
    <div className="h-screen flex flex-col">
      {/* Test controls */}
      <div className="flex-none p-4 bg-yellow-100 border-b">
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-semibold">Video Stream Test Mode</h2>
          <div className="flex items-center space-x-4">
            <label className="flex items-center">
              <input
                type="checkbox"
                checked={testMode}
                onChange={(e) => setTestMode(e.target.checked)}
                className="mr-2"
              />
              Use Mock Data
            </label>
            <div className="text-sm text-gray-600">
              FPS: {Math.round(videoStore.status.fps)} |
              Balls: {videoStore.currentBalls.length} |
              Trajectories: {videoStore.currentTrajectories.length}
            </div>
          </div>
        </div>
      </div>

      {/* Live view */}
      <div className="flex-1">
        <LiveView
          autoConnect={!testMode}
          baseUrl="http://localhost:8000"
        />
      </div>
    </div>
  );
};
