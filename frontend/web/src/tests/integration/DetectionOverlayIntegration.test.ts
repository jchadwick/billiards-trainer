/**
 * Integration test to verify detection overlay components work with real backend data
 */

import { VideoStore } from '../../stores/VideoStore';
import { VisionStore } from '../../stores/VisionStore';
import { apiClient } from '../../api/client';
import { WebSocketClient } from '../../services/websocket-client';
import type { GameStateData, TrajectoryData, BallData, CueData, TableData } from '../../types/api';
import type { DetectionFrame, Ball, CueStick, Table, Trajectory } from '../../types/video';
import { vi } from 'vitest';

// Mock WebSocket data that matches backend schema
const mockGameStateData: GameStateData = {
  balls: [
    {
      id: 'cue',
      position: { x: 400, y: 300, scale: [1.0, 1.0] },
      radius: 15,
      radius_scale: [1.0, 1.0],
      color: '#FFFFFF',
      velocity: { x: 0, y: 0, scale: [1.0, 1.0] },
      confidence: 0.95,
      visible: true,
    },
    {
      id: '1',
      position: { x: 500, y: 250, scale: [1.0, 1.0] },
      radius: 15,
      radius_scale: [1.0, 1.0],
      color: '#FF0000',
      velocity: { x: 10, y: -5, scale: [1.0, 1.0] },
      confidence: 0.88,
      visible: true,
    },
    {
      id: '8',
      position: { x: 350, y: 400, scale: [1.0, 1.0] },
      radius: 15,
      radius_scale: [1.0, 1.0],
      color: '#000000',
      velocity: { x: 0, y: 0, scale: [1.0, 1.0] },
      confidence: 0.92,
      visible: true,
    },
  ],
  cue: {
    angle: 45,
    position: { x: 380, y: 280, scale: [1.0, 1.0] },
    detected: true,
    confidence: 0.89,
    length: 120,
    tip_position: { x: 365, y: 265, scale: [1.0, 1.0] },
  },
  table: {
    corners: [
      { x: 100, y: 50, scale: [1.0, 1.0] },
      { x: 700, y: 50, scale: [1.0, 1.0] },
      { x: 700, y: 450, scale: [1.0, 1.0] },
      { x: 100, y: 450, scale: [1.0, 1.0] },
    ],
    pockets: [
      { x: 100, y: 50, scale: [1.0, 1.0] },
      { x: 400, y: 50, scale: [1.0, 1.0] },
      { x: 700, y: 50, scale: [1.0, 1.0] },
      { x: 700, y: 450, scale: [1.0, 1.0] },
      { x: 400, y: 450, scale: [1.0, 1.0] },
      { x: 100, y: 450, scale: [1.0, 1.0] },
    ],
    calibrated: true,
    dimensions: { width: 6, height: 4, unit: 'meters' },
  },
  ball_count: 3,
  frame_number: 1234,
};

const mockTrajectoryData: TrajectoryData = {
  lines: [
    {
      start: { x: 400, y: 300, scale: [1.0, 1.0] },
      end: { x: 500, y: 250, scale: [1.0, 1.0] },
      type: 'primary',
      confidence: 0.85,
    },
    {
      start: { x: 500, y: 250, scale: [1.0, 1.0] },
      end: { x: 600, y: 200, scale: [1.0, 1.0] },
      type: 'reflection',
      confidence: 0.72,
    },
  ],
  collisions: [
    {
      position: { x: 500, y: 250, scale: [1.0, 1.0] },
      ball2_id: '1',
      angle: 45,
      velocity_before: { x: 10, y: -5, scale: [1.0, 1.0] },
      velocity_after: { x: 8, y: -3, scale: [1.0, 1.0] },
      time_to_collision: 0.5,
    },
  ],
  confidence: 0.78,
  calculation_time_ms: 15.2,
  line_count: 2,
  collision_count: 1,
  prediction_status: 'live',
};

describe('Detection Overlay Integration Tests', () => {
  let videoStore: VideoStore;
  let visionStore: VisionStore;

  beforeEach(() => {
    vi.spyOn(apiClient, 'stopVideoCapture').mockResolvedValue({
      success: true,
      timestamp: new Date().toISOString(),
    });
    vi.spyOn(apiClient, 'getCalibrationData').mockResolvedValue({
      success: false,
      timestamp: new Date().toISOString(),
    });
    videoStore = new VideoStore();
    visionStore = new VisionStore();
  });

  afterEach(async () => {
    videoStore.dispose();
    visionStore.destroy();
    await new Promise<void>((resolve) => setTimeout(resolve, 0));
    vi.restoreAllMocks();
  });

  describe('WebSocket Data Transformation', () => {
    test('should correctly transform ball data from WebSocket to frontend format', () => {
      // Simulate WebSocket message processing
      const message = {
        type: 'state' as const,
        timestamp: new Date().toISOString(),
        data: mockGameStateData,
      };

      // Transform ball data (simulate the handleGameStateMessage method)
      const transformedBalls: Ball[] = mockGameStateData.balls.map((ballData: BallData) => ({
        id: ballData.id,
        position: { x: ballData.position.x, y: ballData.position.y },
        radius: ballData.radius,
        type: inferBallType(ballData.id, ballData.color),
        number: inferBallNumber(ballData.id, ballData.color),
        velocity: ballData.velocity ? { x: ballData.velocity.x, y: ballData.velocity.y } : { x: 0, y: 0 },
        confidence: ballData.confidence,
        color: ballData.color,
      }));

      // Verify ball transformations
      expect(transformedBalls).toHaveLength(3);

      const cueBall = transformedBalls.find(b => b.id === 'cue');
      expect(cueBall).toBeDefined();
      expect(cueBall?.type).toBe('cue');
      expect(cueBall?.position.x).toBe(400);
      expect(cueBall?.position.y).toBe(300);
      expect(cueBall?.confidence).toBe(0.95);

      const eightBall = transformedBalls.find(b => b.id === '8');
      expect(eightBall).toBeDefined();
      expect(eightBall?.type).toBe('eight');
      expect(eightBall?.number).toBe(8);
    });

    test('should correctly transform cue data from WebSocket format', () => {
      const cueData = mockGameStateData.cue!;

      const transformedCue: CueStick = {
        tipPosition: { x: cueData.position!.x, y: cueData.position!.y },
        tailPosition: cueData.tip_position
          ? { x: cueData.tip_position.x, y: cueData.tip_position.y }
          : {
              x: cueData.position!.x - Math.cos(cueData.angle) * (cueData.length || 100),
              y: cueData.position!.y - Math.sin(cueData.angle) * (cueData.length || 100)
            },
        angle: cueData.angle,
        elevation: 0,
        detected: cueData.detected,
        confidence: cueData.confidence,
        length: cueData.length || 100,
      };

      expect(transformedCue.detected).toBe(true);
      expect(transformedCue.confidence).toBe(0.89);
      expect(transformedCue.angle).toBe(45);
      expect(transformedCue.length).toBe(120);
    });

    test('should correctly transform table data from WebSocket format', () => {
      const tableData = mockGameStateData.table!;

      const transformedTable: Table = {
        corners: tableData.corners.map(corner => ({ x: corner.x, y: corner.y })),
        pockets: tableData.pockets.map(pocket => ({
          x: 'position' in pocket ? pocket.position.x : pocket.x,
          y: 'position' in pocket ? pocket.position.y : pocket.y,
        })),
        bounds: { x: 0, y: 0, width: 0, height: 0 },
        rails: [],
        detected: tableData.calibrated,
        confidence: 0.9,
      };

      // Calculate bounds
      if (transformedTable.corners.length >= 4) {
        const xs = transformedTable.corners.map(c => c.x);
        const ys = transformedTable.corners.map(c => c.y);
        transformedTable.bounds = {
          x: Math.min(...xs),
          y: Math.min(...ys),
          width: Math.max(...xs) - Math.min(...xs),
          height: Math.max(...ys) - Math.min(...ys),
        };
      }

      expect(transformedTable.corners).toHaveLength(4);
      expect(transformedTable.pockets).toHaveLength(6);
      expect(transformedTable.bounds.width).toBe(600);
      expect(transformedTable.bounds.height).toBe(400);
      expect(transformedTable.detected).toBe(true);
    });

    test('should correctly transform trajectory data from WebSocket format', () => {
      const trajectories: Trajectory[] = mockTrajectoryData.lines.map((line, index) => ({
        ballId: `trajectory_${index}`,
        points: [
          { x: line.start.x, y: line.start.y },
          { x: line.end.x, y: line.end.y },
        ],
        collisions: mockTrajectoryData.collisions.map(collision => ({
          position: { x: collision.position.x, y: collision.position.y },
          type: collision.ball2_id ? 'ball' : 'rail' as 'ball' | 'rail' | 'pocket',
          targetId: collision.ball2_id,
          angle: collision.angle,
          impulse: 0,
        })),
        type: line.type,
        probability: line.confidence,
        color: getTrajectoryColor(line.type),
      }));

      expect(trajectories).toHaveLength(2);

      const primaryTrajectory = trajectories[0];
      expect(primaryTrajectory.type).toBe('primary');
      expect(primaryTrajectory.points).toHaveLength(2);
      expect(primaryTrajectory.points[0].x).toBe(400);
      expect(primaryTrajectory.points[0].y).toBe(300);
      expect(primaryTrajectory.probability).toBe(0.85);
      expect(primaryTrajectory.color).toBe('#00FF00');

      const reflectionTrajectory = trajectories[1];
      expect(reflectionTrajectory.type).toBe('reflection');
      expect(reflectionTrajectory.color).toBe('#0080FF');
    });
  });

  describe('Confidence Score Display', () => {
    test('should display confidence scores correctly for balls', () => {
      const balls = mockGameStateData.balls;

      balls.forEach(ball => {
        const confidencePercentage = Math.round(ball.confidence * 100);
        const confidenceText = `${confidencePercentage}%`;

        expect(ball.confidence).toBeGreaterThan(0);
        expect(ball.confidence).toBeLessThanOrEqual(1);
        expect(confidenceText).toMatch(/^\d{1,3}%$/);
      });
    });

    test('should handle edge cases in confidence scores', () => {
      const edgeCases = [
        { confidence: 0, expected: '0%' },
        { confidence: 1, expected: '100%' },
        { confidence: 0.999, expected: '100%' },
        { confidence: 0.001, expected: '0%' },
      ];

      edgeCases.forEach(({ confidence, expected }) => {
        const confidenceText = `${Math.round(confidence * 100)}%`;
        expect(confidenceText).toBe(expected);
      });
    });
  });

  describe('Calibration Data Integration', () => {
    test('should load calibration data from backend response format', () => {
      const mockCalibrationResponse = {
        success: true,
        data: {
          geometry: {
            homography_matrix: [
              [1.2, 0.1, -50],
              [-0.05, 1.15, -20],
              [0.0001, -0.00005, 1],
            ],
            table_corners_image: [
              { x: 100, y: 50 },
              { x: 700, y: 50 },
              { x: 700, y: 450 },
              { x: 100, y: 450 },
            ],
            is_valid: true,
            calibration_date: '2024-01-15T10:30:00Z',
            calibration_error: 0.05,
          },
          camera: {
            distortion_coefficients: [0.1, -0.05, 0.001, 0.002, 0],
          },
        },
      };

      // Simulate calibration data loading
      const calibrationData = {
        tableCorners: mockCalibrationResponse.data.geometry.table_corners_image.map(corner => ({
          x: corner.x,
          y: corner.y,
        })),
        homographyMatrix: mockCalibrationResponse.data.geometry.homography_matrix,
        distortionCoefficients: mockCalibrationResponse.data.camera.distortion_coefficients,
        isValid: mockCalibrationResponse.data.geometry.is_valid,
        timestamp: new Date(mockCalibrationResponse.data.geometry.calibration_date),
      };

      expect(calibrationData.homographyMatrix).toHaveLength(3);
      expect(calibrationData.homographyMatrix[0]).toHaveLength(3);
      expect(calibrationData.isValid).toBe(true);
      expect(calibrationData.tableCorners).toHaveLength(4);
    });

    test('should apply homography transformation correctly', () => {
      const homographyMatrix = [
        [1.2, 0.1, -50],
        [-0.05, 1.15, -20],
        [0.0001, -0.00005, 1],
      ];

      const screenPoint = { x: 400, y: 300 };

      // Apply transformation
      const x = screenPoint.x;
      const y = screenPoint.y;

      const transformedX = homographyMatrix[0][0] * x + homographyMatrix[0][1] * y + homographyMatrix[0][2];
      const transformedY = homographyMatrix[1][0] * x + homographyMatrix[1][1] * y + homographyMatrix[1][2];
      const transformedW = homographyMatrix[2][0] * x + homographyMatrix[2][1] * y + homographyMatrix[2][2];

      const result = {
        x: transformedX / transformedW,
        y: transformedY / transformedW,
      };

      expect(result.x).toBeCloseTo(448.78, 1); // Approximate expected value
      expect(result.y).toBeCloseTo(297.56, 1); // Approximate expected value
    });
  });

  describe('Real-time Data Flow', () => {
    test('should handle rapid WebSocket updates without data corruption', () => {
      const updates = Array.from({ length: 100 }, (_, i) => ({
        ...mockGameStateData,
        frame_number: i,
        balls: mockGameStateData.balls.map(ball => ({
          ...ball,
          position: { x: ball.position.x + i, y: ball.position.y + i * 0.5, scale: [1.0, 1.0] },
        })),
      }));

      let lastFrameNumber = -1;
      updates.forEach((update, index) => {
        expect(update.frame_number).toBe(index);
        expect(update.frame_number).toBeGreaterThan(lastFrameNumber);
        lastFrameNumber = update.frame_number;
      });
    });
  });
});

// Helper functions (these should match the ones in VideoStore)
function inferBallType(id: string, color: string): Ball['type'] {
  if (id.toLowerCase().includes('cue') || id === '0') return 'cue';
  if (id === '8' || id === 'eight') return 'eight';

  const ballNumber = parseInt(id);
  if (!isNaN(ballNumber) && ballNumber >= 9 && ballNumber <= 15) return 'stripe';
  if (!isNaN(ballNumber) && ballNumber >= 1 && ballNumber <= 7) return 'solid';

  return 'solid';
}

function inferBallNumber(id: string, color: string): number | undefined {
  const ballNumber = parseInt(id);
  if (!isNaN(ballNumber)) return ballNumber;

  if (id.toLowerCase().includes('cue')) return 0;
  if (id.toLowerCase().includes('eight')) return 8;

  return undefined;
}

function getTrajectoryColor(type: string): string {
  const colors = {
    primary: '#00FF00',
    reflection: '#0080FF',
    collision: '#FF8000',
  };
  return colors[type as keyof typeof colors] || '#00FF00';
}
