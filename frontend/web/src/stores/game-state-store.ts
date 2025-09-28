/**
 * MobX store for game state management
 */

import { makeAutoObservable, runInAction, observable, computed } from 'mobx';
import {
  ProcessedGameState,
  ProcessedFrameData,
  ProcessedTrajectory,
  ProcessedBallData,
  ProcessedCueData,
  ProcessedTableData,
} from '../services/data-handlers';
import { GameStateResponse } from '../types/api';
import type { RootStore } from './index';

export interface GameStatistics {
  totalFrames: number;
  ballsDetected: number;
  averageConfidence: number;
  lastUpdate: Date;
  frameRate: number;
  ballMovements: number;
  cueDetections: number;
  trajectoryPredictions: number;
}

export interface BallTrackingData {
  id: string;
  positions: Array<{ position: [number, number]; timestamp: Date }>;
  averageSpeed: number;
  movementPattern: 'stationary' | 'linear' | 'curved' | 'chaotic';
  lastSeen: Date;
}

export class GameStateStore {
  private rootStore: RootStore;

  // Current state
  currentState: ProcessedGameState | null = null;
  currentFrame: ProcessedFrameData | null = null;
  currentTrajectory: ProcessedTrajectory | null = null;

  // History and tracking
  stateHistory = observable.array<ProcessedGameState>([]);
  frameHistory = observable.array<ProcessedFrameData>([]);
  ballTracking = observable.map<string, BallTrackingData>();

  // Loading and error states
  isLoading = false;
  error: string | null = null;

  // Real-time connection status
  isConnectedToStream = false;
  lastStreamUpdate: Date | null = null;
  streamLatency = 0;

  // Configuration
  maxHistoryLength = 100;
  maxBallTrackingPoints = 50;

  constructor(rootStore: RootStore) {
    makeAutoObservable(this, {}, { autoBind: true });
    this.rootStore = rootStore;
    this.setupRealtimeHandlers();
  }

  // =============================================================================
  // Real-time Data Handlers
  // =============================================================================

  private setupRealtimeHandlers(): void {
    // Handle real-time game state updates
    this.rootStore.apiService.onGameStateData((state) => {
      runInAction(() => {
        this.updateGameState(state);
      });
    });

    // Handle real-time frame updates
    this.rootStore.apiService.onFrameData((frame) => {
      runInAction(() => {
        this.updateFrame(frame);
      });
    });

    // Handle trajectory updates
    this.rootStore.apiService.onTrajectoryData((trajectory) => {
      runInAction(() => {
        this.updateTrajectory(trajectory);
      });
    });

    // Monitor connection status
    this.rootStore.apiService.onConnectionStatus((status) => {
      runInAction(() => {
        this.isConnectedToStream = status.websocket === 'connected';
        if (status.lastWebSocketMessage) {
          this.lastStreamUpdate = status.lastWebSocketMessage;
          this.streamLatency = Date.now() - status.lastWebSocketMessage.getTime();
        }
      });
    });
  }

  // =============================================================================
  // State Updates
  // =============================================================================

  private updateGameState(state: ProcessedGameState): void {
    this.currentState = state;
    this.lastStreamUpdate = new Date();

    // Add to history
    this.stateHistory.push(state);
    if (this.stateHistory.length > this.maxHistoryLength) {
      this.stateHistory.shift();
    }

    // Update ball tracking
    state.balls.forEach(ball => {
      this.updateBallTracking(ball);
    });

    // Clear error on successful update
    if (this.error) {
      this.error = null;
    }
  }

  private updateFrame(frame: ProcessedFrameData): void {
    this.currentFrame = frame;

    // Add to frame history (keep fewer frames due to size)
    this.frameHistory.push(frame);
    if (this.frameHistory.length > 10) {
      this.frameHistory.shift();
    }
  }

  private updateTrajectory(trajectory: ProcessedTrajectory): void {
    this.currentTrajectory = trajectory;
  }

  private updateBallTracking(ball: ProcessedBallData): void {
    const existing = this.ballTracking.get(ball.id);
    const now = new Date();

    if (existing) {
      // Add new position
      existing.positions.push({
        position: ball.position,
        timestamp: now,
      });

      // Limit tracking points
      if (existing.positions.length > this.maxBallTrackingPoints) {
        existing.positions.shift();
      }

      // Update tracking data
      existing.lastSeen = now;
      existing.averageSpeed = this.calculateAverageSpeed(existing.positions);
      existing.movementPattern = this.analyzeMovementPattern(existing.positions);

    } else {
      // Create new tracking entry
      this.ballTracking.set(ball.id, {
        id: ball.id,
        positions: [{ position: ball.position, timestamp: now }],
        averageSpeed: 0,
        movementPattern: 'stationary',
        lastSeen: now,
      });
    }
  }

  // =============================================================================
  // Game State Actions
  // =============================================================================

  async loadCurrentState(): Promise<void> {
    this.isLoading = true;
    this.error = null;

    try {
      const response = await this.rootStore.apiService.getCurrentGameState();

      runInAction(() => {
        // Convert API response to processed state if needed
        // This would be handled by the data processing service
        this.isLoading = false;
      });

    } catch (error) {
      runInAction(() => {
        this.isLoading = false;
        this.error = error instanceof Error ? error.message : 'Failed to load game state';
      });
    }
  }

  async loadGameHistory(options: {
    limit?: number;
    startTime?: Date;
    endTime?: Date;
  } = {}): Promise<void> {
    this.isLoading = true;
    this.error = null;

    try {
      const history = await this.rootStore.apiService.getGameHistory(options);

      runInAction(() => {
        // Process and store history
        this.isLoading = false;
      });

    } catch (error) {
      runInAction(() => {
        this.isLoading = false;
        this.error = error instanceof Error ? error.message : 'Failed to load game history';
      });
    }
  }

  async resetGameState(): Promise<void> {
    this.isLoading = true;
    this.error = null;

    try {
      await this.rootStore.apiService.resetGameState();

      runInAction(() => {
        // Clear current state
        this.currentState = null;
        this.stateHistory.clear();
        this.ballTracking.clear();
        this.isLoading = false;
      });

    } catch (error) {
      runInAction(() => {
        this.isLoading = false;
        this.error = error instanceof Error ? error.message : 'Failed to reset game state';
      });
    }
  }

  // =============================================================================
  // Stream Management
  // =============================================================================

  async subscribeToStreams(): Promise<void> {
    try {
      await this.rootStore.apiService.subscribeToStreams([
        'frame',
        'state',
        'trajectory'
      ]);
    } catch (error) {
      runInAction(() => {
        this.error = error instanceof Error ? error.message : 'Failed to subscribe to streams';
      });
    }
  }

  async unsubscribeFromStreams(): Promise<void> {
    try {
      await this.rootStore.apiService.unsubscribeFromStreams([
        'frame',
        'state',
        'trajectory'
      ]);
    } catch (error) {
      console.warn('Failed to unsubscribe from streams:', error);
    }
  }

  // =============================================================================
  // Computed Properties
  // =============================================================================

  get statistics(): GameStatistics {
    const totalFrames = this.stateHistory.length;
    const ballsDetected = this.currentState?.balls.length || 0;

    // Calculate average confidence
    let totalConfidence = 0;
    let confidenceCount = 0;

    this.stateHistory.forEach(state => {
      totalConfidence += state.confidence;
      confidenceCount++;
    });

    const averageConfidence = confidenceCount > 0 ? totalConfidence / confidenceCount : 0;

    // Calculate frame rate (based on last 10 frames)
    let frameRate = 0;
    const recentFrames = this.stateHistory.slice(-10);
    if (recentFrames.length > 1) {
      const timeSpan = recentFrames[recentFrames.length - 1].timestamp.getTime() -
                     recentFrames[0].timestamp.getTime();
      frameRate = (recentFrames.length - 1) / (timeSpan / 1000);
    }

    // Count various events
    const ballMovements = this.stateHistory.reduce((count, state) => {
      return count + state.balls.filter(ball => ball.isMoving).length;
    }, 0);

    const cueDetections = this.stateHistory.filter(state => state.cue?.detected).length;
    const trajectoryPredictions = this.currentTrajectory ? 1 : 0;

    return {
      totalFrames,
      ballsDetected,
      averageConfidence,
      lastUpdate: this.lastStreamUpdate || new Date(),
      frameRate,
      ballMovements,
      cueDetections,
      trajectoryPredictions,
    };
  }

  get ballsOnTable(): ProcessedBallData[] {
    return this.currentState?.balls || [];
  }

  get cueStick(): ProcessedCueData | null {
    return this.currentState?.cue || null;
  }

  get tableGeometry(): ProcessedTableData | null {
    return this.currentState?.table || null;
  }

  get isGameActive(): boolean {
    return this.ballsOnTable.some(ball => ball.isMoving);
  }

  get ballById(): (id: string) => ProcessedBallData | undefined {
    return (id: string) => this.ballsOnTable.find(ball => ball.id === id);
  }

  get movingBalls(): ProcessedBallData[] {
    return this.ballsOnTable.filter(ball => ball.isMoving);
  }

  get stationaryBalls(): ProcessedBallData[] {
    return this.ballsOnTable.filter(ball => !ball.isMoving);
  }

  get currentFrameRate(): number {
    return this.statistics.frameRate;
  }

  get streamHealth(): {
    status: 'excellent' | 'good' | 'poor' | 'disconnected';
    latency: number;
    frameRate: number;
    lastUpdate: Date | null;
  } {
    if (!this.isConnectedToStream) {
      return {
        status: 'disconnected',
        latency: 0,
        frameRate: 0,
        lastUpdate: null,
      };
    }

    const frameRate = this.currentFrameRate;
    let status: 'excellent' | 'good' | 'poor' | 'disconnected' = 'excellent';

    if (this.streamLatency > 100 || frameRate < 15) {
      status = 'poor';
    } else if (this.streamLatency > 50 || frameRate < 25) {
      status = 'good';
    }

    return {
      status,
      latency: this.streamLatency,
      frameRate,
      lastUpdate: this.lastStreamUpdate,
    };
  }

  // =============================================================================
  // Analysis Helpers
  // =============================================================================

  private calculateAverageSpeed(positions: Array<{ position: [number, number]; timestamp: Date }>): number {
    if (positions.length < 2) return 0;

    let totalDistance = 0;
    let totalTime = 0;

    for (let i = 1; i < positions.length; i++) {
      const prev = positions[i - 1];
      const curr = positions[i];

      const dx = curr.position[0] - prev.position[0];
      const dy = curr.position[1] - prev.position[1];
      const distance = Math.sqrt(dx * dx + dy * dy);

      const timeDiff = curr.timestamp.getTime() - prev.timestamp.getTime();

      totalDistance += distance;
      totalTime += timeDiff;
    }

    return totalTime > 0 ? (totalDistance / totalTime) * 1000 : 0; // pixels per second
  }

  private analyzeMovementPattern(
    positions: Array<{ position: [number, number]; timestamp: Date }>
  ): 'stationary' | 'linear' | 'curved' | 'chaotic' {
    if (positions.length < 3) return 'stationary';

    // Calculate movement variations
    const movements = [];
    for (let i = 1; i < positions.length; i++) {
      const prev = positions[i - 1];
      const curr = positions[i];
      const dx = curr.position[0] - prev.position[0];
      const dy = curr.position[1] - prev.position[1];
      const distance = Math.sqrt(dx * dx + dy * dy);
      movements.push(distance);
    }

    const avgMovement = movements.reduce((a, b) => a + b, 0) / movements.length;

    if (avgMovement < 1) return 'stationary';

    // Analyze direction changes
    const angles = [];
    for (let i = 2; i < positions.length; i++) {
      const p1 = positions[i - 2];
      const p2 = positions[i - 1];
      const p3 = positions[i];

      const angle1 = Math.atan2(p2.position[1] - p1.position[1], p2.position[0] - p1.position[0]);
      const angle2 = Math.atan2(p3.position[1] - p2.position[1], p3.position[0] - p2.position[0]);

      let angleDiff = Math.abs(angle2 - angle1);
      if (angleDiff > Math.PI) angleDiff = 2 * Math.PI - angleDiff;

      angles.push(angleDiff);
    }

    const avgAngleChange = angles.reduce((a, b) => a + b, 0) / angles.length;

    if (avgAngleChange < 0.1) return 'linear';
    if (avgAngleChange < 0.5) return 'curved';
    return 'chaotic';
  }

  // =============================================================================
  // Ball Search and Filtering
  // =============================================================================

  findBallsByColor(color: string): ProcessedBallData[] {
    return this.ballsOnTable.filter(ball =>
      ball.color.toLowerCase().includes(color.toLowerCase())
    );
  }

  findBallsInArea(center: [number, number], radius: number): ProcessedBallData[] {
    return this.ballsOnTable.filter(ball => {
      const dx = ball.position[0] - center[0];
      const dy = ball.position[1] - center[1];
      const distance = Math.sqrt(dx * dx + dy * dy);
      return distance <= radius;
    });
  }

  getBallTrajectory(ballId: string): Array<{ position: [number, number]; timestamp: Date }> {
    const tracking = this.ballTracking.get(ballId);
    return tracking ? [...tracking.positions] : [];
  }

  // =============================================================================
  // Store Lifecycle
  // =============================================================================

  reset(): void {
    this.currentState = null;
    this.currentFrame = null;
    this.currentTrajectory = null;
    this.stateHistory.clear();
    this.frameHistory.clear();
    this.ballTracking.clear();
    this.isLoading = false;
    this.error = null;
    this.isConnectedToStream = false;
    this.lastStreamUpdate = null;
    this.streamLatency = 0;
  }

  destroy(): void {
    this.unsubscribeFromStreams();
    this.reset();
  }
}
