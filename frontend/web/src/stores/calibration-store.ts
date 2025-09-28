/**
 * MobX store for calibration management
 */

import { makeAutoObservable, runInAction, flow, observable } from 'mobx';
import {
  CalibrationStartRequest,
  CalibrationStartResponse,
  CalibrationPointRequest,
  CalibrationPointResponse,
  CalibrationApplyResponse,
  CalibrationSession,
} from '../types/api';
import type { RootStore } from './index';

export interface CalibrationPoint {
  id: string;
  screenX: number;
  screenY: number;
  tableX: number;
  tableY: number;
  accuracy: number;
  timestamp: Date;
}

export interface CalibrationProgress {
  totalPoints: number;
  capturedPoints: number;
  remainingPoints: number;
  canProceed: boolean;
  accuracy: number;
  estimatedTimeRemaining: number; // in seconds
}

export interface CalibrationStats {
  averageAccuracy: number;
  minAccuracy: number;
  maxAccuracy: number;
  totalTime: number; // in seconds
  pointsPerMinute: number;
}

export class CalibrationStore {
  private rootStore: RootStore;

  // Current calibration session
  currentSession: CalibrationSession | null = null;
  sessionInstructions: string[] = [];
  capturedPoints = observable.array<CalibrationPoint>([]);

  // Calibration state
  isActive = false;
  isCapturing = false;
  isApplying = false;
  isCompleted = false;

  // Loading and error states
  isLoading = false;
  error: string | null = null;

  // UI state for calibration process
  currentStep = 0;
  showInstructions = true;
  nextPointPosition: { x: number; y: number } | null = null;

  // Calibration types and settings
  readonly calibrationTypes = [
    { id: 'table_corners', name: 'Table Corners', description: 'Calibrate the four corners of the table' },
    { id: 'full_table', name: 'Full Table', description: 'Comprehensive table calibration with multiple points' },
    { id: 'projector', name: 'Projector', description: 'Calibrate projector alignment' },
    { id: 'camera', name: 'Camera', description: 'Calibrate camera perspective' },
  ];

  constructor(rootStore: RootStore) {
    makeAutoObservable(this, {}, { autoBind: true });
    this.rootStore = rootStore;
  }

  // =============================================================================
  // Calibration Session Management
  // =============================================================================

  startCalibration = flow(function* (this: CalibrationStore, request: CalibrationStartRequest) {
    this.isLoading = true;
    this.error = null;

    try {
      const response: CalibrationStartResponse = yield this.rootStore.apiService.startCalibration(request);

      runInAction(() => {
        this.currentSession = response.session;
        this.sessionInstructions = response.instructions;
        this.isActive = true;
        this.isCompleted = false;
        this.currentStep = 0;
        this.showInstructions = true;
        this.capturedPoints.clear();
        this.isLoading = false;

        // Generate first point position if needed
        this.generateNextPointPosition();
      });

      return response;

    } catch (error) {
      runInAction(() => {
        this.isLoading = false;
        this.error = error instanceof Error ? error.message : 'Failed to start calibration';
      });
      throw error;
    }
  });

  capturePoint = flow(function* (this: CalibrationStore, screenX: number, screenY: number, tableX: number, tableY: number) {
    if (!this.currentSession) {
      throw new Error('No active calibration session');
    }

    this.isCapturing = true;
    this.error = null;

    try {
      const request: CalibrationPointRequest = {
        x: tableX,
        y: tableY,
        screen_x: screenX,
        screen_y: screenY,
      };

      const response: CalibrationPointResponse = yield this.rootStore.apiService.captureCalibrationPoint(
        this.currentSession.session_id,
        request
      );

      runInAction(() => {
        if (response.success) {
          // Add captured point
          const point: CalibrationPoint = {
            id: response.point_id,
            screenX,
            screenY,
            tableX,
            tableY,
            accuracy: response.accuracy,
            timestamp: new Date(),
          };

          this.capturedPoints.push(point);

          // Update session progress
          if (this.currentSession) {
            this.currentSession.points_captured = response.total_points;
            this.currentSession.accuracy = response.accuracy;
          }

          // Move to next step or complete
          if (response.can_proceed && response.remaining_points === 0) {
            this.isCompleted = true;
          } else {
            this.currentStep++;
            this.generateNextPointPosition();
          }
        }

        this.isCapturing = false;
      });

      return response;

    } catch (error) {
      runInAction(() => {
        this.isCapturing = false;
        this.error = error instanceof Error ? error.message : 'Failed to capture calibration point';
      });
      throw error;
    }
  });

  applyCalibration = flow(function* (this: CalibrationStore) {
    if (!this.currentSession) {
      throw new Error('No active calibration session');
    }

    this.isApplying = true;
    this.error = null;

    try {
      const response: CalibrationApplyResponse = yield this.rootStore.apiService.applyCalibration(
        this.currentSession.session_id
      );

      runInAction(() => {
        if (response.success) {
          this.isActive = false;
          this.isCompleted = true;

          // Update session with final results
          if (this.currentSession) {
            this.currentSession.accuracy = response.accuracy;
            this.currentSession.status = 'completed';
          }
        } else {
          this.error = response.errors.join(', ') || 'Failed to apply calibration';
        }

        this.isApplying = false;
      });

      return response;

    } catch (error) {
      runInAction(() => {
        this.isApplying = false;
        this.error = error instanceof Error ? error.message : 'Failed to apply calibration';
      });
      throw error;
    }
  });

  cancelCalibration = flow(function* (this: CalibrationStore) {
    if (!this.currentSession) return;

    this.isLoading = true;
    this.error = null;

    try {
      yield this.rootStore.apiService.cancelCalibration(this.currentSession.session_id);

      runInAction(() => {
        this.resetCalibration();
        this.isLoading = false;
      });

    } catch (error) {
      runInAction(() => {
        this.isLoading = false;
        this.error = error instanceof Error ? error.message : 'Failed to cancel calibration';
      });
    }
  });

  // =============================================================================
  // UI State Management
  // =============================================================================

  setShowInstructions(show: boolean): void {
    this.showInstructions = show;
  }

  nextStep(): void {
    if (this.currentStep < this.sessionInstructions.length - 1) {
      this.currentStep++;
      this.generateNextPointPosition();
    }
  }

  previousStep(): void {
    if (this.currentStep > 0) {
      this.currentStep--;
      this.generateNextPointPosition();
    }
  }

  removeLastPoint(): void {
    if (this.capturedPoints.length > 0) {
      this.capturedPoints.pop();
      if (this.currentSession) {
        this.currentSession.points_captured = this.capturedPoints.length;
      }
      if (this.currentStep > 0) {
        this.currentStep--;
      }
      this.generateNextPointPosition();
    }
  }

  // =============================================================================
  // Computed Properties
  // =============================================================================

  get progress(): CalibrationProgress {
    const totalPoints = this.currentSession?.points_required || 0;
    const capturedPoints = this.capturedPoints.length;
    const remainingPoints = Math.max(0, totalPoints - capturedPoints);

    // Calculate estimated time remaining based on current pace
    let estimatedTimeRemaining = 0;
    if (capturedPoints > 0 && remainingPoints > 0) {
      const totalTime = this.stats.totalTime;
      const averageTimePerPoint = totalTime / capturedPoints;
      estimatedTimeRemaining = averageTimePerPoint * remainingPoints;
    }

    return {
      totalPoints,
      capturedPoints,
      remainingPoints,
      canProceed: remainingPoints === 0 && capturedPoints > 0,
      accuracy: this.currentSession?.accuracy || 0,
      estimatedTimeRemaining,
    };
  }

  get stats(): CalibrationStats {
    if (this.capturedPoints.length === 0) {
      return {
        averageAccuracy: 0,
        minAccuracy: 0,
        maxAccuracy: 0,
        totalTime: 0,
        pointsPerMinute: 0,
      };
    }

    const accuracies = this.capturedPoints.map(p => p.accuracy);
    const averageAccuracy = accuracies.reduce((sum, acc) => sum + acc, 0) / accuracies.length;
    const minAccuracy = Math.min(...accuracies);
    const maxAccuracy = Math.max(...accuracies);

    // Calculate total time from first to last point
    const firstPoint = this.capturedPoints[0];
    const lastPoint = this.capturedPoints[this.capturedPoints.length - 1];
    const totalTime = (lastPoint.timestamp.getTime() - firstPoint.timestamp.getTime()) / 1000;

    const pointsPerMinute = totalTime > 0 ? (this.capturedPoints.length / totalTime) * 60 : 0;

    return {
      averageAccuracy,
      minAccuracy,
      maxAccuracy,
      totalTime,
      pointsPerMinute,
    };
  }

  get currentInstruction(): string {
    if (!this.sessionInstructions || this.currentStep >= this.sessionInstructions.length) {
      return '';
    }
    return this.sessionInstructions[this.currentStep];
  }

  get canProceedToNext(): boolean {
    return this.currentStep < this.sessionInstructions.length - 1;
  }

  get canGoToPrevious(): boolean {
    return this.currentStep > 0;
  }

  get isReadyToApply(): boolean {
    return this.isCompleted && !this.isApplying && this.progress.canProceed;
  }

  get calibrationTypeInfo(): { id: string; name: string; description: string } | null {
    if (!this.currentSession) return null;
    return this.calibrationTypes.find(type => type.id === this.currentSession!.calibration_type) || null;
  }

  // =============================================================================
  // Helper Methods
  // =============================================================================

  private generateNextPointPosition(): void {
    if (!this.currentSession) {
      this.nextPointPosition = null;
      return;
    }

    // Generate suggested position based on calibration type and current step
    const type = this.currentSession.calibration_type;
    const step = this.currentStep;

    switch (type) {
      case 'table_corners':
        this.nextPointPosition = this.generateCornerPosition(step);
        break;

      case 'full_table':
        this.nextPointPosition = this.generateGridPosition(step);
        break;

      case 'projector':
      case 'camera':
        this.nextPointPosition = this.generateCenterOutPosition(step);
        break;

      default:
        this.nextPointPosition = this.generateRandomPosition();
    }
  }

  private generateCornerPosition(step: number): { x: number; y: number } {
    // Generate positions for table corners
    const positions = [
      { x: 0.1, y: 0.1 }, // Top-left
      { x: 0.9, y: 0.1 }, // Top-right
      { x: 0.9, y: 0.9 }, // Bottom-right
      { x: 0.1, y: 0.9 }, // Bottom-left
    ];

    const position = positions[step % positions.length];
    return {
      x: position.x * window.innerWidth,
      y: position.y * window.innerHeight,
    };
  }

  private generateGridPosition(step: number): { x: number; y: number } {
    // Generate positions in a grid pattern
    const gridSize = 4; // 4x4 grid
    const row = Math.floor(step / gridSize);
    const col = step % gridSize;

    const x = (col + 1) / (gridSize + 1);
    const y = (row + 1) / (gridSize + 1);

    return {
      x: x * window.innerWidth,
      y: y * window.innerHeight,
    };
  }

  private generateCenterOutPosition(step: number): { x: number; y: number } {
    // Generate positions starting from center and moving outward
    const positions = [
      { x: 0.5, y: 0.5 }, // Center
      { x: 0.3, y: 0.3 }, // Top-left quadrant
      { x: 0.7, y: 0.3 }, // Top-right quadrant
      { x: 0.7, y: 0.7 }, // Bottom-right quadrant
      { x: 0.3, y: 0.7 }, // Bottom-left quadrant
      { x: 0.5, y: 0.2 }, // Top center
      { x: 0.8, y: 0.5 }, // Right center
      { x: 0.5, y: 0.8 }, // Bottom center
      { x: 0.2, y: 0.5 }, // Left center
    ];

    const position = positions[step % positions.length];
    return {
      x: position.x * window.innerWidth,
      y: position.y * window.innerHeight,
    };
  }

  private generateRandomPosition(): { x: number; y: number } {
    return {
      x: Math.random() * window.innerWidth,
      y: Math.random() * window.innerHeight,
    };
  }

  getCalibrationTypeById(id: string): { id: string; name: string; description: string } | null {
    return this.calibrationTypes.find(type => type.id === id) || null;
  }

  getPointById(id: string): CalibrationPoint | null {
    return this.capturedPoints.find(point => point.id === id) || null;
  }

  // =============================================================================
  // Validation and Quality Checks
  // =============================================================================

  validatePointAccuracy(accuracy: number): boolean {
    return accuracy >= 0.7; // 70% minimum accuracy threshold
  }

  getQualityStatus(): 'excellent' | 'good' | 'fair' | 'poor' {
    const { averageAccuracy } = this.stats;

    if (averageAccuracy >= 0.95) return 'excellent';
    if (averageAccuracy >= 0.85) return 'good';
    if (averageAccuracy >= 0.7) return 'fair';
    return 'poor';
  }

  getQualityRecommendations(): string[] {
    const recommendations: string[] = [];
    const { averageAccuracy, minAccuracy } = this.stats;

    if (averageAccuracy < 0.8) {
      recommendations.push('Consider recalibrating with more precise point placement');
    }

    if (minAccuracy < 0.6) {
      recommendations.push('Some points have very low accuracy - consider removing and recapturing them');
    }

    if (this.capturedPoints.length < 4) {
      recommendations.push('Capture more points for better accuracy');
    }

    const timeBetweenPoints = this.stats.totalTime / Math.max(1, this.capturedPoints.length - 1);
    if (timeBetweenPoints < 2) {
      recommendations.push('Take more time between points for better precision');
    }

    return recommendations;
  }

  // =============================================================================
  // Store Lifecycle
  // =============================================================================

  private resetCalibration(): void {
    this.currentSession = null;
    this.sessionInstructions = [];
    this.capturedPoints.clear();
    this.isActive = false;
    this.isCapturing = false;
    this.isApplying = false;
    this.isCompleted = false;
    this.currentStep = 0;
    this.showInstructions = true;
    this.nextPointPosition = null;
    this.error = null;
  }

  clearError(): void {
    this.error = null;
  }

  reset(): void {
    this.resetCalibration();
    this.isLoading = false;
  }

  destroy(): void {
    // Cancel active calibration if any
    if (this.currentSession && this.isActive) {
      this.cancelCalibration().catch(console.warn);
    }
    this.reset();
  }
}
