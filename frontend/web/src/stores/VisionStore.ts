import { makeAutoObservable, runInAction } from 'mobx';
import type {
  CameraInfo,
  CalibrationData,
  DetectionFrame,
  Ball,
  Cue,
  Point2D,
  ActionResult
} from './types';

export class VisionStore {
  // Observable state
  availableCameras: CameraInfo[] = [];
  selectedCamera: CameraInfo | null = null;
  calibrationData: CalibrationData | null = null;

  // Current detection state
  currentFrame: DetectionFrame | null = null;
  isDetecting: boolean = false;
  isCalibrating: boolean = false;

  // Camera feed
  videoElement: HTMLVideoElement | null = null;
  mediaStream: MediaStream | null = null;

  // Detection statistics
  frameCount: number = 0;
  averageProcessingTime: number = 0;
  detectionAccuracy: number = 0;
  lastFrameTimestamp: Date | null = null;

  // Calibration state
  calibrationStep: 'idle' | 'corners' | 'validation' | 'complete' = 'idle';
  markedCorners: Point2D[] = [];
  validationResults: { accuracy: number; error: string | null } | null = null;

  constructor() {
    makeAutoObservable(this, {
      videoElement: false,
      mediaStream: false
    });
  }

  // Computed values
  get isConnected(): boolean {
    return this.selectedCamera?.isConnected || false;
  }

  get isCalibrated(): boolean {
    return this.calibrationData?.isValid || false;
  }

  get detectionRate(): number {
    if (!this.lastFrameTimestamp) return 0;
    const timeSinceLastFrame = Date.now() - this.lastFrameTimestamp.getTime();
    return timeSinceLastFrame < 1000 ? 1000 / (this.averageProcessingTime || 1) : 0;
  }

  get calibrationProgress(): number {
    switch (this.calibrationStep) {
      case 'idle': return 0;
      case 'corners': return Math.min(this.markedCorners.length / 4, 1) * 0.5;
      case 'validation': return 0.75;
      case 'complete': return 1;
      default: return 0;
    }
  }

  get detectedBalls(): Ball[] {
    return this.currentFrame?.balls || [];
  }

  get detectedCue(): Cue | null {
    return this.currentFrame?.cue || null;
  }

  get isStreamActive(): boolean {
    return this.mediaStream !== null && this.mediaStream.active;
  }

  // Actions
  async discoverCameras(): Promise<ActionResult<CameraInfo[]>> {
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      const videoDevices = devices.filter(device => device.kind === 'videoinput');

      const cameras: CameraInfo[] = [];

      for (const device of videoDevices) {
        try {
          // Test if camera is accessible
          const testStream = await navigator.mediaDevices.getUserMedia({
            video: { deviceId: device.deviceId }
          });

          const track = testStream.getVideoTracks()[0];
          const settings = track.getSettings();

          cameras.push({
            id: device.deviceId,
            name: device.label || `Camera ${cameras.length + 1}`,
            resolution: {
              width: settings.width || 1920,
              height: settings.height || 1080
            },
            fps: settings.frameRate || 30,
            isConnected: true,
            isCalibrated: false
          });

          // Stop test stream
          testStream.getTracks().forEach(track => track.stop());
        } catch (error) {
          // Camera not accessible
          cameras.push({
            id: device.deviceId,
            name: device.label || `Camera ${cameras.length + 1}`,
            resolution: { width: 0, height: 0 },
            fps: 0,
            isConnected: false,
            isCalibrated: false
          });
        }
      }

      runInAction(() => {
        this.availableCameras = cameras;
      });

      return {
        success: true,
        data: cameras,
        timestamp: new Date()
      };
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Failed to discover cameras',
        timestamp: new Date()
      };
    }
  }

  async selectCamera(cameraId: string): Promise<ActionResult> {
    try {
      const camera = this.availableCameras.find(cam => cam.id === cameraId);
      if (!camera) {
        throw new Error('Camera not found');
      }

      if (!camera.isConnected) {
        throw new Error('Camera is not connected');
      }

      // Stop current stream if any
      await this.stopStream();

      // Start new stream
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          deviceId: cameraId,
          width: { ideal: camera.resolution.width },
          height: { ideal: camera.resolution.height },
          frameRate: { ideal: camera.fps }
        }
      });

      runInAction(() => {
        this.selectedCamera = camera;
        this.mediaStream = stream;
      });

      return {
        success: true,
        data: { cameraId },
        timestamp: new Date()
      };
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Failed to select camera',
        timestamp: new Date()
      };
    }
  }

  async startStream(videoElement: HTMLVideoElement): Promise<ActionResult> {
    try {
      if (!this.mediaStream) {
        throw new Error('No camera selected');
      }

      videoElement.srcObject = this.mediaStream;
      await videoElement.play();

      runInAction(() => {
        this.videoElement = videoElement;
      });

      return {
        success: true,
        timestamp: new Date()
      };
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Failed to start stream',
        timestamp: new Date()
      };
    }
  }

  async stopStream(): Promise<ActionResult> {
    try {
      if (this.mediaStream) {
        this.mediaStream.getTracks().forEach(track => track.stop());
      }

      if (this.videoElement) {
        this.videoElement.srcObject = null;
      }

      runInAction(() => {
        this.mediaStream = null;
        this.videoElement = null;
        this.isDetecting = false;
      });

      return {
        success: true,
        timestamp: new Date()
      };
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Failed to stop stream',
        timestamp: new Date()
      };
    }
  }

  startDetection(): void {
    runInAction(() => {
      this.isDetecting = true;
      this.frameCount = 0;
      this.averageProcessingTime = 0;
    });
  }

  stopDetection(): void {
    runInAction(() => {
      this.isDetecting = false;
    });
  }

  updateDetectionFrame(frame: DetectionFrame): void {
    runInAction(() => {
      this.currentFrame = frame;
      this.frameCount++;
      this.lastFrameTimestamp = new Date();

      // Update statistics
      this.averageProcessingTime = this.averageProcessingTime === 0
        ? frame.processingTimeMs
        : (this.averageProcessingTime * 0.9 + frame.processingTimeMs * 0.1);

      this.detectionAccuracy = frame.confidence;
    });
  }

  // Calibration actions
  startCalibration(): void {
    runInAction(() => {
      this.isCalibrating = true;
      this.calibrationStep = 'corners';
      this.markedCorners = [];
      this.validationResults = null;
    });
  }

  addCalibrationCorner(point: Point2D): ActionResult {
    if (this.calibrationStep !== 'corners') {
      return {
        success: false,
        error: 'Not in corner marking mode',
        timestamp: new Date()
      };
    }

    if (this.markedCorners.length >= 4) {
      return {
        success: false,
        error: 'All corners already marked',
        timestamp: new Date()
      };
    }

    runInAction(() => {
      this.markedCorners.push(point);

      if (this.markedCorners.length === 4) {
        this.calibrationStep = 'validation';
      }
    });

    return {
      success: true,
      data: { cornerIndex: this.markedCorners.length - 1 },
      timestamp: new Date()
    };
  }

  removeLastCorner(): ActionResult {
    if (this.calibrationStep !== 'corners' || this.markedCorners.length === 0) {
      return {
        success: false,
        error: 'No corners to remove',
        timestamp: new Date()
      };
    }

    runInAction(() => {
      this.markedCorners.pop();
    });

    return {
      success: true,
      timestamp: new Date()
    };
  }

  async validateCalibration(): Promise<ActionResult<CalibrationData>> {
    try {
      if (this.calibrationStep !== 'validation' || this.markedCorners.length !== 4) {
        throw new Error('Invalid calibration state');
      }

      // In a real implementation, this would send the corners to the backend
      // for homography matrix calculation and validation
      const mockHomographyMatrix = [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
      ];

      const calibrationData: CalibrationData = {
        tableCorners: [...this.markedCorners],
        homographyMatrix: mockHomographyMatrix,
        distortionCoefficients: [0, 0, 0, 0, 0],
        isValid: true,
        timestamp: new Date()
      };

      // Mock validation
      const accuracy = Math.random() * 0.3 + 0.7; // 70-100% accuracy

      runInAction(() => {
        this.calibrationData = calibrationData;
        this.validationResults = {
          accuracy,
          error: accuracy < 0.8 ? 'Low calibration accuracy' : null
        };
        this.calibrationStep = 'complete';

        // Update selected camera calibration status
        if (this.selectedCamera) {
          this.selectedCamera.isCalibrated = true;
        }
      });

      return {
        success: true,
        data: calibrationData,
        timestamp: new Date()
      };
    } catch (error) {
      runInAction(() => {
        this.validationResults = {
          accuracy: 0,
          error: error instanceof Error ? error.message : 'Validation failed'
        };
      });

      return {
        success: false,
        error: error instanceof Error ? error.message : 'Calibration validation failed',
        timestamp: new Date()
      };
    }
  }

  async saveCalibration(): Promise<ActionResult> {
    try {
      if (!this.calibrationData || !this.calibrationData.isValid) {
        throw new Error('No valid calibration data to save');
      }

      // In a real implementation, this would save to backend/local storage
      // For now, just complete the calibration process

      runInAction(() => {
        this.isCalibrating = false;
        this.calibrationStep = 'idle';
      });

      return {
        success: true,
        data: { saved: true },
        timestamp: new Date()
      };
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Failed to save calibration',
        timestamp: new Date()
      };
    }
  }

  cancelCalibration(): void {
    runInAction(() => {
      this.isCalibrating = false;
      this.calibrationStep = 'idle';
      this.markedCorners = [];
      this.validationResults = null;
    });
  }

  // Utility methods
  transformPoint(screenPoint: Point2D): Point2D | null {
    if (!this.calibrationData?.isValid) {
      return null;
    }

    // In a real implementation, this would apply the homography transformation
    // to convert screen coordinates to table coordinates
    // For now, return a mock transformation
    return {
      x: screenPoint.x * 2.54, // Convert to table coordinates (mock)
      y: screenPoint.y * 2.54
    };
  }

  getDetectionConfidence(): number {
    return this.currentFrame?.confidence || 0;
  }

  getBallDetectionCount(): number {
    return this.detectedBalls.length;
  }

  resetStatistics(): void {
    runInAction(() => {
      this.frameCount = 0;
      this.averageProcessingTime = 0;
      this.detectionAccuracy = 0;
      this.lastFrameTimestamp = null;
    });
  }

  // Cleanup
  destroy(): void {
    this.stopStream();
    this.stopDetection();
    this.cancelCalibration();
  }
}
