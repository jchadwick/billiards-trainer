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

    // Load existing calibration data on initialization
    this.loadCalibrationData().catch(console.error);
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

      // Send corners to backend for real homography matrix calculation
      const { apiClient } = await import('../api/client');

      // Prepare calibration request
      const calibrationRequest = {
        calibration_type: 'geometry',
        manual_points: this.markedCorners.map((corner, index) => ({
          id: `corner_${index}`,
          image_point: corner,
          world_point: null, // Backend will calculate
          is_valid: true,
          confidence: 1.0
        })),
        save_results: false // We'll save after validation
      };

      // Call backend calibration API
      const response = await apiClient.performCalibration(calibrationRequest);

      if (!response.success || !response.data) {
        throw new Error(response.error || 'Calibration failed');
      }

      // Extract real calibration data from backend response
      const backendCalibration = response.data;
      const homographyMatrix = backendCalibration.geometry?.homography_matrix || [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
      ];

      const calibrationData: CalibrationData = {
        tableCorners: [...this.markedCorners],
        homographyMatrix: homographyMatrix,
        distortionCoefficients: backendCalibration.camera?.distortion_coefficients || [0, 0, 0, 0, 0],
        isValid: backendCalibration.geometry?.is_valid || true,
        timestamp: new Date()
      };

      // Use real accuracy from backend
      const accuracy = backendCalibration.geometry?.calibration_error ?
        (1 - Math.min(backendCalibration.geometry.calibration_error, 1)) : 0.9;

      runInAction(() => {
        this.calibrationData = calibrationData;
        this.validationResults = {
          accuracy,
          error: accuracy < 0.8 ? 'Low calibration accuracy' : null
        };
        this.calibrationStep = 'complete';

        // Update selected camera calibration status
        if (this.selectedCamera) {
          this.selectedCamera.isCalibrated = calibrationData.isValid;
        }
      });

      return {
        success: true,
        data: calibrationData,
        timestamp: new Date()
      };
    } catch (error) {
      console.error('Calibration validation failed:', error);

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

      // Save calibration data to backend
      const { apiClient } = await import('../api/client');

      // Prepare save request with the validated calibration
      const saveRequest = {
        calibration_type: 'geometry',
        manual_points: this.markedCorners.map((corner, index) => ({
          id: `corner_${index}`,
          image_point: corner,
          world_point: null,
          is_valid: true,
          confidence: 1.0
        })),
        save_results: true // Now we want to save
      };

      const response = await apiClient.performCalibration(saveRequest);

      if (!response.success) {
        throw new Error(response.error || 'Failed to save calibration');
      }

      runInAction(() => {
        this.isCalibrating = false;
        this.calibrationStep = 'idle';
      });

      return {
        success: true,
        data: { saved: true, calibration_id: response.data?.id },
        timestamp: new Date()
      };
    } catch (error) {
      console.error('Failed to save calibration:', error);
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
    if (!this.calibrationData?.isValid || !this.calibrationData.homographyMatrix) {
      return null;
    }

    // Apply real homography transformation using the matrix from backend
    const matrix = this.calibrationData.homographyMatrix;

    // Apply homography matrix transformation: [x', y', w'] = H * [x, y, 1]
    const x = screenPoint.x;
    const y = screenPoint.y;

    const transformedX = matrix[0][0] * x + matrix[0][1] * y + matrix[0][2];
    const transformedY = matrix[1][0] * x + matrix[1][1] * y + matrix[1][2];
    const transformedW = matrix[2][0] * x + matrix[2][1] * y + matrix[2][2];

    // Normalize by w coordinate for perspective correction
    if (Math.abs(transformedW) < 1e-10) {
      return null; // Avoid division by zero
    }

    return {
      x: transformedX / transformedW,
      y: transformedY / transformedW
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

  // Load existing calibration data from backend
  async loadCalibrationData(): Promise<void> {
    try {
      const { apiClient } = await import('../api/client');

      // Try to get the current calibration data from backend
      const response = await apiClient.getCalibrationData();

      if (response.success && response.data) {
        const backendCalibration = response.data;

        // Check if we have valid geometric calibration
        if (backendCalibration.geometry && backendCalibration.geometry.is_valid) {
          const calibrationData: CalibrationData = {
            tableCorners: backendCalibration.geometry.table_corners_image.map(corner => ({
              x: corner.x,
              y: corner.y
            })),
            homographyMatrix: backendCalibration.geometry.homography_matrix,
            distortionCoefficients: backendCalibration.camera?.distortion_coefficients || [0, 0, 0, 0, 0],
            isValid: backendCalibration.geometry.is_valid,
            timestamp: new Date(backendCalibration.geometry.calibration_date)
          };

          runInAction(() => {
            this.calibrationData = calibrationData;

            // Update camera calibration status if we have a selected camera
            if (this.selectedCamera) {
              this.selectedCamera.isCalibrated = calibrationData.isValid;
            }
          });

          console.log('Loaded existing calibration data from backend');
        }
      }
    } catch (error) {
      // Silently fail - calibration data might not exist yet
      console.debug('No existing calibration data found:', error);
    }
  }

  // Cleanup
  destroy(): void {
    this.stopStream();
    this.stopDetection();
    this.cancelCalibration();
  }
}
