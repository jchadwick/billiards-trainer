import { makeAutoObservable, runInAction } from 'mobx';
import type {
  AppConfig,
  CameraConfig,
  DetectionConfig,
  GameConfig,
  UIConfig,
  ActionResult
} from './types';

export class ConfigStore {
  // Observable state
  config: AppConfig = {
    camera: {
      selectedCameraId: null,
      resolution: { width: 1920, height: 1080 },
      fps: 30,
      autoExposure: true,
      exposure: 0.5,
      brightness: 0.5,
      contrast: 0.5
    },
    detection: {
      ballDetectionThreshold: 0.7,
      cueDetectionThreshold: 0.6,
      motionDetectionThreshold: 0.3,
      stabilizationFrames: 5,
      enableTracking: true,
      enablePrediction: true
    },
    game: {
      defaultGameType: 'practice',
      autoStartGames: false,
      enableFoulDetection: true,
      shotTimeout: 120,
      enableShotHistory: true,
      maxHistorySize: 1000
    },
    ui: {
      theme: 'auto',
      showDebugInfo: false,
      enableNotifications: true,
      animationSpeed: 'normal',
      language: 'en'
    }
  };

  private configProfiles: { [name: string]: AppConfig } = {};
  private activeProfile: string = 'default';
  private isDirty: boolean = false;

  constructor() {
    makeAutoObservable(this, {}, { autoBind: true });

    this.loadDefaultProfiles();
    this.loadFromBackend();
  }

  // Computed values
  get currentProfile(): string {
    return this.activeProfile;
  }

  get availableProfiles(): string[] {
    return Object.keys(this.configProfiles);
  }

  get hasUnsavedChanges(): boolean {
    return this.isDirty;
  }

  get validationErrors(): string[] {
    return this.validateConfig(this.config);
  }

  get isValid(): boolean {
    return this.validationErrors.length === 0;
  }

  // Camera config getters
  get cameraConfig(): CameraConfig {
    return this.config.camera;
  }

  get selectedCameraId(): string | null {
    return this.config.camera.selectedCameraId;
  }

  get cameraResolution(): { width: number; height: number } {
    return this.config.camera.resolution;
  }

  get cameraFps(): number {
    return this.config.camera.fps;
  }

  // Detection config getters
  get detectionConfig(): DetectionConfig {
    return this.config.detection;
  }

  get ballDetectionThreshold(): number {
    return this.config.detection.ballDetectionThreshold;
  }

  get isTrackingEnabled(): boolean {
    return this.config.detection.enableTracking;
  }

  // Game config getters
  get gameConfig(): GameConfig {
    return this.config.game;
  }

  get defaultGameType(): GameConfig['defaultGameType'] {
    return this.config.game.defaultGameType;
  }

  get shotTimeout(): number {
    return this.config.game.shotTimeout;
  }

  // UI config getters
  get uiConfig(): UIConfig {
    return this.config.ui;
  }

  get theme(): UIConfig['theme'] {
    return this.config.ui.theme;
  }

  get showDebugInfo(): boolean {
    return this.config.ui.showDebugInfo;
  }

  get language(): string {
    return this.config.ui.language;
  }

  // Actions - Camera Config
  updateCameraConfig(updates: Partial<CameraConfig>): void {
    runInAction(() => {
      Object.assign(this.config.camera, updates);
      this.markDirty();
    });
  }

  setSelectedCamera(cameraId: string | null): void {
    this.updateCameraConfig({ selectedCameraId: cameraId });
  }

  setCameraResolution(width: number, height: number): void {
    this.updateCameraConfig({ resolution: { width, height } });
  }

  setCameraFps(fps: number): void {
    this.updateCameraConfig({ fps });
  }

  setCameraExposure(exposure: number, autoExposure: boolean = false): void {
    this.updateCameraConfig({ exposure, autoExposure });
  }

  setCameraBrightness(brightness: number): void {
    this.updateCameraConfig({ brightness });
  }

  setCameraContrast(contrast: number): void {
    this.updateCameraConfig({ contrast });
  }

  // Actions - Detection Config
  updateDetectionConfig(updates: Partial<DetectionConfig>): void {
    runInAction(() => {
      Object.assign(this.config.detection, updates);
      this.markDirty();
    });
  }

  setBallDetectionThreshold(threshold: number): void {
    this.updateDetectionConfig({ ballDetectionThreshold: threshold });
  }

  setCueDetectionThreshold(threshold: number): void {
    this.updateDetectionConfig({ cueDetectionThreshold: threshold });
  }

  setMotionDetectionThreshold(threshold: number): void {
    this.updateDetectionConfig({ motionDetectionThreshold: threshold });
  }

  setStabilizationFrames(frames: number): void {
    this.updateDetectionConfig({ stabilizationFrames: frames });
  }

  setTrackingEnabled(enabled: boolean): void {
    this.updateDetectionConfig({ enableTracking: enabled });
  }

  setPredictionEnabled(enabled: boolean): void {
    this.updateDetectionConfig({ enablePrediction: enabled });
  }

  // Actions - Game Config
  updateGameConfig(updates: Partial<GameConfig>): void {
    runInAction(() => {
      Object.assign(this.config.game, updates);
      this.markDirty();
    });
  }

  setDefaultGameType(gameType: GameConfig['defaultGameType']): void {
    this.updateGameConfig({ defaultGameType: gameType });
  }

  setAutoStartGames(autoStart: boolean): void {
    this.updateGameConfig({ autoStartGames: autoStart });
  }

  setFoulDetectionEnabled(enabled: boolean): void {
    this.updateGameConfig({ enableFoulDetection: enabled });
  }

  setShotTimeout(timeout: number): void {
    this.updateGameConfig({ shotTimeout: timeout });
  }

  setShotHistoryEnabled(enabled: boolean): void {
    this.updateGameConfig({ enableShotHistory: enabled });
  }

  setMaxHistorySize(size: number): void {
    this.updateGameConfig({ maxHistorySize: size });
  }

  // Actions - UI Config
  updateUIConfig(updates: Partial<UIConfig>): void {
    runInAction(() => {
      Object.assign(this.config.ui, updates);
      this.markDirty();
    });
  }

  setTheme(theme: UIConfig['theme']): void {
    this.updateUIConfig({ theme });
  }

  setShowDebugInfo(show: boolean): void {
    this.updateUIConfig({ showDebugInfo: show });
  }

  setNotificationsEnabled(enabled: boolean): void {
    this.updateUIConfig({ enableNotifications: enabled });
  }

  setAnimationSpeed(speed: UIConfig['animationSpeed']): void {
    this.updateUIConfig({ animationSpeed: speed });
  }

  setLanguage(language: string): void {
    this.updateUIConfig({ language });
  }

  // Profile management (client-side only, stored in browser)
  async saveProfile(name: string): Promise<ActionResult> {
    try {
      if (!name.trim()) {
        throw new Error('Profile name cannot be empty');
      }

      if (!this.isValid) {
        throw new Error('Cannot save invalid configuration');
      }

      const profileConfig = JSON.parse(JSON.stringify(this.config));

      runInAction(() => {
        this.configProfiles[name] = profileConfig;
        this.isDirty = false;
      });

      // Save to localStorage for persistence across sessions
      try {
        localStorage.setItem('billiards-config-profiles', JSON.stringify(this.configProfiles));
      } catch (error) {
        console.warn('Failed to save profiles to localStorage:', error);
      }

      return {
        success: true,
        data: { profileName: name },
        timestamp: new Date()
      };
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Failed to save profile',
        timestamp: new Date()
      };
    }
  }

  async loadProfile(name: string): Promise<ActionResult> {
    try {
      const profile = this.configProfiles[name];
      if (!profile) {
        throw new Error(`Profile '${name}' not found`);
      }

      runInAction(() => {
        this.config = JSON.parse(JSON.stringify(profile));
        this.activeProfile = name;
        this.isDirty = false;
      });

      return {
        success: true,
        data: { profileName: name },
        timestamp: new Date()
      };
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Failed to load profile',
        timestamp: new Date()
      };
    }
  }

  async deleteProfile(name: string): Promise<ActionResult> {
    try {
      if (name === 'default') {
        throw new Error('Cannot delete default profile');
      }

      if (!this.configProfiles[name]) {
        throw new Error(`Profile '${name}' not found`);
      }

      runInAction(() => {
        delete this.configProfiles[name];

        // If we deleted the active profile, switch to default
        if (this.activeProfile === name) {
          this.activeProfile = 'default';
          this.config = JSON.parse(JSON.stringify(this.configProfiles.default));
          this.isDirty = false;
        }
      });

      // Update localStorage
      try {
        localStorage.setItem('billiards-config-profiles', JSON.stringify(this.configProfiles));
      } catch (error) {
        console.warn('Failed to update profiles in localStorage:', error);
      }

      return {
        success: true,
        data: { deletedProfile: name },
        timestamp: new Date()
      };
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Failed to delete profile',
        timestamp: new Date()
      };
    }
  }

  // Config validation and reset
  async resetToDefaults(): Promise<ActionResult> {
    try {
      const defaultConfig = this.createDefaultConfig();

      runInAction(() => {
        this.config = defaultConfig;
        this.activeProfile = 'default';
        this.isDirty = true;
      });

      return {
        success: true,
        timestamp: new Date()
      };
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Failed to reset to defaults',
        timestamp: new Date()
      };
    }
  }

  async importConfig(configData: Partial<AppConfig>): Promise<ActionResult> {
    try {
      const mergedConfig = { ...this.config, ...configData };
      const errors = this.validateConfig(mergedConfig);

      if (errors.length > 0) {
        throw new Error(`Invalid configuration: ${errors.join(', ')}`);
      }

      runInAction(() => {
        this.config = mergedConfig;
        this.markDirty();
      });

      return {
        success: true,
        data: { importedKeys: Object.keys(configData) },
        timestamp: new Date()
      };
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Failed to import configuration',
        timestamp: new Date()
      };
    }
  }

  exportConfig(): AppConfig {
    return JSON.parse(JSON.stringify(this.config));
  }

  // Private methods
  private markDirty(): void {
    this.isDirty = true;
  }

  private createDefaultConfig(): AppConfig {
    return {
      camera: {
        selectedCameraId: null,
        resolution: { width: 1920, height: 1080 },
        fps: 30,
        autoExposure: true,
        exposure: 0.5,
        brightness: 0.5,
        contrast: 0.5
      },
      detection: {
        ballDetectionThreshold: 0.7,
        cueDetectionThreshold: 0.6,
        motionDetectionThreshold: 0.3,
        stabilizationFrames: 5,
        enableTracking: true,
        enablePrediction: true
      },
      game: {
        defaultGameType: 'practice',
        autoStartGames: false,
        enableFoulDetection: true,
        shotTimeout: 120,
        enableShotHistory: true,
        maxHistorySize: 1000
      },
      ui: {
        theme: 'auto',
        showDebugInfo: false,
        enableNotifications: true,
        animationSpeed: 'normal',
        language: 'en'
      }
    };
  }

  private loadDefaultProfiles(): void {
    const defaultConfig = this.createDefaultConfig();

    const performanceConfig: AppConfig = {
      ...defaultConfig,
      camera: {
        ...defaultConfig.camera,
        resolution: { width: 1280, height: 720 },
        fps: 60
      },
      detection: {
        ...defaultConfig.detection,
        ballDetectionThreshold: 0.6,
        stabilizationFrames: 3,
        enablePrediction: false
      }
    };

    const qualityConfig: AppConfig = {
      ...defaultConfig,
      camera: {
        ...defaultConfig.camera,
        resolution: { width: 1920, height: 1080 },
        fps: 30
      },
      detection: {
        ...defaultConfig.detection,
        ballDetectionThreshold: 0.8,
        stabilizationFrames: 10,
        enableTracking: true,
        enablePrediction: true
      }
    };

    // Start with built-in profiles
    this.configProfiles = {
      default: defaultConfig,
      performance: performanceConfig,
      quality: qualityConfig
    };

    // Load saved profiles from localStorage
    try {
      const savedProfiles = localStorage.getItem('billiards-config-profiles');
      if (savedProfiles) {
        const parsed = JSON.parse(savedProfiles);
        // Merge with defaults, allowing localStorage to override
        this.configProfiles = { ...this.configProfiles, ...parsed };
      }
    } catch (error) {
      console.warn('Failed to load profiles from localStorage:', error);
    }
  }

  private validateConfig(config: AppConfig): string[] {
    const errors: string[] = [];

    // Camera validation
    if (config.camera.fps <= 0 || config.camera.fps > 120) {
      errors.push('Camera FPS must be between 1 and 120');
    }

    if (config.camera.resolution.width <= 0 || config.camera.resolution.height <= 0) {
      errors.push('Camera resolution must be positive');
    }

    if (config.camera.exposure < 0 || config.camera.exposure > 1) {
      errors.push('Camera exposure must be between 0 and 1');
    }

    if (config.camera.brightness < 0 || config.camera.brightness > 1) {
      errors.push('Camera brightness must be between 0 and 1');
    }

    if (config.camera.contrast < 0 || config.camera.contrast > 1) {
      errors.push('Camera contrast must be between 0 and 1');
    }

    // Detection validation
    if (config.detection.ballDetectionThreshold < 0 || config.detection.ballDetectionThreshold > 1) {
      errors.push('Ball detection threshold must be between 0 and 1');
    }

    if (config.detection.cueDetectionThreshold < 0 || config.detection.cueDetectionThreshold > 1) {
      errors.push('Cue detection threshold must be between 0 and 1');
    }

    if (config.detection.motionDetectionThreshold < 0 || config.detection.motionDetectionThreshold > 1) {
      errors.push('Motion detection threshold must be between 0 and 1');
    }

    if (config.detection.stabilizationFrames < 1 || config.detection.stabilizationFrames > 60) {
      errors.push('Stabilization frames must be between 1 and 60');
    }

    // Game validation
    if (config.game.shotTimeout <= 0 || config.game.shotTimeout > 3600) {
      errors.push('Shot timeout must be between 1 and 3600 seconds');
    }

    if (config.game.maxHistorySize <= 0 || config.game.maxHistorySize > 10000) {
      errors.push('Max history size must be between 1 and 10000');
    }

    return errors;
  }

  // Backend integration methods
  async loadFromBackend(): Promise<void> {
    try {
      const { apiClient } = await import('../api/client');
      const response = await apiClient.getConfiguration();

      if (response.success && response.data) {
        runInAction(() => {
          // Merge backend config with local config
          if (response.data.values) {
            // Update individual sections if they exist in backend
            if (response.data.values.camera) {
              Object.assign(this.config.camera, response.data.values.camera);
            }
            if (response.data.values.detection) {
              Object.assign(this.config.detection, response.data.values.detection);
            }
            if (response.data.values.game) {
              Object.assign(this.config.game, response.data.values.game);
            }
            if (response.data.values.ui) {
              Object.assign(this.config.ui, response.data.values.ui);
            }
            // Note: Profiles are now managed client-side only, not synced from backend
          }
          this.isDirty = false;
        });
      }
    } catch (error) {
      console.warn('Failed to load configuration from backend:', error);
    }
  }

  async saveToBackend(): Promise<ActionResult> {
    try {
      if (!this.isValid) {
        throw new Error('Cannot save invalid configuration');
      }

      const { apiClient } = await import('../api/client');
      const configData = {
        camera: this.config.camera,
        detection: this.config.detection,
        game: this.config.game,
        ui: this.config.ui
        // Note: Profiles are client-side only, not synced to backend
      };

      const response = await apiClient.updateConfiguration(configData);

      if (response.success) {
        runInAction(() => {
          this.isDirty = false;
        });

        return {
          success: true,
          data: response.data,
          timestamp: new Date()
        };
      } else {
        throw new Error(response.error || 'Failed to save configuration');
      }
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Failed to save configuration to backend',
        timestamp: new Date()
      };
    }
  }

  async resetToBackendDefaults(): Promise<ActionResult> {
    try {
      const { apiClient } = await import('../api/client');
      const response = await apiClient.resetConfiguration(true, true, 'all');

      if (response.success) {
        // Reload configuration from backend
        await this.loadFromBackend();

        return {
          success: true,
          data: response.data,
          timestamp: new Date()
        };
      } else {
        throw new Error(response.error || 'Failed to reset configuration');
      }
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Failed to reset configuration on backend',
        timestamp: new Date()
      };
    }
  }

  async exportToFile(format: 'json' | 'yaml' = 'json'): Promise<ActionResult> {
    try {
      const { apiClient } = await import('../api/client');
      const response = await apiClient.exportConfiguration(format);

      if (response.success && response.data) {
        // Create and trigger download
        const dataStr = format === 'json'
          ? JSON.stringify(response.data.data, null, 2)
          : JSON.stringify(response.data.data); // Would need yaml library for proper YAML export

        const dataBlob = new Blob([dataStr], { type: `application/${format}` });
        const url = URL.createObjectURL(dataBlob);

        const link = document.createElement('a');
        link.href = url;
        link.download = `billiards-config-${new Date().toISOString().split('T')[0]}.${format}`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);

        return {
          success: true,
          data: { format, size: response.data.size },
          timestamp: new Date()
        };
      } else {
        throw new Error(response.error || 'Failed to export configuration');
      }
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Failed to export configuration',
        timestamp: new Date()
      };
    }
  }

  async importFromFile(file: File, mergeStrategy: 'replace' | 'merge' = 'merge'): Promise<ActionResult> {
    try {
      const { apiClient } = await import('../api/client');
      const response = await apiClient.importConfiguration(file, mergeStrategy);

      if (response.success) {
        // Reload configuration from backend to get the updated values
        await this.loadFromBackend();

        return {
          success: true,
          data: response.data,
          timestamp: new Date()
        };
      } else {
        throw new Error(response.error || 'Failed to import configuration');
      }
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Failed to import configuration',
        timestamp: new Date()
      };
    }
  }
}
