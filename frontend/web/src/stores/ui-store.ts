/**
 * MobX store for UI state management
 */

import { makeAutoObservable, observable } from 'mobx';
import type { RootStore } from './index';

export interface NotificationOptions {
  type?: 'info' | 'success' | 'warning' | 'error';
  title?: string;
  duration?: number; // in milliseconds, 0 means persistent
  actions?: Array<{
    label: string;
    action: () => void;
    style?: 'primary' | 'secondary' | 'danger';
  }>;
}

export interface Notification {
  id: string;
  type: 'info' | 'success' | 'warning' | 'error';
  title?: string;
  message: string;
  timestamp: Date;
  duration: number;
  actions: Array<{
    label: string;
    action: () => void;
    style: 'primary' | 'secondary' | 'danger';
  }>;
}

export interface Modal {
  id: string;
  component: string;
  props: Record<string, any>;
  size?: 'small' | 'medium' | 'large' | 'full';
  closable?: boolean;
  backdrop?: boolean;
}

export interface LoadingOverlay {
  id: string;
  message?: string;
  progress?: number; // 0-100
  cancellable?: boolean;
  onCancel?: () => void;
}

export interface Theme {
  name: string;
  primary: string;
  secondary: string;
  background: string;
  surface: string;
  text: string;
  textSecondary: string;
  accent: string;
  success: string;
  warning: string;
  error: string;
  info: string;
}

export type ViewMode = 'desktop' | 'tablet' | 'mobile';
export type Layout = 'standard' | 'compact' | 'fullscreen';

export class UIStore {
  private rootStore: RootStore;

  // Navigation and layout
  currentPath = '/';
  isNavigationOpen = true;
  layout: Layout = 'standard';
  viewMode: ViewMode = 'desktop';

  // Notifications
  notifications = observable.array<Notification>([]);
  maxNotifications = 5;

  // Modals and overlays
  modals = observable.array<Modal>([]);
  loadingOverlays = observable.array<LoadingOverlay>([]);

  // Theme and appearance
  theme: Theme = {
    name: 'default',
    primary: '#3b82f6',
    secondary: '#64748b',
    background: '#ffffff',
    surface: '#f8fafc',
    text: '#1e293b',
    textSecondary: '#64748b',
    accent: '#06b6d4',
    success: '#10b981',
    warning: '#f59e0b',
    error: '#ef4444',
    info: '#3b82f6',
  };

  isDarkMode = false;
  isHighContrastMode = false;
  fontSize = 'medium'; // 'small' | 'medium' | 'large'

  // Sidebar and panels
  isSidebarCollapsed = false;
  rightPanelOpen = false;
  rightPanelContent: string | null = null;

  // Page-specific UI state
  gameViewSettings = {
    showTrajectory: true,
    showBallLabels: true,
    showTableOverlay: true,
    showMetrics: false,
    overlayOpacity: 0.7,
    ballTrailLength: 10,
  };

  configViewSettings = {
    showAdvanced: false,
    groupByCategory: true,
    showValidationErrors: true,
  };

  calibrationViewSettings = {
    showInstructions: true,
    showAccuracyMeter: true,
    gridOverlay: false,
    autoAdvance: false,
  };

  // Window and viewport
  windowSize = { width: 1920, height: 1080 };
  isFullscreen = false;
  keyboardShortcutsEnabled = true;

  // Performance and debugging
  showDebugInfo = false;
  showPerformanceMetrics = false;
  enableAnimations = true;
  reduceMotion = false;

  constructor(rootStore: RootStore) {
    makeAutoObservable(this, {}, { autoBind: true });
    this.rootStore = rootStore;
    this.initializeFromStorage();
    this.setupWindowListeners();
  }

  // =============================================================================
  // Navigation and Layout
  // =============================================================================

  setCurrentPath(path: string): void {
    this.currentPath = path;
  }

  toggleNavigation(): void {
    this.isNavigationOpen = !this.isNavigationOpen;
    this.saveToStorage();
  }

  setLayout(layout: Layout): void {
    this.layout = layout;
    this.saveToStorage();
  }

  setViewMode(mode: ViewMode): void {
    this.viewMode = mode;
  }

  // =============================================================================
  // Notifications
  // =============================================================================

  showNotification(message: string, options: NotificationOptions = {}): string {
    const id = `notification_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;

    const notification: Notification = {
      id,
      type: options.type || 'info',
      title: options.title,
      message,
      timestamp: new Date(),
      duration: options.duration || 5000,
      actions: options.actions || [],
    };

    // Add notification
    this.notifications.push(notification);

    // Remove oldest if we exceed max
    if (this.notifications.length > this.maxNotifications) {
      this.notifications.shift();
    }

    // Auto-remove after duration (if not persistent)
    if (notification.duration > 0) {
      setTimeout(() => {
        this.removeNotification(id);
      }, notification.duration);
    }

    return id;
  }

  removeNotification(id: string): void {
    const index = this.notifications.findIndex(n => n.id === id);
    if (index !== -1) {
      this.notifications.splice(index, 1);
    }
  }

  clearAllNotifications(): void {
    this.notifications.clear();
  }

  // Convenience methods for different notification types
  showSuccess(message: string, title?: string): string {
    return this.showNotification(message, { type: 'success', title });
  }

  showError(message: string, title?: string): string {
    return this.showNotification(message, {
      type: 'error',
      title,
      duration: 0 // Errors are persistent by default
    });
  }

  showWarning(message: string, title?: string): string {
    return this.showNotification(message, { type: 'warning', title });
  }

  showInfo(message: string, title?: string): string {
    return this.showNotification(message, { type: 'info', title });
  }

  // =============================================================================
  // Modals
  // =============================================================================

  openModal(component: string, props: Record<string, any> = {}, options: {
    size?: Modal['size'];
    closable?: boolean;
    backdrop?: boolean;
  } = {}): string {
    const id = `modal_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;

    const modal: Modal = {
      id,
      component,
      props,
      size: options.size || 'medium',
      closable: options.closable !== false,
      backdrop: options.backdrop !== false,
    };

    this.modals.push(modal);
    return id;
  }

  closeModal(id: string): void {
    const index = this.modals.findIndex(m => m.id === id);
    if (index !== -1) {
      this.modals.splice(index, 1);
    }
  }

  closeAllModals(): void {
    this.modals.clear();
  }

  // =============================================================================
  // Loading Overlays
  // =============================================================================

  showLoading(message?: string, options: {
    progress?: number;
    cancellable?: boolean;
    onCancel?: () => void;
  } = {}): string {
    const id = `loading_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;

    const overlay: LoadingOverlay = {
      id,
      message,
      progress: options.progress,
      cancellable: options.cancellable || false,
      onCancel: options.onCancel,
    };

    this.loadingOverlays.push(overlay);
    return id;
  }

  updateLoading(id: string, updates: Partial<LoadingOverlay>): void {
    const overlay = this.loadingOverlays.find(o => o.id === id);
    if (overlay) {
      Object.assign(overlay, updates);
    }
  }

  hideLoading(id: string): void {
    const index = this.loadingOverlays.findIndex(o => o.id === id);
    if (index !== -1) {
      this.loadingOverlays.splice(index, 1);
    }
  }

  hideAllLoading(): void {
    this.loadingOverlays.clear();
  }

  // =============================================================================
  // Theme and Appearance
  // =============================================================================

  setTheme(theme: Partial<Theme>): void {
    this.theme = { ...this.theme, ...theme };
    this.applyTheme();
    this.saveToStorage();
  }

  toggleDarkMode(): void {
    this.isDarkMode = !this.isDarkMode;
    this.applyTheme();
    this.saveToStorage();
  }

  setHighContrastMode(enabled: boolean): void {
    this.isHighContrastMode = enabled;
    this.applyTheme();
    this.saveToStorage();
  }

  setFontSize(size: 'small' | 'medium' | 'large'): void {
    this.fontSize = size;
    this.applyFontSize();
    this.saveToStorage();
  }

  private applyTheme(): void {
    const root = document.documentElement;
    const theme = this.isDarkMode ? this.getDarkTheme() : this.theme;

    Object.entries(theme).forEach(([key, value]) => {
      if (key !== 'name') {
        root.style.setProperty(`--color-${key}`, value);
      }
    });

    root.setAttribute('data-theme', this.isDarkMode ? 'dark' : 'light');
    root.setAttribute('data-high-contrast', this.isHighContrastMode.toString());
  }

  private applyFontSize(): void {
    const root = document.documentElement;
    const sizeMap = {
      small: '14px',
      medium: '16px',
      large: '18px',
    };

    root.style.setProperty('--font-size-base', sizeMap[this.fontSize]);
  }

  private getDarkTheme(): Theme {
    return {
      ...this.theme,
      name: 'dark',
      background: '#0f172a',
      surface: '#1e293b',
      text: '#f1f5f9',
      textSecondary: '#94a3b8',
    };
  }

  // =============================================================================
  // Sidebar and Panels
  // =============================================================================

  toggleSidebar(): void {
    this.isSidebarCollapsed = !this.isSidebarCollapsed;
    this.saveToStorage();
  }

  openRightPanel(content: string): void {
    this.rightPanelContent = content;
    this.rightPanelOpen = true;
  }

  closeRightPanel(): void {
    this.rightPanelOpen = false;
    this.rightPanelContent = null;
  }

  toggleRightPanel(content?: string): void {
    if (this.rightPanelOpen && (!content || this.rightPanelContent === content)) {
      this.closeRightPanel();
    } else {
      this.openRightPanel(content || 'default');
    }
  }

  // =============================================================================
  // View-Specific Settings
  // =============================================================================

  updateGameViewSettings(updates: Partial<typeof this.gameViewSettings>): void {
    this.gameViewSettings = { ...this.gameViewSettings, ...updates };
    this.saveToStorage();
  }

  updateConfigViewSettings(updates: Partial<typeof this.configViewSettings>): void {
    this.configViewSettings = { ...this.configViewSettings, ...updates };
    this.saveToStorage();
  }

  updateCalibrationViewSettings(updates: Partial<typeof this.calibrationViewSettings>): void {
    this.calibrationViewSettings = { ...this.calibrationViewSettings, ...updates };
    this.saveToStorage();
  }

  // =============================================================================
  // Window and Fullscreen
  // =============================================================================

  toggleFullscreen(): void {
    if (!document.fullscreenElement) {
      document.documentElement.requestFullscreen();
    } else {
      document.exitFullscreen();
    }
  }

  private setupWindowListeners(): void {
    // Window resize listener
    const handleResize = () => {
      this.windowSize = {
        width: window.innerWidth,
        height: window.innerHeight,
      };

      // Update view mode based on window size
      if (window.innerWidth < 768) {
        this.viewMode = 'mobile';
      } else if (window.innerWidth < 1024) {
        this.viewMode = 'tablet';
      } else {
        this.viewMode = 'desktop';
      }
    };

    window.addEventListener('resize', handleResize);
    handleResize(); // Initial call

    // Fullscreen change listener
    const handleFullscreenChange = () => {
      this.isFullscreen = !!document.fullscreenElement;
    };

    document.addEventListener('fullscreenchange', handleFullscreenChange);

    // Keyboard listeners
    if (this.keyboardShortcutsEnabled) {
      this.setupKeyboardShortcuts();
    }

    // Reduced motion preference
    const mediaQuery = window.matchMedia('(prefers-reduced-motion: reduce)');
    this.reduceMotion = mediaQuery.matches;
    this.enableAnimations = !this.reduceMotion;

    mediaQuery.addEventListener('change', (e) => {
      this.reduceMotion = e.matches;
      this.enableAnimations = !e.matches;
    });
  }

  private setupKeyboardShortcuts(): void {
    const handleKeyboard = (event: KeyboardEvent) => {
      // Only handle shortcuts when not in input elements
      if (event.target instanceof HTMLInputElement ||
          event.target instanceof HTMLTextAreaElement ||
          event.target instanceof HTMLSelectElement) {
        return;
      }

      const { key, ctrlKey, metaKey, shiftKey } = event;
      const cmdOrCtrl = ctrlKey || metaKey;

      // Toggle navigation: Ctrl/Cmd + B
      if (cmdOrCtrl && key === 'b') {
        event.preventDefault();
        this.toggleNavigation();
      }

      // Toggle fullscreen: F11
      if (key === 'F11') {
        event.preventDefault();
        this.toggleFullscreen();
      }

      // Toggle dark mode: Ctrl/Cmd + Shift + D
      if (cmdOrCtrl && shiftKey && key === 'D') {
        event.preventDefault();
        this.toggleDarkMode();
      }

      // Close modals: Escape
      if (key === 'Escape') {
        if (this.modals.length > 0) {
          event.preventDefault();
          this.closeModal(this.modals[this.modals.length - 1].id);
        }
      }
    };

    document.addEventListener('keydown', handleKeyboard);
  }

  // =============================================================================
  // Performance and Debugging
  // =============================================================================

  toggleDebugInfo(): void {
    this.showDebugInfo = !this.showDebugInfo;
  }

  togglePerformanceMetrics(): void {
    this.showPerformanceMetrics = !this.showPerformanceMetrics;
  }

  setAnimationsEnabled(enabled: boolean): void {
    this.enableAnimations = enabled && !this.reduceMotion;
    this.saveToStorage();
  }

  // =============================================================================
  // Computed Properties
  // =============================================================================

  get isMobile(): boolean {
    return this.viewMode === 'mobile';
  }

  get isTablet(): boolean {
    return this.viewMode === 'tablet';
  }

  get isDesktop(): boolean {
    return this.viewMode === 'desktop';
  }

  get hasActiveModals(): boolean {
    return this.modals.length > 0;
  }

  get hasLoadingOverlays(): boolean {
    return this.loadingOverlays.length > 0;
  }

  get unreadNotifications(): number {
    return this.notifications.length;
  }

  get shouldShowSidebar(): boolean {
    return this.isDesktop && this.isNavigationOpen;
  }

  get contentClassName(): string {
    const classes = ['content'];

    if (this.isSidebarCollapsed) classes.push('sidebar-collapsed');
    if (this.rightPanelOpen) classes.push('right-panel-open');
    if (this.layout === 'fullscreen') classes.push('fullscreen');
    if (this.layout === 'compact') classes.push('compact');
    if (!this.enableAnimations) classes.push('no-animations');

    return classes.join(' ');
  }

  // =============================================================================
  // Persistence
  // =============================================================================

  private initializeFromStorage(): void {
    try {
      const stored = localStorage.getItem('ui_settings');
      if (stored) {
        const settings = JSON.parse(stored);

        // Restore theme settings
        if (settings.isDarkMode !== undefined) {
          this.isDarkMode = settings.isDarkMode;
        }
        if (settings.isHighContrastMode !== undefined) {
          this.isHighContrastMode = settings.isHighContrastMode;
        }
        if (settings.fontSize) {
          this.fontSize = settings.fontSize;
        }

        // Restore layout settings
        if (settings.isNavigationOpen !== undefined) {
          this.isNavigationOpen = settings.isNavigationOpen;
        }
        if (settings.isSidebarCollapsed !== undefined) {
          this.isSidebarCollapsed = settings.isSidebarCollapsed;
        }
        if (settings.layout) {
          this.layout = settings.layout;
        }

        // Restore view settings
        if (settings.gameViewSettings) {
          this.gameViewSettings = { ...this.gameViewSettings, ...settings.gameViewSettings };
        }
        if (settings.configViewSettings) {
          this.configViewSettings = { ...this.configViewSettings, ...settings.configViewSettings };
        }
        if (settings.calibrationViewSettings) {
          this.calibrationViewSettings = { ...this.calibrationViewSettings, ...settings.calibrationViewSettings };
        }

        // Restore preferences
        if (settings.enableAnimations !== undefined) {
          this.enableAnimations = settings.enableAnimations;
        }
        if (settings.keyboardShortcutsEnabled !== undefined) {
          this.keyboardShortcutsEnabled = settings.keyboardShortcutsEnabled;
        }
      }

      // Apply initial theme
      setTimeout(() => {
        this.applyTheme();
        this.applyFontSize();
      }, 0);

    } catch (error) {
      console.warn('Failed to load UI settings from storage:', error);
    }
  }

  private saveToStorage(): void {
    try {
      const settings = {
        isDarkMode: this.isDarkMode,
        isHighContrastMode: this.isHighContrastMode,
        fontSize: this.fontSize,
        isNavigationOpen: this.isNavigationOpen,
        isSidebarCollapsed: this.isSidebarCollapsed,
        layout: this.layout,
        gameViewSettings: this.gameViewSettings,
        configViewSettings: this.configViewSettings,
        calibrationViewSettings: this.calibrationViewSettings,
        enableAnimations: this.enableAnimations,
        keyboardShortcutsEnabled: this.keyboardShortcutsEnabled,
      };

      localStorage.setItem('ui_settings', JSON.stringify(settings));
    } catch (error) {
      console.warn('Failed to save UI settings to storage:', error);
    }
  }

  // =============================================================================
  // Store Lifecycle
  // =============================================================================

  reset(): void {
    this.currentPath = '/';
    this.isNavigationOpen = true;
    this.layout = 'standard';
    this.notifications.clear();
    this.modals.clear();
    this.loadingOverlays.clear();
    this.isDarkMode = false;
    this.isHighContrastMode = false;
    this.fontSize = 'medium';
    this.isSidebarCollapsed = false;
    this.rightPanelOpen = false;
    this.rightPanelContent = null;
    this.isFullscreen = false;
    this.showDebugInfo = false;
    this.showPerformanceMetrics = false;
    this.enableAnimations = true;
    this.keyboardShortcutsEnabled = true;

    // Reset view settings to defaults
    this.gameViewSettings = {
      showTrajectory: true,
      showBallLabels: true,
      showTableOverlay: true,
      showMetrics: false,
      overlayOpacity: 0.7,
      ballTrailLength: 10,
    };

    this.configViewSettings = {
      showAdvanced: false,
      groupByCategory: true,
      showValidationErrors: true,
    };

    this.calibrationViewSettings = {
      showInstructions: true,
      showAccuracyMeter: true,
      gridOverlay: false,
      autoAdvance: false,
    };
  }

  destroy(): void {
    this.reset();
  }
}
