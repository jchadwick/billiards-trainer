import { makeAutoObservable, runInAction } from 'mobx';
import type {
  UIState,
  Notification,
  ActionResult
} from './types';

export class UIStore {
  // Observable state
  uiState: UIState = {
    modals: {
      calibration: false,
      settings: false,
      gameSetup: false,
      shotHistory: false,
      help: false
    },
    notifications: [],
    loading: {
      global: false,
      calibration: false,
      detection: false,
      gameStart: false
    },
    activeTab: 'game',
    sidebarOpen: true
  };

  // Additional UI state
  private notificationIdCounter = 0;
  private autoHideTimers: Map<string, NodeJS.Timeout> = new Map();

  // Screen/viewport information
  windowWidth: number = typeof window !== 'undefined' ? window.innerWidth : 1920;
  windowHeight: number = typeof window !== 'undefined' ? window.innerHeight : 1080;
  isMobile: boolean = this.windowWidth < 768;
  isTablet: boolean = this.windowWidth >= 768 && this.windowWidth < 1024;
  isDesktop: boolean = this.windowWidth >= 1024;

  constructor() {
    makeAutoObservable(this, {}, { autoBind: true });

    // Listen for window resize if in browser
    if (typeof window !== 'undefined') {
      window.addEventListener('resize', this.handleWindowResize);
    }

    // Load initial preferences
    this.loadPreferences();
  }

  // Computed values
  get isAnyModalOpen(): boolean {
    return Object.values(this.uiState.modals).some(isOpen => isOpen);
  }

  get unreadNotifications(): Notification[] {
    return this.uiState.notifications.filter(n => !n.isRead);
  }

  get unreadNotificationCount(): number {
    return this.unreadNotifications.length;
  }

  get isGlobalLoading(): boolean {
    return this.uiState.loading.global;
  }

  get isAnyLoading(): boolean {
    return Object.values(this.uiState.loading).some(loading => loading);
  }

  get recentNotifications(): Notification[] {
    const oneHourAgo = new Date(Date.now() - 60 * 60 * 1000);
    return this.uiState.notifications
      .filter(n => n.timestamp > oneHourAgo)
      .sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime());
  }

  get errorNotifications(): Notification[] {
    return this.uiState.notifications.filter(n => n.type === 'error');
  }

  get shouldCollapseSidebar(): boolean {
    return this.isMobile || (this.isTablet && this.isAnyModalOpen);
  }

  // Modal actions
  openModal(modalName: keyof UIState['modals']): void {
    runInAction(() => {
      // Close other modals first (only one modal at a time)
      Object.keys(this.uiState.modals).forEach(key => {
        this.uiState.modals[key as keyof UIState['modals']] = false;
      });

      this.uiState.modals[modalName] = true;

      // Auto-collapse sidebar on mobile when modal opens
      if (this.shouldCollapseSidebar) {
        this.uiState.sidebarOpen = false;
      }
    });
  }

  closeModal(modalName: keyof UIState['modals']): void {
    runInAction(() => {
      this.uiState.modals[modalName] = false;
    });
  }

  closeAllModals(): void {
    runInAction(() => {
      Object.keys(this.uiState.modals).forEach(key => {
        this.uiState.modals[key as keyof UIState['modals']] = false;
      });
    });
  }

  toggleModal(modalName: keyof UIState['modals']): void {
    if (this.uiState.modals[modalName]) {
      this.closeModal(modalName);
    } else {
      this.openModal(modalName);
    }
  }

  // Loading actions
  setLoading(category: keyof UIState['loading'], isLoading: boolean): void {
    runInAction(() => {
      this.uiState.loading[category] = isLoading;
    });
  }

  setGlobalLoading(isLoading: boolean): void {
    this.setLoading('global', isLoading);
  }

  async withLoading<T>(
    category: keyof UIState['loading'],
    operation: () => Promise<T>
  ): Promise<T> {
    this.setLoading(category, true);
    try {
      return await operation();
    } finally {
      this.setLoading(category, false);
    }
  }

  // Notification actions
  showNotification(
    type: Notification['type'],
    title: string,
    message: string,
    options: {
      autoHide?: boolean;
      duration?: number;
    } = {}
  ): string {
    const id = `notification_${++this.notificationIdCounter}`;
    const { autoHide = true, duration = 5000 } = options;

    const notification: Notification = {
      id,
      type,
      title,
      message,
      timestamp: new Date(),
      isRead: false,
      autoHide,
      duration: autoHide ? duration : undefined
    };

    runInAction(() => {
      this.uiState.notifications.unshift(notification);

      // Keep only the most recent 50 notifications
      if (this.uiState.notifications.length > 50) {
        this.uiState.notifications = this.uiState.notifications.slice(0, 50);
      }
    });

    // Set auto-hide timer if enabled
    if (autoHide && duration) {
      const timer = setTimeout(() => {
        this.removeNotification(id);
      }, duration);
      this.autoHideTimers.set(id, timer);
    }

    return id;
  }

  showSuccess(title: string, message: string, autoHide = true): string {
    return this.showNotification('success', title, message, { autoHide });
  }

  showError(title: string, message: string, autoHide = false): string {
    return this.showNotification('error', title, message, { autoHide });
  }

  showWarning(title: string, message: string, autoHide = true): string {
    return this.showNotification('warning', title, message, { autoHide, duration: 8000 });
  }

  showInfo(title: string, message: string, autoHide = true): string {
    return this.showNotification('info', title, message, { autoHide });
  }

  markNotificationAsRead(notificationId: string): void {
    runInAction(() => {
      const notification = this.uiState.notifications.find(n => n.id === notificationId);
      if (notification) {
        notification.isRead = true;
      }
    });
  }

  markAllNotificationsAsRead(): void {
    runInAction(() => {
      this.uiState.notifications.forEach(notification => {
        notification.isRead = true;
      });
    });
  }

  removeNotification(notificationId: string): void {
    // Clear auto-hide timer if exists
    const timer = this.autoHideTimers.get(notificationId);
    if (timer) {
      clearTimeout(timer);
      this.autoHideTimers.delete(notificationId);
    }

    runInAction(() => {
      this.uiState.notifications = this.uiState.notifications.filter(
        n => n.id !== notificationId
      );
    });
  }

  clearAllNotifications(): void {
    // Clear all auto-hide timers
    this.autoHideTimers.forEach(timer => clearTimeout(timer));
    this.autoHideTimers.clear();

    runInAction(() => {
      this.uiState.notifications = [];
    });
  }

  // Tab management
  setActiveTab(tabName: string): void {
    runInAction(() => {
      this.uiState.activeTab = tabName;
    });
  }

  // Sidebar management
  setSidebarOpen(isOpen: boolean): void {
    runInAction(() => {
      this.uiState.sidebarOpen = isOpen;
    });
  }

  toggleSidebar(): void {
    this.setSidebarOpen(!this.uiState.sidebarOpen);
  }

  // Window/viewport management
  private handleWindowResize = (): void => {
    runInAction(() => {
      this.windowWidth = window.innerWidth;
      this.windowHeight = window.innerHeight;
      this.isMobile = this.windowWidth < 768;
      this.isTablet = this.windowWidth >= 768 && this.windowWidth < 1024;
      this.isDesktop = this.windowWidth >= 1024;

      // Auto-manage sidebar based on screen size
      if (this.shouldCollapseSidebar && this.uiState.sidebarOpen) {
        this.uiState.sidebarOpen = false;
      } else if (this.isDesktop && !this.uiState.sidebarOpen && !this.isAnyModalOpen) {
        this.uiState.sidebarOpen = true;
      }
    });
  };

  // Theme and appearance
  applyTheme(theme: 'light' | 'dark' | 'auto'): void {
    if (typeof document === 'undefined') return;

    const root = document.documentElement;

    if (theme === 'auto') {
      // Use system preference
      const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
      theme = prefersDark ? 'dark' : 'light';
    }

    root.setAttribute('data-theme', theme);
    root.classList.toggle('dark', theme === 'dark');
  }

  // State persistence helpers
  getPersistedUIState(): Pick<UIState, 'activeTab' | 'sidebarOpen'> {
    return {
      activeTab: this.uiState.activeTab,
      sidebarOpen: this.uiState.sidebarOpen
    };
  }

  restoreUIState(state: Partial<Pick<UIState, 'activeTab' | 'sidebarOpen'>>): void {
    runInAction(() => {
      if (state.activeTab) {
        this.uiState.activeTab = state.activeTab;
      }
      if (state.sidebarOpen !== undefined) {
        this.uiState.sidebarOpen = state.sidebarOpen;
      }
    });
  }

  // Preferences persistence
  private loadPreferences(): void {
    try {
      const saved = localStorage.getItem('billiards-ui-preferences');
      if (saved) {
        const preferences = JSON.parse(saved);
        this.restoreUIState(preferences);
      }
    } catch (error) {
      console.warn('Failed to load UI preferences:', error);
    }
  }

  private savePreferences(): void {
    try {
      const preferences = this.getPersistedUIState();
      localStorage.setItem('billiards-ui-preferences', JSON.stringify(preferences));
    } catch (error) {
      console.warn('Failed to save UI preferences:', error);
    }
  }

  // Cleanup
  destroy(): void {
    // Clear all notification timers
    this.autoHideTimers.forEach(timer => clearTimeout(timer));
    this.autoHideTimers.clear();

    // Remove window event listener
    if (typeof window !== 'undefined') {
      window.removeEventListener('resize', this.handleWindowResize);
    }
  }
}
