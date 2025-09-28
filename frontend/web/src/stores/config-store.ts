/**
 * MobX store for configuration management
 */

import { makeAutoObservable, runInAction, flow, observable } from 'mobx';
import {
  ConfigResponse,
  ConfigUpdateRequest,
  ConfigUpdateResponse,
  ConfigValidationError,
} from '../types/api';
import type { RootStore } from './index';

export interface ConfigSection {
  name: string;
  values: Record<string, any>;
  lastModified: Date;
  isValid: boolean;
  validationErrors: ConfigValidationError[];
}

export interface ConfigChange {
  section: string;
  field: string;
  oldValue: any;
  newValue: any;
  timestamp: Date;
  applied: boolean;
}

export class ConfigStore {
  private rootStore: RootStore;

  // Configuration data
  sections = observable.map<string, ConfigSection>();
  allConfig: Record<string, any> = {};
  schemaVersion = '';
  lastModified: Date | null = null;

  // Change tracking
  pendingChanges = observable.array<ConfigChange>([]);
  changeHistory = observable.array<ConfigChange>([]);

  // Loading and error states
  isLoading = false;
  isSaving = false;
  error: string | null = null;
  saveError: string | null = null;

  // Validation
  validationErrors = observable.array<ConfigValidationError>([]);
  isValid = true;

  // Configuration categories for UI organization
  readonly categories = {
    vision: ['camera', 'detection', 'tracking'],
    hardware: ['camera', 'projector', 'calibration'],
    game: ['rules', 'scoring', 'physics'],
    system: ['logging', 'performance', 'security'],
    ui: ['display', 'themes', 'preferences'],
  };

  constructor(rootStore: RootStore) {
    makeAutoObservable(this, {}, { autoBind: true });
    this.rootStore = rootStore;
  }

  // =============================================================================
  // Configuration Loading
  // =============================================================================

  loadAllConfig = flow(function* (this: ConfigStore) {
    this.isLoading = true;
    this.error = null;

    try {
      const response: ConfigResponse = yield this.rootStore.apiService.getConfig();

      runInAction(() => {
        this.allConfig = response.values;
        this.schemaVersion = response.schema_version;
        this.lastModified = new Date(response.last_modified);
        this.isValid = response.is_valid;
        this.validationErrors.replace(response.validation_errors);

        // Organize config into sections
        this.organizeConfigIntoSections(response.values);

        this.isLoading = false;
      });

    } catch (error) {
      runInAction(() => {
        this.isLoading = false;
        this.error = error instanceof Error ? error.message : 'Failed to load configuration';
      });
    }
  });

  loadConfigSection = flow(function* (this: ConfigStore, sectionName: string) {
    this.isLoading = true;
    this.error = null;

    try {
      const response: ConfigResponse = yield this.rootStore.apiService.getConfig(sectionName);

      runInAction(() => {
        const section: ConfigSection = {
          name: sectionName,
          values: response.values,
          lastModified: new Date(response.last_modified),
          isValid: response.is_valid,
          validationErrors: response.validation_errors,
        };

        this.sections.set(sectionName, section);

        // Update global config
        Object.assign(this.allConfig, response.values);

        this.isLoading = false;
      });

    } catch (error) {
      runInAction(() => {
        this.isLoading = false;
        this.error = error instanceof Error ? error.message : `Failed to load ${sectionName} configuration`;
      });
    }
  });

  // =============================================================================
  // Configuration Updates
  // =============================================================================

  updateConfigValue(section: string, field: string, value: any): void {
    const oldValue = this.getConfigValue(section, field);

    if (oldValue !== value) {
      // Track the change
      const change: ConfigChange = {
        section,
        field,
        oldValue,
        newValue: value,
        timestamp: new Date(),
        applied: false,
      };

      this.pendingChanges.push(change);

      // Update local state
      if (!this.allConfig[section]) {
        this.allConfig[section] = {};
      }
      this.allConfig[section][field] = value;

      // Update section if it exists
      const sectionData = this.sections.get(section);
      if (sectionData) {
        sectionData.values[field] = value;
      }
    }
  }

  saveAllChanges = flow(function* (this: ConfigStore) {
    if (this.pendingChanges.length === 0) return;

    this.isSaving = true;
    this.saveError = null;

    try {
      // Group changes by section
      const changesBySection = new Map<string, Record<string, any>>();

      this.pendingChanges.forEach(change => {
        if (!changesBySection.has(change.section)) {
          changesBySection.set(change.section, {});
        }
        changesBySection.get(change.section)![change.field] = change.newValue;
      });

      // Save each section
      for (const [section, values] of changesBySection) {
        const request: ConfigUpdateRequest = {
          section,
          values,
        };

        const response: ConfigUpdateResponse = yield this.rootStore.apiService.updateConfig(request);

        if (!response.success) {
          throw new Error(`Failed to save ${section}: ${response.validation_errors.map(e => e.message).join(', ')}`);
        }
      }

      runInAction(() => {
        // Mark changes as applied
        this.pendingChanges.forEach(change => {
          change.applied = true;
          this.changeHistory.push(change);
        });

        // Clear pending changes
        this.pendingChanges.clear();

        this.isSaving = false;

        // Reload configuration to get latest state
        this.loadAllConfig();
      });

    } catch (error) {
      runInAction(() => {
        this.isSaving = false;
        this.saveError = error instanceof Error ? error.message : 'Failed to save configuration';
      });
    }
  });

  saveSectionChanges = flow(function* (this: ConfigStore, sectionName: string) {
    const sectionChanges = this.pendingChanges.filter(change => change.section === sectionName);

    if (sectionChanges.length === 0) return;

    this.isSaving = true;
    this.saveError = null;

    try {
      const values: Record<string, any> = {};
      sectionChanges.forEach(change => {
        values[change.field] = change.newValue;
      });

      const request: ConfigUpdateRequest = {
        section: sectionName,
        values,
      };

      const response: ConfigUpdateResponse = yield this.rootStore.apiService.updateConfig(request);

      if (!response.success) {
        throw new Error(`Validation errors: ${response.validation_errors.map(e => e.message).join(', ')}`);
      }

      runInAction(() => {
        // Mark section changes as applied
        sectionChanges.forEach(change => {
          change.applied = true;
          this.changeHistory.push(change);
        });

        // Remove from pending changes
        this.pendingChanges.replace(
          this.pendingChanges.filter(change => change.section !== sectionName)
        );

        this.isSaving = false;

        // Reload the section
        this.loadConfigSection(sectionName);
      });

    } catch (error) {
      runInAction(() => {
        this.isSaving = false;
        this.saveError = error instanceof Error ? error.message : `Failed to save ${sectionName} configuration`;
      });
    }
  });

  validateConfig = flow(function* (this: ConfigStore, values: Record<string, any>) {
    try {
      const response: ConfigUpdateResponse = yield this.rootStore.apiService.validateConfig(values);

      runInAction(() => {
        this.validationErrors.replace(response.validation_errors);
        this.isValid = response.validation_errors.length === 0;
      });

      return response.validation_errors;

    } catch (error) {
      runInAction(() => {
        this.error = error instanceof Error ? error.message : 'Configuration validation failed';
      });
      return [];
    }
  });

  // =============================================================================
  // Change Management
  // =============================================================================

  discardChanges(): void {
    this.pendingChanges.clear();
    this.saveError = null;

    // Reload configuration to reset local state
    this.loadAllConfig();
  }

  discardSectionChanges(sectionName: string): void {
    this.pendingChanges.replace(
      this.pendingChanges.filter(change => change.section !== sectionName)
    );

    // Reload the section
    this.loadConfigSection(sectionName);
  }

  undoChange(changeIndex: number): void {
    if (changeIndex >= 0 && changeIndex < this.pendingChanges.length) {
      const change = this.pendingChanges[changeIndex];

      // Revert the value
      this.allConfig[change.section][change.field] = change.oldValue;

      // Update section if it exists
      const section = this.sections.get(change.section);
      if (section) {
        section.values[change.field] = change.oldValue;
      }

      // Remove from pending changes
      this.pendingChanges.splice(changeIndex, 1);
    }
  }

  // =============================================================================
  // Configuration Export/Import
  // =============================================================================

  exportConfig = flow(function* (this: ConfigStore, format: string = 'json') {
    try {
      const blob: Blob = yield this.rootStore.apiService.exportConfig(format);

      // Create download link
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `billiards-config-${new Date().toISOString().split('T')[0]}.${format}`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);

    } catch (error) {
      runInAction(() => {
        this.error = error instanceof Error ? error.message : 'Failed to export configuration';
      });
    }
  });

  // =============================================================================
  // Computed Properties
  // =============================================================================

  get hasPendingChanges(): boolean {
    return this.pendingChanges.length > 0;
  }

  get pendingChangesBySection(): Map<string, ConfigChange[]> {
    const map = new Map<string, ConfigChange[]>();

    this.pendingChanges.forEach(change => {
      if (!map.has(change.section)) {
        map.set(change.section, []);
      }
      map.get(change.section)!.push(change);
    });

    return map;
  }

  get configSectionNames(): string[] {
    return Array.from(this.sections.keys()).sort();
  }

  get categorizedSections(): Record<string, string[]> {
    const result: Record<string, string[]> = {};

    Object.entries(this.categories).forEach(([category, sections]) => {
      result[category] = sections.filter(section => this.sections.has(section));
    });

    // Add uncategorized sections
    const categorizedSectionNames = new Set(
      Object.values(this.categories).flat()
    );

    const uncategorized = this.configSectionNames.filter(
      name => !categorizedSectionNames.has(name)
    );

    if (uncategorized.length > 0) {
      result.other = uncategorized;
    }

    return result;
  }

  // =============================================================================
  // Helper Methods
  // =============================================================================

  getConfigValue(section: string, field: string, defaultValue?: any): any {
    return this.allConfig[section]?.[field] ?? defaultValue;
  }

  getSectionConfig(sectionName: string): Record<string, any> {
    const section = this.sections.get(sectionName);
    return section ? { ...section.values } : {};
  }

  hasSection(sectionName: string): boolean {
    return this.sections.has(sectionName);
  }

  isSectionValid(sectionName: string): boolean {
    const section = this.sections.get(sectionName);
    return section ? section.isValid : true;
  }

  getSectionValidationErrors(sectionName: string): ConfigValidationError[] {
    const section = this.sections.get(sectionName);
    return section ? [...section.validationErrors] : [];
  }

  private organizeConfigIntoSections(config: Record<string, any>): void {
    // Clear existing sections
    this.sections.clear();

    // Group config values by top-level keys (sections)
    Object.entries(config).forEach(([sectionName, sectionValues]) => {
      if (typeof sectionValues === 'object' && sectionValues !== null) {
        const section: ConfigSection = {
          name: sectionName,
          values: { ...sectionValues },
          lastModified: this.lastModified || new Date(),
          isValid: true,
          validationErrors: [],
        };

        this.sections.set(sectionName, section);
      }
    });
  }

  clearError(): void {
    this.error = null;
    this.saveError = null;
  }

  // =============================================================================
  // Store Lifecycle
  // =============================================================================

  reset(): void {
    this.sections.clear();
    this.allConfig = {};
    this.schemaVersion = '';
    this.lastModified = null;
    this.pendingChanges.clear();
    this.changeHistory.clear();
    this.isLoading = false;
    this.isSaving = false;
    this.error = null;
    this.saveError = null;
    this.validationErrors.clear();
    this.isValid = true;
  }

  destroy(): void {
    this.reset();
  }
}
