/**
 * Module status component showing individual module health and diagnostics
 * Monitors config, core, vision, api, and projector modules
 */

import React, { useEffect, useState } from 'react';
import { observer } from 'mobx-react-lite';
import { useStores } from '../../stores/context';
import { StatusIndicator } from './StatusIndicator';
import { StatCard } from './StatCard';
import { ProgressBar } from './ProgressBar';
import { AlertPanel } from './AlertPanel';
import type { Alert } from '../../types/monitoring';
import type { ComponentHealth, HealthStatus } from '../../types/api';

interface ModuleInfo {
  id: string;
  name: string;
  description: string;
  icon: string;
  health: ComponentHealth;
  metrics?: {
    cpu_usage?: number;
    memory_usage?: number;
    requests_per_second?: number;
    error_rate?: number;
    uptime?: number;
    last_restart?: Date;
  };
  actions?: Array<{
    label: string;
    action: () => void;
    type: 'primary' | 'secondary' | 'danger';
  }>;
}

export const ModuleStatus: React.FC = observer(() => {
  const { connectionStore } = useStores();
  const [modules, setModules] = useState<ModuleInfo[]>([]);
  const [selectedModule, setSelectedModule] = useState<string | null>(null);
  const [moduleAlerts, setModuleAlerts] = useState<Alert[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadModuleStatus();
    const interval = setInterval(loadModuleStatus, 10000); // Update every 10 seconds
    return () => clearInterval(interval);
  }, []);

  const loadModuleStatus = async () => {
    try {
      // Simulate module status data
      const mockModules: ModuleInfo[] = [
        {
          id: 'config',
          name: 'Configuration Manager',
          description: 'Manages system configuration and settings',
          icon: '⚙️',
          health: {
            name: 'Configuration Manager',
            status: 'healthy' as HealthStatus,
            message: 'All configurations loaded successfully',
            last_check: new Date().toISOString(),
            uptime: 3600,
            errors: [],
          },
          metrics: {
            cpu_usage: 2 + Math.random() * 3,
            memory_usage: 45 + Math.random() * 10,
            requests_per_second: 0.5 + Math.random() * 1,
            error_rate: 0,
            uptime: 3600,
          },
          actions: [
            {
              label: 'Reload Config',
              action: () => console.log('Reloading config...'),
              type: 'secondary',
            },
            {
              label: 'Export Settings',
              action: () => console.log('Exporting settings...'),
              type: 'secondary',
            },
          ],
        },
        {
          id: 'core',
          name: 'Core Engine',
          description: 'Main application logic and game state management',
          icon: '🎯',
          health: {
            name: 'Core Engine',
            status: 'healthy' as HealthStatus,
            message: 'Game engine running normally',
            last_check: new Date().toISOString(),
            uptime: 3580,
            errors: [],
          },
          metrics: {
            cpu_usage: 15 + Math.random() * 10,
            memory_usage: 35 + Math.random() * 15,
            requests_per_second: 2 + Math.random() * 3,
            error_rate: 0.1,
            uptime: 3580,
          },
          actions: [
            {
              label: 'Reset Game State',
              action: () => console.log('Resetting game state...'),
              type: 'danger',
            },
            {
              label: 'View Logs',
              action: () => console.log('Opening logs...'),
              type: 'secondary',
            },
          ],
        },
        {
          id: 'vision',
          name: 'Vision System',
          description: 'Computer vision and object tracking',
          icon: '👁️',
          health: {
            name: 'Vision System',
            status: 'healthy' as HealthStatus,
            message: 'Camera active, tracking 15 objects',
            last_check: new Date().toISOString(),
            uptime: 3500,
            errors: [],
          },
          metrics: {
            cpu_usage: 45 + Math.random() * 20,
            memory_usage: 60 + Math.random() * 15,
            requests_per_second: 25 + Math.random() * 10, // FPS-like
            error_rate: 0.2,
            uptime: 3500,
          },
          actions: [
            {
              label: 'Recalibrate Camera',
              action: () => console.log('Starting calibration...'),
              type: 'primary',
            },
            {
              label: 'Test Detection',
              action: () => console.log('Testing detection...'),
              type: 'secondary',
            },
          ],
        },
        {
          id: 'api',
          name: 'API Server',
          description: 'REST API and WebSocket server',
          icon: '🌐',
          health: {
            name: 'API Server',
            status: connectionStore.state.isConnected ? 'healthy' : 'unhealthy' as HealthStatus,
            message: connectionStore.state.isConnected
              ? 'Serving requests, 2 active connections'
              : 'Server unreachable',
            last_check: new Date().toISOString(),
            uptime: connectionStore.state.isConnected ? 3600 : 0,
            errors: connectionStore.state.error ? [connectionStore.state.error] : [],
          },
          metrics: {
            cpu_usage: 8 + Math.random() * 5,
            memory_usage: 25 + Math.random() * 10,
            requests_per_second: 5 + Math.random() * 10,
            error_rate: connectionStore.state.isConnected ? 0.1 : 100,
            uptime: connectionStore.state.isConnected ? 3600 : 0,
          },
          actions: [
            {
              label: 'Restart Server',
              action: () => console.log('Restarting server...'),
              type: 'danger',
            },
            {
              label: 'View API Docs',
              action: () => window.open('/docs', '_blank'),
              type: 'secondary',
            },
          ],
        },
        {
          id: 'projector',
          name: 'Projector Control',
          description: 'Projector calibration and overlay rendering',
          icon: '📽️',
          health: {
            name: 'Projector Control',
            status: 'degraded' as HealthStatus,
            message: 'Projector connected, minor calibration drift detected',
            last_check: new Date().toISOString(),
            uptime: 3400,
            errors: ['Calibration accuracy: 95% (target: 98%)'],
          },
          metrics: {
            cpu_usage: 20 + Math.random() * 15,
            memory_usage: 40 + Math.random() * 15,
            requests_per_second: 1 + Math.random() * 2,
            error_rate: 2.5,
            uptime: 3400,
          },
          actions: [
            {
              label: 'Recalibrate Projector',
              action: () => console.log('Starting projector calibration...'),
              type: 'primary',
            },
            {
              label: 'Test Projection',
              action: () => console.log('Testing projection...'),
              type: 'secondary',
            },
          ],
        },
      ];

      setModules(mockModules);
      generateModuleAlerts(mockModules);
      setLoading(false);
    } catch (error) {
      console.error('Failed to load module status:', error);
      setLoading(false);
    }
  };

  const generateModuleAlerts = (modules: ModuleInfo[]) => {
    const alerts: Alert[] = [];

    modules.forEach(module => {
      if (module.health.status === 'unhealthy') {
        alerts.push({
          id: `module-${module.id}-unhealthy`,
          level: 'error',
          title: `${module.name} Unhealthy`,
          message: module.health.message || 'Module is not responding',
          timestamp: new Date(),
          action: {
            label: 'View Details',
            onClick: () => setSelectedModule(module.id),
          },
        });
      } else if (module.health.status === 'degraded') {
        alerts.push({
          id: `module-${module.id}-degraded`,
          level: 'warning',
          title: `${module.name} Degraded`,
          message: module.health.message || 'Module performance is degraded',
          timestamp: new Date(),
          action: {
            label: 'View Details',
            onClick: () => setSelectedModule(module.id),
          },
        });
      }

      // High error rate alert
      if (module.metrics?.error_rate && module.metrics.error_rate > 5) {
        alerts.push({
          id: `module-${module.id}-high-errors`,
          level: 'warning',
          title: `High Error Rate in ${module.name}`,
          message: `Error rate: ${module.metrics.error_rate.toFixed(1)}%`,
          timestamp: new Date(),
        });
      }
    });

    setModuleAlerts(alerts);
  };

  const formatUptime = (seconds: number): string => {
    if (seconds === 0) return 'Offline';
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    return `${hours}h ${minutes}m`;
  };

  const getStatusColor = (status: HealthStatus): 'green' | 'yellow' | 'red' => {
    switch (status) {
      case 'healthy': return 'green';
      case 'degraded': return 'yellow';
      case 'unhealthy': return 'red';
      default: return 'red';
    }
  };

  if (loading) {
    return (
      <div className="space-y-6">
        <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6">
          {[...Array(6)].map((_, i) => (
            <div key={i} className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6">
              <div className="animate-pulse">
                <div className="h-6 bg-gray-300 dark:bg-gray-600 rounded mb-4" />
                <div className="h-4 bg-gray-300 dark:bg-gray-600 rounded mb-2" />
                <div className="h-4 bg-gray-300 dark:bg-gray-600 rounded mb-2" />
              </div>
            </div>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Module Alerts */}
      {moduleAlerts.length > 0 && (
        <div>
          <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">
            Module Alerts
          </h3>
          <AlertPanel alerts={moduleAlerts} />
        </div>
      )}

      {/* Module Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6">
        {modules.map((module) => (
          <div
            key={module.id}
            className={`
              bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6
              ${selectedModule === module.id ? 'ring-2 ring-blue-500' : ''}
              cursor-pointer hover:shadow-md transition-shadow
            `}
            onClick={() => setSelectedModule(selectedModule === module.id ? null : module.id)}
          >
            {/* Module Header */}
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center">
                <span className="text-2xl mr-3">{module.icon}</span>
                <div>
                  <h3 className="text-lg font-medium text-gray-900 dark:text-white">
                    {module.name}
                  </h3>
                  <p className="text-sm text-gray-500 dark:text-gray-400">
                    {module.description}
                  </p>
                </div>
              </div>
              <StatusIndicator
                status={module.health.status}
                size="lg"
                showLabel={false}
              />
            </div>

            {/* Module Status */}
            <div className="mb-4">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                  Status
                </span>
                <StatusIndicator
                  status={module.health.status}
                  size="sm"
                />
              </div>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                {module.health.message}
              </p>
            </div>

            {/* Module Metrics */}
            {module.metrics && (
              <div className="space-y-3 mb-4">
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <span className="text-xs text-gray-500 dark:text-gray-400">CPU</span>
                    <div className="flex items-center">
                      <span className="text-sm font-medium text-gray-900 dark:text-white mr-2">
                        {module.metrics.cpu_usage?.toFixed(1)}%
                      </span>
                      <ProgressBar
                        value={module.metrics.cpu_usage || 0}
                        size="sm"
                        color={module.metrics.cpu_usage! < 60 ? 'green' : module.metrics.cpu_usage! < 80 ? 'yellow' : 'red'}
                        showPercentage={false}
                        className="flex-1"
                      />
                    </div>
                  </div>
                  <div>
                    <span className="text-xs text-gray-500 dark:text-gray-400">Memory</span>
                    <div className="flex items-center">
                      <span className="text-sm font-medium text-gray-900 dark:text-white mr-2">
                        {module.metrics.memory_usage?.toFixed(1)}%
                      </span>
                      <ProgressBar
                        value={module.metrics.memory_usage || 0}
                        size="sm"
                        color={module.metrics.memory_usage! < 70 ? 'green' : module.metrics.memory_usage! < 85 ? 'yellow' : 'red'}
                        showPercentage={false}
                        className="flex-1"
                      />
                    </div>
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <span className="text-gray-500 dark:text-gray-400">Uptime:</span>
                    <span className="ml-1 font-medium text-gray-900 dark:text-white">
                      {formatUptime(module.metrics.uptime || 0)}
                    </span>
                  </div>
                  <div>
                    <span className="text-gray-500 dark:text-gray-400">Error Rate:</span>
                    <span className={`ml-1 font-medium ${
                      module.metrics.error_rate! < 1 ? 'text-green-600' :
                      module.metrics.error_rate! < 5 ? 'text-yellow-600' : 'text-red-600'
                    }`}>
                      {module.metrics.error_rate?.toFixed(1)}%
                    </span>
                  </div>
                </div>
              </div>
            )}

            {/* Expanded Details */}
            {selectedModule === module.id && (
              <div className="border-t border-gray-200 dark:border-gray-700 pt-4 space-y-4">
                {/* Detailed Metrics */}
                {module.metrics && (
                  <div>
                    <h4 className="text-sm font-medium text-gray-900 dark:text-white mb-2">
                      Performance Metrics
                    </h4>
                    <div className="grid grid-cols-2 gap-4 text-sm">
                      <div>
                        <span className="text-gray-500 dark:text-gray-400">Requests/sec:</span>
                        <span className="ml-1 font-medium text-gray-900 dark:text-white">
                          {module.metrics.requests_per_second?.toFixed(1)}
                        </span>
                      </div>
                      <div>
                        <span className="text-gray-500 dark:text-gray-400">Last Check:</span>
                        <span className="ml-1 font-medium text-gray-900 dark:text-white">
                          {new Date(module.health.last_check).toLocaleTimeString()}
                        </span>
                      </div>
                    </div>
                  </div>
                )}

                {/* Errors */}
                {module.health.errors.length > 0 && (
                  <div>
                    <h4 className="text-sm font-medium text-gray-900 dark:text-white mb-2">
                      Recent Errors
                    </h4>
                    <ul className="text-sm text-red-600 dark:text-red-400 space-y-1">
                      {module.health.errors.map((error, index) => (
                        <li key={index}>• {error}</li>
                      ))}
                    </ul>
                  </div>
                )}

                {/* Actions */}
                {module.actions && module.actions.length > 0 && (
                  <div>
                    <h4 className="text-sm font-medium text-gray-900 dark:text-white mb-2">
                      Actions
                    </h4>
                    <div className="flex flex-wrap gap-2">
                      {module.actions.map((action, index) => (
                        <button
                          key={index}
                          onClick={(e) => {
                            e.stopPropagation();
                            action.action();
                          }}
                          className={`
                            px-3 py-1 text-sm rounded-md transition-colors
                            ${action.type === 'primary' ? 'bg-blue-600 hover:bg-blue-700 text-white' :
                              action.type === 'danger' ? 'bg-red-600 hover:bg-red-700 text-white' :
                              'bg-gray-200 hover:bg-gray-300 text-gray-900 dark:bg-gray-600 dark:hover:bg-gray-500 dark:text-white'}
                          `}
                        >
                          {action.label}
                        </button>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        ))}
      </div>

      {/* Summary Stats */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <StatCard
          title="Healthy Modules"
          value={modules.filter(m => m.health.status === 'healthy').length}
          unit={` / ${modules.length}`}
          icon="✅"
          color="green"
        />

        <StatCard
          title="Degraded Modules"
          value={modules.filter(m => m.health.status === 'degraded').length}
          unit={` / ${modules.length}`}
          icon="⚠️"
          color="yellow"
        />

        <StatCard
          title="Failed Modules"
          value={modules.filter(m => m.health.status === 'unhealthy').length}
          unit={` / ${modules.length}`}
          icon="❌"
          color="red"
        />
      </div>
    </div>
  );
});

export default ModuleStatus;
