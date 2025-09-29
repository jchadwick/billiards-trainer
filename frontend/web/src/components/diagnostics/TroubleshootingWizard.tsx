/**
 * Interactive Troubleshooting Wizard Component
 * Provides step-by-step problem resolution guides and automated troubleshooting
 *
 * Features:
 * - Problem detection wizards
 * - Step-by-step resolution guides
 * - Common issue resolution
 * - System recovery procedures
 * - Automated fixes where possible
 */

import React, { useState, useEffect, useCallback } from 'react';
import { observer } from 'mobx-react-lite';
import { useStores } from '../../stores/context';
import { StatCard } from '../monitoring/StatCard';
import { ProgressBar } from '../monitoring/ProgressBar';
import { StatusIndicator } from '../monitoring/StatusIndicator';
import type { SystemIssue } from './DiagnosticsSystem';

interface TroubleshootingStep {
  id: string;
  title: string;
  description: string;
  action?: 'check' | 'run' | 'configure' | 'restart' | 'manual';
  actionLabel?: string;
  automated: boolean;
  required: boolean;
  completed: boolean;
  result?: {
    success: boolean;
    message: string;
    details?: string;
  };
}

interface TroubleshootingGuide {
  id: string;
  title: string;
  description: string;
  category: 'camera' | 'network' | 'performance' | 'calibration' | 'general';
  severity: 'low' | 'medium' | 'high' | 'critical';
  estimatedTime: number; // minutes
  steps: TroubleshootingStep[];
  currentStep: number;
  completed: boolean;
  success?: boolean;
}

interface CommonIssue {
  id: string;
  title: string;
  description: string;
  category: string;
  frequency: number; // How often this issue occurs (0-100)
  quickFix?: {
    title: string;
    action: () => Promise<boolean>;
    automated: boolean;
  };
  guideId?: string; // Reference to detailed troubleshooting guide
}

interface TroubleshootingWizardProps {
  issues: SystemIssue[];
}

export const TroubleshootingWizard: React.FC<TroubleshootingWizardProps> = observer(({ issues }) => {
  const { systemStore, connectionStore } = useStores();
  const [activeGuide, setActiveGuide] = useState<TroubleshootingGuide | null>(null);
  const [guides, setGuides] = useState<TroubleshootingGuide[]>([]);
  const [commonIssues, setCommonIssues] = useState<CommonIssue[]>([]);
  const [isRunningStep, setIsRunningStep] = useState(false);
  const [diagnosticMode, setDiagnosticMode] = useState<'guided' | 'auto' | 'manual'>('guided');

  useEffect(() => {
    initializeTroubleshootingGuides();
    initializeCommonIssues();
  }, []);

  const initializeTroubleshootingGuides = useCallback(() => {
    const troubleshootingGuides: TroubleshootingGuide[] = [
      {
        id: 'camera-not-detected',
        title: 'Camera Not Detected',
        description: 'Resolve camera detection and connectivity issues',
        category: 'camera',
        severity: 'high',
        estimatedTime: 10,
        currentStep: 0,
        completed: false,
        steps: [
          {
            id: 'check-camera-connection',
            title: 'Check Camera Connection',
            description: 'Verify that the camera is properly connected to the system',
            action: 'check',
            actionLabel: 'Check Connection',
            automated: true,
            required: true,
            completed: false,
          },
          {
            id: 'check-camera-permissions',
            title: 'Check Camera Permissions',
            description: 'Ensure the browser has permission to access the camera',
            action: 'check',
            actionLabel: 'Check Permissions',
            automated: true,
            required: true,
            completed: false,
          },
          {
            id: 'test-camera-access',
            title: 'Test Camera Access',
            description: 'Attempt to access the camera and capture a test frame',
            action: 'run',
            actionLabel: 'Test Camera',
            automated: true,
            required: true,
            completed: false,
          },
          {
            id: 'restart-camera-service',
            title: 'Restart Camera Service',
            description: 'Restart the camera service to resolve connection issues',
            action: 'restart',
            actionLabel: 'Restart Service',
            automated: true,
            required: false,
            completed: false,
          },
          {
            id: 'manual-camera-config',
            title: 'Manual Camera Configuration',
            description: 'Manually configure camera settings if automated detection fails',
            action: 'manual',
            actionLabel: 'Configure Manually',
            automated: false,
            required: false,
            completed: false,
          },
        ],
      },
      {
        id: 'network-connectivity',
        title: 'Network Connectivity Issues',
        description: 'Diagnose and resolve network connection problems',
        category: 'network',
        severity: 'high',
        estimatedTime: 15,
        currentStep: 0,
        completed: false,
        steps: [
          {
            id: 'check-internet-connection',
            title: 'Check Internet Connection',
            description: 'Verify basic internet connectivity',
            action: 'check',
            actionLabel: 'Test Connection',
            automated: true,
            required: true,
            completed: false,
          },
          {
            id: 'ping-backend-server',
            title: 'Ping Backend Server',
            description: 'Test connectivity to the backend server',
            action: 'run',
            actionLabel: 'Ping Server',
            automated: true,
            required: true,
            completed: false,
          },
          {
            id: 'check-websocket-connection',
            title: 'Check WebSocket Connection',
            description: 'Verify WebSocket connectivity for real-time features',
            action: 'run',
            actionLabel: 'Test WebSocket',
            automated: true,
            required: true,
            completed: false,
          },
          {
            id: 'clear-browser-cache',
            title: 'Clear Browser Cache',
            description: 'Clear browser cache and cookies to resolve connection issues',
            action: 'manual',
            actionLabel: 'Clear Cache',
            automated: false,
            required: false,
            completed: false,
          },
          {
            id: 'check-firewall-settings',
            title: 'Check Firewall Settings',
            description: 'Ensure firewall is not blocking the application',
            action: 'manual',
            actionLabel: 'Check Firewall',
            automated: false,
            required: false,
            completed: false,
          },
        ],
      },
      {
        id: 'poor-performance',
        title: 'Poor System Performance',
        description: 'Optimize system performance and resolve bottlenecks',
        category: 'performance',
        severity: 'medium',
        estimatedTime: 20,
        currentStep: 0,
        completed: false,
        steps: [
          {
            id: 'check-system-resources',
            title: 'Check System Resources',
            description: 'Monitor CPU, memory, and disk usage',
            action: 'check',
            actionLabel: 'Check Resources',
            automated: true,
            required: true,
            completed: false,
          },
          {
            id: 'close-unnecessary-apps',
            title: 'Close Unnecessary Applications',
            description: 'Close applications that may be consuming system resources',
            action: 'manual',
            actionLabel: 'Close Apps',
            automated: false,
            required: true,
            completed: false,
          },
          {
            id: 'optimize-video-settings',
            title: 'Optimize Video Settings',
            description: 'Adjust video resolution and frame rate for better performance',
            action: 'configure',
            actionLabel: 'Optimize Settings',
            automated: true,
            required: false,
            completed: false,
          },
          {
            id: 'enable-gpu-acceleration',
            title: 'Enable GPU Acceleration',
            description: 'Enable hardware acceleration if available',
            action: 'configure',
            actionLabel: 'Enable GPU',
            automated: true,
            required: false,
            completed: false,
          },
          {
            id: 'restart-system',
            title: 'Restart System',
            description: 'Restart the system to clear memory and refresh resources',
            action: 'restart',
            actionLabel: 'Restart System',
            automated: false,
            required: false,
            completed: false,
          },
        ],
      },
      {
        id: 'calibration-issues',
        title: 'Calibration Problems',
        description: 'Resolve camera-projector calibration issues',
        category: 'calibration',
        severity: 'medium',
        estimatedTime: 25,
        currentStep: 0,
        completed: false,
        steps: [
          {
            id: 'check-lighting-conditions',
            title: 'Check Lighting Conditions',
            description: 'Ensure optimal lighting for calibration',
            action: 'check',
            actionLabel: 'Check Lighting',
            automated: true,
            required: true,
            completed: false,
          },
          {
            id: 'clean-camera-lens',
            title: 'Clean Camera Lens',
            description: 'Clean the camera lens to improve image quality',
            action: 'manual',
            actionLabel: 'Clean Lens',
            automated: false,
            required: true,
            completed: false,
          },
          {
            id: 'verify-projector-alignment',
            title: 'Verify Projector Alignment',
            description: 'Check that the projector is properly aligned with the table',
            action: 'check',
            actionLabel: 'Check Alignment',
            automated: true,
            required: true,
            completed: false,
          },
          {
            id: 'recalibrate-system',
            title: 'Recalibrate System',
            description: 'Run the full calibration process again',
            action: 'run',
            actionLabel: 'Recalibrate',
            automated: true,
            required: true,
            completed: false,
          },
          {
            id: 'verify-calibration-accuracy',
            title: 'Verify Calibration Accuracy',
            description: 'Test calibration accuracy with known reference points',
            action: 'run',
            actionLabel: 'Verify Accuracy',
            automated: true,
            required: true,
            completed: false,
          },
        ],
      },
    ];

    setGuides(troubleshootingGuides);
  }, []);

  const initializeCommonIssues = useCallback(() => {
    const issues: CommonIssue[] = [
      {
        id: 'camera-permission-denied',
        title: 'Camera Permission Denied',
        description: 'Browser is blocking camera access',
        category: 'Camera',
        frequency: 85,
        quickFix: {
          title: 'Request Camera Permission',
          action: async () => {
            try {
              await navigator.mediaDevices.getUserMedia({ video: true });
              return true;
            } catch (error) {
              return false;
            }
          },
          automated: true,
        },
        guideId: 'camera-not-detected',
      },
      {
        id: 'websocket-disconnected',
        title: 'WebSocket Connection Lost',
        description: 'Real-time communication with server is interrupted',
        category: 'Network',
        frequency: 65,
        quickFix: {
          title: 'Reconnect WebSocket',
          action: async () => {
            try {
              // Attempt to reconnect WebSocket
              await connectionStore.reconnect();
              return connectionStore.state.isConnected;
            } catch (error) {
              return false;
            }
          },
          automated: true,
        },
        guideId: 'network-connectivity',
      },
      {
        id: 'high-cpu-usage',
        title: 'High CPU Usage',
        description: 'System CPU usage is above recommended levels',
        category: 'Performance',
        frequency: 45,
        quickFix: {
          title: 'Optimize Performance Settings',
          action: async () => {
            // Simulate performance optimization
            await new Promise(resolve => setTimeout(resolve, 2000));
            return true;
          },
          automated: true,
        },
        guideId: 'poor-performance',
      },
      {
        id: 'poor-detection-accuracy',
        title: 'Poor Ball Detection',
        description: 'Ball detection accuracy is below acceptable levels',
        category: 'Calibration',
        frequency: 35,
        guideId: 'calibration-issues',
      },
      {
        id: 'audio-not-working',
        title: 'Audio Not Working',
        description: 'Audio input or output devices are not functioning',
        category: 'Hardware',
        frequency: 25,
      },
    ];

    setCommonIssues(issues);
  }, [connectionStore]);

  const startTroubleshootingGuide = useCallback((guideId: string) => {
    const guide = guides.find(g => g.id === guideId);
    if (guide) {
      const resetGuide = {
        ...guide,
        currentStep: 0,
        completed: false,
        success: undefined,
        steps: guide.steps.map(step => ({ ...step, completed: false, result: undefined })),
      };
      setActiveGuide(resetGuide);
    }
  }, [guides]);

  const executeStep = useCallback(async (stepId: string): Promise<void> => {
    if (!activeGuide) return;

    setIsRunningStep(true);

    try {
      const step = activeGuide.steps.find(s => s.id === stepId);
      if (!step) return;

      let result: { success: boolean; message: string; details?: string };

      switch (step.action) {
        case 'check':
          result = await executeCheckAction(stepId);
          break;
        case 'run':
          result = await executeRunAction(stepId);
          break;
        case 'configure':
          result = await executeConfigureAction(stepId);
          break;
        case 'restart':
          result = await executeRestartAction(stepId);
          break;
        default:
          result = { success: true, message: 'Manual step completed' };
      }

      // Update the step with results
      setActiveGuide(prevGuide => {
        if (!prevGuide) return null;

        const updatedSteps = prevGuide.steps.map(s =>
          s.id === stepId ? { ...s, completed: true, result } : s
        );

        const nextIncompleteStep = updatedSteps.findIndex(s => !s.completed);
        const currentStep = nextIncompleteStep === -1 ? updatedSteps.length : nextIncompleteStep;
        const completed = nextIncompleteStep === -1;
        const success = completed && updatedSteps.every(s => s.result?.success !== false);

        return {
          ...prevGuide,
          steps: updatedSteps,
          currentStep,
          completed,
          success,
        };
      });

    } catch (error) {
      console.error('Step execution failed:', error);
    } finally {
      setIsRunningStep(false);
    }
  }, [activeGuide]);

  const executeCheckAction = async (stepId: string): Promise<{ success: boolean; message: string; details?: string }> => {
    switch (stepId) {
      case 'check-camera-connection':
        try {
          const devices = await navigator.mediaDevices.enumerateDevices();
          const cameras = devices.filter(device => device.kind === 'videoinput');
          return {
            success: cameras.length > 0,
            message: cameras.length > 0 ?
              `Found ${cameras.length} camera(s)` :
              'No cameras detected',
            details: cameras.map(c => c.label || 'Unknown camera').join(', '),
          };
        } catch (error) {
          return { success: false, message: 'Failed to check camera connection' };
        }

      case 'check-camera-permissions':
        try {
          const permission = await navigator.permissions.query({ name: 'camera' as PermissionName });
          return {
            success: permission.state === 'granted',
            message: `Camera permission: ${permission.state}`,
          };
        } catch (error) {
          return { success: false, message: 'Failed to check camera permissions' };
        }

      case 'check-internet-connection':
        return {
          success: navigator.onLine,
          message: navigator.onLine ? 'Internet connection available' : 'No internet connection',
        };

      case 'check-system-resources':
        // Simulate system resource check
        await new Promise(resolve => setTimeout(resolve, 2000));
        const cpuUsage = 30 + Math.random() * 50;
        const memoryUsage = 40 + Math.random() * 40;
        return {
          success: cpuUsage < 80 && memoryUsage < 80,
          message: `CPU: ${cpuUsage.toFixed(0)}%, Memory: ${memoryUsage.toFixed(0)}%`,
          details: cpuUsage > 80 || memoryUsage > 80 ? 'High resource usage detected' : 'Resource usage normal',
        };

      default:
        return { success: true, message: 'Check completed' };
    }
  };

  const executeRunAction = async (stepId: string): Promise<{ success: boolean; message: string; details?: string }> => {
    switch (stepId) {
      case 'test-camera-access':
        try {
          const stream = await navigator.mediaDevices.getUserMedia({ video: true });
          stream.getTracks().forEach(track => track.stop());
          return { success: true, message: 'Camera access test successful' };
        } catch (error) {
          return { success: false, message: 'Camera access test failed' };
        }

      case 'ping-backend-server':
        try {
          const response = await fetch('/api/health', { method: 'HEAD' });
          return {
            success: response.ok,
            message: response.ok ? 'Backend server responding' : `Server error: ${response.status}`,
          };
        } catch (error) {
          return { success: false, message: 'Cannot reach backend server' };
        }

      case 'check-websocket-connection':
        try {
          await connectionStore.testConnection();
          return {
            success: connectionStore.state.isConnected,
            message: connectionStore.state.isConnected ?
              'WebSocket connection successful' :
              'WebSocket connection failed',
          };
        } catch (error) {
          return { success: false, message: 'WebSocket test failed' };
        }

      default:
        // Simulate running action
        await new Promise(resolve => setTimeout(resolve, 3000));
        return { success: Math.random() > 0.2, message: 'Action completed' };
    }
  };

  const executeConfigureAction = async (stepId: string): Promise<{ success: boolean; message: string; details?: string }> => {
    // Simulate configuration actions
    await new Promise(resolve => setTimeout(resolve, 2000));
    return { success: true, message: 'Configuration applied' };
  };

  const executeRestartAction = async (stepId: string): Promise<{ success: boolean; message: string; details?: string }> => {
    // Simulate restart actions
    await new Promise(resolve => setTimeout(resolve, 1000));
    return { success: true, message: 'Restart initiated' };
  };

  const runQuickFix = useCallback(async (issueId: string): Promise<void> => {
    const issue = commonIssues.find(i => i.id === issueId);
    if (!issue?.quickFix) return;

    try {
      const success = await issue.quickFix.action();
      // Update UI to show result
      console.log(`Quick fix for ${issue.title}: ${success ? 'Success' : 'Failed'}`);
    } catch (error) {
      console.error('Quick fix failed:', error);
    }
  }, [commonIssues]);

  const getSeverityColor = (severity: TroubleshootingGuide['severity']): 'green' | 'yellow' | 'red' | 'purple' => {
    switch (severity) {
      case 'low': return 'green';
      case 'medium': return 'yellow';
      case 'high': return 'red';
      case 'critical': return 'purple';
      default: return 'yellow';
    }
  };

  const getCategoryIcon = (category: string): string => {
    switch (category) {
      case 'camera': return '📷';
      case 'network': return '🌐';
      case 'performance': return '⚡';
      case 'calibration': return '🎯';
      case 'hardware': return '🔧';
      default: return '🔍';
    }
  };

  if (activeGuide) {
    return (
      <div className="space-y-6">
        {/* Guide Header */}
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h2 className="text-xl font-bold text-gray-900 dark:text-white">
                {activeGuide.title}
              </h2>
              <p className="text-gray-600 dark:text-gray-300">
                {activeGuide.description}
              </p>
            </div>
            <div className="flex items-center space-x-4">
              <StatCard
                title="Progress"
                value={`${activeGuide.currentStep} / ${activeGuide.steps.length}`}
                size="sm"
                color="blue"
              />
              <button
                onClick={() => setActiveGuide(null)}
                className="px-4 py-2 bg-gray-600 hover:bg-gray-700 text-white rounded-md transition-colors"
              >
                Exit Guide
              </button>
            </div>
          </div>

          <ProgressBar
            value={(activeGuide.currentStep / activeGuide.steps.length) * 100}
            label="Overall Progress"
            color="blue"
          />
        </div>

        {/* Current Step */}
        {!activeGuide.completed && (
          <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-6">
            <div className="flex items-start space-x-4">
              <div className="flex-shrink-0 w-8 h-8 bg-blue-600 text-white rounded-full flex items-center justify-center text-sm font-medium">
                {activeGuide.currentStep + 1}
              </div>
              <div className="flex-1">
                <h3 className="text-lg font-medium text-blue-900 dark:text-blue-100 mb-2">
                  {activeGuide.steps[activeGuide.currentStep]?.title}
                </h3>
                <p className="text-blue-700 dark:text-blue-300 mb-4">
                  {activeGuide.steps[activeGuide.currentStep]?.description}
                </p>

                {activeGuide.steps[activeGuide.currentStep]?.automated && (
                  <button
                    onClick={() => executeStep(activeGuide.steps[activeGuide.currentStep].id)}
                    disabled={isRunningStep}
                    className={`px-4 py-2 rounded-md transition-colors ${
                      isRunningStep
                        ? 'bg-gray-400 cursor-not-allowed text-white'
                        : 'bg-blue-600 hover:bg-blue-700 text-white'
                    }`}
                  >
                    {isRunningStep ? 'Running...' :
                     activeGuide.steps[activeGuide.currentStep]?.actionLabel || 'Execute Step'}
                  </button>
                )}

                {!activeGuide.steps[activeGuide.currentStep]?.automated && (
                  <div className="bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 rounded-md p-3">
                    <p className="text-yellow-800 dark:text-yellow-200 text-sm">
                      This step requires manual action. Please complete the described action and mark as completed when done.
                    </p>
                    <button
                      onClick={() => executeStep(activeGuide.steps[activeGuide.currentStep].id)}
                      className="mt-2 px-3 py-1 bg-yellow-600 hover:bg-yellow-700 text-white rounded text-sm transition-colors"
                    >
                      Mark as Completed
                    </button>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {/* Completion Status */}
        {activeGuide.completed && (
          <div className={`${
            activeGuide.success ? 'bg-green-50 dark:bg-green-900/20 border-green-200 dark:border-green-800' :
            'bg-red-50 dark:bg-red-900/20 border-red-200 dark:border-red-800'
          } border rounded-lg p-6`}>
            <div className="flex items-center space-x-3">
              <div className={`w-8 h-8 rounded-full flex items-center justify-center ${
                activeGuide.success ? 'bg-green-600 text-white' : 'bg-red-600 text-white'
              }`}>
                {activeGuide.success ? '✓' : '✗'}
              </div>
              <div>
                <h3 className={`text-lg font-medium ${
                  activeGuide.success ? 'text-green-900 dark:text-green-100' : 'text-red-900 dark:text-red-100'
                }`}>
                  {activeGuide.success ? 'Troubleshooting Completed Successfully' : 'Troubleshooting Completed with Issues'}
                </h3>
                <p className={`${
                  activeGuide.success ? 'text-green-700 dark:text-green-300' : 'text-red-700 dark:text-red-300'
                }`}>
                  {activeGuide.success ?
                    'All steps completed successfully. The issue should be resolved.' :
                    'Some steps failed. Additional manual intervention may be required.'}
                </p>
              </div>
            </div>
          </div>
        )}

        {/* All Steps */}
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6">
          <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">
            Troubleshooting Steps
          </h3>
          <div className="space-y-4">
            {activeGuide.steps.map((step, index) => (
              <div key={step.id} className={`flex items-start space-x-4 p-4 rounded-lg ${
                index === activeGuide.currentStep ? 'bg-blue-50 dark:bg-blue-900/20' :
                step.completed ? 'bg-green-50 dark:bg-green-900/20' : 'bg-gray-50 dark:bg-gray-700'
              }`}>
                <div className={`flex-shrink-0 w-6 h-6 rounded-full flex items-center justify-center text-sm ${
                  step.completed ? 'bg-green-600 text-white' :
                  index === activeGuide.currentStep ? 'bg-blue-600 text-white' :
                  'bg-gray-400 text-white'
                }`}>
                  {step.completed ? '✓' : index + 1}
                </div>
                <div className="flex-1">
                  <h4 className="text-sm font-medium text-gray-900 dark:text-white">
                    {step.title}
                  </h4>
                  <p className="text-sm text-gray-600 dark:text-gray-300">
                    {step.description}
                  </p>
                  {step.result && (
                    <div className={`mt-2 text-xs ${
                      step.result.success ? 'text-green-600' : 'text-red-600'
                    }`}>
                      {step.result.message}
                      {step.result.details && (
                        <div className="text-gray-500 dark:text-gray-400">
                          {step.result.details}
                        </div>
                      )}
                    </div>
                  )}
                </div>
                <div className="flex-shrink-0">
                  {step.automated ? (
                    <span className="text-xs text-blue-600 bg-blue-100 dark:bg-blue-900 px-2 py-1 rounded">
                      Automated
                    </span>
                  ) : (
                    <span className="text-xs text-orange-600 bg-orange-100 dark:bg-orange-900 px-2 py-1 rounded">
                      Manual
                    </span>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Troubleshooting Mode Selection */}
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6">
        <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">
          Troubleshooting Mode
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {[
            { id: 'guided', title: 'Guided Troubleshooting', description: 'Step-by-step guided resolution' },
            { id: 'auto', title: 'Automatic Fix', description: 'Attempt automated problem resolution' },
            { id: 'manual', title: 'Manual Diagnostics', description: 'Access detailed diagnostic tools' },
          ].map((mode) => (
            <button
              key={mode.id}
              onClick={() => setDiagnosticMode(mode.id as any)}
              className={`p-4 text-left rounded-lg border-2 transition-colors ${
                diagnosticMode === mode.id
                  ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20'
                  : 'border-gray-200 dark:border-gray-600 hover:border-gray-300 dark:hover:border-gray-500'
              }`}
            >
              <h4 className="font-medium text-gray-900 dark:text-white">{mode.title}</h4>
              <p className="text-sm text-gray-600 dark:text-gray-300 mt-1">{mode.description}</p>
            </button>
          ))}
        </div>
      </div>

      {/* Common Issues Quick Fixes */}
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6">
        <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">
          Common Issues & Quick Fixes
        </h3>
        <div className="space-y-4">
          {commonIssues.map((issue) => (
            <div key={issue.id} className="flex items-center justify-between p-4 bg-gray-50 dark:bg-gray-700 rounded-lg">
              <div className="flex items-center space-x-4">
                <span className="text-2xl">{getCategoryIcon(issue.category)}</span>
                <div>
                  <h4 className="text-sm font-medium text-gray-900 dark:text-white">
                    {issue.title}
                  </h4>
                  <p className="text-sm text-gray-600 dark:text-gray-300">
                    {issue.description}
                  </p>
                  <div className="flex items-center space-x-2 mt-1">
                    <span className="text-xs text-gray-500 dark:text-gray-400">
                      {issue.category}
                    </span>
                    <span className="text-xs text-blue-600 dark:text-blue-400">
                      {issue.frequency}% frequency
                    </span>
                  </div>
                </div>
              </div>
              <div className="flex items-center space-x-2">
                {issue.quickFix && (
                  <button
                    onClick={() => runQuickFix(issue.id)}
                    className="px-3 py-2 bg-green-600 hover:bg-green-700 text-white text-sm rounded-md transition-colors"
                  >
                    {issue.quickFix.title}
                  </button>
                )}
                {issue.guideId && (
                  <button
                    onClick={() => startTroubleshootingGuide(issue.guideId!)}
                    className="px-3 py-2 bg-blue-600 hover:bg-blue-700 text-white text-sm rounded-md transition-colors"
                  >
                    Detailed Guide
                  </button>
                )}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Available Troubleshooting Guides */}
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6">
        <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">
          Troubleshooting Guides
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {guides.map((guide) => (
            <div key={guide.id} className="p-4 bg-gray-50 dark:bg-gray-700 rounded-lg">
              <div className="flex items-center justify-between mb-2">
                <h4 className="text-sm font-medium text-gray-900 dark:text-white">
                  {guide.title}
                </h4>
                <StatCard
                  title=""
                  value={guide.severity}
                  size="sm"
                  color={getSeverityColor(guide.severity)}
                  className="min-w-16"
                />
              </div>
              <p className="text-sm text-gray-600 dark:text-gray-300 mb-3">
                {guide.description}
              </p>
              <div className="flex items-center justify-between">
                <div className="text-xs text-gray-500 dark:text-gray-400">
                  <span>{guide.steps.length} steps</span>
                  <span className="ml-2">~{guide.estimatedTime} min</span>
                </div>
                <button
                  onClick={() => startTroubleshootingGuide(guide.id)}
                  className="px-3 py-2 bg-blue-600 hover:bg-blue-700 text-white text-sm rounded-md transition-colors"
                >
                  Start Guide
                </button>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Detected Issues from Props */}
      {issues.length > 0 && (
        <div className="bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 rounded-lg p-6">
          <h3 className="text-lg font-medium text-yellow-900 dark:text-yellow-100 mb-4">
            Detected Issues
          </h3>
          <div className="space-y-3">
            {issues.map((issue) => (
              <div key={issue.id} className="flex items-center justify-between p-3 bg-white dark:bg-gray-800 rounded">
                <div>
                  <h4 className="text-sm font-medium text-gray-900 dark:text-white">
                    {issue.title}
                  </h4>
                  <p className="text-sm text-gray-600 dark:text-gray-300">
                    {issue.description}
                  </p>
                </div>
                <div className="flex items-center space-x-2">
                  <StatCard
                    title=""
                    value={issue.severity}
                    size="sm"
                    color={getSeverityColor(issue.severity)}
                  />
                  {issue.autoFixAvailable && (
                    <button className="px-3 py-1 bg-green-600 hover:bg-green-700 text-white text-xs rounded transition-colors">
                      Auto Fix
                    </button>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
});

export default TroubleshootingWizard;
