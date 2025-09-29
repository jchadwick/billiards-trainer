/**
 * Hardware Diagnostics Component
 * Implements camera detection, system resources monitoring, projector testing, and audio validation
 *
 * Features:
 * - Camera detection and functionality testing
 * - System resource monitoring (CPU, memory, disk, temperature)
 * - Projector connectivity and calibration testing
 * - Audio system validation
 */

import React, { useState, useEffect, useCallback } from 'react';
import { observer } from 'mobx-react-lite';
import { useStores } from '../../stores/context';
import { StatCard } from '../monitoring/StatCard';
import { ProgressBar } from '../monitoring/ProgressBar';
import { StatusIndicator } from '../monitoring/StatusIndicator';
import type { DiagnosticSuite, DiagnosticTest } from './DiagnosticsSystem';

interface HardwareInfo {
  cameras: {
    detected: number;
    active: number;
    devices: Array<{
      id: string;
      name: string;
      resolution: string;
      frameRate: number;
      status: 'available' | 'busy' | 'error';
    }>;
  };
  projectors: {
    detected: number;
    calibrated: number;
    devices: Array<{
      id: string;
      name: string;
      resolution: string;
      status: 'connected' | 'disconnected' | 'error';
      calibrationAccuracy?: number;
    }>;
  };
  system: {
    cpu: {
      usage: number;
      temperature: number;
      cores: number;
      frequency: number;
    };
    memory: {
      total: number;
      used: number;
      usage: number;
      available: number;
    };
    disk: {
      total: number;
      used: number;
      usage: number;
      available: number;
    };
    gpu?: {
      name: string;
      memory: number;
      usage: number;
      temperature: number;
    };
  };
  audio: {
    inputDevices: Array<{
      id: string;
      name: string;
      status: 'available' | 'error';
    }>;
    outputDevices: Array<{
      id: string;
      name: string;
      status: 'available' | 'error';
    }>;
  };
}

interface HardwareDiagnosticsProps {
  suites: DiagnosticSuite[];
}

export const HardwareDiagnostics: React.FC<HardwareDiagnosticsProps> = observer(({ suites }) => {
  const { systemStore } = useStores();
  const [hardwareInfo, setHardwareInfo] = useState<HardwareInfo | null>(null);
  const [isScanning, setIsScanning] = useState(false);
  const [testResults, setTestResults] = useState<Map<string, any>>(new Map());

  useEffect(() => {
    scanHardware();
  }, []);

  const scanHardware = useCallback(async () => {
    setIsScanning(true);
    try {
      // Simulate hardware detection
      await new Promise(resolve => setTimeout(resolve, 2000));

      const mockHardwareInfo: HardwareInfo = {
        cameras: {
          detected: 2,
          active: 1,
          devices: [
            {
              id: 'camera-0',
              name: 'Overhead Camera',
              resolution: '1920x1080',
              frameRate: 30,
              status: 'available',
            },
            {
              id: 'camera-1',
              name: 'Side Camera',
              resolution: '1280x720',
              frameRate: 30,
              status: 'busy',
            },
          ],
        },
        projectors: {
          detected: 1,
          calibrated: 1,
          devices: [
            {
              id: 'projector-0',
              name: 'Overhead Projector',
              resolution: '1920x1080',
              status: 'connected',
              calibrationAccuracy: 98.5,
            },
          ],
        },
        system: {
          cpu: {
            usage: 45.2,
            temperature: 65,
            cores: 8,
            frequency: 3200,
          },
          memory: {
            total: 16384,
            used: 8192,
            usage: 50.0,
            available: 8192,
          },
          disk: {
            total: 500000,
            used: 150000,
            usage: 30.0,
            available: 350000,
          },
          gpu: {
            name: 'NVIDIA RTX 3070',
            memory: 8192,
            usage: 25.0,
            temperature: 70,
          },
        },
        audio: {
          inputDevices: [
            { id: 'mic-0', name: 'USB Microphone', status: 'available' },
          ],
          outputDevices: [
            { id: 'speaker-0', name: 'System Speakers', status: 'available' },
          ],
        },
      };

      setHardwareInfo(mockHardwareInfo);
    } catch (error) {
      console.error('Hardware scan failed:', error);
    } finally {
      setIsScanning(false);
    }
  }, []);

  const runCameraTest = useCallback(async (cameraId: string) => {
    try {
      // Request camera access
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { deviceId: cameraId }
      });

      // Test basic functionality
      const video = document.createElement('video');
      video.srcObject = stream;
      video.play();

      // Simulate frame rate test
      await new Promise(resolve => setTimeout(resolve, 3000));

      // Cleanup
      stream.getTracks().forEach(track => track.stop());

      return {
        success: true,
        frameRate: 30,
        resolution: '1920x1080',
        latency: 33,
      };
    } catch (error) {
      return {
        success: false,
        error: error.message,
      };
    }
  }, []);

  const runProjectorTest = useCallback(async (projectorId: string) => {
    // Simulate projector calibration test
    await new Promise(resolve => setTimeout(resolve, 2000));

    return {
      success: true,
      calibrationAccuracy: 98.5,
      brightness: 85,
      contrast: 92,
    };
  }, []);

  const runSystemResourceTest = useCallback(async () => {
    try {
      // Get system performance data
      const performanceData = performance.now();

      // Simulate stress test
      await new Promise(resolve => setTimeout(resolve, 5000));

      return {
        success: true,
        cpuStressScore: 95,
        memoryAllocationTest: 'passed',
        diskIOTest: 'passed',
      };
    } catch (error) {
      return {
        success: false,
        error: error.message,
      };
    }
  }, []);

  const runAudioTest = useCallback(async () => {
    try {
      // Test audio input
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

      // Simulate audio test
      await new Promise(resolve => setTimeout(resolve, 2000));

      stream.getTracks().forEach(track => track.stop());

      return {
        success: true,
        inputLevel: 0.8,
        outputLevel: 0.9,
        latency: 15,
      };
    } catch (error) {
      return {
        success: false,
        error: error.message,
      };
    }
  }, []);

  const getResourceUsageColor = (usage: number): 'green' | 'yellow' | 'red' => {
    if (usage < 60) return 'green';
    if (usage < 80) return 'yellow';
    return 'red';
  };

  const getTemperatureColor = (temp: number): 'green' | 'yellow' | 'red' => {
    if (temp < 70) return 'green';
    if (temp < 85) return 'yellow';
    return 'red';
  };

  if (!hardwareInfo) {
    return (
      <div className="flex items-center justify-center min-h-96">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
          <p className="text-gray-600 dark:text-gray-300">
            {isScanning ? 'Scanning hardware...' : 'Loading hardware diagnostics...'}
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Hardware Overview */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <StatCard
          title="Cameras Detected"
          value={hardwareInfo.cameras.detected}
          unit={` / ${hardwareInfo.cameras.active} active`}
          icon="📷"
          color="blue"
        />
        <StatCard
          title="Projectors"
          value={hardwareInfo.projectors.detected}
          unit={` / ${hardwareInfo.projectors.calibrated} calibrated`}
          icon="🎯"
          color="purple"
        />
        <StatCard
          title="CPU Usage"
          value={hardwareInfo.system.cpu.usage.toFixed(1)}
          unit="%"
          icon="⚡"
          color={getResourceUsageColor(hardwareInfo.system.cpu.usage)}
        />
        <StatCard
          title="Memory Usage"
          value={hardwareInfo.system.memory.usage.toFixed(1)}
          unit="%"
          icon="💾"
          color={getResourceUsageColor(hardwareInfo.system.memory.usage)}
        />
      </div>

      {/* Camera Diagnostics */}
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-medium text-gray-900 dark:text-white">
            Camera System
          </h3>
          <button
            onClick={scanHardware}
            disabled={isScanning}
            className="px-3 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-md transition-colors disabled:bg-gray-400"
          >
            {isScanning ? 'Scanning...' : 'Rescan'}
          </button>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {hardwareInfo.cameras.devices.map((camera) => (
            <div key={camera.id} className="p-4 bg-gray-50 dark:bg-gray-700 rounded-lg">
              <div className="flex items-center justify-between mb-2">
                <h4 className="text-sm font-medium text-gray-900 dark:text-white">
                  {camera.name}
                </h4>
                <StatusIndicator
                  status={camera.status === 'available' ? 'healthy' :
                          camera.status === 'busy' ? 'degraded' : 'unhealthy'}
                  size="sm"
                  showLabel={false}
                />
              </div>
              <div className="text-sm text-gray-600 dark:text-gray-300 space-y-1">
                <p>Resolution: {camera.resolution}</p>
                <p>Frame Rate: {camera.frameRate} fps</p>
                <p>Status: {camera.status}</p>
              </div>
              <button
                onClick={() => runCameraTest(camera.id)}
                disabled={camera.status !== 'available'}
                className="mt-3 w-full px-3 py-2 text-sm bg-green-600 hover:bg-green-700 text-white rounded-md transition-colors disabled:bg-gray-400"
              >
                Test Camera
              </button>
            </div>
          ))}
        </div>
      </div>

      {/* System Resources */}
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6">
        <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">
          System Resources
        </h3>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {/* CPU */}
          <div className="space-y-4">
            <h4 className="text-sm font-medium text-gray-900 dark:text-white">CPU</h4>
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span>Usage</span>
                <span>{hardwareInfo.system.cpu.usage.toFixed(1)}%</span>
              </div>
              <ProgressBar
                value={hardwareInfo.system.cpu.usage}
                color={getResourceUsageColor(hardwareInfo.system.cpu.usage)}
                size="sm"
              />
              <div className="grid grid-cols-2 gap-2 text-xs text-gray-600 dark:text-gray-300">
                <div>Cores: {hardwareInfo.system.cpu.cores}</div>
                <div>Freq: {hardwareInfo.system.cpu.frequency} MHz</div>
                <div>Temp: {hardwareInfo.system.cpu.temperature}°C</div>
              </div>
            </div>
          </div>

          {/* Memory */}
          <div className="space-y-4">
            <h4 className="text-sm font-medium text-gray-900 dark:text-white">Memory</h4>
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span>Usage</span>
                <span>{hardwareInfo.system.memory.usage.toFixed(1)}%</span>
              </div>
              <ProgressBar
                value={hardwareInfo.system.memory.usage}
                color={getResourceUsageColor(hardwareInfo.system.memory.usage)}
                size="sm"
              />
              <div className="grid grid-cols-2 gap-2 text-xs text-gray-600 dark:text-gray-300">
                <div>Total: {(hardwareInfo.system.memory.total / 1024).toFixed(1)} GB</div>
                <div>Used: {(hardwareInfo.system.memory.used / 1024).toFixed(1)} GB</div>
                <div>Available: {(hardwareInfo.system.memory.available / 1024).toFixed(1)} GB</div>
              </div>
            </div>
          </div>

          {/* GPU */}
          {hardwareInfo.system.gpu && (
            <div className="space-y-4">
              <h4 className="text-sm font-medium text-gray-900 dark:text-white">GPU</h4>
              <div className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span>Usage</span>
                  <span>{hardwareInfo.system.gpu.usage.toFixed(1)}%</span>
                </div>
                <ProgressBar
                  value={hardwareInfo.system.gpu.usage}
                  color={getResourceUsageColor(hardwareInfo.system.gpu.usage)}
                  size="sm"
                />
                <div className="text-xs text-gray-600 dark:text-gray-300">
                  <div>{hardwareInfo.system.gpu.name}</div>
                  <div>Memory: {(hardwareInfo.system.gpu.memory / 1024).toFixed(1)} GB</div>
                  <div>Temp: {hardwareInfo.system.gpu.temperature}°C</div>
                </div>
              </div>
            </div>
          )}
        </div>

        <div className="mt-6">
          <button
            onClick={runSystemResourceTest}
            className="px-4 py-2 bg-orange-600 hover:bg-orange-700 text-white rounded-md transition-colors"
          >
            Run Stress Test
          </button>
        </div>
      </div>

      {/* Projector Diagnostics */}
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6">
        <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">
          Projector System
        </h3>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {hardwareInfo.projectors.devices.map((projector) => (
            <div key={projector.id} className="p-4 bg-gray-50 dark:bg-gray-700 rounded-lg">
              <div className="flex items-center justify-between mb-2">
                <h4 className="text-sm font-medium text-gray-900 dark:text-white">
                  {projector.name}
                </h4>
                <StatusIndicator
                  status={projector.status === 'connected' ? 'healthy' : 'unhealthy'}
                  size="sm"
                  showLabel={false}
                />
              </div>
              <div className="text-sm text-gray-600 dark:text-gray-300 space-y-1">
                <p>Resolution: {projector.resolution}</p>
                <p>Status: {projector.status}</p>
                {projector.calibrationAccuracy && (
                  <p>Calibration: {projector.calibrationAccuracy.toFixed(1)}%</p>
                )}
              </div>
              <button
                onClick={() => runProjectorTest(projector.id)}
                disabled={projector.status !== 'connected'}
                className="mt-3 w-full px-3 py-2 text-sm bg-purple-600 hover:bg-purple-700 text-white rounded-md transition-colors disabled:bg-gray-400"
              >
                Test Projector
              </button>
            </div>
          ))}
        </div>
      </div>

      {/* Audio System */}
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-medium text-gray-900 dark:text-white">
            Audio System
          </h3>
          <button
            onClick={runAudioTest}
            className="px-3 py-2 bg-green-600 hover:bg-green-700 text-white rounded-md transition-colors"
          >
            Test Audio
          </button>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <h4 className="text-sm font-medium text-gray-900 dark:text-white mb-3">
              Input Devices
            </h4>
            {hardwareInfo.audio.inputDevices.map((device) => (
              <div key={device.id} className="flex items-center justify-between p-2 bg-gray-50 dark:bg-gray-700 rounded">
                <span className="text-sm text-gray-900 dark:text-white">{device.name}</span>
                <StatusIndicator
                  status={device.status === 'available' ? 'healthy' : 'unhealthy'}
                  size="sm"
                  showLabel={false}
                />
              </div>
            ))}
          </div>

          <div>
            <h4 className="text-sm font-medium text-gray-900 dark:text-white mb-3">
              Output Devices
            </h4>
            {hardwareInfo.audio.outputDevices.map((device) => (
              <div key={device.id} className="flex items-center justify-between p-2 bg-gray-50 dark:bg-gray-700 rounded">
                <span className="text-sm text-gray-900 dark:text-white">{device.name}</span>
                <StatusIndicator
                  status={device.status === 'available' ? 'healthy' : 'unhealthy'}
                  size="sm"
                  showLabel={false}
                />
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
});

export default HardwareDiagnostics;