/**
 * Integration test utilities to verify frontend-backend communication
 */

import { apiClient } from '../api/client';
import { rootStore } from '../stores/RootStore';

export interface IntegrationTestResult {
  test: string;
  success: boolean;
  message: string;
  details?: any;
}

export class IntegrationTester {
  private results: IntegrationTestResult[] = [];

  async runAllTests(): Promise<IntegrationTestResult[]> {
    this.results = [];

    await this.testApiHealth();
    await this.testSystemMetrics();
    await this.testConfigurationAccess();
    await this.testVideoStreamStatus();
    await this.testWebSocketConnection();
    await this.testStoreIntegration();

    return this.results;
  }

  private async testApiHealth(): Promise<void> {
    try {
      const response = await apiClient.getHealth(false, false);

      this.results.push({
        test: 'API Health Check',
        success: response.success,
        message: response.success
          ? `Backend is ${response.data?.status || 'responsive'}`
          : `Health check failed: ${response.error}`,
        details: response.data
      });
    } catch (error) {
      this.results.push({
        test: 'API Health Check',
        success: false,
        message: `Network error: ${error instanceof Error ? error.message : 'Unknown error'}`,
      });
    }
  }

  private async testSystemMetrics(): Promise<void> {
    try {
      const response = await apiClient.getMetrics();

      this.results.push({
        test: 'System Metrics',
        success: response.success,
        message: response.success
          ? `Metrics retrieved (CPU: ${response.data?.cpu_usage?.toFixed(1)}%)`
          : `Metrics failed: ${response.error}`,
        details: response.data
      });
    } catch (error) {
      this.results.push({
        test: 'System Metrics',
        success: false,
        message: `Metrics error: ${error instanceof Error ? error.message : 'Unknown error'}`,
      });
    }
  }

  private async testConfigurationAccess(): Promise<void> {
    try {
      const response = await apiClient.getConfiguration();

      this.results.push({
        test: 'Configuration Access',
        success: response.success,
        message: response.success
          ? `Configuration loaded (${Object.keys(response.data?.values || {}).length} sections)`
          : `Config failed: ${response.error}`,
        details: response.data
      });
    } catch (error) {
      this.results.push({
        test: 'Configuration Access',
        success: false,
        message: `Config error: ${error instanceof Error ? error.message : 'Unknown error'}`,
      });
    }
  }

  private async testVideoStreamStatus(): Promise<void> {
    try {
      const response = await apiClient.getStreamStatus();

      this.results.push({
        test: 'Video Stream Status',
        success: response.success,
        message: response.success
          ? `Stream status: ${response.data?.camera?.status || 'unknown'}`
          : `Stream status failed: ${response.error}`,
        details: response.data
      });
    } catch (error) {
      this.results.push({
        test: 'Video Stream Status',
        success: false,
        message: `Stream error: ${error instanceof Error ? error.message : 'Unknown error'}`,
      });
    }
  }

  private async testWebSocketConnection(): Promise<void> {
    try {
      // Test WebSocket connection through SystemStore
      const websocketUrl = 'ws://localhost:8080/api/v1/ws';
      const connectResult = await rootStore.system.connect(websocketUrl);

      if (connectResult.success) {
        // Wait a moment for connection to establish
        await new Promise(resolve => setTimeout(resolve, 1000));

        const isConnected = rootStore.system.status.isConnected;

        this.results.push({
          test: 'WebSocket Connection',
          success: isConnected,
          message: isConnected
            ? 'WebSocket connected successfully'
            : 'WebSocket connection failed',
          details: { status: rootStore.system.status.websocketStatus }
        });

        // Disconnect after test
        rootStore.system.disconnect();
      } else {
        this.results.push({
          test: 'WebSocket Connection',
          success: false,
          message: `WebSocket connection failed: ${connectResult.error}`,
        });
      }
    } catch (error) {
      this.results.push({
        test: 'WebSocket Connection',
        success: false,
        message: `WebSocket error: ${error instanceof Error ? error.message : 'Unknown error'}`,
      });
    }
  }

  private async testStoreIntegration(): Promise<void> {
    try {
      // Test SystemStore integration
      await rootStore.system.refreshMetrics();
      const hasMetrics = rootStore.system.lastMetrics !== null;

      // Test ConfigStore integration
      await rootStore.config.loadFromBackend();
      const configLoaded = Object.keys(rootStore.config.config).length > 0;

      // Test ConnectionStore integration
      await rootStore.connection.connect();
      const connectionWorks = rootStore.connection.state.isConnected;

      const allStoresWork = hasMetrics && configLoaded && connectionWorks;

      this.results.push({
        test: 'Store Integration',
        success: allStoresWork,
        message: allStoresWork
          ? 'All stores integrated successfully'
          : 'Some store integrations failed',
        details: {
          systemStore: hasMetrics,
          configStore: configLoaded,
          connectionStore: connectionWorks
        }
      });
    } catch (error) {
      this.results.push({
        test: 'Store Integration',
        success: false,
        message: `Store integration error: ${error instanceof Error ? error.message : 'Unknown error'}`,
      });
    }
  }

  getTestSummary(): { total: number; passed: number; failed: number; successRate: number } {
    const total = this.results.length;
    const passed = this.results.filter(r => r.success).length;
    const failed = total - passed;
    const successRate = total > 0 ? (passed / total) * 100 : 0;

    return { total, passed, failed, successRate };
  }

  printResults(): void {
    console.group('🧪 Integration Test Results');

    this.results.forEach(result => {
      const icon = result.success ? '✅' : '❌';
      console.log(`${icon} ${result.test}: ${result.message}`);
      if (result.details) {
        console.log('   Details:', result.details);
      }
    });

    const summary = this.getTestSummary();
    console.log(`\n📊 Summary: ${summary.passed}/${summary.total} tests passed (${summary.successRate.toFixed(1)}%)`);

    console.groupEnd();
  }
}

// Export a singleton instance
export const integrationTester = new IntegrationTester();
