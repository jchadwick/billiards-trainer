/**
 * Event timeline component showing chronological system events and user actions
 * Provides filterable timeline view of system activity
 */

import React, { useState, useEffect, useMemo } from 'react';
import { observer } from 'mobx-react-lite';

export interface SystemEvent {
  id: string;
  timestamp: Date;
  type: 'system' | 'user' | 'game' | 'calibration' | 'error' | 'performance';
  category: string;
  title: string;
  description: string;
  metadata?: Record<string, any>;
  severity?: 'low' | 'medium' | 'high' | 'critical';
  userId?: string;
  sessionId?: string;
}

export const EventTimeline: React.FC = observer(() => {
  const [events, setEvents] = useState<SystemEvent[]>([]);
  const [filteredEvents, setFilteredEvents] = useState<SystemEvent[]>([]);
  const [filterType, setFilterType] = useState<string>('ALL');
  const [filterSeverity, setFilterSeverity] = useState<string>('ALL');
  const [searchTerm, setSearchTerm] = useState<string>('');
  const [timeRange, setTimeRange] = useState<string>('24h');
  const [selectedEvent, setSelectedEvent] = useState<SystemEvent | null>(null);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    generateMockEvents();
    if (autoRefresh) {
      const interval = setInterval(addNewEvents, 15000); // Add new events every 15 seconds
      return () => clearInterval(interval);
    }
  }, [autoRefresh]);

  const generateMockEvents = () => {
    const eventTypes = ['system', 'user', 'game', 'calibration', 'error', 'performance'] as const;
    const severities = ['low', 'medium', 'high', 'critical'] as const;

    const sampleEvents = {
      system: [
        { title: 'System Started', description: 'Billiards trainer system initialized successfully', category: 'startup' },
        { title: 'Configuration Loaded', description: 'System configuration loaded from config.json', category: 'config' },
        { title: 'Database Connected', description: 'Successfully connected to PostgreSQL database', category: 'database' },
        { title: 'Service Restart', description: 'Vision service restarted due to memory usage', category: 'service' },
        { title: 'Health Check', description: 'Periodic health check completed successfully', category: 'monitoring' },
      ],
      user: [
        { title: 'User Login', description: 'User admin logged in successfully', category: 'authentication' },
        { title: 'User Logout', description: 'User session ended normally', category: 'authentication' },
        { title: 'Settings Updated', description: 'System settings modified by user', category: 'configuration' },
        { title: 'Manual Calibration', description: 'User initiated manual calibration process', category: 'calibration' },
        { title: 'Game Started', description: 'New game session started by user', category: 'gameplay' },
      ],
      game: [
        { title: 'Game Session Started', description: 'New billiards game session initiated', category: 'session' },
        { title: 'Shot Detected', description: 'Ball movement detected, analyzing trajectory', category: 'tracking' },
        { title: 'Pocket Made', description: 'Ball successfully potted in corner pocket', category: 'scoring' },
        { title: 'Game Ended', description: 'Game session completed, final score recorded', category: 'session' },
        { title: 'Statistics Updated', description: 'Player statistics updated with session data', category: 'analytics' },
      ],
      calibration: [
        { title: 'Calibration Started', description: 'Automatic calibration process initiated', category: 'auto' },
        { title: 'Calibration Point Added', description: 'Calibration reference point captured', category: 'manual' },
        { title: 'Calibration Completed', description: 'Camera-projector calibration completed successfully', category: 'completion' },
        { title: 'Calibration Failed', description: 'Calibration process failed due to insufficient points', category: 'error' },
        { title: 'Calibration Drift', description: 'Calibration accuracy degraded, recalibration recommended', category: 'maintenance' },
      ],
      error: [
        { title: 'Camera Error', description: 'Failed to capture frame from camera device', category: 'hardware' },
        { title: 'Network Error', description: 'WebSocket connection lost, attempting reconnection', category: 'network' },
        { title: 'Processing Error', description: 'Frame processing failed due to invalid input', category: 'processing' },
        { title: 'Database Error', description: 'Failed to save game data to database', category: 'persistence' },
        { title: 'API Error', description: 'External API request failed after 3 retries', category: 'external' },
      ],
      performance: [
        { title: 'High CPU Usage', description: 'CPU usage exceeded 80% threshold', category: 'resources' },
        { title: 'Memory Warning', description: 'Memory usage approaching system limits', category: 'resources' },
        { title: 'Slow Processing', description: 'Frame processing time exceeded target latency', category: 'latency' },
        { title: 'Performance Improved', description: 'System performance returned to normal levels', category: 'optimization' },
        { title: 'Cache Cleared', description: 'System cache cleared to improve performance', category: 'optimization' },
      ],
    };

    const generateRandomEvent = (baseTime: Date = new Date()): SystemEvent => {
      const type = eventTypes[Math.floor(Math.random() * eventTypes.length)];
      const eventTemplates = sampleEvents[type];
      const template = eventTemplates[Math.floor(Math.random() * eventTemplates.length)];
      const severity = severities[Math.floor(Math.random() * severities.length)];

      return {
        id: `event-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
        timestamp: new Date(baseTime.getTime() - Math.random() * 86400000), // Random time in last 24 hours
        type,
        category: template.category,
        title: template.title,
        description: template.description,
        severity,
        metadata: {
          systemLoad: Math.floor(Math.random() * 100),
          memoryUsage: Math.floor(Math.random() * 100),
          activeConnections: Math.floor(Math.random() * 10),
        },
        userId: type === 'user' ? `user-${Math.random().toString(36).substr(2, 8)}` : undefined,
        sessionId: `session-${Math.random().toString(36).substr(2, 8)}`,
      };
    };

    // Generate initial events
    const initialEvents = Array.from({ length: 50 }, () => generateRandomEvent())
      .sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime());

    setEvents(initialEvents);
    setLoading(false);
  };

  const addNewEvents = () => {
    const newEvents = Array.from({ length: Math.floor(Math.random() * 3) + 1 }, () =>
      generateRandomEvent(new Date())
    );

    setEvents(prevEvents => {
      const allEvents = [...newEvents, ...prevEvents]
        .sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime())
        .slice(0, 500); // Keep only last 500 events

      return allEvents;
    });
  };

  const generateRandomEvent = (baseTime: Date): SystemEvent => {
    const eventTypes = ['system', 'user', 'game', 'calibration', 'error', 'performance'] as const;
    const severities = ['low', 'medium', 'high', 'critical'] as const;

    const sampleEvents = {
      system: [
        { title: 'Health Check', description: 'Periodic system health check completed', category: 'monitoring' },
        { title: 'Service Status', description: 'All services running normally', category: 'monitoring' },
      ],
      user: [
        { title: 'User Activity', description: 'User performed system action', category: 'interaction' },
      ],
      game: [
        { title: 'Shot Analysis', description: 'Ball trajectory analyzed and recorded', category: 'tracking' },
        { title: 'Frame Processed', description: 'Video frame processed for object detection', category: 'processing' },
      ],
      calibration: [
        { title: 'Auto Adjustment', description: 'Minor calibration adjustment applied', category: 'auto' },
      ],
      error: [
        { title: 'Minor Error', description: 'Recoverable error occurred and was handled', category: 'recovery' },
      ],
      performance: [
        { title: 'Performance Update', description: 'System performance metrics updated', category: 'metrics' },
      ],
    };

    const type = eventTypes[Math.floor(Math.random() * eventTypes.length)];
    const eventTemplates = sampleEvents[type];
    const template = eventTemplates[Math.floor(Math.random() * eventTemplates.length)];
    const severity = severities[Math.floor(Math.random() * severities.length)];

    return {
      id: `event-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      timestamp: new Date(baseTime.getTime() - Math.random() * 300000), // Random time in last 5 minutes
      type,
      category: template.category,
      title: template.title,
      description: template.description,
      severity,
      metadata: {
        systemLoad: Math.floor(Math.random() * 100),
        memoryUsage: Math.floor(Math.random() * 100),
        activeConnections: Math.floor(Math.random() * 10),
      },
      sessionId: `session-${Math.random().toString(36).substr(2, 8)}`,
    };
  };

  // Apply filters
  useEffect(() => {
    let filtered = events;

    // Filter by time range
    const now = new Date();
    const timeRanges = {
      '1h': 60 * 60 * 1000,
      '6h': 6 * 60 * 60 * 1000,
      '24h': 24 * 60 * 60 * 1000,
      '7d': 7 * 24 * 60 * 60 * 1000,
      '30d': 30 * 24 * 60 * 60 * 1000,
    };

    if (timeRange !== 'ALL') {
      const cutoff = new Date(now.getTime() - timeRanges[timeRange as keyof typeof timeRanges]);
      filtered = filtered.filter(event => event.timestamp >= cutoff);
    }

    // Filter by type
    if (filterType !== 'ALL') {
      filtered = filtered.filter(event => event.type === filterType);
    }

    // Filter by severity
    if (filterSeverity !== 'ALL') {
      filtered = filtered.filter(event => event.severity === filterSeverity);
    }

    // Filter by search term
    if (searchTerm) {
      const term = searchTerm.toLowerCase();
      filtered = filtered.filter(event =>
        event.title.toLowerCase().includes(term) ||
        event.description.toLowerCase().includes(term) ||
        event.category.toLowerCase().includes(term)
      );
    }

    setFilteredEvents(filtered);
  }, [events, filterType, filterSeverity, searchTerm, timeRange]);

  const eventTypes = useMemo(() => {
    const typeSet = new Set(events.map(event => event.type));
    return Array.from(typeSet).sort();
  }, [events]);

  const getEventIcon = (type: SystemEvent['type']): string => {
    switch (type) {
      case 'system': return '⚙️';
      case 'user': return '👤';
      case 'game': return '🎱';
      case 'calibration': return '🎯';
      case 'error': return '❌';
      case 'performance': return '📊';
      default: return '📝';
    }
  };

  const getSeverityColor = (severity: SystemEvent['severity']): string => {
    switch (severity) {
      case 'low': return 'text-green-600 bg-green-50 border-green-200';
      case 'medium': return 'text-blue-600 bg-blue-50 border-blue-200';
      case 'high': return 'text-yellow-600 bg-yellow-50 border-yellow-200';
      case 'critical': return 'text-red-600 bg-red-50 border-red-200';
      default: return 'text-gray-600 bg-gray-50 border-gray-200';
    }
  };

  const formatTimeAgo = (timestamp: Date): string => {
    const now = new Date();
    const diff = now.getTime() - timestamp.getTime();
    const minutes = Math.floor(diff / 60000);
    const hours = Math.floor(diff / 3600000);
    const days = Math.floor(diff / 86400000);

    if (days > 0) {
      return `${days}d ago`;
    } else if (hours > 0) {
      return `${hours}h ago`;
    } else if (minutes > 0) {
      return `${minutes}m ago`;
    } else {
      return 'Just now';
    }
  };

  return (
    <div className="space-y-6">
      {/* Controls */}
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-4">
        <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between space-y-4 lg:space-y-0">
          <div className="flex flex-wrap items-center space-x-4">
            {/* Search */}
            <div className="flex-1 min-w-64">
              <input
                type="text"
                placeholder="Search events..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
              />
            </div>

            {/* Time range filter */}
            <select
              value={timeRange}
              onChange={(e) => setTimeRange(e.target.value)}
              className="px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
            >
              <option value="1h">Last Hour</option>
              <option value="6h">Last 6 Hours</option>
              <option value="24h">Last 24 Hours</option>
              <option value="7d">Last 7 Days</option>
              <option value="30d">Last 30 Days</option>
              <option value="ALL">All Time</option>
            </select>

            {/* Type filter */}
            <select
              value={filterType}
              onChange={(e) => setFilterType(e.target.value)}
              className="px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
            >
              <option value="ALL">All Types</option>
              {eventTypes.map(type => (
                <option key={type} value={type}>{type.charAt(0).toUpperCase() + type.slice(1)}</option>
              ))}
            </select>

            {/* Severity filter */}
            <select
              value={filterSeverity}
              onChange={(e) => setFilterSeverity(e.target.value)}
              className="px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
            >
              <option value="ALL">All Severities</option>
              <option value="low">Low</option>
              <option value="medium">Medium</option>
              <option value="high">High</option>
              <option value="critical">Critical</option>
            </select>
          </div>

          <div className="flex items-center space-x-4">
            {/* Auto-refresh toggle */}
            <label className="flex items-center cursor-pointer">
              <input
                type="checkbox"
                checked={autoRefresh}
                onChange={(e) => setAutoRefresh(e.target.checked)}
                className="sr-only"
              />
              <div className={`relative w-10 h-6 transition-colors duration-200 ease-in-out rounded-full ${
                autoRefresh ? 'bg-blue-600' : 'bg-gray-200 dark:bg-gray-600'
              }`}>
                <div className={`absolute left-1 top-1 bg-white w-4 h-4 rounded-full transition-transform duration-200 ease-in-out ${
                  autoRefresh ? 'transform translate-x-4' : ''
                }`} />
              </div>
              <span className="ml-2 text-sm text-gray-600 dark:text-gray-300">
                Auto-refresh
              </span>
            </label>
          </div>
        </div>
      </div>

      {/* Event Timeline */}
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700">
        <div className="p-4 border-b border-gray-200 dark:border-gray-700">
          <h3 className="text-lg font-medium text-gray-900 dark:text-white">
            Event Timeline ({filteredEvents.length})
          </h3>
        </div>

        <div className="max-h-96 overflow-y-auto">
          {loading ? (
            <div className="p-8 text-center">
              <div className="animate-spin h-8 w-8 border-4 border-blue-500 border-t-transparent rounded-full mx-auto mb-4"></div>
              <p className="text-gray-500 dark:text-gray-400">Loading events...</p>
            </div>
          ) : filteredEvents.length === 0 ? (
            <div className="p-8 text-center">
              <p className="text-gray-500 dark:text-gray-400">No events found matching your criteria.</p>
            </div>
          ) : (
            <div className="relative">
              {/* Timeline line */}
              <div className="absolute left-8 top-0 bottom-0 w-0.5 bg-gray-200 dark:bg-gray-600"></div>

              <div className="space-y-1">
                {filteredEvents.map((event, index) => (
                  <div
                    key={event.id}
                    className="relative flex items-start space-x-4 p-4 hover:bg-gray-50 dark:hover:bg-gray-700 cursor-pointer"
                    onClick={() => setSelectedEvent(event)}
                  >
                    {/* Timeline dot */}
                    <div className={`relative z-10 flex items-center justify-center w-8 h-8 rounded-full border-2 bg-white dark:bg-gray-800 ${
                      event.severity === 'critical' ? 'border-red-500' :
                      event.severity === 'high' ? 'border-yellow-500' :
                      event.severity === 'medium' ? 'border-blue-500' :
                      'border-green-500'
                    }`}>
                      <span className="text-sm">{getEventIcon(event.type)}</span>
                    </div>

                    <div className="flex-1 min-w-0">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center space-x-2">
                          <h4 className="text-sm font-medium text-gray-900 dark:text-white">
                            {event.title}
                          </h4>
                          <span className={`px-2 py-1 text-xs font-medium rounded-full border ${getSeverityColor(event.severity)}`}>
                            {event.severity}
                          </span>
                          <span className="px-2 py-1 text-xs font-medium rounded-full bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300">
                            {event.type}
                          </span>
                        </div>
                        <span className="text-sm text-gray-500 dark:text-gray-400">
                          {formatTimeAgo(event.timestamp)}
                        </span>
                      </div>
                      <p className="mt-1 text-sm text-gray-600 dark:text-gray-300">
                        {event.description}
                      </p>
                      <div className="mt-1 text-xs text-gray-500 dark:text-gray-400">
                        Category: {event.category} • Session: {event.sessionId?.slice(-8)}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Event Detail Modal */}
      {selectedEvent && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
          <div className="bg-white dark:bg-gray-800 rounded-lg max-w-2xl w-full max-h-96 overflow-y-auto">
            <div className="p-6">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-medium text-gray-900 dark:text-white">
                  Event Details
                </h3>
                <button
                  onClick={() => setSelectedEvent(null)}
                  className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-300"
                >
                  ✕
                </button>
              </div>

              <div className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="text-sm font-medium text-gray-500 dark:text-gray-400">Type</label>
                    <div className="mt-1 flex items-center">
                      <span className="mr-2">{getEventIcon(selectedEvent.type)}</span>
                      <span className="text-sm text-gray-900 dark:text-white">{selectedEvent.type}</span>
                    </div>
                  </div>
                  <div>
                    <label className="text-sm font-medium text-gray-500 dark:text-gray-400">Severity</label>
                    <div className={`mt-1 px-2 py-1 text-sm font-medium rounded border ${getSeverityColor(selectedEvent.severity)} inline-block`}>
                      {selectedEvent.severity}
                    </div>
                  </div>
                  <div>
                    <label className="text-sm font-medium text-gray-500 dark:text-gray-400">Timestamp</label>
                    <p className="mt-1 text-sm text-gray-900 dark:text-white">{selectedEvent.timestamp.toLocaleString()}</p>
                  </div>
                  <div>
                    <label className="text-sm font-medium text-gray-500 dark:text-gray-400">Category</label>
                    <p className="mt-1 text-sm text-gray-900 dark:text-white">{selectedEvent.category}</p>
                  </div>
                </div>

                <div>
                  <label className="text-sm font-medium text-gray-500 dark:text-gray-400">Title</label>
                  <p className="mt-1 text-sm text-gray-900 dark:text-white">{selectedEvent.title}</p>
                </div>

                <div>
                  <label className="text-sm font-medium text-gray-500 dark:text-gray-400">Description</label>
                  <p className="mt-1 text-sm text-gray-900 dark:text-white">{selectedEvent.description}</p>
                </div>

                <div>
                  <label className="text-sm font-medium text-gray-500 dark:text-gray-400">Session ID</label>
                  <p className="mt-1 text-sm text-gray-900 dark:text-white font-mono">{selectedEvent.sessionId}</p>
                </div>

                {selectedEvent.metadata && (
                  <div>
                    <label className="text-sm font-medium text-gray-500 dark:text-gray-400">Metadata</label>
                    <pre className="mt-1 text-sm text-gray-900 dark:text-white bg-gray-100 dark:bg-gray-700 p-3 rounded overflow-x-auto">
                      {JSON.stringify(selectedEvent.metadata, null, 2)}
                    </pre>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
});

export default EventTimeline;
