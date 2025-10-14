import React, { useState, useEffect, useRef } from 'react';
import { observer } from 'mobx-react-lite';
import { createWebSocketClient, type WebSocketClient, type ConnectionState } from '../../services/websocket-client';
import type { WebSocketMessage, MessageType } from '../../types';
import { Card } from '../ui/Card';
import { Button } from '../ui/Button';
import { Checkbox } from '../ui/Checkbox';

interface EventLog {
  id: string;
  timestamp: Date;
  message: WebSocketMessage;
}

/**
 * WebSocket Event Monitor - Diagnostic page for viewing raw WebSocket events
 *
 * Displays all incoming WebSocket messages in real-time with:
 * - Raw JSON display
 * - Message filtering by type
 * - Connection statistics
 * - Auto-scroll option
 * - Message history management
 */
export const WebSocketEventMonitor: React.FC = observer(() => {
  const [wsClient, setWsClient] = useState<WebSocketClient | null>(null);
  const [connectionState, setConnectionState] = useState<ConnectionState>('disconnected');
  const [events, setEvents] = useState<EventLog[]>([]);
  const [filteredTypes, setFilteredTypes] = useState<Set<MessageType>>(new Set());
  const [autoScroll, setAutoScroll] = useState(true);
  const [maxEvents, setMaxEvents] = useState(100);
  const [isPaused, setIsPaused] = useState(false);
  const eventsContainerRef = useRef<HTMLDivElement>(null);
  const eventCountRef = useRef(0);

  // Initialize WebSocket client
  useEffect(() => {
    // Determine WebSocket URL from current location
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.hostname}:8000/api/v1/ws`;

    const client = createWebSocketClient({
      url: wsUrl,
      autoReconnect: true,
      maxReconnectAttempts: 10,
      reconnectDelay: 1000,
    });

    // Set up connection state handler
    client.onConnectionState((state) => {
      setConnectionState(state);
    });

    // Set up message handlers for all message types
    const messageTypes: MessageType[] = [
      'frame', 'state', 'trajectory', 'alert', 'config', 'metrics',
      'connection', 'ping', 'pong', 'subscribe', 'unsubscribe',
      'subscribed', 'unsubscribed', 'status', 'error'
    ];

    messageTypes.forEach(type => {
      client.on(type, (message) => {
        if (!isPaused) {
          eventCountRef.current += 1;
          const newEvent: EventLog = {
            id: `${Date.now()}-${eventCountRef.current}`,
            timestamp: new Date(),
            message,
          };

          setEvents(prev => {
            const updated = [...prev, newEvent];
            // Limit to maxEvents
            if (updated.length > maxEvents) {
              return updated.slice(updated.length - maxEvents);
            }
            return updated;
          });
        }
      });
    });

    // Connect to WebSocket
    client.connect().catch(err => {
      console.error('Failed to connect to WebSocket:', err);
    });

    setWsClient(client);

    // Cleanup
    return () => {
      client.destroy();
    };
  }, [maxEvents, isPaused]);

  // Auto-scroll effect
  useEffect(() => {
    if (autoScroll && eventsContainerRef.current) {
      eventsContainerRef.current.scrollTop = eventsContainerRef.current.scrollHeight;
    }
  }, [events, autoScroll]);

  // Message type filter handlers
  const toggleTypeFilter = (type: MessageType) => {
    setFilteredTypes(prev => {
      const newSet = new Set(prev);
      if (newSet.has(type)) {
        newSet.delete(type);
      } else {
        newSet.add(type);
      }
      return newSet;
    });
  };

  const clearAllFilters = () => {
    setFilteredTypes(new Set());
  };

  // Get filtered events
  const filteredEvents = events.filter(event => {
    if (filteredTypes.size === 0) return true;
    return filteredTypes.has(event.message.type);
  });

  // Clear events
  const clearEvents = () => {
    setEvents([]);
    eventCountRef.current = 0;
  };

  // Connection state colors
  const getConnectionStateColor = (state: ConnectionState): string => {
    switch (state) {
      case 'connected': return 'text-green-600 bg-green-100';
      case 'connecting': return 'text-yellow-600 bg-yellow-100';
      case 'reconnecting': return 'text-orange-600 bg-orange-100';
      case 'disconnected': return 'text-gray-600 bg-gray-100';
      case 'error': return 'text-red-600 bg-red-100';
      default: return 'text-gray-600 bg-gray-100';
    }
  };

  // Message type colors
  const getMessageTypeColor = (type: MessageType): string => {
    switch (type) {
      case 'frame': return 'bg-blue-100 text-blue-800';
      case 'state': return 'bg-green-100 text-green-800';
      case 'trajectory': return 'bg-purple-100 text-purple-800';
      case 'alert': return 'bg-red-100 text-red-800';
      case 'error': return 'bg-red-200 text-red-900';
      case 'config': return 'bg-yellow-100 text-yellow-800';
      case 'metrics': return 'bg-cyan-100 text-cyan-800';
      case 'ping':
      case 'pong': return 'bg-gray-100 text-gray-600';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  // Get connection stats
  const stats = wsClient?.connectionStats || {
    uptime: 0,
    messages_sent: 0,
    messages_received: 0,
    bytes_sent: 0,
    bytes_received: 0,
    reconnect_count: 0,
  };

  // Available message types
  const availableTypes: MessageType[] = [
    'frame', 'state', 'trajectory', 'alert', 'config', 'metrics',
    'connection', 'ping', 'pong', 'subscribe', 'unsubscribe',
    'subscribed', 'unsubscribed', 'status', 'error'
  ];

  return (
    <div className="container mx-auto p-6 space-y-6">
      <div className="flex justify-between items-center">
        <h1 className="text-3xl font-bold">WebSocket Event Monitor</h1>
        <div className={`px-4 py-2 rounded-lg font-semibold ${getConnectionStateColor(connectionState)}`}>
          {connectionState.toUpperCase()}
        </div>
      </div>

      {/* Statistics */}
      <Card>
        <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4">
          <div>
            <div className="text-sm text-gray-500">Total Events</div>
            <div className="text-2xl font-bold">{events.length}</div>
          </div>
          <div>
            <div className="text-sm text-gray-500">Filtered Events</div>
            <div className="text-2xl font-bold">{filteredEvents.length}</div>
          </div>
          <div>
            <div className="text-sm text-gray-500">Messages Sent</div>
            <div className="text-2xl font-bold">{stats.messages_sent}</div>
          </div>
          <div>
            <div className="text-sm text-gray-500">Messages Received</div>
            <div className="text-2xl font-bold">{stats.messages_received}</div>
          </div>
          <div>
            <div className="text-sm text-gray-500">Reconnects</div>
            <div className="text-2xl font-bold">{stats.reconnect_count}</div>
          </div>
          <div>
            <div className="text-sm text-gray-500">Uptime (s)</div>
            <div className="text-2xl font-bold">{Math.floor(stats.uptime / 1000)}</div>
          </div>
        </div>
      </Card>

      {/* Controls */}
      <Card>
        <div className="space-y-4">
          <div className="flex flex-wrap gap-4 items-center">
            <Button
              onClick={clearEvents}
              variant="secondary"
              disabled={events.length === 0}
            >
              Clear Events
            </Button>

            <Button
              onClick={() => setIsPaused(!isPaused)}
              variant={isPaused ? "primary" : "secondary"}
            >
              {isPaused ? 'Resume' : 'Pause'}
            </Button>

            <label className="flex items-center gap-2">
              <Checkbox
                checked={autoScroll}
                onChange={(e) => setAutoScroll(e.target.checked)}
              />
              <span className="text-sm">Auto-scroll</span>
            </label>

            <label className="flex items-center gap-2">
              <span className="text-sm">Max Events:</span>
              <input
                type="number"
                min="10"
                max="1000"
                value={maxEvents}
                onChange={(e) => setMaxEvents(parseInt(e.target.value) || 100)}
                className="w-20 px-2 py-1 border rounded"
              />
            </label>
          </div>

          {/* Message Type Filters */}
          <div>
            <div className="flex items-center gap-2 mb-2">
              <span className="text-sm font-semibold">Filter by Type:</span>
              <Button
                onClick={clearAllFilters}
                variant="secondary"
                className="text-xs"
                disabled={filteredTypes.size === 0}
              >
                Clear Filters
              </Button>
            </div>
            <div className="flex flex-wrap gap-2">
              {availableTypes.map(type => (
                <button
                  key={type}
                  onClick={() => toggleTypeFilter(type)}
                  className={`px-3 py-1 rounded-full text-sm font-medium transition-opacity ${
                    filteredTypes.size === 0 || filteredTypes.has(type)
                      ? getMessageTypeColor(type)
                      : 'bg-gray-200 text-gray-400 opacity-50'
                  }`}
                >
                  {type}
                </button>
              ))}
            </div>
          </div>
        </div>
      </Card>

      {/* Events Display */}
      <Card>
        <h2 className="text-xl font-bold mb-4">Event Log</h2>
        <div
          ref={eventsContainerRef}
          className="space-y-2 max-h-[600px] overflow-y-auto"
        >
          {filteredEvents.length === 0 ? (
            <div className="text-center text-gray-500 py-8">
              {events.length === 0 ? 'No events received yet' : 'No events match the selected filters'}
            </div>
          ) : (
            filteredEvents.map(event => (
              <div
                key={event.id}
                className="border border-gray-200 rounded-lg p-4 hover:bg-gray-50 transition-colors"
              >
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center gap-2">
                    <span className={`px-2 py-1 rounded text-xs font-semibold ${getMessageTypeColor(event.message.type)}`}>
                      {event.message.type}
                    </span>
                    {event.message.sequence !== undefined && (
                      <span className="text-xs text-gray-500">
                        seq: {event.message.sequence}
                      </span>
                    )}
                  </div>
                  <span className="text-xs text-gray-500">
                    {event.timestamp.toLocaleTimeString()}.{event.timestamp.getMilliseconds().toString().padStart(3, '0')}
                  </span>
                </div>
                <pre className="text-xs bg-gray-100 p-3 rounded overflow-x-auto">
                  {JSON.stringify(event.message, null, 2)}
                </pre>
              </div>
            ))
          )}
        </div>
      </Card>
    </div>
  );
});
