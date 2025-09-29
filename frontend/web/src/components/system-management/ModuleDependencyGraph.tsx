import React, { useEffect, useRef, useState } from 'react'
import { observer } from 'mobx-react-lite'
import { Card, CardContent, CardHeader, CardTitle, Button } from '../ui'

interface ModuleNode {
  id: string
  name: string
  status: 'running' | 'stopped' | 'error' | 'starting' | 'stopping'
  dependencies: string[]
  dependents: string[]
  position?: { x: number; y: number }
}

const modules: ModuleNode[] = [
  {
    id: 'config',
    name: 'Configuration',
    status: 'running',
    dependencies: [],
    dependents: ['core', 'vision', 'api', 'projector'],
  },
  {
    id: 'core',
    name: 'Core Engine',
    status: 'running',
    dependencies: ['config'],
    dependents: ['vision', 'api', 'projector'],
  },
  {
    id: 'vision',
    name: 'Vision Processing',
    status: 'running',
    dependencies: ['config', 'core'],
    dependents: ['projector'],
  },
  {
    id: 'api',
    name: 'API Server',
    status: 'running',
    dependencies: ['config', 'core'],
    dependents: [],
  },
  {
    id: 'projector',
    name: 'Projector',
    status: 'error',
    dependencies: ['config', 'core', 'vision'],
    dependents: [],
  },
]

export const ModuleDependencyGraph = observer(() => {
  const svgRef = useRef<SVGSVGElement>(null)
  const [selectedModule, setSelectedModule] = useState<string | null>(null)
  const [graphLayout, setGraphLayout] = useState<'tree' | 'circular' | 'force'>('tree')

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'running':
        return '#10b981' // success-500
      case 'error':
        return '#ef4444' // error-500
      case 'stopped':
        return '#6b7280' // secondary-500
      case 'starting':
      case 'stopping':
        return '#f59e0b' // warning-500
      default:
        return '#6b7280'
    }
  }

  const calculateNodePositions = (layout: 'tree' | 'circular' | 'force') => {
    const svgWidth = 400
    const svgHeight = 300
    const nodeRadius = 40

    switch (layout) {
      case 'tree': {
        // Tree layout with config at the top
        const levels = {
          config: 0,
          core: 1,
          vision: 2,
          api: 2,
          projector: 3,
        }

        const levelCounts = Object.values(levels).reduce((acc, level) => {
          acc[level] = (acc[level] || 0) + 1
          return acc
        }, {} as Record<number, number>)

        const levelIndex = {
          config: 0,
          core: 0,
          vision: 0,
          api: 1,
          projector: 0,
        }

        return modules.map((module) => {
          const level = levels[module.id as keyof typeof levels]
          const count = levelCounts[level]
          const index = levelIndex[module.id as keyof typeof levelIndex]

          const y = (level + 1) * (svgHeight / 5)
          const x = count === 1
            ? svgWidth / 2
            : (index + 1) * (svgWidth / (count + 1))

          return {
            ...module,
            position: { x, y }
          }
        })
      }

      case 'circular': {
        return modules.map((module, index) => {
          const angle = (index * 2 * Math.PI) / modules.length
          const radius = Math.min(svgWidth, svgHeight) / 3
          const centerX = svgWidth / 2
          const centerY = svgHeight / 2

          const x = centerX + radius * Math.cos(angle)
          const y = centerY + radius * Math.sin(angle)

          return {
            ...module,
            position: { x, y }
          }
        })
      }

      case 'force': {
        // Simple force-based layout simulation
        const positions = modules.map((module, index) => ({
          ...module,
          position: {
            x: Math.random() * (svgWidth - 2 * nodeRadius) + nodeRadius,
            y: Math.random() * (svgHeight - 2 * nodeRadius) + nodeRadius
          }
        }))

        // Simple force simulation iterations
        for (let iteration = 0; iteration < 50; iteration++) {
          positions.forEach((node, i) => {
            let fx = 0, fy = 0

            // Repulsion between all nodes
            positions.forEach((other, j) => {
              if (i !== j) {
                const dx = node.position!.x - other.position!.x
                const dy = node.position!.y - other.position!.y
                const distance = Math.sqrt(dx * dx + dy * dy)
                if (distance > 0) {
                  const force = 1000 / (distance * distance)
                  fx += (dx / distance) * force
                  fy += (dy / distance) * force
                }
              }
            })

            // Attraction to dependencies
            node.dependencies.forEach(depId => {
              const dep = positions.find(p => p.id === depId)
              if (dep) {
                const dx = dep.position!.x - node.position!.x
                const dy = dep.position!.y - node.position!.y
                const distance = Math.sqrt(dx * dx + dy * dy)
                if (distance > 0) {
                  const force = 0.1
                  fx += (dx / distance) * force * distance
                  fy += (dy / distance) * force * distance
                }
              }
            })

            // Update position
            node.position!.x += fx * 0.01
            node.position!.y += fy * 0.01

            // Keep within bounds
            node.position!.x = Math.max(nodeRadius, Math.min(svgWidth - nodeRadius, node.position!.x))
            node.position!.y = Math.max(nodeRadius, Math.min(svgHeight - nodeRadius, node.position!.y))
          })
        }

        return positions
      }

      default:
        return modules.map((module, index) => ({
          ...module,
          position: { x: index * 80 + 50, y: 150 }
        }))
    }
  }

  const [nodesWithPositions, setNodesWithPositions] = useState(calculateNodePositions(graphLayout))

  useEffect(() => {
    setNodesWithPositions(calculateNodePositions(graphLayout))
  }, [graphLayout])

  const renderDependencyLines = () => {
    return nodesWithPositions.flatMap((node) =>
      node.dependencies.map((depId) => {
        const dependency = nodesWithPositions.find(n => n.id === depId)
        if (!dependency || !node.position || !dependency.position) return null

        const isHighlighted = selectedModule === node.id || selectedModule === depId

        return (
          <line
            key={`${depId}-${node.id}`}
            x1={dependency.position.x}
            y1={dependency.position.y}
            x2={node.position.x}
            y2={node.position.y}
            stroke={isHighlighted ? '#3b82f6' : '#d1d5db'}
            strokeWidth={isHighlighted ? 3 : 2}
            strokeOpacity={isHighlighted ? 1 : 0.6}
            markerEnd="url(#arrowhead)"
          />
        )
      }).filter(Boolean)
    )
  }

  const renderNodes = () => {
    return nodesWithPositions.map((node) => {
      if (!node.position) return null

      const isSelected = selectedModule === node.id
      const isConnected = selectedModule ? (
        node.dependencies.includes(selectedModule) ||
        node.dependents.includes(selectedModule)
      ) : false

      const opacity = selectedModule && !isSelected && !isConnected ? 0.3 : 1

      return (
        <g
          key={node.id}
          className="cursor-pointer"
          onClick={() => setSelectedModule(selectedModule === node.id ? null : node.id)}
          opacity={opacity}
        >
          <circle
            cx={node.position.x}
            cy={node.position.y}
            r={isSelected ? 45 : 40}
            fill={getStatusColor(node.status)}
            stroke={isSelected ? '#3b82f6' : '#ffffff'}
            strokeWidth={isSelected ? 3 : 2}
          />
          <text
            x={node.position.x}
            y={node.position.y + 5}
            textAnchor="middle"
            className="fill-white text-xs font-medium pointer-events-none"
            style={{ fontSize: '11px' }}
          >
            {node.name.split(' ').map((word, i) => (
              <tspan key={i} x={node.position!.x} dy={i === 0 ? 0 : 12}>
                {word}
              </tspan>
            ))}
          </text>
        </g>
      )
    })
  }

  const selectedModuleData = selectedModule ? nodesWithPositions.find(n => n.id === selectedModule) : null

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between">
        <CardTitle>Module Dependency Graph</CardTitle>
        <div className="flex space-x-2">
          <select
            value={graphLayout}
            onChange={(e) => setGraphLayout(e.target.value as any)}
            className="text-sm border border-secondary-300 dark:border-secondary-600 rounded px-2 py-1 bg-white dark:bg-secondary-800"
          >
            <option value="tree">Tree Layout</option>
            <option value="circular">Circular Layout</option>
            <option value="force">Force Layout</option>
          </select>
          <Button
            variant="outline"
            size="sm"
            onClick={() => setSelectedModule(null)}
          >
            Clear Selection
          </Button>
        </div>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          {/* SVG Graph */}
          <div className="bg-white dark:bg-secondary-900 border border-secondary-200 dark:border-secondary-700 rounded-lg p-4">
            <svg
              ref={svgRef}
              width="100%"
              height="300"
              viewBox="0 0 400 300"
              className="border border-secondary-200 dark:border-secondary-700 rounded"
            >
              <defs>
                <marker
                  id="arrowhead"
                  markerWidth="10"
                  markerHeight="7"
                  refX="9"
                  refY="3.5"
                  orient="auto"
                >
                  <polygon
                    points="0 0, 10 3.5, 0 7"
                    fill="#d1d5db"
                  />
                </marker>
              </defs>
              {renderDependencyLines()}
              {renderNodes()}
            </svg>
          </div>

          {/* Module Details */}
          {selectedModuleData && (
            <div className="p-4 bg-secondary-50 dark:bg-secondary-800 rounded-lg border border-secondary-200 dark:border-secondary-700">
              <h4 className="text-sm font-medium text-secondary-900 dark:text-secondary-100 mb-3">
                {selectedModuleData.name} Details
              </h4>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
                <div>
                  <span className="font-medium text-secondary-700 dark:text-secondary-300">Status:</span>
                  <span className={`ml-2 px-2 py-1 rounded-full text-xs font-medium capitalize ${
                    selectedModuleData.status === 'running'
                      ? 'bg-success-100 text-success-800 dark:bg-success-900 dark:text-success-200'
                      : selectedModuleData.status === 'error'
                      ? 'bg-error-100 text-error-800 dark:bg-error-900 dark:text-error-200'
                      : 'bg-secondary-100 text-secondary-800 dark:bg-secondary-900 dark:text-secondary-200'
                  }`}>
                    {selectedModuleData.status}
                  </span>
                </div>
                <div>
                  <span className="font-medium text-secondary-700 dark:text-secondary-300">Dependencies:</span>
                  <span className="ml-2 text-secondary-600 dark:text-secondary-400">
                    {selectedModuleData.dependencies.length > 0
                      ? selectedModuleData.dependencies.join(', ')
                      : 'None'
                    }
                  </span>
                </div>
                <div>
                  <span className="font-medium text-secondary-700 dark:text-secondary-300">Dependents:</span>
                  <span className="ml-2 text-secondary-600 dark:text-secondary-400">
                    {selectedModuleData.dependents.length > 0
                      ? selectedModuleData.dependents.join(', ')
                      : 'None'
                    }
                  </span>
                </div>
                <div>
                  <span className="font-medium text-secondary-700 dark:text-secondary-300">Impact Level:</span>
                  <span className="ml-2 text-secondary-600 dark:text-secondary-400">
                    {selectedModuleData.dependents.length === 0 ? 'Low' :
                     selectedModuleData.dependents.length <= 2 ? 'Medium' : 'High'
                    }
                  </span>
                </div>
              </div>
            </div>
          )}

          {/* Legend */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <h4 className="text-sm font-medium text-secondary-900 dark:text-secondary-100 mb-2">
                Status Legend
              </h4>
              <div className="space-y-1 text-xs">
                <div className="flex items-center space-x-2">
                  <div className="w-3 h-3 rounded-full bg-success-500" />
                  <span className="text-secondary-600 dark:text-secondary-400">Running</span>
                </div>
                <div className="flex items-center space-x-2">
                  <div className="w-3 h-3 rounded-full bg-error-500" />
                  <span className="text-secondary-600 dark:text-secondary-400">Error</span>
                </div>
                <div className="flex items-center space-x-2">
                  <div className="w-3 h-3 rounded-full bg-warning-500" />
                  <span className="text-secondary-600 dark:text-secondary-400">Starting/Stopping</span>
                </div>
                <div className="flex items-center space-x-2">
                  <div className="w-3 h-3 rounded-full bg-secondary-500" />
                  <span className="text-secondary-600 dark:text-secondary-400">Stopped</span>
                </div>
              </div>
            </div>
            <div>
              <h4 className="text-sm font-medium text-secondary-900 dark:text-secondary-100 mb-2">
                Interactions
              </h4>
              <div className="space-y-1 text-xs text-secondary-600 dark:text-secondary-400">
                <div>• Click on a module to highlight its dependencies</div>
                <div>• Arrows show dependency direction</div>
                <div>• Switch layouts to see different perspectives</div>
                <div>• Modules with more dependents have higher impact</div>
              </div>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  )
})