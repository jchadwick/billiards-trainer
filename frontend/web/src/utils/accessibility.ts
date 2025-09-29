/**
 * ARIA utilities and accessibility helpers
 */

export interface AriaAttributes {
  role?: string
  'aria-label'?: string
  'aria-labelledby'?: string
  'aria-describedby'?: string
  'aria-expanded'?: boolean
  'aria-hidden'?: boolean
  'aria-live'?: 'polite' | 'assertive' | 'off'
  'aria-atomic'?: boolean
  'aria-relevant'?: 'all' | 'additions' | 'removals' | 'text' | 'additions text' | 'additions removals' | 'removals additions' | 'removals text' | 'text additions' | 'text removals'
  'aria-busy'?: boolean
  'aria-current'?: 'page' | 'step' | 'location' | 'date' | 'time' | boolean
  'aria-disabled'?: boolean
  'aria-invalid'?: boolean | 'grammar' | 'spelling'
  'aria-pressed'?: boolean
  'aria-selected'?: boolean
  'aria-checked'?: boolean | 'mixed'
  'aria-required'?: boolean
  'aria-modal'?: boolean
  'aria-setsize'?: number
  'aria-posinset'?: number
  'aria-controls'?: string
  'aria-valuenow'?: number
  'aria-valuemin'?: number
  'aria-valuemax'?: number
  'aria-valuetext'?: string
  tabIndex?: number
}

/**
 * Generate unique IDs for ARIA relationships
 */
let idCounter = 0
export function generateId(prefix: string = 'accessibility'): string {
  return `${prefix}-${++idCounter}`
}

/**
 * Create ARIA attributes for form controls
 */
export function createFormControlAria(options: {
  label?: string
  labelId?: string
  description?: string
  descriptionId?: string
  error?: string
  errorId?: string
  required?: boolean
  invalid?: boolean
}): AriaAttributes {
  const { label, labelId, description, descriptionId, error, errorId, required, invalid } = options

  const aria: AriaAttributes = {}

  if (label && !labelId) {
    aria['aria-label'] = label
  } else if (labelId) {
    aria['aria-labelledby'] = labelId
  }

  const describedBy: string[] = []
  if (description && descriptionId) {
    describedBy.push(descriptionId)
  }
  if (error && errorId && invalid) {
    describedBy.push(errorId)
  }

  if (describedBy.length > 0) {
    aria['aria-describedby'] = describedBy.join(' ')
  }

  if (required) {
    aria['aria-required'] = true
  }

  if (invalid !== undefined) {
    aria['aria-invalid'] = invalid
  }

  return aria
}

/**
 * Create ARIA attributes for buttons
 */
export function createButtonAria(options: {
  label?: string
  expanded?: boolean
  pressed?: boolean
  disabled?: boolean
  describedBy?: string
}): AriaAttributes {
  const { label, expanded, pressed, disabled, describedBy } = options

  const aria: AriaAttributes = {}

  if (label) {
    aria['aria-label'] = label
  }

  if (expanded !== undefined) {
    aria['aria-expanded'] = expanded
  }

  if (pressed !== undefined) {
    aria['aria-pressed'] = pressed
  }

  if (disabled) {
    aria['aria-disabled'] = true
  }

  if (describedBy) {
    aria['aria-describedby'] = describedBy
  }

  return aria
}

/**
 * Create ARIA attributes for navigation
 */
export function createNavigationAria(options: {
  label?: string
  current?: 'page' | 'step' | 'location' | boolean
}): AriaAttributes {
  const { label, current } = options

  const aria: AriaAttributes = {
    role: 'navigation'
  }

  if (label) {
    aria['aria-label'] = label
  }

  if (current !== undefined) {
    aria['aria-current'] = current
  }

  return aria
}

/**
 * Create ARIA attributes for live regions
 */
export function createLiveRegionAria(options: {
  level?: 'polite' | 'assertive'
  atomic?: boolean
  relevant?: 'all' | 'additions' | 'removals' | 'text' | 'additions text' | 'additions removals' | 'removals additions' | 'removals text' | 'text additions' | 'text removals'
}): AriaAttributes {
  const { level = 'polite', atomic = true, relevant = 'additions text' } = options

  return {
    'aria-live': level,
    'aria-atomic': atomic,
    'aria-relevant': relevant
  }
}

/**
 * Create ARIA attributes for modal dialogs
 */
export function createModalAria(options: {
  labelId?: string
  descriptionId?: string
  modal?: boolean
}): AriaAttributes {
  const { labelId, descriptionId, modal = true } = options

  const aria: AriaAttributes = {
    role: 'dialog'
  }

  if (modal) {
    aria.role = 'dialog'
    aria['aria-modal'] = true
  }

  if (labelId) {
    aria['aria-labelledby'] = labelId
  }

  if (descriptionId) {
    aria['aria-describedby'] = descriptionId
  }

  return aria
}

/**
 * Create ARIA attributes for lists
 */
export function createListAria(options: {
  label?: string
  itemCount?: number
}): AriaAttributes {
  const { label, itemCount } = options

  const aria: AriaAttributes = {}

  if (label) {
    aria['aria-label'] = label
  }

  if (itemCount !== undefined) {
    aria['aria-setsize'] = itemCount
  }

  return aria
}

/**
 * Create ARIA attributes for list items
 */
export function createListItemAria(options: {
  position?: number
  total?: number
  selected?: boolean
  current?: boolean
}): AriaAttributes {
  const { position, total, selected, current } = options

  const aria: AriaAttributes = {}

  if (position !== undefined) {
    aria['aria-posinset'] = position
  }

  if (total !== undefined) {
    aria['aria-setsize'] = total
  }

  if (selected !== undefined) {
    aria['aria-selected'] = selected
  }

  if (current) {
    aria['aria-current'] = true
  }

  return aria
}

/**
 * Create ARIA attributes for tabs
 */
export function createTabAria(options: {
  selected?: boolean
  controls?: string
  index?: number
}): AriaAttributes {
  const { selected, controls, index } = options

  const aria: AriaAttributes = {
    role: 'tab'
  }

  if (selected !== undefined) {
    aria['aria-selected'] = selected
  }

  if (controls) {
    aria['aria-controls'] = controls
  }

  if (index !== undefined) {
    aria.tabIndex = selected ? 0 : -1
  }

  return aria
}

/**
 * Create ARIA attributes for tab panels
 */
export function createTabPanelAria(options: {
  labelledBy?: string
  hidden?: boolean
}): AriaAttributes {
  const { labelledBy, hidden } = options

  const aria: AriaAttributes = {
    role: 'tabpanel'
  }

  if (labelledBy) {
    aria['aria-labelledby'] = labelledBy
  }

  if (hidden !== undefined) {
    aria['aria-hidden'] = hidden
    aria.tabIndex = hidden ? -1 : 0
  }

  return aria
}

/**
 * Create ARIA attributes for video controls
 */
export function createVideoControlAria(options: {
  label?: string
  state?: 'playing' | 'paused' | 'stopped' | 'muted' | 'unmuted'
  pressed?: boolean
  describedBy?: string
}): AriaAttributes {
  const { label, state, pressed, describedBy } = options

  const aria: AriaAttributes = {}

  if (label) {
    let fullLabel = label
    if (state) {
      const stateLabels = {
        playing: 'Currently playing',
        paused: 'Currently paused',
        stopped: 'Currently stopped',
        muted: 'Currently muted',
        unmuted: 'Currently unmuted'
      }
      fullLabel = `${label}. ${stateLabels[state]}`
    }
    aria['aria-label'] = fullLabel
  }

  if (pressed !== undefined) {
    aria['aria-pressed'] = pressed
  }

  if (describedBy) {
    aria['aria-describedby'] = describedBy
  }

  return aria
}

/**
 * Create ARIA attributes for overlay elements (like ball markers, trajectories)
 */
export function createOverlayAria(options: {
  label: string
  description?: string
  hidden?: boolean
  role?: string
}): AriaAttributes {
  const { label, description, hidden = false, role = 'img' } = options

  const aria: AriaAttributes = {
    role,
    'aria-label': label
  }

  if (description) {
    aria['aria-describedby'] = generateId('overlay-desc')
  }

  if (hidden) {
    aria['aria-hidden'] = true
  }

  return aria
}

/**
 * Create ARIA attributes for progress indicators
 */
export function createProgressAria(options: {
  label?: string
  value?: number
  min?: number
  max?: number
  description?: string
}): AriaAttributes {
  const { label, value, min = 0, max = 100, description } = options

  const aria: AriaAttributes = {
    role: 'progressbar'
  }

  if (label) {
    aria['aria-label'] = label
  }

  if (value !== undefined) {
    aria['aria-valuenow'] = value
    aria['aria-valuemin'] = min
    aria['aria-valuemax'] = max

    // Add value text for screen readers
    const percentage = Math.round((value / max) * 100)
    aria['aria-valuetext'] = `${percentage}%`
  }

  if (description) {
    aria['aria-describedby'] = generateId('progress-desc')
  }

  return aria
}

/**
 * Screen reader text utility
 */
export function createScreenReaderText(text: string): string {
  return text
}

/**
 * Generate descriptive text for complex visual elements
 */
export function createVisualDescription(type: 'ball' | 'trajectory' | 'table' | 'cue', data: any): string {
  switch (type) {
    case 'ball':
      return `Ball ${data.number || data.id} at position ${Math.round(data.x)}, ${Math.round(data.y)}${data.velocity ? ` moving at ${Math.round(data.velocity)} units per second` : ''}`

    case 'trajectory':
      return `Trajectory path with ${data.points?.length || 0} points, probability ${Math.round((data.probability || 0) * 100)}%`

    case 'table':
      return `Pool table detected with ${data.pockets?.length || 6} pockets`

    case 'cue':
      return `Cue stick detected at angle ${Math.round(data.angle || 0)} degrees${data.confidence ? `, confidence ${Math.round(data.confidence * 100)}%` : ''}`

    default:
      return 'Visual element'
  }
}

/**
 * Convert keyboard event to readable string
 */
export function getKeyboardEventDescription(event: KeyboardEvent): string {
  const parts: string[] = []

  if (event.ctrlKey) parts.push('Ctrl')
  if (event.altKey) parts.push('Alt')
  if (event.shiftKey) parts.push('Shift')
  if (event.metaKey) parts.push('Cmd')

  parts.push(event.key)

  return parts.join(' + ')
}

/**
 * Common keyboard shortcuts for billiards trainer
 */
export const KEYBOARD_SHORTCUTS: Record<string, string> = {
  // Global navigation
  'Alt + 1': 'Go to Live View',
  'Alt + 2': 'Go to Calibration',
  'Alt + 3': 'Go to Configuration',
  'Alt + 4': 'Go to Diagnostics',

  // Video controls
  'Space': 'Play/Pause video',
  'F': 'Toggle fullscreen',
  'M': 'Toggle mute',
  'S': 'Take screenshot',

  // Accessibility
  'Alt + A': 'Open accessibility settings',
  'Alt + H': 'Show keyboard shortcuts help',
  'Escape': 'Close current overlay or modal',

  // Navigation
  'Tab': 'Move to next element',
  'Shift + Tab': 'Move to previous element',
  'Enter': 'Activate current element',
  'Arrow Keys': 'Navigate within components',

  // Overlay controls
  'B': 'Toggle ball overlay visibility',
  'T': 'Toggle trajectory overlay visibility',
  'G': 'Toggle table overlay visibility',
  'C': 'Toggle cue overlay visibility',
}

/**
 * Get keyboard shortcut description for screen readers
 */
export function getShortcutDescription(key: string): string {
  return KEYBOARD_SHORTCUTS[key] || 'Unknown shortcut'
}
