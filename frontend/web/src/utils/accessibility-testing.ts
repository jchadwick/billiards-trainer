/**
 * Accessibility Testing Utilities
 * Provides tools for testing and validating accessibility features
 */

export interface AccessibilityTestResult {
  passed: boolean
  message: string
  element?: HTMLElement
  rule: string
  severity: 'error' | 'warning' | 'info'
}

export interface AccessibilityTestReport {
  overall: 'pass' | 'fail' | 'warning'
  results: AccessibilityTestResult[]
  score: number
  totalTests: number
  passedTests: number
  failedTests: number
  warnings: number
}

/**
 * Core accessibility testing functions
 */
export class AccessibilityTester {
  private results: AccessibilityTestResult[] = []

  /**
   * Run all accessibility tests on a container element
   */
  async testAccessibility(container: HTMLElement = document.body): Promise<AccessibilityTestReport> {
    this.results = []

    // Run all tests
    this.testHeadingStructure(container)
    this.testImageAltText(container)
    this.testFormLabels(container)
    this.testFocusableElements(container)
    this.testColorContrast(container)
    this.testAriaAttributes(container)
    this.testKeyboardNavigation(container)
    this.testLiveRegions(container)
    this.testSemanticMarkup(container)
    this.testSkipLinks(container)

    return this.generateReport()
  }

  /**
   * Test heading structure (h1-h6 hierarchy)
   */
  private testHeadingStructure(container: HTMLElement): void {
    const headings = container.querySelectorAll('h1, h2, h3, h4, h5, h6')
    let prevLevel = 0

    headings.forEach((heading) => {
      const level = parseInt(heading.tagName.substring(1))

      if (level > prevLevel + 1) {
        this.addResult({
          passed: false,
          message: `Heading level jumps from h${prevLevel} to h${level}. Use proper heading hierarchy.`,
          element: heading as HTMLElement,
          rule: 'heading-hierarchy',
          severity: 'error'
        })
      } else {
        this.addResult({
          passed: true,
          message: `Heading h${level} follows proper hierarchy`,
          element: heading as HTMLElement,
          rule: 'heading-hierarchy',
          severity: 'info'
        })
      }

      prevLevel = level
    })

    // Check for h1 presence
    const h1Elements = container.querySelectorAll('h1')
    if (h1Elements.length === 0) {
      this.addResult({
        passed: false,
        message: 'No h1 element found. Each page should have exactly one h1.',
        rule: 'h1-required',
        severity: 'error'
      })
    } else if (h1Elements.length > 1) {
      this.addResult({
        passed: false,
        message: `Multiple h1 elements found (${h1Elements.length}). Each page should have exactly one h1.`,
        rule: 'h1-unique',
        severity: 'warning'
      })
    }
  }

  /**
   * Test image alt text
   */
  private testImageAltText(container: HTMLElement): void {
    const images = container.querySelectorAll('img')

    images.forEach((img) => {
      const alt = img.getAttribute('alt')
      const ariaLabel = img.getAttribute('aria-label')
      const ariaHidden = img.getAttribute('aria-hidden')

      if (ariaHidden === 'true') {
        this.addResult({
          passed: true,
          message: 'Decorative image properly hidden from screen readers',
          element: img,
          rule: 'img-alt',
          severity: 'info'
        })
      } else if (!alt && !ariaLabel) {
        this.addResult({
          passed: false,
          message: 'Image missing alt text or aria-label',
          element: img,
          rule: 'img-alt',
          severity: 'error'
        })
      } else if (alt === '' && !ariaLabel) {
        this.addResult({
          passed: false,
          message: 'Image has empty alt text but is not marked as decorative',
          element: img,
          rule: 'img-alt',
          severity: 'warning'
        })
      } else {
        this.addResult({
          passed: true,
          message: 'Image has appropriate alt text',
          element: img,
          rule: 'img-alt',
          severity: 'info'
        })
      }
    })
  }

  /**
   * Test form labels
   */
  private testFormLabels(container: HTMLElement): void {
    const formControls = container.querySelectorAll('input, select, textarea')

    formControls.forEach((control) => {
      const id = control.getAttribute('id')
      const ariaLabel = control.getAttribute('aria-label')
      const ariaLabelledBy = control.getAttribute('aria-labelledby')
      const label = id ? container.querySelector(`label[for="${id}"]`) : null
      const type = control.getAttribute('type')

      // Skip hidden inputs
      if (type === 'hidden') return

      if (!label && !ariaLabel && !ariaLabelledBy) {
        this.addResult({
          passed: false,
          message: 'Form control missing label',
          element: control as HTMLElement,
          rule: 'form-labels',
          severity: 'error'
        })
      } else {
        this.addResult({
          passed: true,
          message: 'Form control has appropriate label',
          element: control as HTMLElement,
          rule: 'form-labels',
          severity: 'info'
        })
      }
    })
  }

  /**
   * Test focusable elements
   */
  private testFocusableElements(container: HTMLElement): void {
    const focusableSelector = 'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
    const focusableElements = container.querySelectorAll(focusableSelector)

    focusableElements.forEach((element) => {
      const el = element as HTMLElement
      const tabIndex = el.getAttribute('tabindex')
      const disabled = el.hasAttribute('disabled')
      const ariaHidden = el.getAttribute('aria-hidden')

      // Check for positive tabindex (anti-pattern)
      if (tabIndex && parseInt(tabIndex) > 0) {
        this.addResult({
          passed: false,
          message: 'Avoid positive tabindex values. Use DOM order instead.',
          element: el,
          rule: 'tabindex-positive',
          severity: 'warning'
        })
      }

      // Check for focusable but hidden elements
      if (ariaHidden === 'true' && !disabled) {
        this.addResult({
          passed: false,
          message: 'Focusable element is hidden from screen readers',
          element: el,
          rule: 'focusable-hidden',
          severity: 'error'
        })
      }

      // Check if element is actually focusable
      try {
        const rect = el.getBoundingClientRect()
        if (rect.width === 0 && rect.height === 0) {
          this.addResult({
            passed: false,
            message: 'Focusable element has no visible dimensions',
            element: el,
            rule: 'focusable-visible',
            severity: 'warning'
          })
        }
      } catch (error) {
        // Element might not be in DOM
      }
    })
  }

  /**
   * Test color contrast (basic check)
   */
  private testColorContrast(container: HTMLElement): void {
    const textElements = container.querySelectorAll('p, span, div, h1, h2, h3, h4, h5, h6, a, button, label')

    textElements.forEach((element) => {
      const el = element as HTMLElement
      const styles = window.getComputedStyle(el)
      const color = styles.color
      const backgroundColor = styles.backgroundColor

      // Basic contrast check (simplified)
      if (color && backgroundColor && backgroundColor !== 'rgba(0, 0, 0, 0)') {
        const contrastRatio = this.calculateContrastRatio(color, backgroundColor)

        if (contrastRatio < 4.5) {
          this.addResult({
            passed: false,
            message: `Low color contrast ratio: ${contrastRatio.toFixed(2)}. WCAG requires 4.5:1 for normal text.`,
            element: el,
            rule: 'color-contrast',
            severity: 'error'
          })
        } else {
          this.addResult({
            passed: true,
            message: `Good color contrast ratio: ${contrastRatio.toFixed(2)}`,
            element: el,
            rule: 'color-contrast',
            severity: 'info'
          })
        }
      }
    })
  }

  /**
   * Test ARIA attributes
   */
  private testAriaAttributes(container: HTMLElement): void {
    const elementsWithAria = container.querySelectorAll('[aria-label], [aria-labelledby], [aria-describedby], [role]')

    elementsWithAria.forEach((element) => {
      const el = element as HTMLElement
      const ariaLabelledBy = el.getAttribute('aria-labelledby')
      const ariaDescribedBy = el.getAttribute('aria-describedby')
      const role = el.getAttribute('role')

      // Check if aria-labelledby references exist
      if (ariaLabelledBy) {
        const ids = ariaLabelledBy.split(' ')
        ids.forEach((id) => {
          const referencedElement = container.querySelector(`#${id}`)
          if (!referencedElement) {
            this.addResult({
              passed: false,
              message: `aria-labelledby references non-existent element with id="${id}"`,
              element: el,
              rule: 'aria-references',
              severity: 'error'
            })
          }
        })
      }

      // Check if aria-describedby references exist
      if (ariaDescribedBy) {
        const ids = ariaDescribedBy.split(' ')
        ids.forEach((id) => {
          const referencedElement = container.querySelector(`#${id}`)
          if (!referencedElement) {
            this.addResult({
              passed: false,
              message: `aria-describedby references non-existent element with id="${id}"`,
              element: el,
              rule: 'aria-references',
              severity: 'error'
            })
          }
        })
      }

      // Check for valid roles
      if (role) {
        const validRoles = [
          'alert', 'alertdialog', 'application', 'article', 'banner', 'button', 'cell',
          'checkbox', 'columnheader', 'combobox', 'complementary', 'contentinfo',
          'definition', 'dialog', 'directory', 'document', 'feed', 'figure', 'form',
          'grid', 'gridcell', 'group', 'heading', 'img', 'link', 'list', 'listbox',
          'listitem', 'log', 'main', 'marquee', 'math', 'menu', 'menubar', 'menuitem',
          'menuitemcheckbox', 'menuitemradio', 'navigation', 'none', 'note', 'option',
          'presentation', 'progressbar', 'radio', 'radiogroup', 'region', 'row',
          'rowgroup', 'rowheader', 'scrollbar', 'search', 'searchbox', 'separator',
          'slider', 'spinbutton', 'status', 'switch', 'tab', 'table', 'tablist',
          'tabpanel', 'term', 'textbox', 'timer', 'toolbar', 'tooltip', 'tree',
          'treegrid', 'treeitem'
        ]

        if (!validRoles.includes(role)) {
          this.addResult({
            passed: false,
            message: `Invalid ARIA role: "${role}"`,
            element: el,
            rule: 'aria-valid-roles',
            severity: 'error'
          })
        }
      }
    })
  }

  /**
   * Test keyboard navigation
   */
  private testKeyboardNavigation(container: HTMLElement): void {
    const focusableElements = container.querySelectorAll(
      'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
    )

    let tabOrder: number[] = []
    focusableElements.forEach((element) => {
      const tabIndex = (element as HTMLElement).tabIndex
      tabOrder.push(tabIndex)
    })

    // Check for proper tab order
    let hasPositiveTabIndex = false
    tabOrder.forEach((tabIndex) => {
      if (tabIndex > 0) {
        hasPositiveTabIndex = true
      }
    })

    if (hasPositiveTabIndex) {
      this.addResult({
        passed: false,
        message: 'Positive tabindex values found. This can disrupt natural tab order.',
        rule: 'tab-order',
        severity: 'warning'
      })
    }

    // Check for skip links
    const skipLinks = container.querySelectorAll('a[href^="#"]')
    let hasSkipToMain = false

    skipLinks.forEach((link) => {
      const href = link.getAttribute('href')
      if (href === '#main-content' || href === '#main') {
        hasSkipToMain = true
      }
    })

    if (!hasSkipToMain && focusableElements.length > 5) {
      this.addResult({
        passed: false,
        message: 'No "skip to main content" link found. Consider adding skip links for keyboard users.',
        rule: 'skip-links',
        severity: 'warning'
      })
    }
  }

  /**
   * Test live regions
   */
  private testLiveRegions(container: HTMLElement): void {
    const liveRegions = container.querySelectorAll('[aria-live], [role="status"], [role="alert"]')

    liveRegions.forEach((region) => {
      const el = region as HTMLElement
      const ariaLive = el.getAttribute('aria-live')
      const role = el.getAttribute('role')

      if (role === 'alert' && ariaLive && ariaLive !== 'assertive') {
        this.addResult({
          passed: false,
          message: 'Alert role should have aria-live="assertive" or no aria-live attribute',
          element: el,
          rule: 'live-regions',
          severity: 'warning'
        })
      }

      if (role === 'status' && ariaLive && ariaLive !== 'polite') {
        this.addResult({
          passed: false,
          message: 'Status role should have aria-live="polite" or no aria-live attribute',
          element: el,
          rule: 'live-regions',
          severity: 'warning'
        })
      }
    })
  }

  /**
   * Test semantic markup
   */
  private testSemanticMarkup(container: HTMLElement): void {
    // Check for proper landmark usage
    const main = container.querySelectorAll('main, [role="main"]')
    if (main.length === 0) {
      this.addResult({
        passed: false,
        message: 'No main landmark found. Add <main> element or role="main"',
        rule: 'landmarks',
        severity: 'error'
      })
    } else if (main.length > 1) {
      this.addResult({
        passed: false,
        message: 'Multiple main landmarks found. There should be only one per page.',
        rule: 'landmarks',
        severity: 'error'
      })
    }

    // Check for navigation landmarks
    const nav = container.querySelectorAll('nav, [role="navigation"]')
    nav.forEach((navElement) => {
      const el = navElement as HTMLElement
      const ariaLabel = el.getAttribute('aria-label')
      const ariaLabelledBy = el.getAttribute('aria-labelledby')

      if (nav.length > 1 && !ariaLabel && !ariaLabelledBy) {
        this.addResult({
          passed: false,
          message: 'Multiple navigation landmarks should have unique labels',
          element: el,
          rule: 'landmarks',
          severity: 'warning'
        })
      }
    })

    // Check for list markup
    const lists = container.querySelectorAll('ul, ol')
    lists.forEach((list) => {
      const listItems = list.querySelectorAll('> li')
      if (listItems.length === 0) {
        this.addResult({
          passed: false,
          message: 'List element contains no list items',
          element: list as HTMLElement,
          rule: 'list-structure',
          severity: 'error'
        })
      }
    })
  }

  /**
   * Test skip links
   */
  private testSkipLinks(container: HTMLElement): void {
    const skipLinks = container.querySelectorAll('a[href^="#"]')

    skipLinks.forEach((link) => {
      const href = link.getAttribute('href')
      if (href && href !== '#') {
        const target = container.querySelector(href)
        if (!target) {
          this.addResult({
            passed: false,
            message: `Skip link points to non-existent target: ${href}`,
            element: link as HTMLElement,
            rule: 'skip-links',
            severity: 'error'
          })
        } else {
          // Check if target is focusable
          const targetEl = target as HTMLElement
          const tabIndex = targetEl.getAttribute('tabindex')
          if (tabIndex !== '0' && tabIndex !== '-1' && !this.isFocusableElement(targetEl)) {
            this.addResult({
              passed: false,
              message: `Skip link target is not focusable: ${href}`,
              element: link as HTMLElement,
              rule: 'skip-links',
              severity: 'warning'
            })
          }
        }
      }
    })
  }

  /**
   * Add a test result
   */
  private addResult(result: AccessibilityTestResult): void {
    this.results.push(result)
  }

  /**
   * Generate final accessibility report
   */
  private generateReport(): AccessibilityTestReport {
    const passedTests = this.results.filter(r => r.passed).length
    const failedTests = this.results.filter(r => !r.passed && r.severity === 'error').length
    const warnings = this.results.filter(r => !r.passed && r.severity === 'warning').length
    const totalTests = this.results.length

    const score = totalTests > 0 ? Math.round((passedTests / totalTests) * 100) : 100

    let overall: 'pass' | 'fail' | 'warning' = 'pass'
    if (failedTests > 0) {
      overall = 'fail'
    } else if (warnings > 0) {
      overall = 'warning'
    }

    return {
      overall,
      results: this.results,
      score,
      totalTests,
      passedTests,
      failedTests,
      warnings
    }
  }

  /**
   * Check if element is naturally focusable
   */
  private isFocusableElement(element: HTMLElement): boolean {
    const focusableTags = ['A', 'BUTTON', 'INPUT', 'SELECT', 'TEXTAREA']
    return focusableTags.includes(element.tagName)
  }

  /**
   * Calculate color contrast ratio (simplified)
   */
  private calculateContrastRatio(color1: string, color2: string): number {
    // This is a simplified implementation
    // In a real application, you'd want a more robust color parsing and contrast calculation

    const getLuminance = (color: string): number => {
      // Very basic luminance calculation
      // Real implementation should handle rgb(), rgba(), hex, hsl(), etc.
      if (color.includes('rgb')) {
        const match = color.match(/\d+/g)
        if (match && match.length >= 3) {
          const [r, g, b] = match.map(n => parseInt(n) / 255)
          return 0.299 * r + 0.587 * g + 0.114 * b
        }
      }
      return 0.5 // Default middle value
    }

    const lum1 = getLuminance(color1)
    const lum2 = getLuminance(color2)
    const brightest = Math.max(lum1, lum2)
    const darkest = Math.min(lum1, lum2)

    return (brightest + 0.05) / (darkest + 0.05)
  }
}

/**
 * Utility functions for accessibility testing
 */
export const AccessibilityTestUtils = {
  /**
   * Test if element is keyboard accessible
   */
  async testKeyboardAccess(element: HTMLElement): Promise<boolean> {
    return new Promise((resolve) => {
      const originalFocus = document.activeElement

      try {
        element.focus()
        const focused = document.activeElement === element

        // Restore original focus
        if (originalFocus instanceof HTMLElement) {
          originalFocus.focus()
        }

        resolve(focused)
      } catch (error) {
        resolve(false)
      }
    })
  },

  /**
   * Test if element is visible to screen readers
   */
  isVisibleToScreenReader(element: HTMLElement): boolean {
    const styles = window.getComputedStyle(element)
    const ariaHidden = element.getAttribute('aria-hidden')

    return (
      styles.display !== 'none' &&
      styles.visibility !== 'hidden' &&
      styles.opacity !== '0' &&
      ariaHidden !== 'true'
    )
  },

  /**
   * Get accessible name for element
   */
  getAccessibleName(element: HTMLElement): string {
    const ariaLabel = element.getAttribute('aria-label')
    if (ariaLabel) return ariaLabel

    const ariaLabelledBy = element.getAttribute('aria-labelledby')
    if (ariaLabelledBy) {
      const referencedElements = ariaLabelledBy.split(' ')
        .map(id => document.getElementById(id))
        .filter(el => el !== null)

      return referencedElements.map(el => el!.textContent || '').join(' ').trim()
    }

    const id = element.getAttribute('id')
    if (id) {
      const label = document.querySelector(`label[for="${id}"]`)
      if (label) return label.textContent || ''
    }

    return element.textContent || ''
  },

  /**
   * Simulate screen reader announcement
   */
  announceToScreenReader(message: string, priority: 'polite' | 'assertive' = 'polite'): void {
    const announcement = document.createElement('div')
    announcement.setAttribute('aria-live', priority)
    announcement.setAttribute('aria-atomic', 'true')
    announcement.className = 'sr-only'
    announcement.textContent = message

    document.body.appendChild(announcement)

    setTimeout(() => {
      document.body.removeChild(announcement)
    }, 1000)
  },

  /**
   * Test tab order
   */
  getTabOrder(container: HTMLElement = document.body): HTMLElement[] {
    const focusableElements = container.querySelectorAll(
      'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
    ) as NodeListOf<HTMLElement>

    return Array.from(focusableElements).sort((a, b) => {
      const aIndex = a.tabIndex
      const bIndex = b.tabIndex

      // Elements with positive tabindex come first
      if (aIndex > 0 && bIndex > 0) return aIndex - bIndex
      if (aIndex > 0) return -1
      if (bIndex > 0) return 1

      // Then elements in DOM order
      const position = a.compareDocumentPosition(b)
      return position & Node.DOCUMENT_POSITION_FOLLOWING ? -1 : 1
    })
  }
}

/**
 * Quick accessibility test function
 */
export async function quickAccessibilityTest(container: HTMLElement = document.body): Promise<AccessibilityTestReport> {
  const tester = new AccessibilityTester()
  return await tester.testAccessibility(container)
}

/**
 * Console-friendly accessibility test
 */
export async function logAccessibilityReport(container: HTMLElement = document.body): Promise<void> {
  const report = await quickAccessibilityTest(container)

  console.group(`Accessibility Test Report - ${report.overall.toUpperCase()}`)
  console.log(`Score: ${report.score}/100`)
  console.log(`Passed: ${report.passedTests}/${report.totalTests}`)
  console.log(`Failed: ${report.failedTests}`)
  console.log(`Warnings: ${report.warnings}`)

  if (report.failedTests > 0) {
    console.group('Errors')
    report.results
      .filter(r => !r.passed && r.severity === 'error')
      .forEach(result => {
        console.error(`${result.rule}: ${result.message}`, result.element)
      })
    console.groupEnd()
  }

  if (report.warnings > 0) {
    console.group('Warnings')
    report.results
      .filter(r => !r.passed && r.severity === 'warning')
      .forEach(result => {
        console.warn(`${result.rule}: ${result.message}`, result.element)
      })
    console.groupEnd()
  }

  console.groupEnd()
}
