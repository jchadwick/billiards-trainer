import { vi } from 'vitest'

// API-backed modules validate configuration during import. Tests mock network
// access, so provide the same local base URL their transport expectations use.
vi.stubEnv('VITE_API_BASE_URL', 'http://localhost:8000')
