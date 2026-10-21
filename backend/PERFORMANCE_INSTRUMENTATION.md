# Performance audit

The repeatable opt-in workflow, configuration defaults, live collector, recorded
input benchmark, metrics, limitations, scenario matrix, and investigation
criteria are documented in [Performance Audit](../docs/PERFORMANCE_AUDIT.md).

The existing `/api/v1/vision/performance` profiler is part of that audit path;
the legacy `vision.performance.enable_profiling` setting does not activate
collectors. Use `performance_audit.enabled` or launch the backend with
`--performance-audit`, then enable the desired state, trajectory, or viewer
domain.
