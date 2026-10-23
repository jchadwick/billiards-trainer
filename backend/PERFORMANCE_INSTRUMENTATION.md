# Python compatibility-service instrumentation

This note applies only to the retained Python compatibility service. Its
optional `/api/v1/vision/performance` endpoint and in-process timing recorder
report that service's own measurements; they do not inspect the supported Rust
server and are not a current performance-evaluation workflow. The Python
collector, comparison evaluator, and associated command wrappers have been
retired.

Use the Rust [performance and localization evaluation workflow](../docs/PERFORMANCE_AUDIT.md)
for supported measurements. Treat measurements from the Python compatibility
service as historical diagnostics while that service remains in the tree.
