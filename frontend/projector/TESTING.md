# Projector contract checks

The retained tests validate the geometry contract without opening a camera,
network socket, or display. They do not establish backend integration or
rendering correctness.

From the repository root, run:

```bash
lua frontend/projector/test_coordinate_contract.lua
luac -p frontend/projector/core/coordinate_contract.lua
```

The test covers normalized table points, boundaries and bounds, projector
pixel radii, motion vectors, event payloads, and meter-tagged physical records.
Coordinate-bearing packets without an owning space and identity scale are
rejected before reaching callbacks. The standalone projector has no production
data publisher, so there is no synthetic sender or fake end-to-end test.
