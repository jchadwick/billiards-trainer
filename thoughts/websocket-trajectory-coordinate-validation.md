# WebSocket trajectory coordinate validation

The trajectory broadcaster now enforces the scale-aware image-coordinate
contract at the transport boundary. Every line must provide `start` and `end`
points with finite numeric `x`/`y` values and exactly two positive finite scale
factors. A collision's `position`, when present, must meet the same contract.
Legacy collision objects may keep top-level `x`/`y` for older consumers, but
that pair must also include scale metadata; a valid nested `position` does not
implicitly scale duplicate coordinates.
NumPy numeric scalars are accepted before JSON encoding; booleans, missing scale,
non-finite values, and zero or negative factors are rejected. Validation does
not reorder or rewrite the payload.

Focused validator tests and the WebSocket delivery suite passed (19 tests in
total). Ruff, Black, isort, and strict mypy on `broadcaster.py` also passed. The
real-server replay exposed a separate integration serializer issue: collision
objects carry redundant top-level bare `x` and `y` beside their correctly
scaled `position`. The broadcaster now rejects that malformed payload until
the integration serializer removes the redundant fields or gives them their
own scale. That follow-up belongs to the integration serializer stage; this
WebSocket-only change leaves it to that owner.
