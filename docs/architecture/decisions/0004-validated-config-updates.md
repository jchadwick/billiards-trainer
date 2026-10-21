# ADR-0004: Validate and atomically persist configuration updates

- Status: accepted
- Date: 2026-10-18
- Deciders: Project maintainers
- Supersedes: None
- Superseded by: None

## Context

Configuration is consumed from a writable runtime file, but API updates
previously applied only shallow example checks and used attribute assignment
fallbacks that did not update the store. Individual writes spawned background
saves that could overlap, truncate the file, or report success before
persistence completed. Import and calibration paths did not share the update
validation rules. Using the active runtime file as the schema template would
also let a partial replacement erase supported keys and change the schema
after replacement or restart.

## Decision

Use the bundled `backend/config_schema_defaults.json` as the immutable source
for supported keys, JSON types, and schema defaults. Keep its supported keys
and defaults synchronized with the checked-in startup `config.json`, but never
derive schema rules from the active writable runtime file. Build the
configuration API schema from the bundled snapshot, reject unknown keys and
incorrect types, enforce explicit numeric and cross-field constraints, and
allow only documented dynamic maps. Validate partial changes against the
merged current configuration. The `Config` store serializes changes and writes
them to a same-directory temporary file before atomically replacing the
configured file; memory changes become visible only after persistence succeeds.
Update, import, and table calibration endpoints use this shared validation and
store API. Validation-only requests never write. Forced updates may proceed
through warnings but cannot bypass validation errors.

## Consequences

### Positive

- API documentation and runtime validation derive from one supported-key
  source.
- Failed or invalid writes leave both persisted and in-memory configuration
  unchanged.
- Imports and calibration use the same checks as ordinary partial updates.

### Negative

- Adding a supported configuration key requires updating both the immutable
  schema/default snapshot and the checked-in startup `config.json`.
- Atomic persistence is synchronous, so configuration writes wait for disk
  completion before returning.
- Only explicitly documented dynamic maps accept arbitrary keys.

## Alternatives considered

### Keep endpoint-specific validation

This would preserve divergent validation behavior across updates, imports, and
calibration, and would continue to let the schema endpoint document a different
contract from the runtime store.

### Publish memory before writing asynchronously

This can report success when disk persistence fails and can expose in-memory
state that will disappear after restart. It also allows concurrent writers to
overwrite each other's files.

## Verification

- `backend/tests/unit/test_config_validation.py` covers schema-derived type and
  key checks, cross-field rules, validation-only behavior, section updates,
  persistence/reload, atomic failure rollback, and partial replacement without
  mutating the immutable schema source.
- Focused backend tests run with `pytest --no-cov`.
- Black and Ruff check the changed Python modules.
- `manage-architecture-decisions/scripts/validate_adrs.py` validates the ADR.
