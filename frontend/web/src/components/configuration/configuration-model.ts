/** PUT merges documents. Refuse deletions so the editor never promises replacement semantics. */
export function deletedPaths(baseline: unknown, draft: unknown, path = ''): string[] {
  if (!baseline || typeof baseline !== 'object' || Array.isArray(baseline) || !draft || typeof draft !== 'object' || Array.isArray(draft)) return [];
  return Object.keys(baseline).flatMap(key => {
    const next = path ? `${path}.${key}` : key;
    return !(key in draft) ? [next] : deletedPaths((baseline as Record<string, unknown>)[key], (draft as Record<string, unknown>)[key], next);
  });
}
export type Configuration = Record<string, any>;
export function parseConfiguration(text: string): Configuration {
  const value = JSON.parse(text);
  if (!value || typeof value !== 'object' || Array.isArray(value) || !Object.keys(value).length) throw new Error('Configuration must be a nonempty JSON object.');
  return value;
}
/** Strip store-owned properties using the same schema that drives Monaco. */
export function editableConfiguration(value: any, schema: any, path = ''): any {
  if (Array.isArray(value)) return value.map(item => editableConfiguration(item, schema?.items, path));
  if (!value || typeof value !== 'object') return value;
  return Object.fromEntries(Object.entries(value).filter(([key]) => !schema?.properties?.[key]?.readOnly && !(path === 'metadata' && key === 'configuration_revision')).map(([key, item]) => [key, editableConfiguration(item, schema?.properties?.[key], path ? `${path}.${key}` : key)]));
}
export function pretty(value: unknown) { return JSON.stringify(value, null, 2); }

/** Return paths for changed submitted values, without copying settings or treating omissions as deletions. */
export function changedConfigurationPaths(
  baseline: unknown,
  proposed: unknown,
  path = "",
): string[] {
  const baselineObject = baseline && typeof baseline === "object" && !Array.isArray(baseline)
    ? baseline as Record<string, unknown>
    : null;
  const proposedObject = proposed && typeof proposed === "object" && !Array.isArray(proposed)
    ? proposed as Record<string, unknown>
    : null;
  if (baselineObject && proposedObject) {
    const keys = Object.keys(proposedObject).sort();
    return keys.flatMap((key) => {
      if ((path === "metadata" || path === "/metadata") && key === "configuration_revision") return [];
      const childPath = `${path}/${key.replaceAll("~", "~0").replaceAll("/", "~1")}`;
      if (!(key in baselineObject)) return [childPath];
      return changedConfigurationPaths(baselineObject[key], proposedObject[key], childPath);
    });
  }
  if (JSON.stringify(baseline) === JSON.stringify(proposed)) return [];
  return [path || "/"];
}
