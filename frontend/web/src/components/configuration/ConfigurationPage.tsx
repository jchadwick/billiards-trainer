import { useEffect, useRef, useState } from 'react';
import type { editor as MonacoEditor } from 'monaco-editor';
import axios from 'axios';
import client from '../../api/axios-client';
import { Button } from '../ui/button';
import { MonacoConfigurationEditor } from './MonacoConfigurationEditor';
import type { EditorHandle } from './MonacoConfigurationEditor';
import { changedConfigurationPaths, deletedPaths, editableConfiguration, parseConfiguration, pretty } from './configuration-model';
import type { Configuration } from './configuration-model';
import { useRootStore } from '../../stores/context';

type Validation = { success: boolean; validation_errors: { field?: string; message: string }[]; warnings?: unknown[]; restart_required?: boolean; updated_fields?: unknown[]; configuration_revision?: string | number };
type Loaded = { values: Configuration; schema: any; loadedAt: string };
function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}
// Drafts survive route changes in this tab, but never persist sensitive settings to storage.
let retained: { loaded: Loaded; draft: string } | undefined;
function errorMessage(error: unknown) {
  if (axios.isAxiosError(error)) return error.response?.data?.message || error.message;
  return error instanceof Error ? error.message : String(error);
}
export function ConfigurationPage() {
  const rootStore = useRootStore();
  const [loaded, setLoaded] = useState<Loaded | undefined>(retained?.loaded);
  const [draft, setDraft] = useState(retained?.draft ?? '');
  const [busy, setBusy] = useState('');
  const [message, setMessage] = useState('');
  const [validation, setValidation] = useState<Validation>();
  const [markers, setMarkers] = useState<MonacoEditor.IMarker[]>([]);
  const [review, setReview] = useState(false);
  const [confirmReload, setConfirmReload] = useState(false);
  const [overrideWarnings, setOverrideWarnings] = useState(false);
  const [lastApplied, setLastApplied] = useState('');
  const editor = useRef<EditorHandle | null>(null);
  const input = useRef<HTMLInputElement>(null);
  const dirty = !!loaded && draft !== pretty(loaded.values);
  useEffect(() => { if (loaded) retained = { loaded, draft }; }, [loaded, draft]);
  useEffect(() => {
    const guard = (event: BeforeUnloadEvent) => { if (dirty) { event.preventDefault(); event.returnValue = ''; } };
    window.addEventListener('beforeunload', guard); return () => window.removeEventListener('beforeunload', guard);
  }, [dirty]);
  async function fetchActive() {
    const [config, schema] = await Promise.all([client.get<{ values: Configuration }>('/api/v1/config'), client.get('/api/v1/config/schema')]);
    if (!isRecord(config.data) || !isRecord(config.data.values) || !isRecord(schema.data)) {
      throw new Error('Configuration API returned an unexpected response. Check that the backend is reachable and supports configuration.');
    }
    return { values: config.data.values as Configuration, schema: schema.data, loadedAt: new Date().toLocaleTimeString() };
  }
  async function reload() {
    setConfirmReload(false); setBusy('Loading active configuration…'); setMessage('');
    try { const next = await fetchActive(); setLoaded(next); setDraft(pretty(next.values)); setValidation(undefined); setReview(false); }
    catch (error) { setMessage(errorMessage(error)); } finally { setBusy(''); }
  }
  useEffect(() => { if (!retained) void reload(); }, []);
  function update(text: string) { setDraft(text); setValidation(undefined); setReview(false); setOverrideWarnings(false); setMessage(''); }
  function payload() {
    const parsed = parseConfiguration(draft);
    if (!loaded) throw new Error('Load configuration before validating.');
    const removed = deletedPaths(loaded.values, parsed);
    if (removed.length) throw new Error(`The configuration API merges updates and cannot delete these keys: ${removed.join(', ')}. Restore them before applying.`);
    if (parsed.metadata?.configuration_revision !== loaded.values.metadata?.configuration_revision) throw new Error('metadata.configuration_revision is read-only. Restore its loaded value.');
    return editableConfiguration(parsed, loaded.schema);
  }
  async function validate(toReview = false) {
    setBusy('Validating configuration…'); setMessage('');
    try {
      const response = await client.getInstance().put<Validation>('/api/v1/config', payload(), { params: { validate_only: true } });
      setValidation(response.data);
      if (response.data.success && toReview) setReview(true);
      else if (response.data.success) setMessage('Configuration is valid · nothing applied');
    } catch (error) { setMessage(errorMessage(error)); } finally { setBusy(''); }
  }
  async function apply() {
    setBusy('Applying configuration…'); setMessage('');
    try {
      const data = payload();
      const current = await fetchActive();
      if (pretty(current.values) !== pretty(loaded?.values)) { setMessage('Configuration changed elsewhere. Your draft is preserved. Reload active configuration before reviewing again.'); setReview(false); return; }
      const result = await client.getInstance().put<Validation>('/api/v1/config', data, { params: { force_update: overrideWarnings } });
      if (!result.data.success) { setValidation(result.data); setMessage('Configuration was rejected. Your draft is preserved.'); return; }
      rootStore.recordConfigurationApplied({
        changedFieldPaths: (Array.isArray(result.data.updated_fields)
          ? result.data.updated_fields.filter((path): path is string => typeof path === 'string')
          : changedConfigurationPaths(loaded?.values, data)),
        restartRequired: result.data.restart_required === true,
        previousRevision: loaded?.values.metadata?.configuration_revision,
        appliedRevision: result.data.configuration_revision,
      });
      rootStore.notifyVideoConfigurationApplied();
      setLastApplied(new Date().toLocaleTimeString());
      setMessage(result.data.restart_required ? 'Applied · restart required. Arrange restart outside this interface.' : 'Applied successfully');
      // Re-read the merged canonical document, including its newly assigned revision.
      try { const next = await fetchActive(); setLoaded(next); setDraft(pretty(next.values)); setReview(false); setValidation(undefined); }
      catch { setMessage('Apply was confirmed, but the active document could not be reloaded. Your draft is preserved. Reload active before applying again.'); setReview(false); }
    } catch (error) { setMessage(`Apply was not confirmed: ${errorMessage(error)}. Your draft is preserved; reload active to verify before retrying.`); }
    finally { setBusy(''); }
  }
  async function importFile(file?: File) {
    if (!file) return;
    try { const text = await file.text(); parseConfiguration(text); if (!dirty || window.confirm("Replace the current unapplied draft with this imported configuration?")) update(text); }
    catch (error) { setMessage(`Import could not be completed: ${errorMessage(error)}. Your previous draft is preserved.`); }
  }
  function download() {
    const url = URL.createObjectURL(new Blob([draft], { type: 'application/json' }));
    const link = document.createElement('a'); link.href = url; link.download = 'billiards-trainer-draft.json'; link.click(); URL.revokeObjectURL(url);
  }
  return <div className="space-y-5">
    <div className="flex flex-wrap items-center justify-between gap-4"><h1 className="text-2xl font-semibold">Configuration</h1><div className="flex flex-wrap gap-6 text-xs text-muted-foreground"><span>Active revision <code className="ml-2 text-foreground">{loaded?.values.metadata?.configuration_revision ?? 'Not available'}</code></span><span>Last loaded / applied <code className="ml-2">{loaded?.loadedAt ?? 'Not available'} / {lastApplied || 'Not available'}</code></span><span>JSON · schema-backed</span></div></div>
    {message && <div role="status" className="rounded-md border border-border bg-card p-4 text-sm">{message}</div>}
    {confirmReload && <div role="alertdialog" aria-label="Replace draft with active configuration?" className="rounded-md border border-border bg-card p-4"><p className="mb-3">Replace draft with active configuration? This discards unapplied edits.</p><div className="flex gap-2"><Button onClick={() => setConfirmReload(false)}>Keep editing</Button><Button onClick={() => void reload()}>Reload active</Button></div></div>}
    <section className="overflow-hidden rounded-lg border border-border bg-card">
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-border p-3"><div className="flex flex-wrap gap-2">
        <Button disabled={!!busy} onClick={() => dirty ? setConfirmReload(true) : void reload()}>Reload active</Button>
        <Button disabled={!!busy || !loaded} onClick={() => void validate()}>Validate</Button>
        <Button disabled={!!busy} onClick={() => input.current?.click()}>Import</Button>
        <Button disabled={!loaded} onClick={download}>Export</Button>
        <Button className="highlight-issue" disabled title="The API reset action applies defaults immediately; a draft-only defaults source is not exposed.">Reset defaults</Button>
        <input ref={input} className="sr-only" aria-label="Import configuration JSON" type="file" accept="application/json,.json" onChange={event => { void importFile(event.target.files?.[0]); event.target.value = ''; }} />
      </div><div className="flex gap-2">{review ? <><Button disabled={!!busy} onClick={() => setReview(false)}>Back to editor</Button><Button disabled={!!busy || !dirty || (!!validation?.warnings?.length && !overrideWarnings)} onClick={() => void apply()}>Apply configuration</Button></> : <Button disabled={!!busy || !dirty} onClick={() => void validate(true)}>Review changes</Button>}</div></div>
      {busy && <p className="border-b border-border px-4 py-2 text-sm" role="status">{busy}</p>}
      <div className="flex flex-wrap justify-between gap-3 border-b border-border px-4 py-3 text-xs"><strong>{review ? 'Review configuration changes' : 'configuration.json'} {dirty && <span className="text-amber-400">● Unsaved edits</span>}</strong><span className="text-muted-foreground">Backend JSON schema</span><div className="flex gap-4">{!review && <><button onClick={() => editor.current?.getAction('actions.find')?.run()}>Find / replace</button><button onClick={() => editor.current?.getAction('editor.action.formatDocument')?.run()}>Format</button></>}</div></div>
      {review && <div className="space-y-2 border-b border-border p-4 text-sm"><p>Loaded configuration → proposed configuration · not applied</p><p className="text-muted-foreground">Validation passed. Apply merges these settings into the active configuration. The API does not expose a browser revision precondition.</p>{!!validation?.warnings?.length && <label className="flex gap-2"><input type="checkbox" checked={overrideWarnings} onChange={event => setOverrideWarnings(event.target.checked)} />Apply despite the reported warnings</label>}</div>}
      {loaded ? <MonacoConfigurationEditor key={review ? 'review' : 'edit'} value={draft} original={review ? pretty(loaded.values) : undefined} schema={loaded.schema} readOnly={!!busy} onChange={update} onMarkers={setMarkers} onMount={value => { editor.current = value; }} /> : <p className="p-8 text-sm text-muted-foreground">{busy || 'Configuration unavailable. Reload active to retry.'}</p>}
      <div className="flex justify-between border-t border-border px-4 py-2 text-xs text-muted-foreground"><span>{review ? 'Read-only diff review' : 'JSON · UTF-8 · 2-space indentation'}</span><span>Schema source: /api/v1/config/schema</span></div>
      <details open className="border-t border-border"><summary className="cursor-pointer px-4 py-3 text-sm font-semibold">Problems ({validation ? validation.validation_errors.length : markers.length})</summary><div className="space-y-2 border-t border-border p-4 text-sm">
        {validation ? validation.validation_errors.length ? validation.validation_errors.map((issue, index) => <p key={index} className="text-red-300">{issue.field}: {issue.message}</p>) : <p>No validation errors. Validation does not apply configuration.</p> : markers.length ? markers.map((issue, index) => <button className="block text-left text-amber-300" key={index} onClick={() => { editor.current?.revealLineInCenter(issue.startLineNumber); editor.current?.setPosition({ lineNumber: issue.startLineNumber, column: issue.startColumn }); editor.current?.focus(); }}>{issue.message} · Ln {issue.startLineNumber}</button>) : <p className="text-muted-foreground">No editor problems. Use Validate for backend validation.</p>}
        {validation?.warnings?.map((warning, index) => <p key={index} className="text-amber-300">Warning: {typeof warning === 'string' ? warning : pretty(warning)}</p>)}
      </div></details>
    </section>
    <p className="text-xs text-muted-foreground">The backend owns schema and configuration. Editor contents and the loaded baseline remain separate until apply is confirmed.</p>
  </div>;
}
