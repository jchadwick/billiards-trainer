import Editor, { DiffEditor, loader } from '@monaco-editor/react';
import * as monaco from 'monaco-editor/editor/editor.api.js';
import { jsonDefaults } from 'monaco-editor/languages/features/json/register.js';
import EditorWorker from 'monaco-editor/editor/editor.worker.js?worker';
import JsonWorker from 'monaco-editor/language/json/json.worker.js?worker';

// Keep both editor and JSON validation workers local; installations need no CDN access.
self.MonacoEnvironment = { getWorker: (_moduleId, label) => label === 'json' ? new JsonWorker() : new EditorWorker() };
loader.config({ monaco });
export type EditorHandle = monaco.editor.IStandaloneCodeEditor;
export function MonacoConfigurationEditor({ value, original, schema, readOnly = false, onChange, onMarkers, onMount }: {
  value: string; original?: string; schema: object; readOnly?: boolean; onChange: (value: string) => void;
  onMarkers: (markers: monaco.editor.IMarker[]) => void; onMount: (editor: EditorHandle) => void;
}) {
  function configure() {
    jsonDefaults.setDiagnosticsOptions({ validate: true, allowComments: false, enableSchemaRequest: false,
      schemas: [{ uri: 'inmemory://billiards/configuration-schema.json', fileMatch: ['*'], schema }] });
  }
  const options: monaco.editor.IStandaloneEditorConstructionOptions = { readOnly, automaticLayout: true, minimap: { enabled: false }, fontSize: 14, lineHeight: 24, tabSize: 2, scrollBeyondLastLine: false, padding: { top: 18 }, wordWrap: 'on', ariaLabel: 'Backend configuration JSON editor' };
  return original === undefined
    ? <Editor height="60vh" language="json" theme="vs-dark" value={value} beforeMount={configure} onMount={onMount} onChange={text => onChange(text ?? '')} onValidate={onMarkers} options={options} loading={<p className="p-6">Loading configuration editor…</p>} />
    : <DiffEditor height="60vh" language="json" theme="vs-dark" original={original} modified={value} beforeMount={configure} options={{ ...options, readOnly: true, originalEditable: false }} loading={<p className="p-6">Loading change review…</p>} />;
}
