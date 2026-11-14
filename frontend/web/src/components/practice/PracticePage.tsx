import { useMemo, useState } from 'react';
import type { MouseEvent } from 'react';
import { TrainingTabs } from './TrainingTabs';
import {
  PRACTICE_TABLE_DEFAULTS,
  standardPockets,
  suggestDirectPots,
  type PracticeBall,
  type PracticeTable,
} from '../video/shot-prediction';

type PlacementMode = 'cue' | 'object';

/** A self-contained table-plane coaching sandbox. It never reads or writes live state. */
export function PracticePage() {
  const [table, setTable] = useState<PracticeTable>({ ...PRACTICE_TABLE_DEFAULTS });
  const [cueBall, setCueBall] = useState<PracticeBall | null>({ id: 'cue', x: table.width * 0.17, y: table.height * 0.5, radius: table.ballRadius });
  const [cueConfirmed, setCueConfirmed] = useState(false);
  const [balls, setBalls] = useState<PracticeBall[]>([
    { id: 'ball-1', x: table.width * 0.53, y: table.height * 0.5, radius: table.ballRadius },
    { id: 'ball-2', x: table.width * 0.72, y: table.height * 0.31, radius: table.ballRadius },
    { id: 'ball-3', x: table.width * 0.78, y: table.height * 0.68, radius: table.ballRadius },
  ]);
  const [placementMode, setPlacementMode] = useState<PlacementMode>('cue');
  const [placementX, setPlacementX] = useState(Math.round(table.width / 2));
  const [placementY, setPlacementY] = useState(Math.round(table.height / 2));
  const [targetId, setTargetId] = useState('');
  const [pocketId, setPocketId] = useState('');
  const [nextBallNumber, setNextBallNumber] = useState(4);

  const pockets = useMemo(() => standardPockets(table), [table]);
  const candidates = useMemo(() => cueConfirmed
    ? suggestDirectPots(cueBall, balls, table, {
      ...(targetId ? { targetId } : {}),
      ...(pocketId ? { pocketId } : {}),
    })
    : [], [cueBall, cueConfirmed, balls, table, targetId, pocketId]);
  const selected = candidates[0];

  function updateTable(update: Partial<PracticeTable>) {
    if (Object.values(update).some((value) => !Number.isFinite(value) || value <= 0)) return;
    const next = { ...table, ...update };
    if (next.width <= next.ballRadius * 2 || next.height <= next.ballRadius * 2) return;
    setTable(next);
    if (update.ballRadius !== undefined) {
      setCueBall((current) => current ? { ...current, radius: next.ballRadius } : null);
      setBalls((current) => current.map((ball) => ({ ...ball, radius: next.ballRadius })));
    }
  }

  function placeAt(xValue: number, yValue: number) {
    const x = Math.max(table.ballRadius, Math.min(table.width - table.ballRadius, xValue));
    const y = Math.max(table.ballRadius, Math.min(table.height - table.ballRadius, yValue));
    setPlacementX(Math.round(x));
    setPlacementY(Math.round(y));
    if (placementMode === 'cue') {
      setCueBall({ id: 'cue', x, y, radius: table.ballRadius });
      setCueConfirmed(false);
    } else {
      const ball = { id: `ball-${nextBallNumber}`, x, y, radius: table.ballRadius };
      setBalls((current) => [...current, ball]);
      setNextBallNumber((number) => number + 1);
      setTargetId(ball.id);
    }
  }

  function placeBall(event: MouseEvent<SVGSVGElement>) {
    // The screen transform includes the border and viewBox letterboxing. A
    // bounding-box ratio alone places balls away from the pointer on tall views.
    const inverse = event.currentTarget.getScreenCTM()?.inverse();
    if (!inverse) return;
    placeAt(inverse.a * event.clientX + inverse.c * event.clientY + inverse.e,
      inverse.b * event.clientX + inverse.d * event.clientY + inverse.f);
  }

  function removeBall(id: string) {
    setBalls((current) => current.filter((ball) => ball.id !== id));
    if (targetId === id) setTargetId('');
  }

  return (
    <section className="practice-page" aria-labelledby="practice-title">
      <TrainingTabs active="practice" />
      <header className="practice-heading">
        <div>
          <p className="practice-eyebrow">Practice sandbox · simulated</p>
          <h1 id="practice-title">Plan a direct pot</h1>
          <p>Place the balls on a physical table layout, confirm the cue ball, then compare direct-pot options.</p>
        </div>
        <span className="practice-simulation-badge">SIMULATION · NOT LIVE DETECTION</span>
      </header>

      <div className="practice-grid">
        <section className="practice-table-card" aria-label="Editable practice table">
          <div className="practice-table-tools" role="group" aria-label="Ball placement tool">
            <button type="button" className={placementMode === 'cue' ? 'selected' : ''} aria-pressed={placementMode === 'cue'} onClick={() => setPlacementMode('cue')}>Place cue ball</button>
            <button type="button" className={placementMode === 'object' ? 'selected' : ''} aria-pressed={placementMode === 'object'} onClick={() => setPlacementMode('object')}>Add object ball</button>
            <button type="button" onClick={() => { setBalls([]); setTargetId(''); }}>Clear object balls</button>
          </div>
          <p className="practice-instructions" id="practice-instructions">{placementMode === 'cue' ? 'Click the table to position the cue ball.' : 'Click the table to add an object ball. Select a ball below to remove it.'}</p>
          <form className="practice-coordinate-entry" aria-label="Place ball by table coordinates" onSubmit={(event) => { event.preventDefault(); placeAt(placementX, placementY); }}>
            <label>Table X (mm)<input type="number" step="any" min={table.ballRadius} max={table.width - table.ballRadius} value={placementX} onChange={(event) => setPlacementX(Number(event.target.value))} /></label>
            <label>Table Y (mm)<input type="number" step="any" min={table.ballRadius} max={table.height - table.ballRadius} value={placementY} onChange={(event) => setPlacementY(Number(event.target.value))} /></label>
            <button type="submit">Place selected ball</button>
          </form>
          <svg
            className="practice-table"
            viewBox={`0 0 ${table.width} ${table.height}`}
            role="img"
            aria-label="Simulated pool table. Click to place the selected ball type."
            aria-describedby="practice-instructions"
            onClick={placeBall}
            data-testid="practice-table"
          >
            <rect x="0" y="0" width={table.width} height={table.height} rx={table.ballRadius * 1.5} className="practice-cloth" />
            {pockets.map((pocket) => <circle key={pocket.id} cx={pocket.x} cy={pocket.y} r={table.ballRadius * 1.9} className="practice-pocket" />)}
            {selected && <>
              <line x1={selected.cueBall.x} y1={selected.cueBall.y} x2={selected.contactCenter.x} y2={selected.contactCenter.y} className="practice-cue-path" />
              <line x1={selected.objectPathStart.x} y1={selected.objectPathStart.y} x2={selected.objectPathEnd.x} y2={selected.objectPathEnd.y} className="practice-object-path" />
              <circle cx={selected.contactCenter.x} cy={selected.contactCenter.y} r={selected.cueBall.radius} className="practice-ghost-ball" />
              <circle cx={selected.contactPoint.x} cy={selected.contactPoint.y} r={table.ballRadius * 0.22} className="practice-contact-point" />
            </>}
            {balls.map((ball, index) => <g key={ball.id} onClick={(event) => { event.stopPropagation(); setTargetId(ball.id); }}>
              <circle cx={ball.x} cy={ball.y} r={ball.radius} className={`practice-ball ${targetId === ball.id ? 'target' : ''}`} />
              <text x={ball.x} y={ball.y} className="practice-ball-label">{index + 1}</text>
            </g>)}
            {cueBall && <g>
              <circle cx={cueBall.x} cy={cueBall.y} r={cueBall.radius} className="practice-cue-ball" />
              <text x={cueBall.x} y={cueBall.y} className="practice-cue-label">C</text>
            </g>}
          </svg>
          <div className="practice-ball-list" aria-label="Simulated object balls">
            {balls.map((ball, index) => <div className="practice-ball-row" key={ball.id}>
              <button type="button" className={targetId === ball.id ? 'selected' : ''} aria-pressed={targetId === ball.id} onClick={() => setTargetId(ball.id)}>
                Ball {index + 1} · {Math.round(ball.x)}, {Math.round(ball.y)}
              </button>
              <button type="button" className="practice-remove-ball" aria-label={`Remove ball ${index + 1}`} onClick={() => removeBall(ball.id)}>Remove</button>
            </div>)}
            {balls.length === 0 && <p>No object balls placed. Choose “Add object ball” and click the table.</p>}
          </div>
          <details className="practice-dimensions">
            <summary>Table dimensions and ball size</summary>
            <div>
              <label>Length (mm)<input type="number" step="any" min="100" value={table.width} onChange={(event) => updateTable({ width: Number(event.target.value) })} /></label>
              <label>Width (mm)<input type="number" step="any" min="100" value={table.height} onChange={(event) => updateTable({ height: Number(event.target.value) })} /></label>
              <label>Ball radius (mm)<input type="number" step="any" min="1" value={table.ballRadius} onChange={(event) => updateTable({ ballRadius: Number(event.target.value) })} /></label>
            </div>
            <small>Offline defaults follow the existing physical localization configuration.</small>
          </details>
        </section>

        <aside className="practice-coach-card" aria-label="Shot coaching">
          <h2>Choose the shot</h2>
          <p className="practice-muted">All positions are editable simulation inputs. No camera or server data is used.</p>
          <label className="practice-control">Cue ball
            <span className="practice-cue-confirmation">{cueBall ? `Placed at ${Math.round(cueBall.x)}, ${Math.round(cueBall.y)}` : 'Not placed'}</span>
          </label>
          <button type="button" className="practice-confirm" disabled={!cueBall} aria-pressed={cueConfirmed} onClick={() => setCueConfirmed((confirmed) => !confirmed)}>
            {cueConfirmed ? 'Cue ball confirmed · click to reset' : 'Confirm cue ball'}
          </button>
          <label className="practice-control" htmlFor="practice-target">Target ball
            <select id="practice-target" value={targetId} onChange={(event) => setTargetId(event.target.value)}>
              <option value="">Automatic suggestion</option>
              {balls.map((ball, index) => <option key={ball.id} value={ball.id}>Ball {index + 1}</option>)}
            </select>
          </label>
          <label className="practice-control" htmlFor="practice-pocket">Pocket
            <select id="practice-pocket" value={pocketId} onChange={(event) => setPocketId(event.target.value)}>
              <option value="">Automatic suggestion</option>
              {pockets.map((pocket) => <option key={pocket.id} value={pocket.id}>{pocket.label}</option>)}
            </select>
          </label>

          {!cueConfirmed ? <div className="practice-coach-result" role="status"><strong>Confirm the cue ball to calculate a shot.</strong><p>The aim line appears only after you confirm this simulated position.</p></div>
            : selected ? <div className="practice-coach-result" role="status" aria-live="polite">
              <p className="practice-result-label">Suggested direct pot</p>
              <strong>Ball {balls.findIndex((ball) => ball.id === selected.targetId) + 1} → {selected.pocket.label}</strong>
              <p>{(selected.cutAngleRadians * 180 / Math.PI).toFixed(1)}° cut · {(selected.totalTravel / 1000).toFixed(2)} m combined travel</p>
              <p>Aim the cue ball center through the marked ghost-ball center. The dot marks the object-ball contact point; the green line estimates its initial path to the pocket.</p>
              <p className="practice-muted">Direct-pot geometry only. Spin, banks, combinations, and post-contact physics are not modeled.</p>
            </div> : <div className="practice-coach-result practice-no-shot" role="status" aria-live="polite">
              <strong>No supported direct pot for this selection.</strong>
              <p>Check the ball placement or choose automatic target and pocket. Blocked paths and out-of-bounds centers are excluded.</p>
            </div>}
        </aside>
      </div>
    </section>
  );
}
