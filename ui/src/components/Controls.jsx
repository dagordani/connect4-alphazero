// Opponent settings and game buttons.

function checkpointLabel(ck) {
  if (ck.iter == null) return ck.name.replace(/_/g, ' ');
  const m = ck.name.match(/_(\d{4}-\d{2}-\d{2})_(\d{2})h(\d{2})m/);
  return m ? `iter ${ck.iter} · ${m[1]} ${m[2]}:${m[3]}` : `iter ${ck.iter}`;
}

function groupByDir(checkpoints) {
  const groups = new Map();
  for (const ck of checkpoints) {
    if (!groups.has(ck.dir)) groups.set(ck.dir, []);
    groups.get(ck.dir).push(ck);
  }
  return [...groups.entries()];
}

export default function Controls({
  checkpoints,
  checkpoint,
  onCheckpoint,
  human,
  onHuman,
  simulations,
  onSimulations,
  sample,
  onSample,
  autoReply,
  onAutoReply,
  busy,
  canUndo,
  canEngineMove,
  onNewGame,
  onUndo,
  onEngineMove,
  onHint,
}) {
  const current = checkpoints.find((c) => c.id === checkpoint);

  return (
    <section className="panel">
      <h2>Opponent</h2>

      <label htmlFor="checkpoint">Checkpoint</label>
      <select id="checkpoint" value={checkpoint} disabled={busy} onChange={(e) => onCheckpoint(e.target.value)}>
        {groupByDir(checkpoints).map(([dir, cks]) => (
          <optgroup key={dir} label={dir}>
            {cks.map((ck) => (
              <option key={ck.id} value={ck.id} disabled={!ck.ok} title={ck.ok ? ck.id : ck.error}>
                {checkpointLabel(ck)}
                {ck.ok ? '' : ' (incompatible)'}
              </option>
            ))}
          </optgroup>
        ))}
      </select>
      {current?.config && (
        <div className="meta">
          {current.config.num_channels} channels × {current.config.num_blocks} blocks
        </div>
      )}

      <label>You play</label>
      <div className="segmented">
        {['red', 'yellow'].map((side) => (
          <button
            key={side}
            className={human === side ? 'on' : ''}
            disabled={busy}
            onClick={() => human !== side && onHuman(side)}
          >
            <span className={`chip ${side}`} />
            {side === 'red' ? 'Red · first' : 'Yellow · second'}
          </button>
        ))}
      </div>

      <label htmlFor="sims">Simulations per move</label>
      <input
        id="sims"
        type="number"
        min="0"
        max="10000"
        step="25"
        value={simulations}
        onChange={(e) => onSimulations(Math.max(0, Math.min(10000, parseInt(e.target.value, 10) || 0)))}
      />
      <div className="meta">0 = no search: the network's policy picks the move</div>

      <label>Engine move choice</label>
      <div className="segmented">
        {[
          [false, 'Best move'],
          [true, 'Sample'],
        ].map(([value, text]) => (
          <button key={text} className={sample === value ? 'on' : ''} onClick={() => onSample(value)}>
            {text}
          </button>
        ))}
      </div>
      <div className="meta">
        {sample
          ? 'Random, in proportion to the visit counts (to the policy with 0 simulations)'
          : 'Always the most-visited move (highest prior with 0 simulations)'}
      </div>

      <label className="check">
        <input type="checkbox" checked={autoReply} onChange={(e) => onAutoReply(e.target.checked)} />
        engine replies automatically
      </label>

      <div className="buttons">
        <button onClick={onNewGame} disabled={busy}>New game</button>
        <button onClick={onUndo} disabled={busy || !canUndo}>Undo</button>
        <button onClick={onEngineMove} disabled={busy || !canEngineMove}>Engine move</button>
        <button onClick={onHint} disabled={busy || !canEngineMove}>Hint</button>
      </div>
    </section>
  );
}
