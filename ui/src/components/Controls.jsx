// Opponent settings and game buttons.

const OPPONENTS = [
  ['neural', 'Neural MCTS'],
  ['classical', 'Classical MCTS'],
];

export default function Controls({
  opponent,
  onOpponent,
  levels,
  level,
  onLevel,
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
  const current = levels.find((lv) => lv.id === level);
  const neural = opponent === 'neural';
  const anyLevel = levels.some((lv) => lv.ok);

  return (
    <section className="panel">
      <h2>Opponent</h2>

      <div className="segmented">
        {OPPONENTS.map(([value, text]) => (
          <button
            key={value}
            className={opponent === value ? 'on' : ''}
            disabled={busy || (value === 'neural' && !anyLevel)}
            title={value === 'neural' && !anyLevel ? 'no difficulty level could be loaded' : undefined}
            onClick={() => opponent !== value && onOpponent(value)}
          >
            {text}
          </button>
        ))}
      </div>
      <div className="meta">
        {neural
          ? 'Tree search guided by the trained network'
          : 'Pure tree search with random rollouts: no neural network'}
      </div>

      {neural && (
        <>
          <label>Difficulty</label>
          <div className="segmented">
            {levels.map((lv) => (
              <button
                key={lv.id}
                className={level === lv.id ? 'on' : ''}
                disabled={busy || !lv.ok}
                title={lv.ok ? undefined : lv.error}
                onClick={() => level !== lv.id && onLevel(lv.id)}
              >
                {lv.name}
              </button>
            ))}
          </div>
          {current && (
            <div className="meta">Network trained on {current.games.toLocaleString()} self-play games</div>
          )}
        </>
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
        min={neural ? 0 : 1}
        max="10000"
        step="25"
        value={simulations}
        onChange={(e) =>
          onSimulations(Math.max(neural ? 0 : 1, Math.min(10000, parseInt(e.target.value, 10) || 0)))
        }
      />
      <div className="meta">
        {neural
          ? "0 = no search: the network's policy picks the move"
          : 'Each simulation adds one node and plays one random game from it'}
      </div>

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
          ? `Random, in proportion to the visit counts${neural ? ' (to the policy with 0 simulations)' : ''}`
          : `Always the most-visited move${neural ? ' (highest prior with 0 simulations)' : ''}`}
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
