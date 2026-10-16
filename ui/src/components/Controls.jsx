// Opponent settings and game buttons, in a fixed grid of named slots (see .controls in
// styles.css) so fields keep their place when the opponent changes.

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
    <section className="panel controls">
      <div className="field f-opponent">
        <label>Opponent</label>
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
          {neural ? 'Tree search guided by the trained network' : 'Pure tree search with random rollouts, no network'}
        </div>
      </div>

      <div className="field f-level">
        <label>Difficulty</label>
        {neural ? (
          <>
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
            {current && <div className="meta">Network trained on {current.games.toLocaleString()} self-play games</div>}
          </>
        ) : (
          <div className="meta">No network: the search gets stronger with more simulations per move.</div>
        )}
      </div>

      <div className="field f-side">
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
      </div>

      <div className="field f-sims">
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
          {neural ? '0 = no search: the policy picks' : 'One node + one random game each'}
        </div>
      </div>

      <div className="field f-choice">
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
          {sample ? 'In proportion to the visit counts' : 'Always the most-visited move'}
          {neural && ` (${sample ? 'the policy' : 'highest prior'} with 0 simulations)`}
        </div>
      </div>

      <div className="field f-game">
        <label>Game</label>
        <div className="buttons">
          <button onClick={onNewGame} disabled={busy}>New game</button>
          <button onClick={onUndo} disabled={busy || !canUndo}>Undo</button>
          <button onClick={onEngineMove} disabled={busy || !canEngineMove}>Engine move</button>
          <button onClick={onHint} disabled={busy || !canEngineMove}>Hint</button>
        </div>
        <label className="check">
          <input type="checkbox" checked={autoReply} onChange={(e) => onAutoReply(e.target.checked)} />
          engine replies automatically
        </label>
      </div>
    </section>
  );
}
