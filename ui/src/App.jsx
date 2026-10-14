import { useEffect, useRef, useState } from 'react';

import { api } from './api.js';
import Board from './components/Board.jsx';
import Controls from './components/Controls.jsx';
import SearchPanel from './components/SearchPanel.jsx';
import { capitalize } from './game.js';

// The API is stateless: this component owns the list of columns played (`moves`)
// and sends it with every request. `game` is the server's description of the
// position after those moves.

// Default opponent: the newest loadable checkpoint (last folder, highest iteration).
function defaultCheckpoint(checkpoints) {
  const ok = checkpoints.filter((c) => c.ok);
  return ok.length ? ok[ok.length - 1].id : '';
}

export default function App() {
  const [checkpoints, setCheckpoints] = useState([]);
  const [settings, setSettings] = useState({ checkpoint: '', human: 'red', simulations: 100, sample: false });
  const [autoReply, setAutoReply] = useState(true);
  const [moves, setMoves] = useState([]);
  const [game, setGame] = useState(null);
  const [analysis, setAnalysis] = useState(null);
  const [animate, setAnimate] = useState(false); // drop animation only for new moves, not undo
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const busyRef = useRef(false);
  const started = useRef(false);

  // Serialise server calls: a search can take a while and clicks arrive fast.
  async function run(task) {
    if (busyRef.current) return;
    busyRef.current = true;
    setBusy(true);
    setError('');
    try {
      await task();
    } catch (e) {
      setError(e.message);
    } finally {
      busyRef.current = false;
      setBusy(false);
    }
  }

  function show(newMoves, state, newAnalysis, isNewMove) {
    setMoves(newMoves);
    setGame(state);
    setAnalysis(newAnalysis);
    setAnimate(isNewMove);
  }

  async function engineMove(current, s) {
    const res = await api.move(current, s.checkpoint, s.simulations, s.sample);
    show([...current, res.move], res.state, res.analysis, true);
  }

  // Start over; the engine moves first when you play yellow.
  async function newGame(s) {
    const state = await api.state([]);
    show([], state, null, false);
    if (s.checkpoint && state.turn !== s.human) await engineMove([], s);
  }

  // Changing the opponent or your side starts a new game with the new settings.
  function changeSettings(patch, restart) {
    const next = { ...settings, ...patch };
    setSettings(next);
    if (restart) run(() => newGame(next));
  }

  useEffect(() => {
    if (started.current) return; // StrictMode runs effects twice in development
    started.current = true;
    run(async () => {
      const list = await api.checkpoints();
      setCheckpoints(list);
      const s = { ...settings, checkpoint: defaultCheckpoint(list) };
      setSettings(s);
      if (!s.checkpoint) throw new Error('no loadable checkpoint found in checkpoints*/');
      await newGame(s);
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const yourTurn = game && !game.over && game.turn === settings.human;

  function play(column) {
    if (!yourTurn || !game.legal.includes(column)) return;
    run(async () => {
      const next = [...moves, column];
      const state = await api.state(next);
      show(next, state, null, true);
      if (autoReply && !state.over) await engineMove(next, settings);
    });
  }

  function undo() {
    // Back to your own turn: two plies if the engine replied, otherwise one.
    const n = Math.min(moves.length, autoReply && game?.turn === settings.human ? 2 : 1);
    if (!n) return;
    run(async () => {
      const prev = moves.slice(0, moves.length - n);
      show(prev, await api.state(prev), null, false);
    });
  }

  function hint() {
    run(async () => {
      const res = await api.analyse(moves, settings.checkpoint, settings.simulations);
      setAnalysis(res.analysis);
    });
  }

  return (
    <main className="app">
      <section className="game">
        <h1>Connect 4</h1>
        {game && (
          <Board
            board={game.board}
            lastMove={moves.length ? moves[moves.length - 1] : null}
            animate={animate}
            canPlay={yourTurn && !busy}
            legal={game.legal}
            humanColor={settings.human}
            analysis={analysis}
            onPlay={play}
          />
        )}
        <Status game={game} human={settings.human} busy={busy} error={error} />
      </section>

      <aside className="side">
        <Controls
          checkpoints={checkpoints}
          checkpoint={settings.checkpoint}
          onCheckpoint={(checkpoint) => changeSettings({ checkpoint }, true)}
          human={settings.human}
          onHuman={(human) => changeSettings({ human }, true)}
          simulations={settings.simulations}
          onSimulations={(simulations) => changeSettings({ simulations }, false)}
          sample={settings.sample}
          onSample={(sample) => changeSettings({ sample }, false)}
          autoReply={autoReply}
          onAutoReply={setAutoReply}
          busy={busy}
          canUndo={moves.length > 0}
          canEngineMove={Boolean(game && !game.over && settings.checkpoint)}
          onNewGame={() => run(() => newGame(settings))}
          onUndo={undo}
          onEngineMove={() => run(() => engineMove(moves, settings))}
          onHint={hint}
        />
        <SearchPanel analysis={analysis} />
      </aside>
    </main>
  );
}

function Status({ game, human, busy, error }) {
  if (error) return <div className="status error">{error}</div>;
  if (!game) return <div className="status">{busy ? 'Loading the engine…' : ''}</div>;

  const chip = (side) => <span className={`chip ${side}`} />;
  if (game.over) {
    if (game.winner === 'draw') return <div className="status">Draw: the board is full.</div>;
    return (
      <div className="status">
        {chip(game.winner)} {capitalize(game.winner)} wins: {game.winner === human ? 'you win!' : 'the engine wins.'}
      </div>
    );
  }
  if (busy && game.turn !== human) return <div className="status">{chip(game.turn)} Engine is thinking…</div>;
  return (
    <div className="status">
      {chip(game.turn)} {capitalize(game.turn)} to move · {game.turn === human ? 'your turn' : "engine's turn"} · move{' '}
      {game.ply + 1}
    </div>
  );
}
