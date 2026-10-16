import { useEffect, useRef, useState } from 'react';

import { api } from './api.js';
import Board from './components/Board.jsx';
import Controls from './components/Controls.jsx';
import SearchPanel from './components/SearchPanel.jsx';
import TreeView from './components/TreeView.jsx';
import { capitalize } from './game.js';
import { TreeReplay } from './tree.js';

// The API is stateless: this component owns the list of columns played (`moves`)
// and sends it with every request. `game` is the engine's description of the
// position after those moves.

// Default opponent: the easiest loadable level.
function defaultLevel(levels) {
  return levels.find((lv) => lv.ok)?.id ?? '';
}

// Classical MCTS needs no network; neural MCTS needs a level.
const hasEngine = (s) => s.opponent === 'classical' || Boolean(s.level);

// The tree replay shows one more node per frame; bigger searches add several per
// frame so no animation takes longer than about MAX_FRAMES / 60 seconds.
const MAX_FRAMES = 240;

// The analysis as of the current replay step: N, Q and the search value come from
// the replayed tree, so the panels count up with the animation.
function liveAnalysis(analysis, tree) {
  if (!analysis?.tree) return analysis;
  const kids = new Map(tree.root.children.map((c) => [c.action, c]));
  return {
    ...analysis,
    search_value: tree.root.n ? tree.root.w / tree.root.n : null,
    moves: analysis.moves.map((m) => {
      const c = kids.get(m.move);
      return { ...m, visits: c ? c.n : 0, q: c?.n ? -c.w / c.n : null };
    }),
  };
}

export default function App() {
  const [levels, setLevels] = useState([]);
  const [settings, setSettings] = useState({
    opponent: 'neural',
    level: '',
    human: 'red',
    simulations: 100,
    sample: false,
  });
  const [autoReply, setAutoReply] = useState(true);
  const [moves, setMoves] = useState([]);
  const [game, setGame] = useState(null);
  const [analysis, setAnalysis] = useState(null);
  const [animate, setAnimate] = useState(false); // drop animation only for new moves, not undo
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [step, setStep] = useState(0); // tree events replayed so far
  const [animating, setAnimating] = useState(false);
  const busyRef = useRef(false);
  const started = useRef(false);
  const skipRef = useRef(false);

  // Replay of analysis.tree up to `step`, rebuilt when a new analysis arrives.
  const replayRef = useRef(null);
  const events = analysis?.tree ?? null;
  if (!replayRef.current || replayRef.current.events !== events) replayRef.current = new TreeReplay(events ?? []);
  replayRef.current.advanceTo(step);
  const tree = replayRef.current;
  const live = liveAnalysis(analysis, tree);

  // Serialise engine calls: a search can take a while and clicks arrive fast.
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

  // Show a new analysis and replay its tree one node at a time; resolves when done.
  function playTree(newAnalysis) {
    setAnalysis(newAnalysis);
    const total = newAnalysis?.tree?.length ?? 0;
    setStep(0);
    if (!total) return Promise.resolve();
    const perFrame = Math.max(1, Math.ceil(total / MAX_FRAMES));
    skipRef.current = false;
    setAnimating(true);
    return new Promise((resolve) => {
      let k = 0;
      const tick = () => {
        k = skipRef.current ? total : Math.min(total, k + perFrame);
        setStep(k);
        if (k < total) return requestAnimationFrame(tick);
        setAnimating(false);
        resolve();
      };
      requestAnimationFrame(tick);
    });
  }

  // The engine's disc drops once its search tree has finished drawing.
  async function engineMove(current, s) {
    const res = await api.move(current, s);
    await playTree(res.analysis);
    show([...current, res.move], res.state, res.analysis, true);
  }

  // Start over; the engine moves first when you play yellow.
  async function newGame(s) {
    const state = await api.state([]);
    show([], state, null, false);
    if (hasEngine(s) && state.turn !== s.human) await engineMove([], s);
  }

  // Changing your side starts a new game with the new settings (`restart`); switching the
  // difficulty level or between neural and classical search continues the current game.
  function changeSettings(patch, restart) {
    const next = { ...settings, ...patch };
    setSettings(next);
    if (restart) run(() => newGame(next));
  }

  useEffect(() => {
    if (started.current) return; // StrictMode runs effects twice in development
    started.current = true;
    run(async () => {
      const list = await api.levels();
      setLevels(list);
      const level = defaultLevel(list);
      // without a usable level, start against the pure-search opponent instead
      const s = { ...settings, level, opponent: level ? settings.opponent : 'classical' };
      setSettings(s);
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
      const res = await api.analyse(moves, settings);
      await playTree(res.analysis);
    });
  }

  return (
    <main className="app">
      <h1>Connect 4</h1>

      <div className="top">
        <Controls
          opponent={settings.opponent}
          onOpponent={(opponent) =>
            // switching search keeps the game going: the new opponent plays the next engine move.
            // Classical has no policy-only mode, so 0 simulations becomes 1.
            changeSettings({ opponent, simulations: opponent === 'classical' ? Math.max(1, settings.simulations) : settings.simulations }, false)
          }
          levels={levels}
          level={settings.level}
          onLevel={(level) => changeSettings({ level }, false)}
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
          canEngineMove={Boolean(game && !game.over && hasEngine(settings))}
          onNewGame={() => run(() => newGame(settings))}
          onUndo={undo}
          onEngineMove={() => run(() => engineMove(moves, settings))}
          onHint={hint}
        />

        <SearchPanel analysis={live} />
      </div>

      <div className="play">
        <section className="game">
          {game && (
            <Board
              board={game.board}
              lastMove={moves.length ? moves[moves.length - 1] : null}
              animate={animate}
              canPlay={yourTurn && !busy}
              legal={game.legal}
              humanColor={settings.human}
              analysis={live}
              onPlay={play}
            />
          )}
          <Status game={game} human={settings.human} busy={busy} error={error} />
        </section>

        <TreeView
          analysis={analysis}
          tree={tree}
          step={step}
          animating={animating}
          busy={busy}
          onSkip={() => (skipRef.current = true)}
        />
      </div>
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
