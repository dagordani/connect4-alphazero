// The engine the UI talks to: port of the request handling in engine.py (replay,
// describe, search, classical_search) with the same result shapes, so the UI is the
// same whichever engine answers. Runs in the browser (worker.js) and in Node
// (scripts/parity.mjs); the network comes in through `getEvaluate(levelId)`.
//
// Commands:
//   levels                                              difficulty levels and board size
//   state    {moves}                                    describe the position
//   move     {moves, opponent, level, simulations, sample}  search, play the chosen move
//   analyse  {moves, opponent, level, simulations}      search without playing
//
// opponent "neural" (default) = NeuralMCTS with the level's network; "classical" =
// ClassicalMCTS (no network). simulations 0 = no search: the policy head picks (neural only).

import { ClassicalMCTS } from './classical.js';
import { argmax, Node, sampleIndex, State } from './game.js';
import { maskedSoftmax, NeuralMCTS } from './neural.js';

export const MAX_SIMULATIONS = 10000;

const round4 = (x) => Math.round(x * 1e4) / 1e4;

// --- game state ----------------------------------------------------------------
export function replay(moves, rows, cols) {
  if (!Array.isArray(moves) || moves.length > rows * cols) {
    throw new Error(`moves must be a list of at most ${rows * cols} columns`);
  }
  let state = State.fromEmptyBoard(rows, cols);
  moves.forEach((m, i) => {
    if (!Number.isInteger(m)) throw new Error(`move ${i} is not an integer: ${m}`);
    if (state.gameEnded) throw new Error(`move ${i} (${m}) played after the game ended`);
    if (!state.allowedMoves.includes(m)) {
      throw new Error(`move ${i} (${m}) is illegal; legal: [${state.allowedMoves.join(', ')}]`);
    }
    state = state.transitionState(m);
  });
  return state;
}

// Position in absolute colours: +1 red (moves first), -1 yellow, 0 empty. State.board is in
// the side-to-move frame and flips after every move, so on even plies red is to move.
export function describe(state, ply) {
  const sign = ply % 2 === 0 ? 1 : -1;
  const turn = sign === 1 ? 'red' : 'yellow';
  const board = [];
  for (let r = 0; r < state.rows; r++) {
    board.push(Array.from(state.board.subarray(r * state.cols, (r + 1) * state.cols), (v) => v * sign + 0));
  }
  let winner = null;
  if (state.gameEnded) {
    // reward -1: the player to move lost, i.e. the previous mover won
    winner = state.reward === 0 ? 'draw' : turn === 'red' ? 'yellow' : 'red';
  }
  return {
    board,
    turn,
    ply,
    legal: state.gameEnded ? [] : [...state.allowedMoves],
    over: state.gameEnded,
    winner,
  };
}

// --- search --------------------------------------------------------------------
function checkSearchParams(simulations, sample, minimum = 0) {
  if (!Number.isInteger(simulations) || simulations < minimum || simulations > MAX_SIMULATIONS) {
    throw new Error(`simulations must be an integer in ${minimum}..${MAX_SIMULATIONS}`);
  }
  if (typeof sample !== 'boolean') throw new Error('sample must be true or false');
}

// root children by action -> rows of the analysis table
function moveRows(state, root, priors) {
  const children = new Map(root.children.map(([a, c]) => [a, c]));
  return state.allowedMoves.map((a) => {
    const c = children.get(a);
    return {
      move: a,
      prior: priors ? round4(priors[a]) : null,
      visits: c ? c.n : 0,
      // a child's w is from the opponent's point of view, so negate for the player to move
      q: c && c.n ? round4(-c.w / c.n) : null,
    };
  });
}

export async function search(state, evaluate, simulations, sample = false, random = Math.random) {
  checkSearchParams(simulations, sample);
  const net = await evaluate(state);
  const policy = maskedSoftmax(net.logits, state.allowedMoves);
  const legal = state.allowedMoves;

  if (simulations === 0) {
    // no search: the policy head picks -- its favourite legal move, or a draw from it
    const move = sample ? legal[sampleIndex(legal.map((a) => policy[a]), random)] : legal[argmax(legal.map((a) => policy[a]))];
    const rows = legal.map((a) => ({ move: a, prior: round4(policy[a]), visits: null, q: null }));
    rows.sort((x, y) => y.prior - x.prior);
    return [
      move,
      { opponent: 'neural', simulations: 0, sampled: sample, search_value: null, net_value: round4(net.value), moves: rows },
    ];
  }

  const root = new Node(0, state);
  const tree = new NeuralMCTS(root, evaluate, Math.max(2, simulations));
  await tree.buildTree();
  const move = tree.chooseMove(sample, random);
  const rows = moveRows(state, root, root.prior);
  rows.sort((x, y) => y.visits - x.visits || y.prior - x.prior);
  return [
    move,
    {
      opponent: 'neural',
      simulations,
      sampled: sample,
      search_value: root.n ? round4(root.w / root.n) : null,
      net_value: round4(net.value),
      moves: rows,
      tree: tree.events,
    },
  ];
}

// ClassicalMCTS (UCT + random rollouts, no network): same analysis shape, no priors or network value.
export function classicalSearch(state, simulations, maxSimPlies, sample = false, random = Math.random) {
  checkSearchParams(simulations, sample, 1);
  const root = new Node(0, state);
  const tree = new ClassicalMCTS(root, simulations, maxSimPlies, random);
  tree.buildTree();

  let move;
  if (sample) {
    // ClassicalMCTS.chooseMove has no sampling, so draw in proportion to the root's visit counts here
    const actions = root.children.map(([a]) => a);
    move = actions[sampleIndex(root.children.map(([, c]) => c.n), random)];
  } else {
    move = tree.chooseMove();
  }

  const rows = moveRows(state, root, null);
  rows.sort((x, y) => y.visits - x.visits);
  return [
    move,
    {
      opponent: 'classical',
      simulations,
      sampled: sample,
      search_value: root.n ? round4(root.w / root.n) : null,
      net_value: null,
      moves: rows,
      tree: tree.events,
    },
  ];
}

// --- commands ------------------------------------------------------------------
// config: {rows, cols, maxSimPlies, levels}; getEvaluate(levelId) -> async evaluate(state)
export function createEngine(config, getEvaluate, random = Math.random) {
  const { rows, cols, maxSimPlies } = config;

  async function runSearch(req, state, sample = false) {
    const opponent = req.opponent ?? 'neural';
    const simulations = req.simulations ?? 100;
    if (opponent === 'neural') {
      if (!config.levels.some((lv) => lv.id === req.level)) throw new Error(`unknown level ${JSON.stringify(req.level)}`);
      return search(state, await getEvaluate(req.level), simulations, sample, random);
    }
    if (opponent === 'classical') return classicalSearch(state, simulations, maxSimPlies, sample, random);
    throw new Error(`opponent must be 'neural' or 'classical', got ${JSON.stringify(opponent)}`);
  }

  return {
    levels: () => ({ levels: config.levels.map(({ id, name, games }) => ({ id, name, games, ok: true })), rows, cols }),

    state: (req) => {
      const moves = req.moves ?? [];
      return describe(replay(moves, rows, cols), moves.length);
    },

    analyse: async (req) => {
      const moves = req.moves ?? [];
      const state = replay(moves, rows, cols);
      if (state.gameEnded) return { analysis: null, state: describe(state, moves.length) };
      const [, analysis] = await runSearch(req, state);
      return { analysis, state: describe(state, moves.length) };
    },

    move: async (req) => {
      const moves = req.moves ?? [];
      const state = replay(moves, rows, cols);
      if (state.gameEnded) throw new Error('the game is already over');
      const [move, analysis] = await runSearch(req, state, req.sample ?? false);
      const after = state.transitionState(move);
      return { move, analysis, state: describe(after, moves.length + 1) };
    },
  };
}
