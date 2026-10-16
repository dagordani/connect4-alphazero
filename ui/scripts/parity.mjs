// Parity test: the JS engine (src/engine/, what the website runs) against the Python
// reference (engine.py: game_interface.py + neural_mcts.py + classical_mcts.py + torch).
//
//   cd ui && npm run parity        (needs `uv` and the exported models in public/models)
//
// 1. Rules: random games, every position described identically (board, turn, legal, winner).
// 2. Neural search: same positions x every level -> same move, visit counts and tree.
// 3. Classical search (random, so not comparable move by move): the tree events replay to
//    the reported stats, and it finds an immediate win / blocks an immediate loss.

import { spawn } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import readline from 'node:readline';
import { fileURLToPath } from 'node:url';

import * as ort from 'onnxruntime-web/wasm';

import { createEngine } from '../src/engine/search.js';
import { TreeReplay } from '../src/tree.js';

const UI = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const ROOT = path.resolve(UI, '..');
const config = JSON.parse(fs.readFileSync(path.join(UI, 'public', 'app-config.json'), 'utf8'));

const NUM_GAMES = 2000;
const NUM_POSITIONS = 30;
const SIMULATIONS = 200;

// deterministic random numbers for the test (mulberry32)
function rng(seed) {
  return () => {
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const random = rng(42);

// --- engines ------------------------------------------------------------------
ort.env.wasm.numThreads = 1;
const sessions = new Map();
async function getEvaluate(levelId) {
  if (!sessions.has(levelId)) {
    const level = config.levels.find((lv) => lv.id === levelId);
    const bytes = fs.readFileSync(path.join(UI, 'public', level.model));
    sessions.set(levelId, await ort.InferenceSession.create(bytes));
  }
  const session = sessions.get(levelId);
  return async (state) => {
    const board = new ort.Tensor('float32', Float32Array.from(state.board), [1, 1, state.rows, state.cols]);
    const out = await session.run({ board });
    return { value: out.value.data[0], logits: out.logits.data };
  };
}
const js = createEngine(config, getEvaluate, random);

// engine.py over stdin/stdout, one JSON request per line
const py = spawn('uv', ['run', '--quiet', 'python', 'engine.py'], { cwd: ROOT, stdio: ['pipe', 'pipe', 'inherit'] });
const waiting = new Map();
let nextId = 1;
readline.createInterface({ input: py.stdout }).on('line', (line) => {
  const reply = JSON.parse(line);
  const { resolve, reject } = waiting.get(reply.id);
  waiting.delete(reply.id);
  reply.ok ? resolve(reply.result) : reject(new Error(reply.error));
});
function python(cmd, params) {
  const id = nextId++;
  return new Promise((resolve, reject) => {
    waiting.set(id, { resolve, reject });
    py.stdin.write(JSON.stringify({ id, cmd, ...params }) + '\n');
  });
}

// --- helpers ------------------------------------------------------------------
let failures = 0;
function check(ok, message) {
  if (!ok) {
    failures++;
    if (failures <= 20) console.log(`  FAIL ${message}`);
  }
  return ok;
}

function randomGame() {
  const moves = [];
  for (let state = js.state({ moves }); !state.over; state = js.state({ moves })) {
    moves.push(state.legal[Math.floor(random() * state.legal.length)]);
  }
  return moves;
}

const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
const byMove = (rows) => Object.fromEntries(rows.map((r) => [r.move, r]));

// --- 1. rules -------------------------------------------------------------------
console.log(`1. rules: ${NUM_GAMES} random games, every position`);
const games = Array.from({ length: NUM_GAMES }, randomGame);
let positions = 0;
const outcomes = { red: 0, yellow: 0, draw: 0 };
await Promise.all(
  games.map(async (moves, g) => {
    for (let k = 0; k <= moves.length; k++) {
      const prefix = moves.slice(0, k);
      const [a, b] = [js.state({ moves: prefix }), await python('state', { moves: prefix })];
      positions++;
      if (!check(same(a, b), `game ${g} after ${k} moves ${JSON.stringify(prefix)}`)) break;
    }
    outcomes[js.state({ moves }).winner]++;
  }),
);
console.log(`   ${positions} positions compared; results red ${outcomes.red} / yellow ${outcomes.yellow} / draw ${outcomes.draw}`);

// --- 2. neural search -------------------------------------------------------------
console.log(`2. neural search: ${NUM_POSITIONS} positions x ${config.levels.length} levels, ${SIMULATIONS} simulations`);
const searchPositions = [[]];
for (const g of games) {
  if (searchPositions.length === NUM_POSITIONS) break;
  if (g.length > 4) searchPositions.push(g.slice(0, Math.floor(random() * (g.length - 1))));
}
let searches = 0;
let exact = 0;
let worstValue = 0;
for (const level of config.levels) {
  for (const moves of searchPositions) {
    const req = { moves, opponent: 'neural', level: level.id, simulations: SIMULATIONS };
    const [a, b] = [await js.move(req), await python('move', req)];
    searches++;
    const ma = byMove(a.analysis.moves);
    const mb = byMove(b.analysis.moves);
    const sameVisits = Object.keys(mb).every((m) => ma[m]?.visits === mb[m].visits);
    const sameTree =
      a.analysis.tree.length === b.analysis.tree.length &&
      a.analysis.tree.every((e, i) => e[0] === b.analysis.tree[i][0] && e[1] === b.analysis.tree[i][1] && e[2] === b.analysis.tree[i][2]);
    for (const m of Object.keys(mb)) worstValue = Math.max(worstValue, Math.abs(ma[m].prior - mb[m].prior));
    worstValue = Math.max(worstValue, Math.abs(a.analysis.net_value - b.analysis.net_value));
    if (a.move === b.move && sameVisits && sameTree) exact++;
    check(a.move === b.move, `${level.id} ${JSON.stringify(moves)}: move js ${a.move} vs python ${b.move}`);
    check(sameVisits, `${level.id} ${JSON.stringify(moves)}: visit counts differ`);
  }
}
console.log(`   ${exact}/${searches} searches identical (move, visits, tree); max prior/value difference at the 4 reported decimals: ${worstValue.toExponential(1)}`);

// policy only (0 simulations)
for (const level of config.levels) {
  const req = { moves: [3, 3, 2], opponent: 'neural', level: level.id, simulations: 0 };
  const [a, b] = [await js.move(req), await python('move', req)];
  check(a.move === b.move, `${level.id} policy-only move js ${a.move} vs python ${b.move}`);
}

// --- 3. classical search ----------------------------------------------------------
console.log('3. classical search: tree replay and tactics');
for (const moves of searchPositions.slice(0, 10)) {
  const { analysis } = await js.move({ moves, opponent: 'classical', simulations: 2000 });
  const t = new TreeReplay(analysis.tree);
  t.advanceTo(analysis.tree.length);
  const kids = new Map(t.root.children.map((c) => [c.action, c]));
  check(
    analysis.moves.every((m) => (kids.get(m.move)?.n ?? 0) === m.visits),
    `classical ${JSON.stringify(moves)}: replayed visits differ from reported`,
  );
}
const tactics = [
  { moves: [0, 6, 0, 6, 0, 6], best: 0, what: 'red wins in column 0' },
  { moves: [0, 6, 0, 6, 0], best: 0, what: 'yellow blocks column 0' },
  { moves: [1, 1, 2, 2, 3, 3], best: [0, 4], what: 'red completes the bottom row' },
];
for (const { moves, best, what } of tactics) {
  const { move } = await js.move({ moves, opponent: 'classical', simulations: 2000 });
  check([best].flat().includes(move), `classical should play ${best} (${what}), played ${move}`);
}

py.stdin.end();
console.log(failures ? `\n${failures} check(s) failed` : '\nall checks passed');
process.exit(failures ? 1 : 0);
