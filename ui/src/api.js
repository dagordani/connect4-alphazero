// The UI's interface to the engine, which runs in the browser in a Web Worker
// (engine/worker.js). Every call returns a promise and rejects with the engine's
// error message, so the UI can show it as-is.

const worker = new Worker(new URL('./engine/worker.js', import.meta.url), { type: 'module' });
const pending = new Map(); // id -> {resolve, reject}
let nextId = 1;

worker.onmessage = ({ data }) => {
  const entry = pending.get(data.id);
  if (!entry) return;
  pending.delete(data.id);
  if (data.ok) entry.resolve(data.result);
  else entry.reject(new Error(data.error));
};

worker.onerror = (e) => {
  for (const { reject } of pending.values()) reject(new Error(`engine crashed: ${e.message}`));
  pending.clear();
};

function request(cmd, params = {}) {
  const id = nextId++;
  return new Promise((resolve, reject) => {
    pending.set(id, { resolve, reject });
    worker.postMessage({ id, cmd, ...params });
  });
}

export const api = {
  levels: () => request('levels').then((d) => d.levels),
  state: (moves) => request('state', { moves }),
  // `s` is the opponent settings: { opponent: 'neural' | 'classical', level, simulations, sample }
  move: (moves, s) =>
    request('move', { moves, opponent: s.opponent, level: s.level, simulations: s.simulations, sample: s.sample }),
  analyse: (moves, s) => request('analyse', { moves, opponent: s.opponent, level: s.level, simulations: s.simulations }),
};
