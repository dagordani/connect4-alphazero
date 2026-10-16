// Thin wrapper over the server's JSON API. Every call throws an Error carrying
// the server's message on failure, so the UI can show it as-is.

async function request(path, body) {
  const res = await fetch(path, {
    method: body === undefined ? 'GET' : 'POST',
    headers: body === undefined ? undefined : { 'Content-Type': 'application/json' },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  const data = await res.json().catch(() => ({ error: `${res.status} ${res.statusText}` }));
  if (!res.ok) throw new Error(data.error ?? `${res.status} ${res.statusText}`);
  return data;
}

export const api = {
  levels: () => request('/api/levels').then((d) => d.levels),
  state: (moves) => request('/api/state', { moves }),
  // `s` is the opponent settings: { opponent: 'neural' | 'classical', level, simulations, sample }
  move: (moves, s) =>
    request('/api/move', { moves, opponent: s.opponent, level: s.level, simulations: s.simulations, sample: s.sample }),
  analyse: (moves, s) =>
    request('/api/analyse', { moves, opponent: s.opponent, level: s.level, simulations: s.simulations }),
};
