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
  checkpoints: () => request('/api/checkpoints').then((d) => d.checkpoints),
  state: (moves) => request('/api/state', { moves }),
  move: (moves, checkpoint, simulations, sample) => request('/api/move', { moves, checkpoint, simulations, sample }),
  analyse: (moves, checkpoint, simulations) => request('/api/analyse', { moves, checkpoint, simulations }),
};
