// HTTP server for the Connect 4 UI. Node 20, no dependencies.
//
//   GET  /api/levels                                       difficulty levels, whether they load, board size
//   POST /api/state    {moves}                             describe a position
//   POST /api/move     {moves, opponent, level, simulations, sample}  engine picks and plays a move
//   POST /api/analyse  {moves, opponent, level, simulations}          search without playing (hint)
//
// `opponent` is "neural" (default; NeuralMCTS with the network of `level`) or "classical"
// (ClassicalMCTS: pure search with random rollouts, no network, no level).
// The levels are defined in config.toml ([[levels]]).
// `sample: true` makes the engine draw its move in proportion to the visit
// counts (or the policy, with simulations 0) instead of playing the best one.
// The board size comes from config.toml; the engine checks moves against it.
//
// The API is stateless: the browser keeps the list of columns played and sends
// it with every request. Everything else is served from ../ui/dist (the built
// React app). During development, run the Vite dev server instead; it proxies
// /api here.

import fs from 'node:fs';
import http from 'node:http';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { EngineError, EngineProcess } from './engine.js';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const PROJECT_DIR = path.resolve(HERE, '..');
const UI_DIST = path.join(PROJECT_DIR, 'ui', 'dist');
const HOST = process.env.HOST ?? '127.0.0.1';
const PORT = Number(process.env.PORT ?? 3001);
const MAX_BODY_BYTES = 64 * 1024;
const MAX_SIMULATIONS = 10000;
const MAX_MOVES = 400; // sanity cap only: the engine checks moves against the configured board
const OPPONENTS = ['neural', 'classical'];

const engine = new EngineProcess(PROJECT_DIR);

class HttpError extends Error {
  constructor(status, message) {
    super(message);
    this.status = status;
  }
}

// --- request helpers -----------------------------------------------------------
async function readJson(req) {
  let size = 0;
  const chunks = [];
  for await (const chunk of req) {
    size += chunk.length;
    if (size > MAX_BODY_BYTES) throw new HttpError(413, 'request body too large');
    chunks.push(chunk);
  }
  if (size === 0) return {};
  try {
    return JSON.parse(Buffer.concat(chunks).toString('utf8'));
  } catch {
    throw new HttpError(400, 'body is not valid JSON');
  }
}

function validMoves(moves) {
  if (moves === undefined) return [];
  const ok =
    Array.isArray(moves) &&
    moves.length <= MAX_MOVES &&
    moves.every((m) => Number.isInteger(m) && m >= 0);
  if (!ok) throw new HttpError(400, '`moves` must be an array of column numbers');
  return moves;
}

function searchParams(body) {
  const moves = validMoves(body.moves);
  const opponent = body.opponent ?? 'neural';
  if (!OPPONENTS.includes(opponent)) throw new HttpError(400, '`opponent` must be "neural" or "classical"');
  const neural = opponent === 'neural';
  if (neural && (typeof body.level !== 'string' || body.level.length === 0 || body.level.length > 50)) {
    throw new HttpError(400, '`level` must be a level id from /api/levels');
  }
  const simulations = body.simulations ?? 100;
  const minSimulations = neural ? 0 : 1; // classical has no policy to fall back on
  if (!Number.isInteger(simulations) || simulations < minSimulations || simulations > MAX_SIMULATIONS) {
    throw new HttpError(400, `\`simulations\` must be an integer in ${minSimulations}..${MAX_SIMULATIONS}`);
  }
  return neural ? { moves, opponent, level: body.level, simulations } : { moves, opponent, simulations };
}

function sendJson(res, status, payload) {
  const body = JSON.stringify(payload);
  res.writeHead(status, { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(body) });
  res.end(body);
}

// --- API -----------------------------------------------------------------------
const routes = {
  'GET /api/levels': () => engine.request('levels'),
  'POST /api/state': async (req) => engine.request('state', { moves: validMoves((await readJson(req)).moves) }),
  'POST /api/move': async (req) => {
    const body = await readJson(req);
    const sample = body.sample ?? false;
    if (typeof sample !== 'boolean') throw new HttpError(400, '`sample` must be true or false');
    return engine.request('move', { ...searchParams(body), sample });
  },
  'POST /api/analyse': async (req) => engine.request('analyse', searchParams(await readJson(req))),
};

async function handleApi(req, res, pathname) {
  const route = routes[`${req.method} ${pathname}`];
  if (!route) return sendJson(res, 404, { error: `no route ${req.method} ${pathname}` });
  try {
    sendJson(res, 200, await route(req));
  } catch (err) {
    if (err instanceof HttpError) return sendJson(res, err.status, { error: err.message });
    if (err instanceof EngineError) return sendJson(res, 400, { error: err.message });
    console.error(err);
    sendJson(res, 500, { error: err.message });
  }
}

// --- static files (the built UI) -----------------------------------------------
const CONTENT_TYPES = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.svg': 'image/svg+xml',
  '.png': 'image/png',
  '.ico': 'image/x-icon',
  '.json': 'application/json',
  '.woff2': 'font/woff2',
};

function serveStatic(res, pathname) {
  if (!fs.existsSync(UI_DIST)) {
    res.writeHead(404, { 'Content-Type': 'text/plain; charset=utf-8' });
    return res.end(
      'The UI has not been built.\n' +
        'Production: cd ui && npm install && npm run build, then restart this server.\n' +
        'Development: cd ui && npm run dev, and open the URL Vite prints.\n',
    );
  }
  let decoded;
  try {
    decoded = decodeURIComponent(pathname);
  } catch {
    res.writeHead(400);
    return res.end(); // malformed %-encoding
  }
  // Resolve inside ui/dist only; anything that escapes it (../) is refused.
  let file = path.resolve(UI_DIST, '.' + decoded);
  if (file !== UI_DIST && !file.startsWith(UI_DIST + path.sep)) {
    res.writeHead(403);
    return res.end();
  }
  if (!fs.existsSync(file) || fs.statSync(file).isDirectory()) file = path.join(UI_DIST, 'index.html');
  res.writeHead(200, { 'Content-Type': CONTENT_TYPES[path.extname(file)] ?? 'application/octet-stream' });
  fs.createReadStream(file)
    .on('error', () => res.destroy()) // e.g. the file vanished during a rebuild
    .pipe(res);
}

// --- server --------------------------------------------------------------------
const server = http.createServer((req, res) => {
  try {
    const { pathname } = new URL(req.url, 'http://localhost');
    if (pathname.startsWith('/api/')) return handleApi(req, res, pathname);
    if (req.method !== 'GET' && req.method !== 'HEAD') {
      res.writeHead(405);
      return res.end();
    }
    serveStatic(res, pathname);
  } catch (err) {
    // A bad request must never take the server down.
    console.error(err);
    if (!res.headersSent) res.writeHead(500);
    res.end();
  }
});

server.listen(PORT, HOST, () => {
  console.log(`\n  Connect 4 server on http://${HOST}:${PORT}   (ctrl-c to stop)\n`);
  engine.start(); // warm up: importing torch takes a few seconds
});

for (const signal of ['SIGINT', 'SIGTERM']) {
  process.on(signal, () => {
    engine.stop();
    server.close();
    process.exit(0);
  });
}
