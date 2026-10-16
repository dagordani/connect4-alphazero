// Web Worker running the engine (search.js) so searches never block the page.
// Each level's network is an ONNX file (exported by export_onnx.py into
// public/models/) run with onnxruntime-web's WASM backend, loaded on first use.
//
// Protocol: {id, cmd, ...params} in, {id, ok: true, result} or {id, ok: false, error} out.
// Requests are handled one at a time, in order.

import * as ort from 'onnxruntime-web/wasm';

import { createEngine } from './search.js';

const BASE = import.meta.env.BASE_URL;

// One thread: multi-threaded WASM needs cross-origin isolation headers, and this network
// is tiny anyway. (Vite bundles the .wasm file itself; see vite.config.js.)
ort.env.wasm.numThreads = 1;

const sessions = new Map(); // level id -> Promise<InferenceSession>

async function loadConfig() {
  const res = await fetch(`${BASE}app-config.json`);
  if (!res.ok) throw new Error(`could not load app-config.json (${res.status}); run export_onnx.py`);
  return res.json();
}

function makeEngine(config) {
  function getSession(levelId) {
    if (!sessions.has(levelId)) {
      const level = config.levels.find((lv) => lv.id === levelId);
      const session = ort.InferenceSession.create(`${BASE}${level.model}`, { executionProviders: ['wasm'] });
      session.catch(() => sessions.delete(levelId)); // retry on the next request
      sessions.set(levelId, session);
    }
    return sessions.get(levelId);
  }

  async function getEvaluate(levelId) {
    const session = await getSession(levelId);
    return async (state) => {
      const board = new ort.Tensor('float32', Float32Array.from(state.board), [1, 1, state.rows, state.cols]);
      const out = await session.run({ board });
      return { value: out.value.data[0], logits: out.logits.data };
    };
  }

  return createEngine(config, getEvaluate);
}

const enginePromise = loadConfig().then(makeEngine);

let queue = Promise.resolve();
self.onmessage = ({ data }) => {
  const { id, cmd, ...params } = data;
  queue = queue.then(async () => {
    try {
      const engine = await enginePromise;
      const handler = engine[cmd];
      if (!handler) throw new Error(`unknown command ${JSON.stringify(cmd)}`);
      self.postMessage({ id, ok: true, result: await handler(params) });
    } catch (e) {
      self.postMessage({ id, ok: false, error: e.message ?? String(e) });
    }
  });
};
