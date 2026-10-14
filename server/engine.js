// Manages the Python engine process (../engine.py) and its JSON-lines protocol.
//
// One long-lived process: loading PyTorch and the checkpoints takes seconds, so
// we pay that once. Requests are written to its stdin one JSON object per line;
// replies come back on stdout tagged with the same id. The engine answers in
// order, one at a time, so concurrent requests simply queue in the pipe.

import { spawn } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import readline from 'node:readline';

const DEFAULT_TIMEOUT_MS = 120_000;

// PYTHON env var first, then a virtualenv in this project or (while it still
// lives inside the alpha-zero repo) the parent's, then whatever python3 is on PATH.
function pickPython(projectDir) {
  if (process.env.PYTHON) return process.env.PYTHON;
  const candidates = [
    path.join(projectDir, '.venv', 'bin', 'python'),
    path.join(projectDir, '..', '.venv', 'bin', 'python'),
  ];
  return candidates.find((p) => fs.existsSync(p)) ?? 'python3';
}

export class EngineProcess {
  constructor(projectDir) {
    this.projectDir = projectDir;
    this.python = pickPython(projectDir);
    this.proc = null;
    this.pending = new Map(); // id -> { resolve, reject, timer }
    this.nextId = 1;
  }

  start() {
    console.log(`starting engine: ${this.python} engine.py`);
    const proc = spawn(this.python, ['-u', 'engine.py'], {
      cwd: this.projectDir,
      stdio: ['pipe', 'pipe', 'inherit'], // engine warnings/tracebacks show up in this terminal
    });
    this.proc = proc;

    readline.createInterface({ input: proc.stdout }).on('line', (line) => this.#onLine(line));

    // Only the *current* process's failures reject pending requests: an old
    // process exiting after a respawn must not fail the new one's requests.
    proc.on('error', (err) => {
      console.error(`engine failed to start (${this.python}): ${err.message}`);
      if (this.proc !== proc) return;
      this.proc = null;
      this.#failAll(new Error(`engine failed to start: ${err.message}`));
    });
    proc.on('exit', (code, signal) => {
      if (this.proc !== proc) return;
      this.proc = null; // the next request respawns it
      this.#failAll(new Error(`engine exited (code ${code}, signal ${signal})`));
    });
    proc.stdin.on('error', () => {}); // EPIPE after a crash; the 'exit'/'error' handlers report it
  }

  request(cmd, params = {}, timeoutMs = DEFAULT_TIMEOUT_MS) {
    if (!this.proc) this.start();
    const id = this.nextId++;
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pending.delete(id);
        reject(new Error(`engine timed out after ${timeoutMs / 1000}s on '${cmd}'`));
      }, timeoutMs);
      this.pending.set(id, { resolve, reject, timer });
      this.proc.stdin.write(JSON.stringify({ id, cmd, ...params }) + '\n');
    });
  }

  stop() {
    if (this.proc) this.proc.kill();
    this.proc = null;
  }

  #onLine(line) {
    let reply;
    try {
      reply = JSON.parse(line);
    } catch {
      console.error(`engine wrote a non-JSON line: ${line}`);
      return;
    }
    const entry = this.pending.get(reply.id);
    if (!entry) return; // timed out already
    this.pending.delete(reply.id);
    clearTimeout(entry.timer);
    if (reply.ok) entry.resolve(reply.result);
    else entry.reject(new EngineError(reply.error));
  }

  #failAll(err) {
    for (const { reject, timer } of this.pending.values()) {
      clearTimeout(timer);
      reject(err);
    }
    this.pending.clear();
  }
}

// An error the engine reported about the request itself (illegal move, unknown
// checkpoint, ...), as opposed to the engine crashing.
export class EngineError extends Error {}
