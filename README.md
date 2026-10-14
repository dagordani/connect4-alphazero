# Connect 4, AlphaZero style

A from-scratch AlphaZero for Connect 4, plus a web UI to play against any checkpoint.

| path | what it is |
|---|---|
| `interface.py` | game rules (`State`), search tree (`Node`), PUCT search (`MCTS`) |
| `model.py` | the network (`Model`, with `save` / `load`), loss, policy-only arena |
| `data.py` | self-play and the replay buffer |
| `play.py` | training loop and evaluation (runs training when executed) |
| `engine.py` | JSON-lines engine the web server talks to |
| `server/` | Node.js HTTP server (no npm dependencies) |
| `ui/` | React + Vite front end |
| `checkpoints*/` | saved models (`*.pt`); every folder matching `checkpoints*` shows up in the UI |

## Play in the browser

Requirements: Python with `torch` and `numpy`, and Node.js ≥ 20.19.

```
browser (React) --/api--> server/server.js (Node) --stdin/stdout--> engine.py (Python: MCTS + Model)
```

The server starts `engine.py` once and keeps it running. It uses the Python in `$PYTHON` if set, otherwise
`.venv/bin/python` in this folder, then `../.venv/bin/python`, then `python3`.

**Development** (hot reload), in two terminals:

```bash
cd server && npm run dev          # API on http://127.0.0.1:3001
cd ui && npm install && npm run dev   # open the URL Vite prints (http://localhost:5173)
```

**Production** (one server):

```bash
cd ui && npm install && npm run build
cd ../server && npm start         # http://127.0.0.1:3001
```

Environment variables: `PORT` / `HOST` for the server (default `127.0.0.1:3001`), `PYTHON` for the
interpreter, `API_TARGET` for the Vite dev proxy if the server isn't on port 3001.

In the UI: choose a checkpoint and a side, set the simulations per move (0 = no search, the policy head picks),
and play by clicking a column. **Hint** shows the search's analysis without playing. Checkpoints saved with an
older architecture are listed but disabled.

## API

The server is stateless: the client sends the full list of columns played so far.

| method | path | body | returns |
|---|---|---|---|
| GET | `/api/checkpoints` | | `{checkpoints: [{id, dir, name, iter, ok, config, error}]}` |
| POST | `/api/state` | `{moves}` | `{board, turn, ply, legal, over, winner}` |
| POST | `/api/move` | `{moves, checkpoint, simulations}` | `{move, analysis, state}` |
| POST | `/api/analyse` | `{moves, checkpoint, simulations}` | `{analysis, state}` |

`board[row][col]`: row 0 is the top, `1` = red (moves first), `-1` = yellow, `0` = empty.
