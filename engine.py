"""Engine process for the web UI: answers JSON requests with the AlphaZero search.

The Node server (server/) starts this once and talks to it over stdin/stdout,
one JSON object per line:

    request:  {"id": 7, "cmd": "move", "moves": [3, 3, 2], "checkpoint": "checkpoints_64_4/i3_....pt", "simulations": 100}
    reply:    {"id": 7, "ok": true, "result": {...}}   or   {"id": 7, "ok": false, "error": "..."}

It is stateless: every request carries the full list of columns played so far,
and the game is replayed from the empty board with State.transition_state, which
also checks that every move is legal. Undo is just a shorter list.

Commands:
    checkpoints                              list checkpoint files and whether they load
    state     {moves}                        describe the position
    move      {moves, checkpoint, simulations, sample}  search, play the chosen move, describe the result
    analyse   {moves, checkpoint, simulations}          search without playing

simulations == 0 means "no search": the move comes straight from the policy head.
sample == false (default) plays the best move: most visits, or highest prior without search.
sample == true draws the move in proportion to the visit counts (temperature 1),
or to the policy without search. Either way there is no Dirichlet noise.
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from interface import COLS, ROWS, MCTS, Node, State
from model import Model

HERE = Path(__file__).resolve().parent
MAX_SIMULATIONS = 10000

_checkpoint_info: dict[str, tuple[float, dict]] = {}  # id -> (mtime, listing entry)
_models: dict[str, tuple[float, Model]] = {}  # id -> (mtime, loaded model)


# --- checkpoints -------------------------------------------------------------
def _iteration(name: str):
    m = re.match(r"i(\d+)_", name)
    return int(m.group(1)) if m else None


def list_checkpoints() -> list[dict]:
    """Every checkpoints*/*.pt file, marked ok if it loads into the current Model."""
    out = []
    for path in HERE.glob("checkpoints*/*.pt"):
        cid = path.relative_to(HERE).as_posix()
        mtime = path.stat().st_mtime
        cached = _checkpoint_info.get(cid)
        if cached and cached[0] == mtime:
            out.append(cached[1])
            continue
        entry = {"id": cid, "dir": path.parent.name, "name": path.stem, "iter": _iteration(path.stem)}
        try:
            model = Model.load(path).eval()
            _models[cid] = (mtime, model)
            entry.update(ok=True, config=torch.load(path, weights_only=True)["config"])
        except Exception as e:  # noqa: BLE001 -- an old architecture, a corrupt file, ...
            entry.update(ok=False, error=f"{type(e).__name__}: {str(e).splitlines()[0][:200]}")
        _checkpoint_info[cid] = (mtime, entry)
        out.append(entry)
    # random_weights first, then by iteration
    out.sort(key=lambda e: (e["dir"], e["iter"] is not None, e["iter"] or 0, e["name"]))
    return out


def get_model(cid: str) -> Model:
    """Load a checkpoint by id. Only ids from the scanned listing are accepted."""
    entries = {e["id"]: e for e in list_checkpoints()}
    entry = entries.get(cid)
    if entry is None:
        raise ValueError(f"unknown checkpoint {cid!r}")
    if not entry["ok"]:
        raise ValueError(f"checkpoint {cid!r} does not load: {entry['error']}")
    mtime = (HERE / cid).stat().st_mtime
    cached = _models.get(cid)
    if cached is None or cached[0] != mtime:
        _models[cid] = (mtime, Model.load(HERE / cid).eval())
    return _models[cid][1]


# --- game state ----------------------------------------------------------------
def replay(moves) -> State:
    if not isinstance(moves, list) or len(moves) > ROWS * COLS:
        raise ValueError("moves must be a list of at most 42 columns")
    state = State.from_empty_board(ROWS, COLS)
    for i, m in enumerate(moves):
        if not isinstance(m, int) or isinstance(m, bool):
            raise ValueError(f"move {i} is not an integer: {m!r}")
        if state.game_ended:
            raise ValueError(f"move {i} ({m}) played after the game ended")
        if m not in state.allowed_moves:
            raise ValueError(f"move {i} ({m}) is illegal; legal: {state.allowed_moves.tolist()}")
        state = state.transition_state(m)
    return state


def describe(state: State, ply: int) -> dict:
    """Position in absolute colours: +1 red (moves first), -1 yellow, 0 empty.

    State.board is in the side-to-move frame (+1 = player to move) and flips
    after every move, so on even plies the player to move is red.
    """
    sign = 1 if ply % 2 == 0 else -1
    turn = "red" if sign == 1 else "yellow"
    board = [[int(v) for v in row] for row in (state.board * sign)]
    winner = None
    if state.game_ended:
        # reward -1: the player to move lost, i.e. the previous mover won
        winner = "draw" if state.reward == 0 else ("yellow" if turn == "red" else "red")
    return {
        "board": board,
        "turn": turn,
        "ply": ply,
        "legal": [] if state.game_ended else [int(a) for a in state.allowed_moves],
        "over": bool(state.game_ended),
        "winner": winner,
    }


# --- search --------------------------------------------------------------------
def _net(model: Model, state: State) -> tuple[float, np.ndarray]:
    """Raw network output: value for the player to move, masked policy over columns."""
    x = torch.tensor(state.board, dtype=torch.float).reshape(1, 1, ROWS, COLS)
    with torch.no_grad():
        value, logits = model(x)
    mask = torch.full((COLS,), -np.inf)
    mask[list(state.allowed_moves)] = 0
    probs = F.softmax(logits[0] + mask, dim=0).numpy()
    return float(value.item()), probs


def search(state: State, model: Model, simulations: int, sample: bool = False) -> tuple[int, dict]:
    if not isinstance(simulations, int) or not 0 <= simulations <= MAX_SIMULATIONS:
        raise ValueError(f"simulations must be an integer in 0..{MAX_SIMULATIONS}")
    if not isinstance(sample, bool):
        raise ValueError("sample must be true or false")
    net_value, policy = _net(model, state)
    legal = [int(a) for a in state.allowed_moves]

    if simulations == 0:
        # no search: the policy head picks -- its favourite legal move, or a draw from it
        if sample:
            p = np.array([policy[a] for a in legal], dtype=np.float64)
            move = int(np.random.default_rng().choice(legal, p=p / p.sum()))
        else:
            move = int(max(legal, key=lambda a: policy[a]))
        rows = [{"move": a, "prior": round(float(policy[a]), 4), "visits": None, "q": None} for a in legal]
        rows.sort(key=lambda r: -r["prior"])
        return move, {"simulations": 0, "sampled": sample, "search_value": None,
                      "net_value": round(net_value, 4), "moves": rows}

    root = Node(node_id=0, state=state)
    tree = MCTS(root=root, model=model, num_nodes=max(2, simulations), add_noise=False)
    tree.build_tree()
    move = int(tree.choose_move(sample=sample))

    children = {int(a): c for a, c in root.children}
    rows = []
    for a in legal:
        c = children.get(a)
        rows.append({
            "move": a,
            "prior": round(float(root.prior[0, a]), 4),
            "visits": c.n if c else 0,
            # a child's w is from the opponent's point of view, so negate for the player to move
            "q": round(-c.w / c.n, 4) if c and c.n else None,
        })
    rows.sort(key=lambda r: (-r["visits"], -r["prior"]))
    return move, {
        "simulations": simulations,
        "sampled": sample,
        "search_value": round(root.w / root.n, 4) if root.n else None,
        "net_value": round(net_value, 4),
        "moves": rows,
    }


# --- commands ------------------------------------------------------------------
def cmd_checkpoints(_req):
    return {"checkpoints": list_checkpoints()}


def cmd_state(req):
    moves = req.get("moves", [])
    return describe(replay(moves), len(moves))


def cmd_analyse(req):
    moves = req.get("moves", [])
    state = replay(moves)
    if state.game_ended:
        return {"analysis": None, "state": describe(state, len(moves))}
    _, analysis = search(state, get_model(req.get("checkpoint")), req.get("simulations", 100))
    return {"analysis": analysis, "state": describe(state, len(moves))}


def cmd_move(req):
    moves = req.get("moves", [])
    state = replay(moves)
    if state.game_ended:
        raise ValueError("the game is already over")
    move, analysis = search(state, get_model(req.get("checkpoint")), req.get("simulations", 100),
                            req.get("sample", False))
    after = state.transition_state(move)
    return {"move": move, "analysis": analysis, "state": describe(after, len(moves) + 1)}


COMMANDS = {"checkpoints": cmd_checkpoints, "state": cmd_state, "analyse": cmd_analyse, "move": cmd_move}


def main():
    # Replies go to the real stdout; anything else that prints (library warnings,
    # a stray debug print in interface.py) goes to stderr so it can't corrupt the protocol.
    out = sys.stdout
    sys.stdout = sys.stderr
    torch.set_grad_enabled(False)
    for line in sys.stdin:
        if not line.strip():
            continue
        req_id = None
        try:
            req = json.loads(line)
            req_id = req.get("id")
            handler = COMMANDS.get(req.get("cmd"))
            if handler is None:
                raise ValueError(f"unknown command {req.get('cmd')!r}")
            reply = {"id": req_id, "ok": True, "result": handler(req)}
        except Exception as e:  # noqa: BLE001 -- report every failure to the caller
            reply = {"id": req_id, "ok": False, "error": f"{type(e).__name__}: {e}"}
        out.write(json.dumps(reply) + "\n")
        out.flush()


if __name__ == "__main__":
    main()
