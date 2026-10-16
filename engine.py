"""Engine process for the web UI: answers JSON requests with the AlphaZero search.

The Node server (server/) starts this once and talks to it over stdin/stdout,
one JSON object per line:

    request:  {"id": 7, "cmd": "move", "moves": [3, 3, 2], "opponent": "neural",
               "level": "medium", "simulations": 100}
    reply:    {"id": 7, "ok": true, "result": {...}}   or   {"id": 7, "ok": false, "error": "..."}

It is stateless: every request carries the full list of columns played so far,
and the game is replayed from the empty board with State.transition_state, which
also checks that every move is legal. Undo is just a shorter list.

Commands:
    levels                                   list difficulty levels, whether they load, and the board size
    state     {moves}                        describe the position
    move      {moves, opponent, level, simulations, sample}  search, play the chosen move, describe the result
    analyse   {moves, opponent, level, simulations}          search without playing

opponent is "neural" (default: NeuralMCTS with the network of `level`) or "classical"
(ClassicalMCTS: UCT with random rollouts, no network, `level` is ignored).

The board size and the difficulty levels (id, name, checkpoint file) come from
config.toml ([board], [[levels]]); a level whose checkpoint doesn't load or was
trained on a different board size is listed as not ok.

simulations == 0 means "no search": the move comes straight from the policy head
(neural only; classical needs at least 1).
sample == false (default) plays the best move: most visits, or highest prior without search.
sample == true draws the move in proportion to the visit counts (temperature 1),
or to the policy without search. Either way there is no Dirichlet noise.
"""

import json
import sys
import tomllib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from classical_mcts import ClassicalMCTS
from game_interface import Node, State
from model import Model
from neural_mcts import NeuralMCTS

HERE = Path(__file__).resolve().parent
MAX_SIMULATIONS = 10000

with open(HERE / "config.toml", "rb") as f:
    CFG = tomllib.load(f)
ROWS = CFG["board"]["rows"]
COLS = CFG["board"]["cols"]
MAX_SIM_PLIES = CFG["inference"]["max_moves_per_sim"]

LEVELS = CFG["levels"]  # [{id, name, checkpoint, games}], easiest first

_models: dict[str, tuple[float, Model]] = {}  # level id -> (mtime, loaded model)


# --- levels --------------------------------------------------------------------
def _load_level(level: dict) -> Model:
    """Load a level's checkpoint, cached until the file changes."""
    path = HERE / level["checkpoint"]
    mtime = path.stat().st_mtime
    cached = _models.get(level["id"])
    if cached is None or cached[0] != mtime:
        model = Model.load(path).eval()
        if (model.rows, model.cols) != (ROWS, COLS):
            raise ValueError(f"trained on a {model.rows}x{model.cols} board, config.toml board is {ROWS}x{COLS}")
        _models[level["id"]] = (mtime, model)
    return _models[level["id"]][1]


def list_levels() -> list[dict]:
    """The difficulty levels from config.toml, marked ok if their checkpoint loads into the
    current Model and was trained on the board size in config.toml."""
    out = []
    for level in LEVELS:
        entry = {"id": level["id"], "name": level["name"], "games": level["games"]}
        try:
            _load_level(level)
            entry["ok"] = True
        except Exception as e:  # noqa: BLE001 -- a missing file, an old architecture, ...
            entry.update(ok=False, error=f"{type(e).__name__}: {str(e).splitlines()[0][:200]}")
        out.append(entry)
    return out


def get_model(level_id) -> Model:
    """Model for a level id. Only ids from config.toml are accepted, never file paths."""
    level = next((lv for lv in LEVELS if lv["id"] == level_id), None)
    if level is None:
        raise ValueError(f"unknown level {level_id!r}")
    return _load_level(level)


# --- game state ----------------------------------------------------------------
def replay(moves) -> State:
    if not isinstance(moves, list) or len(moves) > ROWS * COLS:
        raise ValueError(f"moves must be a list of at most {ROWS * COLS} columns")
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
    x = torch.tensor(state.board, dtype=torch.float).reshape(1, 1, state.rows, state.cols)
    with torch.no_grad():
        value, logits = model(x)
    mask = torch.full((state.cols,), -np.inf)
    mask[list(state.allowed_moves)] = 0
    probs = F.softmax(logits[0] + mask, dim=0).numpy()
    return float(value.item()), probs


def _check_search_params(simulations, sample, minimum=0):
    if not isinstance(simulations, int) or isinstance(simulations, bool) or not minimum <= simulations <= MAX_SIMULATIONS:
        raise ValueError(f"simulations must be an integer in {minimum}..{MAX_SIMULATIONS}")
    if not isinstance(sample, bool):
        raise ValueError("sample must be true or false")


def search(state: State, model: Model, simulations: int, sample: bool = False) -> tuple[int, dict]:
    _check_search_params(simulations, sample)
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
        return move, {"opponent": "neural", "simulations": 0, "sampled": sample, "search_value": None,
                      "net_value": round(net_value, 4), "moves": rows}

    root = Node(node_id=0, state=state)
    tree = NeuralMCTS(root=root, model=model, num_nodes=max(2, simulations), add_noise=False)
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
        "opponent": "neural",
        "simulations": simulations,
        "sampled": sample,
        "search_value": round(root.w / root.n, 4) if root.n else None,
        "net_value": round(net_value, 4),
        "moves": rows,
    }


def classical_search(state: State, simulations: int, sample: bool = False) -> tuple[int, dict]:
    """ClassicalMCTS (UCT + random rollouts, no network). Same analysis shape as search(),
    with no priors or network value."""
    _check_search_params(simulations, sample, minimum=1)
    root = Node(node_id=0, state=state)
    tree = ClassicalMCTS(root=root, num_nodes=simulations, max_sim_plies=MAX_SIM_PLIES)
    tree.build_tree()

    children = {int(a): c for a, c in root.children}
    if sample:
        # ClassicalMCTS.choose_move has no sampling, so draw in proportion to the root's visit counts here
        actions = list(children)
        visits = np.array([children[a].n for a in actions], dtype=np.float64)
        move = int(np.random.default_rng().choice(actions, p=visits / visits.sum()))
    else:
        move = int(tree.choose_move())

    rows = []
    for a in (int(a) for a in state.allowed_moves):
        c = children.get(a)
        rows.append({
            "move": a,
            "prior": None,
            "visits": c.n if c else 0,
            # a child's w is from the opponent's point of view, so negate for the player to move
            "q": round(-c.w / c.n, 4) if c and c.n else None,
        })
    rows.sort(key=lambda r: -r["visits"])
    return move, {
        "opponent": "classical",
        "simulations": simulations,
        "sampled": sample,
        "search_value": round(root.w / root.n, 4) if root.n else None,
        "net_value": None,
        "moves": rows,
    }


def run_search(req, state: State, sample: bool = False) -> tuple[int, dict]:
    opponent = req.get("opponent", "neural")
    simulations = req.get("simulations", 100)
    if opponent == "neural":
        return search(state, get_model(req.get("level")), simulations, sample)
    if opponent == "classical":
        return classical_search(state, simulations, sample)
    raise ValueError(f"opponent must be 'neural' or 'classical', got {opponent!r}")


# --- commands ------------------------------------------------------------------
def cmd_levels(_req):
    return {"levels": list_levels(), "rows": ROWS, "cols": COLS}


def cmd_state(req):
    moves = req.get("moves", [])
    return describe(replay(moves), len(moves))


def cmd_analyse(req):
    moves = req.get("moves", [])
    state = replay(moves)
    if state.game_ended:
        return {"analysis": None, "state": describe(state, len(moves))}
    _, analysis = run_search(req, state)
    return {"analysis": analysis, "state": describe(state, len(moves))}


def cmd_move(req):
    moves = req.get("moves", [])
    state = replay(moves)
    if state.game_ended:
        raise ValueError("the game is already over")
    move, analysis = run_search(req, state, req.get("sample", False))
    after = state.transition_state(move)
    return {"move": move, "analysis": analysis, "state": describe(after, len(moves) + 1)}


COMMANDS = {"levels": cmd_levels, "state": cmd_state, "analyse": cmd_analyse, "move": cmd_move}


def main():
    # Replies go to the real stdout; anything else that prints (library warnings,
    # a stray debug print in the game code) goes to stderr so it can't corrupt the protocol.
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
