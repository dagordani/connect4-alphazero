"""Export the web UI's difficulty levels to ONNX for the in-browser engine.

Reads the [[levels]] list in config.toml and, for each level:
  - loads its checkpoint (models/*.pt) and exports it to ui/public/models/<id>.onnx
    (input "board" [1, 1, rows, cols], outputs "value" [1, 1] and "logits" [1, cols])
  - checks onnxruntime gives the same outputs as torch on random positions

Then writes ui/public/app-config.json (board size, rollout cap, levels) for the UI.
The UI reads only these generated files, so after changing a level in config.toml,
rerun this script and commit its output:

    uv run python export_onnx.py
"""

import json
import tomllib
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from game_interface import State
from model import Model

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "ui" / "public"
MODELS_DIR = OUT_DIR / "models"
NUM_CHECKS = 100
TOLERANCE = 1e-4


def random_positions(rows, cols, n, rng):
    """Boards (side-to-move frame) from random games, so the check sees realistic inputs."""
    boards = []
    while len(boards) < n:
        state = State.from_empty_board(rows, cols)
        for _ in range(rng.integers(0, rows * cols)):
            if state.game_ended:
                break
            boards.append(state.board.copy())
            state = state.transition_state(int(rng.choice(state.allowed_moves)))
    return boards[:n]


def export_level(level, rows, cols, rng):
    model = Model.load(HERE / level["checkpoint"]).eval()
    if (model.rows, model.cols) != (rows, cols):
        raise SystemExit(f"{level['checkpoint']} was trained on a {model.rows}x{model.cols} board, "
                         f"config.toml board is {rows}x{cols}")

    path = MODELS_DIR / f"{level['id']}.onnx"
    dummy = torch.zeros(1, 1, rows, cols)
    torch.onnx.export(model, (dummy,), path, input_names=["board"], output_names=["value", "logits"],
                      opset_version=18, external_data=False)  # one self-contained file per level

    # same outputs as torch?
    session = ort.InferenceSession(str(path))
    worst = 0.0
    for board in random_positions(rows, cols, NUM_CHECKS, rng):
        x = torch.tensor(board, dtype=torch.float).reshape(1, 1, rows, cols)
        with torch.no_grad():
            value, logits = model(x)
        ort_value, ort_logits = session.run(None, {"board": x.numpy()})
        worst = max(worst, np.abs(ort_value - value.numpy()).max(), np.abs(ort_logits - logits.numpy()).max())
    if worst > TOLERANCE:
        raise SystemExit(f"{path.name}: onnxruntime differs from torch by {worst:.2e} (> {TOLERANCE})")

    print(f"{level['id']:>7}: {level['checkpoint']} -> {path.relative_to(HERE)} "
          f"({path.stat().st_size / 1e6:.2f} MB, max diff vs torch {worst:.1e})")
    return {"id": level["id"], "name": level["name"], "games": level["games"], "model": f"models/{path.name}"}


def main():
    with open(HERE / "config.toml", "rb") as f:
        cfg = tomllib.load(f)
    rows, cols = cfg["board"]["rows"], cfg["board"]["cols"]
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(0)
    levels = [export_level(level, rows, cols, rng) for level in cfg["levels"]]

    app_config = {
        "rows": rows,
        "cols": cols,
        "maxSimPlies": cfg["inference"]["max_moves_per_sim"],
        "levels": levels,
    }
    out = OUT_DIR / "app-config.json"
    out.write_text(json.dumps(app_config, indent=2) + "\n")
    print(f"wrote {out.relative_to(HERE)}")


if __name__ == "__main__":
    main()
