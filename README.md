# Connect 4, AlphaZero style

A from-scratch AlphaZero for Connect 4: self-play training in PyTorch, plus a web app to
play against the trained networks (or a classical MCTS with no network) and watch the
search tree being built move by move. The web app runs entirely in the browser.

| path | what it is |
|---|---|
| `config.toml` | board size, model, training and inference settings, and the web app's difficulty levels |
| `main.py` | command-line entry point: train, or play in the terminal |
| `game_interface.py` | game rules (`State`), tree node (`Node`) |
| `neural_mcts.py` / `classical_mcts.py` | PUCT search guided by the network / UCT search with random rollouts |
| `model.py` | the network (`Model`, with `save` / `load`) and its loss |
| `dataset.py`, `train.py` | self-play replay buffer, training loop and evaluation arena |
| `play.py` | terminal games against either search |
| `viz.py` | matplotlib drawing of a search tree |
| `models/` | the 4 checkpoints behind the web app's difficulty levels (committed) |
| `checkpoints/run_NN/` | training runs (not committed) |
| `export_onnx.py` | exports the difficulty levels to ONNX for the web app |
| `engine.py` | Python reference engine the web app's JavaScript engine is tested against |
| `ui/` | the web app (React + Vite); `ui/src/engine/` is the in-browser engine |

## Setup

Python 3.11+ with [uv](https://docs.astral.sh/uv/), Node.js 20.19+ (or 22.12+).

```bash
uv sync                  # creates .venv with torch, numpy, ... (and the ONNX tools)
cd ui && npm install
```

## Train and play from the terminal

```bash
uv run python main.py train                                            # new run: checkpoints/run_NN/
uv run python main.py train --checkpoint checkpoints/run_01/i48_....pt  # continue a run
uv run python main.py play --checkpoint models/3072games.pt            # vs neural MCTS
uv run python main.py play --opponent classical                        # vs classical MCTS
```

Training from scratch creates the next `checkpoints/run_NN/` folder and saves `random_weights.pt`
there first; checkpoints are saved every `checkpoint_frequency` iterations and evaluated against
the random weights and the best checkpoint so far. Settings are in `config.toml`.

## The web app

```bash
cd ui && npm run dev      # http://localhost:5173
```

Play against **Neural MCTS** at one of four difficulty levels (the network after 768, 1536, 2304
or 3072 self-play games) or **Classical MCTS** (no network: tree search with random rollouts).
You can switch level or opponent in the middle of a game. The search tree is drawn as it is
built, one node per frame, with the visit counts and values; big trees are cropped to the
most-visited nodes (toggle **Full tree**, then drag to pan and ctrl/⌘ + scroll to zoom). With 0
simulations there is no search, and the panel shows the network's value and policy instead.

There is no server: the page loads the networks as ONNX files and runs the search in a Web
Worker with [onnxruntime-web](https://onnxruntime.ai/docs/tutorials/web/).

```
browser: React UI --postMessage--> Web Worker: src/engine/ (rules + both MCTS) --> onnxruntime-web (models/*.onnx)
```

### Changing the difficulty levels

The levels are listed once, in `config.toml` (`[[levels]]`: id, name, checkpoint, games).
After changing them, export and commit the results:

```bash
uv run python export_onnx.py    # -> ui/public/models/<id>.onnx + ui/public/app-config.json
```

The script checks that onnxruntime gives the same outputs as PyTorch for every model.

### Testing the JavaScript engine

`ui/src/engine/` is a port of `game_interface.py`, `neural_mcts.py` and `classical_mcts.py`.
The parity test plays random games and searches through both and compares them (rules on every
position, neural searches move for move and visit for visit, classical search tactics):

```bash
cd ui && npm run parity
```

Run it after changing either side.

# TODO:
* save logs from training
* show number of unique positions and number of total positions the model was trained on in the UI
* show the number of look-ahead moves (average tree height) the model was trained on

# fun TODO:
* implement training with accelerator
* add value batching to do search in parallel (speedup self-play game generation)
* data augmentation, ex: sample flipped board around y-axis in get_batch() with 50% change, probs tensor would also have to be flipped, value remains the same
* experiment training with much larger trees to see if policy becomes confidently wrong (not enough variance in decisions/games)
* experiment using much larger network / batchnorm in heads / using max(columns) in policy head