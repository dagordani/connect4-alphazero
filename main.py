import argparse
import re
import tomllib
from pathlib import Path

import torch.optim as optim

from model import Model, compute_loss
from dataset import Dataset
from train import train
from play import play_vs_classical_MCTS, play_vs_neural_MCTS

CONFIG_PATH = "config.toml"


def parse_args():
    parser = argparse.ArgumentParser(description="Connect 4 AlphaZero: train a model or play against an AI")
    parser.add_argument("mode", choices=["train", "play"], help="train the network or play against an AI")
    parser.add_argument("--checkpoint", default=None,
                        help='model checkpoint path (ex: "checkpoints/run_01/i15_2026-10-12_02h05m42s.pt") to continue training from or to play against')
    parser.add_argument("--opponent", choices=["classical", "neural"], default="neural",
                        help="when playing: play against classical MCTS or neural MCTS (default: neural)")
    args = parser.parse_args()

    if args.mode == "play" and args.opponent == "neural" and args.checkpoint is None:
        parser.error("--checkpoint is required to play against neural MCTS")
    return args


def new_run_dir(checkpoints_dir: Path) -> Path:
    """creates and returns the next run dir, e.g. checkpoints/run_03 if run_01 and run_02 exist"""
    nums = [int(m.group(1)) for d in checkpoints_dir.glob("run_*") if (m := re.fullmatch(r"run_(\d+)", d.name))]
    run_dir = checkpoints_dir / f"run_{max(nums, default=0) + 1:02d}"
    run_dir.mkdir(parents=True)
    return run_dir


if __name__ == '__main__':

    ## load config
    with open(CONFIG_PATH, "rb") as f:
        cfg = tomllib.load(f)

    args = parse_args()

    # read config
    rows = cfg["board"]["rows"]
    cols = cfg["board"]["cols"]

    channels = cfg["model"]["channels"]
    kernel_size = cfg["model"]["kernel_size"]
    blocks = cfg["model"]["blocks"]

    if args.mode == "train":
        learning_rate = cfg["training"]["learning_rate"]
        buffer_size = cfg["training"]["buffer_size"]

        ## train network: either from scratch or from a checkpoint
        # instantiate model; new checkpoints go in the same run dir as the checkpoint we resume from
        if args.checkpoint is not None:
            net = Model.load(args.checkpoint)
            run_dir = Path(args.checkpoint).parent
            best_model_path = Path(args.checkpoint)
        else:
            net = Model(channels, kernel_size, blocks, rows, cols)
            run_dir = new_run_dir(Path(cfg["paths"]["checkpoints_dir"]))
            best_model_path = run_dir / "random_weights.pt"
            net.save(best_model_path)
        print(f"run dir: {run_dir}")

        optimizer = optim.AdamW(
            net.parameters(),
            lr=learning_rate,
        )

        # instantiate data buffer
        data = Dataset(buffer_size, rows, cols)

        # train
        train(model=net, optimizer=optimizer, data=data, loss_fn=compute_loss, cfg=cfg, run_dir=run_dir, best_model_path=best_model_path)

    else:
        ## play: either against Classical MCTS or against Neural MCTS
        if args.opponent == "classical":
            classical_mcts_tree_size = cfg["inference"]["classical_mcts_tree_size"]
            max_moves_per_sim = cfg["inference"]["max_moves_per_sim"]
            play_vs_classical_MCTS(rows=rows, cols=cols, num_nodes=classical_mcts_tree_size, max_sim_plies=max_moves_per_sim)
        else:
            net = Model.load(args.checkpoint)
            net.eval()
            sample = cfg["inference"]["sample"]
            search = cfg["inference"]["search"]
            neural_mcts_tree_size = cfg["inference"]["neural_mcts_tree_size"]
            play_vs_neural_MCTS(rows=rows, cols=cols, model=net, sample=sample, search=search, num_nodes=neural_mcts_tree_size)
