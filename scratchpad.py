# import numpy as np
# import torch
# import torch.optim as optim
# import torch.nn.functional as F
# from model import Model, test_model_overfits, compute_loss, play_arena
# from data import Dataset
# from interface import ROWS, COLS, NUM_NODES, State, Node, MCTS
# import time
# from datetime import datetime
# from pathlib import Path



## training
# CHANNELS, KERNEL, BLOCKS = 64, 3, 4
# BASE_PATH = Path("./connect4_alphazero_style")
# CHECKPOINTS_PATH = BASE_PATH / "checkpoints_64_4"
# RANDOM_WEIGHTS_PATH = CHECKPOINTS_PATH / "random_weights.pt"

## start new model
# net = Model(CHANNELS, KERNEL, BLOCKS, ROWS, COLS)
# optimizer = optim.AdamW(
#     net.parameters(),
#     lr=1e-3,
# )

## test fwd pass
# x = torch.tensor([[[[ 0.,  0.,  1.,  0.,  0.,  0.,  0.],
#             [ 0.,  0.,  1.,  0.,  0.,  0.,  0.],
#             [ 0.,  0., -1., -1.,  0., -1., -1.],
#             [ 0.,  0.,  1.,  1.,  0., -1.,  1.],
#             [ 0.,  0.,  1., -1.,  0.,  1., -1.],
#             [ 1.,  1.,  1., -1., -1.,  1., -1.]]]])
# print(net(x))

# net.save(RANDOM_WEIGHTS_PATH)

# roots, terminal_state = self_play_game(10, net)
# print(roots)
# print(terminal_state)
# xx, yvals, yprobs = build_dataset(roots, terminal_state)
# print(yvals)

# print(test_model_overfits(net, optimizer))

# xx, yvals, yprobs = build_replay_buffer(10, net)
# print(xx.shape, yvals.shape, yprobs.shape)

# a, b, c = get_batch(xx, yvals, yprobs, batch_size=32)
# print(a.shape, b.shape, c.shape)


## load model
# path = CHECKPOINTS_PATH / "i15_2026-10-12_02h05m42s.pt"
# net = Model.load(path)
# optimizer = optim.AdamW(
#     net.parameters(),
#     lr=1e-3,
# )

## AI vs AI arena
# A = 'connect4_alphazero_style/checkpoints_64_4/i12_2026-10-14_01h30m59s'
# A = 'connect4_alphazero_style/checkpoints_64_4/random_weights.pt'
# B = 'connect4_alphazero_style/checkpoints_64_4/i6_2026-10-14_01h29m43s'
# print(play_arena(A, B, 100, 10, search=True, sample=True))

# train(model=net, optimizer=optimizer, best_model_path="connect4_alphazero_style/checkpoints_64_4/i36_2026-10-14_05h52m17s.pt")

## play vs AI
# network = Model(64, 3, num_blocks=4)
# network = Model.load(B)
# network.load_state_dict(torch.load(B, weights_only=True))
# play_game(model=network, sample=False, search=True, num_nodes=500)