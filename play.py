import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from model import Model, test_model_overfits, compute_loss, play_arena
# from data import self_play_game, build_dataset, get_batch, build_replay_buffer
from data import Dataset
from interface import ROWS, COLS, NUM_NODES, State, Node, MCTS
import time
from datetime import datetime
from pathlib import Path

# 10/11 TODO:
# make buffer of fixed size and much larger. Allow to add new games and drop old ones.
# modify arena to be a formal eval to be called during training loop every-so-often to confirm model is improving before saving checkpoint
# modify training loop to call arena and stop training if not improving, and to measure avg lenght of game
    # batch_size should be a fixed number of positions for the fwd pass, but it should also equal roughly 32 games to average out noise around z
    # assume avg length of game is 10 plies, then batch_size = 32 * 10 = 320
    # say we fill buffer 10% of the way each time, therefore batch_size should <= 10% of buffer size, say 10%, so buffer size is 3200
        # and we need to play 32 games to fill 10% of buffer since that's ~320 positions or 10% of 3200
    # and say we want to sample each position around 3x, then number of batches = buffer filled size / batch size * 3
    # training should be >=10k games, or ~200k positions. If buffer=32k and takes 10 iterations to fill it and we need 7 turnovers we need at least 70 iterations

# 10/12 TODO:
# add hidden layer to value head (transform to scalar in 2 layers instead of 1)
# change num games, num iterations, to sample each datapoint ~5-10x
# add data mirror to double dataset and cut training time
# add search and remove sampling in arena to get robust comparisson
# print number of draws from each arena, to compute baseline loss to measure against
# evaluate after each iteration, win rate against random weights (with search and no sampling) should be 100% by iteration 28
# only change best model if win rate is >=60% instead of 50%

# 10/15 TODO:
# create formal Logs from training
# create webapp to play vs AI and visualize tree building
# allow user to select number of games or average positions model was trained on, and number of nodes or moves-look-ahead (tree height - on average) for inference
# also with/without search and with/without sampling

# fun TODO:
# print avg height of trees in self-play (need to add self.level attribute to MCTS class)
# implement training with Apple accelerator
# do value batching by doing search in parallel and in steplock do inference of 640 nodes (batch_size) in different trees
# experiment adding data augmentation in get_batch with 50% change
# experiment training with much larger trees
# experiment using much larger network / batchnorm in heads / using max(columns) in policy head

## training

CHANNELS, KERNEL, BLOCKS = 64, 3, 4
BASE_PATH = Path("./connect4_alphazero_style")
CHECKPOINTS_PATH = BASE_PATH / "checkpoints_64_4"
RANDOM_WEIGHTS_PATH = CHECKPOINTS_PATH / "random_weights.pt"

## start new model
net = Model(CHANNELS, KERNEL, BLOCKS, ROWS, COLS)
optimizer = optim.AdamW(
    net.parameters(),
    lr=1e-3,
)

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

def train(model, optimizer, best_model_path):
    # iterations = 12
    num_games = 64
    tree_size = 300
    batch_size = 640
    num_batches = 10
    buffer_size = 6400
    data = Dataset(buffer_size)
    best_model_path = best_model_path
    total_num_samples = 0

    for i in range(37, 49):
        t0 = time.time()
        num_plies_per_game = data.build_replay_buffer(num_games, tree_size, model)
        t1 = time.time()
        
        print(f"iter: {i}, num self-play games: {num_games}, time taken to populate buffer: {round(t1 - t0, 3)}s, number of plies: {int(num_plies_per_game.sum())}, avg plies per game: {num_plies_per_game.mean()}, total unique datapoints: {data.datapoints}")
        
        value_losses = np.zeros(num_batches)
        policy_losses = np.zeros(num_batches)
        losses = np.zeros(num_batches)
        for j in range(num_batches):
        
            x, val, probs = data.get_batch(batch_size)
            pred_val, pred_probs = model(x)
            
            optimizer.zero_grad()
            value_loss, policy_loss = compute_loss(v=pred_val, p=pred_probs, z=val, pi=probs)
            loss = value_loss + policy_loss
            loss.backward()
            optimizer.step()

            # store batch losses
            value_losses[j] = value_loss.item()
            policy_losses[j] = policy_loss.item()
            losses[j] = loss.item()

            total_num_samples += batch_size
            # print(f"batch: {j}, total sampled datapoints: {total_num_samples}, batch loss: {round(loss.item(), 3)}, batch value loss: {round(value_loss.item(), 3)}, batch policy loss: {round(policy_loss.item(), 3)}")

        print(f"iter: {i}, iteration loss: {round(losses.mean(), 3)}, iteration value loss: {round(value_losses.mean(), 3)}, iteration policy loss: {round(policy_losses.mean(), 3)}")
        
        # save and test model
        if i % 3 == 0:
            # save checkpoint
            cur_date_time = datetime.now().strftime("%Y-%m-%d_%Hh%Mm%Ss")
            path = CHECKPOINTS_PATH / f"i{i}_{cur_date_time}.pt"
            model.save(path)

            # evaluate in arena
            print("===== eval starts =====")
            rand, best = eval(path, best_model_path)
            eval_games = rand['model 1 wins'] + rand['model 2 wins'] + rand['draws']
            rand_win_rate = rand['model 1 wins'] / (eval_games - rand['draws'])
            best_win_rate = best['model 1 wins'] / (eval_games - best['draws'])
            print(f"iter: {i}. Outcome vs random model: {rand}, {rand_win_rate*100}% win. Outcome vs previous best model (path {best_model_path}): {best}, {best_win_rate*100}% win")
            print("===== eval ends =====")
            if rand_win_rate <= 0.5:
                raise SystemExit("Latest model checkpoint is worse than random weights")
            if best_win_rate >= 0.6:
                best_model_path = path


def eval(cur_model_path, best_model_path):
    num_games = 100
    num_nodes = 100

    # vs random model
    rand_out = play_arena(cur_model_path, RANDOM_WEIGHTS_PATH, num_games=num_games, num_nodes=num_nodes, search=True, sample=True, sample_first_n=5)

    # vs previous best model
    best_out = play_arena(cur_model_path, best_model_path, num_games=num_games, num_nodes=num_nodes, search=True, sample=True, sample_first_n=5)

    return rand_out, best_out

## play vs AI
def play_game(model: Model, sample: bool, search: bool, num_nodes: int, ):
    plies = 0
    user_state = State.from_empty_board(ROWS, COLS)
    while not user_state.game_ended:
        print(f"Plies: {plies}")
        print(user_state)
        user_action = int(input(f"choose action among allowed moves: "))
        while user_action not in list(user_state.allowed_moves):
            user_action = int(input(f"Invalid action, choose among allowed moves {user_state.allowed_moves}: "))
            
        ai_state = user_state.transition_state(user_action)
        plies += 1
        if ai_state.game_ended:
            if ai_state.reward == 0:
                print("\n============== It's a draw! ==============")
            else:    
                print("\n============== You Win! ==============")
            print(f"Plies: {plies}")
            ai_state.board *= -1
            print(ai_state)
            break
        if search:
            root = Node(node_id=0, state=ai_state)
            tree = MCTS(root=root, model=model, num_nodes=num_nodes, add_noise=False)
            tree.build_tree()
            ai_action = tree.choose_move(sample=sample)
        else:
            # turn state into tensor
            x = torch.tensor(ai_state.board, dtype=torch.float).reshape(1, 1, ROWS, COLS)
    
            # fwd pass, mask illegal moves so they get 0 prob
            _, action_logits = model(x)  # logits shape: [1, 7]
            for i in list(set(range(COLS)) - set(ai_state.allowed_moves)):
                action_logits[0,i] = -np.inf
            action_probs = F.softmax(action_logits, dim=1)
            
            # choose action
            ai_action = None
            if sample:
                ai_action = torch.multinomial(action_probs, num_samples=1)[0]
            else:
                ai_action = action_probs.argmax(dim=1)[0]

        user_state = ai_state.transition_state(ai_action)
        plies += 1
        if user_state.game_ended:
            if user_state.reward == 0:
                print("\n============== It's a draw! ==============")
            else:    
                print("\n============== You Lost! ==============")            
            print(f"Plies: {plies}")
            print(user_state)
            break
    return

## AI vs AI arena
# A = 'connect4_alphazero_style/checkpoints_64_4/i12_2026-10-14_01h30m59s'
# A = 'connect4_alphazero_style/checkpoints_64_4/random_weights.pt'
# B = 'connect4_alphazero_style/checkpoints_64_4/i6_2026-10-14_01h29m43s'
# print(play_arena(A, B, 100, 10, search=True, sample=True))

train(model=net, optimizer=optimizer, best_model_path="connect4_alphazero_style/checkpoints_64_4/i36_2026-10-14_05h52m17s.pt")

## play vs AI
# network = Model(64, 3, num_blocks=4)
# network = Model.load(B)
# network.load_state_dict(torch.load(B, weights_only=True))
# play_game(model=network, sample=False, search=True, num_nodes=500)