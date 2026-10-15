import numpy as np
from data import Dataset
from datetime import datetime
import time
from neural_mcts import NeuralMCTS


def train(model, optimizer, data, loss_fn, cfg):

    # read config
    iterations = cfg["training"]["iterations"]
    num_games = cfg["training"]["num_games"]
    tree_size = cfg["training"]["tree_size"]
    batch_size = cfg["training"]["batch_size"]
    num_batches = cfg["training"]["num_batches"]
    checkpoint_frequency = cfg["training"]["checkpoint_frequency"]

    random_weights_path = cfg["training"]["random_weights_path"]
    best_model_path = cfg["training"]["best_model_path"]
    eval_num_games = cfg["training"]["eval_num_games"]
    eval_tree_size = cfg["training"]["eval_tree_size"]

    train_sample_first_n = cfg["training"]["self_play_sample_first_n"]
    eval_sample_first_n = cfg["training"]["eval_arena_sample_first_n"]

    total_num_samples = 0
    for i in range(1, iterations + 1):
        t0 = time.time()
        num_plies_per_game = data.build_replay_buffer(model=model, num_games=num_games, num_nodes=tree_size, sample_first_n=train_sample_first_n)
        t1 = time.time()
        
        print(f"iter: {i}, num self-play games: {num_games}, time taken to populate buffer: {round(t1 - t0, 3)}s, number of plies: {int(num_plies_per_game.sum())}, avg plies per game: {num_plies_per_game.mean()}, total unique datapoints: {data.datapoints}")
        
        value_losses = np.zeros(num_batches)
        policy_losses = np.zeros(num_batches)
        losses = np.zeros(num_batches)
        for j in range(num_batches):
        
            x, val, probs = data.get_batch(batch_size)
            pred_val, pred_probs = model(x)
            
            optimizer.zero_grad()
            value_loss, policy_loss = loss_fn(v=pred_val, p=pred_probs, z=val, pi=probs)
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
        if i % checkpoint_frequency == 0:
            # save checkpoint
            cur_date_time = datetime.now().strftime("%Y-%m-%d_%Hh%Mm%Ss")
            path = CHECKPOINTS_PATH / f"i{i}_{cur_date_time}.pt"
            model.save(path)

            # evaluate in arena
            print("===== eval starts =====")
            rand = play_arena(cur_model_path, random_weights_path, num_games=eval_num_games, num_nodes=eval_tree_size, search=True, sample=True, sample_first_n=eval_sample_first_n)
            best = play_arena(cur_model_path, best_model_path, num_games=eval_num_games, num_nodes=eval_tree_size, search=True, sample=True, sample_first_n=eval_sample_first_n)
            rand_win_rate = rand['model 1 wins'] / (eval_num_games - rand['draws'])
            best_win_rate = best['model 1 wins'] / (eval_num_games - best['draws'])
            print(f"iter: {i}. Outcome vs random model: {rand}, {rand_win_rate*100}% win. Outcome vs previous best model (path {best_model_path}): {best}, {best_win_rate*100}% win")
            print("===== eval ends =====")
            if rand_win_rate <= 0.5:
                raise SystemExit(f"Error: latest model checkpoint ({path}) is worse than random weights model ({random_weights_path})")
            if best_win_rate >= 0.6:
                best_model_path = path

def play_arena(path1, path2, num_games, num_nodes, search=True, sample=False, sample_first_n=5):
    """
    plays one model against another
    returns terminal state, where -1 is the player than won

    It works with or without search, and with or without sampling the first few moves
    Note:
    - search=False means we only use the model networks, but this mostly makes use of policy head. That's why for evaluation we add search to test value head,
      and num_ndoes is chosen to closely match inference/test time. Also don't use Dirichlet noise since that deviates performance from actual inference/test time
    - we use sampling over first few moves to get diversity of games, otherwise even with search the models are deterministic and the 100 eval games
      are just 2 games repeated 50 times each (game where model 1 start and game where model 2 starts)
    """
    model1 = Model.load(path1)
    model2 = Model.load(path2)

    def _play_one_game(modelA, modelB):
        state = State.from_empty_board(ROWS, COLS)
        model = modelA
        plies = 0
        while not state.game_ended:

            if search:
                root = Node(node_id=0, state=state)
                tree = NeuralMCTS(root=root, model=model, num_nodes=num_nodes, add_noise=False)
                tree.build_tree()
                action = tree.choose_move(sample=(sample and plies < sample_first_n))
            else:
                # turn state into tensor
                x = torch.tensor(state.board, dtype=torch.float).reshape(1, 1, ROWS, COLS)

                # fwd pass, mask illegal moves so they get 0 prob
                _, action_logits = model(x)  # logits shape: [1, 7]
                for i in list(set(range(COLS)) - set(state.allowed_moves)):
                    action_logits[0,i] = -np.inf
                action_probs = F.softmax(action_logits, dim=1)
            
                # choose action
                action = None
                if sample and plies < sample_first_n:
                    action = torch.multinomial(action_probs, num_samples=1)[0]
                else:
                    action = action_probs.argmax(dim=1)[0]

            # pass turn
            state = state.transition_state(action)
            model = modelB if plies % 2 == 0 else modelA
            plies += 1
    
        if state.reward == 0:
            return 'draw'
        else:
            if plies % 2 == 0:
                return 'B'
            else:
                return 'A'
        
    outcomes = {"draws": 0, "model 1 wins": 0, "model 2 wins": 0}
    num_games_model1_starts = num_games // 2
    num_games_model2_starts = num_games - num_games_model1_starts
    for _ in range(num_games_model1_starts):
        out = _play_one_game(model1, model2)
        if out == 'draw':
            outcomes['draws'] += 1
        elif out == 'A':
            outcomes['model 1 wins'] += 1
        elif out == 'B':
            outcomes['model 2 wins'] += 1

    for _ in range(num_games_model2_starts):
        out = _play_one_game(model2, model1)
        if out == 'draw':
            outcomes['draws'] += 1
        elif out == 'A':
            outcomes['model 2 wins'] += 1
        elif out == 'B':
            outcomes['model 1 wins'] += 1
    
    return outcomes  