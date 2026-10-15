import numpy as np
import torch
import torch.nn.functional as F
from game_interface import State, Node
from model import Model
from classical_mcts import ClassicalMCTS
from neural_mcts import NeuralMCTS


def play_vs_classical_MCTS(rows: int, cols: int, num_nodes: int, max_sim_plies: int):
    plies = 0
    user_state = State.from_empty_board(rows, cols)
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
        root = Node(node_id=0, state=ai_state)
        tree = ClassicalMCTS(root=root, num_nodes=num_nodes, max_sim_plies=max_sim_plies)
        tree.build_tree()
        ai_action = tree.choose_move()
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

def play_vs_neural_MCTS(rows: int, cols: int, model: Model, sample: bool, search: bool, num_nodes: int):
    plies = 0
    user_state = State.from_empty_board(rows, cols)
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
            tree = NeuralMCTS(root=root, model=model, num_nodes=num_nodes, add_noise=False)
            tree.build_tree()
            ai_action = tree.choose_move(sample=sample)
        else:
            # turn state into tensor
            x = torch.tensor(ai_state.board, dtype=torch.float).reshape(1, 1, ai_state.rows, ai_state.cols)
    
            # fwd pass, mask illegal moves so they get 0 prob
            _, action_logits = model(x)  # logits shape: [1, 7]
            for i in list(set(range(ai_state.cols)) - set(ai_state.allowed_moves)):
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