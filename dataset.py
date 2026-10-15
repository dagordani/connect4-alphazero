import numpy as np
import torch
from interface import State, Node
from neural_mcts import NeuralMCTS



# synthetic dataset:
# use modified version of MCTS where instead of using simulate() we call policy() to get value (discard action probs)
# play a full game of AI vs AI, say its 20 plies, therefore we end up with 20 trees, these are 20 data points to update policy gradients
# after each ply (or after entire game) we call policy() on root to get predicted action probs and predicted value of root
# we normalize the root's children visit counts to get actual action probs and the root's Q is the actual value
# batch size for updating policy gradients can be anything, we just stack these 20 data points with other synthetic games until B size

class Dataset:
    def __init__(self, buffer_size, rows, cols):
        # buffer tensors with pointer that points to next position to fill ("top of stack")
        self.buffer_size = buffer_size
        self.x = torch.zeros(buffer_size, 1, rows, cols)
        self.yvals = torch.zeros(buffer_size, 1)
        self.yprobs = torch.zeros(buffer_size, cols)
        self.ptr = 0
        self.datapoints = 0

    def _self_play_game(self, model, num_nodes, sample_first_n):
        """plays one full game with current model and returns list of root nodes for each move and the terminal state board array"""

        state = State.from_empty_board()
        root_nodes = []
        while not state.game_ended:
            # build node from state and store
            root = Node(node_id=0, state=state)
            root_nodes.append(root)

            # buld tree from node
            tree = NeuralMCTS(root, model, num_nodes, add_noise=True)
            tree.build_tree()

            # make move with sampling (equivalent to temperature=1) for first few moves, then argmax (equivalent to temperature=0), to build variety of opennings for training
            action = tree.choose_move(sample=(len(root_nodes) <= sample_first_n))

            # pass to opponent
            state = state.transition_state(action)  # create copy to use as root for another tree for opponent's move
        
        # return terminal state just to show final board but don't build graph since game is over (not a datapoint for policy training)
        return root_nodes, state

    def _add_game_to_buffer(self, root_nodes: list[Node], final_state: State):
        
        reward = final_state.reward
        val = -1 * reward if len(root_nodes) % 2 == 0 else reward 
        for root in root_nodes:

            # predictions
            board = torch.tensor(root.state.board, dtype=torch.float).unsqueeze(0)  # shape: [1, 6, 7]
            
            # actual game statistics
            val = -1 * val
            val_tensor = torch.tensor([val]) # shape: [1]
            probs = torch.zeros(7)  # shape: [7]
            
            # populate prob by the action index so neural net learns consistent distribution
            for action, node in root.children:
                probs[action] = node.n/root.n

            # fill buffer
            self.x[self.ptr, :, :, :] = board
            self.yvals[self.ptr, :] = val_tensor
            self.yprobs[self.ptr, :] = probs

            # move pointer: if currently pointing at last slot of buffer we move it to the beginning, else increment
            if self.ptr == self.buffer_size - 1:
                self.ptr = 0
            else:
                self.ptr += 1
            self.datapoints += 1

    def build_replay_buffer(self, model, num_games, num_nodes, sample_first_n):
        num_plies_per_game = np.zeros(num_games)
        for g in range(num_games):
            roots, terminal_state = self._self_play_game(model, num_nodes, sample_first_n)
            self._add_game_to_buffer(roots, terminal_state)
            num_plies_per_game[g] = len(roots)
        return num_plies_per_game

    def get_batch(self, batch_size):
        """
        inputs: replay buffer dataset tensors
        outputs: random batch_size subset of tensors

        Note: we generate batch_size number of random indices and index into them because the alternative is generating one random index and
        returning a batch_size slice, but then shuffling the order of the datapoints afterwards requires creating new tensor so it costs memory
        """
        rng = np.random.default_rng()
        buffer_filled_size = self.ptr if self.datapoints // self.buffer_size == 0 else self.buffer_size
        ids = rng.integers(low=0, high=buffer_filled_size, size=batch_size) # inclusive lower bound and exclusive upper bound
        return self.x[ids], self.yvals[ids], self.yprobs[ids]