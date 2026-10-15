import numpy as np


class State:
    def __init__(self, board, last_move_coordinates=None, game_ended=False, reward=None):
        self.board = board
        self.rows = board.shape[0]
        self.cols = board.shape[1]
        self.allowed_moves = np.where(board[0] == 0)[0]
        self.skyline = (board != 0.0).sum(axis=0)
        self.last_move_coordinates = last_move_coordinates
        self.game_ended = game_ended
        self.reward = reward

    @classmethod
    def from_empty_board(cls, rows, cols):
        return cls(np.zeros((rows, cols)))

    @classmethod
    def from_state(cls, state):
        """Creates a fresh instance directly from another State instance."""
        return cls(
            board=state.board.copy(),
            last_move_coordinates=state.last_move_coordinates,
            game_ended=state.game_ended,
            reward=state.reward,
        )

    def copy_state(self):
        return State.from_state(self)
    
    def _update_board(self, action: int) -> None:
        x = int(self.rows - self.skyline[action] - 1)
        y = action
        self.last_move_coordinates = (x, y)
        self.board[x, y] = 1  # update board
        self.skyline[y] += 1  # update skyline
        self.allowed_moves = np.where(self.board[0] == 0)[0]   # update allowed moves
    
    def transition_state_in_place(self, action):
        """used once in expand(), use many times in simulate()"""
        self._update_board(action)  # make move
        self._check_winning_condition()  # update .game_ended and .reward
        self.board *= -1  # flip board to pass to other player. Call after ._check_winning_condition since that checks for +1 pieces
    
    def transition_state(self, action):
        """for expand()"""
        state = self.copy_state()  # creates state: last_move_coordinates=None and game_ended=False
        state.transition_state_in_place(action)  # updates last_move_coordinates via _update_board() and game_ended via check_winning_condition()
        return state

    def _check_winning_condition(self) -> None:
        """check if latest move ended the game through win (return 1) or draw (return 0) or None for neither"""

        # iterate from -3 to +3 position from index of latest move in all 4 directions (horizontal, vertical, both diagonals)
        # we do sliding window of 4 along 7 positions: start from 3,2,1,0 positions to left, and check the 4 right positions
        CONN = 4
        DIRS = [(0, 1), (-1, 0), (-1, 1), (1, 1)]  # right, up, up-right, down-right
        x, y = self.last_move_coordinates
        x_bound, y_bound = self.rows, self.cols
        for dx, dy in DIRS:
            for d in range(-CONN + 1, 1):
                # move in negative direction to set starting point of window
                start_x, start_y = x + d*dx, y + d*dy
                
                # skip out of bounds
                if start_x < 0 or start_x >= x_bound or start_y < 0:                    
                    continue
                    
                # move in positive direction to traverse window
                count = 0
                for i in range(CONN):
                    xx, yy = start_x + i*dx, start_y + i*dy
                    
                    # move within bounds
                    if xx >= 0 and xx < x_bound and yy >= 0 and yy < y_bound:
                        count += self.board[xx, yy] == 1
                        
                    if count == CONN:
                        # player that made last move won, but since the resulting state is of the opponent we set reward to -1
                        self.game_ended = True
                        self.reward = -1
                        return

        # board is full, and win wasn't found, therefore it's a draw
        if self.allowed_moves.size == 0:
            self.game_ended = True
            self.reward = 0
        return

    def __repr__(self):
        return f"Game_Ended? {self.game_ended}\n  board:\n{self.board}\n  skyline: {self.skyline}\n  allowed moves: {self.allowed_moves}"
        

class Game:
    def __init__(self, state: State):
        self.state = state

    def move(self, action):
        self.state.transition_state_in_place(action)


class Node:
    def __init__(self, node_id: int, state: State):
        self.id = node_id  # order of nodes created
        self.state = state
        self.n = 0  # visit count (int)
        self.w = 0.0  # cumulative reward (float)
        self.parent: Node = None
        self.children: list[tuple[int, Node]] = []
        self.fully_expanded = False # for classical MCTS
        self.prior = None  # action probs over its children assigned by policy head
        self.evaluated = False  # whether node has been through model's fwd pass

    def add_child(self, action, child):
        assert isinstance(child, Node)
        self.children.append((action, child))
        child.parent = self
        child.children = []
    
    def __repr__(self):
        return f"id: {self.id}, visit count: {self.n}, value: {self.w}, num_children: {len(self.children)}, fully_expanded? {self.fully_expanded}, prior: {self.prior}, evaluated? {self.evaluated}\n{self.state}"